"""独立探索入口；默认只有CPU核查，GPU与开发GT必须另行授权。"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from core import (
    OUTPUT,
    Store,
    authorize,
    code_manifest,
    load_adapter,
    read,
    require,
    selected_rows,
    sha,
    tensor_sha,
    verify_sources,
)
from lifecycle import (
    identity,
    record_closure,
    record_exit,
    session_directories,
    verified_checkpoint,
)


def validate_config(path):
    """固定预算、种子和资源合同，不允许从CLI悄悄扩训。"""
    config = read(path)
    require(config["scope"] == "INDEPENDENT_EXPLORATORY_NOT_FORMAL", "范围错误")
    require(
        (config["train_frames"], config["updates"]) in ((512, 256), (256, 64)),
        "预算未登记",
    )
    require(
        config["seed"] == 20261101 and config["evaluation_frames"] == 320, "协议改变"
    )
    require(config["preflight_pause_updates"] == 4, "预检须在4更新处验证独立暂停")
    require(
        config["session_seconds"] == 27000 and config["save_margin_seconds"] == 600,
        "会话时间协议改变",
    )
    require(config["output_root"] == str(OUTPUT), "输出根改变")
    require(
        config["asset_limit_bytes"] == 8_000_000_000
        and config["combined_limit_bytes"] == 30_000_000_000
        and config["free_reserve_bytes"] == 250_000_000_000,
        "存储合同改变",
    )
    require(re.fullmatch(r"lg_bev_explore_[a-z0-9_]+", config["run_id"]), "非法run-id")
    verify_sources(config)
    return config


def run_id(config, action):
    return (
        config["run_id"]
        + {"train": "", "preflight": "_preflight", "evaluate": "_evaluation"}[action]
    )


def require_preflight(store, config, config_path):
    """新训练必须有同配置/源码的独立GPU预检，不接受原A128或CPU测试替代。"""
    state, end, _ = verified_checkpoint(
        store,
        run_id(config, "preflight"),
        identity(config_path, "preflight"),
        require_outcome="PREFLIGHT_PASS",
    )
    require(state["updates"] == 8 and len(state["tokens"]) == 32, "预检不是32帧8更新")
    require(end["outcome"] == "PREFLIGHT_PASS", "预检未通过")


def trained_endpoint(store, config, config_path):
    state, _, _ = verified_checkpoint(
        store,
        config["run_id"],
        identity(config_path, "train"),
        require_outcome="COMPLETED",
    )
    require(
        state["updates"] == config["updates"] and state["budget"] == config["updates"],
        "不是固定独立终点",
    )
    state["adapter_digest"] = tensor_sha(state["adapter"])
    return state


def child_command(args, command):
    result = [
        sys.executable,
        "-B",
        str(HERE / "cli.py"),
        command,
        "--config",
        str(args.config.resolve()),
        "--action",
        args.action,
        "--authorization",
        str(args.authorization.resolve()),
        "--session-id",
        args.session_id,
        "--started",
        str(args.started),
    ]
    if args.continue_run:
        result += ["--continue-run"]
    return result


def previous_state(store, rid, expected, session_id):
    """新会话记录明确绑定上一会话路径，禁止依据mtime或旧run猜测。"""
    launch = read(store.path(f"{rid}/sessions/{session_id}/launch.json"))
    path = store.path(launch["previous_checkpoint"])
    require(sha(path) == launch["previous_checkpoint_sha256"], "继续Checkpoint损坏")
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    require(state["identity"] == expected, "不能加载旧正式或其他实验状态")
    return state


def launch(args, config, store):
    """外层等待监督进程，真实观察退出后才封存closure。"""
    expected, rid = identity(args.config, args.action), run_id(config, args.action)
    args.started = time.monotonic()
    store.reserve_for_artifact(20_000_000)
    store.root.mkdir(parents=True, exist_ok=True)
    with store.path("experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "train":
            require_preflight(store, config, args.config)
        if args.action == "evaluate":
            trained_endpoint(store, config, args.config)
        prior = None
        if args.continue_run:
            _, _, prior = verified_checkpoint(
                store, rid, expected, require_outcome="PAUSED"
            )
        else:
            require(
                not session_directories(store, rid),
                "实验已存在，请使用本实验continue入口",
            )
        if args.action == "preflight":
            require(
                len(session_directories(store, rid)) < 2,
                "本次预检最多两个会话，禁止自动重试",
            )
        args.session_id = str(time.time_ns()) + "_" + uuid.uuid4().hex[:8]
        record = {
            "identity": expected,
            "launcher_pid": os.getpid(),
            "session_id": args.session_id,
            "requested_at": time.time(),
        }
        if prior:
            record.update(
                previous_checkpoint=str(prior.relative_to(store.root)),
                previous_checkpoint_sha256=sha(prior),
            )
        store.json(f"{rid}/sessions/{args.session_id}/launch.json", record)
        with store.path(f"{rid}/sessions/{args.session_id}/console.log").open(
            "xb"
        ) as log:
            supervisor = subprocess.Popen(
                child_command(args, "_session"),
                stdout=log,
                stderr=log,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            )
            code = supervisor.wait()
        if code != 0:
            store.json(
                f"{rid}/sessions/{args.session_id}/failure_observation.json",
                {
                    "supervisor_pid": supervisor.pid,
                    "actual_exit_code": code,
                    "observed_at": time.time(),
                    "safe_to_shutdown": False,
                    "说明": "启动器真实观察；不是成功退出回执，不允许自动继续",
                },
            )
        record_closure(
            store, rid, args.session_id, expected, supervisor, code, args.started
        )
        _, end, _ = verified_checkpoint(store, rid, expected)
        print(
            json.dumps(
                {
                    "outcome": end["outcome"],
                    "run_id": rid,
                    "safe_to_shutdown": True,
                    "can_continue_exploration": end["outcome"] == "PAUSED",
                    "formal_run_resume_allowed": False,
                },
                ensure_ascii=False,
            )
        )


def internal(args, config, store):
    """内部子进程也检查授权及父进程归属，不允许直接绕过启动器。"""
    rid, expected = run_id(config, args.action), identity(args.config, args.action)
    require(
        args.session_id and re.fullmatch(r"[0-9]+_[a-f0-9]+", args.session_id),
        "非法会话",
    )
    session = f"{rid}/sessions/{args.session_id}"
    launch_record = read(store.path(session + "/launch.json"))
    require(launch_record["identity"] == expected, "启动身份错误")
    if args.command == "_session":
        require(
            os.getppid() == launch_record["launcher_pid"], "监督进程不是启动器子进程"
        )
        store.json(session + "/supervisor.json", {"pid": os.getpid()})
        child = subprocess.Popen(
            child_command(args, "_worker"),
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )
        code = child.wait()
        if code != 0:
            store.json(
                session + "/worker_failure_observation.json",
                {
                    "worker_pid": child.pid,
                    "actual_exit_code": code,
                    "observed_at": time.time(),
                    "说明": "Supervisor真实观察到的失败退出；不是成功回执",
                },
            )
        record_exit(store, rid, args.session_id, expected, child, code, args.started)
        return
    require(
        os.getppid() == read(store.path(session + "/supervisor.json"))["pid"],
        "Worker父进程错误",
    )
    previous = (
        previous_state(store, rid, expected, args.session_id)
        if args.continue_run
        else None
    )
    trained = (
        trained_endpoint(store, config, args.config)
        if args.action == "evaluate"
        else None
    )
    from backend import environment

    environment(store)
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from worker import train, evaluate

    if args.action == "evaluate":
        evaluate(
            config,
            store,
            rid,
            args.session_id,
            expected,
            args.started,
            trained,
            previous=previous,
        )
    else:
        train(
            config,
            store,
            rid,
            args.session_id,
            expected,
            args.started,
            previous=previous,
            preflight=args.action == "preflight",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "cpu-check",
            "preflight",
            "train",
            "evaluate",
            "status",
            "stop",
            "_session",
            "_worker",
        ),
    )
    parser.add_argument("--config", type=Path, default=HERE / "configs/main.json")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--continue-run", action="store_true")
    parser.add_argument(
        "--action", choices=("preflight", "train", "evaluate"), default="train"
    )
    parser.add_argument("--session-id")
    parser.add_argument("--started", type=float)
    args = parser.parse_args()
    config = validate_config(args.config)
    store = Store()
    if args.command == "cpu-check":
        require(args.authorization is None, "CPU核查不使用GPU授权")
        import torch

        a, initial = load_adapter(config), load_adapter(config, "a0")
        print(
            json.dumps(
                {
                    "status": "CPU_PREPARATION_ONLY",
                    "cuda_initialized": torch.cuda.is_initialized(),
                    "a9562_sha": tensor_sha(a.state_dict()),
                    "a0_sha": tensor_sha(initial.state_dict()),
                    "train_frames": len(selected_rows(config, "train")),
                    "evaluation_metadata_frames": len(
                        selected_rows(config, "development")
                    ),
                    "development_GT_read": False,
                    "code_sha256": code_manifest(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command in ("status", "stop"):
        rid, expected = run_id(config, args.action), identity(args.config, args.action)
        if args.command == "status":
            _, end, _ = verified_checkpoint(store, rid, expected)
            print(
                json.dumps(
                    {
                        "safe_to_shutdown": True,
                        "outcome": end["outcome"],
                        "can_continue_exploration": end["outcome"] == "PAUSED",
                        "formal_run_resume_allowed": False,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            sessions = session_directories(store, rid)
            require(sessions, "没有可停止的独立实验")
            relative = str(sessions[-1].relative_to(store.root)) + "/stop.request.json"
            if not store.path(relative).exists():
                store.json(
                    relative, {"requested_at": time.time(), "reason": "USER_STOP"}
                )
        return
    if args.command in ("preflight", "train", "evaluate"):
        args.action = args.command
    require(args.authorization is not None, "本轮未授权GPU；请提供新的明确授权文件")
    authorize(args.config, args.authorization, args.action)
    if args.command in ("_session", "_worker"):
        internal(args, config, store)
    else:
        launch(args, config, store)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import traceback

        traceback.print_exc()
        print(
            json.dumps(
                {
                    "status": "BLOCKED_OR_FAILED",
                    "safe_to_shutdown": False,
                    "error": repr(error),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2)

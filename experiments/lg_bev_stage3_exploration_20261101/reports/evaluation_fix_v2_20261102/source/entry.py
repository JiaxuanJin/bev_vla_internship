"""版本化评价入口；修复版本须重新绑定授权，原训练源码保持封存。"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

SOURCE = Path(__file__).resolve().parent
REPORT = SOURCE.parent
ORIGINAL = REPORT.parents[1]
sys.path.insert(0, str(ORIGINAL))
sys.path.insert(0, str(ORIGINAL.parents[1]))
from core import Store, read, require, sha, tensor_sha
from cli import validate_config, trained_endpoint
from lifecycle import (
    verified_checkpoint,
    session_directories,
    record_exit,
    record_closure,
)

CONFIG_PATH = ORIGINAL / "configs/main.json"


def checked_protocol():
    p = read(REPORT / "protocol.json")
    require(
        sha(REPORT / "protocol.json")
        == (REPORT / "protocol.sha256").read_text().strip(),
        "Protocol SHA changed",
    )
    require(
        p["source_sha256"] == {f.name: sha(f) for f in sorted(SOURCE.glob("*.py"))},
        "Evaluation source changed",
    )
    approval_path = REPORT / "authorization.approved.json"
    require(approval_path.exists(), "修复版本 v2 尚未重新绑定授权，评价保持暂停")
    approval = read(approval_path)
    require(
        approval.get("approved") is True
        and approval.get("protocol_sha256") == sha(REPORT / "protocol.json")
        and approval.get("source_sha256") == p["source_sha256"]
        and approval.get("development_gt") is True
        and approval.get("cold_final") is False,
        "修复版本授权与封存协议不一致",
    )
    require(
        p["development_frames"] == 320
        and p["session_seconds"] == 27000
        and p["save_margin_seconds"] == 600,
        "Protocol changed",
    )
    config = validate_config(CONFIG_PATH)
    require(sha(CONFIG_PATH) == p["training_config_sha256"], "Training config changed")
    return p, config


def expected(action):
    return dict(
        schema="LG_BEV_UNIFIED_EVAL_FIX_20261102_V2",
        action=action,
        protocol_sha256=sha(REPORT / "protocol.json"),
    )


def run_id(action):
    return "lg_bev_explore_main_20261101_unified_eval_fix_v2_20261102_" + action


def endpoint(store, config, protocol):
    trained = trained_endpoint(store, config, CONFIG_PATH)
    require(
        trained["adapter_digest"] == protocol["trained_adapter_sha256"],
        "Trained adapter changed",
    )
    require(
        sha(protocol["trained_checkpoint"]) == protocol["trained_checkpoint_sha256"],
        "Trained checkpoint changed",
    )
    return trained


def preflight_gate(store):
    # 模型、评分规则及清单不变；复用已完成的 8 帧 GPU 结果，补充修复专用 CPU 核验。
    previous_report = REPORT.parent / "evaluation_20261102"
    previous_protocol = read(previous_report / "protocol.json")
    require(
        previous_protocol["source_sha256"]
        == {
            path.name: sha(path)
            for path in sorted((previous_report / "source").glob("*.py"))
        },
        "原评价封存源码发生变化",
    )
    previous_identity = dict(
        schema="LG_BEV_UNIFIED_EVAL_20261102_V1",
        action="diagnostic",
        protocol_sha256=sha(previous_report / "protocol.json"),
    )
    state, end, _ = verified_checkpoint(
        store,
        "lg_bev_explore_main_20261101_unified_eval_20261102_diagnostic",
        previous_identity,
        require_outcome="PREFLIGHT_PASS",
    )
    require(
        state["preflight_passed"] and end["preflight_passed"],
        "Diagnostic preflight failed",
    )
    cpu = read(REPORT / "cpu_preflight.json")
    require(
        cpu["passed"]
        and cpu["source_sha256"] == read(REPORT / "protocol.json")["source_sha256"],
        "CPU preflight missing or stale",
    )
    require(cpu["sdk_velocity_exact"] and cpu["cleanup_fence_passed"], "修复核验不完整")


def command(args, role):
    result = [
        sys.executable,
        "-B",
        str(SOURCE / "entry.py"),
        role,
        "--action",
        args.action,
        "--session-id",
        args.session_id,
        "--started",
        str(args.started),
    ]
    if args.continue_run:
        result.append("--continue-run")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "diagnostic",
            "development",
            "status",
            "stop",
            "_supervisor",
            "_worker",
        ),
    )
    parser.add_argument(
        "--action", choices=("diagnostic", "development"), default="development"
    )
    parser.add_argument("--session-id")
    parser.add_argument("--started", type=float)
    parser.add_argument("--continue-run", action="store_true")
    args = parser.parse_args()
    if args.command in ("diagnostic", "development"):
        args.action = args.command
    protocol, config = checked_protocol()
    store = Store()
    rid, ident = run_id(args.action), expected(args.action)
    if args.command == "status":
        state, end, cp = verified_checkpoint(store, rid, ident)
        print(
            json.dumps(
                dict(
                    outcome=end["outcome"],
                    cursor=state["cursor"],
                    safe_to_shutdown=True,
                    can_resume=end["outcome"] == "PAUSED",
                    checkpoint=str(cp),
                    sha256=sha(cp),
                )
            )
        )
        return
    if args.command == "stop":
        sessions = session_directories(store, rid)
        require(sessions, "No session to stop")
        rel = str(sessions[-1].relative_to(store.root)) + "/stop.request.json"
        if not store.path(rel).exists():
            store.json(rel, dict(requested_at=time.time(), reason="USER_STOP"))
        return
    trained = endpoint(store, config, protocol)
    if args.action == "development":
        preflight_gate(store)
    if args.command in ("diagnostic", "development"):
        args.started = time.monotonic()
        with store.path("experiment.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            sessions = session_directories(store, rid)
            prior = None
            if args.continue_run:
                _, _, prior = verified_checkpoint(
                    store, rid, ident, require_outcome="PAUSED"
                )
            else:
                require(not sessions, "Existing run: no retry or overwrite permitted")
            args.session_id = str(time.time_ns()) + "_" + uuid.uuid4().hex[:8]
            rel = f"{rid}/sessions/{args.session_id}"
            launch = dict(
                identity=ident,
                session_id=args.session_id,
                launcher_pid=os.getpid(),
                requested_at=time.time(),
            )
            if prior:
                launch.update(
                    previous_checkpoint=str(prior.relative_to(store.root)),
                    previous_checkpoint_sha256=sha(prior),
                )
            store.json(rel + "/launch.json", launch)
            with store.path(rel + "/console.log").open("xb") as log:
                child = subprocess.Popen(
                    command(args, "_supervisor"), stdout=log, stderr=log
                )
                code = child.wait()
            if code:
                store.json(
                    rel + "/failure_observation.json",
                    dict(
                        supervisor_pid=child.pid,
                        actual_exit_code=code,
                        safe_to_shutdown=False,
                    ),
                )
            record_closure(
                store, rid, args.session_id, ident, child, code, args.started
            )
            state, end, cp = verified_checkpoint(store, rid, ident)
            print(
                json.dumps(
                    dict(
                        outcome=end["outcome"],
                        cursor=state["cursor"],
                        safe_to_shutdown=True,
                        checkpoint=str(cp),
                    )
                )
            )
            return
    rel = f"{rid}/sessions/{args.session_id}"
    launch = read(store.path(rel + "/launch.json"))
    require(launch["identity"] == ident, "Session identity changed")
    if args.command == "_supervisor":
        require(os.getppid() == launch["launcher_pid"], "Wrong launcher")
        store.json(rel + "/supervisor.json", dict(pid=os.getpid()))
        child = subprocess.Popen(command(args, "_worker"))
        code = child.wait()
        if code:
            store.json(
                rel + "/worker_failure_observation.json",
                dict(
                    worker_pid=child.pid, actual_exit_code=code, safe_to_shutdown=False
                ),
            )
        record_exit(store, rid, args.session_id, ident, child, code, args.started)
        return
    require(
        os.getppid() == read(store.path(rel + "/supervisor.json"))["pid"],
        "Wrong supervisor",
    )
    previous = None
    if args.continue_run:
        import torch

        cp = store.path(launch["previous_checkpoint"])
        require(
            sha(cp) == launch["previous_checkpoint_sha256"],
            "Previous checkpoint changed",
        )
        previous = torch.load(cp, map_location="cpu", weights_only=False)
        require(previous["identity"] == ident, "Previous identity changed")
    from runtime_v2 import environment

    environment(store)
    from job import evaluate

    evaluate(args, config, protocol, store, rid, ident, trained, previous)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(2)

"""本地离线start/status/stop/resume；仅首次会话获授权，不自动跨天启动。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import psutil
import torch

from scripts.stage3.lg_bev_common import ROOT, read, sha, write
from scripts.stage3.lg_bev_engineering_cache import CACHE_ROOT, RollingCache, contained
from scripts.stage3.lg_bev_daily_state import (
    RUN_ID,
    RUN,
    OUT,
    POLICY,
    DATA_REPORT,
    SESSION_SECONDS,
    TMUX_SOCKET,
    TMUX_NAME,
    SessionClock,
    atomic_json,
    event,
    latest_checkpoint,
    source_manifest,
    validate_policy,
)


def offline_environment() -> None:
    """本地资源及所有可能的大型运行时缓存固定到G，不修改Windows网络。"""
    for key, value in dict(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        WANDB_MODE="offline",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        LG_BEV_DYNAMIC_CUDA="0",
        LG_BEV_UPSAMPLE_CUDA="0",
        LG_BEV_FIXED_CUDA="1",
        LG_BEV_CPU_DIRECT="1",
    ).items():
        os.environ[key] = value
    for key in (
        "TMPDIR",
        "TEMP",
        "TMP",
        "HF_HOME",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        path = CACHE_ROOT / "runtime" / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)


def deny_network() -> list:
    """实际拒绝Python TCP/UDP互联网连接；UNIX本机IPC保持可用。"""
    original = socket.socket.connect
    attempts = []

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            attempts.append(str(address))
            raise OSError("LG-BEV离线训练禁止网络连接")
        return original(sock, address)

    socket.socket.connect = connect
    # 实际执行拒绝探针，随后本地模型运行不得再请求网络。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.connect(("127.0.0.1", 9))
        except OSError:
            pass
        else:
            raise RuntimeError("离线网络拒绝未生效")
    attempts.clear()
    return attempts


def options(args) -> tuple[Path, dict | None]:
    """正式run固定；验收仅能使用明确独立、不可晋升正式的有界配置。"""
    config = (
        read(args.acceptance_config)
        if getattr(args, "acceptance_config", None)
        else None
    )
    if config:
        if (
            config.get("scope") != "bounded_real_train_acceptance"
            or not config["run_id"].startswith("daily_acceptance_")
            or config.get("formal_training") is not False
            or config.get("total_microsteps") != 32
            or config.get("budget_updates") != 19124
            or len(set(config.get("tokens", []))) != 32
        ):
            raise ValueError("验收配置不得伪装正式run")
        return contained(CACHE_ROOT, "runs/" + config["run_id"]), config
    return RUN, None


def admission() -> dict:
    """准入包含全量数据、跨场景、离线、停止恢复和48小时资源证据。"""
    validate_policy(POLICY)
    path = OUT / "admission.json"
    if not path.exists():
        raise ValueError("BLOCKED：正式准入尚未封存，不能启动长训")
    gate = read(path)
    required = {
        "full_train_labels",
        "cross_scene_native_FP32",
        "real_backward_resume",
        "offline_local_assets",
        "detached_manager",
        "session_and_total_budget",
        "storage_and_memory",
        "resource_48h",
    }
    if (
        gate["status"] != "READY"
        or set(gate["gates"]) != required
        or not all(v == "PASS" for v in gate["gates"].values())
        or gate["conservative_active_hours"] > 48
        or gate["policy_sha256"] != sha(POLICY)
    ):
        raise ValueError("BLOCKED：关键准入不通过或48小时不足")
    if gate["source_sha256"] != source_manifest():
        raise ValueError("BLOCKED：源码偏离准入封存")
    for path, digest in gate["artifacts"].items():
        if sha(Path(path)) != digest:
            raise ValueError("BLOCKED：准入证据损坏/变化")
    return gate


def require_launch_authorization() -> None:
    """日期到达不能自动解锁；必须有绑定最新hold的新明确用户授权记录。"""
    hold = OUT / "launch_hold_20261027.json"
    authorization = OUT / "launch_authorization.json"
    if not hold.exists() or not authorization.exists():
        raise ValueError(
            "AWAITING_AUTHORIZATION：今天禁止正式训练；明天仍须用户新的明确授权"
        )
    approval = read(authorization)
    if (
        approval.get("status") != "AUTHORIZED_BY_NEW_EXPLICIT_USER_REQUEST"
        or approval.get("hold_sha256") != sha(hold)
        or approval.get("policy_sha256") != sha(POLICY)
        or not approval.get("user_request_sha256")
        or approval.get("authorization_date", "") < "2026-10-28"
        or datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
        < approval["authorization_date"]
    ):
        raise ValueError("AWAITING_AUTHORIZATION：授权记录无效，不能按日期自动启动")


def alive(summary: dict, run: Path) -> bool:
    """检查PID及明确run参数，防止PID复用误认和误发信号。"""
    try:
        process = psutil.Process(summary["pid"])
        command = process.cmdline()
        return (
            process.is_running()
            and process.status() != psutil.STATUS_ZOMBIE
            and "scripts.stage3.lg_bev_daily" in command
            and run.name in command
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError):
        return False


def status(run: Path) -> dict:
    """读取真实派生摘要并校对进程；不因Codex断线改变训练。"""
    if not (run / "status.json").exists():
        return dict(
            status=(
                (
                    "READY_AWAITING_AUTHORIZATION"
                    if (OUT / "admission.json").exists()
                    and read(OUT / "admission.json")["status"] == "READY"
                    else "BLOCKED"
                )
                if run == RUN
                else "NOT_STARTED"
            ),
            run_id=run.name,
            reason="尚未启动；请先通过完整准入",
            pid=None,
        )
    value = read(run / "status.json")
    if value["status"] == "STARTING" and value.get("pid") is None:
        # new-session已返回而Python仍在导入模块：从tmux取得真实pane PID，
        # 随后仍执行cmdline/run归属校验，不能将启动窗口误报FAILED_UNCLEAN。
        try:
            result = subprocess.run(
                [
                    "tmux",
                    "-S",
                    TMUX_SOCKET,
                    "display-message",
                    "-p",
                    "-t",
                    value["tmux_session"],
                    "#{pane_pid}",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                value["pid"] = int(result.stdout.strip())
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    value["process_alive"] = alive(value, run)
    if (
        value["status"] in ("RUNNING", "STARTING", "STOPPING")
        and not value["process_alive"]
    ):
        value["status"] = "FAILED_UNCLEAN"
        value["safe_to_shutdown"] = False
    if (
        value["status"] in ("STOPPED", "COMPLETED")
        or value.get("checkpoint_verified_after_stop")
    ) and not value["process_alive"]:
        try:
            latest_checkpoint(run, load=False)
            value["safe_to_shutdown"] = True
        except (ValueError, OSError) as exc:
            value.update(
                status="FAILED_CHECKPOINT",
                safe_to_shutdown=False,
                can_resume=False,
                error=str(exc),
            )
    return value


def start(args, *, resume: bool) -> dict:
    """验证后创建独立tmux会话；完成/失败状态不盲目重跑。"""
    run, config = options(args)
    if config is None:
        require_launch_authorization()
        admission()
    cache = RollingCache()
    existing = status(run)
    if existing.get("process_alive"):
        raise ValueError("该run已有活跃进程")
    if existing["status"] == "COMPLETED" or (run / "terminal.json").exists():
        raise ValueError("COMPLETED：拒绝重复训练")
    if resume:
        if existing["status"] != "STOPPED" or not existing.get("can_resume"):
            raise ValueError("只允许从确认安全停止的同一run恢复；失败需先诊断")
        checkpoint = latest_checkpoint(run, load=False)
    else:
        if run.exists():
            raise ValueError("首次start拒绝既有run；请status或resume，禁止覆盖")
        run.mkdir(parents=True, exist_ok=False)
        write(
            run / "prelaunch.json",
            dict(
                owner=run.name, policy_sha256=sha(POLICY), acceptance_only=bool(config)
            ),
        )
        checkpoint = None
    session_id = uuid.uuid4().hex[:12]
    log = run / "console" / (session_id + ".log")
    log.parent.mkdir(exist_ok=True)
    command = [
        str(ROOT / ".venv-bevformer/bin/python"),
        "-u",
        "-m",
        "scripts.stage3.lg_bev_daily",
        "_worker",
        "--run-id",
        run.name,
        "--session-id",
        session_id,
    ]
    if resume:
        command += ["--checkpoint", checkpoint["path"]]
    if config:
        command += ["--acceptance-config", str(args.acceptance_config.resolve())]
    session = TMUX_NAME if config is None else run.name
    atomic_json(
        run / "status.json",
        dict(
            status="STARTING",
            run_id=run.name,
            session_id=session_id,
            pid=None,
            can_resume=False,
            safe_to_shutdown=False,
            console_log=str(log),
            tmux_session=session,
            previous_total_active_seconds=existing.get("total_active_seconds", 0),
            checkpoint=existing.get("checkpoint"),
            checkpoint_sha256=existing.get("checkpoint_sha256"),
        ),
        cache,
    )
    shell = "cd " + shlex.quote(str(ROOT)) + " && exec " + shlex.join(command)
    shell += " > " + shlex.quote(str(log)) + " 2>&1"
    result = subprocess.run(
        ["tmux", "-S", TMUX_SOCKET, "new-session", "-d", "-s", session, shell],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        atomic_json(
            run / "status.json",
            dict(
                status="FAILED",
                run_id=run.name,
                can_resume=False,
                safe_to_shutdown=False,
                error=result.stderr,
            ),
            cache,
        )
        raise RuntimeError(result.stderr)
    return dict(
        status="STARTING",
        run_id=run.name,
        session_id=session_id,
        tmux_session=session,
        tmux_socket=TMUX_SOCKET,
        console_log=str(log),
    )


def stop(args) -> dict:
    """复用原停止入口；只有进程已退出且checkpoint校验通过才可关机。"""
    run, _ = options(args)
    cache = RollingCache()
    value = status(run)
    if value["status"] == "COMPLETED" and not value.get("process_alive"):
        latest_checkpoint(run, load=False)
        return {**value, "safe_to_shutdown": True}
    if value["status"] == "STOPPED" and not value.get("process_alive"):
        latest_checkpoint(run, load=False)
        return {**value, "safe_to_shutdown": True}
    if not value.get("process_alive"):
        raise ValueError("没有可安全暂停的活跃训练；不能宣称已保存")
    atomic_json(
        run / "stop.request",
        dict(session_id=value["session_id"], time=time.time()),
        cache,
    )
    if (run / "control.jsonl").exists() and value["status"] == "RUNNING":
        # 仍以原训练器SIGUSR1处理器为准；请求文件覆盖初始化/probe边界。
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.stage3.lg_bev_engineering_full",
                "--stop-run",
                str(run),
            ],
            cwd=ROOT,
            capture_output=True,
        )
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        value = status(run)
        if not value.get("process_alive"):
            if value["status"] not in ("STOPPED", "COMPLETED"):
                raise RuntimeError(
                    "进程退出但没有确认安全保存，不能关机：" + str(value)
                )
            latest_checkpoint(run, load=False)
            return {**value, "safe_to_shutdown": True}
        time.sleep(1)
    return {
        **value,
        "status": "STOPPING",
        "safe_to_shutdown": False,
        "reason": "仍在完成安全保存；请继续status，不能据此关机",
    }


class Worker:
    """摘要派生自唯一control与训练器，保存失败一律FAILED。"""

    def __init__(self, run: Path, args, config: dict | None) -> None:
        self.cache = RollingCache()
        self.run, self.args, self.config = run, args, config
        previous = 0.0
        self.last_checkpoint = {}
        if args.checkpoint:
            latest = latest_checkpoint(run)
            if Path(latest["path"]) != args.checkpoint:
                raise ValueError("恢复必须使用最新有效提交checkpoint")
            previous = max(
                latest["wall_seconds"],
                read(run / "status.json").get("previous_total_active_seconds", 0),
            )
            self.last_checkpoint = {
                "checkpoint": latest["path"],
                "checkpoint_sha256": latest["sha256"],
            }
        stop_path = run / "stop.request"
        if stop_path.exists():
            # 归属明确的单次请求不是历史实验；改名留存，不删除。
            if read(stop_path).get("session_id") != args.session_id:
                os.rename(stop_path, run / ("stop.request.handled_" + args.session_id))
        self.clock = SessionClock(
            previous,
            stop_path=stop_path,
            session_seconds=config["session_seconds"] if config else SESSION_SECONDS,
            margin_seconds=0 if config else 120,
        )
        self.last_write = 0.0
        self.last_persist = 0.0
        self.last_record = {}
        self.process = psutil.Process()
        self.process.cpu_percent()
        self.network_attempts = deny_network()
        self.source_sha256 = hashlib.sha256(
            json.dumps(source_manifest(), sort_keys=True).encode()
        ).hexdigest()
        self.gpu_sample = {}
        self.gpu_sample_at = 0.0

        def request(_signum, _frame):
            self.clock.requested = True

        for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, request)
        value = read(run / "status.json")
        value.update(
            pid=os.getpid(),
            total_active_seconds=self.clock.total(),
            session_active_seconds=self.clock.elapsed(),
        )
        atomic_json(run / "status.json", value, self.cache)

    def progress(self, trainer, record: dict, *, force: bool = False) -> None:
        self.last_record = record
        if not force and time.monotonic() - self.last_write < 2:
            return
        self.last_write = time.monotonic()
        if self.last_write - self.gpu_sample_at >= 30:
            self.gpu_sample_at = self.last_write
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.gpu_sample = dict(
                    raw=result.stdout.strip(), returncode=result.returncode
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.gpu_sample = dict(error=str(exc))
        global_updates = trainer.updates + (
            9562 if trainer.stage == "B" and not self.config else 0
        )
        value = dict(
            status="RUNNING",
            run_id=self.run.name,
            pid=os.getpid(),
            session_id=self.args.session_id,
            stage=trainer.stage,
            epoch=trainer.epoch + 1,
            global_epoch=(trainer.epoch + 1 + (2 if trainer.stage == "B" else 0)),
            frame=trainer.cursor,
            microstep=trainer.epoch * len(trainer.tokens) + trainer.cursor,
            stage_updates=trainer.updates,
            updates=global_updates,
            total_planned_updates=(
                28686 if self.config is None else self.config["total_microsteps"] // 4
            ),
            formal_training=self.config is None,
            partial_accumulation=trainer.accumulated,
            loss=record.get("losses"),
            learning_rate=trainer.optimizer.param_groups[0]["lr"],
            session_active_seconds=self.clock.elapsed(),
            total_active_seconds=self.clock.total(),
            session_limit_seconds=self.clock.session_seconds,
            total_limit_seconds=self.clock.total_seconds,
            policy_sha256=sha(POLICY),
            execution_sha256=trainer.protocol_sha256,
            source_manifest_sha256=self.source_sha256,
            gpu_sample=self.gpu_sample,
            process_parent_pid=self.process.ppid(),
            detached_tmux=bool(os.environ.get("TMUX")),
            cpu_rss_bytes=self.process.memory_info().rss,
            cpu_percent=self.process.cpu_percent(),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            console_log=str(self.run / "console" / (self.args.session_id + ".log")),
            can_resume=False,
            safe_to_shutdown=False,
            updated_at=time.time(),
        )
        value.update(self.last_checkpoint)
        # 进度是可重建遥测，2秒刷终端、60秒更新磁盘；不为每条进度
        # fsync全G盘。checkpoint/session_end仍逐次持久化，安全判定不变。
        if force or self.last_write - self.last_persist >= 60:
            self.last_persist = self.last_write
            atomic_json(self.run / "status.json", value, self.cache, durable=force)
            event(self.run, dict(event="progress", **value), self.cache, durable=force)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def note_checkpoint(self, _trainer, path: Path, digest: str) -> None:
        """只派生最近checkpoint显示，不创建第二套训练状态。"""
        self.last_checkpoint = dict(checkpoint=str(path), checkpoint_sha256=digest)

    def finalize(self, trainer, state: str, *, reason: str | None = None) -> None:
        if self.network_attempts:
            raise RuntimeError("离线运行中出现未授权网络请求")
        self.last_write = 0
        self.progress(trainer, self.last_record, force=True)
        path = (
            self.run
            / "checkpoints"
            / f"{trainer.stage}_u{trainer.updates:06d}_m{trainer.cursor:06d}_{uuid.uuid4().hex[:10]}.pt"
        )
        digest = trainer.checkpoint(path, self.cache)
        receipt = path.with_name(path.name + ".receipt.pt")
        record = dict(
            event="checkpoint",
            stage=trainer.stage,
            updates=trainer.updates,
            cursor=trainer.cursor,
            path=str(path),
            sha256=digest,
            receipt_sha256=sha(receipt),
        )
        event(self.run, record, self.cache)
        latest_checkpoint(self.run)  # 真正反序列化、SHA与协议核验，不仅检查存在。
        value = read(self.run / "status.json")
        value.update(
            status=state,
            stage=trainer.stage,
            stage_updates=trainer.updates,
            updates=trainer.updates
            + (9562 if trainer.stage == "B" and not self.config else 0),
            frame=trainer.cursor,
            epoch=trainer.epoch + 1,
            partial_accumulation=trainer.accumulated,
            session_active_seconds=self.clock.elapsed(),
            total_active_seconds=self.clock.total(),
            checkpoint=str(path),
            checkpoint_sha256=digest,
            checkpoint_receipt_sha256=sha(receipt),
            stop_reason=reason or self.clock.reason() or "USER_STOP",
            safe_to_shutdown=False,
            can_resume=state == "STOPPED"
            and self.clock.total() < self.clock.total_seconds
            and self.clock.reason() != "TOTAL_ACTIVE_TIME_LIMIT",
            checkpoint_verified_after_stop=True,
            resume_command=".venv-bevformer/bin/python -m scripts.stage3.lg_bev_daily resume",
            offline_network_attempts=len(self.network_attempts),
        )
        atomic_json(self.run / "status.json", value, self.cache)
        event(self.run, dict(event="session_end", **value), self.cache)
        print(
            "SESSION_SAVED_VERIFIED", json.dumps(value, ensure_ascii=False), flush=True
        )


def worker(args) -> None:
    """独立tmux里的实际进程；异常退出不报告可以关机。"""
    offline_environment()
    run, config = options(args)
    work = Worker(run, args, config)
    try:
        validate_policy(POLICY)
        if config:
            from scripts.stage3.lg_bev_daily_acceptance import acceptance_worker

            acceptance_worker(work, config)
        else:
            require_launch_authorization()
            admission()
            from types import SimpleNamespace
            from scripts.stage3.lg_bev_engineering_full import full_training

            full_training(
                SimpleNamespace(
                    authorize_full_training=True,
                    cache_gb=None,
                    engineering_out=ROOT
                    / "results/lg_bev_engineering_handoff_20261027/final",
                    daily_policy=POLICY,
                    daily_admission=OUT / "admission.json",
                    data_report=DATA_REPORT,
                    source_provenance=DATA_REPORT.parent / "source_provenance.json",
                    prepare_data_if_needed=False,
                    data_run_id="formal_gt_20261027_v1",
                    run_id=RUN_ID,
                    resume=args.checkpoint,
                    session_clock=work.clock,
                    progress_callback=work.progress,
                    checkpoint_callback=work.note_checkpoint,
                    finalize=work.finalize,
                )
            )
    except BaseException as exc:
        failure = dict(
            status="FAILED",
            run_id=run.name,
            pid=os.getpid(),
            session_id=args.session_id,
            safe_to_shutdown=False,
            can_resume=False,
            error=repr(exc),
            traceback=traceback.format_exc(),
            total_active_seconds=work.clock.total(),
            session_active_seconds=work.clock.elapsed(),
        )
        try:
            atomic_json(run / "status.json", failure, work.cache)
            event(run, dict(event="fatal_error", **failure), work.cache)
        finally:
            traceback.print_exc()
        raise


def main() -> None:
    """四个可复制命令；watch的Ctrl-C仅退出监控，不给训练进程发信号。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=["start", "status", "stop", "resume", "_worker"]
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--session-id")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--acceptance-config", type=Path)
    args = parser.parse_args()
    if args.action == "_worker":
        worker(args)
        return
    offline_environment()
    try:
        if args.action in ("start", "resume"):
            print(
                json.dumps(
                    start(args, resume=args.action == "resume"),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        elif args.action == "stop":
            print(json.dumps(stop(args), ensure_ascii=False, indent=2))
        else:
            run, _ = options(args)
            while True:
                print(json.dumps(status(run), ensure_ascii=False, indent=2), flush=True)
                if not args.watch:
                    break
                time.sleep(2)
    except KeyboardInterrupt:
        print("监控退出；训练保持独立运行", flush=True)
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps(dict(status="BLOCKED", error=str(exc)), ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()

"""Sequential fresh-process micro stages with a host-memory watchdog."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .runtime_memory import host_memory


def run_worker(args, worker, directory, protocol=None):
    directory.mkdir(parents=True, exist_ok=False)
    result = directory / "result.json"
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("run_stage2.py")),
        "--mode",
        "micro-train",
        "--micro-worker",
        worker,
        "--output",
        str(result),
        "--model",
        str(args.model),
        "--author-root",
        str(args.author_root.resolve()),
        "--joint-checkpoint",
        str(args.joint_checkpoint.resolve()),
        "--seed",
        str(args.seed),
        "--micro-steps",
        str(args.micro_steps),
        "--min-host-available-mib",
        str(args.min_host_available_mib),
        "--min-cuda-free-mib",
        str(args.min_cuda_free_mib),
        "--max-worker-rss-mib",
        str(args.max_worker_rss_mib),
    ]
    if args.joint_config:
        command += ["--joint-config", str(args.joint_config.resolve())]
    if protocol:
        command += ["--protocol", str(protocol)]
    if args.resume:
        command += ["--resume", str(args.resume.resolve())]
    if worker == "protocol":
        command += [
            "--attention-tv-weight",
            str(getattr(args, "attention_tv_weight", 0.0)),
        ]
    if worker == "smoke" and getattr(args, "fusion_checkpoint", None):
        command += ["--fusion-checkpoint", str(args.fusion_checkpoint)]
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MPLCONFIGDIR": "/tmp/stage2-mpl",
        "STAGE2_MEMORY_LOG": str(directory / "memory.jsonl"),
    }
    return monitor_command(command, directory, args, worker=worker, env=env)


def monitor_command(command, directory, args, *, worker, env=None, result_path=None):
    """复用宿主保护监控，包括固定旧源码的只读重放。目录由调用方排他创建。"""
    result = result_path or directory / "result.json"
    summary = {
        "worker": worker,
        "command": command,
        "peak_rss_mib": 0,
        "min_available_mib": float("inf"),
        "guard_triggered": False,
    }
    with (directory / "runtime.log").open("x") as log, (
        directory / "watchdog.jsonl"
    ).open("x") as watch:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=env
        )
        try:
            last_print = 0
            while process.poll() is None:
                sample = host_memory(process.pid)
                sample["elapsed_monotonic"] = time.monotonic()
                summary["peak_rss_mib"] = max(
                    summary["peak_rss_mib"],
                    sample.get("VmRSS_mib", 0),
                    sample.get("VmHWM_mib", 0),
                )
                summary["min_available_mib"] = min(
                    summary["min_available_mib"], sample["MemAvailable_mib"]
                )
                watch.write(json.dumps(sample) + "\n")
                watch.flush()
                if (
                    sample["MemAvailable_mib"] < args.min_host_available_mib
                    or sample.get("VmRSS_mib", 0) > args.max_worker_rss_mib
                ):
                    summary["guard_triggered"] = True
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                if time.monotonic() - last_print >= 30:
                    print(
                        f"[watchdog] {worker} rss={sample.get('VmRSS_mib', 0):.0f} MiB available={sample['MemAvailable_mib']:.0f} MiB log={directory / 'runtime.log'}",
                        flush=True,
                    )
                    last_print = time.monotonic()
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    summary["exit_code"] = process.returncode
    if result.exists():
        summary["result"] = json.loads(result.read_text())
    with (directory / "process_summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
    if process.returncode or summary["guard_triggered"]:
        raise RuntimeError(
            f"Micro {worker} failed; see {directory}; guard={summary['guard_triggered']}"
        )
    return summary


def run_micro_stages(args):
    output = args.output.parent
    # Claim a fresh run directory before launching any child or writing artifacts.
    output.mkdir(parents=True, exist_ok=False)
    if not 1 <= args.micro_steps <= 20:
        raise ValueError("--micro-steps must be between 1 and 20")
    if args.micro_phase in ("step", "protocol", "smoke"):
        if args.protocol or args.resume or args.micro_steps != 1:
            raise ValueError(
                "Fresh run requires one step and no protocol/resume; use continue explicitly"
            )
        fixed = run_worker(args, "protocol", output / "protocol")
        protocol = output / "protocol/protocol.json"
        if args.micro_phase == "protocol":
            return {
                "status": "PASS_FIXED_PROTOCOL",
                "optimizer_steps": 0,
                "processes": [fixed],
            }
        if args.micro_phase == "smoke":
            smoke = run_worker(args, "smoke", output / "smoke", protocol)
            return {
                "status": smoke["result"]["status"],
                "optimizer_steps": 0,
                "processes": [fixed, smoke],
            }
        trained = run_worker(args, "train", output / "train", protocol)
        return {
            "status": "PASS_ONE_STEP_PRESSURE_TEST",
            "optimizer_steps": 1,
            "processes": [fixed, trained],
        }
    if args.protocol is None or args.resume is None:
        raise ValueError("continue/validate require --protocol and --resume")
    summary = run_worker(
        args,
        "train" if args.micro_phase == "continue" else "validate",
        output / args.micro_phase,
        args.protocol.resolve(),
    )
    return {
        "status": summary["result"]["status"],
        "optimizer_steps": summary["result"].get("optimizer_steps", 0),
        "processes": [summary],
    }

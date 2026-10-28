"""仅独立32帧验收run的真实tmux启动/自动暂停/人工暂停/恢复，不允许正式run。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.stage3.lg_bev_common import ROOT, read, write
from scripts.stage3.lg_bev_daily_state import OUT, RUN, latest_checkpoint
from scripts.stage3.lg_bev_daily import status, require_launch_authorization


def check(config_path: Path) -> dict:
    """三个独立会话合计32microsteps/8测试更新，完成后不再续训。"""
    config = read(config_path)
    if config["formal_training"] or not config["run_id"].startswith(
        "daily_acceptance_"
    ):
        raise ValueError("只允许明确有界测试run")
    from scripts.stage3.lg_bev_engineering_cache import CACHE_ROOT

    run = CACHE_ROOT / "runs" / config["run_id"]
    command = [sys.executable, "-m", "scripts.stage3.lg_bev_daily"]
    calls = []

    def invoke(action: str, expected: int = 0) -> str:
        result = subprocess.run(
            command + [action, "--acceptance-config", str(config_path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=150,
        )
        calls.append(
            dict(
                action=action,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
        if result.returncode != expected:
            raise RuntimeError(str(calls[-1]))
        return result.stdout

    def wait_for(target: str, timeout: float = 600) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = status(run)
            if value["status"].startswith("FAILED"):
                raise RuntimeError(str(value))
            if value["status"] == target and (
                target == "RUNNING" or not value.get("process_alive")
            ):
                return value
            time.sleep(0.3)
        raise TimeoutError(str(status(run)))

    invoke("start")
    automatic = wait_for("STOPPED")
    invoke("status")
    if automatic["stop_reason"] != "DAILY_SESSION_LIMIT":
        raise ValueError("3秒同一会话时钟未自动暂停")
    first = latest_checkpoint(run)["state"]
    invoke("resume")
    running = wait_for("RUNNING")
    invoke("stop")
    manual = wait_for("STOPPED")
    if manual["stop_reason"] != "USER_STOP":
        raise ValueError("人工停止未经过安全边界")
    second = latest_checkpoint(run)["state"]
    invoke("resume")
    completed = wait_for("COMPLETED")
    invoke("resume", expected=2)
    final = latest_checkpoint(run)["state"]
    if RUN.exists():
        raise ValueError("测试期间正式run不应出现")
    try:
        require_launch_authorization()
    except ValueError:
        locked = True
    else:
        locked = False
    events = [
        json.loads(line) for line in (run / "control.jsonl").read_text().splitlines()
    ]
    end = [e for e in events if e.get("event") == "session_end"]
    if len(end) != 3 or final["updates"] != 8:
        raise ValueError("会话或更新数异常")
    if not any(s["accumulated"] for s in (first, second)):
        raise ValueError("测试未覆盖部分累积梯度中断，不能声称验收")
    if not all(e["offline_network_attempts"] == 0 for e in end):
        raise ValueError("发现联网尝试")
    if (
        not automatic["safe_to_shutdown"]
        or not manual["safe_to_shutdown"]
        or not completed["safe_to_shutdown"]
    ):
        raise ValueError("checkpoint或进程退出未满足安全条件")
    # optimizer/RNG均从真实checkpoint读取；后续compare对照不中断终点。
    if not final["optimizer"]["state"] or not final.get("rng"):
        raise ValueError("持久状态不完整")
    result = dict(
        status="PASS",
        run_id=run.name,
        scope="bounded_32frames_only; formal updates zero",
        automatic_stop_verified=True,
        manual_stop_verified=True,
        partial_gradient_resume_verified=True,
        safe_to_shutdown_after_exit=True,
        completed_refuses_resume=True,
        formal_launch_locked=locked,
        formal_run_exists=False,
        detached_tmux=running["detached_tmux"],
        detached_parent_pid=running["process_parent_pid"],
        offline_network_attempts=0,
        test_stage_updates=final["updates"],
        formal_updates=0,
        session_summaries=end,
        checkpoint_keys=sorted(final),
        intermediate_accumulation=[first["accumulated"], second["accumulated"]],
        commands=calls,
    )
    write(OUT / "manager_control_acceptance.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.config), ensure_ascii=False, indent=2))

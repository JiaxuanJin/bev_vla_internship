"""从唯一控制事件/真实checkpoint追加当日中文记录；不启动或恢复训练。"""

from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
from zoneinfo import ZoneInfo

from scripts.stage3.lg_bev_common import ROOT
from scripts.stage3.lg_bev_daily_state import RUN, latest_checkpoint
from scripts.stage3.lg_bev_daily import status


def record() -> str:
    """同一会话同一checkpoint只登记一次，按上海日期选择唯一logbook。"""
    summary = status(RUN)
    if summary["status"] not in ("STOPPED", "COMPLETED"):
        raise ValueError("只记录已退出且checkpoint已校验的真实正式会话")
    if not summary.get("safe_to_shutdown"):
        raise ValueError("进程未退出或checkpoint未通过核验")
    checkpoint = latest_checkpoint(RUN, load=False)
    date = datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()
    target = ROOT / "docs" / ("logbook_" + date + ".md")
    marker = (
        "<!-- LG-BEV:" + summary["session_id"] + ":" + checkpoint["sha256"] + " -->"
    )
    value = {
        k: summary.get(k)
        for k in (
            "run_id",
            "session_id",
            "status",
            "stage",
            "epoch",
            "frame",
            "updates",
            "partial_accumulation",
            "loss",
            "learning_rate",
            "session_active_seconds",
            "total_active_seconds",
            "stop_reason",
            "checkpoint",
            "checkpoint_sha256",
            "offline_network_attempts",
            "safe_to_shutdown",
            "can_resume",
        )
    }
    with target.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0)
        content = stream.read()
        if marker in content:
            return str(target) + "：同一提交已记录，未重复追加"
        if not content:
            stream.write("# " + date + " LG-BEV 实验记录\n")
        stream.write(
            "\n## 正式离线会话的安全暂停记录\n\n"
            + marker
            + "\n\n本条来自真实 control.jsonl/status/checkpoint，未运行开发或封存集评价。"
            + "下一段训练仍需用户明确执行 resume，不自动启动。\n\n```json\n"
            + json.dumps(value, ensure_ascii=False, indent=2)
            + "\n```\n"
        )
        stream.flush()
        import os

        os.fsync(stream.fileno())
    return str(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", required=True)
    parser.parse_args()
    print(record())

"""48小时累计/7.5小时会话的离线控制原语；不包含模型计算。"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
from typing import Any

from scripts.stage3.lg_bev_common import ROOT, read, sha
from scripts.stage3.lg_bev_engineering_cache import CACHE_ROOT, RollingCache, contained

RUN_ID = "lg_bev_formal_48h_20261027_v1"
RUN = CACHE_ROOT / "runs" / RUN_ID
OUT = ROOT / "results/lg_bev_daily_20261027"
POLICY = OUT / "time_protocol_v2.json"
DATA_REPORT = CACHE_ROOT / "runs/formal_gt_20261027_v1/data_report.json"
TOTAL_SECONDS = 48 * 3600
SESSION_SECONDS = 7.5 * 3600
TMUX_SOCKET = "/tmp/lg_bev_daily_20261027.sock"
TMUX_NAME = "lg-bev-formal"


def source_manifest() -> dict[str, str]:
    """登记原工程及新控制源码，不依赖联网包或浮动分支。"""
    paths = sorted((ROOT / "scripts/stage3").glob("lg_bev_engineering_*.py"))
    paths += sorted((ROOT / "scripts/stage3").glob("lg_bev_daily*.py"))
    for relative in [
        "scripts/stage3/lg_bev_prototype.py",
        "scripts/stage3/lg_bev_native.py",
        "scripts/stage3/lg_bev_data.py",
        "scripts/stage3/lg_bev_supervision_space.py",
        "scripts/stage2/language_encoder.py",
        "scripts/stage2/spatial_v3_model.py",
        "scripts/stage2/bev_coordinates.py",
        "scripts/stage2/micro_training.py",
        "scripts/stage2/joint_runtime.py",
        "scripts/stage2/spatial_annotations.py",
        "scripts/stage2/low_memory_dataset.py",
    ]:
        paths.append(ROOT / relative)
    return {str(p): sha(p) for p in paths}


def atomic_json(
    path: Path, value: Any, cache: RollingCache, *, durable: bool = True
) -> None:
    """只更新本次明确归属run的可变状态；历史实验和不可变结果不覆盖。"""
    contained(cache.root, str(path.relative_to(cache.root)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with cache.locked():
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        cache.reserve_for_artifact(len(encoded) + 8192)
        pending = path.with_name(path.name + ".pending-" + uuid.uuid4().hex)
        with pending.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            if durable:
                os.fsync(stream.fileno())
        os.replace(pending, path)


def event(run: Path, value: dict, cache: RollingCache, *, durable: bool = True) -> None:
    """唯一control事件源，所有摘要从此源/真实checkpoint派生。"""
    line = json.dumps(
        {"time": time.time(), **value}, ensure_ascii=False, allow_nan=False
    )
    with cache.locked():
        cache.reserve_for_artifact(len(line.encode()) + 8192)
        with (run / "control.jsonl").open("a") as stream:
            stream.write(line + "\n")
            stream.flush()
            if durable:
                os.fsync(stream.fileno())


class SessionClock:
    """共享A/B/训练probe的单会话时钟；累计基数由持久checkpoint恢复。"""

    def __init__(
        self,
        previous: float,
        *,
        session_seconds: float = SESSION_SECONDS,
        total_seconds: float = TOTAL_SECONDS,
        stop_path: Path | None = None,
        margin_seconds: float = 120.0,
    ) -> None:
        if not math.isfinite(previous) or previous < 0 or previous > total_seconds:
            raise ValueError("累计活跃时间无效或已超预算")
        self.previous = previous
        self.started = time.perf_counter()
        self.session_seconds = session_seconds
        self.total_seconds = total_seconds
        self.margin_seconds = min(margin_seconds, session_seconds / 10)
        self.stop_path = stop_path

    def elapsed(self) -> float:
        """当前会话活跃秒数，进程重启重新计时。"""
        return time.perf_counter() - self.started

    def total(self) -> float:
        """跨天累计，用户停机时间不计入。"""
        return self.previous + self.elapsed()

    def reason(self) -> str | None:
        """预留保存时间，在微步边界停止；不承诺精确秒级退出。"""
        if getattr(self, "requested", False):
            return "USER_STOP"
        if self.total() >= self.total_seconds - self.margin_seconds:
            return "TOTAL_ACTIVE_TIME_LIMIT"
        if self.stop_path is not None and self.stop_path.exists():
            return "USER_STOP"
        if self.elapsed() >= self.session_seconds - self.margin_seconds:
            return "DAILY_SESSION_LIMIT"
        return None


def validate_policy(path: Path) -> dict:
    """48小时只修订资源时间，父合同及工程缓存补充仍需精确绑定。"""
    value = read(path)
    parent = (
        ROOT
        / "results/lg_bev_formal_training_audit_20261027/formal_execution_contract.json"
    )
    if (
        value["parent_contract_sha256"] != sha(parent)
        or value["total_active_seconds"] != TOTAL_SECONDS
        or value["session_active_seconds"] != SESSION_SECONDS
        or value["training_content_changed"] is not False
    ):
        raise ValueError("时间协议不符合用户授权或父合同变化")
    if (
        value["training_content_sha256"]
        != hashlib.sha256(json.dumps(read(parent), sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("训练内容摘要不符")
    return value


def checkpoint_valid(run: Path, record: dict, *, load: bool = True) -> dict:
    """先SHA再读取；损坏最新提交不得静默回退重做有效更新。"""
    import torch

    path = Path(record["path"])
    if (
        not path.resolve().is_relative_to((run / "checkpoints").resolve())
        and path != run / "adapter_terminal.pt"
    ):
        raise ValueError("checkpoint不属于该run")
    if sha(path) != record["sha256"]:
        raise ValueError("checkpoint损坏或SHA不符，禁止恢复")
    receipt_path = path.with_name(path.name + ".receipt.pt")
    if not receipt_path.exists():
        raise ValueError("checkpoint缺少持久时间receipt，禁止猜测累计时间")
    receipt = torch.load(receipt_path, map_location="cpu", weights_only=False)
    if receipt["checkpoint_sha256"] != record["sha256"]:
        raise ValueError("checkpoint receipt摘要不符")
    if record.get("receipt_sha256") and sha(receipt_path) != record["receipt_sha256"]:
        raise ValueError("时间receipt损坏")
    result = {**record, "wall_seconds": receipt["wall_seconds"]}
    if load:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["protocol_sha256"] != sha(run / "execution.json"):
            raise ValueError("checkpoint执行协议不符")
        result["state"] = state
    return result


def latest_checkpoint(run: Path, *, load: bool = True) -> dict:
    """以最后一个已fsync提交事件为准，不按文件mtime猜测。"""
    lines = (run / "control.jsonl").read_text().splitlines()
    for line in reversed(lines):
        record = json.loads(line)
        if record.get("event") == "checkpoint":
            return checkpoint_valid(run, record, load=load)
    raise ValueError("尚无有效提交checkpoint")

"""LG-BEV 独立实验的只增写文件、哈希与协议校验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results/lg_bev_20261027"
SEED = 20261027
FORBIDDEN = {"scene-0061", "scene-0655", "scene-0553", "scene-0001", "scene-0002"}


def sha(path: str | Path) -> str:
    """流式计算文件内容摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: Any) -> None:
    """拒绝覆盖已有结果，且拒绝写入 NaN。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def read(path: Path) -> Any:
    """读取 UTF-8 JSON。"""
    return json.loads(path.read_text(encoding="utf-8"))


def assert_splits(frames: list[dict]) -> None:
    """禁止场景、帧 token 跨 split 或触及封存场景。"""
    if not frames or {r["split"] for r in frames} != {"train", "internal_dev"}:
        raise ValueError("必须包含 train 与 internal_dev")
    seen_tokens: set[str] = set()
    scenes: dict[str, str] = {}
    for row in frames:
        if row["scene_name"] in FORBIDDEN:
            raise ValueError("封存场景不能进入新任务")
        token, scene, split = row["sample_token"], row["scene_token"], row["split"]
        if token in seen_tokens:
            raise ValueError("重复或跨 split 的帧 token")
        if scene in scenes and scenes[scene] != split:
            raise ValueError("场景跨 split")
        seen_tokens.add(token)
        scenes[scene] = split


def verify_protocol(out: Path) -> dict:
    """训练和评估入口重新检查固定来源，防止静默改协议。"""
    protocol = read(out / "protocol.json")
    assert_splits(protocol["frames"])
    for path, digest in protocol["lineage_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"来源文件已改变: {path}")
    for path, digest in protocol["source_sha256"].items():
        if sha(ROOT / path) != digest:
            raise ValueError(f"实验代码已改变: {path}")
    if sha(out / "manifest.json") != protocol["manifest_sha256"]:
        raise ValueError("实验清单已改变")
    return protocol

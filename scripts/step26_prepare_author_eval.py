#!/usr/bin/env python3
"""准备作者原始分割评估入口的 30 帧独立复现。

只生成子集 PKL、隔离工作目录与兼容版 test.py；不修改第三方源码。
"""

from __future__ import annotations

import difflib
import hashlib
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer-seg"
OUTPUT = ROOT / "outputs/drivelm_check/root_cause/step26_author_eval"
SOURCE_PKL = (
    ROOT
    / "outputs/drivelm_check/root_cause/step21_full_val"
    / "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
)
SOURCE_ENTRY = REPO / "tools/test.py"
DEST_ENTRY = ROOT / "scripts/step26_author_test_entry.py"


def sha256(path: Path) -> str:
    """读取文件并计算 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """生成连续 30 帧索引与轻量兼容启动副本。"""
    if not SOURCE_PKL.is_file() or not SOURCE_ENTRY.is_file():
        raise FileNotFoundError("找不到完整 Val PKL 或上游 tools/test.py")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with SOURCE_PKL.open("rb") as handle:
        source = pickle.load(handle)
    ordered = sorted(source["infos"], key=lambda info: info["timestamp"])
    if len(ordered) != 6019:
        raise ValueError(f"意外的 Val 帧数: {len(ordered)}")
    preview = ordered[:30]
    if len({item["scene_token"] for item in preview}) != 1:
        raise ValueError("前 30 帧不是同一场景，请先确认 Step 23 的抽样索引")
    if any(
        preview[i]["prev"] != preview[i - 1]["token"]
        for i in range(1, len(preview))
    ):
        raise ValueError("前 30 帧不是直接连续前驱，请核对 Step 23 的抽样规则")

    subset = dict(source)
    subset["infos"] = preview
    subset_path = OUTPUT / "val30_author.pkl"
    with subset_path.open("wb") as handle:
        pickle.dump(subset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    original = SOURCE_ENTRY.read_text(encoding="utf-8")
    guard = "    assert args.out or args.eval or args.format_only or args.show"
    after_guard = "    if args.eval and args.format_only:"
    if original.count(guard) != 1 or original.count(after_guard) != 1:
        raise RuntimeError("上游 test.py 的参数检查结构已变化，停止自动生成")
    beginning = original.index(guard)
    ending = original.index(after_guard, beginning)
    modified = (
        original[:beginning]
        + "    # 分割指标由上游 single_gpu_test 内部打印，不需要检测 --eval 参数。\n"
        + original[ending:]
    )

    old = "model = MMDataParallel(model, device_ids=[0])"
    new = "model = MMDataParallel(model.cuda(), device_ids=[0])"
    if modified.count(old) != 1:
        raise RuntimeError("上游 MMDataParallel 调用已变化，停止自动生成")
    modified = modified.replace(old, new, 1)
    DEST_ENTRY.write_text(modified, encoding="utf-8")

    patch = "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile="upstream/tools/test.py",
            tofile="scripts/step26_author_test_entry.py",
        )
    )
    (OUTPUT / "minimal_compatibility.patch").write_text(
        patch, encoding="utf-8"
    )

    workspace = OUTPUT / "workspace"
    data_dir = workspace / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_link = data_dir / "nuscenes"
    if data_link.exists() or data_link.is_symlink():
        if data_link.resolve() != (REPO / "data/nuscenes").resolve():
            raise RuntimeError("隔离目录的数据软链接指向未知位置，停止操作")
    else:
        data_link.symlink_to((REPO / "data/nuscenes").resolve(), target_is_directory=True)

    manifest = {
        "dataset": "v1.0-trainval",
        "n_frames": len(preview),
        "n_scenes": 1,
        "first_token": preview[0]["token"],
        "last_token": preview[-1]["token"],
        "checkpoint": "/mnt/g/nus/bevformer_small_seg.pth",
        "source_entry_sha256": sha256(SOURCE_ENTRY),
        "compatibility_copy_sha256": sha256(DEST_ENTRY),
        "subset_sha256": sha256(subset_path),
        "changes": ["仅跳过纯分割测试入口的 CLI 参数强制检查", "启动时将模型显式移至 CUDA"],
        "important_note": "保留上游模型、forward、DataLoader 和 IoU 计算；不是原文件的零修改执行。",
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Step 26 准备完成：", OUTPUT)
    print("数据样本：", len(preview), "帧，同一场景且连续")
    print("子集：", subset_path)
    print("上游入口兼容副本：", DEST_ENTRY)
    print("仅修改启动兼容性，源码差异保存在 minimal_compatibility.patch")
    print("Step 26 prepare: PASS")


if __name__ == "__main__":
    main()

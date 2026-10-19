#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将本地已完成实验的小型可审核结果拷贝到 Git 可追踪目录。

只复制 JSON/CSV/TXT/小型 PNG；不复制大型日志、权重、PKL、npy、npz。
原始 outputs 路径始终保留不变，缺失文件仅报告，不伪造结果。
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/drivelm_check/root_cause"
DEST = ROOT / "results/root_cause_2026-10-19"
MAX_BYTES = 3 * 1024 * 1024

FILES = {
    "step9_drivelm799/results.json": "step09_drivelm799_alignment.json",
    "step11_reference_frame/results.json": "step11_reference_frame.json",
    "step15_temporal_ab/results.json": "step15_temporal_ab.json",
    "step21_full_val/generation_summary.json": "step21_val_index_summary.json",
    "step23_full_val_preview/results.json": "step23_val30_results.json",
    "step25_full_val_6019/results.json": "step25_val6019_results.json",
    "step31_direct_verification/results.json": "step31_attention_and_geometry.json",
    "step32_calibrated_geometry/results.json": "step32_real_calibration.json",
    "step33_geometry_ab/results.json": "step33_geometry_ab.json",
    "step33_geometry_ab/per_frame.csv": "step33_per_frame.csv",
    "step28_checkpoint_identity.txt": "step28_checkpoint_identity.txt",
    "step29_environment_audit.txt": "step29_environment_audit.txt",
}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    copied = []
    missing = []
    for source, target in FILES.items():
        path = BASE / source
        if not path.is_file():
            missing.append(source)
            continue
        size = path.stat().st_size
        if size > MAX_BYTES:
            print(f"跳过超过 {MAX_BYTES // 1048576} MB 的文件：{source}")
            continue
        shutil.copy2(path, DEST / target)
        copied.append(target)
    (DEST / "README.md").write_text(
        "# 2026-10-19 根因排查证据\n\n"
        "本目录包含经筛选的小型实验输出副本；原始日志、checkpoint、"
        "完整 Val PKL、稠密预测数组保留在本地 outputs/ 下。\n\n"
        "## 已归档\n" + "".join(f"- `{name}`\n" for name in copied)
        + "\n## 当前未找到（不可视为实验已完成）\n"
        + "".join(f"- `{name}`\n" for name in missing),
        encoding="utf-8",
    )
    print(f"证据目录：{DEST}")
    print(f"已复制：{len(copied)} 项；缺失：{len(missing)} 项")
    for name in copied:
        print(" +", name)
    for name in missing:
        print(" -", name)


if __name__ == "__main__":
    main()

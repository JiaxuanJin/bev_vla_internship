#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对照 Issue #37 的 map_y 反向采样方案，自动恢复全部原始文件。"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEAD_PATH = (
    ROOT
    / "BEVFormer-seg/projects/mmdet3d_plugin/bevformer/dense_heads/"
    / "bevformer_head.py"
)
METRICS_DIR = ROOT / "outputs/bev_seg"
OUTPUT_DIR = (
    ROOT / "outputs/drivelm_check/root_cause/step43_issue37_map_y_ab"
)
RESULT_FILES = (
    "seg_minival_metrics.json",
    "seg_minival_per_frame.csv",
    "seg_minival_confusion.npy",
)
MAP_Y_PATTERN = re.compile(
    r"(?P<indent>^[ \t]*)self\.map_y\s*=\s*torch\.arange\(\s*"
    r"map_bev_start_position\[1\]\s*,\s*"
    r"map_grid_conf\[['\"]ybound['\"]\]\[1\]\s*,\s*"
    r"map_bev_resolution\[1\]\s*\)",
    re.MULTILINE,
)
FLIP_PATTERN = re.compile(
    r"(?P<indent>^[ \t]*)seg_bev\s*=\s*torch\.flip\("
    r"\s*seg_bev\s*,\s*dims\s*=\s*\[2\]\s*\)",
    re.MULTILINE,
)


def make_issue37_patch(original: str) -> str:
    """应用原讨论的两个配套改动；若版本不符则拒绝修改。"""
    if len(MAP_Y_PATTERN.findall(original)) != 1:
        raise RuntimeError("源码中的 map_y 赋值不是唯一匹配，拒绝修改")
    if len(FLIP_PATTERN.findall(original)) != 1:
        raise RuntimeError("源码中的 flip(dim=2) 不是唯一匹配，拒绝修改")

    def replace_map(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}self.map_y = torch.arange(\n"
            f"{indent}    map_grid_conf['ybound'][1] - map_bev_resolution[1],\n"
            f"{indent}    map_bev_start_position[1] - map_bev_resolution[1],\n"
            f"{indent}    -map_bev_resolution[1]\n"
            f"{indent})"
        )

    updated = MAP_Y_PATTERN.sub(replace_map, original, count=1)
    updated = FLIP_PATTERN.sub(
        lambda match: (
            f"{match.group('indent')}# Issue37: Y 采样方向反转，"
            "此处不再翻转 seg_bev"
        ),
        updated,
        count=1,
    )
    return updated


def foreground_score(path: Path) -> float | None:
    """读取现有 Mini 评估的前景指标。"""
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    value = obj.get("foreground_mIoU")
    return None if value is None else float(value)


def main() -> None:
    """安全备份、执行 81 帧 A/B、保存比较，再恢复源文件和旧指标。"""
    if not HEAD_PATH.exists():
        raise FileNotFoundError(f"找不到分割头源码：{HEAD_PATH}")

    original_bytes = HEAD_PATH.read_bytes()
    original_text = original_bytes.decode("utf-8")
    updated_text = make_issue37_patch(original_text)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HEAD_PATH, OUTPUT_DIR / "bevformer_head_original.py.bak")

    # 先保存旧评估，避免对照实验覆盖已有 Mini 基线。
    existed: dict[str, bool] = {}
    for filename in RESULT_FILES:
        source = METRICS_DIR / filename
        existed[filename] = source.exists()
        if source.exists():
            shutil.copy2(source, OUTPUT_DIR / f"baseline_{filename}")
    baseline = foreground_score(OUTPUT_DIR / "baseline_seg_minival_metrics.json")

    print("\nStep43：Issue #37 map_y 方向 + 移除 flip(dim=2)", flush=True)
    print(f"原始 Mini mIoU: {baseline * 100:.4f}%" if baseline is not None
          else "未发现原始 Mini 结果", flush=True)
    print("[1/3] 临时修改分割头，开始 81 帧 Mini 评估", flush=True)

    succeeded = False
    revised = None
    try:
        HEAD_PATH.write_text(updated_text, encoding="utf-8")
        # 新进程导入修改后的第三方模块，避免 Python 已导入模块的缓存。
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts/run_bev_seg_minival.py"),
                "--limit", "81",
                "--checkpoint", "/mnt/g/nus/bevformer_small_seg.pth",
                "--preview-indices", "",
            ],
            cwd=ROOT,
            check=True,
        )
        revised = foreground_score(METRICS_DIR / "seg_minival_metrics.json")
        for filename in RESULT_FILES:
            src = METRICS_DIR / filename
            if src.exists():
                shutil.copy2(src, OUTPUT_DIR / f"issue37_{filename}")
        succeeded = True
    finally:
        # 无论运行成功还是中途异常，都恢复源码和原始 Mini 指标。
        HEAD_PATH.write_bytes(original_bytes)
        for filename in RESULT_FILES:
            target = METRICS_DIR / filename
            backup = OUTPUT_DIR / f"baseline_{filename}"
            if existed[filename]:
                shutil.copy2(backup, target)
            elif target.exists():
                target.unlink()
        print("[2/3] 原版 bevformer_head.py 与旧 Mini 指标已恢复", flush=True)

    if succeeded:
        report = {
            "experiment": "Issue37 reversed map_y + remove feature flip(dim=2)",
            "dataset": "nuScenes v1.0-mini / mini_val, 81 frames",
            "baseline_foreground_miou": baseline,
            "issue37_foreground_miou": revised,
            "delta_percentage_points": (
                (revised - baseline) * 100
                if revised is not None and baseline is not None
                else None
            ),
            "source_restored": True,
            "original_metrics_restored": True,
        }
        (OUTPUT_DIR / "comparison.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("[3/3] A/B 对照完成", flush=True)
        print(f"原版 mIoU: {baseline * 100:.4f}%" if baseline is not None
              else "原版无指标", flush=True)
        print(f"Issue37 mIoU: {revised * 100:.4f}%" if revised is not None
              else "新结果无指标", flush=True)
        print(f"对照报告: {OUTPUT_DIR / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提取 BEVFormer-Seg 训练/推理坐标链的关键源码证据。

只读取本地源码与 Step34 CSV，不加载模型、不修改第三方仓库，
供人工核对坐标系、轴方向与地图栅格化流程。
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "BEVFormer-seg"
RESULT_DIR = ROOT / "outputs/drivelm_check/root_cause/step36_coordinate_contract"
PER_FRAME = (
    ROOT
    / "outputs/drivelm_check/root_cause/step34_full_val_orientation_ab/"
    "per_frame.csv"
)

# 保留完整关键函数，以及训练/验证入口附近的代码，行号按本地源码输出。
SOURCE_SECTIONS: dict[str, list[tuple[int, int]]] = {
    "projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py": [
        (1, 120), (145, 190), (255, 315),
    ],
    "projects/configs/bevformer/bevformer_small_seg.py": [
        (15, 42), (54, 110), (175, 250),
    ],
    "projects/mmdet3d_plugin/datasets/pipelines/rasterize.py": [(1, 120)],
    "projects/mmdet3d_plugin/datasets/pipelines/bevsegmentation.py": [
        (40, 110),
    ],
    "projects/mmdet3d_plugin/datasets/mtl_nuscenes_dataset_ego.py": [
        (255, 287), (376, 402), (470, 512),
    ],
    "projects/mmdet3d_plugin/datasets/nuscenes_dataset.py": [
        (100, 140),
    ],
    "projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py": [
        (45, 112), (210, 285),
    ],
}

PATTERNS = (
    "ego2global", "lidar2ego", "sensor2lidar", "get_map_vectors",
    "get_map_mask", "RasterizeMapVectors", "reference_points",
    "grid_sample", "feat_cropper", "rot90(", "semantic_indices",
)


def source_excerpt(path: Path, start: int, end: int) -> str:
    """按文件真实行号读取源码片段，缺失文件不报致命错误。

    Args:
        path: 源码绝对路径。
        start: 起始行号（含）。
        end: 结束行号（含）。

    Returns:
        带行号的代码块文本。
    """
    if not path.is_file():
        return f"（文件不存在：{path}）\n"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if start > len(lines):
        return "（文件行数不足）\n"
    clipped = lines[start - 1 : min(end, len(lines))]
    return "\n".join(
        f"{lineno:4d} | {line}"
        for lineno, line in enumerate(clipped, start=start)
    )


def geometric_identity_check() -> list[str]:
    """用非正方形坐标场证明原版与 Step34 候选的轴排列关系。"""
    # 使用 3x5 而非 150x150，避免正方形输入掩盖维度互换错误。
    source = np.arange(15).reshape(3, 5)
    original = np.fliplr(np.rot90(source, k=-1))
    candidate = np.flipud(original)
    transpose_ok = np.array_equal(original, source.T)
    ccw_ok = np.array_equal(candidate, np.rot90(source, k=1))
    assert transpose_ok and ccw_ok, "合成空间变换恒等式未通过"
    return [
        "- 非正方形合成网格（3×5）测试：通过。",
        "- 原版 `rot90(-1) + flip(dim=3)` = 源网格转置：通过。",
        "- Step34 在 Slicer 前增加 `flip(dim=2)` = 源网格逆时针旋转 90°：通过。",
        "- 上述只是张量轴排列恒等式，**不等于物理 LiDAR→Ego 标定验证**。",
    ]


def append_focus_frames(out: list[str]) -> None:
    """保留 Step34 中指定退化样本的完整 CSV 原始字段。"""
    if not PER_FRAME.is_file():
        out.append(f"- 未找到 CSV：`{PER_FRAME}`")
        return
    focus_indices = {3703, 3704, 3712, 3713, 3714, 3715, 3716,
                     3717, 3718, 3719, 3720, 3721}
    with PER_FRAME.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        out.append("- CSV 为空，无法提取焦点帧。")
        return
    focus = [row for row in rows if str(row.get("index", "")) in {
        str(value) for value in focus_indices
    }]
    result_csv = RESULT_DIR / "focus_frames.csv"
    with result_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(focus)
    out.append(f"- Step34 CSV 总记录数：{len(rows)}。")
    out.append(f"- 焦点样本数：{len(focus)}；保存至 `{result_csv}`。")
    out.append("- 重点对照 3715、3718、3719 帧是否属于同一场景，")
    out.append("  以及周围帧的连贯退化是否与地图内容/轨迹有关。")


def scan_relevant_locations(base: Path, patterns: Iterable[str]) -> list[str]:
    """搜索与参考系转换相关的代码位置，限制输出避免冗长。"""
    results: list[str] = []
    if not base.is_dir():
        return ["- 第三方源码目录不存在，跳过搜索。"]
    targets = tuple(patterns)
    project = base / "projects/mmdet3d_plugin"
    for path in sorted(project.rglob("*.py")):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            if any(pattern in line for pattern in targets):
                relative = path.relative_to(base)
                results.append(f"- `{relative}:{number}`: `{line.strip()[:170]}`")
                if len(results) >= 110:
                    results.append("- （仅展示前 110 处；完整源文件在本地仓库。）")
                    return results
    return results


def main() -> None:
    """写出一个可以直接上传的 Markdown 证据包。"""
    if not UPSTREAM.is_dir():
        raise FileNotFoundError(f"未找到第三方项目源码：{UPSTREAM}")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    out: list[str] = [
        "# Step 36：训练/推理坐标链源码证据",
        "",
        f"记录日期：{date.today().isoformat()}",
        "",
        "## 1. 目标与解释边界",
        "",
        "该文件提取本地实际源码、固定退化帧信息与张量轴排列恒等式。",
        "**不对根因预设结论；不修改权重、GT、分割头或任何模型文件。**",
        "",
        "## 2. 张量空间变换的合成测试",
        "",
        *geometric_identity_check(),
        "",
        "## 3. Step34 异常帧回溯",
        "",
    ]
    append_focus_frames(out)

    out.extend([
        "", "## 4. 关键源码全文片段（以本地行号为准）", "",
    ])
    for rel_path, sections in SOURCE_SECTIONS.items():
        path = UPSTREAM / rel_path
        for start, end in sections:
            out.extend([
                f"### `{rel_path}` 第 {start}–{end} 行",
                "",
                "```text",
                source_excerpt(path, start, end),
                "```",
                "",
            ])

    out.extend([
        "## 5. 与坐标转换相关的其他代码位置", "",
        *scan_relevant_locations(UPSTREAM, PATTERNS),
        "",
        "## 6. 后续判断需要核对的问题", "",
        "- `RasterizeMapVectors` 的输入 vectors 属于 Ego 还是 LiDAR 坐标？",
        "- BEV encoder 的 `(row, col)` 对应哪个物理轴及正方向？",
        "- `BevFeatureSlicer` 采用哪个 reference frame、网格中心与方向？",
        "- Train 与 Val 是否使用相同地图 rasterization 和同一坐标定义？",
        "- 原始 checkpoint 的训练代码版本与当前推理代码是否一致？",
        "- CCW90 后仍退化的帧是否集中于特定地图或场景？",
        "",
    ])

    report = RESULT_DIR / "source_evidence.md"
    report.write_text("\n".join(out), encoding="utf-8")
    print("Step36 完成：不需要 GPU，未修改上游项目。")
    print(f"坐标链证据：{report}")
    if (RESULT_DIR / "focus_frames.csv").is_file():
        print(f"退化焦点帧：{RESULT_DIR / 'focus_frames.csv'}")
    print("请上传 source_evidence.md，以便逐段定位具体坐标变换。")


if __name__ == "__main__":
    main()

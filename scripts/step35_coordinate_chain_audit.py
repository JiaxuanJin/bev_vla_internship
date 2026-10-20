#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 35：Step34 异常场景统计与训练/推理坐标链源代码取证。

不调用 GPU，不覆盖任何已有结果；仅汇总 Step34 的 per_frame.csv，
并提取本地 BEVFormer-Seg 中需要人工核对的代码位置。

运行：python scripts/step35_coordinate_chain_audit.py
输出：outputs/drivelm_check/root_cause/step35_coordinate_audit/
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEG_ROOT = ROOT / "BEVFormer-seg"
STEP34 = ROOT / "outputs/drivelm_check/root_cause/step34_full_val_orientation_ab"
REPORT_DIR = ROOT / "outputs/drivelm_check/root_cause/step35_coordinate_audit"

# 按物理坐标链检索；匹配结果只代表定位线索，不代表已发现 bug。
SOURCE_GROUPS: dict[str, tuple[str, ...]] = {
    "BEV 分割头方向与 Slicer": (
        "rot90", "torch.flip", "feat_cropper", "BevFeatureSlicer"
    ),
    "GT 地图生成与参考系": (
        "semantic_indices", "get_map_mask", "ego2global", "sensor2lidar",
        "lidar2ego", "patch_angle", "patch_box"
    ),
    "训练/测试配置": (
        "train_pipeline", "test_pipeline", "grid_conf", "xbound", "ybound",
        "bev_h_", "bev_w_"
    ),
}


def percentage(value: float) -> str:
    """小数表示的 IoU 转为百分比。"""
    return f"{100.0 * value:.4f}%"


def load_step34() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """加载真实 Step34 结果，拒绝将未完成实验当作证据。"""
    result_path = STEP34 / "results.json"
    csv_path = STEP34 / "per_frame.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not result.get("complete") or result.get("processed_frames") != 6019:
        raise ValueError("Step34 不是完整 6019 帧结果，停止分析")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        raw_rows = list(csv.DictReader(stream))
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        rows.append({
            "index": int(raw["index"]),
            "sample_token": raw["sample_token"],
            "scene_token": raw["scene_token"],
            "original_fg_miou": float(raw["original_fg_miou"]),
            "candidate_fg_miou": float(raw["candidate_fg_miou"]),
            "delta_fg_miou": float(raw["delta_fg_miou"]),
            "changed_pixels": int(raw["changed_pixels"]),
        })
    if len(rows) != 6019:
        raise ValueError(f"CSV 只有 {len(rows)} 帧，不是完整 6019 帧")
    if len({row["index"] for row in rows}) != 6019:
        raise ValueError("CSV 帧索引有重复或缺失")
    return result, rows


def source_candidates() -> list[Path]:
    """只扫描第三方项目的配置和插件 Python 源码，避免翻遍全部依赖。"""
    config = SEG_ROOT / "projects/configs/bevformer/bevformer_small_seg.py"
    project = SEG_ROOT / "projects/mmdet3d_plugin"
    found = list(project.rglob("*.py")) if project.exists() else []
    return ([config] if config.is_file() else []) + sorted(found)


def scan_sources() -> list[tuple[str, int, str, str]]:
    """返回分类、行号、路径、真实源码，不修改任何上游文件。"""
    matches: list[tuple[str, int, str, str]] = []
    for source in source_candidates():
        rel = str(source.relative_to(ROOT))
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
        is_config = "projects/configs/" in rel
        is_head = "bevformer_seg_head.py" in rel
        for number, line in enumerate(lines, 1):
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            if is_head and any(token in line for token in SOURCE_GROUPS["BEV 分割头方向与 Slicer"]):
                matches.append(("head", number, rel, cleaned))
            if any(token in line for token in SOURCE_GROUPS["GT 地图生成与参考系"]):
                matches.append(("gt", number, rel, cleaned))
            if is_config and any(token in line for token in SOURCE_GROUPS["训练/测试配置"]):
                matches.append(("config", number, rel, cleaned))
            if "class BevFeatureSlicer" in line:
                matches.append(("slicer", number, rel, cleaned))
    return matches


def format_source_evidence(matches: list[tuple[str, int, str, str]]) -> list[str]:
    """精简源码定位清单，优先保留训练/推理核心文件。"""
    out: list[str] = []
    caps = {"head": 28, "slicer": 6, "gt": 45, "config": 30}
    for key, title in (
        ("head", "分割头 BEV 方向变换"),
        ("slicer", "BevFeatureSlicer 定义"),
        ("gt", "GT 地图坐标与语义标签入口"),
        ("config", "Train/Test 配置关键入口"),
    ):
        entries = [item for item in matches if item[0] == key]
        if key == "gt":
            # GT 相关命中可能很多，优先 semantic_indices、地图 API。
            def rank(item: tuple[str, int, str, str]) -> tuple[int, str, int]:
                line = item[3]
                weight = 0 if ("semantic_indices" in line or "get_map_mask" in line) else 1
                return weight, item[2], item[1]
            entries = sorted(entries, key=rank)
        out.extend([f"### {title}", ""])
        if not entries:
            out.extend(["未在扫描范围中找到对应符号，需要检查实际项目文件路径。", ""])
            continue
        for _, number, path, line in entries[:caps[key]]:
            safe_line = line[:180].replace("`", "'")
            out.append(f"- `{path}:{number}`: `{safe_line}`")
        if len(entries) > caps[key]:
            out.append(f"- 另有 {len(entries) - caps[key]} 处命中，未在此展开。")
        out.append("")
    return out


def main() -> None:
    """输出场景例外名单与可验证的源码入口，不推断未经验证的根因。"""
    result, rows = load_step34()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["scene_token"]].append(row)
    scene_rows: list[dict[str, Any]] = []
    for scene, records in groups.items():
        total = len(records)
        avg_delta = sum(record["delta_fg_miou"] for record in records) / total
        scene_rows.append({
            "scene_token": scene,
            "frames": total,
            "improved": sum(record["delta_fg_miou"] > 0 for record in records),
            "worsened": sum(record["delta_fg_miou"] < 0 for record in records),
            "equal": sum(record["delta_fg_miou"] == 0 for record in records),
            "mean_frame_delta_pp": 100 * avg_delta,
        })
    scene_rows.sort(key=lambda row: row["mean_frame_delta_pp"])
    with (REPORT_DIR / "scene_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scene_rows[0]))
        writer.writeheader()
        writer.writerows(scene_rows)

    worsened = sorted(
        (row for row in rows if row["delta_fg_miou"] < 0),
        key=lambda row: row["delta_fg_miou"],
    )
    with (REPORT_DIR / "worsened_frames.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(worsened)

    source_hits = scan_sources()
    text: list[str] = [
        "# Step 35：方向消融的退化帧与训练/推理坐标链静态审计",
        "",
        "## 1. Step34 全量结果（累计像素 IoU）",
        "",
        f"- 验证帧数：{len(rows)}；覆盖场景：{len(groups)}。",
        f"- 原始前景 mIoU：{percentage(result['original']['foreground_miou'])}。",
        f"- CCW90 前景 mIoU：{percentage(result['physical_ccw_candidate']['foreground_miou'])}。",
        f"- 变化：{result['aggregate_delta_percentage_points']:+.4f} 个百分点。",
        f"- 逐帧改善 {result['frames_improved']} / 退化 {result['frames_worsened']} / 不变 {result['frames_equal']}。",
        "",
        "## 2. 例外场景（平均逐帧 mIoU 差值最低）",
        "",
        "注意：本节是 **逐帧 mIoU 差值的算术平均**，不是按场景像素累计的 mIoU。",
        "| scene_token | 帧数 | 改善 | 退化 | 平均逐帧差值(pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in scene_rows[:12]:
        text.append(
            f"| `{record['scene_token']}` | {record['frames']} | "
            f"{record['improved']} | {record['worsened']} | "
            f"{record['mean_frame_delta_pp']:+.3f} |"
        )
    text.extend([
        "", "## 3. 退化最明显的 12 个样本", "",
        "| index | scene_token | 原始帧 IoU | CCW90 帧 IoU | 差值(pp) |",
        "|---:|---|---:|---:|---:|",
    ])
    for row in worsened[:12]:
        text.append(
            f"| {row['index']} | `{row['scene_token']}` | "
            f"{percentage(row['original_fg_miou'])} | "
            f"{percentage(row['candidate_fg_miou'])} | "
            f"{100 * row['delta_fg_miou']:+.3f} |"
        )
    text.extend([
        "", "## 4. 源码中需要核实的坐标链入口", "",
        "以下仅为静态代码位置取证。**尚未证明具体是哪个源码环节错误**。",
        "",
    ])
    text.extend(format_source_evidence(source_hits))
    text.extend([
        "## 5. 下一步需核验的事实", "",
        "- Train 与 Val 的 `semantic_indices` 生成逻辑、地图参考系和栅格方向是否一致。",
        "- BEVFormer encoder 输出 BEV 的轴顺序，分割头 `rot90/flip` 与 Slicer 的轴和取样范围。",
        "- GT 以 Ego 为参考、预测 BEV 以 LiDAR 为参考时，是否使用了真实外参进行完整坐标映射。",
        "- 针对退化帧保存预测/GT 叠图以解释 150 帧例外，并检查 `ped_crossing` 提升较小的原因。",
        "- 若需修复，必须明确训练端与推理端一致的映射定义，重新跑固定评估协议。",
        "",
    ])
    output_path = REPORT_DIR / "audit.md"
    output_path.write_text("\n".join(text), encoding="utf-8")
    print("Step 35：CPU 静态审计完成")
    print(f"场景数：{len(groups)}，退化帧：{len(worsened)}")
    print(f"源码命中：{len(source_hits)} 处")
    print(f"审计报告：{output_path}")
    print(f"场景排名：{REPORT_DIR / 'scene_summary.csv'}")
    print(f"退化样本：{REPORT_DIR / 'worsened_frames.csv'}")
    print("注意：本步不改模型，不会产生新的模型评估指标。")


if __name__ == "__main__":
    main()

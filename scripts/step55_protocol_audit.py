#!/usr/bin/env python3
"""检查 BEVFormer-seg 的数据与分割协议，输出只读审计快照。

运行：python3 scripts/step55_protocol_audit.py
可选：python3 scripts/step55_protocol_audit.py --project ~/projects/bev_vla

脚本不会加载模型、不会改动 GT、权重或实验结果。
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED = {
    "bev_h_": 150,
    "bev_w_": 150,
    "queue_length": 3,
    "map_grid_conf": {
        "xbound": [-30.0, 30.0, 0.15],
        "ybound": [-15.0, 15.0, 0.15],
        "zbound": [-10.0, 10.0, 20.0],
        "dbound": [1.0, 60.0, 1.0],
    },
}
OFFICIAL_CONFIG_URL = (
    "https://github.com/Bin-ze/BEVFormer_segmentation_detection/"
    "blob/master/projects/configs/bevformer/bevformer_small_seg.py"
)


def digest(path: Path) -> str:
    """对文本源码计算哈希，方便定位本地版本差异。"""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def path_info(path: Path, hash_small: bool = False) -> dict[str, Any]:
    """记录路径、存在性、大小与可选文本哈希。"""
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        size = path.stat().st_size
        result["bytes"] = size
        if hash_small and size <= 10 * 1024 * 1024:
            result["sha256"] = digest(path)
    return result


def find_assignment(text: str, name: str) -> Any:
    """仅提取常量赋值，不执行 config 代码。"""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(isinstance(t, ast.Name) and t.id == name for t in targets):
            continue
        try:
            return ast.literal_eval(node.value)
        except (ValueError, TypeError, SyntaxError, RecursionError):
            return None
    return None


def check_config(path: Path) -> dict[str, Any]:
    """抽取 BEV / 分割配置，并保留需要人工复核的条件。"""
    result = path_info(path, hash_small=True)
    if not path.is_file():
        return result
    text = path.read_text(encoding="utf-8", errors="replace")
    extracted: dict[str, Any] = {}
    fields = ("bev_h_", "bev_w_", "queue_length", "map_grid_conf", "point_cloud_range")
    for key in fields:
        extracted[key] = find_assignment(text, key)
    checks = {
        "video_test_mode_true": bool(re.search(r"video_test_mode\s*=\s*True", text)),
        "map_rasterization_present": "RasterizeMapVectors" in text,
        "segmentation_four_outputs": bool(re.search(r"outC\s*=\s*4", text)),
        "segmentation_only": bool(re.search(r"'seg'\s*:\s*True", text))
        and bool(re.search(r"'det'\s*:\s*False", text)),
        "bev_temporal_rotate": bool(re.search(r"rotate_prev_bev\s*=\s*True", text)),
        "bev_temporal_shift": bool(re.search(r"use_shift\s*=\s*True", text)),
    }
    mismatches = [
        key for key, expected in EXPECTED.items()
        if extracted.get(key) is not None and extracted[key] != expected
    ]
    result.update(
        {"config": extracted, "checks": checks, "differences_from_upstream": mismatches}
    )
    return result


def map_header(path: Path) -> dict[str, Any]:
    """仅读地图 JSON 开头，不把大型地图文件整体装入内存。"""
    item = path_info(path)
    try:
        with path.open("rb") as handle:
            header = handle.read(1024 * 1024).decode("utf-8", errors="replace")
        found = re.search(r'"version"\s*:\s*"([^"]+)"', header)
        item["map_version"] = found.group(1) if found else "未在文件头部识别，需手动核对"
    except OSError as exc:
        item["read_error"] = str(exc)
    return item


def candidate_data_roots(project: Path) -> list[Path]:
    """扫描常见数据位置；若使用其他磁盘，可用 --data-root 指定。"""
    roots = [
        project / "data/nuscenes",
        project / "BEVFormer-seg/data/nuscenes",
        project / "BEVFormer/data/nuscenes",
        Path.home() / "projects/bevformer_author_clean_5080/data/nuscenes",
        Path("/mnt/g/nus"),
        Path("/mnt/g/BEV_VLA_Datasets/nuscenes"),
    ]
    return list(dict.fromkeys(path.expanduser() for path in roots))


def inspect_data_root(root: Path) -> dict[str, Any]:
    """记录数据版本、地图 expansion、pkl，不改文件。"""
    item: dict[str, Any] = {"root": str(root), "exists": root.is_dir()}
    if not root.is_dir():
        return item
    item["dataset_folders"] = [
        name for name in ("v1.0-mini", "v1.0-trainval", "v1.0-test")
        if (root / name).is_dir()
    ]
    item["map_expansion_json"] = [
        map_header(path) for path in sorted((root / "maps/expansion").glob("*.json"))
    ]
    item["annotation_pkls"] = [
        path_info(path)
        for path in sorted(root.glob("nuscenes_infos_temporal_*.pkl"))
    ]
    return item


def main() -> None:
    """写入 JSON 和 Markdown 复核摘要。"""
    parser = argparse.ArgumentParser(description="BEV 分割训练/评估协议只读审计")
    parser.add_argument(
        "--project", type=Path, default=Path.home() / "projects/bev_vla"
    )
    parser.add_argument("--data-root", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    out = args.out or (project / "outputs/protocol_audit_2026-10-21")
    out.mkdir(parents=True, exist_ok=True)

    config_relative = "projects/configs/bevformer/bevformer_small_seg.py"
    author_root = Path.home() / "projects/bevformer_author_clean_5080"
    code_roots = [project / "BEVFormer-seg", author_root]
    configs = [check_config(root / config_relative) for root in code_roots]
    gt_relative = "projects/mmdet3d_plugin/datasets/pipelines/rasterize.py"
    dataset_relative = "projects/mmdet3d_plugin/datasets/nuscenes_dataset.py"
    ground_truth_sources = [
        path_info(root / rel, hash_small=True)
        for root in code_roots for rel in (gt_relative, dataset_relative)
    ]
    hardcoded_versions = {}
    for source in ground_truth_sources:
        source_path = Path(source["path"])
        if source_path.is_file():
            content = source_path.read_text(encoding="utf-8", errors="replace")
            hardcoded_versions[str(source_path)] = re.findall(
                r"NuScenes\(\s*version\s*=\s*['\"]([^'\"]+)", content
            )
    data_roots = candidate_data_roots(project) + list(args.data_root)
    data = [inspect_data_root(root) for root in dict.fromkeys(data_roots)]
    ckpt = path_info(Path("/mnt/g/nus/bevformer_small_seg.pth"))
    ckpt["recorded_sha256_from_2026_10_20_logbook"] = (
        "b9e369ba2e4514050fefe914f57565a31b918edaa735958a1f877184ad75dc68"
    )

    result = {
        "audit": "Step55: BEVFormer-seg protocol snapshot",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project": str(project),
        "official_reference": OFFICIAL_CONFIG_URL,
        "official_expected": EXPECTED,
        "local_configs": configs,
        "gt_sources": ground_truth_sources,
        "nuscenes_constructor_version_strings": hardcoded_versions,
        "data_roots": data,
        "checkpoint": ckpt,
        "known_baseline": {
            "data_split": "nuScenes v1.0-mini / mini_val",
            "frames": 81,
            "temporal": True,
            "foreground_miou_pct": 6.38,
            "divider_pct": 2.23,
            "ped_crossing_pct": 8.73,
            "boundary_pct": 8.17,
            "source": "docs/logbook_2026-10-20.md Step50",
        },
    }
    json_path = out / "step55_protocol_snapshot.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Step55：训练产物、nuScenes Map 与评估协议只读审计",
        "",
        "本报告只检查本机文件和静态配置，不修改 checkpoint、GT 或模型。",
        "",
        "## 冻结的评估基线",
        "",
        "- nuScenes v1.0-mini，Mini Val 81 帧，六相机，时序开启。",
        "- 前景三类 mIoU：6.38%；Divider 2.23%、Ped Crossing 8.73%、Boundary 8.17%。",
        "- 不与 DriveLM799 独立帧或官方论文完整 Val 成绩直接比较。",
        "",
        "## 本地 Config",
        "",
    ]
    for item in configs:
        lines.append(f"- `{item['path']}`：{'找到' if item['exists'] else '未找到'}")
        if item.get("config"):
            fields = item["config"]
            lines.append(
                f"  - bev_h={fields.get('bev_h_')}，bev_w={fields.get('bev_w_')}，"
                f"queue_length={fields.get('queue_length')}，"
                f"map_grid_conf={fields.get('map_grid_conf')}"
            )
            lines.append(f"  - 与公开配置的已提取字段差异：{item['differences_from_upstream']}")
            lines.append(f"  - 结构检查：{item['checks']}")
    lines.extend(["", "## nuScenes 数据及地图版本", ""])
    for item in data:
        if not item["exists"]:
            continue
        lines.append(f"- 数据根目录：`{item['root']}`；数据版本目录：{item['dataset_folders']}")
        if not item["map_expansion_json"]:
            lines.append("  - 未在该根目录发现 maps/expansion/*.json")
        for mp in item["map_expansion_json"]:
            lines.append(f"  - 地图：`{mp['path']}`；版本：{mp.get('map_version')}")
        for pkl in item["annotation_pkls"]:
            lines.append(f"  - PKL：`{pkl['path']}`；字节数：{pkl.get('bytes')}")
    lines.extend([
        "", "## 重点判断", "",
        "1. NuScenes 数据集版本、Map Expansion 版本、nuscenes-devkit 包版本分别记录。",
        "2. 公开 config 的 GT 使用 RasterizeMapVectors，语义类别含 background + 3 foreground。",
        "3. 必须继续确认发布权重对应的数据分割、地图包、训练代码 commit、评估命令。",
        "4. Step54 推理端校正后的 mIoU 下降，不能当作正式修复。",
        "5. 若权重发布协议仍不可核实，标注 6.38% 为受限实验基线，阶段二保持严格同条件 A/B。",
        "", "## 官方参考", "", f"- {OFFICIAL_CONFIG_URL}",
        "- https://github.com/Bin-ze/BEVFormer_segmentation_detection",
        "- https://github.com/nutonomy/nuscenes-devkit#map-expansion",
        "",
    ])
    markdown_path = out / "step55_protocol_review.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n=== Step55：只读协议审计 ===")
    print("项目目录：", project)
    for cfg in configs:
        print("Config:", "OK" if cfg["exists"] else "MISSING", cfg["path"])
        if cfg.get("config"):
            print("  已提取字段差异:", cfg["differences_from_upstream"])
    for d in data:
        if d["exists"]:
            print("数据根目录：", d["root"])
            print("  数据版本：", d["dataset_folders"])
            print("  地图扩展 JSON：", len(d["map_expansion_json"]))
            for m in d["map_expansion_json"]:
                print("   ", Path(m["path"]).name, "version=", m.get("map_version"))
    print("保存 JSON：", json_path)
    print("保存审计 Markdown：", markdown_path)
    print("\n说明：没有对模型进行推理，也没有宣称排查完成。")


if __name__ == "__main__":
    main()

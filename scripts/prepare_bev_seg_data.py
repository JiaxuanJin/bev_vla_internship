#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""准备独立的 BEVFormer-Seg 数据目录，并检查 nuScenes Mini 与 Trainval 元数据关联。

只创建符号链接，不复制大型图像、不覆盖原数据、不生成 Markdown。
运行位置：~/projects/bev_vla（也可以从其他目录运行）。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from datetime import datetime

ROOT = Path.home() / "projects" / "bev_vla"
SOURCE = ROOT / "BEVFormer" / "data" / "nuscenes"
FULL = ROOT / "data" / "nuscenes_trainval_meta" / "v1.0-trainval"
MAP = ROOT / "data" / "nuscenes_map_expansion"
STAGE = ROOT / "data" / "bev_seg_nuscenes"
SEG = ROOT / "BEVFormer-seg"
SUMMARY = ROOT / "outputs" / "bev_seg" / "seg_data_precheck.json"


def link_safe(dst: Path, target: Path, required: bool = True) -> bool:
    """只在目标存在、目的位置未占用时创建软链接；已正确链接则保持不动。"""
    if not target.exists():
        print(f"[{'缺失' if required else '跳过'}] {target}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if dst.resolve() == target.resolve():
            print(f"[已存在] {dst}")
            return True
        raise RuntimeError(f"已有其他目标的符号链接，不自动覆盖：{dst} -> {dst.resolve()}")
    if dst.exists():
        raise RuntimeError(f"目标路径已有真实文件/目录，不自动覆盖：{dst}")
    dst.symlink_to(target.resolve(), target_is_directory=target.is_dir())
    print(f"[已链接] {dst} -> {target.resolve()}")
    return True


def main() -> None:
    print("========== BEVFormer-Seg 数据集准备 ==========")
    required = {
        "原始 Mini 数据目录": SOURCE,
        "完整 Trainval 元数据": FULL / "sample.json",
        "矢量地图扩展包": MAP / "expansion" / "boston-seaport.json",
        "分割代码仓库": SEG / "projects" / "configs" / "bevformer" / "bevformer_small_seg.py",
        "Mini 验证 PKL": SOURCE / "nuscenes_infos_temporal_val.pkl",
    }
    missing = [label for label, p in required.items() if not p.exists()]
    for label, path in required.items():
        print(f"[{'存在' if path.exists() else '缺失'}] {label}: {path}")
    if missing:
        raise SystemExit(f"[停止] 缺少必要文件：{', '.join(missing)}；没有修改任何数据。")

    if STAGE.is_symlink():
        raise SystemExit(f"[停止] {STAGE} 为现有符号链接，为防止写入其他目录请先检查。")
    STAGE.mkdir(parents=True, exist_ok=True)
    links = {
        "v1.0-trainval": FULL,
        "v1.0-mini": SOURCE / "v1.0-mini",
        "samples": SOURCE / "samples",
        "sweeps": SOURCE / "sweeps",
        "nuscenes_infos_temporal_train.pkl": SOURCE / "nuscenes_infos_temporal_train.pkl",
        "nuscenes_infos_temporal_val.pkl": SOURCE / "nuscenes_infos_temporal_val.pkl",
    }
    print("\n[1/3] 连接原始数据与元数据")
    for name, target in links.items():
        link_safe(STAGE / name, target, required=name not in ("sweeps",))

    print("\n[2/3] 连接地图与分割仓库")
    map_root = STAGE / "maps"
    for png in sorted((SOURCE / "maps").glob("*.png")):
        link_safe(map_root / png.name, png)
    link_safe(map_root / "expansion", MAP / "expansion")
    link_safe(map_root / "basemap", MAP / "basemap", required=False)
    link_safe(SEG / "data" / "nuscenes", STAGE)

    print("\n[3/3] 对比 Mini 验证索引与完整元数据（仅检查，不启动模型）")
    # 这些 PKL 来自用户本地已经生成的可信预处理结果，不加载未知来源的 pickle。
    with (SOURCE / "nuscenes_infos_temporal_val.pkl").open("rb") as f:
        val = pickle.load(f)
    infos = val.get("infos", [])
    val_tokens = {x.get("token") for x in infos if x.get("token")}
    with (FULL / "sample.json").open(encoding="utf-8") as f:
        full_samples = json.load(f)
    trainval_tokens = {x["token"] for x in full_samples if "token" in x}
    overlap = val_tokens & trainval_tokens
    bad_files = []
    for info in infos:
        for camera, data in info.get("cams", {}).items():
            p = Path(data.get("data_path", ""))
            candidates = [p if p.is_absolute() else SEG / p,
                          STAGE / "samples" / camera / p.name]
            if not any(x.is_file() for x in candidates):
                bad_files.append({"token": info.get("token"), "camera": camera, "path": str(p)})
                if len(bad_files) >= 12:
                    break
        if len(bad_files) >= 12:
            break

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage_root": str(STAGE),
        "segmentation_repo_data_link": str(SEG / "data" / "nuscenes"),
        "validation_pkl_metadata": val.get("metadata", {}),
        "mini_val_samples": len(val_tokens),
        "full_trainval_sample_count": len(trainval_tokens),
        "mini_val_tokens_in_trainval_metadata": len(overlap),
        "mini_val_tokens_absent_from_trainval_metadata": len(val_tokens - trainval_tokens),
        "camera_file_missing_examples_max12": bad_files,
        "warning": "准备数据目录不等于已经生成分割标签或计算 mIoU。",
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Mini 验证集样本数：{len(val_tokens)}")
    print(f"其中可在 Trainval 元数据定位：{len(overlap)}")
    print(f"缺失相机图像示例数（至多 12）：{len(bad_files)}")
    print(f"[已保存] {SUMMARY}")
    if len(overlap) < len(val_tokens) or bad_files:
        print("[注意] 有数据关联问题，下一步先修复，不应直接运行完整评估。")
    else:
        print("[通过] 数据软链接和基础关联检查完成，可以继续测试分割数据加载器。")


if __name__ == "__main__":
    main()

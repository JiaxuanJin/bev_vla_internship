#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于上游 BEVFormer-Seg 转换逻辑生成 6019 帧 nuScenes Val PKL。

运行：python -u scripts/step21_generate_full_val.py

仅处理官方 Val 场景，不触碰已有的 Mini PKL、DriveLM799 PKL 或训练集。
生成新文件位于 outputs/drivelm_check/root_cause/step21_full_val/。
"""

import hashlib
import importlib
import json
import pickle
import sys
from pathlib import Path

from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

ROOT = Path(__file__).resolve().parents[1]
SEG_REPO = ROOT / "BEVFormer-seg"
DATA_ROOT = SEG_REPO / "data/nuscenes"
OUT_DIR = ROOT / "outputs/drivelm_check/root_cause/step21_full_val"
OUT_PKL = OUT_DIR / "nuscenes_infos_temporal_val_full6019.pkl"
OUT_META = OUT_DIR / "generation_summary.json"
CAN_BUS_PARENTS = (
    Path("/mnt/g/BEV_VLA_Datasets"),
    Path("/mnt/g/BEV_VLA_Datasets/nuscenes"),
    SEG_REPO / "data",
)


class ValOnlyNuScenes:
    """仅让上游转换器遍历 Val 样本，其余 SDK 查表行为保持不变。"""

    def __init__(self, original: NuScenes, val_samples: list[dict]):
        self._original = original
        self.sample = val_samples

    def __getattr__(self, name: str):
        return getattr(self._original, name)


def file_sha256(path: Path) -> str:
    """计算生成文件的 SHA256，便于师傅核对复现实验身份。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_can_bus_parent() -> Path:
    """NuScenesCanBus 需要传入含 can_bus 子目录的父目录。"""
    for parent in CAN_BUS_PARENTS:
        if (parent / "can_bus").is_dir():
            return parent
    raise FileNotFoundError("找不到 CAN bus，期望的父目录: " + str(CAN_BUS_PARENTS))


def main() -> None:
    if not SEG_REPO.is_dir():
        raise FileNotFoundError(f"缺少 BEVFormer-Seg 源码目录: {SEG_REPO}")
    if OUT_PKL.exists():
        raise FileExistsError(f"已有目标 PKL，拒绝覆盖: {OUT_PKL}")

    can_bus_parent = locate_can_bus_parent()
    print("[1/5] 加载官方 nuScenes Trainval 元数据", flush=True)
    print(f"      数据根目录: {DATA_ROOT}", flush=True)
    print(f"      CAN bus 父目录: {can_bus_parent}", flush=True)
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(DATA_ROOT), verbose=False)

    val_scene_names = set(splits.val)
    val_scene_tokens = {
        scene["token"] for scene in nusc.scene if scene["name"] in val_scene_names
    }
    val_samples = [
        sample for sample in nusc.sample
        if sample["scene_token"] in val_scene_tokens
    ]
    print(f"      Val 场景: {len(val_scene_tokens)}，Val 帧: {len(val_samples)}", flush=True)
    if len(val_scene_tokens) != 150 or len(val_samples) != 6019:
        raise RuntimeError("Val 场景或帧数量不符合预期，停止生成")

    print("[2/5] 注册作者原始转换器（仅调用 Val 信息生成函数）", flush=True)
    sys.path.insert(0, str(SEG_REPO))
    converter = importlib.import_module("tools.data_converter.nuscenes_converter")
    can_bus = NuScenesCanBus(dataroot=str(can_bus_parent))
    val_only_nusc = ValOnlyNuScenes(nusc, val_samples)

    print("[3/5] 生成标准 Val 字段（约 6019 帧，时间视磁盘性能而定）", flush=True)
    # 完全复用上游相机外参、LiDAR、CAN bus 和历史 sweeps 转换实现。
    # 注意：对其进行的唯一改动是限制遍历样本为官方 Val，不生成 train/test PKL。
    train_infos, val_infos = converter._fill_trainval_infos(
        val_only_nusc,
        can_bus,
        train_scenes=set(),
        val_scenes=val_scene_tokens,
        test=False,
        max_sweeps=10,
    )
    if train_infos or len(val_infos) != 6019:
        raise RuntimeError(
            f"意外输出数量: train={len(train_infos)}, val={len(val_infos)}"
        )
    if len({row["token"] for row in val_infos}) != 6019:
        raise RuntimeError("Val PKL 存在重复 sample token")

    print("[4/5] 校验全部六路相机及关键帧 LiDAR 路径", flush=True)
    expected_cameras = {
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK",
        "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
    }
    for idx, info in enumerate(val_infos):
        if set(info["cams"]) != expected_cameras:
            raise RuntimeError(f"样本 {idx}: 六路摄像头字段不完整")
        paths = [info["lidar_path"]] + [
            value["data_path"] for value in info["cams"].values()
        ]
        for value in paths:
            path = Path(str(value))
            if not any(candidate.is_file() for candidate in (path, SEG_REPO / path, DATA_ROOT / path)):
                raise FileNotFoundError(f"样本 {idx} 引用不存在的文件: {value}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUT_DIR / "nuscenes_infos_temporal_val_full6019.tmp"
    payload = {"infos": val_infos, "metadata": {"version": "v1.0-trainval"}}
    print("[5/5] 保存独立 Val PKL 和复现摘要（绝不覆盖 Mini）", flush=True)
    try:
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(OUT_PKL)
    finally:
        temporary.unlink(missing_ok=True)

    summary = {
        "dataset_version": "v1.0-trainval",
        "split": "official nuScenes val",
        "num_scenes": len(val_scene_tokens),
        "num_samples": len(val_infos),
        "source_repo": "Bin-ze/BEVFormer_segmentation_detection",
        "source_converter": "tools/data_converter/nuscenes_converter.py",
        "generator": "_fill_trainval_infos (official val samples only)",
        "can_bus_parent": str(can_bus_parent),
        "output_pkl": str(OUT_PKL),
        "output_size_bytes": OUT_PKL.stat().st_size,
        "sha256": file_sha256(OUT_PKL),
        "original_mini_pkl_untouched": True,
    }
    OUT_META.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n===== 生成成功 =====", flush=True)
    print(f"Val 场景数: {len(val_scene_tokens)}", flush=True)
    print(f"Val 帧数: {len(val_infos)}", flush=True)
    print(f"PKL: {OUT_PKL}", flush=True)
    print(f"SHA256: {summary['sha256']}", flush=True)
    print(f"摘要: {OUT_META}", flush=True)
    print("注意：此步仅生成评估索引，不代表模型已完成官方 Val 评估。", flush=True)


if __name__ == "__main__":
    main()

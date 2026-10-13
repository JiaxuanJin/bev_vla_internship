#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DriveLM 799 帧：使用 BEVFormer 官方测试数据管线验证新 PKL（无 GPU 推理）。

在项目根目录运行：
    python scripts/test_drivelm_799_bevformer_loader.py --indices 0
    python scripts/test_drivelm_799_bevformer_loader.py --indices 0,399,798

读取之前构建的 independent PKL，不修改检测/分割仓库或 nuScenes Mini。
输出仅包含控制台信息及 outputs/drivelm_check/drivelm_799_dataloader_smoke.json。
注意：这验证数据加载，不是模型预测成功或 mAP/mIoU 成绩。
"""

import argparse
import copy
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer"
CONFIG = REPO / "projects/configs/bevformer/bevformer_small.py"
PKL = ROOT / "data/drivelm/nuscenes_infos_temporal_drivelm799_independent.pkl"
REPORT = ROOT / "outputs/drivelm_check/drivelm_799_dataloader_smoke.json"
CAMERAS = {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}


def unpack(value):
    """只展开单元素测试增强列表和 MMCV DataContainer；不擅自合并增强。"""
    from mmcv.parallel import DataContainer
    for _ in range(8):
        if isinstance(value, DataContainer):
            value = value.data
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        else:
            break
    return value


def check_frame(dataset, index):
    import torch

    info = dataset.data_infos[index]
    if set(info.get("cams", {})) != CAMERAS:
        raise ValueError(f"PKL 第 {index} 帧相机通道不完整")
    # 仅使用磁盘上的图像，LiDAR 路径为参考坐标用途，不尝试读取不存在的点云 .bin。
    for cam, record in info["cams"].items():
        image = Path(record["data_path"])
        if not image.is_file() or image.stat().st_size == 0:
            raise FileNotFoundError(f"第 {index} 帧 {cam} 图片缺失: {image}")
        k = np.asarray(record["cam_intrinsic"], dtype=np.float64)
        r = np.asarray(record["sensor2lidar_rotation"], dtype=np.float64)
        t = np.asarray(record["sensor2lidar_translation"], dtype=np.float64)
        if k.shape != (3, 3) or r.shape != (3, 3) or t.shape != (3,):
            raise ValueError(f"第 {index} 帧 {cam} 标定矩阵形状有误")
        if not (np.isfinite(k).all() and np.isfinite(r).all() and np.isfinite(t).all()):
            raise ValueError(f"第 {index} 帧 {cam} 标定出现非有限数值")

    # 真正执行 BEVFormer 官方 test_pipeline（图片读取、预处理、封装），不是只检查文件名。
    sample = dataset[index]
    if sample is None or not isinstance(sample, dict):
        raise RuntimeError(f"第 {index} 帧数据管线返回异常: {type(sample).__name__}")
    if "img" not in sample or "img_metas" not in sample:
        raise KeyError(f"第 {index} 帧缺失必要输出字段，得到 {list(sample)}")

    img = unpack(sample["img"])
    meta = unpack(sample["img_metas"])
    if not torch.is_tensor(img):
        raise TypeError(f"第 {index} 帧图像应为 Tensor，实际: {type(img).__name__}")
    if img.ndim != 4 or img.shape[0] != 6 or img.shape[1] != 3:
        raise ValueError(f"第 {index} 帧图像应为 (6,3,H,W)，实际: {tuple(img.shape)}")
    if not torch.isfinite(img).all().item():
        raise ValueError(f"第 {index} 帧图像存在 NaN/Inf")
    if not isinstance(meta, dict):
        raise TypeError(f"第 {index} 帧 img_metas 应为 dict，实际: {type(meta).__name__}")

    matrices = meta.get("lidar2img")
    if not isinstance(matrices, (list, tuple)) or len(matrices) != 6:
        raise ValueError(f"第 {index} 帧 img_metas 缺少 6 份 lidar2img 标定")
    for j, mat in enumerate(matrices):
        a = np.asarray(mat, dtype=np.float64)
        if a.shape != (4, 4) or not np.isfinite(a).all():
            raise ValueError(f"第 {index} 帧第 {j} 份 lidar2img 标定不合法")

    token = str(info["token"])
    if meta.get("sample_idx") != token:
        raise ValueError(f"第 {index} 帧 token 不一致：PKL={token}, meta={meta.get('sample_idx')}")
    if not str(info["scene_token"]).startswith("drive_independent_"):
        raise ValueError("当前测试必须使用独立帧 PKL，避免将稀疏帧当作连续时序")

    print(f"[通过] index={index} | token={token} | images={tuple(img.shape)} | "
          f"lidar2img=6/6 | scene=independent", flush=True)
    return {"index": index, "sample_token": token, "image_shape": list(img.shape),
            "meta_keys": sorted(meta.keys()), "camera_count": 6,
            "independent_scene_token": str(info["scene_token"]), "status": "passed"}


def main():
    parser = argparse.ArgumentParser(description="DriveLM 799 帧 BEVFormer 数据加载冒烟测试")
    parser.add_argument("--indices", default="0", help="测试样本下标，逗号分隔；如 0,399,798")
    parser.add_argument("--pkl", type=Path, default=PKL, help="默认使用独立帧 PKL")
    args = parser.parse_args()
    indices = [int(s.strip()) for s in args.indices.split(",") if s.strip()]
    if not indices:
        raise ValueError("请通过 --indices 指定至少一个帧序号")
    if not CONFIG.is_file():
        raise FileNotFoundError(f"找不到 BEVFormer 模型配置：{CONFIG}")
    pkl = args.pkl.resolve()
    if not pkl.is_file():
        raise FileNotFoundError(f"找不到 PKL：{pkl}")

    # 从原始检测仓库注册插件，不导入 BEVFormer-seg，避免同名注册冲突。
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    print("[1/3] 加载 BEVFormer-Small 原始检测配置并注册插件...", flush=True)
    importlib.import_module("projects.mmdet3d_plugin")
    cfg = Config.fromfile(str(CONFIG))
    val_cfg = copy.deepcopy(cfg.data.test)
    val_cfg["ann_file"] = str(pkl)
    val_cfg["test_mode"] = True
    val_cfg.pop("samples_per_gpu", None)
    val_cfg.pop("workers_per_gpu", None)
    # PKL 中相机路径是绝对路径，不需要覆盖项目原有 data_root。
    if not val_cfg.get("modality", {}).get("use_camera", False) or val_cfg.get("modality", {}).get("use_lidar", True):
        raise ValueError("检测配置不是 use_camera=True 且 use_lidar=False，请勿直接尝试")

    print("[2/3] 构建检测验证 Dataset...", flush=True)
    dataset = build_dataset(val_cfg)
    size = len(dataset)
    print(f"[数据集] 读取到 {size} 帧", flush=True)
    if size != 799:
        raise ValueError(f"预期 799 帧，实际 {size}；停止后续检查")

    print(f"[3/3] 调用官方 test_pipeline，检查帧 {indices}...", flush=True)
    rows = []
    for i in indices:
        if not 0 <= i < size:
            raise IndexError(f"超出样本下标范围：{i}, 有效范围 0 ~ {size-1}")
        rows.append(check_frame(dataset, i))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"dataset": "DriveLM v1.1 val 799 independent-frame index",
                                  "pkl": str(pkl), "config": str(CONFIG),
                                  "total_frames": size, "tested": rows,
                                  "scope": "Only data loader and image preprocessing; no GPU inference or detection metrics."},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[保存] {REPORT}", flush=True)
    print("[成功] 指定帧的真实图像读取和 BEVFormer test_pipeline 测试通过。", flush=True)
    print("[提醒] 这尚不是 GPU 推理，也不是 DriveLM mAP/mIoU。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[失败] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)

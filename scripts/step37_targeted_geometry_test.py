#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 37：固定 Val 样本上比较原始、转置和真实外参映射。

说明：
    A = 本地实际 BEVFormerHead.forward 的原始 CCW90 方向。
    B = A 在 feat_cropper 前额外 flip(dim=2)，等价于源 BEV 转置。
    C = 取消 A 的纯索引旋转，按逐帧 LiDAR->Ego 外参将源 BEV
        重采样至 Ego 网格，再送入原始 feat_cropper。

三个分支使用相同的 checkpoint、图像和 GT，且均关闭历史 BEV。
C 为以坐标定义为前提的物理对照，不代表 checkpoint 已兼容此映射。
不修改第三方源码、权重或已有 Step34 输出。
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer-seg"
CONFIG = REPO / "projects/configs/bevformer/bevformer_small_seg.py"
PKL = (
    ROOT
    / "outputs/drivelm_check/root_cause/step21_full_val/"
    "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
)
CKPT = Path("/mnt/g/nus/bevformer_small_seg.pth")
OUTPUT = ROOT / "outputs/drivelm_check/root_cause/step37_targeted_geometry"
DEFAULT_INDICES = "0,300,900,1500,2400,3000,3600,3715,3718,3719,4500,6018"
CLASSES = ("background", "divider", "ped_crossing", "boundary")
VARIANTS = ("original_ccw90", "transpose_candidate", "lidar_to_ego_grid")


def confusion_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """计算 4 类混淆矩阵，GT 为行，预测为列。"""
    if gt.shape != (200, 400) or pred.shape != gt.shape:
        raise ValueError(f"地图尺寸异常：GT={gt.shape}, pred={pred.shape}")
    gt = gt.astype(np.int64)
    pred = pred.astype(np.int64)
    if gt.min() < 0 or gt.max() > 3 or pred.min() < 0 or pred.max() > 3:
        raise ValueError("只支持标签 0、1、2、3")
    return np.bincount((gt * 4 + pred).ravel(), minlength=16).reshape(4, 4)


def metric(matrix: np.ndarray) -> dict[str, Any]:
    """在累计像素混淆矩阵上计算各类别 IoU。"""
    tp = matrix.diagonal()
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - tp
    values = np.divide(
        tp, union, out=np.zeros(4, dtype=float), where=union > 0
    )
    return {
        "foreground_miou": float(values[1:].mean()),
        "class_iou": dict(zip(CLASSES, values.astype(float).tolist())),
        "confusion_gt_rows_pred_cols": matrix.tolist(),
    }


def lidar_to_ego_sampling_grid(
    height: int,
    width: int,
    point_cloud_range: list[float],
    rotation: np.ndarray,
    translation: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """为 Ego 输出格点构造对应的 LiDAR 输入归一化采样网格。

    Encoder 中 W 轴是 LiDAR X，H 轴是 LiDAR Y。
    对每个 Ego 坐标使用 p_lidar = R.T @ (p_ego - t)。
    源/目标网格均采用 BEV Encoder 的像素中心与 point_cloud_range。
    配合 F.grid_sample(align_corners=False) 使用。
    """
    xmin, ymin, _, xmax, ymax, _ = map(float, point_cloud_range)
    xs = xmin + (np.arange(width) + 0.5) * (xmax - xmin) / width
    ys = ymin + (np.arange(height) + 0.5) * (ymax - ymin) / height
    ego_y, ego_x = np.meshgrid(ys, xs, indexing="ij")
    ego = np.stack((ego_x, ego_y, np.zeros_like(ego_x)), axis=0)
    lidar = rotation.T @ (ego.reshape(3, -1) - translation[:, None])
    lidar_x = lidar[0].reshape(height, width)
    lidar_y = lidar[1].reshape(height, width)
    normalized_x = 2.0 * (lidar_x - xmin) / (xmax - xmin) - 1.0
    normalized_y = 2.0 * (lidar_y - ymin) / (ymax - ymin) - 1.0
    grid = np.stack((normalized_x, normalized_y), axis=-1)
    return torch.as_tensor(grid[None], device=device, dtype=dtype)


def run_self_test() -> None:
    """先用人工坐标验证方向恒等式和零外参的网格采样。"""
    raw = torch.arange(15, dtype=torch.float32).reshape(1, 1, 3, 5)
    ccw = torch.rot90(raw, k=1, dims=(-2, -1))
    restored = torch.rot90(ccw, k=-1, dims=(-2, -1))
    assert torch.equal(restored, raw)
    assert torch.equal(torch.flip(ccw, dims=(2,)), raw.transpose(2, 3))

    test = torch.arange(49, dtype=torch.float32).reshape(1, 1, 7, 7)
    point_cloud_range = [-7.0, -7.0, -1.0, 7.0, 7.0, 1.0]
    grid = lidar_to_ego_sampling_grid(
        7, 7, point_cloud_range, np.eye(3), np.zeros(3),
        test.device, test.dtype,
    )
    actual = F.grid_sample(test, grid, mode="bilinear", align_corners=False)
    assert torch.allclose(actual, test, atol=1e-5)
    print("[PASS] 非方形轴排列与零外参物理重栅格化", flush=True)


def write_visual(index: int, gt: np.ndarray, preds: dict[str, np.ndarray]) -> None:
    """绘制固定颜色的 GT 和三个候选预测，保留退化帧证据。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#20232e", "#f18f43", "#2cb4ad", "#e5ce5a"])
    images = [("GT", gt)] + [(name, preds[name]) for name in VARIANTS]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for axis, (name, img) in zip(axes, images):
        axis.imshow(img, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
        axis.set_title(name, fontsize=10)
        axis.axis("off")
    fig.suptitle(f"nuScenes Val index={index} | 独立帧对照（无历史）")
    fig.tight_layout()
    folder = OUTPUT / "images"
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"val_{index:04d}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    """运行不破坏历史产物的三分支小样本模型实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--indices", default=DEFAULT_INDICES)
    args = parser.parse_args()
    run_self_test()
    if args.self_test:
        return

    indices = [int(number.strip()) for number in args.indices.split(",")]
    if len(indices) != len(set(indices)) or not indices:
        raise ValueError("样本索引不能为空或重复")
    for path in (CONFIG, PKL, CKPT):
        if not path.is_file():
            raise FileNotFoundError(f"缺少现有文件：{path}")
    if not torch.cuda.is_available():
        raise RuntimeError("当前已使用的 BEVFormer CUDA 环境未启用")
    if (OUTPUT / "results.json").exists():
        raise FileExistsError("已有 Step37 结果，请先备份或指定新实验")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_bev_seg_one_sample import (
        segmentation_classes,
        to_numpy,
        unwrap_single_aug,
    )
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    importlib.import_module("projects.mmdet3d_plugin")
    cfg = Config.fromfile(str(CONFIG))
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.ann_file = str(PKL)
    dataset_cfg.test_mode = True
    for key in ("samples_per_gpu", "workers_per_gpu"):
        dataset_cfg.pop(key, None)
    dataset = build_dataset(dataset_cfg)
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise IndexError("固定样本索引超出官方 Val 范围")

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if type(model.pts_bbox_head).__name__ != "BEVFormerHead":
        raise RuntimeError("实际运行的不是 BEVFormerHead，停止方向解释")
    state = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    gc.collect()
    model = model.cuda().eval()
    cropper = model.pts_bbox_head.feat_cropper
    crop_forward = cropper.forward
    point_cloud_range = list(cfg.point_cloud_range)
    matrices = {name: np.zeros((4, 4), dtype=np.int64) for name in VARIANTS}
    records = []
    t_start = time.monotonic()

    try:
        for count, index in enumerate(indices, 1):
            info = dataset.data_infos[index]
            rotation = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
            translation = np.asarray(info["lidar2ego_translation"], dtype=float)
            item = dataset[index]
            image = unwrap_single_aug(item["img"])
            meta = unwrap_single_aug(item["img_metas"])
            gt = np.squeeze(
                to_numpy(unwrap_single_aug(item["semantic_indices"]))
            ).astype(np.uint8)
            if image.ndim == 4:
                image = image.unsqueeze(0)
            if image.ndim != 5 or tuple(image.shape[:2]) != (1, 6):
                raise ValueError(f"第 {index} 帧图像形状错误：{tuple(image.shape)}")
            image_gpu = image.cuda()
            preds = {}
            errors = {}

            for variant in VARIANTS:
                def mapped_crop(bev: torch.Tensor) -> torch.Tensor:
                    """按当前 variant 切换唯一的空间干预环节。"""
                    if variant == "original_ccw90":
                        return crop_forward(bev)
                    if variant == "transpose_candidate":
                        return crop_forward(torch.flip(bev, dims=(2,)))
                    # BEVFormerHead 的原始输入已为 raw_bev 的 CCW90。
                    raw_bev = torch.rot90(bev, k=-1, dims=(2, 3))
                    if raw_bev.shape[-2:] != (150, 150):
                        raise ValueError("物理标定分支只支持当前 150x150 模型")
                    grid = lidar_to_ego_sampling_grid(
                        raw_bev.shape[-2], raw_bev.shape[-1],
                        point_cloud_range, rotation, translation,
                        raw_bev.device, raw_bev.dtype,
                    )
                    ego_bev = F.grid_sample(
                        raw_bev, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=False,
                    )
                    return crop_forward(ego_bev)

                cropper.forward = mapped_crop
                with torch.inference_mode():
                    prediction_output = model(
                        return_loss=False,
                        rescale=True,
                        img=[image_gpu.clone()],
                        img_metas=[[copy.deepcopy(meta)]],
                    )
                pred, _ = segmentation_classes(prediction_output[0]["seg_preds"])
                preds[variant] = pred
                matrices[variant] += confusion_matrix(gt, pred)
                errors[variant] = metric(confusion_matrix(gt, pred))[
                    "foreground_miou"
                ]
                del prediction_output

            record = {
                "index": index,
                "sample_token": str(info["token"]),
                "scene_token": str(info["scene_token"]),
                "original_ccw90_miou": errors["original_ccw90"],
                "transpose_candidate_miou": errors["transpose_candidate"],
                "lidar_to_ego_grid_miou": errors["lidar_to_ego_grid"],
                "transpose_minus_original_pp": 100 * (
                    errors["transpose_candidate"] - errors["original_ccw90"]
                ),
                "physical_minus_original_pp": 100 * (
                    errors["lidar_to_ego_grid"] - errors["original_ccw90"]
                ),
            }
            records.append(record)
            if index in (0, 3000, 3715, 3718, 3719, 6018):
                write_visual(index, gt, preds)
            print(
                f"[{count}/{len(indices)}] {index}: "
                f"A={errors['original_ccw90']:.2%} "
                f"B={errors['transpose_candidate']:.2%} "
                f"C={errors['lidar_to_ego_grid']:.2%}",
                flush=True,
            )
            del image_gpu, preds
    finally:
        cropper.forward = crop_forward

    output = {
        "experiment": "Step37 targeted geometry / official Val sparse samples",
        "protocol": "same GT, checkpoint, frames; no temporal history; no retraining",
        "variant_definitions": {
            "original_ccw90": "actual local BEVFormerHead forward output",
            "transpose_candidate": "extra flip(dim=2) before feat_cropper",
            "lidar_to_ego_grid": (
                "undo head CCW90, then per-frame inverse LiDAR-to-Ego "
                "grid_sample raw BEV onto Ego coordinates before Slicer"
            ),
        },
        "important_limit": (
            "物理分支依据 Encoder ref_x/ref_y 的物理约定构造；只做推理端修正，"
            "不保证符合已有 checkpoint 训练时的空间排列。独立帧结果不可与"
            "Step34 连续时序 6019 帧 mIoU 直接比较。"
        ),
        "indices": indices,
        "elapsed_seconds": time.monotonic() - t_start,
        "results": {name: metric(matrix) for name, matrix in matrices.items()},
        "per_frame": records,
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (OUTPUT / "per_frame.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print("\n===== Step37 累计前景 mIoU（独立帧）=====", flush=True)
    for name in VARIANTS:
        print(f"{name}: {output['results'][name]['foreground_miou']:.4%}")
    print(f"报告：{OUTPUT / 'results.json'}")
    print(f"可视化：{OUTPUT / 'images'}")
    print("注意：物理坐标候选是诊断实验，不是已验证修复。", flush=True)


if __name__ == "__main__":
    main()

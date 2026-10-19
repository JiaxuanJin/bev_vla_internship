#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""以真实 Val 样本比较原始分割头与标定驱动的方向候选。

实验保持权重、输入、GT 和网络其他结构一致，仅在 BevFeatureSlicer 输入
处将原始转置后的 BEV 再沿行翻转，等价于对原始 BEV 做逆时针旋转 90 度。
这是候选消融而非预设修复，所得指标不替代完整 Val 结果。
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
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer-seg"
CONFIG = REPO / "projects/configs/bevformer/bevformer_small_seg.py"
PKL = (
    ROOT
    / "outputs/drivelm_check/root_cause/step21_full_val/"
    "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
)
CHECKPOINT = Path("/mnt/g/nus/bevformer_small_seg.pth")
OUTPUT = ROOT / "outputs/drivelm_check/root_cause/step33_geometry_ab"
DEFAULT_INDICES = "0,300,600,900,1200,1500,1800,2100,2400,3000,3600,4500,5300,6018"


def frame_confusion(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """以 GT 为行、预测为列统计四类混淆矩阵。"""
    if gt.shape != prediction.shape or gt.shape != (200, 400):
        raise ValueError(f"GT/预测尺寸错误：{gt.shape} / {prediction.shape}")
    gt_values = gt.ravel().astype(np.int64)
    pred_values = prediction.ravel().astype(np.int64)
    if not np.isin(gt_values, [0, 1, 2, 3]).all():
        raise ValueError("GT 类别必须属于 0..3")
    if not np.isin(pred_values, [0, 1, 2, 3]).all():
        raise ValueError("预测类别必须属于 0..3")
    return np.bincount(gt_values * 4 + pred_values, minlength=16).reshape(4, 4)


def summarize(matrix: np.ndarray) -> dict:
    """与作者实现一致：四类别累计 IoU，前景三类简单平均，缺席类记 0。"""
    true_positive = np.diag(matrix)
    union = matrix.sum(0) + matrix.sum(1) - true_positive
    values = np.divide(
        true_positive,
        union,
        out=np.zeros(4, dtype=float),
        where=union > 0,
    )
    return {
        "class_iou": {
            key: float(value)
            for key, value in zip(
                ("background", "divider", "ped_crossing", "boundary"),
                values,
            )
        },
        "foreground_miou": float(values[1:].mean()),
        "confusion_gt_rows_pred_cols": matrix.tolist(),
    }


def main() -> None:
    """运行原始方向与物理标定候选方向的同输入、无历史配对对照。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default=DEFAULT_INDICES)
    args = parser.parse_args()
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("indices 不能为空或重复")
    if not PKL.is_file() or not CHECKPOINT.is_file():
        raise FileNotFoundError("完整 Val PKL 或 checkpoint 不存在")
    if not torch.cuda.is_available():
        raise RuntimeError("需要已经验证可用的 CUDA 环境")
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

    # 相对 data/nuscenes 路径仅在第三方项目根目录下有效。
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    importlib.import_module("projects.mmdet3d_plugin")
    config = Config.fromfile(str(CONFIG))
    dataset_config = copy.deepcopy(config.data.test)
    dataset_config.ann_file = str(PKL)
    dataset_config.test_mode = True
    for name in ("samples_per_gpu", "workers_per_gpu"):
        dataset_config.pop(name, None)
    dataset = build_dataset(dataset_config)
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise IndexError(f"样本序号不在 0..{len(dataset)-1} 内")

    config.model.pretrained = None
    config.model.train_cfg = None
    config.model.video_test_mode = False
    model = build_model(config.model, test_cfg=config.get("test_cfg"))
    checkpoint = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    loaded = model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    del checkpoint
    gc.collect()
    model = model.to("cuda:0").eval()
    cropper = model.pts_bbox_head.feat_cropper
    original_crop = cropper.forward
    matrices = {
        "original": np.zeros((4, 4), dtype=np.int64),
        "physical_ccw_candidate": np.zeros((4, 4), dtype=np.int64),
    }
    scene_ids: set[str] = set()
    records: list[dict] = []

    def physical_ccw_crop(bev: torch.Tensor) -> torch.Tensor:
        """原始分割头的 transpose 后沿 H 轴翻转，对应源 BEV 的 CCW90。"""
        return original_crop(torch.flip(bev, dims=[2]))

    try:
        for count, index in enumerate(indices, start=1):
            item = dataset[index]
            image = unwrap_single_aug(item["img"])
            meta = unwrap_single_aug(item["img_metas"])
            gt = np.squeeze(
                to_numpy(unwrap_single_aug(item["semantic_indices"]))
            ).astype(np.uint8)
            if image.ndim == 4:
                image = image.unsqueeze(0)
            if image.ndim != 5 or image.shape[0] != 1 or image.shape[1] != 6:
                raise ValueError(f"样本 {index} 的相机维度异常：{tuple(image.shape)}")
            image_gpu = image.to("cuda:0")
            scene_id = str(dataset.data_infos[index]["scene_token"])
            scene_ids.add(scene_id)
            preds = {}
            for variant in ("original", "physical_ccw_candidate"):
                cropper.forward = (
                    original_crop if variant == "original" else physical_ccw_crop
                )
                with torch.inference_mode():
                    output = model(
                        return_loss=False,
                        rescale=True,
                        img=[image_gpu.clone()],
                        img_metas=[[copy.deepcopy(meta)]],
                    )
                prediction, _ = segmentation_classes(output[0]["seg_preds"])
                matrices[variant] += frame_confusion(gt, prediction)
                preds[variant] = prediction
                del output
            changed = int(np.count_nonzero(preds["original"] != preds["physical_ccw_candidate"]))
            entry = {
                "index": index,
                "sample_token": str(dataset.data_infos[index]["token"]),
                "scene_token": scene_id,
                "raw_fg_miou": summarize(frame_confusion(gt, preds["original"]))["foreground_miou"],
                "candidate_fg_miou": summarize(frame_confusion(gt, preds["physical_ccw_candidate"]))["foreground_miou"],
                "changed_pixels": changed,
            }
            entry["delta_fg_miou"] = entry["candidate_fg_miou"] - entry["raw_fg_miou"]
            records.append(entry)
            print(
                f"[{count}/{len(indices)}] frame={index} "
                f"raw={entry['raw_fg_miou']:.4%} "
                f"ccw={entry['candidate_fg_miou']:.4%} "
                f"pixels_changed={changed}",
                flush=True,
            )
            del image_gpu, preds
    finally:
        cropper.forward = original_crop

    base = summarize(matrices["original"])
    trial = summarize(matrices["physical_ccw_candidate"])
    result = {
        "experiment": "Step33 physically motivated BEV orientation counterfactual",
        "protocol": "full Val sparse independent frames; identical data/GT/weights; no temporal history",
        "calibration_hypothesis": "LiDAR->Ego yaw close to -90 deg; compare source transpose vs source BEV CCW90",
        "note": (
            "诊断性因果消融；模型权重在原始方向下训练，推理阶段单独改动不能"
            "视为正式修复。不得与完整 6019 帧时序 mIoU 直接作同协议比较。"
        ),
        "indices": indices,
        "scenes_represented": len(scene_ids),
        "original": base,
        "physical_ccw_candidate": trial,
        "aggregate_delta_percentage_points": (
            trial["foreground_miou"] - base["foreground_miou"]
        ) * 100,
        "frames_improved": sum(x["delta_fg_miou"] > 0 for x in records),
        "frames_worsened": sum(x["delta_fg_miou"] < 0 for x in records),
        "frames_equal": sum(x["delta_fg_miou"] == 0 for x in records),
        "per_frame": records,
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print("\n===== 固定验证集累计结果 =====")
    print(f"覆盖场景数：{len(scene_ids)}")
    print(f"原始前景 mIoU：{base['foreground_miou']:.4%}")
    print(f"物理方向候选前景 mIoU：{trial['foreground_miou']:.4%}")
    print(f"变化：{result['aggregate_delta_percentage_points']:+.3f} 个百分点")
    print(f"改善/下降/不变：{result['frames_improved']}/{result['frames_worsened']}/{result['frames_equal']}")
    print(f"结果：{OUTPUT / 'results.json'}")
    print("注意：只检验推理阶段方向候选的因果影响，不是训练后修复证明。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 39：BEVFormer-Seg 空间映射修复候选筛选。

用同一帧的 BEV encoder 输出，逐一送入八种离散坐标排列及原始分割 decoder。
验证目标：候选是否在既定诊断帧、预留验证帧及已知退化帧上同时稳定。

注意：此步骤不改动原始源码或 checkpoint；无时序的抽样结果不是正式基线。
命令：python -u scripts/step39_orientation_repair_screen.py
"""

from __future__ import annotations

import copy
import csv
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer-seg"
CONFIG = REPO / "projects/configs/bevformer/bevformer_small_seg.py"
CHECKPOINT = Path("/mnt/g/nus/bevformer_small_seg.pth")
VAL_PKL = (
    ROOT / "outputs/drivelm_check/root_cause/step21_full_val/"
    "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
)
OUTPUT = ROOT / "outputs/drivelm_check/root_cause/step39_orientation_screen"
CLASS_NAMES = ("background", "divider", "ped_crossing", "boundary")

# 用固定索引避免人为挑选最好看的帧；各组相互不重叠。
SCREEN = (
    0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000,
    3300, 3600, 3900, 4200, 4500, 4800, 5100, 5400, 5700, 6018,
)
VERIFY = (150, 750, 1350, 1950, 2550, 3150, 4350, 4950, 5550)
REGRESSION = (3715, 3718, 3719)


def get_orientations(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    """对原始 Encoder 排列 [B,C,Y,X] 应用八种 D4 方格方向。"""
    transpose = raw.transpose(-2, -1)
    return {
        "identity_raw_xy": raw,
        "original_ccw90": torch.rot90(raw, 1, (-2, -1)),
        "rot90_cw": torch.rot90(raw, -1, (-2, -1)),
        "rot180": torch.rot90(raw, 2, (-2, -1)),
        "transpose_candidate": transpose,
        "anti_transpose": torch.flip(transpose, (-2, -1)),
        "flip_rows": torch.flip(raw, (-2,)),
        "flip_columns": torch.flip(raw, (-1,)),
    }


def confusion(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """GT 为行、Prediction 为列，生成四类混淆矩阵。"""
    if gt.shape != (200, 400) or pred.shape != gt.shape:
        raise ValueError(f"GT 与预测尺寸有误：{gt.shape} / {pred.shape}")
    if not np.isin(gt, [0, 1, 2, 3]).all():
        raise ValueError("GT 不是 0~3 的四分类掩码")
    if not np.isin(pred, [0, 1, 2, 3]).all():
        raise ValueError("Prediction 不是 0~3 的四分类掩码")
    pairs = gt.astype(np.int64).ravel() * 4 + pred.astype(np.int64).ravel()
    return np.bincount(pairs, minlength=16).reshape(4, 4)


def summarize(cm: np.ndarray) -> dict:
    """和作者评估器一致：全局每类 IoU，均值仅含三类前景。"""
    tp = np.diag(cm)
    unions = cm.sum(axis=0) + cm.sum(axis=1) - tp
    iou = np.divide(
        tp, unions, out=np.zeros(4, dtype=np.float64), where=unions != 0
    )
    return {
        "foreground_miou": float(iou[1:].mean()),
        "class_iou": {name: float(value) for name, value in zip(CLASS_NAMES, iou)},
        "confusion_gt_rows_pred_cols": cm.tolist(),
    }


def get_masks(logits: torch.Tensor) -> np.ndarray:
    """将 4 类 logits 转为 200×400 的类别掩码。"""
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 4:
        raise RuntimeError(f"分割头输出形状异常：{tuple(logits.shape)}")
    return logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)


def init_runtime():
    """沿用 Step25 已证实的作者数据管线，严格加载模型权重。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，Step39 需要在现有 BEVFormer GPU 环境执行")
    for file in (CONFIG, CHECKPOINT, VAL_PKL):
        if not file.is_file():
            raise FileNotFoundError(f"缺少文件：{file}")

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_bev_seg_one_sample import (  # pylint: disable=import-outside-toplevel
        to_numpy,
        unwrap_single_aug,
    )

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from mmcv import Config  # pylint: disable=import-outside-toplevel
    from mmdet3d.datasets import build_dataset  # pylint: disable=import-outside-toplevel
    from mmdet3d.models import build_model  # pylint: disable=import-outside-toplevel

    importlib.import_module("projects.mmdet3d_plugin")
    cfg = Config.fromfile(str(CONFIG))
    data_cfg = copy.deepcopy(cfg.data.test)
    data_cfg.ann_file = str(VAL_PKL)
    data_cfg.test_mode = True
    data_cfg.pop("samples_per_gpu", None)
    data_cfg.pop("workers_per_gpu", None)
    dataset = build_dataset(data_cfg)
    if len(dataset) != 6019:
        raise RuntimeError(f"Val 帧数不是 6019，而是 {len(dataset)}")

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False  # 固定帧评估，避免时序历史干扰。
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = torch.load(
        str(CHECKPOINT), map_location="cpu", weights_only=True
    )
    report = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError("Checkpoint 与实际模型结构不一致")
    del checkpoint
    gc.collect()
    model = model.to("cuda:0").eval()
    return dataset, model, to_numpy, unwrap_single_aug


def main() -> None:
    """一次 Encoder 推理输出八种修复候选，分屏选与留出集。"""
    indices = [*SCREEN, *VERIFY, *REGRESSION]
    if len(indices) != len(set(indices)):
        raise RuntimeError("预定义索引存在重复")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset, model, to_numpy, unwrap = init_runtime()
    cropper = model.pts_bbox_head.feat_cropper
    decoder = model.pts_bbox_head.seg_decoder
    original_forward: Callable = cropper.forward

    names = tuple(get_orientations(torch.zeros(1, 1, 2, 2)))
    totals = {
        split: {name: np.zeros((4, 4), dtype=np.int64) for name in names}
        for split in ("screen", "verify", "regression")
    }
    rows: list[dict] = []
    beginning = time.monotonic()
    group_of = {i: "screen" for i in SCREEN}
    group_of.update({i: "verify" for i in VERIFY})
    group_of.update({i: "regression" for i in REGRESSION})

    try:
        for count, index in enumerate(indices, 1):
            item = dataset[index]
            img = unwrap(item["img"])
            meta = unwrap(item["img_metas"])
            gt = np.squeeze(to_numpy(unwrap(item["semantic_indices"]))).astype(
                np.uint8
            )
            if img.ndim == 4:
                img = img.unsqueeze(0)
            if img.ndim != 5 or tuple(img.shape[:2]) != (1, 6):
                raise RuntimeError(f"帧 {index} 相机张量维度异常：{img.shape}")
            if not isinstance(meta, dict):
                raise TypeError(f"帧 {index} 图像元数据不是字典")

            capture: list[torch.Tensor] = []

            def record_pre_crop(bev: torch.Tensor) -> torch.Tensor:
                """只拦截实际使用的 BEVFormerHead Slicer 输入，不修改其算法。"""
                capture.append(bev.detach())
                return original_forward(bev)

            cropper.forward = record_pre_crop
            try:
                with torch.inference_mode():
                    output = model(
                        return_loss=False,
                        rescale=True,
                        img=[img.to("cuda:0").clone()],
                        img_metas=[[copy.deepcopy(meta)]],
                    )
            finally:
                cropper.forward = original_forward

            if len(capture) != 1:
                raise RuntimeError(f"帧 {index}: Slicer 被调用 {len(capture)} 次")
            original_logits = output[0].get("seg_preds")
            if original_logits is None:
                raise RuntimeError(f"帧 {index} 没有 seg_preds")

            # 本地实际 BEVFormerHead 的输入 A 等价于 CCW90(raw)。
            # 将其反旋转，得到与 Encoder 网格行列相同的 raw 排列。
            a_input = capture[0]
            if a_input.shape[-2:] != (150, 150):
                raise RuntimeError(f"BEV 特征尺寸异常：{a_input.shape}")
            raw_bev = torch.rot90(a_input, -1, (-2, -1))
            variants = get_orientations(raw_bev)
            if not torch.equal(variants["original_ccw90"], a_input):
                raise RuntimeError("原始输入重建失败，立即停止")

            group = group_of[index]
            record = {
                "index": index,
                "group": group,
                "sample_token": str(dataset.data_infos[index]["token"]),
                "scene_token": str(dataset.data_infos[index]["scene_token"]),
            }
            reference_pred = get_masks(original_logits)
            with torch.inference_mode():
                for name, variant in variants.items():
                    reconstructed = decoder(original_forward(variant))
                    prediction = get_masks(reconstructed)
                    if name == "original_ccw90" and not np.array_equal(
                        reference_pred, prediction
                    ):
                        raise RuntimeError(
                            f"帧 {index}: 原始预测无法由采集的 BEV 重现"
                        )
                    cm = confusion(gt, prediction)
                    totals[group][name] += cm
                    record[f"{name}_miou"] = summarize(cm)["foreground_miou"]

            rows.append(record)
            print(
                f"[{count}/{len(indices)}] index={index} {group} "
                f"原始={record['original_ccw90_miou']:.2%} "
                f"Transpose={record['transpose_candidate_miou']:.2%} "
                f"Raw={record['identity_raw_xy_miou']:.2%}",
                flush=True,
            )
            del output, img, variants, raw_bev, capture

    finally:
        cropper.forward = original_forward

    grouped = {
        group: {name: summarize(cm) for name, cm in variants.items()}
        for group, variants in totals.items()
    }
    selected = max(
        names,
        key=lambda name: grouped["screen"][name]["foreground_miou"],
    )
    ranking = sorted(
        names,
        key=lambda name: grouped["screen"][name]["foreground_miou"],
        reverse=True,
    )
    result = {
        "experiment": "Step39 D4 feature orientation, same Encoder per frame",
        "protocol": (
            "固定索引，无时序；同一 Encoder 输出、权重、GT、Slicer、decoder；"
            "变化项仅是传入 Slicer 前的空间索引排列"
        ),
        "limitation": (
            "此为诊断和候选筛选；并非真实 LiDAR->Ego 物理刚体校准，"
            "也不是独立官方测试集评价。不得将最大 IoU 直接视为已修复训练坐标协议。"
        ),
        "model_forward_reproduction_passed": True,
        "screen_indices": list(SCREEN),
        "verify_indices": list(VERIFY),
        "known_regression_indices": list(REGRESSION),
        "screen_ranking": ranking,
        "screen_selected": selected,
        "metrics_by_group": grouped,
        "elapsed_seconds": round(time.monotonic() - beginning, 2),
    }
    with (OUTPUT / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    with (OUTPUT / "per_frame.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== Step39 汇总（前景 mIoU，按全局混淆矩阵）=====", flush=True)
    for group in ("screen", "verify", "regression"):
        print(f"\n[{group}]", flush=True)
        for name in ranking:
            print(f"  {name:<23} {grouped[group][name]['foreground_miou']:.3%}", flush=True)
    print("\n仅按 screen 选出的候选:", selected, flush=True)
    print("结果:", OUTPUT / "results.json", flush=True)
    print("提醒：本次只是空间变换筛选，不是训练/推理坐标修复完成。", flush=True)


if __name__ == "__main__":
    main()

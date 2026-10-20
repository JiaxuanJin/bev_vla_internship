#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step44：Mini Val 预测与地图 GT 的空间、类别一致性定点诊断。

运行：python -u scripts/step44_minival_gt_alignment_diagnostic.py
仅通过前景累计 IoU 筛选诊断假设，不修改模型、GT 文件或权重。
任何从验证集搜索出的转换，均不得作为正式性能或训练目标。
"""

from __future__ import annotations

import copy
import gc
import importlib
import json
import os
import sys
import time
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer-seg"
CONFIG = REPO / "projects/configs/bevformer/bevformer_small_seg.py"
CKPT = Path("/mnt/g/nus/bevformer_small_seg.pth")
OUT = ROOT / "outputs/drivelm_check/root_cause/step44_gt_alignment"
N_CLASSES = 4
CLASS_NAMES = ("background", "divider", "ped_crossing", "boundary")


def confusion(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """按 GT 行/预测列统计混淆矩阵。"""
    packed = gt.astype(np.int64).ravel() * N_CLASSES + pred.ravel()
    return np.bincount(packed, minlength=N_CLASSES**2).reshape(4, 4)


def miou(matrix: np.ndarray) -> float:
    """与作者前景 mIoU 定义一致，三类始终计入平均。"""
    tp = matrix.diagonal()
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - tp
    scores = np.divide(tp, union, out=np.zeros(4), where=union != 0)
    return float(scores[1:].mean())


def transform_mask(gt: np.ndarray, operation: str) -> np.ndarray:
    """不改变 H×W 的三个镜像候选（仅用于异常原因诊断）。"""
    if operation == "none":
        return gt
    if operation == "flip_rows":
        return gt[::-1, :]
    if operation == "flip_columns":
        return gt[:, ::-1]
    if operation == "rotate_180":
        return gt[::-1, ::-1]
    raise ValueError(f"未知空间候选：{operation}")


def shift_mask(gt: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """将 GT 向右 dx、向下 dy 平移；越界用背景 0 填充。"""
    h, w = gt.shape
    shifted = np.zeros_like(gt)
    x_dst_start, x_dst_end = max(dx, 0), min(w + dx, w)
    y_dst_start, y_dst_end = max(dy, 0), min(h + dy, h)
    if x_dst_start < x_dst_end and y_dst_start < y_dst_end:
        shifted[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = gt[
            y_dst_start - dy:y_dst_end - dy,
            x_dst_start - dx:x_dst_end - dx,
        ]
    return shifted


def evaluate(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    operation: str = "none",
    dx: int = 0,
    dy: int = 0,
    labels: tuple[int, int, int] = (1, 2, 3),
) -> tuple[float, np.ndarray]:
    """固定变换下计算所有帧的全局 IoU；不平均逐帧 mIoU。"""
    total = np.zeros((4, 4), dtype=np.int64)
    remap = np.array((0, *labels), dtype=np.uint8)
    for gt, pred in pairs:
        mapped = transform_mask(gt, operation)
        if dx or dy:
            mapped = shift_mask(mapped, dx, dy)
        if labels != (1, 2, 3):
            mapped = remap[mapped]
        total += confusion(mapped, pred)
    return miou(total), total


def save_report(obj: dict[str, Any]) -> None:
    """保存可复核 JSON。"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def self_test() -> None:
    """验证平移方向、类别转换以及混淆矩阵定义。"""
    gt = np.array([[0, 1, 2], [0, 3, 0]], dtype=np.uint8)
    assert np.array_equal(shift_mask(gt, 1, 0),
                          np.array([[0, 0, 1], [0, 0, 3]], dtype=np.uint8))
    assert np.array_equal(shift_mask(gt, 0, -1),
                          np.array([[0, 3, 0], [0, 0, 0]], dtype=np.uint8))
    assert confusion(gt, gt).trace() == gt.size
    assert abs(miou(confusion(gt, gt)) - 1.0) < 1e-12
    print("[self-test] PASS", flush=True)


def main() -> None:
    """同一次 Mini 推理收集预测，在两场景上分别筛选/验证假设。"""
    self_test()
    if not CKPT.is_file():
        raise FileNotFoundError(f"缺少权重：{CKPT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_bev_seg_one_sample import (  # pylint: disable=import-outside-toplevel
        segmentation_classes,
        to_numpy,
        unwrap_single_aug,
    )

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    importlib.import_module("projects.mmdet3d_plugin")
    cfg = Config.fromfile(str(CONFIG))
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.test_mode = True
    dataset_cfg.pop("samples_per_gpu", None)
    dataset_cfg.pop("workers_per_gpu", None)
    dataset = build_dataset(dataset_cfg)
    if len(dataset) != 81 or dataset.version != "v1.0-mini":
        raise RuntimeError(
            f"需要严格的 Mini Val 81 帧；当前={len(dataset)}, version={dataset.version}"
        )

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    state = torch.load(CKPT, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()
    model = model.cuda().eval()

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    scene_tokens: list[str] = []
    start = time.monotonic()
    for i in range(len(dataset)):
        item = dataset[i]
        img = unwrap_single_aug(item["img"])
        meta = unwrap_single_aug(item["img_metas"])
        gt = np.squeeze(to_numpy(unwrap_single_aug(item["semantic_indices"])))
        gt = gt.astype(np.uint8)
        if img.ndim == 4:
            img = img.unsqueeze(0)
        with torch.no_grad():
            output = model(
                return_loss=False,
                rescale=True,
                img=[img.to("cuda")],
                img_metas=[[copy.deepcopy(meta)]],
            )
        pred, _ = segmentation_classes(output[0]["seg_preds"])
        if gt.shape != (200, 400) or pred.shape != gt.shape:
            raise RuntimeError(f"样本{i}形状不符：{gt.shape}, {pred.shape}")
        pairs.append((gt.copy(), pred.copy()))
        scene_tokens.append(str(meta["scene_token"]))
        if (i + 1) % 20 == 0 or i + 1 == len(dataset):
            print(f"[inference] {i+1}/{len(dataset)}", flush=True)
        del output, img

    # 同批数据集两场景中的第一场景用于筛选；其余场景只用于验证。
    first_scene = scene_tokens[0]
    fit = [pair for pair, scene in zip(pairs, scene_tokens) if scene == first_scene]
    holdout = [pair for pair, scene in zip(pairs, scene_tokens) if scene != first_scene]
    if not holdout:
        raise RuntimeError("Mini Val 没有独立场景，拒绝用同一场景筛选和验证")

    baseline_all, baseline_cm = evaluate(pairs)
    baseline_fit, _ = evaluate(fit)
    baseline_holdout, _ = evaluate(holdout)
    print(
        f"[baseline] all={baseline_all:.4%} fit={baseline_fit:.4%} "
        f"holdout={baseline_holdout:.4%}",
        flush=True,
    )

    operations = ("none", "flip_rows", "flip_columns", "rotate_180")
    # 只针对有限、可解释的全局轴反转和平移假设进行一次诊断。
    offsets = (-40, -20, 0, 20, 40)
    candidates = []
    for op in operations:
        for dy in offsets:
            for dx in offsets:
                score, _ = evaluate(fit, op, dx, dy)
                candidates.append({"operation": op, "dx": dx, "dy": dy,
                                   "fit_miou": score})
    candidates.sort(key=lambda d: d["fit_miou"], reverse=True)
    best = candidates[0]
    best_holdout, best_cm = evaluate(
        holdout, best["operation"], best["dx"], best["dy"]
    )

    perm_scores = []
    for p in permutations((1, 2, 3)):
        score, _ = evaluate(fit, labels=p)
        perm_scores.append({"labels": list(p), "fit_miou": score})
    perm_scores.sort(key=lambda d: d["fit_miou"], reverse=True)
    best_perm = perm_scores[0]
    perm_holdout, _ = evaluate(holdout, labels=tuple(best_perm["labels"]))

    payload = {
        "experiment": "Step44 Mini Val GT spatial and class-label diagnostic",
        "scope": "81-frame official mini_val, frozen author checkpoint, temporal ON",
        "caution": "GT 变换由一个验证场景选出，另一个场景交叉验证；仅用于定位标签约定，不能作为正式修复、指标或训练结果",
        "scene_ids": list(dict.fromkeys(scene_tokens)),
        "fit_frames": len(fit),
        "holdout_frames": len(holdout),
        "baseline_all_miou": baseline_all,
        "baseline_fit_miou": baseline_fit,
        "baseline_holdout_miou": baseline_holdout,
        "baseline_confusion_gt_rows_pred_cols": baseline_cm.tolist(),
        "gt_class_pixels": baseline_cm.sum(axis=1).tolist(),
        "pred_class_pixels": baseline_cm.sum(axis=0).tolist(),
        "best_gt_spatial_candidate": {
            **best,
            "holdout_miou": best_holdout,
            "holdout_delta_pp": 100 * (best_holdout - baseline_holdout),
            "holdout_confusion": best_cm.tolist(),
        },
        "top_10_spatial_fit": candidates[:10],
        "best_gt_class_permutation": {
            **best_perm,
            "holdout_miou": perm_holdout,
            "holdout_delta_pp": 100 * (perm_holdout - baseline_holdout),
        },
        "class_permutations_fit": perm_scores,
        "elapsed_seconds": round(time.monotonic() - start, 1),
    }
    save_report(payload)
    print("\n====== Step44 ======", flush=True)
    print(f"Baseline mini_val: {baseline_all*100:.3f}%", flush=True)
    print(
        f"场景 A({len(fit)}帧) {baseline_fit*100:.3f}% | "
        f"场景 B({len(holdout)}帧) {baseline_holdout*100:.3f}%",
        flush=True,
    )
    print(
        "GT空间变换最佳候选:", best["operation"],
        f"dx={best['dx']}, dy={best['dy']}; "
        f"筛选场景={best['fit_miou']*100:.3f}% "
        f"独立场景={best_holdout*100:.3f}%",
        flush=True,
    )
    print(
        "类别索引排列最佳候选:", best_perm["labels"],
        f"筛选场景={best_perm['fit_miou']*100:.3f}% "
        f"独立场景={perm_holdout*100:.3f}%",
        flush=True,
    )
    print(f"报告：{OUT / 'results.json'}", flush=True)
    print("注意：本次从 Val 选取的候选不能用于正式成绩。", flush=True)


if __name__ == "__main__":
    main()

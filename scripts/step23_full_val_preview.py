#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整 nuScenes Val 的连续帧 GPU 冒烟评估（非完整榜单成绩）。

保持 checkpoint、模型配置和官方地图 GT 生成管线不变，在同一场景选取
连续若干帧；使用上游 IntersectionOverUnion 与 NumPy 混淆矩阵分别计算指标。

用法（在项目根目录执行）：
    python -u scripts/step23_full_val_preview.py --limit 30
"""

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer-seg"
SCRIPTS = ROOT / "scripts"
CONFIG = REPO / "projects/configs/bevformer/bevformer_small_seg.py"
CHECKPOINT = Path("/mnt/g/nus/bevformer_small_seg.pth")
FULL_VAL_PKL = (
    ROOT
    / "outputs/drivelm_check/root_cause/step21_full_val"
    / "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
)
OUTPUT_DIR = ROOT / "outputs/drivelm_check/root_cause/step23_full_val_preview"
CLASS_NAMES = ("background", "divider", "ped_crossing", "boundary")
NUM_CLASSES = len(CLASS_NAMES)


def file_sha256(path: Path) -> str:
    """计算文件 SHA256，记录实验所用 checkpoint 身份。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def choose_contiguous_indices(infos: list[dict], count: int) -> list[int]:
    """找出同一 scene 中严格以前一 token 衔接的 count 个样本。"""
    if count < 2:
        raise ValueError("--limit 至少为 2，才能验证时序连续性")

    for start in range(len(infos)):
        selected = [start]
        for index in range(start + 1, len(infos)):
            previous = infos[index - 1]
            current = infos[index]
            if current["scene_token"] != previous["scene_token"]:
                break
            if current.get("prev") != previous.get("token"):
                break
            selected.append(index)
            if len(selected) == count:
                return selected
    raise RuntimeError(f"完整 Val 中没有找到连续 {count} 帧的片段")


def confusion_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """累计四类像素混淆矩阵，行 GT、列 Prediction。"""
    if gt.shape != pred.shape or gt.shape != (200, 400):
        raise ValueError(f"预测和标签尺寸不正确：gt={gt.shape}, pred={pred.shape}")
    if not np.isin(gt, np.arange(NUM_CLASSES)).all():
        raise ValueError("GT 存在不支持的类别")
    if not np.isin(pred, np.arange(NUM_CLASSES)).all():
        raise ValueError("Prediction 存在不支持的类别")
    values = NUM_CLASSES * gt.astype(np.int64).ravel() + pred.astype(np.int64).ravel()
    return np.bincount(values, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def iou_from_confusion(matrix: np.ndarray) -> list[float]:
    """按上游 absent_score=0 规则统计 IoU，包括完全缺席类别。"""
    tp = np.diag(matrix)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - tp
    return [float(tp[i] / union[i]) if union[i] else 0.0 for i in range(NUM_CLASSES)]


def main() -> None:
    """执行一次严格连续帧、双评价器的 Val GPU 预检查。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30, help="同场景连续样本数")
    args = parser.parse_args()

    if not FULL_VAL_PKL.is_file() or not CHECKPOINT.is_file():
        raise FileNotFoundError("未找到修复后的完整 Val PKL 或 checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请激活 .venv-bevformer")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(SCRIPTS))
    from run_bev_seg_one_sample import (  # pylint: disable=import-outside-toplevel
        segmentation_classes,
        to_numpy,
        unwrap_single_aug,
    )

    # 上游相对路径 data/nuscenes 必须从 BEVFormer-seg 工作目录解析。
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from mmcv import Config  # pylint: disable=import-outside-toplevel
    from mmdet3d.datasets import build_dataset  # pylint: disable=import-outside-toplevel
    from mmdet3d.models import build_model  # pylint: disable=import-outside-toplevel

    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.metrics import (  # pylint: disable=import-outside-toplevel
        IntersectionOverUnion,
    )

    cfg = Config.fromfile(str(CONFIG))
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.ann_file = str(FULL_VAL_PKL)
    dataset_cfg.test_mode = True
    dataset_cfg.pop("samples_per_gpu", None)
    dataset_cfg.pop("workers_per_gpu", None)
    dataset = build_dataset(dataset_cfg)
    if len(dataset) != 6019:
        raise ValueError(f"预期6019帧，实际{len(dataset)}帧")

    selected = choose_contiguous_indices(dataset.data_infos, args.limit)
    first_info = dataset.data_infos[selected[0]]
    print("===== Step 23：完整 Val 连续帧小规模 GPU 核查 =====", flush=True)
    print(f"全量 Val: {len(dataset)} 帧", flush=True)
    print(f"本次索引: {selected[0]}~{selected[-1]}", flush=True)
    print(f"场景 Token: {first_info['scene_token']}", flush=True)
    print("模型时序模式:", cfg.model.video_test_mode, flush=True)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    payload = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    report = model.load_state_dict(payload["state_dict"], strict=True)
    print(
        f"checkpoint 严格加载: missing={len(report.missing_keys)}, "
        f"unexpected={len(report.unexpected_keys)}",
        flush=True,
    )
    del payload
    gc.collect()

    device = torch.device("cuda:0")
    model.to(device).eval()
    upstream_metric = IntersectionOverUnion(NUM_CLASSES).to(device)
    cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_frame = []
    start_time = time.monotonic()

    for position, index in enumerate(selected, start=1):
        item = dataset[index]
        img = unwrap_single_aug(item["img"])
        meta = unwrap_single_aug(item["img_metas"])
        gt = np.squeeze(to_numpy(unwrap_single_aug(item["semantic_indices"]))).astype(
            np.uint8
        )
        if not isinstance(meta, dict) or not torch.is_tensor(img):
            raise TypeError(f"第 {index} 帧输入格式异常")
        if img.ndim == 4:
            img = img.unsqueeze(0)
        if img.ndim != 5 or tuple(img.shape[:2]) != (1, 6):
            raise ValueError(f"第 {index} 帧六路相机输入维度异常：{tuple(img.shape)}")
        with torch.inference_mode():
            output = model(
                return_loss=False,
                rescale=True,
                img=[img.to(device).clone()],
                img_metas=[[copy.deepcopy(meta)]],
            )
        logits = output[0]["seg_preds"]
        pred, _ = segmentation_classes(logits)
        frame_cm = confusion_matrix(gt, pred)
        cm_total += frame_cm

        # 同一预测与 GT，分别由作者原始 IoU 类和混淆矩阵进行计算。
        upstream_metric.update(
            torch.as_tensor(pred[None], device=device, dtype=torch.long),
            torch.as_tensor(gt[None], device=device, dtype=torch.long),
        )
        scores = iou_from_confusion(frame_cm)
        per_frame.append(
            {
                "index": index,
                "sample_token": dataset.data_infos[index]["token"],
                "scene_token": dataset.data_infos[index]["scene_token"],
                "foreground_miou_single_frame": float(np.mean(scores[1:])),
            }
        )
        if position % 5 == 0 or position == len(selected):
            running_scores = iou_from_confusion(cm_total)
            print(
                f"[{position}/{len(selected)}] 累计前景 mIoU "
                f"{np.mean(running_scores[1:]):.4%}",
                flush=True,
            )

    local_scores = np.array(iou_from_confusion(cm_total))
    official_scores = upstream_metric.compute().detach().cpu().numpy()
    diff = float(np.max(np.abs(local_scores - official_scores)))
    result = {
        "experiment": "full nuScenes Val contiguous subset - smoke test only",
        "important_note": "仅30帧预检查，不是6019帧完整Val成绩或DriveLM799正式成绩",
        "full_val_samples": len(dataset),
        "tested_samples": len(selected),
        "scene_token": first_info["scene_token"],
        "selected_indices": selected,
        "video_test_mode": bool(cfg.model.video_test_mode),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "pkl": str(FULL_VAL_PKL),
        "class_names": list(CLASS_NAMES),
        "per_class_iou_numpy": local_scores.tolist(),
        "per_class_iou_upstream": official_scores.tolist(),
        "numpy_foreground_miou": float(np.mean(local_scores[1:])),
        "upstream_foreground_miou": float(np.mean(official_scores[1:])),
        "max_iou_difference": diff,
        "metric_match": bool(diff < 1e-5),
        "confusion_gt_rows_pred_cols": cm_total.tolist(),
        "per_frame": per_frame,
        "elapsed_seconds": round(time.monotonic() - start_time, 1),
    }
    path = OUTPUT_DIR / "results.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("===== 本次诊断结果 =====", flush=True)
    for name, a, b in zip(CLASS_NAMES, local_scores, official_scores):
        print(f"{name}: local={a:.4%}, upstream={b:.4%}", flush=True)
    print(f"前景 mIoU: {result['upstream_foreground_miou']:.4%}", flush=True)
    print(f"作者指标与混淆矩阵一致: {result['metric_match']}", flush=True)
    print(f"结果文件: {path}", flush=True)
    if not result["metric_match"]:
        raise RuntimeError("两套 IoU 指标不同，请检查结果 JSON 后继续")
    print("Step 23: PASS", flush=True)


if __name__ == "__main__":
    main()

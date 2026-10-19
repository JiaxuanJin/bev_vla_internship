#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BEVFormer-Seg: 完整 nuScenes Val 6019 帧前景分割性能复现。

使用上游数据管线 / forward_test / IntersectionOverUnion 语义，按真实 Val
数据顺序运行，保留各场景的历史 BEV，并与独立 NumPy 混淆矩阵指标相互检查。
输出目录独立于 Mini 和 DriveLM799，周期性保存推理时序状态以供续跑。

项目根目录执行：
    python -u scripts/step25_full_val_reproduce.py
断点续跑：
    python -u scripts/step25_full_val_reproduce.py --resume

注意：不等同于验证 checkpoint 与上游发布权重具有相同 SHA256。
"""

import argparse
import copy
import csv
import gc
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer-seg"
CONFIG = REPO / "projects/configs/bevformer/bevformer_small_seg.py"
CHECKPOINT = Path("/mnt/g/nus/bevformer_small_seg.pth")
FULL_VAL_PKL = (
    ROOT
    / "outputs/drivelm_check/root_cause/step21_full_val"
    / "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
)
OUTPUT_DIR = ROOT / "outputs/drivelm_check/root_cause/step25_full_val_6019"
CLASS_NAMES = ("background", "divider", "ped_crossing", "boundary")
NUM_CLASSES = len(CLASS_NAMES)
ALL_SAMPLES = 6019
PROGRESS_PATH = OUTPUT_DIR / "progress.pt"
RESULT_PATH = OUTPUT_DIR / "results.json"
CSV_PATH = OUTPUT_DIR / "per_frame.csv"


def sha256_file(path: Path) -> str:
    """计算文件的 SHA256，用于检验断点恢复时实验身份一致性。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    """先写临时文件后替换，降低中断导致 JSON 损坏的风险。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def confusion_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """返回 4x4 混淆矩阵；行为 GT，列为预测。"""
    if gt.shape != (200, 400) or pred.shape != gt.shape:
        raise ValueError(f"预测与 GT 尺寸不一致：GT={gt.shape}, pred={pred.shape}")
    if np.any(gt > 3) or np.any(pred > 3):
        raise ValueError("检测到范围 0~3 以外的类别索引")
    packed = gt.astype(np.int64).ravel() * NUM_CLASSES + pred.astype(np.int64).ravel()
    return np.bincount(packed, minlength=16).reshape(NUM_CLASSES, NUM_CLASSES)


def class_ious(matrix: np.ndarray) -> list[float]:
    """与上游 absent_score=0 对齐，始终保留三类前景。"""
    tp = np.diag(matrix)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - tp
    return [float(tp[i] / union[i]) if union[i] else 0.0 for i in range(NUM_CLASSES)]


def serialize_temporal_state(model: torch.nn.Module) -> dict:
    """保存 forward_test 的历史 BEV 与车辆位姿，以便断点续跑。"""
    state = model.prev_frame_info
    bev = state.get("prev_bev")
    pos = state.get("prev_pos")
    angle = state.get("prev_angle")
    return {
        "prev_bev": bev.detach().cpu().clone() if bev is not None else None,
        "scene_token": state.get("scene_token"),
        "prev_pos": np.asarray(pos).tolist() if pos is not None else None,
        "prev_angle": float(angle) if angle is not None else None,
    }


def restore_temporal_state(model: torch.nn.Module, saved: dict, device: torch.device) -> None:
    """恢复模型在上次成功保存帧之后的时序状态。"""
    state = model.prev_frame_info
    bev = saved.get("prev_bev")
    state["prev_bev"] = bev.to(device) if bev is not None else None
    state["scene_token"] = saved.get("scene_token")
    pos = saved.get("prev_pos")
    state["prev_pos"] = np.asarray(pos, dtype=np.float64) if pos is not None else 0
    angle = saved.get("prev_angle")
    state["prev_angle"] = float(angle) if angle is not None else 0.0


def write_csv(per_frame: list[dict]) -> None:
    """保存按帧的轻量明细，方便进一步定位异常场景。"""
    fields = (
        "index", "sample_token", "scene_token", "foreground_miou_single_frame",
        "iou_divider", "iou_ped_crossing", "iou_boundary", "seconds",
    )
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_frame)
    os.replace(tmp, CSV_PATH)


def summarize(
    confusion: np.ndarray,
    upstream_metric: Any,
    metadata: dict,
    per_frame: list[dict],
    elapsed_seconds: float,
    complete: bool,
) -> dict:
    """输出两套评价器结果及差异；中途保存亦可读取。"""
    local = np.asarray(class_ious(confusion), dtype=np.float64)
    official = upstream_metric.compute().detach().cpu().numpy()
    max_diff = float(np.max(np.abs(local - official)))
    return {
        **metadata,
        "complete": complete,
        "processed_samples": len(per_frame),
        "foreground_miou_local": float(np.mean(local[1:])),
        "foreground_miou_upstream": float(np.mean(official[1:])),
        "class_names": list(CLASS_NAMES),
        "per_class_iou_local": local.tolist(),
        "per_class_iou_upstream": official.tolist(),
        "max_metric_difference": max_diff,
        "metric_match": bool(max_diff < 1e-5),
        "confusion_gt_rows_pred_cols": confusion.tolist(),
        "total_gt_pixels": int(confusion.sum()),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "per_frame_csv": str(CSV_PATH),
    }


def save_progress(
    model: torch.nn.Module,
    upstream_metric: Any,
    confusion: np.ndarray,
    per_frame: list[dict],
    metadata: dict,
    elapsed_seconds: float,
    *,
    complete: bool = False,
) -> dict:
    """持久化可恢复状态、累积指标和按帧信息。"""
    stats = summarize(confusion, upstream_metric, metadata, per_frame, elapsed_seconds, complete)
    saved = {
        "identity": metadata["identity"],
        "next_index": len(per_frame),
        "confusion": torch.from_numpy(confusion.copy()),
        "per_frame": per_frame,
        "elapsed_seconds": elapsed_seconds,
        "temporal_state": serialize_temporal_state(model),
        "metric_state": {
            name: getattr(upstream_metric, name).detach().cpu().clone()
            for name in ("true_positive", "false_positive", "false_negative", "support")
        },
    }
    tmp = PROGRESS_PATH.with_suffix(".pt.tmp")
    torch.save(saved, tmp)
    os.replace(tmp, PROGRESS_PATH)
    atomic_json(RESULT_PATH, stats)
    write_csv(per_frame)
    return stats


def main() -> None:
    """完整 Val 逐帧评估，支持独立输出和严格身份验证的断点续跑。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="从保存的时序状态恢复")
    parser.add_argument("--save-every", type=int, default=50, help="每 N 帧保存一次状态")
    args = parser.parse_args()
    if args.save_every < 1:
        raise ValueError("--save-every 必须大于 0")
    for path in (CONFIG, FULL_VAL_PKL, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f"缺少必要文件：{path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查当前 BEVFormer 环境")
    if args.resume and not PROGRESS_PATH.is_file():
        raise FileNotFoundError("没有找到断点；首次运行请去掉 --resume")
    if not args.resume and (RESULT_PATH.exists() or PROGRESS_PATH.exists()):
        raise FileExistsError(
            "目标目录已有结果，为防覆盖已停止；续跑请加 --resume"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_bev_seg_one_sample import (  # pylint: disable=import-outside-toplevel
        segmentation_classes,
        to_numpy,
        unwrap_single_aug,
    )

    # 上游测试管线的相对路径 data/nuscenes 要从项目仓库下解析。
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
    if len(dataset) != ALL_SAMPLES:
        raise RuntimeError(f"完整 Val 应为 6019 帧，实际为 {len(dataset)}")
    if not cfg.model.video_test_mode:
        raise RuntimeError("当前配置关闭了时序推理，拒绝继续")

    identity = {
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "config_sha256": sha256_file(CONFIG),
        "val_pkl_sha256": sha256_file(FULL_VAL_PKL),
        "sample_count": len(dataset),
    }
    metadata = {
        "experiment": "nuScenes v1.0-trainval official split, 6019-frame baseline",
        "note": (
            "遵循上游数据管线、模型 forward_test 和 IoU 类进行全量复现；"
            "不是作者原始 tools/test.py 的逐字相同执行入口，"
            "也未验证权重与作者发布 checkpoint 的 SHA256 是否相同。"
        ),
        "identity": identity,
        "checkpoint_path": str(CHECKPOINT),
        "val_pkl_path": str(FULL_VAL_PKL),
        "video_test_mode": bool(cfg.model.video_test_mode),
        "expected_samples": ALL_SAMPLES,
        "absent_class_iou": 0.0,
        "mIoU_definition": "global confusion then per-class IoU; mean foreground classes 1,2,3",
    }

    print("===== Step 25：nuScenes Val 6019 帧完整基线复现 =====", flush=True)
    print("场景数量:", len({info["scene_token"] for info in dataset.data_infos}), flush=True)
    print("验证帧数:", len(dataset), flush=True)
    print("时序模式:", cfg.model.video_test_mode, flush=True)
    print("Checkpoint SHA256:", identity["checkpoint_sha256"], flush=True)
    print("索引 SHA256:", identity["val_pkl_sha256"], flush=True)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    report = model.load_state_dict(checkpoint["state_dict"], strict=True)
    print(
        f"权重严格加载: missing={len(report.missing_keys)}, "
        f"unexpected={len(report.unexpected_keys)}",
        flush=True,
    )
    del checkpoint
    gc.collect()

    device = torch.device("cuda:0")
    model = model.to(device).eval()
    upstream_metric = IntersectionOverUnion(NUM_CLASSES).to(device)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_frame: list[dict] = []
    earlier_elapsed = 0.0

    if args.resume:
        progress = torch.load(str(PROGRESS_PATH), map_location="cpu", weights_only=True)
        if progress["identity"] != identity:
            raise RuntimeError("断点与当前 checkpoint/config/PKL 不一致，拒绝混用")
        per_frame = progress["per_frame"]
        confusion = progress["confusion"].cpu().numpy().astype(np.int64)
        earlier_elapsed = float(progress.get("elapsed_seconds", 0.0))
        restore_temporal_state(model, progress["temporal_state"], device)
        for name, value in progress["metric_state"].items():
            setattr(upstream_metric, name, value.to(device))
        if int(progress["next_index"]) != len(per_frame):
            raise RuntimeError("断点帧数与已保存记录不一致")
        print(f"断点恢复成功：下一帧索引 {len(per_frame)}", flush=True)
    else:
        print("全新开始：索引 0", flush=True)

    start = time.monotonic()
    next_index = len(per_frame)
    if next_index == ALL_SAMPLES:
        print("全部帧已经完成，无需再次推理", flush=True)
        return

    try:
        for index in range(next_index, ALL_SAMPLES):
            frame_start = time.monotonic()
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
                raise ValueError(f"第 {index} 帧六路相机形状异常：{tuple(img.shape)}")
            with torch.inference_mode():
                output = model(
                    return_loss=False,
                    rescale=True,
                    img=[img.to(device).clone()],
                    img_metas=[[copy.deepcopy(meta)]],
                )
            logits = output[0].get("seg_preds")
            if logits is None:
                raise RuntimeError(f"第 {index} 帧未返回 seg_preds")
            pred, _ = segmentation_classes(logits)
            this_confusion = confusion_matrix(gt, pred)
            confusion += this_confusion
            upstream_metric.update(
                torch.as_tensor(pred[None], device=device, dtype=torch.long),
                torch.as_tensor(gt[None], device=device, dtype=torch.long),
            )
            single = class_ious(this_confusion)
            per_frame.append({
                "index": index,
                "sample_token": str(dataset.data_infos[index]["token"]),
                "scene_token": str(dataset.data_infos[index]["scene_token"]),
                "foreground_miou_single_frame": float(np.mean(single[1:])),
                "iou_divider": single[1],
                "iou_ped_crossing": single[2],
                "iou_boundary": single[3],
                "seconds": round(time.monotonic() - frame_start, 3),
            })

            done = index + 1
            if done % args.save_every == 0 or done == ALL_SAMPLES:
                elapsed = earlier_elapsed + time.monotonic() - start
                stats = save_progress(
                    model, upstream_metric, confusion, per_frame, metadata,
                    elapsed, complete=done == ALL_SAMPLES,
                )
                print(
                    f"[{done}/{ALL_SAMPLES}] 前景 mIoU="
                    f"{stats['foreground_miou_upstream']:.4%} | "
                    f"两套评价器一致={stats['metric_match']} | "
                    f"累计耗时={elapsed / 60:.1f} 分钟",
                    flush=True,
                )
                if not stats["metric_match"]:
                    raise RuntimeError("上游评价器与 NumPy IoU 不一致，已停止")
    except Exception:
        print(
            f"[中断] 已保存断点至 {PROGRESS_PATH}；"
            "修复问题后用 --resume 从最后一次保存位置恢复",
            flush=True,
        )
        raise

    final = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    print("\n===== 完整 Val 最终结果 =====", flush=True)
    for name, value in zip(CLASS_NAMES, final["per_class_iou_upstream"]):
        print(f"{name}: {value:.4%}", flush=True)
    print(f"前景 mIoU: {final['foreground_miou_upstream']:.4%}", flush=True)
    print(f"评价器一致: {final['metric_match']}", flush=True)
    print(f"结果: {RESULT_PATH}", flush=True)
    print("Step 25: PASS", flush=True)


if __name__ == "__main__":
    main()

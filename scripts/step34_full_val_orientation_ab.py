#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 34：官方 nuScenes Val 6019 帧的连续时序 BEV 朝向配对消融。

A: 原始 BEVFormer-Seg 分割头；B: Step 33 中的 CCW90 方向候选。
两组使用同一 config/checkpoint/数据/GT 与 IoU 计算方法，并独立保存
各自的历史 BEV。该实验只是推理端反事实测试，不等于训练坐标修复。

运行：
    python -u scripts/step34_full_val_orientation_ab.py --save-every 50 \
        2>&1 | tee outputs/step34_run.log
中断后：
    python -u scripts/step34_full_val_orientation_ab.py --resume --save-every 50 \
        2>&1 | tee -a outputs/step34_run.log
"""

from __future__ import annotations

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
SEG_REPO = ROOT / "BEVFormer-seg"
CONFIG = SEG_REPO / "projects/configs/bevformer/bevformer_small_seg.py"
CHECKPOINT = Path("/mnt/g/nus/bevformer_small_seg.pth")
VAL_PKL = (
    ROOT
    / "outputs/drivelm_check/root_cause/step21_full_val/"
    "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
)
OUTPUT = ROOT / "outputs/drivelm_check/root_cause/step34_full_val_orientation_ab"
PROGRESS_PATH = OUTPUT / "progress.pt"
RESULTS_PATH = OUTPUT / "results.json"
PER_FRAME_PATH = OUTPUT / "per_frame.csv"
CLASS_NAMES = ("background", "divider", "ped_crossing", "boundary")
VARIANTS = ("original", "physical_ccw_candidate")
TOTAL_FRAMES = 6019
REFERENCE_FG_MIOU = 0.046416  # Step 25 完整 Val，连续时序基线（四舍五入）


def sha256_file(path: Path) -> str:
    """以分块读取方式计算文件摘要，不把大权重一次性读进内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def confusion_matrix(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """计算以 GT 为行、预测为列的四分类混淆矩阵。"""
    if gt.shape != (200, 400) or prediction.shape != gt.shape:
        raise ValueError(f"GT 和预测尺寸不符：{gt.shape} / {prediction.shape}")
    gt_flat = gt.ravel().astype(np.int64)
    pred_flat = prediction.ravel().astype(np.int64)
    if not np.isin(gt_flat, [0, 1, 2, 3]).all():
        raise ValueError("GT 中存在 0..3 之外的类别")
    if not np.isin(pred_flat, [0, 1, 2, 3]).all():
        raise ValueError("预测中存在 0..3 之外的类别")
    return np.bincount(gt_flat * 4 + pred_flat, minlength=16).reshape(4, 4)


def metrics(matrix: np.ndarray) -> dict[str, Any]:
    """先累积像素混淆矩阵，再算逐类 IoU 和前三景类 mIoU。"""
    tp = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - tp
    ious = np.divide(tp, union, out=np.zeros(4), where=union > 0)
    return {
        "class_iou": dict(zip(CLASS_NAMES, [float(x) for x in ious])),
        "foreground_miou": float(ious[1:].mean()),
        "confusion_gt_rows_pred_cols": matrix.tolist(),
    }


def capture_temporal_state(model: torch.nn.Module, *, to_cpu: bool) -> dict:
    """复制模型时间状态；磁盘断点保存时将 prev_bev 转移到 CPU。"""
    state = model.prev_frame_info
    bev = state.get("prev_bev")
    if bev is not None:
        bev = bev.detach().clone()
        if to_cpu:
            bev = bev.cpu()
    pos = state.get("prev_pos")
    angle = state.get("prev_angle")
    return {
        "prev_bev": bev,
        "scene_token": state.get("scene_token"),
        "prev_pos": np.asarray(pos).tolist() if pos is not None else None,
        "prev_angle": float(angle) if angle is not None else None,
    }


def restore_temporal_state(
    model: torch.nn.Module, state: dict, device: torch.device
) -> None:
    """在每个 A/B forward 之前恢复该分支自己的连续时序状态。"""
    target = model.prev_frame_info
    bev = state.get("prev_bev")
    target["prev_bev"] = bev.to(device) if bev is not None else None
    target["scene_token"] = state.get("scene_token")
    pos = state.get("prev_pos")
    target["prev_pos"] = np.asarray(pos, dtype=np.float64) if pos is not None else 0
    angle = state.get("prev_angle")
    target["prev_angle"] = float(angle) if angle is not None else 0.0


def atomic_json(path: Path, data: dict) -> None:
    """原子更新 JSON，避免中断留下半截结果。"""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def save_csv(records: list[dict]) -> None:
    """按帧记录用于识别改善/退化的场景和样本。"""
    fields = (
        "index", "sample_token", "scene_token", "original_fg_miou",
        "candidate_fg_miou", "delta_fg_miou", "changed_pixels", "seconds"
    )
    tmp = PER_FRAME_PATH.with_name(PER_FRAME_PATH.name + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    os.replace(tmp, PER_FRAME_PATH)


def build_result(
    identity: dict, matrices: dict[str, np.ndarray],
    records: list[dict], elapsed: float
) -> dict:
    """形成可直接归档的评估报告；未完成时标记为 partial。"""
    a = metrics(matrices["original"])
    b = metrics(matrices["physical_ccw_candidate"])
    return {
        "experiment": "Step34 full nuScenes Val orientation A/B, temporal on",
        "complete": len(records) == TOTAL_FRAMES,
        "processed_frames": len(records),
        "expected_frames": TOTAL_FRAMES,
        "scenes_represented": len({record["scene_token"] for record in records}),
        "identity": identity,
        "protocol": (
            "Same model, weights, 6019-sample Val, GT, and global IoU. "
            "Temporal ON, independent previous-BEV states per branch."
        ),
        "intervention": (
            "A: original seg head; B: flip BEV along dim=2 immediately "
            "before original feat_cropper.forward (Step33 CCW90 candidate)."
        ),
        "interpretation_limit": (
            "推理端空间方向反事实消融；未验证训练时坐标链和分割标签 "
            "的完整约定之前，不得作为正式模型修复。"
        ),
        "original": a,
        "physical_ccw_candidate": b,
        "aggregate_delta_percentage_points": round(
            100 * (b["foreground_miou"] - a["foreground_miou"]), 6
        ),
        "frames_improved": sum(row["delta_fg_miou"] > 0 for row in records),
        "frames_worsened": sum(row["delta_fg_miou"] < 0 for row in records),
        "frames_equal": sum(row["delta_fg_miou"] == 0 for row in records),
        "reference_step25_temporal_fg_miou": REFERENCE_FG_MIOU,
        "delta_original_vs_step25_percentage_points": round(
            100 * (a["foreground_miou"] - REFERENCE_FG_MIOU), 6
        ),
        "elapsed_seconds": round(elapsed, 2),
        "per_frame_csv": str(PER_FRAME_PATH),
    }


def save_progress(
    identity: dict, matrices: dict[str, np.ndarray], records: list[dict],
    branch_states: dict[str, dict], elapsed: float
) -> dict:
    """保存可恢复的两路时序状态及各自累计指标。"""
    saved_states = {}
    for name, state in branch_states.items():
        copied = dict(state)
        if copied["prev_bev"] is not None:
            copied["prev_bev"] = copied["prev_bev"].detach().cpu().clone()
        saved_states[name] = copied
    payload = {
        "identity": identity,
        "next_index": len(records),
        "matrices": {
            name: torch.from_numpy(matrix.copy()) for name, matrix in matrices.items()
        },
        "records": records,
        "branch_states": saved_states,
        "elapsed_seconds": elapsed,
    }
    tmp = PROGRESS_PATH.with_name(PROGRESS_PATH.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, PROGRESS_PATH)
    result = build_result(identity, matrices, records, elapsed)
    atomic_json(RESULTS_PATH, result)
    save_csv(records)
    return result


def main() -> None:
    """按时间顺序运行全量 A/B，定期保存并支持中断后原位续跑。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="从保存的进度续跑")
    parser.add_argument("--save-every", type=int, default=50)
    args = parser.parse_args()
    if args.save_every < 1:
        raise ValueError("--save-every 必须 >= 1")
    for path in (CONFIG, VAL_PKL, CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(f"缺少输入文件：{path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    if args.resume and not PROGRESS_PATH.is_file():
        raise FileNotFoundError("还没有断点；第一次运行不要加 --resume")
    if not args.resume and (PROGRESS_PATH.exists() or RESULTS_PATH.exists()):
        raise FileExistsError("Step34 输出已存在，为避免覆盖请使用 --resume")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # 使用 Step25/33 同一模型加载方式和实际 DataLoader。
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_bev_seg_one_sample import (
        segmentation_classes,
        to_numpy,
        unwrap_single_aug,
    )
    os.chdir(SEG_REPO)
    sys.path.insert(0, str(SEG_REPO))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    importlib.import_module("projects.mmdet3d_plugin")
    cfg = Config.fromfile(str(CONFIG))
    ds_cfg = copy.deepcopy(cfg.data.test)
    ds_cfg.ann_file = str(VAL_PKL)
    ds_cfg.test_mode = True
    ds_cfg.pop("samples_per_gpu", None)
    ds_cfg.pop("workers_per_gpu", None)
    dataset = build_dataset(ds_cfg)
    if len(dataset) != TOTAL_FRAMES:
        raise RuntimeError(f"预期 6019 帧，实际 {len(dataset)} 帧")
    if not cfg.model.video_test_mode:
        raise RuntimeError("配置没有启用 video_test_mode，不能与 Step25 直接比较")

    identity = {
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "config_sha256": sha256_file(CONFIG),
        "val_pkl_sha256": sha256_file(VAL_PKL),
        "sample_count": len(dataset),
        "video_test_mode": True,
        "intervention": "flip_dim2_before_feat_cropper",
    }
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    loaded = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("分割权重没有严格匹配")
    del checkpoint
    gc.collect()
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    cropper = model.pts_bbox_head.feat_cropper
    original_crop = cropper.forward

    def candidate_crop(bev: torch.Tensor) -> torch.Tensor:
        """只改变 Step33 已测试的方向候选，不修改网络权重。"""
        return original_crop(torch.flip(bev, dims=[2]))

    matrices = {
        name: np.zeros((4, 4), dtype=np.int64) for name in VARIANTS
    }
    branch_states = {
        name: capture_temporal_state(model, to_cpu=False) for name in VARIANTS
    }
    records: list[dict] = []
    previous_seconds = 0.0
    if args.resume:
        progress = torch.load(
            str(PROGRESS_PATH), map_location="cpu", weights_only=True
        )
        if progress["identity"] != identity:
            raise RuntimeError("本次环境、config、数据或 checkpoint 与断点不一致")
        records = progress["records"]
        if int(progress["next_index"]) != len(records):
            raise RuntimeError("断点索引与帧记录不一致")
        matrices = {
            name: progress["matrices"][name].numpy().astype(np.int64)
            for name in VARIANTS
        }
        branch_states = progress["branch_states"]
        previous_seconds = float(progress.get("elapsed_seconds", 0.0))
        print(f"[续跑] 从帧 {len(records)}/{TOTAL_FRAMES} 恢复", flush=True)
    else:
        print("[首跑] 从第 0 帧开始，全量连续时序 A/B 对照", flush=True)
    print(f"checkpoint: {identity['checkpoint_sha256']}", flush=True)
    print(f"Val PKL: {identity['val_pkl_sha256']}", flush=True)
    print(f"样本数: {len(dataset)}", flush=True)

    started = time.monotonic()
    try:
        for index in range(len(records), TOTAL_FRAMES):
            frame_start = time.monotonic()
            item = dataset[index]
            img = unwrap_single_aug(item["img"])
            meta = unwrap_single_aug(item["img_metas"])
            gt = np.squeeze(
                to_numpy(unwrap_single_aug(item["semantic_indices"]))
            ).astype(np.uint8)
            if not torch.is_tensor(img) or not isinstance(meta, dict):
                raise TypeError(f"帧 {index} 的数据格式异常")
            if img.ndim == 4:
                img = img.unsqueeze(0)
            if img.ndim != 5 or tuple(img.shape[:2]) != (1, 6):
                raise ValueError(f"帧 {index} 的六路相机尺寸异常：{tuple(img.shape)}")
            img_gpu = img.to(device)
            predictions = {}
            for variant in VARIANTS:
                restore_temporal_state(model, branch_states[variant], device)
                cropper.forward = (
                    original_crop if variant == "original" else candidate_crop
                )
                with torch.inference_mode():
                    output = model(
                        return_loss=False,
                        rescale=True,
                        img=[img_gpu.clone()],
                        img_metas=[[copy.deepcopy(meta)]],
                    )
                logits = output[0].get("seg_preds")
                if logits is None:
                    raise RuntimeError(f"帧 {index} ({variant}) 没有 seg_preds")
                pred, _ = segmentation_classes(logits)
                matrices[variant] += confusion_matrix(gt, pred)
                predictions[variant] = pred
                branch_states[variant] = capture_temporal_state(
                    model, to_cpu=False
                )
                del output, logits

            raw = metrics(confusion_matrix(gt, predictions["original"]))
            candidate = metrics(
                confusion_matrix(gt, predictions["physical_ccw_candidate"])
            )
            info = dataset.data_infos[index]
            records.append({
                "index": index,
                "sample_token": str(info["token"]),
                "scene_token": str(info["scene_token"]),
                "original_fg_miou": raw["foreground_miou"],
                "candidate_fg_miou": candidate["foreground_miou"],
                "delta_fg_miou": (
                    candidate["foreground_miou"] - raw["foreground_miou"]
                ),
                "changed_pixels": int(np.count_nonzero(
                    predictions["original"] != predictions["physical_ccw_candidate"]
                )),
                "seconds": round(time.monotonic() - frame_start, 3),
            })
            del img_gpu, predictions

            done = index + 1
            if done % args.save_every == 0 or done == TOTAL_FRAMES:
                elapsed = previous_seconds + time.monotonic() - started
                summary = save_progress(
                    identity, matrices, records, branch_states, elapsed
                )
                raw_fg = summary["original"]["foreground_miou"]
                candidate_fg = summary["physical_ccw_candidate"]["foreground_miou"]
                print(
                    f"[{done}/{TOTAL_FRAMES}] "
                    f"原始={raw_fg:.4%} | CCW90={candidate_fg:.4%} | "
                    f"提升={summary['aggregate_delta_percentage_points']:+.3f}pp | "
                    f"耗时={elapsed / 60:.1f}min",
                    flush=True,
                )
    except (Exception, KeyboardInterrupt):
        print(
            f"[中断] 最近成功保存的断点：{PROGRESS_PATH}；"
            "使用 --resume 从上一次保存位置恢复。",
            flush=True,
        )
        raise
    finally:
        cropper.forward = original_crop

    final = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    print("\n===== Step 34 最终结果：全量 Val、连续时序 =====", flush=True)
    for variant in VARIANTS:
        result = final[variant]
        print(f"{variant} 前景 mIoU：{result['foreground_miou']:.4%}")
        for name, value in result["class_iou"].items():
            print(f"  {name}: {value:.4%}")
    print(
        f"总提升：{final['aggregate_delta_percentage_points']:+.4f} 百分点",
        flush=True,
    )
    print(
        "逐帧改善/恶化/相同："
        f"{final['frames_improved']}/{final['frames_worsened']}/"
        f"{final['frames_equal']}",
        flush=True,
    )
    print(
        "原始分支 vs Step25 历史基线差值："
        f"{final['delta_original_vs_step25_percentage_points']:+.5f} 百分点",
        flush=True,
    )
    print(f"结果文件：{RESULTS_PATH}", flush=True)
    print("注意：候选不代表已完成训练/推理空间坐标修复。", flush=True)


if __name__ == "__main__":
    main()

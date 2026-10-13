#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BEVFormer-Small：nuScenes Mini 验证集逐帧 GPU 分割评估。

在项目根目录运行：
    python scripts/run_bev_seg_minival.py --limit 81

特点：
- 复用已跑通的 run_bev_seg_one_sample.py 的输入展开及标签处理方法。
- 只构建一次数据集/模型、只加载一次权重，再依序处理全部样本。
- 使用整个验证集各类别累计 intersection/union 计算 mIoU，
  而非先求每帧 IoU 再平均（两种结果并不相同）。
- 保存阶段性结果，出错时也尽量保留已完成的评估记录。
- 不修改检测仓库、不训练模型、不创建 Markdown。

数据集：nuScenes v1.0-mini / mini_val；不是完整 DriveLM 验证集。
"""

import argparse
import copy
import csv
import gc
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

# 从已成功运行的单帧版本复用确定无误的工具方法，保持数据处理一致。
from run_bev_seg_one_sample import (
    ROOT, SEG_REPO, OUT, CONFIG, DEFAULT_CKPT,
    unwrap_single_aug, to_numpy, segmentation_classes, make_figure,
)

OUT = ROOT / "outputs" / "bev_seg_no_temporal"

CLASS_NAMES = ("background", "divider", "ped_crossing", "boundary")
NUM_CLASSES = len(CLASS_NAMES)


def confusion_of_frame(gt, pred):
    """行是真值、列是预测；使用像素级混淆矩阵加速累计。"""
    if gt.shape != pred.shape:
        raise ValueError(f"预测真值维度不符：GT {gt.shape} vs PRED {pred.shape}")
    g = gt.reshape(-1).astype(np.int64, copy=False)
    p = pred.reshape(-1).astype(np.int64, copy=False)
    if np.any((g < 0) | (g >= NUM_CLASSES)):
        raise ValueError("GT 含未知类别，停止而不是悄悄丢弃像素")
    if np.any((p < 0) | (p >= NUM_CLASSES)):
        raise ValueError("预测含未知类别，停止而不是悄悄丢弃像素")
    return np.bincount(g * NUM_CLASSES + p, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def stats_from_confusion(confusion):
    gt_count = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    intersection = np.diag(confusion)
    union = gt_count + pred_count - intersection
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        rows.append({
            "class_id": i,
            "class_name": name,
            "gt_pixels": int(gt_count[i]),
            "pred_pixels": int(pred_count[i]),
            "intersection": int(intersection[i]),
            "union": int(union[i]),
            "iou": float(intersection[i] / union[i]) if union[i] else None,
        })
    fg_values = [r["iou"] for r in rows[1:] if r["iou"] is not None]
    all_values = [r["iou"] for r in rows if r["iou"] is not None]
    return rows, (float(np.mean(fg_values)) if fg_values else None), (float(np.mean(all_values)) if all_values else None)


def save_progress(confusion, per_frame, output_dir, requested, dataset_len, complete=False, error=None):
    """原子写入 JSON/CSV；终端异常退出时仍保留之前已成功的帧。"""
    rows, fg, all_cls = stats_from_confusion(confusion)
    info = {
        "dataset": "nuScenes v1.0-mini / mini_val",
        "important_note": "仅为 nuScenes Mini BEV 分割评估，不是 DriveLM 正式验证集成绩",
        "model": "BEVFormer-Small segmentation",
        "requested_samples": requested,
        "dataset_samples": dataset_len,
        "evaluated_samples": len(per_frame),
        "complete": bool(complete),
        "error": error,
        "foreground_mIoU": fg,
        "mIoU_including_background": all_cls,
        "classes": rows,
        "confusion_matrix_gt_rows_pred_cols": confusion.tolist(),
        "aggregation": "global intersection/union summed over frames; NOT average of per-frame IoUs",
        "input_order": "sequential validation dataset order; model forward_test retains its native temporal state",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "seg_minival_metrics.json"
    temp = output_dir / "seg_minival_metrics.json.tmp"
    temp.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(target)
    np.save(output_dir / "seg_minival_confusion.npy", confusion)
    csv_path = output_dir / "seg_minival_per_frame.csv"
    tmp_csv = output_dir / "seg_minival_per_frame.csv.tmp"
    with tmp_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "sample_token", "scene_token", "foreground_mIoU_single_frame",
            "IoU_background", "IoU_divider", "IoU_ped_crossing", "IoU_boundary", "seconds",
        ])
        writer.writeheader()
        writer.writerows(per_frame)
    tmp_csv.replace(csv_path)
    return info


def main():
    parser = argparse.ArgumentParser(description="BEVFormer-Small 分割：81 帧 Mini 验证集 mIoU 评估")
    parser.add_argument("--limit", type=int, default=81, help="最多处理多少帧；默认 81")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--save-every", type=int, default=10, help="每几帧保存一次中间结果")
    parser.add_argument("--preview-indices", default="0,40,80", help="保存对比图的帧下标，如 0,40,80；留空关闭")
    args = parser.parse_args()
    if args.limit <= 0 or args.save_every <= 0:
        raise ValueError("--limit 和 --save-every 必须为正整数")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。请激活之前的 .venv-bevformer 环境")
    if not args.checkpoint.is_file() or not CONFIG.is_file():
        raise FileNotFoundError("分割权重或模型配置不存在")
    preview = {int(x.strip()) for x in args.preview_indices.split(",") if x.strip()}
    OUT.mkdir(parents=True, exist_ok=True)
    image_dir = OUT / "images"
    if preview:
        image_dir.mkdir(parents=True, exist_ok=True)

    # 配置内使用相对路径 data/nuscenes，沿用单帧成功验证的目录和插件导入方式。
    os.chdir(SEG_REPO)
    sys.path.insert(0, str(SEG_REPO))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    print("[1/4] 导入分割插件，构建验证集", flush=True)
    importlib.import_module("projects.mmdet3d_plugin")
    cfg = Config.fromfile(str(CONFIG))
    cfg.model.video_test_mode = False
    print("[对照实验] 已关闭历史 BEV", flush=True)
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.test_mode = True
    for key in ("samples_per_gpu", "workers_per_gpu"):
        dataset_cfg.pop(key, None)
    dataset = build_dataset(dataset_cfg)
    total = min(len(dataset), args.limit)
    print(f"[数据集] 共 {len(dataset)} 帧，本次处理 {total} 帧", flush=True)

    print("[2/4] 构建 GPU 分割模型并严格加载权重（只执行一次）", flush=True)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    result = model.load_state_dict(state, strict=True)
    print(f"[权重] missing={len(result.missing_keys)}，unexpected={len(result.unexpected_keys)}", flush=True)
    del state, ckpt
    gc.collect()
    model = model.to(torch.device("cuda:0")).eval()

    print("[3/4] 开始逐帧 GPU 推理和像素级统计", flush=True)
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_frame = []
    start_time = time.monotonic()
    active_index = -1
    try:
        for index in range(total):
            active_index = index
            began = time.monotonic()
            item = dataset[index]
            img = unwrap_single_aug(item["img"])
            meta = unwrap_single_aug(item["img_metas"])
            gt_raw = unwrap_single_aug(item["semantic_indices"])
            if not torch.is_tensor(img) or not isinstance(meta, dict):
                raise TypeError(f"第 {index} 帧数据结构异常：img={type(img).__name__}, meta={type(meta).__name__}")
            if img.ndim == 4:
                img = img.unsqueeze(0)
            if img.ndim != 5 or img.shape[0] != 1 or img.shape[1] != 6:
                raise ValueError(f"第 {index} 帧六路相机维度异常：{list(img.shape)}")
            gt = np.squeeze(to_numpy(gt_raw)).astype(np.int64)
            if gt.ndim != 2:
                raise ValueError(f"第 {index} 帧 GT 维度异常：{gt.shape}")
            img = img.to(torch.device("cuda:0"), non_blocking=False)
            with torch.no_grad():
                outputs = model(return_loss=False, rescale=True, img=[img], img_metas=[[meta]])
            if not isinstance(outputs, (list, tuple)) or not outputs or not isinstance(outputs[0], dict):
                raise TypeError(f"第 {index} 帧推理输出格式异常：{type(outputs).__name__}")
            logits = outputs[0].get("seg_preds")
            if logits is None:
                raise RuntimeError(f"第 {index} 帧推理没有 seg_preds 字段")
            pred, logits_shape = segmentation_classes(logits)
            if index == 0:
                print(f"[预测] logits 形状={logits_shape}, GT 形状={gt.shape}", flush=True)
            cm = confusion_of_frame(gt, pred)
            confusion += cm
            one_rows, one_fg, _ = stats_from_confusion(cm)
            # 第一帧应与此前单帧实验一致；若不同，提醒检查时序状态或测试配置。
            if index == 0:
                old_path = OUT / "seg_metrics_sample0.json"
                if old_path.exists():
                    try:
                        prior = json.loads(old_path.read_text(encoding="utf-8")).get(
                            "foreground_mIoU_single_frame")
                        if prior is not None:
                            diff = abs(float(prior) - float(one_fg))
                            print(f"[复现检查] 第一帧 mIoU：本次={one_fg:.6f}，此前={float(prior):.6f}，差值={diff:.8f}", flush=True)
                            if diff > 1e-4:
                                print("[警告] 第一帧与此前结果不一致，请核查随机性、缓存或输入流程", flush=True)
                    except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
                        print(f"[提示] 无法比较既有单帧指标：{exc}", flush=True)
            elapsed = round(time.monotonic() - began, 3)
            per_frame.append({
                "index": index,
                "sample_token": str(dataset.data_infos[index].get("token", "")),
                "scene_token": str(meta.get("scene_token", "")),
                "foreground_mIoU_single_frame": one_fg,
                **{f"IoU_{r['class_name']}": r["iou"] for r in one_rows},
                "seconds": elapsed,
            })
            if index in preview:
                figure_path = image_dir / f"seg_comparison_minival_{index:02d}.png"
                make_figure(gt, pred, figure_path)
                print(f"  [图片] {figure_path}", flush=True)
            if (index + 1) % args.save_every == 0 or index + 1 == total:
                progress = save_progress(confusion, per_frame, OUT, total, len(dataset), complete=index + 1 == total)
                print(f"[{index+1}/{total}] 累计前景 mIoU={progress['foreground_mIoU']:.6f}，已保存中间结果", flush=True)
            else:
                print(f"[{index+1}/{total}] 当前帧前景 mIoU={one_fg:.6f}，耗时={elapsed:.2f}s", flush=True)
            del img, outputs, logits
    except BaseException as exc:
        save_progress(confusion, per_frame, OUT, total, len(dataset), complete=False,
                      error=f"frame_index={active_index}; {type(exc).__name__}: {exc}")
        print(f"[错误] 第 {active_index} 帧失败；已完成 {len(per_frame)} 帧，中间结果已保存", flush=True)
        raise

    print("[4/4] 完整验证集评估完成", flush=True)
    result = save_progress(confusion, per_frame, OUT, total, len(dataset), complete=True)
    print("=" * 52)
    print(f"评估帧数：{len(per_frame)}/{len(dataset)}")
    for row in result["classes"]:
        print(f"{row['class_name']}: IoU={row['iou']}，GT 像素={row['gt_pixels']}，预测像素={row['pred_pixels']}")
    print(f"Mini 验证集前景 mIoU：{result['foreground_mIoU']:.6f}")
    print(f"Mini 验证集含背景 mIoU：{result['mIoU_including_background']:.6f}")
    print(f"总耗时：{time.monotonic()-start_time:.1f}s")
    print("指标 JSON：", OUT / "seg_minival_metrics.json")
    print("逐帧 CSV：", OUT / "seg_minival_per_frame.csv")
    print("注意：这不是 DriveLM 正式验证集评估成绩")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[失败] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)

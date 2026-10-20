#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 42：作者原始测试入口与 Step25 手动推理的成对对照。

只读官方权重和完整 Val 元数据；生成独立前 N 帧子集，并且只构建一个模型。
作者原始 single_gpu_test 使用 MMDataParallel/DataLoader；Step25 使用手动包装。
两次测试使用相同模型、同一子集、时序 ON，记录各类别累计 IoU。

使用：python -u scripts/step42_upstream_vs_manual.py --count 30
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer-seg"
CONFIG = REPO / "projects/configs/bevformer/bevformer_small_seg.py"
WEIGHTS = Path("/mnt/g/nus/bevformer_small_seg.pth")
FULL_VAL = (
    ROOT / "outputs/drivelm_check/root_cause/step21_full_val"
    / "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
)
OUTPUT = ROOT / "outputs/drivelm_check/root_cause/step42_upstream_entry_ab"
CLASS_NAMES = ("background", "divider", "ped_crossing", "boundary")


def confusion_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """GT 为行，Prediction 为列的 4×4 混淆矩阵。"""
    if gt.shape != (200, 400) or pred.shape != gt.shape:
        raise ValueError(f"GT/预测尺寸异常：{gt.shape}/{pred.shape}")
    if gt.min() < 0 or pred.min() < 0 or gt.max() > 3 or pred.max() > 3:
        raise ValueError("分割标签不属于 0..3")
    return np.bincount(
        (gt.ravel().astype(np.int64) * 4 + pred.ravel()).astype(np.int64),
        minlength=16,
    ).reshape(4, 4)


def summarize(confusion: np.ndarray) -> dict[str, Any]:
    """采用作者前景 mIoU 公式汇总累计混淆矩阵。"""
    tp = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - np.diag(confusion)
    vals = np.divide(tp, union, out=np.zeros(4), where=union > 0)
    return {
        "class_iou": {name: float(v) for name, v in zip(CLASS_NAMES, vals)},
        "foreground_miou": float(vals[1:].mean()),
        "confusion_gt_rows_pred_cols": confusion.tolist(),
    }


def main() -> None:
    """运行上游路径/手动路径对照，不修改原仓库文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()
    if not 2 <= args.count <= 150:
        parser.error("--count 应处于 2～150，先进行小规模诊断")
    for required in (CONFIG, WEIGHTS, FULL_VAL):
        if not required.is_file():
            raise FileNotFoundError(f"缺少文件：{required}")
    if not torch.cuda.is_available():
        raise RuntimeError("当前 CUDA 不可用")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    subset_path = OUTPUT / "official_val_first_n.pkl"
    with FULL_VAL.open("rb") as file:
        obj = pickle.load(file)
    ordered = sorted(obj["infos"], key=lambda row: row["timestamp"])
    subset = {"infos": ordered[: args.count], "metadata": obj["metadata"]}
    with subset_path.open("wb") as file:
        pickle.dump(subset, file, protocol=pickle.HIGHEST_PROTOCOL)
    del obj, ordered

    # 维持作者原始项目的相对数据路径和类注册机制。
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(ROOT / "scripts"))
    importlib.import_module("projects.mmdet3d_plugin")

    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.apis import test as upstream_test
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from run_bev_seg_one_sample import (
        segmentation_classes,
        to_numpy,
        unwrap_single_aug,
    )

    cfg = Config.fromfile(str(CONFIG))
    cfg.data.test.ann_file = str(subset_path)
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    if not cfg.model.video_test_mode:
        raise RuntimeError("必须保持与完整 Val 一致的 temporal ON 模式")
    dataset = build_dataset(cfg.data.test)
    if len(dataset) != args.count:
        raise RuntimeError(f"样本数量不匹配：{len(dataset)}")

    # 原作者使用的 DataLoader 路径。workers=0 减少内存压力。
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(WEIGHTS), map_location="cpu")
    model = model.cuda().eval()

    captured = {}
    metric_cls = upstream_test.IntersectionOverUnion

    class CapturedMetric(metric_cls):
        """不改变原指标，仅记录原作者调用 compute() 的结果。"""

        def compute(self):
            values = super().compute()
            captured["class_iou"] = values.detach().cpu().tolist()
            return values

    upstream_test.IntersectionOverUnion = CapturedMetric
    print(f"[1/2] 作者 single_gpu_test · {args.count} 帧 · 时序 ON", flush=True)
    try:
        upstream_test.single_gpu_test(
            MMDataParallel(model, device_ids=[0]), loader, False, None
        )
    finally:
        upstream_test.IntersectionOverUnion = metric_cls
    if "class_iou" not in captured:
        raise RuntimeError("作者测试入口未计算分割 IoU，请检查日志")

    # 复用同一个模型，重置历史状态，再运行 Step25 的手动包装。
    state = model.prev_frame_info
    state.update({
        "scene_token": None,
        "prev_bev": None,
        "prev_pos": 0,
        "prev_angle": 0,
    })
    torch.cuda.empty_cache()
    manual_conf = np.zeros((4, 4), dtype=np.int64)
    per_frame = []
    print(f"[2/2] Step25 手动测试 · {args.count} 帧 · 时序 ON", flush=True)
    for index in range(args.count):
        item = dataset[index]
        image = unwrap_single_aug(item["img"])
        meta = unwrap_single_aug(item["img_metas"])
        gt = np.squeeze(
            to_numpy(unwrap_single_aug(item["semantic_indices"]))
        ).astype(np.uint8)
        if image.ndim == 4:
            image = image.unsqueeze(0)
        if not torch.is_tensor(image) or image.ndim != 5:
            raise ValueError(f"第 {index} 帧相机张量格式异常")
        with torch.inference_mode():
            output = model(
                return_loss=False,
                rescale=True,
                img=[image.cuda(non_blocking=False).clone()],
                img_metas=[[copy.deepcopy(meta)]],
            )
        pred, _ = segmentation_classes(output[0]["seg_preds"])
        matrix = confusion_matrix(gt, pred)
        manual_conf += matrix
        per_frame.append({
            "index": index,
            "token": dataset.data_infos[index]["token"],
            "scene_token": dataset.data_infos[index]["scene_token"],
            "single_frame_miou": summarize(matrix)["foreground_miou"],
        })
        if (index + 1) % 10 == 0 or index + 1 == args.count:
            print(f"  [{index + 1}/{args.count}]", flush=True)
    manual = summarize(manual_conf)
    upstream_iou = np.asarray(captured["class_iou"], dtype=float)
    manual_iou = np.asarray(list(manual["class_iou"].values()), dtype=float)
    maximum_difference = float(np.max(np.abs(upstream_iou - manual_iou)))
    results = {
        "purpose": "same checkpoint, same subset, original author test path vs Step25 path",
        "temporal_on": True,
        "frames": args.count,
        "author_class_iou": dict(zip(CLASS_NAMES, upstream_iou.tolist())),
        "author_foreground_miou": float(upstream_iou[1:].mean()),
        "manual": manual,
        "maximum_class_iou_difference": maximum_difference,
        "matching_within_1e_5": maximum_difference < 1e-5,
        "limit": (
            "对比推理入口是否等价；即便通过也不能证明原作者报告的"
            "42.62% 与当前官方 Val 的数据协议完全相同"
        ),
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "per_frame_manual.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_frame[0]))
        writer.writeheader()
        writer.writerows(per_frame)
    print("\n=== Step42 双路径结果 ===", flush=True)
    print(f"作者上游 mIoU: {results['author_foreground_miou']:.4%}", flush=True)
    print(f"Step25 手动 mIoU: {manual['foreground_miou']:.4%}", flush=True)
    print(f"类别最大差异: {maximum_difference:.8f}", flush=True)
    print(f"两条推理入口一致: {results['matching_within_1e_5']}", flush=True)
    print(f"结果: {OUTPUT / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()

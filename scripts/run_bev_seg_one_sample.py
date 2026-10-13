#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BEVFormer-Small 分割：单帧 GPU 推理 + 真实地图标签比较（不修改原仓库）。

用法（在项目根目录执行）：
    python scripts/run_bev_seg_one_sample.py --index 0

输出目录：outputs/bev_seg/
    seg_pred_sample0.npy      单帧预测标签
    seg_gt_gpu_sample0.npy    同一次数据管线生成的真实标签
    seg_comparison_sample0.png 预测与真值对比图
    seg_metrics_sample0.json  类别 IoU、前景 mIoU 等

注意：单帧 mIoU 只是调试指标，不代表完整 Mini 或 DriveLM 正式验证集成绩。
"""

import argparse
import copy
import gc
import importlib
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SEG_REPO = ROOT / "BEVFormer-seg"
OUT = ROOT / "outputs" / "bev_seg"
DEFAULT_CKPT = Path("checkpoints/bevformer_small_seg.pth")
CONFIG = SEG_REPO / "projects/configs/bevformer/bevformer_small_seg.py"


def unwrap_single_aug(value):
    """提取 MMDetection 测试管线中一份增强数据的底层对象。"""
    from mmcv.parallel import DataContainer
    while True:
        if isinstance(value, DataContainer):
            value = value.data
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        else:
            return value


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def segmentation_classes(tensor):
    """仅接受明确的分割 logits 形状，避免错误轴计算。"""
    x = to_numpy(tensor)
    if x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == 4:
        return np.argmax(x[0], axis=0).astype(np.uint8), list(x.shape)
    if x.ndim == 3 and x.shape[0] == 4:
        return np.argmax(x, axis=0).astype(np.uint8), list(x.shape)
    raise ValueError(f"模型输出 seg_preds 形状不符合预期（B,4,H,W）：{x.shape}")


def calculate_metrics(gt, pred):
    """按像素交并比计算类别指标；无该类且无预测时不计入均值。"""
    labels = ["背景", "divider", "ped_crossing", "boundary"]
    rows = []
    for c, name in enumerate(labels):
        intersection = int(np.logical_and(gt == c, pred == c).sum())
        union = int(np.logical_or(gt == c, pred == c).sum())
        rows.append({
            "class_id": c, "class_name": name,
            "intersection": intersection, "union": union,
            "gt_pixels": int((gt == c).sum()),
            "pred_pixels": int((pred == c).sum()),
            "iou": round(intersection / union, 6) if union > 0 else None,
        })
    fg = [r["iou"] for r in rows[1:] if r["iou"] is not None]
    return rows, (float(np.mean(fg)) if fg else None)


def make_figure(gt, pred, output_file):
    # 使用同一色阶便于对照；没有对预测做任何几何校正或后处理。
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    for ax, data, title in zip(axes, [gt, pred], ["Map Ground Truth", "BEVFormer-Small Prediction"]):
        im = ax.imshow(data, origin="lower", cmap="viridis", vmin=0, vmax=3,
                       interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
    fig.colorbar(im, ax=axes, ticks=[0, 1, 2, 3], label="Class Index", shrink=0.8)
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="BEVFormer 分割单帧 GPU 推理与真值比较")
    parser.add_argument("--index", type=int, default=0, help="Mini 验证集中的样本序号")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查当前 .venv-bevformer 环境")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到分割权重：{args.checkpoint}")
    if not CONFIG.is_file():
        raise FileNotFoundError(f"找不到分割配置：{CONFIG}")
    OUT.mkdir(parents=True, exist_ok=True)

    # 这个第三方仓库使用相对 data/nuscenes 路径；先切换进去，再注册插件。
    os.chdir(SEG_REPO)
    sys.path.insert(0, str(SEG_REPO))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    print("[1/5] 导入第三方分割插件，读取配置", flush=True)
    importlib.import_module("projects.mmdet3d_plugin")
    cfg = Config.fromfile(str(CONFIG))
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.test_mode = True
    dataset_cfg.pop("samples_per_gpu", None)  # DataLoader 参数，不能交给数据集 __init__
    print("[2/5] 构建验证集并读取单帧图像 / 地图真值", flush=True)
    dataset = build_dataset(dataset_cfg)
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"样本序号 {args.index} 超出范围 (0~{len(dataset)-1})")
    item = dataset[args.index]
    print("样本 keys:", list(item), flush=True)

    img = unwrap_single_aug(item["img"])
    meta = unwrap_single_aug(item["img_metas"])
    gt_raw = unwrap_single_aug(item["semantic_indices"])
    if not torch.is_tensor(img) or not isinstance(meta, dict):
        raise TypeError(f"意外的数据结构：img={type(img)}, img_metas={type(meta)}")
    if img.ndim == 4:  # (六路相机,C,H,W) -> (batch,六路相机,C,H,W)
        img = img.unsqueeze(0)
    if img.ndim != 5 or img.shape[0] != 1 or img.shape[1] != 6:
        raise ValueError(f"图像维度不符合预期 (1,6,C,H,W)：{tuple(img.shape)}")
    gt = np.squeeze(to_numpy(gt_raw)).astype(np.int64)
    if gt.ndim != 2 or not set(np.unique(gt).tolist()).issubset({0, 1, 2, 3}):
        raise ValueError(f"真值标签异常：shape={gt.shape}, unique={np.unique(gt)}")
    print(f"图像形状: {tuple(img.shape)}；真值形状: {gt.shape}；类别: {np.unique(gt).tolist()}", flush=True)

    print("[3/5] 构建模型并严格加载预训练权重（可能需要几十秒）", flush=True)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    load_result = model.load_state_dict(state, strict=True)
    print(f"权重加载完成: missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}", flush=True)
    del ckpt, state
    gc.collect()
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    img = img.to(device, non_blocking=False)

    # 采用仓库定义的真实 forward_test 路径（首次样本 prev_bev=None）。
    # 这里不用旧版 MMCV 的数据并行封装，避免 PyTorch 2.8 API 兼容性问题。
    print("[4/5] 执行单帧 GPU 分割推理", flush=True)
    with torch.no_grad():
        outputs = model(return_loss=False, rescale=True, img=[img], img_metas=[[meta]])
    if not isinstance(outputs, (list, tuple)) or not outputs or not isinstance(outputs[0], dict):
        raise TypeError(f"推理结果不是期望的 list[dict]：{type(outputs)}")
    logits = outputs[0].get("seg_preds")
    if logits is None:
        raise RuntimeError(f"未发现分割结果 seg_preds；返回字段：{list(outputs[0])}")
    pred, logits_shape = segmentation_classes(logits)
    if pred.shape != gt.shape:
        raise ValueError(f"预测与真值网格不一致：pred={pred.shape}, gt={gt.shape}；不做未经验证的转置或缩放")

    print("[5/5] 保存预测、对比图及单帧 IoU", flush=True)
    np.save(OUT / f"seg_pred_sample{args.index}.npy", pred)
    np.save(OUT / f"seg_gt_gpu_sample{args.index}.npy", gt.astype(np.uint8))
    make_figure(gt, pred, OUT / f"seg_comparison_sample{args.index}.png")
    rows, fg_miou = calculate_metrics(gt, pred)
    payload = {
        "说明": "这是 nuScenes Mini 单帧分割评估，不是 DriveLM 正式验证集 mIoU",
        "dataset": "nuScenes Mini / mini_val",
        "sample_index": args.index,
        "sample_token": str(dataset.data_infos[args.index].get("token", "")),
        "checkpoint": str(args.checkpoint),
        "prediction_logits_shape": logits_shape,
        "mask_shape": list(gt.shape),
        "per_class": rows,
        "foreground_mIoU_single_frame": round(fg_miou, 6) if fg_miou is not None else None,
        "classes_included_in_foreground_mean": [r["class_id"] for r in rows[1:] if r["iou"] is not None],
    }
    (OUT / f"seg_metrics_sample{args.index}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n========== 单帧 GPU 分割推理成功 ==========")
    print("预测 logits 形状:", logits_shape)
    for r in rows:
        print(f"  {r['class_name']}: IoU={r['iou']} | GT 像素={r['gt_pixels']} | 预测像素={r['pred_pixels']}")
    print("该帧前景 mIoU:", fg_miou)
    print("图片:", OUT / f"seg_comparison_sample{args.index}.png")
    print("下一步：确认预测合理，再扩展到完整 Mini 验证集")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[失败] {type(exc).__name__}: {exc}", flush=True)
        raise

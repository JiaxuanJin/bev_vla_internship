"""对 BEVFormer-Seg 执行非破坏性的真实预测轴转置消融。

实验仅在同一输入、配置、权重下取消分割头送入 Slicer 前的空间转置。
通过临时 forward_pre_hook 实现，不修改第三方仓库或原始检查点。
所有输出写到 outputs/drivelm_check/root_cause/ 下，不是官方评估成绩。

使用方法（在 ~/projects/bev_vla 根目录运行）：
    python -u scripts/step7_bev_axis_ablation.py --indices 0,1,80
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_bev_seg_minival import CLASS_NAMES, confusion_of_frame, stats_from_confusion
from run_bev_seg_one_sample import (
    CONFIG,
    DEFAULT_CKPT,
    OUT as MINI_OUT,
    ROOT,
    SEG_REPO,
    segmentation_classes,
    to_numpy,
    unwrap_single_aug,
)

OUTPUT_DIR = ROOT / "outputs" / "drivelm_check" / "root_cause"


def sha256_file(path: Path) -> str:
    """计算配置或检查点的 SHA256，用于保证实验可追溯。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """将类别 1–3 合并为前景，计算不区分类别的二值 IoU。"""
    target = gt > 0
    predicted = pred > 0
    union = np.count_nonzero(target | predicted)
    return float(np.count_nonzero(target & predicted) / union) if union else 1.0


def compute_metrics(gt: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    """计算四类分割 IoU 与辅助诊断二值 IoU。"""
    confusion = confusion_of_frame(gt, pred)
    classes, foreground_miou, all_miou = stats_from_confusion(confusion)
    return {
        "foreground_binary_iou": binary_iou(gt, pred),
        "foreground_miou": foreground_miou,
        "including_background_miou": all_miou,
        "classes": classes,
        "confusion_matrix": confusion.tolist(),
    }


def undo_spatial_swap(_module: torch.nn.Module, args: tuple) -> tuple:
    """仅对进入 BevFeatureSlicer 的张量再转置一次，抵消前面的转置。"""
    if len(args) != 1 or not torch.is_tensor(args[0]):
        raise TypeError("BevFeatureSlicer 的输入结构不符合预期，已停止消融")
    bev = args[0]
    if bev.ndim != 4:
        raise ValueError(f"预期形状为 B,C,H,W，实际为 {tuple(bev.shape)}")
    return (bev.transpose(-2, -1),)


def reset_temporal_state(model: torch.nn.Module) -> None:
    """每次推理前清空历史 BEV，防止配对实验相互影响。"""
    if not hasattr(model, "prev_frame_info"):
        raise AttributeError("模型没有 prev_frame_info，无法验证时序状态隔离")
    model.prev_frame_info = {
        "prev_bev": None,
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }


def infer(
    model: torch.nn.Module,
    image: torch.Tensor,
    metadata: dict[str, Any],
    *,
    remove_swap: bool,
) -> tuple[np.ndarray, list[int]]:
    """固定输入，按需取消 BEV 转置，并返回模型原始预测类别。"""
    reset_temporal_state(model)
    handle = None
    if remove_swap:
        handle = model.pts_bbox_head.feat_cropper.register_forward_pre_hook(
            undo_spatial_swap
        )
    try:
        # forward_test 会原位更改 can_bus，因此必须给每次推理独立元数据副本。
        safe_metadata = copy.deepcopy(metadata)
        with torch.no_grad():
            prediction = model(
                return_loss=False,
                rescale=True,
                img=[image.clone()],
                img_metas=[[safe_metadata]],
            )
        if not isinstance(prediction, (list, tuple)) or not prediction:
            raise RuntimeError("模型推理输出不是非空 list/tuple")
        seg_logits = prediction[0].get("seg_preds")
        if seg_logits is None:
            raise RuntimeError("推理输出缺少 seg_preds")
        return segmentation_classes(seg_logits)
    finally:
        if handle is not None:
            handle.remove()


def render_comparison(
    gt: np.ndarray, original: np.ndarray, ablated: np.ndarray, path: Path
) -> None:
    """保存同一色阶的 GT、原始输出与轴消融输出，禁止调整预测。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    samples = (("Ground Truth", gt), ("Original", original), ("No axis swap", ablated))
    for axis, (title, image) in zip(axes, samples):
        axis.imshow(
            image,
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=3,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel("X / map column")
        axis.set_ylabel("Y / map row")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    """构建一次模型，配对比较三个样本的原始预测与真实轴消融预测。"""
    parser = argparse.ArgumentParser(description="BEV 分割空间轴转置真实推理消融")
    parser.add_argument("--indices", default="0,1,80", help="Mini 验证帧，例如 0,1,80")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    args = parser.parse_args()

    indices = [int(value.strip()) for value in args.indices.split(",")]
    if not indices or len(indices) != len(set(indices)) or any(i < 0 for i in indices):
        raise ValueError("indices 不能为空、重复或为负数")
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA，请激活原来的 .venv-bevformer")
    # 原始单帧脚本在进入 Seg 仓库后解析相对 checkpoint 路径。
    # 为了尽可能复现该入口，优先使用 Seg 仓库下的同名文件。
    if args.checkpoint.is_absolute():
        checkpoint_path = args.checkpoint.resolve()
    else:
        candidates = (SEG_REPO / args.checkpoint, ROOT / args.checkpoint)
        checkpoint_path = next((p.resolve() for p in candidates if p.is_file()), None)
    if not CONFIG.is_file():
        raise FileNotFoundError(f"分割配置文件不存在：{CONFIG}")
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(f"找不到 checkpoint：{args.checkpoint}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(SEG_REPO)
    sys.path.insert(0, str(SEG_REPO))

    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    importlib.import_module("projects.mmdet3d_plugin")
    cfg = Config.fromfile(str(CONFIG))
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.test_mode = True
    dataset_cfg.pop("samples_per_gpu", None)
    dataset_cfg.pop("workers_per_gpu", None)
    dataset = build_dataset(dataset_cfg)
    if max(indices) >= len(dataset):
        raise IndexError(f"样本索引超出 Mini 验证集范围，共 {len(dataset)} 帧")

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    matched = model.load_state_dict(state, strict=True)
    if matched.missing_keys or matched.unexpected_keys:
        raise RuntimeError("模型权重键不完全一致")
    del ckpt, state
    gc.collect()
    model = model.cuda().eval()
    if model.video_test_mode is not False:
        raise RuntimeError("历史 BEV 没有禁用")
    if not hasattr(model.pts_bbox_head, "feat_cropper"):
        raise AttributeError("找不到 feat_cropper，不能做安全消融")

    print("===== BEV 真实推理：轴转置配对消融 =====", flush=True)
    print(f"Mini 验证集：{len(dataset)} 帧；当前样本：{indices}", flush=True)
    print(f"checkpoint：{checkpoint_path}", flush=True)
    print("严格加载 checkpoint 成功；video_test_mode=False", flush=True)
    print("每帧对照：同权重、同输入、相同预处理，且强制清空 prev_bev", flush=True)

    rows = []
    total_original = np.zeros((4, 4), dtype=np.int64)
    total_ablated = np.zeros((4, 4), dtype=np.int64)
    for index in indices:
        item = dataset[index]
        image = unwrap_single_aug(item["img"])
        metadata = unwrap_single_aug(item["img_metas"])
        gt = np.squeeze(to_numpy(unwrap_single_aug(item["semantic_indices"])))
        if not torch.is_tensor(image) or not isinstance(metadata, dict):
            raise TypeError("数据集返回的图像或元数据类型不符合预期")
        if image.ndim == 4:
            image = image.unsqueeze(0)
        if tuple(image.shape[:2]) != (1, 6) or tuple(gt.shape) != (200, 400):
            raise ValueError(f"异常输入维度：image={image.shape}, GT={gt.shape}")
        gt = gt.astype(np.int64)
        image = image.to("cuda:0")

        baseline, baseline_shape = infer(
            model, image, metadata, remove_swap=False
        )
        ablated, ablated_shape = infer(
            model, image, metadata, remove_swap=True
        )
        if baseline.shape != gt.shape or ablated.shape != gt.shape:
            raise RuntimeError("预测与 GT 的图像形状不一致")

        original_metrics = compute_metrics(gt, baseline)
        ablation_metrics = compute_metrics(gt, ablated)
        total_original += np.asarray(original_metrics["confusion_matrix"])
        total_ablated += np.asarray(ablation_metrics["confusion_matrix"])

        cached_gt_path = MINI_OUT / f"seg_gt_gpu_sample{index}.npy"
        cached_pred_path = MINI_OUT / f"seg_pred_sample{index}.npy"
        gt_equal = (
            bool(np.array_equal(gt, np.load(cached_gt_path, allow_pickle=False)))
            if cached_gt_path.is_file()
            else None
        )
        baseline_equal = (
            bool(np.array_equal(baseline, np.load(cached_pred_path, allow_pickle=False)))
            if cached_pred_path.is_file()
            else None
        )
        fig = OUTPUT_DIR / f"step7_axis_ablation_sample_{index:03d}.png"
        render_comparison(gt, baseline, ablated, fig)

        row = {
            "index": index,
            "sample_token": str(dataset.data_infos[index].get("token", "")),
            "scene_token": str(dataset.data_infos[index].get("scene_token", "")),
            "baseline_logits_shape": baseline_shape,
            "ablated_logits_shape": ablated_shape,
            "gt_same_as_prior_saved_file": gt_equal,
            "baseline_same_as_prior_saved_file": baseline_equal,
            "pixel_labels_changed": int(np.count_nonzero(baseline != ablated)),
            "original": original_metrics,
            "axis_swap_disabled": ablation_metrics,
            "figure": str(fig),
        }
        rows.append(row)
        print(
            f"[样本 {index}] 原始：fg_mIoU={original_metrics['foreground_miou']:.4%} "
            f"binary={original_metrics['foreground_binary_iou']:.4%}；"
            f"取消转置：fg_mIoU={ablation_metrics['foreground_miou']:.4%} "
            f"binary={ablation_metrics['foreground_binary_iou']:.4%}",
            flush=True,
        )
        print(
            f"  与历史缓存一致：GT={gt_equal}，原始预测={baseline_equal}；"
            f"变化像素={row['pixel_labels_changed']}；图片={fig.name}",
            flush=True,
        )
        del item, image

    orig_classes, orig_fg, orig_all = stats_from_confusion(total_original)
    abla_classes, abla_fg, abla_all = stats_from_confusion(total_ablated)
    payload = {
        "experiment": "Mini paired test; disable feature spatial transpose with temporary hook",
        "important_limit": (
            "仅使用选择的 Mini 样本，是诊断性真实模型消融，不是 81/799 帧正式成绩；"
            "checkpoint 可能按原始转置约定训练，单纯取消转置不能直接证明源码缺陷。"
        ),
        "config_path": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "video_test_mode": False,
        "sample_indices": indices,
        "rows": rows,
        "aggregate_over_selected_samples": {
            "original": {
                "foreground_miou": orig_fg,
                "including_background_miou": orig_all,
                "classes": orig_classes,
            },
            "axis_swap_disabled": {
                "foreground_miou": abla_fg,
                "including_background_miou": abla_all,
                "classes": abla_classes,
            },
        },
    }
    path = OUTPUT_DIR / "step7_axis_ablation.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("===== 选中样本累计统计（非官方成绩） =====", flush=True)
    print(f"原始前景 mIoU：{orig_fg:.4%}；取消转置：{abla_fg:.4%}", flush=True)
    print(f"详细结果：{path}", flush=True)
    print("提醒：没有修改模型源码、checkpoint 或已有 81/799 帧结果。", flush=True)


if __name__ == "__main__":
    main()

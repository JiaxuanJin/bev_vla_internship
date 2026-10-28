"""冻结 BEVFormer 原生双 Head 的实际 CUDA 前向与损失接口。"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from scripts.stage2.bev_coordinates import BEVGrid


def load_native() -> tuple[Any, Any, nn.Module, Any]:
    """复用已验证的本地构建器，不加载 V1–V5 已训练适配器权重。"""
    from scripts.stage2.spatial_experiment import initialize
    from scripts.stage2.spatial_v3_model import InstanceReadout

    model, wrapped, pipeline, dataset = initialize("train")
    pipeline.requires_grad_(False).eval()
    encoder = InstanceReadout(pipeline.encoder).cuda().eval()
    encoder.requires_grad_(False)
    model.requires_grad_(False).eval()
    return model, wrapped, encoder, dataset


def dual_head(model: nn.Module, enhanced_bev: Tensor) -> dict[str, Tensor]:
    """同一增强 BEV 通过原作者检测解码器和分割解码器。

    缓存边界为真实冻结图像主干输出；形状占位相机特征不参与数值计算。
    此函数不读任何 GT，不代表执行了新的六相机主干前向。
    """
    if enhanced_bev.shape != (1, 256, 150, 150):
        raise ValueError("当前真实双 Head 接口只接受单帧原生 150×150 BEV")
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("作者模型必须冻结")
    transformer = model.pts_bbox_head.transformer
    original = transformer.get_bev_features
    sequence = BEVGrid().to_sequence(enhanced_bev)
    transformer.get_bev_features = lambda *_args, **_kwargs: sequence
    try:
        return model.pts_bbox_head([enhanced_bev.new_empty((1, 6, 256, 1, 1))], [{}])
    finally:
        transformer.get_bev_features = original


def native_losses(model: nn.Module, prediction: dict, ann: dict, gt: Tensor) -> dict:
    """恢复作者范围/类别过滤；分割调用原生带类别权重的 CE。"""
    from scripts.stage2.multitask_training import aggregate_detection_losses
    from scripts.stage2.spatial_v3_range_correction import filter_ann

    ann, stats = filter_ann(ann, model.pts_bbox_head.pc_range)
    device = prediction["seg_preds"].device
    boxes = ann["gt_bboxes_3d"].to(device)
    labels = torch.as_tensor(ann["gt_labels_3d"], device=device, dtype=torch.long)
    losses = model.pts_bbox_head.loss([boxes], [labels], prediction)
    detection = aggregate_detection_losses(
        {k: v for k, v in losses.items() if "seg" not in k}
    )
    segmentation = model.pts_bbox_head.seg_loss(gt[None].to(device), prediction)[
        "loss_seg"
    ]
    return dict(detection=detection, segmentation=segmentation, filter_stats=stats)


def module_sha(module: nn.Module) -> str:
    """逐张量哈希含 buffer 的完整状态，不保留主干额外副本。"""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_audit(loss: Tensor, module: nn.Module) -> dict:
    """逐参数报告 None/精确零/非零及有限性，保留任务计算图。"""
    parameters = list(module.named_parameters())
    gradients = torch.autograd.grad(
        loss, [p for _, p in parameters], retain_graph=True, allow_unused=True
    )
    result = {}
    for (name, _), gradient in zip(parameters, gradients):
        value = float(gradient.detach().norm()) if gradient is not None else 0.0
        result[name] = dict(
            l2=value,
            connected=gradient is not None,
            finite=gradient is None or bool(torch.isfinite(gradient).all()),
        )
    return dict(
        parameters=result,
        total_l2=float(np.sqrt(sum(v["l2"] ** 2 for v in result.values()))),
        all_finite=all(v["finite"] for v in result.values()),
    )


def decode(model: nn.Module, prediction: dict) -> dict[str, np.ndarray]:
    """读取原生解码器全部 top300，不在开发集选择 score 阈值。"""
    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    boxes, scores, labels = model.pts_bbox_head.get_bboxes(
        prediction, [dict(box_type_3d=LiDARInstance3DBoxes)], rescale=False
    )[0]
    return dict(
        boxes=boxes.tensor.detach().cpu().numpy(),
        scores=scores.detach().cpu().numpy(),
        labels=labels.detach().cpu().numpy(),
    )

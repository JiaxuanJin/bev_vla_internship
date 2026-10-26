"""指令指定实例的空间监督：依赖显式标注，不回退到全类别掩码。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class SpatialTarget:
    """一条指令在分割 GT 像素网格上的目标实例与评价区域。

    Args:
        instruction_id: 稳定的指令唯一标识，可用于严格 A/B 配对。
        target_class: 受支持地图类别的整数 ID（1/2/3）。
        instance_id: 经标注确认的实例/目标标识；不能使用类别名称替代。
        instance_mask: [H,W] 的 bool 真值，且必须属于 target_class。
        related_region: [H,W] 的 bool 评价区域，包含 instance_mask。
            该区域必须由指令/独立标注确定，而非预测结果或 GT 全类别反推。
        annotation_source: 非空来源，例如 manual_verified/dataset_instance。
        coordinate_frame: 固定表示两种掩码均属于原始分割 GT 像素网格。

    注意：标注字段是证据来源声明，不能取代人工/数据集配准核验。
    """

    instruction_id: str
    target_class: int
    instance_id: str
    instance_mask: Tensor
    related_region: Tensor
    annotation_source: str
    coordinate_frame: str = "segmentation_gt_pixel_grid"


def validate_spatial_target(
    target: SpatialTarget, gt: Tensor, *, ignore_index: int = 255
) -> Tensor:
    """校验单个实例标签的坐标、类别、ROI 与 GT 一致性。"""
    if not isinstance(target, SpatialTarget):
        raise TypeError("必须显式提供 SpatialTarget，而不是类别 ID")
    if target.target_class not in (1, 2, 3):
        raise ValueError("只接受类别 1/2/3")
    for field in ("instruction_id", "instance_id", "annotation_source"):
        value = getattr(target, field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} 必须包含可追溯的非空标识")
    if target.annotation_source.lower() in {"class_mask", "auto_class", "unknown"}:
        raise ValueError("禁止将类别掩码冒充目标实例标注")
    if target.coordinate_frame != "segmentation_gt_pixel_grid":
        raise ValueError("空间目标必须已配准到原始分割 GT 像素网格")
    if gt.ndim != 2 or gt.dtype != torch.long:
        raise ValueError("单帧 GT 必须是 [H,W] 的 long 类别索引")
    if not bool(((gt == ignore_index) | ((gt >= 0) & (gt <= 3))).all()):
        raise ValueError("GT 含无效类别")
    for name, mask in (
        ("instance_mask", target.instance_mask),
        ("related_region", target.related_region),
    ):
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
            raise TypeError(f"{name} 必须为 bool Tensor，不允许隐式阈值化")
        if mask.shape != gt.shape or mask.device != gt.device:
            raise ValueError(f"{name} 必须与 GT 完全同尺寸、同设备；禁止 resize")
    instance = target.instance_mask
    region = target.related_region
    valid = gt != ignore_index
    if not bool(instance.any()):
        raise ValueError("该指令目标实例不存在，不能伪造注意力正样本")
    if not bool((instance <= (gt == target.target_class)).all()):
        raise ValueError("目标实例掩码包含其他类别或 ignore 区域")
    if not bool((instance <= region).all()):
        raise ValueError("评价 ROI 必须覆盖目标实例")
    if not bool((region & valid).any() and ((~region) & valid).any()):
        raise ValueError("相关/无关评价区域均必须有有效像素")
    return valid


def spatial_attention_loss(
    attention_logits: Tensor,
    segmentation_gt: Tensor,
    spatial_targets: list[SpatialTarget] | tuple[SpatialTarget, ...],
    *,
    ignore_index: int = 255,
    dice_weight: float = 0.25,
    max_pos_weight: float = 20.0,
) -> Tensor:
    """仅学习每条指令指定的目标实例，其他同类实例也作为负例。

    输入是 [B,1,H,W] logits，GT 为 [B,H,W]；两者必须为相同网格。
    使用全图有效像素 BCE + soft Dice；ROI 只供分项评价，绝不泄漏到推理。
    """
    if attention_logits.ndim != 4 or attention_logits.shape[1] != 1:
        raise ValueError("attention_logits 必须为 [B,1,H,W]")
    if tuple(segmentation_gt.shape) != (
        attention_logits.shape[0], *attention_logits.shape[2:]
    ):
        raise ValueError("注意力和 GT 未对齐，禁止自动 resize")
    if len(spatial_targets) != attention_logits.shape[0]:
        raise ValueError("每帧必须对应一份唯一的空间实例标注")
    if not attention_logits.is_floating_point() or not bool(
        torch.isfinite(attention_logits).all()
    ):
        raise ValueError("注意力 logits 必须为有限浮点数")
    if dice_weight < 0 or not 1 <= max_pos_weight <= 1000:
        raise ValueError("损失超参数不在允许范围")

    losses: list[Tensor] = []
    for i, target in enumerate(spatial_targets):
        valid = validate_spatial_target(target, segmentation_gt[i], ignore_index=ignore_index)
        truth = target.instance_mask.float()
        n_pos = truth.sum()
        n_neg = valid.sum().to(truth.dtype) - n_pos
        positive_weight = (n_neg / n_pos).clamp(min=1.0, max=max_pos_weight)
        logits = attention_logits[i, 0]
        per_pixel = F.binary_cross_entropy_with_logits(
            logits, truth, pos_weight=positive_weight, reduction="none"
        )
        bce = (per_pixel * valid).sum() / valid.sum().clamp_min(1)
        prob = logits.sigmoid() * valid
        intersection = (prob * truth).sum()
        dice = 1 - (2 * intersection + 1) / (prob.sum() + n_pos + 1)
        losses.append(bce + dice_weight * dice)
    return torch.stack(losses).mean()

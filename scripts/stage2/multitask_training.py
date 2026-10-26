"""第二阶段统一多任务损失实现。

支持真实检测头 loss 字典、四类地图分割 logits 及语言类别注意力。
注意：这是可训练损失接口，不会自行启动检测头，也不等于联合训练已完成。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class LossConfig:
    """多任务损失超参数。

    Args:
        detection_weight: 检测损失权重。
        segmentation_weight: 分割损失权重。
        attention_weight: GT 类别注意力监督权重（BCE + Dice）。
        ignore_index: 分割 GT 中需要忽略的标签值。
        attention_dice_weight: 注意力 BCE 之外的 Soft Dice 系数。
        attention_tv_weight: 无需 GT 的概率掩码空间 TV 正则权重；0 保持历史公式。
    """

    detection_weight: float = 1.0
    segmentation_weight: float = 1.0
    attention_weight: float = 0.1
    ignore_index: int = 255
    attention_dice_weight: float = 0.25
    attention_tv_weight: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.detection_weight,
            self.segmentation_weight,
            self.attention_weight,
            self.attention_dice_weight,
            self.attention_tv_weight,
        )
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("所有损失权重都必须是非负的有限数")
        if self.detection_weight + self.segmentation_weight <= 0:
            raise ValueError("检测损失与分割损失不能同时禁用")


def _loss_key(name: str) -> bool:
    """兼容 MMDetection `loss_cls` 与 `d0.loss_cls` 等命名。"""
    return name.startswith("loss") or ".loss" in name


def _scalar_loss(value: Any, key: str) -> Tensor:
    """允许 decoder 多层 loss 以 list / tuple 形式返回。"""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{key} 为空列表")
        parts = [_scalar_loss(item, key) for item in value]
        return torch.stack(parts).sum()
    if not isinstance(value, Tensor):
        raise TypeError(f"{key} 必须由 Tensor 构成，收到 {type(value).__name__}")
    if value.numel() != 1:
        raise ValueError(f"{key} 不是标量损失，形状={tuple(value.shape)}")
    value = value.reshape(())
    if not bool(torch.isfinite(value.detach()).item()):
        raise ValueError(f"{key} 出现 NaN 或 Inf")
    return value


def aggregate_detection_losses(loss_dict: Mapping[str, Any]) -> Tensor:
    """按 mmdet loss 命名聚合，不把 `acc` 等日志指标作为损失。"""
    if not isinstance(loss_dict, Mapping):
        raise TypeError("检测头必须返回字典形式的 loss")
    selected = [(k, v) for k, v in loss_dict.items() if _loss_key(k)]
    if not selected:
        raise ValueError("找不到真实检测损失键，禁止伪造 detection loss")
    total = None
    for key, value in selected:
        part = _scalar_loss(value, key)
        total = part if total is None else total + part
    assert total is not None
    return total


def _validate_gt(
    seg_logits: Tensor, segmentation_gt: Tensor, cfg: LossConfig
) -> Tensor:
    if seg_logits.ndim != 4 or seg_logits.shape[1] != 4:
        raise ValueError("分割 logits 必须为 [B,4,H,W]")
    shape = (seg_logits.shape[0], seg_logits.shape[2], seg_logits.shape[3])
    if tuple(segmentation_gt.shape) != shape:
        raise ValueError(f"GT 形状应为 {shape}，实际 {tuple(segmentation_gt.shape)}")
    if segmentation_gt.dtype not in (torch.int64, torch.long):
        raise TypeError("分割 GT 必须为 long 类型的类别索引")
    if segmentation_gt.device != seg_logits.device:
        raise ValueError("分割 GT 与分割 logits 必须在同一个设备")
    valid = segmentation_gt != cfg.ignore_index
    if not valid.any():
        raise ValueError("该批次没有有效分割 GT")
    labels = segmentation_gt[valid]
    if labels.min().item() < 0 or labels.max().item() >= 4:
        raise ValueError("分割标签仅支持背景 0 与前景类别 1～3")
    if not torch.isfinite(seg_logits.detach()).all():
        raise ValueError("分割 logits 存在 NaN 或 Inf")
    return valid


def supervised_attention_loss(
    attention_logits: Tensor,
    segmentation_gt: Tensor,
    target_classes: Tensor,
    cfg: LossConfig,
    *,
    spatial_targets: Sequence[Any] | None = None,
) -> Tensor:
    """同空间下的类别条件注意力损失。

    注意力输入是未经 sigmoid 的 logits。[B,1,H,W]；训练 GT 不进入推理。
    注意力图与分割 GT 不同尺寸时严格拒绝，避免默默缩放掩盖空间协议问题。
    """
    if spatial_targets is not None:
        from .spatial_supervision import spatial_attention_loss

        if target_classes.shape != (segmentation_gt.shape[0],):
            raise ValueError("target_classes 必须逐帧指定")
        if target_classes.dtype != torch.long or (
            target_classes.device != attention_logits.device
        ):
            raise ValueError("指令类别 dtype/设备错误")
        if len(spatial_targets) != segmentation_gt.shape[0]:
            raise ValueError("实例标注数量必须等于 batch")
        if any(
            item.target_class != int(target_classes[i].item())
            for i, item in enumerate(spatial_targets)
        ):
            raise ValueError("显式实例标签与指令类别不一致")
        return spatial_attention_loss(
            attention_logits, segmentation_gt, spatial_targets,
            ignore_index=cfg.ignore_index,
            dice_weight=cfg.attention_dice_weight,
        )
    if attention_logits.ndim != 4 or attention_logits.shape[1] != 1:
        raise ValueError("注意力 logits 必须是 [B,1,H,W]")
    if tuple(attention_logits.shape[0:1] + attention_logits.shape[2:4]) != tuple(
        segmentation_gt.shape
    ):
        raise ValueError("注意力空间尺寸与 GT 不匹配；请显式进行坐标配准，禁止直接插值")
    if attention_logits.device != segmentation_gt.device:
        raise ValueError("注意力 logits / GT 设备不一致")
    if segmentation_gt.dtype != torch.long:
        raise TypeError("注意力监督 GT 必须为 long")
    valid = segmentation_gt != cfg.ignore_index
    if ((segmentation_gt[valid] < 0) | (segmentation_gt[valid] > 3)).any():
        raise ValueError("注意力监督 GT 类别必须为 0～3")
    if target_classes.shape != (segmentation_gt.shape[0],):
        raise ValueError("target_classes 必须是 [B]，逐帧指定类别")
    if target_classes.dtype != torch.long:
        raise TypeError("target_classes 必须为 torch.long")
    if target_classes.device != attention_logits.device:
        raise ValueError("target_classes 与注意力设备不一致")
    if (target_classes < 1).any() or (target_classes > 3).any():
        raise ValueError("注意力目标类别必须属于 1、2、3")
    if not torch.isfinite(attention_logits.detach()).all():
        raise ValueError("注意力出现 NaN 或 Inf")
    valid = segmentation_gt != cfg.ignore_index
    truth = (segmentation_gt == target_classes[:, None, None]).float()
    positive_present = (truth * valid.float()).flatten(1).sum(dim=1) > 0
    if not positive_present.any():
        # 批内目标全部缺席：不臆造正样本注意力监督。
        return attention_logits.sum() * 0.0

    selected = positive_present
    mask = valid[selected].unsqueeze(1)
    truth = truth[selected].unsqueeze(1)
    logits = attention_logits[selected]
    bce = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    bce = (bce * mask).sum() / mask.sum().clamp_min(1)

    prob = logits.sigmoid() * mask
    intersection = (prob * truth).sum(dim=(1, 2, 3))
    total = prob.sum(dim=(1, 2, 3)) + (truth * mask).sum(dim=(1, 2, 3))
    dice = 1 - (2 * intersection + 1) / (total + 1)
    return bce + cfg.attention_dice_weight * dice.mean()


def attention_regularization_loss(attention_logits: Tensor) -> Tensor:
    """GT 无关的空间平滑先验：sigmoid 概率相邻差的各轴均值之和。

    不约束注意力面积或正负比例。单像素/单行/单列跳过不存在的边，
    保留可反传的零值；float32 计算避免半精度归约不稳定。
    """
    if (
        attention_logits.ndim != 4
        or attention_logits.shape[1] != 1
        or min(attention_logits.shape) <= 0
    ):
        raise ValueError("TV 输入必须为非空 [B,1,H,W] logits")
    if not attention_logits.is_floating_point():
        raise TypeError("TV 输入必须为浮点 logits")
    if not torch.isfinite(attention_logits).all():
        raise ValueError("TV 输入包含 NaN/Inf")
    probability = attention_logits.float().sigmoid()
    value = probability.sum() * 0.0
    if probability.shape[-2] > 1:
        value = value + probability.diff(dim=-2).abs().mean()
    if probability.shape[-1] > 1:
        value = value + probability.diff(dim=-1).abs().mean()
    return value


class Stage2MultiTaskLoss(nn.Module):
    """接收检测头真实 loss、分割 logits、真实地图 GT、类别注意力。"""

    def __init__(self, config: LossConfig | None = None) -> None:
        super().__init__()
        self.config = config or LossConfig()

    def forward(
        self,
        detection_losses: Mapping[str, Any],
        segmentation_logits: Tensor,
        segmentation_gt: Tensor,
        attention_logits: Tensor,
        target_classes: Tensor,
        *,
        spatial_targets: Sequence[Any] | None = None,
        regularization_logits: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """返回保留梯度的总损失和各项 detached 诊断指标。"""
        valid = _validate_gt(segmentation_logits, segmentation_gt, self.config)
        det = aggregate_detection_losses(detection_losses)
        if det.device != segmentation_logits.device:
            raise ValueError("检测损失与分割分支不在同一设备")
        if valid.sum().item() <= 0:
            raise ValueError("有效分割像素数为零")
        seg = F.cross_entropy(
            segmentation_logits,
            segmentation_gt,
            ignore_index=self.config.ignore_index,
        )
        att = supervised_attention_loss(
            attention_logits, segmentation_gt, target_classes, self.config,
            spatial_targets=spatial_targets,
        )
        # 零权重不额外计算正则，历史实验的 total 保持逐位相同。
        regularization = att.new_zeros(())
        if self.config.attention_tv_weight:
            if regularization_logits is not None and (
                regularization_logits.device != attention_logits.device
                or regularization_logits.shape[0] != attention_logits.shape[0]
            ):
                raise ValueError("正则 logits 与监督 logits 的 batch/设备不一致")
            regularization = attention_regularization_loss(
                attention_logits
                if regularization_logits is None
                else regularization_logits
            )
        total = (
            det * self.config.detection_weight
            + seg * self.config.segmentation_weight
            + att * self.config.attention_weight
        )
        if self.config.attention_tv_weight:
            total = total + self.config.attention_tv_weight * regularization
        if not bool(torch.isfinite(total.detach()).item()):
            raise ValueError("总损失包含 NaN/Inf")
        parts = {
            "detection_loss": det.detach(),
            "segmentation_loss": seg.detach(),
            "attention_loss": att.detach(),
            "attention_supervision_loss": att.detach(),
            "attention_regularization_loss": regularization.detach(),
            "total_loss": total.detach(),
        }
        return total, parts


__all__ = [
    "LossConfig",
    "Stage2MultiTaskLoss",
    "aggregate_detection_losses",
    "supervised_attention_loss",
    "attention_regularization_loss",
]

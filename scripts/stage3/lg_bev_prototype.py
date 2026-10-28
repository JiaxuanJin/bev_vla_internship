"""LG-BEV：语言条件化 BEV 空间注意力最小原型。

本文件仅实现新研究分支的独立模块、掩码监督与合成单元测试。
不会读取、修改 V1–V5 checkpoint，也不会自动访问任何验证集。

输入协议（参考 V5）：
    bev: [B, 256, H, W]，冻结 BEVFormer 的原生 BEV 特征。
    text: [B, 3072]，预先计算的冻结语言特征。
    position: [B, 5, H, W]，仅由标定生成的米制空间特征。

真实数据、原生双 Head 接入与逐任务梯度审计分别在同目录的
lg_bev_data.py、lg_bev_native.py、lg_bev_train.py；本文件自检只运行合成数据。
"""

from __future__ import annotations

import argparse
import math
from typing import TypedDict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AdapterOutput(TypedDict):
    """语言引导模块的输出张量。"""

    enhanced_bev: Tensor
    attention_logits: Tensor
    attention_mask: Tensor
    residual: Tensor


class LanguageGroundedBEVAdapter(nn.Module):
    """生成语言相关的 BEV 注意力，使用受限残差增强原生 BEV。

    Args:
        bev_channels: 原生 BEV 特征通道数。
        text_dim: 冻结语言特征的维度（V5 为 3072）。
        position_channels: 标定生成的空间输入通道数。
        hidden_channels: 新增模块的内部通道数，必须是 4 的倍数。
        relative_bound: 残差相对每通道原始 BEV RMS 的幅度上界。
    """

    def __init__(
        self,
        bev_channels: int = 256,
        text_dim: int = 3072,
        position_channels: int = 5,
        hidden_channels: int = 24,
        relative_bound: float = 0.05,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0 or hidden_channels % 4 != 0:
            raise ValueError("hidden_channels 必须是正数且能被 4 整除")
        if not 0.0 < relative_bound <= 1.0:
            raise ValueError("relative_bound 必须位于 (0, 1] 区间")

        self.relative_bound = relative_bound
        self.visual = nn.Sequential(
            nn.Conv2d(bev_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(4, hidden_channels),
            nn.GELU(),
        )
        self.position = nn.Conv2d(position_channels, hidden_channels, 1)
        self.language = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.key = nn.Conv2d(hidden_channels, hidden_channels, 1)
        self.mask_refine = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.up = nn.Conv2d(hidden_channels, bev_channels, kernel_size=1)

        # 训练启动时完全不扰动原始检测和分割特征。
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        # 初始掩码不能默认全部开启，后续用真实目标掩码训练。
        nn.init.zeros_(self.mask_refine.bias)

    def forward(self, bev: Tensor, text: Tensor, position: Tensor) -> AdapterOutput:
        """执行轻量级的语言查询-空间键匹配和安全残差融合。

        Args:
            bev: [B, C, H, W]，原始 BEV 特征。
            text: [B, text_dim]，冻结的预计算语言特征。
            position: [B, P, H, W]，标定几何；不得混入 GT。

        Returns:
            包含增强 BEV、注意力 logits、掩码及真实残差的字典。

        Raises:
            ValueError: 输入形状或 batch / 空间尺寸不匹配。
        """
        if bev.ndim != 4 or text.ndim != 2 or position.ndim != 4:
            raise ValueError("bev / text / position 维度应分别为 4 / 2 / 4")
        if bev.shape[0] != text.shape[0] or bev.shape[0] != position.shape[0]:
            raise ValueError("batch 大小必须一致")
        if bev.shape[-2:] != position.shape[-2:]:
            raise ValueError("BEV 与标定几何的空间尺寸必须一致")

        # 语言 query 与逐像素视觉 key 内积；一条指令生成一幅 BEV 掩码。
        hidden = self.visual(bev) + self.position(position.to(bev))
        query = self.language(text.to(bev))[:, :, None, None]
        key = self.key(hidden)
        score = (key * query).sum(dim=1, keepdim=True)
        score = score / math.sqrt(key.shape[1])
        logits = score + self.mask_refine(score)
        attention = logits.sigmoid()

        # 注意力在空间维度逐位置约束残差；不向主干输入 GT。
        rms = bev.detach().square().mean((2, 3), keepdim=True)
        rms = rms.sqrt().clamp_min(1e-6)
        residual = self.relative_bound * rms * self.up(hidden).tanh() * attention
        return {
            "enhanced_bev": bev + residual,
            "attention_logits": logits,
            "attention_mask": attention,
            "residual": residual,
        }


def attention_supervision_loss(
    logits: Tensor,
    target_mask: Tensor,
    distractor_mask: Tensor | None = None,
    distractor_weight: float = 0.5,
    valid_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """计算平衡 BCE、Dice 和可选的同类干扰目标负样本损失。

    支持无目标指令：该样本只计负样本 BCE，不计算空目标 Dice。
    真实项目应从 **训练集 GT** 构建目标和干扰掩码，禁止使用验证集 GT。

    Args:
        logits: [B, 1, H, W]，未经过 sigmoid 的注意力输出。
        target_mask: 与 logits 同形状的 0/1 目标掩码。
        distractor_mask: 可选，同形状的 0/1 干扰实例掩码。
        distractor_weight: 附加干扰负样本损失系数。
        valid_mask: 可选的有效监督区域；无效像素不进入 BCE/Dice/干扰损失。

    Returns:
        带梯度的 total、bce、dice、distractor 损失字典。

    Raises:
        ValueError: 掩码形状、取值或正负目标互斥性不符合要求。
    """
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits 必须为 [B, 1, H, W]")
    if target_mask.shape != logits.shape:
        raise ValueError("target_mask 形状必须与 logits 完全一致")
    if not bool(((target_mask == 0) | (target_mask == 1)).all()):
        raise ValueError("target_mask 只能包含 0/1")
    if distractor_mask is not None:
        if distractor_mask.shape != logits.shape:
            raise ValueError("distractor_mask 形状不匹配")
        if not bool(((distractor_mask == 0) | (distractor_mask == 1)).all()):
            raise ValueError("distractor_mask 只能包含 0/1")
        if bool(((target_mask == 1) & (distractor_mask == 1)).any()):
            raise ValueError("目标掩码与干扰实例掩码不能重叠")
    if distractor_weight < 0:
        raise ValueError("distractor_weight 不允许小于 0")

    if valid_mask is None:
        valid_mask = torch.ones_like(target_mask, dtype=torch.bool)
    if valid_mask.shape != logits.shape or not bool(
        ((valid_mask == 0) | (valid_mask == 1)).all()
    ):
        raise ValueError("valid_mask 必须与 logits 同形状且为 0/1")
    valid = valid_mask.to(device=logits.device, dtype=torch.bool)
    gt = target_mask.to(logits) * valid
    element_bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    is_positive = (gt > 0) & valid
    is_negative = (gt == 0) & valid
    # 每个样本先做正负均衡，避免背景像素数量主导损失。
    reduce_dims = (1, 2, 3)
    n_pos = is_positive.sum(dim=reduce_dims).clamp_min(1)
    n_neg = is_negative.sum(dim=reduce_dims).clamp_min(1)
    pos_loss = (element_bce * is_positive).sum(dim=reduce_dims) / n_pos
    neg_loss = (element_bce * is_negative).sum(dim=reduce_dims) / n_neg
    positive_sample = is_positive.flatten(1).any(dim=1)
    has_valid = valid.flatten(1).any(dim=1)
    per_sample_bce = (pos_loss + neg_loss) / (
        positive_sample.float() + is_negative.flatten(1).any(1).float()
    ).clamp_min(1)
    # 非空/空目标分别均值再平均，避免大量空指令压倒非空监督。
    strata = [
        per_sample_bce[group].mean()
        for group in (positive_sample, ~positive_sample & has_valid)
        if bool(group.any())
    ]
    balanced_bce = torch.stack(strata).mean() if strata else logits.sum() * 0

    probs = logits.sigmoid() * valid
    intersection = (probs * gt).sum(dim=reduce_dims)
    total_mass = probs.sum(dim=reduce_dims) + gt.sum(dim=reduce_dims)
    per_sample_dice = 1 - (2 * intersection + 1) / (total_mass + 1)
    if bool(positive_sample.any()):
        dice = per_sample_dice[positive_sample].mean()
    else:
        dice = logits.sum() * 0

    distractor = logits.sum() * 0
    if distractor_mask is not None:
        is_distractor = (
            distractor_mask.to(device=logits.device, dtype=torch.bool) & valid
        )
        n_distractor = is_distractor.sum(dim=reduce_dims).clamp_min(1)
        negative_bce = F.softplus(logits)
        per_sample = (negative_bce * is_distractor).sum(dim=reduce_dims)
        has_distractor = is_distractor.flatten(1).any(dim=1)
        if bool(has_distractor.any()):
            distractor = (per_sample / n_distractor)[has_distractor].mean()

    total = balanced_bce + dice + distractor_weight * distractor
    return {
        "total": total,
        "bce": balanced_bce,
        "dice": dice,
        "distractor": distractor,
    }


def self_test() -> None:
    """不读取任何项目数据，验证形状、恒等初始化和反向梯度。"""
    torch.manual_seed(20261027)
    module = LanguageGroundedBEVAdapter()
    bev = torch.randn(2, 256, 12, 12)
    text = torch.randn(2, 3072)
    position = torch.randn(2, 5, 12, 12)
    outputs = module(bev, text, position)

    assert outputs["enhanced_bev"].shape == bev.shape
    assert outputs["attention_mask"].shape == (2, 1, 12, 12)
    assert torch.equal(outputs["enhanced_bev"], bev)
    assert torch.all(outputs["attention_mask"] >= 0)
    assert torch.all(outputs["attention_mask"] <= 1)
    print("[PASS] 输入输出形状、注意力范围、零残差初始化")

    # 真正改变指令输入；检查掩码输出确实与语言相关。
    swapped = module(bev, text.flip(0), position)
    language_delta = (
        (outputs["attention_logits"] - swapped["attention_logits"]).abs().max().item()
    )
    assert language_delta > 1e-8, "注意力没有响应语言变化"
    print(f"[PASS] 语言条件响应，max_logit_delta={language_delta:.6f}")

    target_mask = torch.zeros(2, 1, 12, 12)
    target_mask[0, 0, 2:5, 3:6] = 1
    distractor_mask = torch.zeros_like(target_mask)
    distractor_mask[0, 0, 7:9, 7:9] = 1
    # 第二条指令无目标；用于测试空指令分支不会产生 NaN。
    losses = attention_supervision_loss(
        outputs["attention_logits"], target_mask, distractor_mask
    )
    objective = losses["total"] + outputs["enhanced_bev"].square().mean()
    assert torch.isfinite(objective)
    objective.backward()
    assert module.up.weight.grad is not None
    assert bool(torch.isfinite(module.up.weight.grad).all())
    assert float(module.up.weight.grad.abs().sum()) > 0
    first_linear = module.language[1]
    assert first_linear.weight.grad is not None
    assert float(first_linear.weight.grad.abs().sum()) > 0
    print("[PASS] 含空目标的 BCE/Dice/干扰监督，梯度可达语言分支和残差头")

    empty = attention_supervision_loss(torch.randn(2, 1, 8, 8), torch.zeros(2, 1, 8, 8))
    assert torch.isfinite(empty["total"])
    assert float(empty["dice"]) == 0
    print("[PASS] 全空目标情况下损失数值有效")
    print("[INFO] 合成测试通过；本自检未执行真实双 Head 或精度评估")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="运行合成单元测试")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()

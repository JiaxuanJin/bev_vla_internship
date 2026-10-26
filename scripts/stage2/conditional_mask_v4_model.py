"""独立语言条件目标 Mask Head；推理接口不接受 GT、候选实例或实例距离。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LanguageConditionedTargetMaskHead(nn.Module):
    """用真实 BEV、米制坐标和冻结文本向量预测目标实例。

    Args:
        channels: 中间通道数，所有语言消融完全相同。
    """

    def __init__(self, channels: int = 24) -> None:
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(256, channels, 1), nn.GroupNorm(4, channels), nn.GELU()
        )
        self.position = nn.Conv2d(5, channels, 1)
        self.text = nn.Sequential(
            nn.LayerNorm(3072),
            nn.Linear(3072, 96),
            nn.GELU(),
            nn.Linear(96, channels * 2),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(
        self, bev: Tensor, text: Tensor, sampling_grid: Tensor, ego_features: Tensor
    ) -> Tensor:
        """输出 200×400 实例 logits，几何输入只依赖固定标定。

        Args:
            bev: 原作者冻结的 B×256×150×150 LiDAR BEV。
            text: 冻结 DistilBERT 四路读出 B×3072。
            sampling_grid: Ego 地面像素经标定映射至 LiDAR 归一化 XY。
            ego_features: 归一化 x、y、半径、xy、常数，B×5×200×400。

        Returns:
            B×1×200×400 的条件实例 logits。
        """
        native = self.visual(bev)
        features = F.grid_sample(
            native,
            sampling_grid.to(native),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        features = features + self.position(ego_features.to(features))
        scale, bias = self.text(text).chunk(2, dim=1)
        conditioned = features * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
        return self.decoder(conditioned)


def mask_loss(
    logits: Tensor, target: Tensor, same_class_other: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    """分组 BCE、Dice 和真实难负例；共享目标像素不会同时充当负例。

    Args:
        logits: 目标输出。
        target: 真实实例正例，空目标拒绝。
        same_class_other: 同类非目标像素，来自训练标签。

    Returns:
        标量损失与各分量；GT 只进入本函数。
    """
    value = logits.reshape(target.shape)
    if not target.any() or target.all():
        raise ValueError("NOT_EVALUABLE: empty or invalid target")
    negative = ~target
    other = same_class_other & negative
    positive = F.softplus(-value[target]).mean()
    background = F.softplus(value[negative]).mean()
    distractor = F.softplus(value[other]).mean() if other.any() else background
    probability = value.sigmoid()
    dice = 1 - (2 * probability[target].sum() + 1) / (
        probability.sum() + target.sum() + 1
    )
    negatives = F.softplus(value[negative])
    k = min(negatives.numel(), max(32, 4 * int(target.sum())))
    hard = negatives.topk(k).values.mean()
    parts = dict(
        positive_bce=positive,
        negative_bce=background,
        distractor_bce=distractor,
        dice=dice,
        hard_negative=hard,
    )
    return (
        0.5 * positive
        + 0.25 * background
        + 0.25 * distractor
        + 0.25 * dice
        + 0.1 * hard,
        parts,
    )


class ZeroChannelResidual(nn.Module):
    """仅作因果梯度诊断的零初始化通道残差；不修改作者 Head。"""

    def __init__(self) -> None:
        super().__init__()
        self.down = nn.Conv2d(256, 16, 1)
        self.gate = nn.Linear(3072, 16)
        self.up = nn.Conv2d(16, 256, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, bev: Tensor, text: Tensor, attention: Tensor) -> Tensor:
        """同一固定 alpha=0.0129582463，避免放大幅度制造收益。"""
        residual = self.up(
            F.gelu(self.down(bev)) * self.gate(text).sigmoid()[:, :, None, None]
        )
        return bev + 0.0129582463 * attention * residual

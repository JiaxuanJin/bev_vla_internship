"""V5：语言条件通道残差进入真实原生双 Head，不新增实例 Head。"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LanguageConditionedBEVResidualAdapter(nn.Module):
    """零初始化、幅度受限的原生 BEV 残差。

    Args:
        channels: 低秩通道数，所有等预算分支完全一致。
        relative_bound: 每通道残差绝对值相对输入 RMS 的固定上界。
    """

    def __init__(self, channels: int = 16, relative_bound: float = 0.05) -> None:
        super().__init__()
        self.relative_bound = relative_bound
        self.visual = nn.Sequential(
            nn.Conv2d(256, channels, 1), nn.GroupNorm(4, channels), nn.GELU()
        )
        self.position = nn.Conv2d(5, channels, 1)
        self.language = nn.Sequential(
            nn.LayerNorm(3072),
            nn.Linear(3072, 64),
            nn.GELU(),
            nn.Linear(64, 2 * channels),
        )
        self.up = nn.Conv2d(channels, 256, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, bev: Tensor, text: Tensor, position: Tensor) -> dict[str, Tensor]:
        """输出进入真实 Detection/Segmentation Heads 的增强 BEV。

        Args:
            bev: 原作者冻结 BEV，B×256×150×150。
            text: 冻结 DistilBERT 向量，B×3072；无语言对照传全零。
            position: 仅由标定生成的原生5通道米制几何，不接受GT。

        Returns:
            增强BEV及实际残差；同一输出同时进入两个原生Head。
        """
        hidden = self.visual(bev) + self.position(position.to(bev))
        scale, bias = self.language(text).chunk(2, dim=1)
        conditioned = hidden * (1 + scale.tanh()[:, :, None, None])
        conditioned = F.gelu(conditioned + bias[:, :, None, None])
        rms = bev.detach().square().mean((2, 3), keepdim=True).sqrt().clamp_min(1e-6)
        residual = self.relative_bound * rms * self.up(conditioned).tanh()
        return dict(enhanced_bev=bev + residual, residual=residual)

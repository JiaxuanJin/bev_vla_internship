"""将语言的全局语义向量映射到 BEV 特征的通道空间。"""

import torch
from torch import nn


class SemanticProjection(nn.Module):
    """将任意文本隐藏维度映射到指定 BEV 通道数。

    Args:
        text_dim: 文本隐藏层的特征维度。
        bev_channels: BEV 特征通道数量。
    """

    def __init__(self, text_dim: int, bev_channels: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, bev_channels),
            nn.GELU(),
            nn.Linear(bev_channels, bev_channels),
        )

    def forward(self, pooled_text: torch.Tensor) -> torch.Tensor:
        """接收 (B, D_text)，返回 (B, C_bev)。"""
        if pooled_text.ndim != 2:
            raise ValueError("语言向量必须为 (B, D_text) 二维张量")
        return self.net(pooled_text)

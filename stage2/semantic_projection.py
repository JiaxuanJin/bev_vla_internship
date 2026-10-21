"""将 DistilBERT 文本语义投影到 BEVFormer 通道空间。"""

import torch
from torch import Tensor, nn


class SemanticProjection(nn.Module):
    """文本向量 (B, D_text) -> BEV 引导向量 (B, C_bev)。

    Args:
        text_dim: 语言编码器输出维度，标准 DistilBERT 为 768。
        bev_dim: BEVFormer-Small 的特征通道，首版为 256。
    """

    def __init__(self, text_dim: int = 768, bev_dim: int = 256) -> None:
        super().__init__()
        if text_dim <= 0 or bev_dim <= 0:
            raise ValueError("text_dim 和 bev_dim 必须为正整数")
        self.text_dim = text_dim
        self.bev_dim = bev_dim
        self.norm = nn.LayerNorm(text_dim)
        self.projection = nn.Linear(text_dim, bev_dim)

    def forward(self, text_features: Tensor) -> Tensor:
        """返回 (B, C_bev) 特征，供次日空间门控模块使用。"""
        if text_features.ndim != 2 or text_features.shape[1] != self.text_dim:
            raise ValueError(
                f"期望 (B,{self.text_dim}) 文本特征，实际 {tuple(text_features.shape)}"
            )
        return self.projection(self.norm(text_features))

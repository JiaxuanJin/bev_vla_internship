"""将语言的全局语义向量映射到 BEV 特征的通道空间。"""

import torch
from torch import nn
from torch.nn import functional as F


class SemanticProjection(nn.Module):
    """将任意文本隐藏维度映射到指定 BEV 通道数。

    Args:
        text_dim: 文本隐藏层的特征维度。
        bev_channels: BEV 特征通道数量。
    """

    def __init__(self, text_dim: int, bev_channels: int = 256) -> None:
        super().__init__()
        if text_dim <= 0 or bev_channels <= 0:
            raise ValueError("text_dim 和 bev_channels 必须为正数")
        self.text_dim = text_dim
        self.bev_channels = bev_channels
        self.net = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, bev_channels),
            nn.GELU(),
            nn.Linear(bev_channels, bev_channels),
        )

    def forward(self, pooled_text: torch.Tensor) -> torch.Tensor:
        """接收 (B, D_text)，返回 (B, C_bev)。"""
        if pooled_text.ndim != 2 or pooled_text.shape[1] != self.text_dim:
            raise ValueError(f"语言向量必须为 (B,{self.text_dim}) 二维张量")
        if not torch.isfinite(pooled_text).all():
            raise ValueError("语言向量包含 NaN/Inf")
        return self.net(pooled_text)

    def forward_spatial(self, pooled_text: torch.Tensor, visual: torch.Tensor):
        """文本通道 query 与归一化视觉栅格交互，输出 Attention 之前的二维特征。

        S[b,c,r,k] = normalize(q)[b,c] * V[b,c,r,k]。
        该视觉条件映射保持原余弦门控的参数与数值路径，不声明定位准确性。
        """
        query = self(pooled_text)
        if visual.ndim != 4 or tuple(visual.shape[:2]) != tuple(query.shape) or min(visual.shape) <= 0:
            raise ValueError("视觉特征必须为与 query 一致的 [B,C,H,W]")
        if visual.device != query.device or visual.dtype != query.dtype:
            raise ValueError("二维语义映射的设备/类型不一致")
        if not torch.isfinite(visual).all():
            raise ValueError("二维语义映射的视觉特征非有限")
        semantic = F.normalize(query, dim=1, eps=1e-6)[:, :, None, None] * visual
        return query, semantic

    @staticmethod
    def to_spatial(query: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        """S[b,c,r,k] = query[b,c] * attention[b,0,r,k]，返回二维引导特征。

        空间选择来自视觉条件 attention；广播文本本身不产生定位真值。
        """
        if query.ndim != 2 or attention.ndim != 4 or attention.shape[1] != 1:
            raise ValueError("需要 query [B,C] 和 attention [B,1,H,W]")
        if query.shape[0] != attention.shape[0] or min(attention.shape) <= 0:
            raise ValueError("二维语义映射的 batch/空间尺寸错误")
        if query.device != attention.device or query.dtype != attention.dtype:
            raise ValueError("query 与 attention 的设备/类型必须一致")
        if min(query.shape) <= 0 or not query.is_floating_point():
            raise ValueError("query 必须是非空浮点张量")
        if not torch.isfinite(query).all() or not torch.isfinite(attention).all():
            raise ValueError("二维语义映射包含 NaN/Inf")
        if (attention < 0).any() or (attention > 1).any():
            raise ValueError("attention 必须是 [0,1] 概率")
        return query[:, :, None, None] * attention

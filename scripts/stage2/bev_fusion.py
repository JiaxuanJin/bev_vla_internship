"""可训练的语言条件 BEV 注意力残差融合层。"""

import math

import torch
from torch import nn

from .language_bev_attention import LanguageBEVAttention
from .semantic_projection import SemanticProjection


class LanguageBEVFusion(nn.Module):
    """从语言向量生成注意力图，并以小幅残差增强 BEV。

    Args:
        text_dim: 文本编码器隐藏维度。
        bev_channels: 输入 BEV 特征通道数。
        alpha_init: 初始增强系数；取小值降低未训练模块的干扰。
    """

    def __init__(
        self,
        text_dim: int,
        bev_channels: int = 256,
        alpha_init: float = 0.01,
    ) -> None:
        super().__init__()
        if not math.isfinite(alpha_init):
            raise ValueError("alpha_init 必须有限")
        self.semantic_projection = SemanticProjection(text_dim, bev_channels)
        self.attention = LanguageBEVAttention(bev_channels)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self,
        bev_features: torch.Tensor,
        pooled_text: torch.Tensor,
        enabled: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回增强 BEV (B,C,H,W) 和注意力 (B,1,H,W)。"""
        if bev_features.ndim != 4:
            raise ValueError("BEV 必须是 (B,C,H,W) 张量")
        if pooled_text.ndim != 2 or pooled_text.shape[0] != bev_features.shape[0]:
            raise ValueError("文本 batch 数必须等于 BEV batch 数")

        if not enabled:
            # 保证基线回退不修改或重新采样任何输入特征。
            blank = bev_features.new_zeros(
                (bev_features.shape[0], 1, *bev_features.shape[2:])
            )
            return bev_features, blank

        query = self.semantic_projection(pooled_text)
        attention = self.attention(bev_features, query)
        enhanced = bev_features + self.alpha * bev_features * attention
        return enhanced, attention

    def forward_with_details(
        self, bev_features: torch.Tensor, pooled_text: torch.Tensor,
        enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        """先生成二维语义，再由它产生掩码；默认 forward 和权重键不变。"""
        if not enabled:
            enhanced, attention = self(bev_features, pooled_text, enabled=False)
            return {"enhanced_bev": enhanced, "attention": attention,
                    "semantic_bev": torch.zeros_like(bev_features)}
        if bev_features.ndim != 4 or pooled_text.ndim != 2:
            raise ValueError("需要 BEV [B,C,H,W] 与文本 [B,D]")
        visual = self.attention.visual_features(bev_features)
        query, semantic = self.semantic_projection.forward_spatial(pooled_text, visual)
        attention = self.attention.from_semantic(semantic)
        enhanced = bev_features + self.alpha * bev_features * attention
        if not torch.isfinite(enhanced).all():
            raise ValueError("融合结果包含 NaN/Inf")
        return {"enhanced_bev": enhanced, "attention": attention,
                "text_query": query,
                "semantic_bev": semantic}

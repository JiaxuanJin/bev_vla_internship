"""计算语言语义与每个 BEV 栅格位置的相似度。"""

import torch
from torch import nn
from torch.nn import functional as F


class LanguageBEVAttention(nn.Module):
    """用通道投影与余弦相似度生成空间注意力。

    Args:
        bev_channels: BEV 通道数。
        temperature: Sigmoid 之前的相似度缩放温度。
    """

    def __init__(
        self,
        bev_channels: int = 256,
        temperature: float = 0.25,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature 必须大于零")
        self.visual_projection = nn.Conv2d(
            bev_channels, bev_channels, kernel_size=1, bias=False
        )
        self.temperature = temperature

    def forward(
        self,
        bev_features: torch.Tensor,
        text_query: torch.Tensor,
    ) -> torch.Tensor:
        """输入 BEV (B,C,H,W) 和语义 (B,C)，输出 (B,1,H,W)。"""
        if bev_features.ndim != 4 or text_query.ndim != 2:
            raise ValueError("输入维度应分别为 (B,C,H,W) 与 (B,C)")
        if bev_features.shape[:2] != text_query.shape:
            raise ValueError("文本和 BEV 的 batch / channel 不一致")
        spatial = F.normalize(
            self.visual_projection(bev_features), dim=1, eps=1e-6
        )
        query = F.normalize(text_query, dim=1, eps=1e-6)
        similarity = (spatial * query[:, :, None, None]).sum(
            dim=1, keepdim=True
        )
        return torch.sigmoid(similarity / self.temperature)

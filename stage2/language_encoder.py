"""DistilBERT 语言编码器。

编码器输出 768 维句向量；冻结主干后，后续可训练投影层仍正常反向传播。
不在模块导入时下载预训练模型，模型文件由下载脚本统一管理。
"""

from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer


class LanguageEncoder(nn.Module):
    """将 token 序列转换为句级语义特征。

    Args:
        backbone: 已加载的 Hugging Face Transformer 主干。
        freeze_backbone: 是否冻结语言主干并禁止其梯度计算。
    """

    def __init__(self, backbone: nn.Module, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        # DistilBERT 使用 dim 字段，兼容其他模型的 hidden_size 字段。
        config = backbone.config
        hidden_size = getattr(config, "dim", None)
        if hidden_size is None:
            hidden_size = getattr(config, "hidden_size")
        self.output_dim = int(hidden_size)

        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @classmethod
    def from_local_model(
        cls, model_dir: str | Path, freeze_backbone: bool = True
    ) -> "LanguageEncoder":
        """仅从本地目录加载模型，避免运行阶段再次访问网络。"""
        backbone = AutoModel.from_pretrained(
            str(model_dir), local_files_only=True, use_safetensors=True
        )
        return cls(backbone=backbone, freeze_backbone=freeze_backbone)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """对有效 token 做 masked mean pooling，输出 (B, D_text)。

        Args:
            input_ids: token ID，形状 (B, L)。
            attention_mask: 1 表示有效 token，0 表示 padding，形状 (B, L)。

        Returns:
            按有效 token 平均池化的文本特征，形状 (B, D_text)。

        Raises:
            ValueError: 输入形状不一致，或有样本不包含有效 token。
        """
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids 和 attention_mask 必须具有相同的 (B,L) 形状")
        if not torch.all(attention_mask.sum(dim=1) > 0).item():
            raise ValueError("每条文本至少需要一个有效 token")

        if self.freeze_backbone:
            # 冻结时保持 dropout 关闭；投影模块的梯度不受影响。
            self.backbone.eval()
        grad_context = torch.no_grad() if self.freeze_backbone else nullcontext()
        with grad_context:
            hidden = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state

        valid = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return pooled


def prepare_text_batch(
    texts: List[str], tokenizer: object, max_length: int = 64
) -> Dict[str, Tensor]:
    """将英文指令批量编码为可用于 DistilBERT 的张量。

    使用当前 batch 内最长序列补齐，固定截断上限为 max_length。
    """
    if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("texts 必须为非空字符串列表")
    if max_length <= 0:
        raise ValueError("max_length 必须是正整数")
    encoded = tokenizer(
        texts,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }


def load_local_tokenizer(model_dir: str | Path) -> object:
    """加载已缓存至 models/ 下的 tokenizer，不触发在线下载。"""
    return AutoTokenizer.from_pretrained(
        str(model_dir), use_fast=True, local_files_only=True
    )

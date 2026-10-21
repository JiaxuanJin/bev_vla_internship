"""DistilBERT 语言编码器封装；真实运行时才加载预训练权重。"""

import re

import torch
from torch import Tensor, nn


def normalize_instructions(texts: list[str]) -> list[str]:
    """只规范空白，不删除否定、方位或改变目标顺序。"""
    if not isinstance(texts, (list, tuple)) or not texts:
        raise ValueError("texts 必须是非空字符串列表")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("每条指令必须是非空字符串")
    return [re.sub(r"\s+", " ", text).strip() for text in texts]


class DistilBERTLanguageEncoder(nn.Module):
    """使用 Hugging Face 预训练模型编码短文本指令。

    Args:
        model_name: 预训练模型名或本地模型目录。
        freeze: 是否冻结语言主干参数。
        local_files_only: 是否禁止模型的在线下载。
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        freeze: bool = True,
        local_files_only: bool = False,
        *,
        tokenizer=None,
        model=None,
    ) -> None:
        super().__init__()
        if (tokenizer is None) != (model is None):
            raise ValueError("测试注入时 tokenizer 和 model 必须同时指定")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "缺少 transformers，请先安装后再使用 DistilBERT 模式"
            ) from exc
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only, clean_up_tokenization_spaces=False
        )
        self.model = model if model is not None else AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only, use_safetensors=True
        )
        self.hidden_size = int(self.model.config.hidden_size)
        self.freeze = freeze
        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def tokenize(self, texts: list[str], device: str) -> dict[str, Tensor]:
        """保留原指令含义，按 batch 最长序列补齐并截断至 64 tokens。"""
        encoded = self.tokenizer(
            normalize_instructions(texts),
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        return {key: value.to(device) for key, value in encoded.items()}

    def forward(self, texts: list[str], device: str) -> Tensor:
        """输出 masked mean 向量；有限数值不等价于正确语义定位。"""
        tokens = self.tokenize(texts, device)
        if self.freeze:
            self.model.eval()  # 上层 train() 不能重新启用冻结主干的 dropout。
            with torch.no_grad():
                hidden = self.model(**tokens).last_hidden_state
        else:
            hidden = self.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if not torch.isfinite(pooled).all():
            raise ValueError("文本特征包含 NaN/Inf")
        return pooled

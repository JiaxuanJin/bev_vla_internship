"""DistilBERT 语言编码器封装；真实运行时才加载预训练权重。"""

from torch import Tensor, nn


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
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "缺少 transformers，请先安装后再使用 DistilBERT 模式"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.hidden_size = int(self.model.config.hidden_size)
        self.freeze = freeze
        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def forward(self, texts: list[str], device: str) -> Tensor:
        """将文本列表映射到 (B,D_text) 的 mask 感知池化向量。"""
        import torch

        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        if self.freeze:
            with torch.no_grad():
                hidden = self.model(**tokens).last_hidden_state
        else:
            hidden = self.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

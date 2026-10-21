"""统一文本→二维语义/注意力→共享 BEV 接口，不包含 GT 或优化器。"""

from contextlib import contextmanager

import torch
from torch import Tensor, nn

from .bev_coordinates import BEVGrid
from .bev_fusion import LanguageBEVFusion
from .language_encoder import DistilBERTLanguageEncoder, normalize_instructions
from .semantic_instruction import parse_instruction


class Stage2Pipeline(nn.Module):
    """组合既有编码、SemanticProjection、Attention 和 Fusion。

    规则解析只提供受约束输入校验和元数据，不将 GT 或方向模板注入注意力。
    二维 semantic_bev 是 normalize(q)*V，并作为 Attention 输入；
    残差仍为 BEV + alpha*BEV*A，保持既有权重键。
    """

    def __init__(self, encoder: nn.Module, grid: BEVGrid | None = None):
        super().__init__()
        self.encoder = encoder
        self.grid = grid or BEVGrid()
        self.fusion = LanguageBEVFusion(encoder.hidden_size, self.grid.channels)

    @classmethod
    def from_local_model(cls, model_dir, grid: BEVGrid | None = None):
        return cls(DistilBERTLanguageEncoder(
            model_name=str(model_dir), freeze=True, local_files_only=True,
        ), grid)

    def encode(self, texts, device, *, strict=True):
        normalized = normalize_instructions(texts)
        instructions = []
        for text in normalized:
            try:
                instructions.append(parse_instruction(text).to_dict())
            except ValueError:
                if strict:
                    raise
                instructions.append(None)
        embedding = self.encoder(normalized, str(device))
        return embedding, instructions

    def forward(self, bev: Tensor, texts=None, *, enabled=True, strict=True):
        if bev.ndim != 4 or tuple(bev.shape[1:]) != (
            self.grid.channels, self.grid.height, self.grid.width
        ) or bev.shape[0] <= 0:
            raise ValueError("BEV 必须符合声明的 [B,C,H,W] 网格，不能 resize")
        if not enabled:
            # 真正旁路：不调用 tokenizer、编码器或投影层，保留原 tensor。
            return {"enhanced_bev": bev,
                    "attention": bev.new_zeros((bev.shape[0], 1, *bev.shape[2:])),
                    "semantic_bev": torch.zeros_like(bev), "instructions": []}
        embedding, instructions = self.encode(texts, bev.device, strict=strict)
        return self.fuse_encoded(bev, embedding) | {"instructions": instructions}

    def fuse_encoded(self, bev: Tensor, embedding: Tensor):
        if embedding.shape != (bev.shape[0], self.encoder.hidden_size):
            raise ValueError("一帧必须对应一条指令，文本 batch 不得隐式广播")
        if embedding.device != bev.device:
            raise ValueError("语言与 BEV 设备不一致")
        dtype = self.fusion.alpha.dtype
        details = self.fusion.forward_with_details(bev.to(dtype), embedding.to(dtype))
        details["enhanced_bev"] = details["enhanced_bev"].to(bev.dtype)
        details["text_embedding"] = embedding
        return details

    def fuse_sequence(self, sequence: Tensor, embedding: Tensor):
        bev = self.grid.to_map(sequence)
        details = self.fuse_encoded(bev, embedding)
        return self.grid.to_sequence(details["enhanced_bev"]), details

    @contextmanager
    def inject_shared_bev(
        self, transformer, texts=None, *, enabled=True, strict=True,
        capture_details=False,
    ):
        """临时注入 get_bev_features，在异常和正常退出时均恢复原方法。"""
        original = transformer.get_bev_features
        if getattr(original, "_stage2_hook", False):
            raise RuntimeError("同一 Transformer 不允许嵌套 Stage2 注入")
        had_local = "get_bev_features" in transformer.__dict__
        local_value = transformer.__dict__.get("get_bev_features")
        state = {"hook_calls": 0, "enabled": enabled, "trace": {}}
        embedding = None
        if enabled:
            device = self.fusion.alpha.device
            embedding, state["instructions"] = self.encode(texts, device, strict=strict)

        def injected(*args, **kwargs):
            sequence = original(*args, **kwargs)
            self.grid.to_map(sequence)  # 即使旁路也检查作者张量协议。
            state["hook_calls"] += 1
            if not enabled:
                return sequence
            enhanced, details = self.fuse_sequence(sequence, embedding)
            state["trace"] = {name: list(value.shape) for name, value in details.items()}
            if capture_details:
                # 仅反传验收保留同一次前向的计算图；默认推理不保留。
                state["details"] = details
            return enhanced

        injected._stage2_hook = True
        transformer.get_bev_features = injected
        try:
            yield state
        finally:
            if had_local:
                transformer.get_bev_features = local_value
            else:
                delattr(transformer, "get_bev_features")

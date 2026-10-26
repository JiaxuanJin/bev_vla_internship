"""阶段二 Day 1 单元测试：无需网络与模型下载即可运行。"""

import sys
from pathlib import Path

import pytest
import torch
from transformers import DistilBertConfig, DistilBertModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage2.language_encoder import LanguageEncoder  # noqa: E402
from stage2.semantic_projection import SemanticProjection  # noqa: E402


@pytest.fixture()
def tiny_encoder() -> LanguageEncoder:
    """使用随机初始化的极小 DistilBERT 检测模块逻辑，不下载模型。"""
    torch.manual_seed(123)
    config = DistilBertConfig(
        vocab_size=100,
        max_position_embeddings=32,
        n_layers=1,
        n_heads=4,
        dim=32,
        hidden_dim=64,
        dropout=0.0,
        attention_dropout=0.0,
    )
    return LanguageEncoder(DistilBertModel(config), freeze_backbone=True)


def test_batch_and_padding_shapes(tiny_encoder: LanguageEncoder) -> None:
    """不同长度的两条指令可以同批编码。"""
    input_ids = torch.tensor([[5, 6, 7, 8, 0], [5, 6, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])
    result = tiny_encoder(input_ids, mask)
    assert result.shape == (2, 32)
    assert torch.isfinite(result).all().item()


def test_pad_tokens_do_not_change_pooled_features(
    tiny_encoder: LanguageEncoder,
) -> None:
    """在 attention_mask=0 的位置替换 token，不应改变有效文本特征。"""
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    one = tiny_encoder(torch.tensor([[5, 6, 7, 0, 0]]), mask)
    two = tiny_encoder(torch.tensor([[5, 6, 7, 8, 9]]), mask)
    torch.testing.assert_close(one, two, rtol=1e-5, atol=1e-6)


def test_projector_shape_and_gradient(tiny_encoder: LanguageEncoder) -> None:
    """验证映射维度，以及冻结语言主干时投影层仍可反传。"""
    ids = torch.tensor([[5, 6, 7], [4, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    projector = SemanticProjection(text_dim=32, bev_dim=256)
    guidance = projector(tiny_encoder(ids, mask))
    assert guidance.shape == (2, 256)
    guidance.square().mean().backward()
    assert projector.projection.weight.grad is not None
    assert projector.projection.weight.grad.abs().sum().item() > 0
    assert all(not p.requires_grad for p in tiny_encoder.parameters())


def test_invalid_empty_mask_is_rejected(tiny_encoder: LanguageEncoder) -> None:
    """空有效 token 样本直接报错，避免 silent invalid 池化。"""
    with pytest.raises(ValueError, match="至少需要一个有效 token"):
        tiny_encoder(torch.tensor([[0, 0]]), torch.tensor([[0, 0]]))


def test_wrong_feature_dim_is_rejected() -> None:
    """错误的文本向量维度不能被隐式广播。"""
    projector = SemanticProjection(text_dim=768, bev_dim=256)
    with pytest.raises(ValueError, match="期望"):
        projector(torch.zeros(2, 256))

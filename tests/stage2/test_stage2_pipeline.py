"""阶段二工程契约测试；极小随机 DistilBERT 和合成 BEV 不代表定位准确率。"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.bev_fusion import LanguageBEVFusion
from scripts.stage2.language_bev_attention import LanguageBEVAttention
from scripts.stage2.language_encoder import DistilBERTLanguageEncoder, normalize_instructions
from scripts.stage2.multitask_training import LossConfig, Stage2MultiTaskLoss, supervised_attention_loss
from scripts.stage2.pipeline import Stage2Pipeline
from scripts.stage2.semantic_instruction import ego_direction_mask, parse_instruction
from scripts.stage2.semantic_projection import SemanticProjection


@pytest.fixture()
def encoder(tmp_path):
    words = ("[PAD] [UNK] [CLS] [SEP] [MASK] find focus show locate highlight lane markings "
             "lines dividers road edges boundaries curb pedestrian crossings crosswalk zebra "
             "ahead front rear left right on the and but ignore in behind").split()
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(words) + "\n")
    tokenizer = DistilBertTokenizerFast(vocab_file=str(vocab), do_lower_case=True,
                                       clean_up_tokenization_spaces=False)
    config = DistilBertConfig(vocab_size=len(words), dim=32, hidden_dim=48, n_layers=1,
                             n_heads=4, max_position_embeddings=64, dropout=0.3, attention_dropout=0.3)
    torch.manual_seed(123)
    return DistilBERTLanguageEncoder(tokenizer=tokenizer, model=DistilBertModel(config), freeze=True)


@pytest.mark.parametrize("text,targets,excluded,fb,lr", [
    ("Find lane markings ahead", ("divider",), (), "front", None),
    ("Locate lane dividers in front", ("divider",), (), "front", None),
    ("Show the crosswalk on the front right", ("ped_crossing",), (), "front", "right"),
    ("Show the zebra crossing on the front right", ("ped_crossing",), (), "front", "right"),
    ("Find road edges behind on the left", ("boundary",), (), "rear", "left"),
    ("Find the curb behind on the left", ("boundary",), (), "rear", "left"),
    ("Find lane lines and road edges ahead", ("divider", "boundary"), (), "front", None),
    ("Highlight road boundaries but ignore pedestrian crossings", ("boundary",), ("ped_crossing",), None, None),
    ("Find lane lines without road edges", ("divider",), ("boundary",), None, None),
])
def test_supported_instructions(text, targets, excluded, fb, lr):
    ins = parse_instruction(text)
    assert (ins.targets, ins.excluded, ins.front_back, ins.left_right) == (targets, excluded, fb, lr)


@pytest.mark.parametrize("text", [
    "", "Ignore lane markings", "Find lane lines and traffic lights", "Find traffic lights",
    "Do not ignore lane lines", "Find lane lines or road boundaries",
    "Find lane lines in front and behind", "Find lane lines on the left and right",
    "Find lane lines but ignore lane lines", "Find lane lines not in front",
    "Find lane lines but ignore road edges and crosswalks",
    "ahead markings lane focus",  # 词序置乱破坏目标短语；编码器仍可单独编码。
])
def test_unsupported_or_ambiguous_instructions_are_rejected(text):
    with pytest.raises(ValueError):
        parse_instruction(text)


@pytest.mark.parametrize("invalid", [None, "Find lane markings", [], [""], [None], [3]])
def test_preprocessing_rejects_invalid_batch(invalid):
    with pytest.raises(ValueError):
        normalize_instructions(invalid)


def test_normalization_preserves_negation_and_directions():
    assert normalize_instructions([" \tFind  lane lines\nbut ignore road edges on the right  "]) == [
        "Find lane lines but ignore road edges on the right"
    ]


def test_encoder_padding_mask_and_frozen_dropout(encoder):
    encoder.train()
    texts = ["Find lane markings", "Highlight road boundaries but ignore pedestrian crossings on the right"]
    one = encoder(texts[:1], "cpu")
    batch = encoder(texts, "cpu")
    tokens = encoder.tokenize(texts, "cpu")
    assert (tokens["attention_mask"] == 0).any()
    torch.testing.assert_close(one[0], batch[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batch, encoder(texts, "cpu"), rtol=0, atol=0)
    assert not encoder.model.training
    assert batch.shape == (2, 32) and torch.isfinite(batch).all()
    assert all(not p.requires_grad for p in encoder.model.parameters())


def test_masked_padding_token_content_does_not_affect_encoding(encoder):
    tokens = encoder.tokenize(["Find lane markings", "Find road boundaries on the front right"], "cpu")
    altered = {k: v.clone() for k, v in tokens.items()}
    altered["input_ids"][tokens["attention_mask"] == 0] = encoder.tokenizer.convert_tokens_to_ids("curb")
    with torch.no_grad():
        original = encoder.model(**tokens).last_hidden_state
        changed = encoder.model(**altered).last_hidden_state
    valid = tokens["attention_mask"].bool()
    torch.testing.assert_close(original[valid], changed[valid], rtol=1e-5, atol=1e-6)


def test_batch_shuffle_and_word_shuffle_are_distinct_checks(encoder):
    texts = ["Find lane markings ahead", "Find the curb on the right"]
    vectors = encoder(texts, "cpu")
    torch.testing.assert_close(encoder(list(reversed(texts)), "cpu").flip(0), vectors)
    shuffled = "ahead markings lane find"
    encoded = encoder([shuffled], "cpu")
    assert torch.isfinite(encoded).all() and encoded.shape == (1, 32)
    with pytest.raises(ValueError):
        parse_instruction(shuffled)
    # 不要求同义文本向量相等，也不把不同向量视为正确定位。


def test_tokenizer_truncation_is_explicit(encoder):
    tokens = encoder.tokenize(["lane " * 100], "cpu")
    assert tokens["input_ids"].shape == (1, 64)
    assert tokens["input_ids"][0, -1] == encoder.tokenizer.sep_token_id


def test_non_square_row_major_roundtrip_has_no_transpose():
    grid = BEVGrid(2, 3, 4)
    sequence = torch.arange(2 * 6 * 4).reshape(2, 6, 4)
    mapped = grid.to_map(sequence)
    assert mapped.shape == (2, 4, 2, 3)
    assert mapped[1, 3, 1, 2] == sequence[1, 1 * 3 + 2, 3]
    torch.testing.assert_close(grid.to_sequence(mapped), sequence)
    noncontiguous = torch.randn(2, 4, 3, 2).transpose(2, 3)
    torch.testing.assert_close(grid.to_map(grid.to_sequence(noncontiguous)), noncontiguous)


@pytest.mark.parametrize("bad", [torch.zeros(1, 4, 6), torch.zeros(1, 5, 4), torch.zeros(6, 4)])
def test_grid_rejects_wrong_sequence_shape(bad):
    with pytest.raises(ValueError):
        BEVGrid(2, 3, 4).to_map(bad)


def test_metric_centers_rotation_translation_and_ego_front_right():
    grid = BEVGrid(2, 3, 4, (-3, -2, -1, 3, 2, 1))
    lidar = grid.lidar_centers()
    torch.testing.assert_close(lidar[0, 0], torch.tensor([-2., -1., 0.], dtype=torch.float64))
    torch.testing.assert_close(lidar[1, 2], torch.tensor([2., 1., 0.], dtype=torch.float64))
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
    shift = torch.tensor([1., 2., 0.], dtype=torch.float64)
    ego = grid.ego_centers(rotation, shift)
    torch.testing.assert_close(ego[0, 0], torch.tensor([2., 0., 0.], dtype=torch.float64))
    torch.testing.assert_close((ego - shift) @ rotation, lidar)
    direction = parse_instruction("Show crosswalk on the front right")
    actual = ego_direction_mask(np.array([1., -1., 1., -1., 0.]), np.array([-1., -1., 1., 1., -1.]), direction)
    assert actual.tolist() == [True, False, False, False, False]


@pytest.mark.parametrize("rotation", [torch.diag(torch.tensor([-1., 1., 1.])), torch.zeros(3, 3), torch.full((3, 3), float('nan'))])
def test_invalid_calibration_is_rejected(rotation):
    with pytest.raises(ValueError):
        BEVGrid().ego_centers(rotation, torch.zeros(3))


@pytest.mark.parametrize("direction,expected", [
    ("front right", [True, False, False, False]),
    ("front left", [False, True, False, False]),
    ("rear right", [False, False, True, False]),
    ("rear left", [False, False, False, True]),
    ("front", [True, True, False, False]),
    ("rear", [False, False, True, True]),
])
def test_direction_quadrants_use_ego_axes(direction, expected):
    ins = parse_instruction(f"Find lane lines on the {direction}")
    actual = ego_direction_mask(np.array([1., 1., -1., -1.]), np.array([-1., 1., -1., 1.]), ins)
    assert actual.tolist() == expected


def test_attention_uses_visual_locations_and_query_channels():
    attention = LanguageBEVAttention(2)
    with torch.no_grad():
        attention.visual_projection.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
    bev = torch.tensor([[[[1., -1., 0.]], [[0., 0., 1.]]]])
    mask = attention(bev, torch.tensor([[1., 0.]]))
    assert mask.shape == (1, 1, 1, 3)
    assert mask[0, 0, 0, 0] > 0.9 and mask[0, 0, 0, 1] < 0.1
    assert mask[0, 0, 0, 2] == 0.5
    spatial = SemanticProjection.to_spatial(torch.tensor([[2., 3.]]), mask)
    assert spatial.shape == (1, 2, 1, 3)
    torch.testing.assert_close(spatial[0, :, 0, 2], torch.tensor([1., 1.5]))


def test_768_to_256_to_two_dimensional_features_and_legacy_equivalence():
    fusion = LanguageBEVFusion(768).eval()
    bev, text = torch.randn(2, 256, 3, 5), torch.randn(2, 768)
    result = fusion.forward_with_details(bev, text)
    assert result["text_query"].shape == (2, 256)
    assert result["semantic_bev"].shape == (2, 256, 3, 5)
    assert result["attention"].shape == (2, 1, 3, 5)
    original, mask = fusion(bev, text)
    torch.testing.assert_close(original, result["enhanced_bev"], rtol=0, atol=0)
    torch.testing.assert_close(mask, result["attention"], rtol=0, atol=0)
    assert fusion(bev, text, enabled=False)[0] is bev


@pytest.mark.parametrize("text", [torch.zeros(2, 16), torch.full((2, 768), float('nan'))])
def test_projection_rejects_incorrect_or_nonfinite_features(text):
    with pytest.raises(ValueError):
        SemanticProjection(768)(text)


def test_attention_rejects_batch_broadcast_and_nonfinite_values():
    layer = LanguageBEVAttention(4)
    with pytest.raises(ValueError):
        layer(torch.zeros(2, 4, 2, 3), torch.zeros(1, 4))
    with pytest.raises(ValueError):
        layer(torch.full((1, 4, 2, 3), float('inf')), torch.zeros(1, 4))


@pytest.mark.parametrize("query,attention", [
    (torch.zeros(2, 4), torch.zeros(1, 1, 3, 5)),
    (torch.zeros(1, 4), torch.zeros(1, 2, 3, 5)),
    (torch.zeros(1, 4), torch.full((1, 1, 3, 5), 1.1)),
    (torch.zeros(1, 4), torch.full((1, 1, 3, 5), float('nan'))),
])
def test_spatial_semantics_rejects_invalid_shapes_and_probabilities(query, attention):
    with pytest.raises(ValueError):
        SemanticProjection.to_spatial(query, attention)


def test_same_bev_paired_with_different_text_is_a_functional_check(encoder):
    pipeline = Stage2Pipeline(encoder, BEVGrid(2, 3, 4)).eval()
    bev = torch.randn(1, 4, 2, 3).expand(2, -1, -1, -1)
    texts = ["Find lane lines", "Find road boundaries on the left"]
    normal = pipeline(bev, texts)
    shuffled = pipeline(bev, list(reversed(texts)))
    torch.testing.assert_close(normal["attention"], shuffled["attention"].flip(0))
    # 没有语言-区域真值，不设置语义定位准确率或提升通过阈值。


def test_fusion_state_dict_keys_remain_compatible():
    module = LanguageBEVFusion(768)
    expected = {"alpha", "semantic_projection.net.0.weight", "semantic_projection.net.0.bias",
                "semantic_projection.net.1.weight", "semantic_projection.net.1.bias",
                "semantic_projection.net.3.weight", "semantic_projection.net.3.bias",
                "attention.visual_projection.weight"}
    assert set(module.state_dict()) == expected
    LanguageBEVFusion(768).load_state_dict(module.state_dict(), strict=True)


def test_attention_consumes_spatial_head_output_before_mask():
    fusion = LanguageBEVFusion(768, bev_channels=4).eval()
    bev, text = torch.randn(2, 4, 3, 5), torch.randn(2, 768)
    details = fusion.forward_with_details(bev, text)
    semantic = details["semantic_bev"]
    assert semantic.shape == (2, 4, 3, 5) and semantic.requires_grad
    torch.testing.assert_close(fusion.attention.from_semantic(semantic), details["attention"], rtol=0, atol=0)
    # 强制改变二维语义输入会改变 mask；它是实际依赖，不能只是后置展示。
    neutral = fusion.attention.from_semantic(torch.zeros_like(semantic))
    torch.testing.assert_close(neutral, torch.full_like(neutral, 0.5))
    derivative = torch.autograd.grad(details["attention"].sum(), semantic)[0]
    assert torch.isfinite(derivative).all() and derivative.abs().sum() > 0


def test_captured_hook_details_retain_same_forward_graph(encoder):
    pipeline = Stage2Pipeline(encoder, BEVGrid(2, 3, 4))
    transformer = FakeTransformer()
    with pipeline.inject_shared_bev(transformer, ["Find lane lines"], capture_details=True) as state:
        enhanced = transformer.get_bev_features(torch.randn(1, 6, 4))
    assert state["hook_calls"] == 1 and state["details"]["attention"].requires_grad
    loss = enhanced.square().mean() + state["details"]["attention"].square().mean()
    loss.backward()
    assert pipeline.fusion.alpha.grad is not None
    assert "get_bev_features" not in transformer.__dict__


class FakeTransformer(nn.Module):
    def get_bev_features(self, sequence):
        return sequence


class FakeDualHead(nn.Module):
    """仅用于 Hook 集成契约测试，绝不是 BEVFormer 真实模型。"""

    def __init__(self, grid):
        super().__init__()
        self.grid = grid
        self.transformer = FakeTransformer()
        self.cls = nn.Linear(grid.channels, 10)
        self.box = nn.Linear(grid.channels, 10)
        self.seg = nn.Conv2d(grid.channels, 4, 1)

    def forward(self, sequence):
        shared = self.transformer.get_bev_features(sequence)
        return {"all_cls_scores": self.cls(shared), "all_bbox_preds": self.box(shared),
                "seg_preds": self.seg(self.grid.to_map(shared))}


def test_pipeline_shared_hook_both_heads_bypass_and_exception_cleanup(encoder):
    grid = BEVGrid(3, 5, 4)
    pipeline = Stage2Pipeline(encoder, grid).eval()
    head = FakeDualHead(grid).eval()
    sequence = torch.randn(1, 15, 4)
    baseline = head(sequence)
    original = head.transformer.get_bev_features
    with pipeline.inject_shared_bev(head.transformer, enabled=False) as state:
        assert head.transformer.get_bev_features(sequence) is sequence
        bypass = head(sequence)
    assert state["hook_calls"] == 2
    for key in baseline:
        torch.testing.assert_close(baseline[key], bypass[key], rtol=0, atol=0)
    with pipeline.inject_shared_bev(head.transformer, ["Find lane lines ahead"]) as state:
        enhanced = head(sequence)
        with pytest.raises(RuntimeError, match="嵌套"):
            with pipeline.inject_shared_bev(head.transformer, enabled=False):
                pass
    assert state["hook_calls"] == 1
    assert state["trace"]["semantic_bev"] == [1, 4, 3, 5]
    assert state["trace"]["text_embedding"] == [1, 32]
    assert head.transformer.get_bev_features == original
    assert "get_bev_features" not in head.transformer.__dict__
    for key in baseline:
        assert enhanced[key].shape == baseline[key].shape
        assert torch.isfinite(enhanced[key]).all()
    with pytest.raises(RuntimeError, match="intentional"):
        with pipeline.inject_shared_bev(head.transformer, ["Find lane lines"]):
            raise RuntimeError("intentional")
    assert head.transformer.get_bev_features == original


def test_pipeline_strict_input_and_exact_bypass_without_language(encoder):
    pipeline = Stage2Pipeline(encoder, BEVGrid(2, 3, 4))
    bev = torch.randn(2, 4, 2, 3)
    assert pipeline(bev, texts=None, enabled=False)["enhanced_bev"] is bev
    with pytest.raises(ValueError, match="batch"):
        pipeline(bev, ["Find lane lines"])
    with pytest.raises(ValueError):
        pipeline(bev, ["Find traffic lights", "Find lane lines"])
    with pytest.raises(ValueError):
        pipeline(torch.randn(1, 4, 3, 2), ["Find lane lines"])


def test_three_tasks_reach_same_fusion_without_updating_frozen_heads(encoder):
    grid = BEVGrid(3, 5, 4)
    pipeline = Stage2Pipeline(encoder, grid)
    head = FakeDualHead(grid).requires_grad_(False)
    before = {k: v.clone() for k, v in head.state_dict().items()}
    result = pipeline(torch.randn(2, 4, 3, 5), ["Find lane lines", "Find crosswalk on the left"])
    predictions = head(grid.to_sequence(result["enhanced_bev"]))
    gt = torch.zeros(2, 3, 5, dtype=torch.long)
    gt[0, 1, 1:4], gt[1, 1:, 1:3] = 1, 2
    target = torch.tensor([1, 2])
    det = {"loss_cls": predictions["all_cls_scores"].square().mean(),
           "loss_bbox": predictions["all_bbox_preds"].square().mean()}
    attention_logits = torch.logit(result["attention"].clamp(1e-6, 1 - 1e-6))
    cfg = LossConfig()
    seg = torch.nn.functional.cross_entropy(predictions["seg_preds"], gt)
    att = supervised_attention_loss(attention_logits, gt, target, cfg)
    params = list(pipeline.fusion.parameters())
    for loss in (sum(det.values()), seg, att):
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        valid = [g for g in grads if g is not None]
        assert valid and all(torch.isfinite(g).all() for g in valid)
        assert sum(float(g.abs().sum()) for g in valid) > 0
    total, parts = Stage2MultiTaskLoss(cfg)(det, predictions["seg_preds"], gt, attention_logits, target)
    assert all(torch.isfinite(v) for v in parts.values())
    total.backward()
    assert pipeline.fusion.alpha.grad is not None
    assert all(p.grad is None for p in encoder.parameters())
    assert all(p.grad is None for p in head.parameters())
    for k, value in head.state_dict().items():
        torch.testing.assert_close(value, before[k], rtol=0, atol=0)

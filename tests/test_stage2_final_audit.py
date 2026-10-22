"""2026-10-22 缺口修复测试：正则定义、指令绑定、Hook 与只读复现。"""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2.multitask_training import (
    LossConfig,
    Stage2MultiTaskLoss,
    attention_regularization_loss,
    supervised_attention_loss,
)
from scripts.stage2.semantic_instruction import attention_target_class
from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.pipeline import Stage2Pipeline


@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 1), (2, 1, 1, 5), (1, 1, 5, 1), (2, 1, 3, 5)]
)
def test_tv_constant_and_degenerate_grids_have_finite_zero_gradient(shape):
    logits = torch.full(shape, 3.0, requires_grad=True)
    value = attention_regularization_loss(logits)
    assert value.item() == 0
    value.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_tv_formula_gradients_and_saturation_are_stable():
    logits = torch.tensor(
        [[[[-1000.0, 0.0, 1000.0], [1.0, -1.0, 2.0]]]], requires_grad=True
    )
    probability = logits.sigmoid()
    expected = (probability[:, :, 1:] - probability[:, :, :-1]).abs().mean() + (
        probability[:, :, :, 1:] - probability[:, :, :, :-1]
    ).abs().mean()
    value = attention_regularization_loss(logits)
    torch.testing.assert_close(value, expected)
    value.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "invalid",
    [
        torch.zeros(1, 2, 3, 4),
        torch.zeros(1, 1, 0, 4),
        torch.zeros(1, 3),
        torch.full((1, 1, 2, 3), float("nan")),
        torch.ones(1, 1, 2, 3, dtype=torch.long),
    ],
)
def test_tv_rejects_invalid_inputs(invalid):
    with pytest.raises((TypeError, ValueError)):
        attention_regularization_loss(invalid)


@pytest.mark.parametrize("weight", [-0.1, float("inf"), float("nan")])
def test_tv_weight_rejects_invalid_config(weight):
    with pytest.raises(ValueError):
        LossConfig(attention_tv_weight=weight)


def loss_inputs():
    torch.manual_seed(422)
    det = {"loss_cls": torch.tensor(2.0, requires_grad=True)}
    seg = torch.randn(1, 4, 2, 3, requires_grad=True)
    gt = torch.tensor([[[0, 1, 255], [0, 1, 0]]])
    att = torch.randn(1, 1, 2, 3, requires_grad=True)
    return det, seg, gt, att, torch.tensor([1])


def test_zero_tv_weight_is_exact_legacy_formula_and_gradient():
    det, seg, gt, att, target = loss_inputs()
    cfg = LossConfig()
    old = (
        det["loss_cls"]
        + torch.nn.functional.cross_entropy(seg, gt, ignore_index=255)
        + 0.1 * supervised_attention_loss(att, gt, target, cfg)
    )
    new, parts = Stage2MultiTaskLoss(cfg)(det, seg, gt, att, target)
    assert torch.equal(new, old)
    old_gradient = torch.autograd.grad(old, (seg, att), retain_graph=True)
    new_gradient = torch.autograd.grad(new, (seg, att))
    assert all(torch.equal(a, b) for a, b in zip(old_gradient, new_gradient))
    assert parts["attention_regularization_loss"] == 0
    assert torch.equal(parts["attention_loss"], parts["attention_supervision_loss"])


def test_native_grid_regularization_is_weighted_once_and_independent_of_gt_presence():
    det, seg, gt, att, target = loss_inputs()
    native = torch.randn(1, 1, 3, 5, requires_grad=True)
    base, _ = Stage2MultiTaskLoss()(det, seg, gt, att, target)
    combined, parts = Stage2MultiTaskLoss(LossConfig(attention_tv_weight=0.01))(
        det, seg, gt, att, target, regularization_logits=native
    )
    torch.testing.assert_close(
        combined, base + 0.01 * attention_regularization_loss(native), rtol=0, atol=0
    )
    combined.backward()
    assert native.grad.abs().sum() > 0 and torch.isfinite(native.grad).all()
    assert parts["attention_regularization_loss"] > 0
    absent_gt = torch.zeros_like(gt)
    _, parts = Stage2MultiTaskLoss(LossConfig(attention_tv_weight=0.01))(
        det, seg, absent_gt, att, target, regularization_logits=native
    )
    assert (
        parts["attention_supervision_loss"] == 0
        and parts["attention_regularization_loss"] > 0
    )


def test_supervision_ignore_pixels_and_invalid_label_boundaries():
    _, _, gt, att, target = loss_inputs()
    loss = supervised_attention_loss(att, gt, target, LossConfig())
    grad = torch.autograd.grad(loss, att)[0]
    assert grad[0, 0, 0, 2] == 0  # ignore_index=255 不参与监督。
    for invalid in (gt.float(), torch.full_like(gt, 4)):
        with pytest.raises((ValueError, TypeError)):
            supervised_attention_loss(att, invalid, target, LossConfig())


def test_regularization_batch_mismatch_rejected_and_zero_weight_skips_unused_input():
    det, seg, gt, att, target = loss_inputs()
    invalid = torch.zeros(2, 1, 3, 5)
    with pytest.raises(ValueError, match="batch"):
        Stage2MultiTaskLoss(LossConfig(attention_tv_weight=0.01))(
            det, seg, gt, att, target, regularization_logits=invalid
        )
    # 零权重忽略未使用的正则输入；总损失仍与历史公式相同。
    legacy, _ = Stage2MultiTaskLoss()(det, seg, gt, att, target)
    skipped, _ = Stage2MultiTaskLoss()(
        det, seg, gt, att, target, regularization_logits=invalid
    )
    assert torch.equal(legacy, skipped)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Find lane markings", 1),
        ("Show crosswalk on the left", 2),
        ("Find road boundaries but ignore pedestrian crossings", 3),
    ],
)
def test_supervision_target_comes_from_parsed_text(text, expected):
    assert attention_target_class(text) == expected


@pytest.mark.parametrize(
    "text", ["Find lane markings and road boundaries", "Ignore crosswalk", "unknown"]
)
def test_ambiguous_supervision_target_is_rejected(text):
    with pytest.raises(ValueError):
        attention_target_class(text)


def test_mismatched_label_rejected_before_any_gpu_or_model_call():
    from scripts.stage2.micro_training import real_training_forward

    with pytest.raises(ValueError, match="target_class"):
        real_training_forward(None, None, None, {}, "Find road boundaries", 1)


class Encoder(nn.Module):
    hidden_size = 4

    def forward(self, texts, device):
        return torch.tensor([[1.0, 2.0, 3.0, 4.0] for _ in texts], device=device)


class Transformer:
    def get_bev_features(self, sequence):
        return sequence


def test_hook_restores_existing_instance_override_after_repeated_calls_and_forward_error():
    pipeline = Stage2Pipeline(Encoder(), BEVGrid(2, 3, 4))
    transformer = Transformer()
    original = lambda sequence: sequence
    transformer.get_bev_features = original
    for _ in range(2):
        with pipeline.inject_shared_bev(transformer, ["Find lane markings"]) as state:
            transformer.get_bev_features(torch.ones(1, 6, 4))
            transformer.get_bev_features(torch.zeros(1, 6, 4))
        assert state["hook_calls"] == 2 and transformer.get_bev_features is original
    with pytest.raises(ValueError):
        with pipeline.inject_shared_bev(transformer, ["Find lane markings"]):
            transformer.get_bev_features(torch.zeros(1, 5, 4))
    assert transformer.get_bev_features is original


def test_parser_structure_is_metadata_and_does_not_inject_a_direction_template(
    monkeypatch,
):
    import scripts.stage2.pipeline as module

    pipeline = Stage2Pipeline(Encoder(), BEVGrid(2, 3, 4)).eval()
    bev = torch.randn(1, 4, 2, 3)
    original = pipeline(bev, ["Find lane markings ahead"])
    monkeypatch.setattr(
        module,
        "parse_instruction",
        lambda text: SimpleNamespace(to_dict=lambda: {"fake_direction": "rear"}),
    )
    changed = pipeline(bev, ["Find lane markings ahead"])
    assert changed["instructions"] != original["instructions"]
    assert torch.equal(changed["attention"], original["attention"])


def test_separate_fusion_weight_loading_preserves_encoder_namespace(tmp_path):
    from scripts.stage2.integration_smoke import load_fusion_weights

    pipeline = Stage2Pipeline(Encoder(), BEVGrid(2, 3, 4))
    path = tmp_path / "fusion.pt"
    saved = {k: v.clone() for k, v in pipeline.fusion.state_dict().items()}
    torch.save({"state_dict": saved, "steps": 20}, path)
    with torch.no_grad():
        pipeline.fusion.alpha.fill_(99)
    record = load_fusion_weights(pipeline, path)
    assert record["recorded_steps"] == 20
    assert all(
        torch.equal(v, saved[k]) for k, v in pipeline.fusion.state_dict().items()
    )


def test_snapshot_hash_relocation_checks_archived_bytes(tmp_path):
    from scripts.stage2.replay_stage2 import verify_snapshot, digest

    path = tmp_path / "archive/scripts/stage2/file.py"
    path.parent.mkdir(parents=True)
    path.write_text("original source\n")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps({"source_sha256": {"/old/scripts/stage2/file.py": digest(path)}})
    )
    assert (
        verify_snapshot(tmp_path / "archive", protocol)["/old/scripts/stage2/file.py"]
        == path
    )
    path.write_text("changed source\n")
    with pytest.raises(ValueError):
        verify_snapshot(tmp_path / "archive", protocol)


def test_smoke_orchestration_never_launches_training(tmp_path, monkeypatch):
    import scripts.stage2.micro_process as module

    calls = []

    def worker(args, worker, directory, protocol=None):
        calls.append(worker)
        return {"result": {"status": "PASS_REAL_INTEGRATION_SMOKE_NO_UPDATE"}}

    monkeypatch.setattr(module, "run_worker", worker)
    args = SimpleNamespace(
        output=tmp_path / "new/run.json",
        micro_steps=1,
        micro_phase="smoke",
        protocol=None,
        resume=None,
    )
    result = module.run_micro_stages(args)
    assert calls == ["protocol", "smoke"] and result["optimizer_steps"] == 0

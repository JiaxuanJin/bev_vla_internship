"""LG-BEV 合成契约和真实验收证据；合成 stub 只用于接口测试。"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.spatial_annotations import metric_grid, project_attention
from scripts.stage3.lg_bev_common import assert_splits
from scripts.stage3.lg_bev_data import (
    box_masks,
    exclusive_masks,
    geometry,
    load_supervision,
)
from scripts.stage3.lg_bev_evaluate import paired_forward
from scripts.stage3.lg_bev_native import dual_head, gradient_audit
from scripts.stage3.lg_bev_prototype import (
    LanguageGroundedBEVAdapter,
    attention_supervision_loss,
)
from scripts.stage3.lg_bev_train import frame_input, make_schedule, training_frame


@pytest.fixture
def sample() -> tuple:
    """固定随机样本；小空间不代替真实双 Head 验收。"""
    torch.manual_seed(20261027)
    adapter = LanguageGroundedBEVAdapter(bev_channels=8, text_dim=16, hidden_channels=4)
    return adapter, torch.randn(2, 8, 6, 7), torch.randn(2, 16), torch.randn(2, 5, 6, 7)


def test_shape_identity_and_language(sample: tuple) -> None:
    adapter, bev, text, pos = sample
    correct = adapter(bev, text, pos)
    wrong = adapter(bev, text.flip(0), pos)
    assert torch.equal(correct["enhanced_bev"], bev)
    assert correct["attention_logits"].shape == (2, 1, 6, 7)
    assert not torch.equal(correct["attention_logits"], wrong["attention_logits"])
    assert not torch.count_nonzero(correct["residual"])


def test_zero_up_gradient_routes_then_update(sample: tuple) -> None:
    adapter, bev, text, pos = sample
    detail = adapter(bev, text, pos)
    task_loss = detail["enhanced_bev"].square().mean()
    report = gradient_audit(task_loss, adapter)
    assert report["parameters"]["up.weight"]["l2"] > 0
    assert report["parameters"]["language.1.weight"]["l2"] == 0
    target = torch.zeros(2, 1, 6, 7)
    target[0, 0, 1:3, 1:3] = 1
    mask = attention_supervision_loss(detail["attention_logits"], target)
    report = gradient_audit(mask["total"], adapter)
    assert report["parameters"]["language.1.weight"]["l2"] > 0
    assert not report["parameters"]["up.weight"]["connected"]
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001)
    (task_loss + mask["total"]).backward()
    optimizer.step()
    detail = adapter(bev, text, pos)
    report = gradient_audit(detail["enhanced_bev"].square().mean(), adapter)
    assert report["parameters"]["language.1.weight"]["l2"] > 0
    assert detail["residual"].norm() > 0
    rms = bev.square().mean((2, 3), keepdim=True).sqrt()
    assert bool((detail["residual"].abs() <= 0.05 * rms + 1e-7).all())


def test_empty_target_and_valid_region() -> None:
    logits = torch.randn(2, 1, 5, 5, requires_grad=True)
    empty = torch.zeros_like(logits)
    losses = attention_supervision_loss(logits, empty)
    assert torch.isfinite(losses["total"])
    assert losses["dice"] == 0
    invalid = attention_supervision_loss(logits, empty, valid_mask=empty)
    assert invalid["total"] == 0
    invalid["total"].backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_empty_group_cannot_dominate_positive_group() -> None:
    positive = torch.zeros(1, 1, 4, 4)
    positive[..., 0, 0] = 1
    logits = torch.ones_like(positive)
    first = attention_supervision_loss(
        torch.cat((logits, -logits)), torch.cat((positive, positive * 0))
    )
    repeated = attention_supervision_loss(
        torch.cat((logits, (-logits).repeat(9, 1, 1, 1))),
        torch.cat((positive, positive.repeat(9, 1, 1, 1) * 0)),
    )
    assert torch.allclose(first["bce"], repeated["bce"])


def test_mask_conflict_and_out_of_bounds() -> None:
    masks = np.zeros((2, 5, 5), dtype=bool)
    masks[0, 1:4, 1:4], masks[1, 2:5, 2:5] = True, True
    target, other, conflict = exclusive_masks(masks, [0], [1])
    assert conflict == 4
    assert not (target & other).any()
    with pytest.raises(ValueError, match="重叠"):
        attention_supervision_loss(
            torch.zeros(1, 1, 5, 5),
            torch.from_numpy(target[None, None]),
            torch.from_numpy(target[None, None]),
        )
    boxes = np.array([[100, 100, 0, 1, 1, 1, 0], [0, 0, 0, 0.01, 0.01, 1, 0]])
    raster, fallback = box_masks(boxes)
    assert raster[0].sum() == 0
    assert raster[1].sum() == 1 and fallback == 1


def test_clockwise_native_box_yaw() -> None:
    boxes = np.array([[0, 0, 0, 12, 1, 2, np.pi / 4]])
    masks, _ = box_masks(boxes)
    xy = BEVGrid().lidar_centers().numpy()[..., :2]
    points = xy[masks[0]]
    # 顺时针长轴落在 y=-x，若使用新版逆时针公式此测试必失败。
    assert (points[:, 0] * points[:, 1]).mean() < 0


def test_calibrated_ramp_and_gt_independent_position() -> None:
    row = dict(
        lidar2ego_rotation=[0.7071067811865476, 0, 0, -0.7071067811865476],
        lidar2ego_translation=[1, 0, 1.8],
    )
    grid, _, lidar = metric_grid(row)
    native = BEVGrid().lidar_centers()
    ramp = native[..., :2].permute(2, 0, 1)[None]
    projected = project_attention(ramp, grid)[0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(projected, lidar[..., :2], atol=1e-10)
    before = geometry(row)
    after = geometry({**row, "target_mask": np.ones((150, 150)), "text": "changed"})
    assert torch.equal(before, after)
    # 编码同一物理 Ego z=0 平面，x/30,y/15 严格对应采样网格。
    sampled = F.grid_sample(before.double(), grid, align_corners=False)
    assert abs(float(sampled[0, 0, 100, 200]) - 0.075 / 30) < 1e-7
    assert abs(float(sampled[0, 1, 100, 200]) - 0.075 / 15) < 3e-7


def test_split_and_training_gt_firewall(tmp_path) -> None:
    frames = [
        dict(
            sample_token="a", scene_token="s1", scene_name="scene-0757", split="train"
        ),
        dict(
            sample_token="b",
            scene_token="s2",
            scene_name="scene-0796",
            split="internal_dev",
        ),
    ]
    assert_splits(frames)
    for changed in (
        {"sample_token": "a"},
        {"scene_token": "s1"},
        {"scene_name": "scene-0655"},
    ):
        with pytest.raises(ValueError):
            assert_splits([frames[0], {**frames[1], **changed}])
    row = dict(split="internal_dev")
    with pytest.raises(ValueError, match="训练禁止"):
        load_supervision(row, tmp_path, training=True)
    with pytest.raises(ValueError, match="训练不能"):
        training_frame(row)


def test_schedule_is_train_only_and_repeatable() -> None:
    rows = [
        dict(
            split=split,
            task=task,
            target_present=positive,
            instruction_id=f"{split}_{task}_{positive}",
        )
        for split in ("train", "internal_dev")
        for task in ("map", "object")
        for positive in (True, False)
    ]
    first, second = make_schedule(rows, 16), make_schedule(rows, 16)
    assert first == second
    assert all(r["split"] == "train" for r in first)
    assert sum(not r["target_present"] for r in first if r["task"] == "object") == 4


def test_inference_boundary_has_no_gt_inputs() -> None:
    assert list(inspect.signature(paired_forward).parameters) == [
        "model",
        "adapter",
        "bev",
        "position",
        "correct_text",
        "wrong_text",
    ]
    source = inspect.getsource(frame_input)
    assert "source_file" not in source
    assert "label_file" not in source
    assert "training_frame" not in inspect.getsource(paired_forward)


def test_native_dual_head_cache_boundary_restores_on_failure() -> None:
    class Stub(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = SimpleNamespace(get_bev_features=lambda: "original")

        def forward(self, *_args):
            # 两个 stub 分量共用真实接口返回的同一个序列对象。
            sequence = self.transformer.get_bev_features()
            assert sequence.shape == (1, 22500, 256)
            raise RuntimeError("unit_contract_failure")

    head = Stub()
    model = nn.Module()
    model.pts_bbox_head = head
    original = head.transformer.get_bev_features
    with pytest.raises(RuntimeError, match="unit_contract_failure"):
        dual_head(model, torch.zeros(1, 256, 150, 150))
    assert head.transformer.get_bev_features is original


def test_blocked_pilot_cannot_initialize_model(monkeypatch, tmp_path) -> None:
    from scripts.stage3 import lg_bev_train as training

    monkeypatch.setattr(training, "verify_protocol", lambda _out: {})
    monkeypatch.setattr(
        training,
        "read",
        lambda path: (
            {"status": "PASS"}
            if path.name == "smoke_test.json"
            else {"pilot_allowed": False}
        ),
    )
    monkeypatch.setattr(
        training, "load_native", lambda: pytest.fail("禁止加载 GPU 模型")
    )
    with pytest.raises(ValueError, match="禁止 pilot"):
        training.train(tmp_path, "pilot")
    assert not (tmp_path / "pilot").exists()


def test_blocked_evaluation_cannot_initialize_model(monkeypatch, tmp_path) -> None:
    from scripts.stage3 import lg_bev_evaluate as evaluation

    monkeypatch.setattr(evaluation, "verify_protocol", lambda _out: {})
    monkeypatch.setattr(evaluation, "read", lambda _path: {"pilot_allowed": False})
    monkeypatch.setattr(
        evaluation, "load_native", lambda: pytest.fail("禁止加载 GPU 模型")
    )
    with pytest.raises(ValueError, match="禁止性能对照"):
        evaluation.evaluate(tmp_path)

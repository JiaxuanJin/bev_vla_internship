"""独立 Attention 空间与固定计分协议，不依赖 GPU 或封存验证集。"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from scripts.stage3.lg_bev_supervision_space import (
    AttentionSupervisionSpace,
    bilinear_adjoint_check,
)
from scripts.stage3.lg_bev_supervision_metrics import (
    average_precision,
    match_boxes,
    mask_counts,
    paired_inference,
)


def calibration() -> dict:
    return dict(
        lidar2ego_rotation=[2**-0.5, 0, 0, -(2**-0.5)],
        lidar2ego_translation=[1, 0, 1.8],
    )


def test_mapping_is_independent_of_instance_and_prediction() -> None:
    base = calibration()
    a = AttentionSupervisionSpace.from_calibration("map", base)
    b = AttentionSupervisionSpace.from_calibration(
        "map",
        {
            **base,
            "target_mask": np.ones((200, 400)),
            "instance_id": "changed",
            "prediction": 100,
        },
    )
    assert torch.equal(a.sampling_grid, b.sampling_grid)
    assert a.project(torch.randn(1, 1, 150, 150)).shape == (1, 1, 200, 400)


def test_object_attention_remains_native_tensor() -> None:
    space = AttentionSupervisionSpace.from_calibration("object", {})
    value = torch.randn(1, 1, 150, 150, requires_grad=True)
    assert space.project(value) is value
    with pytest.raises(ValueError, match="未知监督空间"):
        AttentionSupervisionSpace.from_calibration("other", {})


def test_bilinear_adjoint_is_metric_and_preserves_mass() -> None:
    space = AttentionSupervisionSpace.from_calibration("map", calibration())
    for pixel in [(10, 20), (100, 200), (190, 380)]:
        report = bilinear_adjoint_check(space, pixel)
        assert report["pass_check"]
        assert report["gradient_support_cells"] <= 4
        assert abs(report["gradient_mass"] - 1) < 1e-10


def test_invalid_native_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="原生 BEV"):
        AttentionSupervisionSpace.from_calibration("object", {}).project(
            torch.zeros(1, 1, 200, 400)
        )


def test_average_precision_retains_empty_prompt_false_positive() -> None:
    # 最高分来自空目标指令；不能在聚合前删除。
    result = average_precision([(0.99, False), (0.8, True), (0.7, True)], 2)
    assert result["ap"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert result["fp"] == 1 and result["fn"] == 0
    empty = average_precision([(0.8, False)], 0)
    assert empty["ap"] is None and empty["fp"] == 1


def test_detection_matching_is_unique_and_2m_xy() -> None:
    gt = np.array([[0, 0, 0], [5, 0, 0]])
    boxes = np.array([[0, 0, 0], [0.1, 0, 0], [7.01, 0, 0]])
    result = match_boxes(
        gt, np.arange(2), boxes, np.array([0.9, 0.8, 0.7]), np.arange(3)
    )
    assert result == [(0.9, True), (0.8, False), (0.7, False)]


def test_background_fp_does_not_include_other_foreground() -> None:
    active = np.ones((3, 3), bool)
    target = np.zeros_like(active)
    distractor = np.zeros_like(active)
    target[0, 0] = True
    distractor[0, 1] = True
    background = np.zeros_like(active)
    background[2] = True
    counts = mask_counts(active, target, distractor, background)
    assert counts["background_fp"] == 3
    assert counts["distractor_fp"] == 1
    assert counts["union"] == 9


def test_inference_has_no_gt_or_warp_inputs() -> None:
    assert list(inspect.signature(paired_inference).parameters) == [
        "model",
        "adapter",
        "bev",
        "position",
        "correct_text",
        "wrong_text",
    ]
    assert "cropper" not in inspect.getsource(paired_inference)
    assert "calibrated_cropper" not in inspect.getsource(paired_inference)

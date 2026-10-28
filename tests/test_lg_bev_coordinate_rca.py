"""运行时坐标契约、标定桥接、异常恢复和独立 GT 的小规模测试。"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn

from scripts.stage3.lg_bev_coordinate_core import (
    calibrated_cropper,
    calibrated_sample,
    native_to_segment,
    physical_ramp,
    ramp_error,
    require_runtime_head,
    segment_to_native,
)
from scripts.stage3.lg_bev_coordinate_independent_gt import independent_line_mask


@pytest.fixture
def calibration() -> dict:
    """含非零平移的顺时针 LiDAR→Ego 旋转。"""
    return dict(
        lidar2ego_rotation=[2**-0.5, 0, 0, -(2**-0.5)],
        lidar2ego_translation=[0.985793, 0, 1.84019],
    )


def test_active_transform_is_exactly_invertible() -> None:
    value = torch.arange(3 * 5 * 7).reshape(1, 3, 5, 7)
    assert torch.equal(segment_to_native(native_to_segment(value)), value)
    inactive = torch.flip(torch.rot90(value, -1, [2, 3]), [3])
    assert not torch.equal(inactive, native_to_segment(value))


def test_bridge_real_metric_ramp_and_gt_independence(calibration: dict) -> None:
    ramp = physical_ramp(calibration, channels=3)
    sampled = calibrated_sample(native_to_segment(ramp), calibration)
    report = ramp_error(sampled, calibration)
    assert report["max_lidar_xy_m"] < 1e-4
    assert report["max_ego_xyz_m"] < 1e-4
    different_gt = {
        **calibration,
        "target_mask": np.ones((200, 400)),
        "text": "changed",
    }
    assert torch.equal(
        sampled, calibrated_sample(native_to_segment(ramp), different_gt)
    )


def test_bridge_tilted_calibration_uses_ground_plane() -> None:
    calibration = dict(
        lidar2ego_rotation=[
            0.706749235646644,
            -0.015300993788500868,
            0.01739745181256607,
            -0.7070846669051719,
        ],
        lidar2ego_translation=[0.985793, 0, 1.84019],
    )
    ramp = physical_ramp(calibration, channels=3)
    report = ramp_error(
        calibrated_sample(native_to_segment(ramp), calibration), calibration
    )
    assert report["max_ego_xyz_m"] < 1e-4


def test_bridge_gradient_and_no_trainable_transform(calibration: dict) -> None:
    value = torch.randn(1, 3, 150, 150, requires_grad=True)
    sample = calibrated_sample(native_to_segment(value), calibration)
    sample.square().mean().backward()
    assert value.grad is not None
    assert torch.isfinite(value.grad).all() and value.grad.norm() > 0
    assert "GT" not in inspect.signature(calibrated_sample).parameters


def test_unknown_runtime_head_is_rejected() -> None:
    class OtherHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.seg_decoder = nn.Identity()

    with pytest.raises(ValueError, match="未知运行时 Head"):
        require_runtime_head(OtherHead())


def test_cropper_context_restores_after_exception(
    monkeypatch, calibration: dict
) -> None:
    from scripts.stage3 import lg_bev_coordinate_core as core

    monkeypatch.setattr(core, "require_runtime_head", lambda _head: {})
    head = nn.Module()
    head.feat_cropper = nn.Identity()
    original = head.feat_cropper.forward
    with pytest.raises(RuntimeError, match="probe"):
        with calibrated_cropper(head, calibration):
            assert "forward" in head.feat_cropper.__dict__
            raise RuntimeError("probe")
    assert head.feat_cropper.forward == original
    assert "forward" not in head.feat_cropper.__dict__


def test_cropper_context_refuses_nested_patch(monkeypatch, calibration: dict) -> None:
    from scripts.stage3 import lg_bev_coordinate_core as core

    monkeypatch.setattr(core, "require_runtime_head", lambda _head: {})
    head = nn.Module()
    head.feat_cropper = nn.Identity()
    with calibrated_cropper(head, calibration):
        with pytest.raises(RuntimeError, match="拒绝覆盖或嵌套"):
            with calibrated_cropper(head, calibration):
                pass
    assert "forward" not in head.feat_cropper.__dict__


def test_independent_raw_map_axis_and_pixel_centres() -> None:
    frame = dict(
        ego2global_rotation=[1, 0, 0, 0],
        ego2global_translation=[100, 200, 0],
        instance_ids=["map/lane_divider/record"],
    )
    raw = dict(
        node=[dict(token="a", x=106, y=203), dict(token="b", x=109, y=203)],
        line=[dict(token="line", node_tokens=["a", "b"])],
        lane_divider=[dict(token="record", line_token="line")],
        road_divider=[],
    )
    mask, records = independent_line_mask(frame, raw)
    assert mask[120, 250]
    assert not mask[80, 250]
    assert records[0]["max_pixel_center_quantization_m"] <= np.sqrt(2) * 0.075 + 1e-8

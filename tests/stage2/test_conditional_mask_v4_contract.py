"""核验物理采样和重叠实例负例，CPU 检查不代替真实 GPU 实验。"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.conditional_mask_v4_model import mask_loss
from scripts.stage2.spatial_annotations import metric_grid


def test_metric_sampling_preserves_lidar_xy_under_rotation() -> None:
    """转动和平移后线性坐标斜坡仍等于真实 LiDAR 像素坐标。"""
    angle = 0.61
    info = dict(
        lidar2ego_rotation=[np.cos(angle / 2), 0, 0, np.sin(angle / 2)],
        lidar2ego_translation=[0.98, -0.2, 1.8],
    )
    grid, ego, lidar = metric_grid(info)
    native = BEVGrid().lidar_centers(dtype=torch.float64)[..., :2]
    ramp = native.permute(2, 0, 1)[None]
    sampled = F.grid_sample(ramp, grid, align_corners=False)[0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(sampled, lidar[..., :2], atol=1e-10)


def test_overlapping_record_remains_positive_only() -> None:
    """同类记录重叠不能同时把目标像素当成负例训练。"""
    target = torch.zeros((4, 5), dtype=torch.bool)
    target[1, 1:3] = True
    other = target.clone()
    other[3, 3] = True
    logits = torch.zeros((4, 5), requires_grad=True)
    loss, parts = mask_loss(logits, target, other)
    loss.backward()
    assert (logits.grad[target] < 0).all()
    assert logits.grad[3, 3] > 0

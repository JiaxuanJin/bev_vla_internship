"""新物理投影的坐标与梯度契约；真实 GPU 结果由实验日志提供。"""

import numpy as np
import pytest
import torch

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.spatial_annotations import metric_grid, project_attention


@pytest.mark.parametrize("quaternion", [[1., 0., 0., 0.], [.70710678, 0., 0., -.70710678]])
def test_calibrated_grid_preserves_metric_coordinates(quaternion):
    info = {"lidar2ego_rotation": quaternion, "lidar2ego_translation": [1., .2, 1.8]}
    grid, _, lidar = metric_grid(info)
    native = BEVGrid().lidar_centers()[..., :2].permute(2, 0, 1)[None]
    actual = project_attention(native, grid)[0].permute(1, 2, 0).numpy()
    np.testing.assert_allclose(actual, lidar[..., :2], atol=1e-10)


def test_physical_sampling_keeps_attention_gradient():
    grid, _, _ = metric_grid({"lidar2ego_rotation": [1., 0., 0., 0.],
                              "lidar2ego_translation": [1., 0., 1.8]})
    logits = torch.zeros(1, 1, 150, 150, requires_grad=True)
    project_attention(logits, grid).square().add(project_attention(logits, grid)).mean().backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.norm() > 0


def test_projection_rejects_wrong_native_shape():
    with pytest.raises(ValueError, match="原生"):
        project_attention(torch.zeros(1, 1, 149, 150), torch.zeros(1, 200, 400, 2))

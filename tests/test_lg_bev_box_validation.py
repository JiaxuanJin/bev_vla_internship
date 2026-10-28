"""真实FP32边缘反例、闭边界、亚像素协议和严格拒绝的回归。"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from shapely.geometry import Point, Polygon

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage3.lg_bev_data import box_masks, verify_box_corners

# scene-0447/frame8965eb26cbfe4a42b0c43b2493ba0963，原生框索引6。
# 原始annotation转换得到的FP32值；没有调整几何以迎合验证器。
REAL_BOX = [
    15.507814407348633,
    -20.711374282836914,
    -3.2965264320373535,
    3.625999927520752,
    16.341999053955078,
    4.423999786376953,
    -3.0154051780700684,
    0.0,
    0.0,
]


def native(values, dtype=torch.float32):
    """使用实际安装的0.17.1 corners；显式保留测试所需张量精度。"""
    from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes

    tensor = torch.tensor(values, dtype=dtype).reshape(-1, 9)
    boxes = LiDARInstance3DBoxes(tensor, box_dim=9)
    boxes.tensor = tensor
    return boxes


def test_real_failure_is_roundoff_not_contains_vs_covers():
    boxes = native([REAL_BOX])
    generated, _ = box_masks(boxes.tensor.numpy())
    polygon = Polygon(boxes.corners[0, [0, 3, 7, 4], :2].numpy())
    point = Point(BEVGrid().lidar_centers()[45, 100, :2].numpy())
    assert polygon.contains(point) and polygon.covers(point)
    assert not polygon.touches(point)
    assert not generated[0, 45, 100]
    assert 0 < polygon.boundary.distance(point) < 3e-7
    assert verify_box_corners(boxes, generated) == 1


@pytest.mark.parametrize("pixel", [(45, 100), (0, 0), (45, 98)])
def test_even_one_corrupt_pixel_is_rejected(pixel):
    boxes = native([REAL_BOX])
    generated, _ = box_masks(boxes.tensor.numpy())
    generated[(0, *pixel)] ^= True
    with pytest.raises(AssertionError, match="box=0"):
        verify_box_corners(boxes, generated)


def test_exact_boundary_is_included_by_closed_polygon():
    xy = BEVGrid().lidar_centers()[75, 75, :2].numpy()
    boxes = native([[0, 0, 0, 2 * xy[0], 2 * xy[1], 1, 0, 0, 0]], torch.float64)
    generated, _ = box_masks(boxes.tensor.numpy())
    polygon = Polygon(boxes.corners[0, [0, 3, 7, 4], :2].numpy())
    point = Point(xy)
    assert polygon.touches(point) and polygon.covers(point)
    assert not polygon.contains(point) and generated[0, 75, 75]
    assert verify_box_corners(boxes, generated) == 1


@pytest.mark.parametrize("center", [(0, 0), (100, 100), (51.19, -51.19)])
def test_subpixel_fallback_and_out_of_grid_are_checked(center):
    boxes = native([[*center, 0, 0.001, 0.001, 1, 0, 0, 0]])
    generated, _ = box_masks(boxes.tensor.numpy())
    assert verify_box_corners(boxes, generated) == 1
    generated[0, 0, 0] ^= True
    with pytest.raises(AssertionError):
        verify_box_corners(boxes, generated)


def test_wrong_native_yaw_or_corners_are_rejected():
    boxes = native([REAL_BOX])
    generated, _ = box_masks(boxes.tensor.numpy())
    wrong = boxes.corners.clone()
    wrong[:, :, 0] += 0.01

    class ChangedCorners(SimpleNamespace):
        def __len__(self):
            return len(self.tensor)

    with pytest.raises(AssertionError, match="顺时针yaw"):
        verify_box_corners(
            ChangedCorners(tensor=boxes.tensor, corners=wrong), generated
        )


def test_empty_boxes_and_mask_shape_are_checked():
    boxes = native([])
    assert verify_box_corners(boxes, np.zeros((0, 150, 150), dtype=bool)) == 0
    with pytest.raises(ValueError, match="150"):
        verify_box_corners(boxes, np.zeros((1, 150, 150), dtype=bool))

"""BEV 张量行列与 LiDAR 米制中心的显式契约；不推断地图 GT 配准。"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class BEVGrid:
    height: int = 150
    width: int = 150
    channels: int = 256
    point_cloud_range: tuple[float, ...] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)

    def __post_init__(self):
        if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0
               for n in (self.height, self.width, self.channels)):
            raise ValueError("BEV 的 H/W/C 必须是正整数")
        p = self.point_cloud_range
        if len(p) != 6 or not all(math.isfinite(v) for v in p):
            raise ValueError("point_cloud_range 必须是六个有限数")
        if any(p[i] >= p[i + 3] for i in range(3)):
            raise ValueError("point_cloud_range 上下界错误")

    def to_map(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 3 or tuple(sequence.shape[1:]) != (
            self.height * self.width, self.channels
        ) or sequence.shape[0] <= 0:
            raise ValueError("共享 BEV 必须为 [B,H*W,C]，禁止自动转置或缩放")
        return sequence.reshape(-1, self.height, self.width, self.channels).permute(0, 3, 1, 2)

    def to_sequence(self, feature_map: Tensor) -> Tensor:
        if feature_map.ndim != 4 or tuple(feature_map.shape[1:]) != (
            self.channels, self.height, self.width
        ) or feature_map.shape[0] <= 0:
            raise ValueError("BEV 地图必须为 [B,C,H,W]，禁止自动转置或缩放")
        return feature_map.permute(0, 2, 3, 1).reshape(-1, self.height * self.width, self.channels)

    def lidar_centers(self, *, device="cpu", dtype=torch.float64) -> Tensor:
        """返回 [H,W,3]，列增大对应 LiDAR x 增大，行增大对应 LiDAR y 增大。

        flatten index = row * W + col；采用栅格中心，z=0 为方向诊断平面。
        """
        xmin, ymin, _, xmax, ymax, _ = self.point_cloud_range
        x = xmin + (torch.arange(self.width, device=device, dtype=dtype) + 0.5) * (xmax - xmin) / self.width
        y = ymin + (torch.arange(self.height, device=device, dtype=dtype) + 0.5) * (ymax - ymin) / self.height
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1)

    def ego_centers(self, rotation: Tensor, translation: Tensor) -> Tensor:
        """必须显式提供 LiDAR→Ego 标定，不能把张量行列直接叫前后左右。"""
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("LiDAR→Ego 的 R/t 形状错误")
        if not rotation.is_floating_point() or translation.dtype != rotation.dtype or translation.device != rotation.device:
            raise ValueError("R/t 需要一致的浮点类型和设备")
        if not torch.isfinite(rotation).all() or not torch.isfinite(translation).all():
            raise ValueError("R/t 包含 NaN/Inf")
        eye = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
        if not torch.allclose(rotation.T @ rotation, eye, atol=1e-5) or not torch.isclose(
            torch.linalg.det(rotation), rotation.new_tensor(1.0), atol=1e-5
        ):
            raise ValueError("R 必须是右手系正交旋转矩阵")
        return self.lidar_centers(device=rotation.device, dtype=rotation.dtype) @ rotation.T + translation

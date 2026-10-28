"""坐标根因实验：运行时契约、原始 GT 与无参数标定桥接。"""

from __future__ import annotations

from contextlib import contextmanager
import inspect
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.spatial_annotations import DATA, metric_grid
from scripts.stage3.lg_bev_common import DEFAULT_OUT, ROOT, read, sha

COORD_OUT = ROOT / "results/lg_bev_coordinate_rca_20261027"


def legal_frames() -> list[dict]:
    """只允许本次 LG-BEV train 场景进入根因定位。"""
    frames = read(DEFAULT_OUT / "manifest.json")["frames"]
    return [f for f in frames if f["split"] == "train"]


def require_runtime_head(head: nn.Module) -> dict:
    """从实际实例解析 Head/decoder 类，拒绝依据未加载的同名文件猜测。"""
    head_file = Path(inspect.getfile(type(head))).resolve()
    decoder_file = Path(inspect.getfile(type(head.seg_decoder))).resolve()
    source = inspect.getsource(type(head).forward)
    if type(head).__name__ != "BEVFormerHead" or head_file.name != "bevformer_head.py":
        raise ValueError("未知运行时 Head，必须先独立审计，禁止套用已知轴向")
    if (
        ".permute(2, 3, 1, 0)" not in source
        or "torch.flip(seg_bev, dims=[2])" not in source
    ):
        raise ValueError("实际 Head 空间变换不符合已审计契约")
    return dict(
        head_class=f"{type(head).__module__}.{type(head).__name__}",
        head_file=str(head_file),
        head_source_sha256=sha(head_file),
        decoder_class=f"{type(head.seg_decoder).__module__}.{type(head.seg_decoder).__name__}",
        decoder_file=str(decoder_file),
        decoder_source_sha256=sha(decoder_file),
        active_transform="native.transpose(2,3).flip(2)",
        previously_audited_inactive_head=str(
            head_file.with_name("bevformer_seg_head.py")
        ),
        forward_source=source,
        decoder_forward_source=inspect.getsource(type(head.seg_decoder).forward),
        explicit_decoder_coordinate_ops=False,
        decoder_operations=[
            "conv",
            "BatchNorm",
            "ReLU",
            "ResNet",
            "bilinear upsample align_corners=True",
            "skip concatenation",
        ],
        learned_compensation="Not excluded by source inspection; assessed separately on training-only GT",
    )


def native_to_segment(bev: Tensor) -> Tensor:
    """真实 BEVFormerHead 的固定转置和竖直翻转。"""
    return bev.transpose(2, 3).flip(2)


def segment_to_native(segment: Tensor) -> Tensor:
    """精确求逆，不使用 GT 或开发拟合的偏移。"""
    return segment.flip(2).transpose(2, 3)


def calibrated_sample(segment: Tensor, calibration: dict) -> Tensor:
    """原生分割变换求逆后，按逐帧 Ego z=0 标定采样原生 BEV。"""
    native = segment_to_native(segment)
    grid, _, _ = metric_grid(calibration)
    return F.grid_sample(
        native,
        grid.to(native),
        align_corners=False,
        mode="bilinear",
        padding_mode="zeros",
    )


@contextmanager
def calibrated_cropper(head: nn.Module, calibration: dict) -> Iterator[None]:
    """仅当前调用临时替换 cropper.forward；不修改作者文件、参数或 buffer。"""
    require_runtime_head(head)
    cropper = head.feat_cropper
    if "forward" in cropper.__dict__:
        raise RuntimeError("cropper 已有实例级 forward，拒绝覆盖或嵌套")
    original = cropper.forward
    cropper.forward = lambda value: calibrated_sample(value, calibration)
    try:
        yield
    finally:
        del cropper.forward
        assert cropper.forward == original


def physical_ramp(
    calibration: dict, *, device: str = "cpu", channels: int = 256
) -> Tensor:
    """在 Ego z=0 平面构造原生 LiDAR XYZ 坐标，匹配标定投影的平面约束。"""
    from pyquaternion import Quaternion

    xyz = BEVGrid().lidar_centers(device=device, dtype=torch.float32)
    rotation = xyz.new_tensor(
        Quaternion(calibration["lidar2ego_rotation"]).rotation_matrix
    )
    shift = xyz.new_tensor(calibration["lidar2ego_translation"])
    xyz[..., 2] = (
        -shift[2] - xyz[..., 0] * rotation[2, 0] - xyz[..., 1] * rotation[2, 1]
    ) / rotation[2, 2]
    ramp = xyz.new_zeros((1, channels, 150, 150))
    ramp[0, :3] = xyz.permute(2, 0, 1)
    return ramp


def ramp_error(sampled: Tensor, row: dict) -> dict:
    """相同 GT 像素中心比较 LiDAR XY 与 Ego XYZ，避免混用地面高度。"""
    from pyquaternion import Quaternion

    _, ego, lidar = metric_grid(row)
    actual = sampled[0, :3].detach().cpu().permute(1, 2, 0).double().numpy()
    rotation = Quaternion(row["lidar2ego_rotation"]).rotation_matrix
    actual_ego = actual @ rotation.T + np.asarray(row["lidar2ego_translation"])
    xy_error = np.linalg.norm(actual[..., :2] - lidar[..., :2], axis=-1)
    ego_error = np.linalg.norm(actual_ego - ego, axis=-1)
    return dict(
        mean_lidar_xy_m=float(xy_error.mean()),
        max_lidar_xy_m=float(xy_error.max()),
        mean_ego_xyz_m=float(ego_error.mean()),
        max_ego_xyz_m=float(ego_error.max()),
    )


def raw_map_gt(
    frame: dict, info: dict, mapper: Any, raster: Any
) -> tuple[np.ndarray, dict]:
    """从 G 盘原始地图重建 GT，并用独立坐标线验证 raster 行列方向。"""
    from scripts.stage2.spatial_annotations import tagged_vectors

    vectors, tagged = tagged_vectors(mapper, info, frame["location"])
    semantic, *_ = raster.preprocess_map(vectors, (30.0, 60.0), (200, 400), 3, 5, 36)
    gt = (semantic.numpy() * np.arange(1, 4)[:, None, None]).sum(0).astype(np.uint8)
    path = DATA / "maps/expansion" / f"{frame['location']}.json"
    line = dict(pts=np.array([[6.0, 3.0], [9.0, 3.0]]), pts_num=2, type=0)
    probe, *_ = raster.preprocess_map([line], (30.0, 60.0), (200, 400), 3, 1, 36)
    rows, cols = np.nonzero(probe[0].numpy())
    if not (
        rows.min() == rows.max() == 120 and cols.min() == 240 and cols.max() == 260
    ):
        raise ValueError("原始 raster 坐标方向或离散化发生变化")
    return gt, dict(
        raw_map_path=str(path),
        raw_map_sha256=sha(path),
        record_ids=sorted({identity for _, identity in tagged}),
        synthetic_line=dict(ego_xy=[[6, 3], [9, 3]], row=120, col_min=240, col_max=260),
        raster_vertex_quantization="int32 truncation; <=0.15m vertex quantization; supervision uses pixel centres",
    )

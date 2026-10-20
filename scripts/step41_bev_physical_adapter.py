#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 41：BEV LiDAR->Ego 物理坐标适配及可复现的坐标场验证。

本脚本不更改原项目文件、不训练模型、不对 Val 做模型推理。
它实现完整 LiDAR->Ego 逆映射、GT 栅格中心采样，以及边界掩码检查。

运行：
    python scripts/step41_bev_physical_adapter.py

用途：后续训练与推理共用的物理坐标适配候选。旧 checkpoint 的 Decoder
不保证与这种输入坐标排列兼容，不能据此声称 mIoU 已恢复。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional

ROOT = Path(__file__).resolve().parents[1]
VAL_PKL = (
    ROOT
    / "outputs/drivelm_check/root_cause/step21_full_val"
    / "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
)
OUTPUT_DIR = ROOT / "outputs/drivelm_check/root_cause/step41_physical_adapter"
NUSCENES_ROOT = ROOT / "BEVFormer-seg/data/nuscenes"


@dataclass(frozen=True)
class GridBounds:
    """表示栅格在物理 X、Y 轴的范围（米）。"""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    height: int
    width: int

    def centers(self, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """返回与 GT OpenCV 栅格行列一致的 (x, y) 像素中心。"""
        xs = self.x_min + (
            torch.arange(self.width, device=device, dtype=dtype) + 0.5
        ) * ((self.x_max - self.x_min) / self.width)
        ys = self.y_min + (
            torch.arange(self.height, device=device, dtype=dtype) + 0.5
        ) * ((self.y_max - self.y_min) / self.height)
        y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
        return x_grid, y_grid


def build_lidar_to_ego_grid(
    lidar_to_ego: torch.Tensor,
    source_bounds: GridBounds,
    target_bounds: GridBounds,
    *,
    ego_plane_z: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造将 Ego 目标栅格反投影到 LiDAR BEV 的 grid_sample 网格。

    Args:
        lidar_to_ego: [B, 4, 4]，列向量约定 p_ego = R p_lidar + t。
        source_bounds: 输入 BEV 物理尺寸与栅格维度，W 对应 X、H 对应 Y。
        target_bounds: GT 物理尺寸与栅格维度，W 对应 Ego X、H 对应 Ego Y。
        ego_plane_z: 采样的 Ego 平面高度（米），默认 0。

    Returns:
        grid: [B, target_H, target_W, 2]，符合 align_corners=True。
        valid: [B, target_H, target_W]，源 BEV 平面内的有效范围。

    Raises:
        ValueError: 外参或栅格维度不正确。
    """
    if lidar_to_ego.ndim != 3 or lidar_to_ego.shape[-2:] != (4, 4):
        raise ValueError("lidar_to_ego 必须为 [B, 4, 4]")
    if source_bounds.height < 2 or source_bounds.width < 2:
        raise ValueError("source_bounds 的高宽至少为 2")
    if target_bounds.height < 2 or target_bounds.width < 2:
        raise ValueError("target_bounds 的高宽至少为 2")

    dtype, device = lidar_to_ego.dtype, lidar_to_ego.device
    x_ego, y_ego = target_bounds.centers(device=device, dtype=dtype)
    ego_points = torch.stack(
        (x_ego, y_ego, torch.full_like(x_ego, ego_plane_z)), dim=-1
    )

    rotation = lidar_to_ego[:, :3, :3]
    translation = lidar_to_ego[:, :3, 3]
    # p_lidar = R.T @ (p_ego - t)：真实外参的逆变换。
    source_xyz = torch.einsum(
        "bij,bhwj->bhwi",
        rotation.transpose(1, 2),
        ego_points.unsqueeze(0) - translation[:, None, None, :],
    )
    x_lidar, y_lidar = source_xyz[..., 0], source_xyz[..., 1]

    # align_corners=True 必须按照源特征的首尾像素中心归一化，避免半像素偏移。
    dx = (source_bounds.x_max - source_bounds.x_min) / source_bounds.width
    dy = (source_bounds.y_max - source_bounds.y_min) / source_bounds.height
    first_x = source_bounds.x_min + dx / 2
    first_y = source_bounds.y_min + dy / 2
    last_x = source_bounds.x_max - dx / 2
    last_y = source_bounds.y_max - dy / 2
    grid_x = (2 * (x_lidar - first_x) / (last_x - first_x)) - 1
    grid_y = (2 * (y_lidar - first_y) / (last_y - first_y)) - 1
    grid = torch.stack((grid_x, grid_y), dim=-1)
    valid = (grid_x >= -1) & (grid_x <= 1) & (grid_y >= -1) & (grid_y <= 1)
    return grid, valid


def sample_lidar_bev_into_ego(
    bev_features: torch.Tensor,
    lidar_to_ego: torch.Tensor,
    source_bounds: GridBounds,
    target_bounds: GridBounds,
) -> tuple[torch.Tensor, torch.Tensor]:
    """保持梯度的 BEV 物理重采样，可用于以后统一训练和推理。"""
    if bev_features.ndim != 4:
        raise ValueError("bev_features 应为 [B, C, H, W]")
    if bev_features.shape[0] != lidar_to_ego.shape[0]:
        raise ValueError("batch size 不一致")
    if tuple(bev_features.shape[-2:]) != (
        source_bounds.height,
        source_bounds.width,
    ):
        raise ValueError("BEV 高宽与物理范围定义不一致")

    grid, valid = build_lidar_to_ego_grid(
        lidar_to_ego.to(device=bev_features.device, dtype=bev_features.dtype),
        source_bounds,
        target_bounds,
    )
    sampled = functional.grid_sample(
        bev_features, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return sampled, valid


def synthetic_ramp_test(
    lidar_to_ego: torch.Tensor,
    source_bounds: GridBounds,
    target_bounds: GridBounds,
) -> dict[str, Any]:
    """传递已知的 LiDAR XY 斜坡并检查输出与解析逆变换一致。"""
    source_x, source_y = source_bounds.centers(
        device=lidar_to_ego.device, dtype=lidar_to_ego.dtype
    )
    ramp = torch.stack((source_x, source_y), dim=0).unsqueeze(0)
    ramp = ramp.expand(lidar_to_ego.shape[0], -1, -1, -1).contiguous()
    sampled, valid = sample_lidar_bev_into_ego(
        ramp, lidar_to_ego, source_bounds, target_bounds
    )
    target_x, target_y = target_bounds.centers(
        device=lidar_to_ego.device, dtype=lidar_to_ego.dtype
    )
    target_xyz = torch.stack(
        (target_x, target_y, torch.zeros_like(target_x)), dim=-1
    )
    rotation = lidar_to_ego[:, :3, :3]
    shift = lidar_to_ego[:, :3, 3]
    expected = torch.einsum(
        "bij,bhwj->bhwi",
        rotation.transpose(1, 2),
        target_xyz[None] - shift[:, None, None, :],
    )[..., :2].permute(0, 3, 1, 2)
    diff = (sampled - expected).abs()
    mask = valid[:, None].expand_as(diff)
    if not bool(valid.any()):
        raise RuntimeError("所有目标点都在 BEV 范围外，请检查外参或配置")
    max_abs = float(diff[mask].max())
    mean_abs = float(diff[mask].mean())
    if max_abs >= 0.0002:
        raise AssertionError(f"物理斜坡验证失败：最大误差 {max_abs:.6f}m")
    return {
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.numel()),
        "mean_absolute_error_m": mean_abs,
        "max_absolute_error_m": max_abs,
        "pass": True,
    }


def synthetic_transform(angle_deg: float, tx: float, ty: float) -> torch.Tensor:
    """建立用于单元测试的真实 2D 旋转、平移四维齐次矩阵。"""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    matrix = torch.eye(4, dtype=torch.float64)[None]
    matrix[0, :2, :2] = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
    matrix[0, 0, 3], matrix[0, 1, 3] = tx, ty
    return matrix


def real_nuscenes_calibration(index: list[int], infos: list[dict]) -> list[tuple[int, str, torch.Tensor]]:
    """通过真实 nuScenes LIDAR_TOP 获取 LiDAR->Ego 变换。"""
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion

    if not NUSCENES_ROOT.is_dir():
        raise FileNotFoundError(f"nuScenes 数据目录不存在：{NUSCENES_ROOT}")
    # 延迟加载，避免在本脚本不需要真实外参时加载整套官方记录。
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=str(NUSCENES_ROOT), verbose=False
    )
    result = []
    for frame_index in index:
        token = infos[frame_index]["token"]
        sample = nusc.get("sample", token)
        lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cal = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        rotation = torch.tensor(
            Quaternion(cal["rotation"]).rotation_matrix, dtype=torch.float64
        )
        translation = torch.tensor(cal["translation"], dtype=torch.float64)
        transform = torch.eye(4, dtype=torch.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        result.append((frame_index, token, transform))
    return result


def main() -> None:
    """执行坐标单元测试与真实外参一致性试验，不触碰权重。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indices", type=int, nargs="+", default=[0, 3000, 3715, 3718, 3719]
    )
    args = parser.parse_args()
    source = GridBounds(-51.2, 51.2, -51.2, 51.2, 150, 150)
    target = GridBounds(-30, 30, -15, 15, 200, 400)

    result: dict[str, Any] = {
        "purpose": "真实 LiDAR->Ego 坐标模块；不是训练完成或 mIoU 修复",
        "source": source.__dict__,
        "gt": target.__dict__,
        "assumptions": [
            "Encoder BEV W=X_lidar、H=Y_lidar、像素中心网格",
            "GT Rasterize W=X_ego、H=Y_ego、像素中心网格",
            "逆外参 p_lidar=R.T(p_ego-t)，Ego 采样平面 z=0",
            "grid_sample align_corners=True，按 source 像素中心归一化",
        ],
        "synthetic": {},
        "real_samples": [],
    }
    for angle in (0.0, -90.0, 30.0):
        t = synthetic_transform(angle, 1.2, -0.4)
        metrics = synthetic_ramp_test(t, source, target)
        result["synthetic"][str(angle)] = metrics
        print(f"[synthetic yaw={angle:+.1f}] max XY error = {metrics['max_absolute_error_m']:.2e} m", flush=True)

    if VAL_PKL.is_file():
        with VAL_PKL.open("rb") as stream:
            infos = sorted(pickle.load(stream)["infos"], key=lambda row: row["timestamp"])
        if any(i < 0 or i >= len(infos) for i in args.indices):
            raise IndexError("请求的 Val 帧索引超出范围")
        for index, token, transform in real_nuscenes_calibration(args.indices, infos):
            metrics = synthetic_ramp_test(transform.unsqueeze(0), source, target)
            result["real_samples"].append({
                "index": index,
                "sample_token": token,
                "lidar_to_ego_matrix": transform.tolist(),
                **metrics,
            })
            print(f"[nuScenes index={index}] max XY error = {metrics['max_absolute_error_m']:.2e} m", flush=True)
    else:
        result["note"] = "缺少完整 Val PKL，仅完成合成单元测试"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "coordinate_contract.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"完成：{out}", flush=True)
    print("下一步需要在 Train/Val 共用调用路径中集成适配器并重新训练；此步不代表修复完成", flush=True)


if __name__ == "__main__":
    main()

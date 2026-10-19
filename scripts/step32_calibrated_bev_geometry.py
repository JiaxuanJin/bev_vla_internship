"""使用 nuScenes 真实 LiDAR->Ego 标定审计 BEV 坐标链路。

此实验只检验物理坐标映射，不修改模型权重，也不计算正式 mIoU。
"""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv import Config
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


def get_variants(field: torch.Tensor) -> dict[str, torch.Tensor]:
    """创建平面上的八种不同朝向，源代码组合单独标记。"""
    return {
        "identity": field,
        "source_transpose": torch.flip(torch.rot90(field, -1, (-2, -1)), (-1,)),
        "rotate_90_clockwise": torch.rot90(field, -1, (-2, -1)),
        "rotate_90_counterclockwise": torch.rot90(field, 1, (-2, -1)),
        "rotate_180": torch.rot90(field, 2, (-2, -1)),
        "flip_rows": torch.flip(field, (-2,)),
        "flip_columns": torch.flip(field, (-1,)),
        "anti_transpose": torch.flip(torch.rot90(field, 1, (-2, -1)), (-1,)),
    }


def make_physical_field(
    height: int,
    width: int,
    pc_range: list[float],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> torch.Tensor:
    """把 Encoder 网格中的 LiDAR 物理坐标变换到 Ego 坐标。"""
    xs = np.linspace(
        pc_range[0] + (pc_range[3] - pc_range[0]) / (2 * width),
        pc_range[3] - (pc_range[3] - pc_range[0]) / (2 * width),
        width,
        dtype=np.float64,
    )
    ys = np.linspace(
        pc_range[1] + (pc_range[4] - pc_range[1]) / (2 * height),
        pc_range[4] - (pc_range[4] - pc_range[1]) / (2 * height),
        height,
        dtype=np.float64,
    )
    y_grid, x_grid = np.meshgrid(ys, xs, indexing="ij")
    lidar = np.stack([x_grid, y_grid, np.zeros_like(x_grid)], axis=0)
    ego = np.einsum("ij,jhw->ihw", rotation, lidar) + translation[:, None, None]
    return torch.from_numpy(ego[:2].astype(np.float32))[None]


def evaluate_mapping(
    slicer: torch.nn.Module,
    field: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    """比较通过真实 Slicer 后的 Ego X/Y 物理坐标。"""
    with torch.no_grad():
        actual = slicer(field)
    diff = (actual - expected).abs()
    return {
        "x_mae_m": float(diff[:, 0].mean().item()),
        "y_mae_m": float(diff[:, 1].mean().item()),
        "xy_mae_m": float(diff.mean().item()),
        "max_abs_error_m": float(diff.max().item()),
    }


def main() -> None:
    """运行标定坐标检查并保存每个样本的实验结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 3000])
    args = parser.parse_args()

    root = Path.home() / "projects" / "bev_vla"
    repo = root / "BEVFormer-seg"
    output_dir = root / "outputs/drivelm_check/root_cause/step32_calibrated_geometry"
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl = (
        root / "outputs/drivelm_check/root_cause/step21_full_val"
        / "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
    )
    if not pkl.is_file():
        raise FileNotFoundError(f"找不到完整 Val PKL: {pkl}")

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_seg_head import (
        BevFeatureSlicer,
    )
    from mmcv import load

    cfg = Config.fromfile(
        str(repo / "projects/configs/bevformer/bevformer_small_seg.py")
    )
    head_cfg = cfg.model.pts_bbox_head
    slicer = BevFeatureSlicer(
        dict(head_cfg.det_grid_conf), dict(head_cfg.map_grid_conf)
    ).eval()
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=str(repo / "data/nuscenes"), verbose=False
    )
    infos = sorted(load(str(pkl))["infos"], key=lambda record: record["timestamp"])

    head_h = int(head_cfg.bev_h)
    head_w = int(head_cfg.bev_w)
    x_start, x_end, dx = head_cfg.map_grid_conf["xbound"]
    y_start, y_end, dy = head_cfg.map_grid_conf["ybound"]
    rows = int(round((y_end - y_start) / dy))
    cols = int(round((x_end - x_start) / dx))
    x_coords = x_start + (torch.arange(cols) + 0.5) * dx
    y_coords = y_start + (torch.arange(rows) + 0.5) * dy
    target_y, target_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    expected = torch.stack([target_x, target_y], dim=0)[None]

    print("==== Step 32 真实 LiDAR->Ego 标定检查 ====", flush=True)
    print("样本索引:", args.indices, flush=True)
    results = []
    for index in args.indices:
        info = infos[index]
        sample = nusc.get("sample", info["token"])
        lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cal = nusc.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
        rotation = Quaternion(cal["rotation"]).rotation_matrix
        translation = np.asarray(cal["translation"], dtype=np.float64)
        physical_field = make_physical_field(
            head_h, head_w, cfg.point_cloud_range, rotation, translation
        )
        by_orientation = {
            name: evaluate_mapping(slicer, variant, expected)
            for name, variant in get_variants(physical_field).items()
        }
        ranking = sorted(
            by_orientation,
            key=lambda name: by_orientation[name]["xy_mae_m"],
        )
        item = {
            "index": index,
            "sample_token": info["token"],
            "scene_token": info["scene_token"],
            "lidar_to_ego_rotation_xy": rotation[:2, :2].tolist(),
            "lidar_to_ego_translation_xyz_m": translation.tolist(),
            "orientation_results": by_orientation,
            "ranked_by_physical_mae": ranking,
        }
        results.append(item)
        print(f"\n样本 {index}: LiDAR->Ego 二维旋转矩阵:", flush=True)
        print(np.array2string(rotation[:2, :2], precision=4), flush=True)
        print("位移 m:", np.round(translation, 4).tolist(), flush=True)
        print(
            "原始转置 XY MAE(m):",
            round(by_orientation["source_transpose"]["xy_mae_m"], 5),
            flush=True,
        )
        print("8 种物理朝向误差（由小到大）:", flush=True)
        for name in ranking:
            print(f"  {name}: {by_orientation[name]['xy_mae_m']:.5f} m", flush=True)

    payload = {
        "experiment": "Physical BEV coordinate conventions using real LiDAR-to-Ego calibration",
        "dataset": "nuScenes v1.0-trainval official val",
        "note": (
            "本实验使用真实的 LiDAR->Ego 标定，但只测试人工物理坐标场的几何一致性。"
            "它不能单独证明真实模型权重的空间错误，也不是 mIoU 或修复后模型指标。"
        ),
        "samples": results,
    }
    path = output_dir / "results.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n保存结果: {path}", flush=True)
    print("Step 32: PASS（仅表示几何测试运行完成）", flush=True)


if __name__ == "__main__":
    main()

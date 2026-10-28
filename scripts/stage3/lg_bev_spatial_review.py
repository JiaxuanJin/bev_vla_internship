"""CPU 独立复核原作者 cropper 源码的坐标 ramp，禁止任何优化。"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.spatial_annotations import AUTHOR, metric_grid
from scripts.stage3.lg_bev_common import DEFAULT_OUT, read, sha, verify_protocol, write


def review(out: Path) -> None:
    """提取现有纯几何类执行 CPU ramp，保存与 CUDA 审计一致的失败证据。"""
    from mmcv import Config

    verify_protocol(out)
    source = (
        AUTHOR / "projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py"
    )
    parsed = ast.parse(source.read_text())
    nodes = [
        node
        for node in parsed.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and node.name in ("calculate_birds_eye_view_parameters", "BevFeatureSlicer")
    ]
    namespace = dict(torch=torch, nn=nn, F=F)
    # 仅执行原文件中的两个无注册副作用的几何定义，不创建主干或双 Head。
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace
    )
    config_path = AUTHOR / "projects/configs/bevformer/bevformer_small_seg_det.py"
    config = Config.fromfile(str(config_path))
    head = config.model.pts_bbox_head
    slicer = namespace["BevFeatureSlicer"](head.det_grid_conf, head.map_grid_conf)
    token = read(out / "smoke_test.json")["steps"][0]["sample_token"]
    row = next(
        f for f in read(out / "manifest.json")["frames"] if f["sample_token"] == token
    )
    native = (
        BEVGrid().lidar_centers(dtype=torch.float32)[..., :2].permute(2, 0, 1)[None]
    )
    transformed = torch.flip(torch.rot90(native, -1, [2, 3]), [3])
    actual = slicer(transformed)[0].permute(1, 2, 0).numpy()
    _, ego, lidar = metric_grid(row)
    expected = lidar[..., :2]
    error = np.linalg.norm(actual - expected, axis=-1)
    gate = read(out / "spatial_gate.json")
    assert abs(float(error.mean()) - gate["native_seg_cropper_mean_error_m"]) < 1e-4
    design = np.concatenate(
        (ego[..., :2], np.ones((*ego.shape[:2], 1))), axis=-1
    ).reshape(-1, 3)
    actual_affine = np.linalg.lstsq(design, actual.reshape(-1, 2), rcond=None)[0]
    expected_affine = np.linalg.lstsq(design, expected.reshape(-1, 2), rcond=None)[0]
    write(
        out / "spatial_review.json",
        dict(
            status="CONFIRMED_BLOCKER",
            cpu_matches_cuda=True,
            optimizer_updates=0,
            source=str(source),
            source_sha256=sha(source),
            config_sha256=sha(config_path),
            sample_token=token,
            mean_error_m=float(error.mean()),
            max_error_m=float(error.max()),
            native_lidar_xy_from_ego_xy1=actual_affine.tolist(),
            calibrated_lidar_xy_from_ego_xy1=expected_affine.tolist(),
            definition="rows=[ego_x,ego_y,1], cols=[lidar_x,lidar_y]; actual original rot90/flip/slicer vs calibrated Ego z=0",
            interpretation="原生 feature sampling 与标定物理语义不一致；不直接等于最终语义预测错误，尚未验证 decoder 是否学会补偿；禁止据此拟合 GT 平移/阈值。",
            next_minimal_action="只在允许训练场景上核查作者传感器/栅格约定及 learned decoder 的补偿；在不改 baseline 的独立坐标桥接协议得到审计前，保留 BLOCKED。",
        ),
    )
    path = out / "spatial_ramps.npz"
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            actual_lidar_xy=actual,
            calibrated_lidar_xy=expected,
            error_m=error,
            ego_xy=ego[..., :2],
        )
    plot(out, actual, expected, error)
    print("SPATIAL_REVIEW_CONFIRMED_BLOCKER", float(error.mean()), float(error.max()))


def plot(
    out: Path, actual: np.ndarray, expected: np.ndarray, error: np.ndarray
) -> None:
    """可视化原生/标定 LiDAR 坐标 ramp 及物理偏差。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        (actual[..., 0], "Native slicer: LiDAR x", "coolwarm"),
        (expected[..., 0], "Calibrated: LiDAR x", "coolwarm"),
        (actual[..., 1], "Native slicer: LiDAR y", "coolwarm"),
        (expected[..., 1], "Calibrated: LiDAR y", "coolwarm"),
        (error, "XY discrepancy / m", "magma"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), constrained_layout=True)
    for ax, (value, title, cmap) in zip(axes, panels):
        im = ax.imshow(value, origin="lower", extent=(-30, 30, -15, 15), cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("GT Ego x / m")
        ax.set_ylabel("GT Ego y / m")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(out / "figures/native_vs_calibrated_ramp.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    review(parser.parse_args().out.resolve())

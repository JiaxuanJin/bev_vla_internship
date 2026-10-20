#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对照作者的相机投影约定，审计推理时的真实输入和插件来源。

只读取 Mini Val 三帧。无需加载 670MB checkpoint，无 GPU 推理，不修改源码。
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SEG_ROOT = ROOT / "BEVFormer-seg"
CONFIG = SEG_ROOT / "projects/configs/bevformer/bevformer_small_seg.py"
OUT = ROOT / "outputs/drivelm_check/root_cause/step46_projection_runtime"
SAMPLES = (0, 40, 80)


def git_output(*args: str) -> list[str]:
    """读取 BEVFormer-seg 工作树差异，不执行任何 Git 写操作。"""
    result = subprocess.run(
        ["git", "-C", str(SEG_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return [f"GIT_ERROR: {result.stderr.strip()}"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def get_raw(value: Any) -> Any:
    """展开 MMCV 的 DataContainer 和单次 test-time augmentation。"""
    from mmcv.parallel import DataContainer

    while True:
        if isinstance(value, DataContainer):
            value = value.data
        elif isinstance(value, (tuple, list)) and len(value) == 1:
            value = value[0]
        else:
            return value


def points_for_bev(pc_range: list[float], h: int, w: int) -> np.ndarray:
    """严格复现 Encoder.get_reference_points 的像素中心及深度采样。"""
    xmin, ymin, zmin, xmax, ymax, zmax = pc_range
    x = xmin + (np.arange(w, dtype=np.float64) + 0.5) * (xmax - xmin) / w
    y = ymin + (np.arange(h, dtype=np.float64) + 0.5) * (ymax - ymin) / h
    xx, yy = np.meshgrid(x, y, indexing="xy")
    sampled_z = zmin + np.linspace(0.5, zmax - zmin - 0.5, 4)
    points = []
    for z in sampled_z:
        points.append(
            np.stack(
                [xx, yy, np.full_like(xx, z), np.ones_like(xx)], axis=-1
            ).reshape(-1, 4)
        )
    return np.concatenate(points, axis=0)


def mask_statistics(meta: dict[str, Any], pc_range: list[float]) -> dict:
    """沿用作者 point_sampling 的图像宽高和正深度条件。"""
    mats = np.asarray(meta["lidar2img"], dtype=np.float64)
    if mats.shape != (6, 4, 4):
        raise ValueError(f"lidar2img 非 6x4x4：{mats.shape}")
    points = points_for_bev(pc_range, 150, 150)
    proj = np.einsum("cij,nj->cni", mats, points)
    depth = proj[..., 2]
    safe_depth = np.maximum(depth, 1e-5)
    u = proj[..., 0] / safe_depth
    v = proj[..., 1] / safe_depth
    img_shapes = meta["img_shape"]
    # 作者 point_sampling 统一使用 img_shape[0] 的 H、W 归一化。
    ih, iw = (int(img_shapes[0][0]), int(img_shapes[0][1]))
    visible = (depth > 1e-5) & (u > 0) & (u < iw) & (v > 0) & (v < ih)
    # 每个 BEV 查询 4 个高度样本：查询只要有一个有效投影就算可见。
    query_visible = visible.reshape(6, 4, 150 * 150).any(axis=1)
    return {
        "image_hw_used_by_encoder": [ih, iw],
        "all_camera_image_hw": [list(x[:2]) for x in img_shapes],
        "per_camera_visible_bev_query_pct": [
            round(float(v.mean() * 100), 3) for v in query_visible
        ],
        "any_camera_visible_bev_query_pct": round(
            float(query_visible.any(axis=0).mean() * 100), 3
        ),
        "all_cameras_invisible_bev_query_pct": round(
            float((~query_visible.any(axis=0)).mean() * 100), 3
        ),
        "non_finite_projection_values": int((~np.isfinite(proj)).sum()),
        "finite_calibration_matrices": bool(np.isfinite(mats).all()),
    }


def main() -> None:
    """运行三帧真实投影审计，并保存 JSON。"""
    if not CONFIG.is_file():
        raise FileNotFoundError(f"缺少原版配置：{CONFIG}")
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "purpose": "真实图像与相机投影 / 运行时源码审计；不评估 mIoU",
        "tracked_python_changes": git_output(
            "diff", "--name-only", "HEAD", "--", "*.py"
        ),
        "all_worktree_changes": git_output("status", "--short"),
        "sample_results": [],
    }

    os.chdir(SEG_ROOT)
    sys.path.insert(0, str(SEG_ROOT))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset

    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.bevformer.modules.encoder import BEVFormerEncoder
    from projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head import (
        BEVFormerHead,
    )
    from projects.mmdet3d_plugin.datasets.utils.vector_map import VectorizedLocalMap
    from projects.mmdet3d_plugin.datasets.pipelines.rasterize import RasterizeMapVectors

    classes = (BEVFormerEncoder, BEVFormerHead, VectorizedLocalMap, RasterizeMapVectors)
    report["loaded_code_paths"] = {
        cls.__name__: inspect.getfile(cls) for cls in classes
    }
    report["package_versions"] = {
        "torch": torch.__version__,
        "numpy": np.__version__,
    }

    cfg = Config.fromfile(str(CONFIG))
    ds_cfg = copy.deepcopy(cfg.data.test)
    ds_cfg.test_mode = True
    ds_cfg.pop("samples_per_gpu", None)
    ds_cfg.pop("workers_per_gpu", None)
    dataset = build_dataset(ds_cfg)
    pc_range = list(cfg.point_cloud_range)
    print("[Step46] 数据集大小:", len(dataset), flush=True)
    print("[Step46] 修改的上游 Python 文件数:",
          len(report["tracked_python_changes"]), flush=True)
    for name, path in report["loaded_code_paths"].items():
        print(f"[源码] {name}: {path}", flush=True)

    for index in SAMPLES:
        if index >= len(dataset):
            continue
        item = dataset[index]
        meta = get_raw(item["img_metas"])
        img = get_raw(item["img"])
        if not isinstance(meta, dict) or not torch.is_tensor(img):
            raise RuntimeError(f"样本 {index} 的元数据或图像类型不正确")
        if img.ndim == 5 and img.shape[0] == 1:
            img = img[0]
        if img.ndim != 4 or img.shape[0] != 6:
            raise RuntimeError(f"样本 {index} 的相机图像形状异常: {img.shape}")
        img_array = img.float()
        statistics = {
            "index": index,
            "sample_token": str(dataset.data_infos[index]["token"]),
            "camera_image_shape": list(img.shape),
            "normalized_image_mean": float(img_array.mean()),
            "normalized_image_std": float(img_array.std()),
            **mask_statistics(meta, pc_range),
        }
        report["sample_results"].append(statistics)
        print(
            f"[样本 {index}] 6相机可见BEV查询比例="
            f"{statistics['any_camera_visible_bev_query_pct']:.2f}% | "
            f"图像标准差={statistics['normalized_image_std']:.2f} | "
            f"矩阵有限={statistics['finite_calibration_matrices']}",
            flush=True,
        )

    path = OUT / "results.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[完成] 报告：", path, flush=True)
    print("注意：此检查不证明模型投影精度达到作者报告值。", flush=True)


if __name__ == "__main__":
    main()

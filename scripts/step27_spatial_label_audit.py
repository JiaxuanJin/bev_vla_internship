#!/usr/bin/env python3
"""核查 BEV 地图 GT 物理坐标到像素行列的约定。

通过人工 XY 直线与真实 nuScenes 地图矢量覆盖图确认标签方向。
不修改模型、checkpoint 和官方评估结果。
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer-seg"
PKL = (
    ROOT
    / "outputs/drivelm_check/root_cause/step21_full_val"
    / "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
)
OUTPUT = ROOT / "outputs/drivelm_check/root_cause/step27_spatial_labels"


def unwrap(value):
    """提取 MMDetection DataContainer 或单一增强包装。"""
    from mmcv.parallel import DataContainer

    while True:
        if isinstance(value, DataContainer):
            value = value.data
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        else:
            return value


def to_numpy(value) -> np.ndarray:
    """将标签转为 NumPy 数组。"""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main() -> None:
    """保存坐标规则、人工已知线段检查和真实 GT 矢量叠加图。"""
    if not PKL.is_file():
        raise FileNotFoundError(PKL)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.pipelines.rasterize import (
        RasterizeMapVectors,
    )

    cfg = Config.fromfile(
        str(REPO / "projects/configs/bevformer/bevformer_small_seg.py")
    )
    grid = cfg.map_grid_conf
    x_min, x_max, x_res = grid["xbound"]
    y_min, y_max, y_res = grid["ybound"]
    shape = (round((y_max - y_min) / y_res), round((x_max - x_min) / x_res))
    print("栅格定义：X 对应列，Y 对应行；行号随 Y 增大而增大")
    print("X 范围 / 分辨率：", grid["xbound"])
    print("Y 范围 / 分辨率：", grid["ybound"])
    print("输出 [H, W]：", shape)

    # 已知物理线段：水平的 divider，垂直的 pedestrian crossing。
    vectors = [
        {"pts": np.array([[-10.0, 0.0], [10.0, 0.0]]), "pts_num": 2, "type": 0},
        {"pts": np.array([[0.0, -7.0], [0.0, 7.0]]), "pts_num": 2, "type": 1},
    ]
    rasterizer = RasterizeMapVectors(map_grid_conf=grid)
    synthetic = rasterizer({"vectors": copy.deepcopy(vectors)})
    synthetic_gt = to_numpy(synthetic["semantic_indices"])
    assert synthetic_gt.shape == (200, 400)
    assert int(synthetic_gt[100, 150]) == 1, "X 向水平线位置有误"
    assert int(synthetic_gt[70, 200]) == 2, "Y 向垂直线位置有误"
    assert int(synthetic_gt[100, 100]) == 0, "背景位置异常"
    np.save(OUTPUT / "synthetic_gt.npy", synthetic_gt)
    print("人工直线标签：PASS")

    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.ann_file = str(PKL)
    dataset_cfg.test_mode = True
    dataset_cfg.pop("samples_per_gpu", None)
    dataset_cfg.pop("workers_per_gpu", None)
    dataset = build_dataset(dataset_cfg)
    assert len(dataset) == 6019
    audited = []
    for index in (0, 1, 3000):
        record = dataset.get_data_info(index)
        raw_vectors = record["vectors"]
        item = dataset[index]
        gt = np.squeeze(to_numpy(unwrap(item["semantic_indices"])))
        if gt.shape != shape:
            raise ValueError(f"样本 {index} GT 形状错误: {gt.shape}")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
        axes[0].imshow(gt, origin="upper", vmin=0, vmax=3, interpolation="nearest")
        axes[0].set_title(f"Ground truth, index={index}")
        axes[1].imshow(gt, origin="upper", vmin=0, vmax=3, interpolation="nearest", alpha=0.55)
        count = 0
        for vector in raw_vectors:
            coords = np.asarray(vector["pts"])[: int(vector["pts_num"]), :2]
            if len(coords) < 2:
                continue
            columns = (coords[:, 0] - x_min) / x_res
            rows = (coords[:, 1] - y_min) / y_res
            axes[1].plot(columns, rows, linewidth=0.9, alpha=0.7)
            count += 1
        axes[1].set_title("GT + Ego-frame map vectors (XY to row/col)")
        for axis in axes:
            axis.set_xlim(0, shape[1])
            axis.set_ylim(shape[0], 0)
            axis.set_xlabel("column / x")
            axis.set_ylabel("row / y")
        fig.tight_layout()
        fig.savefig(OUTPUT / f"gt_vectors_sample_{index:04d}.png", dpi=150)
        plt.close(fig)
        audited.append({
            "index": index,
            "sample_token": dataset.data_infos[index]["token"],
            "gt_unique_labels": np.unique(gt).tolist(),
            "vector_lines_drawn": count,
        })
        print(f"样本 {index}: {count} 条地图矢量叠加图已保存")

    summary = {
        "scope": "GT local-vector rasterization only, not learned BEV orientation",
        "xbound": grid["xbound"],
        "ybound": grid["ybound"],
        "output_shape": list(shape),
        "mapping": "col=(x-x_min)/x_resolution, row=(y-y_min)/y_resolution",
        "synthetic_line_test": "PASS",
        "samples": audited,
        "limitation": "矢量与 GT 的一致性不能证明模型预测的坐标变换正确。",
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Step 27: PASS；注意：只验证 GT 一侧的坐标和栅格化协议")


if __name__ == "__main__":
    main()

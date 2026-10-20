"""Step 53：作者 Slicer 的真实标定坐标反事实 A/B 核验（只读）。

只对合成 XY Ramp 使用作者的空间变换及 grid_sample；不加载模型权重、
不运行神经网络、不计算 mIoU，也不修改作者源码或训练模型。

特别说明：这里用 z_lidar=0 的二维平面投影求解，严格对应 Step 52
使用的 p_ego[:2] = R[:2,:2] @ p_lidar[:2] + t[:2] 假设。
真实 BEV 特征并不一定表示 z_lidar=0；合成坐标归零不代表分割任务修复。
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """读取作者路径、待测帧及报告路径。"""
    parser = argparse.ArgumentParser(description="Step53 calibration grid A/B")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg",
    )
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 40, 80])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs/drivelm_check/root_cause/step53_runtime_calibration_ab",
    )
    return parser.parse_args()


def stats(torch: Any, dx: Any, dy: Any) -> dict[str, float]:
    """计算像素坐标相对 GT 的物理误差。"""
    distance = torch.hypot(dx, dy)
    return {
        "mean_dx_m": round(dx.mean().item(), 7),
        "mean_dy_m": round(dy.mean().item(), 7),
        "mean_xy_error_m": round(distance.mean().item(), 7),
        "p95_xy_error_m": round(torch.quantile(distance.flatten(), 0.95).item(), 7),
        "max_xy_error_m": round(distance.max().item(), 7),
    }


def main() -> None:
    """运行原始 Slicer 与标定逆变换采样网格的合成坐标对照。"""
    args = parse_args()
    source = args.source.expanduser().resolve()
    sys.path.insert(0, str(source))

    import torch
    import torch.nn.functional as functional
    from mmcv import Config
    from pyquaternion import Quaternion
    from projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head import (
        BevFeatureSlicer,
    )

    cfg = Config.fromfile(
        str(source / "projects/configs/bevformer/bevformer_small_seg.py")
    )
    info_path = source / "data/nuscenes/nuscenes_infos_temporal_val.pkl"
    with info_path.open("rb") as file:
        infos = pickle.load(file)["infos"]

    head = cfg.model.pts_bbox_head
    h = int(head.bev_h)
    w = int(head.bev_w)
    det_grid = head.det_grid_conf
    map_grid = head.map_grid_conf
    x0, x1, _ = det_grid["xbound"]
    y0, y1, _ = det_grid["ybound"]
    gt_x0, gt_x1, gt_dx = map_grid["xbound"]
    gt_y0, gt_y1, gt_dy = map_grid["ybound"]
    gt_w = round((gt_x1 - gt_x0) / gt_dx)
    gt_h = round((gt_y1 - gt_y0) / gt_dy)

    dtype = torch.float64
    x = x0 + (torch.arange(w, dtype=dtype) + 0.5) * (x1 - x0) / w
    y = y0 + (torch.arange(h, dtype=dtype) + 0.5) * (y1 - y0) / h
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    ramp = torch.stack((xx, yy), dim=-1).reshape(h, w, 1, 2)

    # 与 BEVFormerHead.forward 完全相同的排列与翻转，不修改作者代码。
    head_ramp = ramp.permute(2, 3, 1, 0)
    head_ramp = torch.flip(head_ramp, dims=[2])
    gt_xvals = gt_x0 + (torch.arange(gt_w, dtype=dtype) + 0.5) * gt_dx
    gt_yvals = gt_y0 + (torch.arange(gt_h, dtype=dtype) + 0.5) * gt_dy
    gt_y, gt_x = torch.meshgrid(gt_yvals, gt_xvals, indexing="ij")
    ego_xy = torch.stack((gt_x, gt_y), dim=0)

    with torch.no_grad():
        slicer = BevFeatureSlicer(det_grid, map_grid)
        original = slicer(head_ramp)[0]

    entries = []
    for idx in args.indices:
        if idx < 0 or idx >= len(infos):
            raise IndexError(f"Mini Val 帧号越界: {idx}")
        info = infos[idx]
        rotation = torch.tensor(
            Quaternion(info["lidar2ego_rotation"]).rotation_matrix,
            dtype=dtype,
        )
        translation = torch.tensor(info["lidar2ego_translation"], dtype=dtype)
        r_xy = rotation[:2, :2]
        t_xy = translation[:2]
        if abs(torch.linalg.det(r_xy).item()) < 0.9:
            raise ValueError("水平旋转子矩阵不可逆或倾斜异常")

        # 反事实：已知 Ego GT 像素中心，求在 LiDAR 平面(z_lidar=0)
        # 中对应的 XY，再映射到 Head 翻转后的 BEV 特征像素索引。
        lidar_xy = torch.linalg.solve(
            r_xy, (ego_xy - t_xy[:, None, None]).reshape(2, -1)
        ).reshape(2, gt_h, gt_w)
        lidar_x, lidar_y = lidar_xy.unbind(dim=0)
        x_index = (lidar_x - x0) / ((x1 - x0) / w) - 0.5
        y_index = (lidar_y - y0) / ((y1 - y0) / h) - 0.5
        # Head 变换后：行 = W-1-X索引；列 = Y索引。
        src_row = w - 1 - x_index
        src_col = y_index
        normalized_col = 2.0 * src_col / (h - 1) - 1.0
        normalized_row = 2.0 * src_row / (w - 1) - 1.0
        sampling_grid = torch.stack(
            (normalized_col, normalized_row), dim=-1
        ).unsqueeze(0)
        valid = (
            (normalized_col >= -1.0)
            & (normalized_col <= 1.0)
            & (normalized_row >= -1.0)
            & (normalized_row <= 1.0)
        )
        if valid.float().mean().item() < 0.99:
            raise ValueError("有效采样像素少于 99%，先核查尺寸和轴向")

        with torch.no_grad():
            corrected = functional.grid_sample(
                head_ramp,
                sampling_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0]

        original_ego = torch.einsum("ab,bhw->ahw", r_xy, original)
        corrected_ego = torch.einsum("ab,bhw->ahw", r_xy, corrected)
        original_ego += t_xy[:, None, None]
        corrected_ego += t_xy[:, None, None]
        baseline_err = stats(
            torch, original_ego[0] - gt_x, original_ego[1] - gt_y
        )
        fixed_err = stats(
            torch,
            (corrected_ego[0] - gt_x)[valid],
            (corrected_ego[1] - gt_y)[valid],
        )
        entry = {
            "index": idx,
            "token": info.get("token"),
            "translation_xy_m": [round(v, 6) for v in t_xy.tolist()],
            "valid_pixel_fraction": round(valid.float().mean().item(), 7),
            "author_slicer_baseline": baseline_err,
            "calibrated_grid_counterfactual": fixed_err,
        }
        entries.append(entry)
        print(
            f"[{idx:02d}] tX={t_xy[0].item():+.4f} m "
            f"原始={baseline_err['mean_xy_error_m']:.5f} m "
            f"校正={fixed_err['mean_xy_error_m']:.7f} m "
            f"有效像素={entry['valid_pixel_fraction']:.2%}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "Step53 author slicer vs inverse calibration grid",
        "scope": "synthetic XY coordinate ramps only; NO MODEL FORWARD/IoU",
        "assumption": "z_lidar=0 planar mapping; consistent with Step52",
        "source": str(source),
        "frames": entries,
        "conclusion_limit": (
            "校准后坐标误差变小仅验证二维映射可构造，"
            "不证明训练 checkpoint 的坐标、权重或分割 mIoU 已修复。"
        ),
    }
    (args.output / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Step 53：真实标定二维坐标反事实核验",
        "",
        "只读 CPU 合成 Ramp 测试，不执行模型前向、不评估 mIoU。",
        "",
        "| 帧 | LiDAR→Ego X 平移 (m) | 原始误差 (m) | 校正误差 (m) |",
        "|---|---:|---:|---:|",
    ]
    for item in entries:
        lines.append(
            f"| {item['index']} | {item['translation_xy_m'][0]:+.4f} | "
            f"{item['author_slicer_baseline']['mean_xy_error_m']:.6f} | "
            f"{item['calibrated_grid_counterfactual']['mean_xy_error_m']:.6f} |"
        )
    lines += [
        "",
        "## 结论边界",
        "",
        "这是使用真实标定的合成坐标验证，不代表模型已经修复。",
        "只有在冻结 checkpoint、相同样本和评估协议下进行独立 A/B，",
        "确认三类 IoU 稳健改善后才能提出正式代码修复。",
        "",
    ]
    (args.output / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("[Step53] 完成，报告：", args.output / "audit.md", flush=True)


if __name__ == "__main__":
    main()

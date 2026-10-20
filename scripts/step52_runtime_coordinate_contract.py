"""使用真实 nuScenes 标定核验作者 BEV 分割空间链路。

本脚本只运行 CPU 合成坐标 Ramp，不执行模型前向、训练或 IoU 评估。
保留作者 ``BevFeatureSlicer`` 的实际实现；不修改源码和 checkpoint。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析独立环境路径和待核验的 Mini Val 帧编号。"""
    parser = argparse.ArgumentParser(description="BEV/Ego runtime coordinate audit")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home()
        / "projects/bevformer_author_clean_5080/BEVFormer-seg",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 40, 80],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs/drivelm_check/root_cause/step52_runtime_coordinate_contract",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """计算源码 SHA256，便于识别被执行的作者代码。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_ramps(
    torch: Any, height: int, width: int, det_grid: Any
) -> Any:
    """按 Encoder 查询展平顺序构造物理 X/Y 坐标场。"""
    x_min, x_max, _ = det_grid["xbound"]
    y_min, y_max, _ = det_grid["ybound"]
    x = x_min + (torch.arange(width, dtype=torch.float64) + 0.5) * (
        (x_max - x_min) / width
    )
    y = y_min + (torch.arange(height, dtype=torch.float64) + 0.5) * (
        (y_max - y_min) / height
    )
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(height * width, 1, 2)


def get_gt_coordinates(torch: Any, map_grid: Any) -> tuple[Any, Any]:
    """构造 GT Rasterize 的行 Y / 列 X 像素中心物理坐标。"""
    x_min, x_max, x_step = map_grid["xbound"]
    y_min, y_max, y_step = map_grid["ybound"]
    width = round((x_max - x_min) / x_step)
    height = round((y_max - y_min) / y_step)
    x = x_min + (torch.arange(width, dtype=torch.float64) + 0.5) * x_step
    y = y_min + (torch.arange(height, dtype=torch.float64) + 0.5) * y_step
    gt_y, gt_x = torch.meshgrid(y, x, indexing="ij")
    return gt_x, gt_y


def summarize_diff(torch: Any, dx: Any, dy: Any) -> dict[str, float]:
    """以米为单位输出每个坐标轴的残差统计。"""
    dist = torch.hypot(dx, dy)
    return {
        "signed_dx_mean_m": round(dx.mean().item(), 6),
        "signed_dy_mean_m": round(dy.mean().item(), 6),
        "abs_dx_mean_m": round(dx.abs().mean().item(), 6),
        "abs_dy_mean_m": round(dy.abs().mean().item(), 6),
        "mean_distance_m": round(dist.mean().item(), 6),
        "p95_distance_m": round(torch.quantile(dist.flatten(), 0.95).item(), 6),
        "max_distance_m": round(dist.max().item(), 6),
    }


def main() -> None:
    """核查实际 Slicer 采样位置及真实 LiDAR/Ego 物理一致性。"""
    args = parse_args()
    source = args.source.expanduser().resolve()
    config_path = source / "projects/configs/bevformer/bevformer_small_seg.py"
    ann_file = source / "data/nuscenes/nuscenes_infos_temporal_val.pkl"
    head_file = (
        source
        / "projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py"
    )
    for path in (source, config_path, ann_file, head_file):
        if not path.exists():
            raise FileNotFoundError(f"缺少输入路径: {path}")

    # 在干净进程内优先导入作者源码；不安装或修改任何依赖。
    sys.path.insert(0, str(source))
    import torch
    from mmcv import Config
    from pyquaternion import Quaternion
    from projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_head import (
        BevFeatureSlicer,
    )

    cfg = Config.fromfile(str(config_path))
    head = cfg.model.pts_bbox_head
    det_grid = head.det_grid_conf
    map_grid = head.map_grid_conf
    bev_h = int(head.bev_h)
    bev_w = int(head.bev_w)

    # 使用源码中的真实 Slicer，而不是自行假设其 grid_sample 行为。
    with torch.no_grad():
        ramps = get_ramps(torch, bev_h, bev_w, det_grid)
        seg_bev = ramps.reshape(bev_h, bev_w, 1, 2).permute(2, 3, 1, 0)
        seg_bev = torch.flip(seg_bev, dims=[2])
        cropper = BevFeatureSlicer(det_grid, map_grid)
        sampled = cropper(seg_bev).squeeze(0)
        lidar_x, lidar_y = sampled[0], sampled[1]
        gt_x, gt_y = get_gt_coordinates(torch, map_grid)

    if lidar_x.shape != gt_x.shape or lidar_y.shape != gt_y.shape:
        raise ValueError(
            f"裁剪后坐标形状与 GT 不同: {tuple(lidar_x.shape)} "
            f"vs {tuple(gt_x.shape)}"
        )

    # 仅加载可信的本地数据标注。样本不参与模型推理或模型训练。
    with ann_file.open("rb") as file:
        infos = pickle.load(file)["infos"]

    output: dict[str, Any] = {
        "purpose": "BEV ramp -> author head transform -> real Slicer -> LiDAR/Ego",
        "scope": "CPU synthetic coordinate fields + real calibration; no model forward",
        "source": str(source),
        "source_head_sha256": sha256_file(head_file),
        "source_config_sha256": sha256_file(config_path),
        "checkpoint_modified": False,
        "source_modified": False,
        "bev_hw": [bev_h, bev_w],
        "after_head_hw": list(seg_bev.shape[-2:]),
        "after_slicer_hw": list(sampled.shape[-2:]),
        "gt_hw": list(gt_x.shape),
        "assumptions": [
            "Encoder flat query order: row=Y_lidar, column=X_lidar",
            "Head: reshape->permute(2,3,1,0)->flip(dim=2)",
            "GT raster: row=Y_ego, column=X_ego, pixel-centre coordinates",
            "LiDAR->Ego: p_ego=R*p_lidar+t, z omitted in XY ramp",
            "This diagnostic does not establish the checkpoint training runtime",
        ],
        "frames": [],
    }
    print("[Step52] 使用作者 BevFeatureSlicer 真实采样，CPU 运行")
    print(f"BEV: {bev_h}x{bev_w}; GT: {tuple(gt_x.shape)}")

    for index in args.indices:
        if not (0 <= index < len(infos)):
            raise IndexError(f"帧号越界: {index}, 总帧数={len(infos)}")
        info = infos[index]
        if "lidar2ego_rotation" not in info or "lidar2ego_translation" not in info:
            raise KeyError(
                "PKL 不包含 LiDAR->Ego 标定；可用字段: "
                + ", ".join(sorted(info))
            )
        quat = Quaternion(info["lidar2ego_rotation"])
        rot = torch.tensor(quat.rotation_matrix, dtype=torch.float64)
        t = torch.as_tensor(info["lidar2ego_translation"], dtype=torch.float64)
        # Ramp 使用 z=0 作为核验平面，真实特征并没有单一 z 值。
        ego_x = rot[0, 0] * lidar_x + rot[0, 1] * lidar_y + t[0]
        ego_y = rot[1, 0] * lidar_x + rot[1, 1] * lidar_y + t[1]
        dx = ego_x - gt_x
        dy = ego_y - gt_y
        measurements = summarize_diff(torch, dx, dy)
        entry = {
            "index": index,
            "sample_token": info.get("token"),
            "lidar2ego_translation_m": t.tolist(),
            "lidar2ego_rotation": list(info["lidar2ego_rotation"]),
            "coordinate_error": measurements,
            "center_error_xy_m": [
                round(dx[gt_x.shape[0] // 2, gt_x.shape[1] // 2].item(), 6),
                round(dy[gt_x.shape[0] // 2, gt_x.shape[1] // 2].item(), 6),
            ],
        }
        output["frames"].append(entry)
        print(
            f"[Val {index:02d}] mean dx={measurements['signed_dx_mean_m']:+.3f}m, "
            f"dy={measurements['signed_dy_mean_m']:+.3f}m, "
            f"mean XY distance={measurements['mean_distance_m']:.3f}m"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    result_json = args.output / "results.json"
    result_json.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Step 52：BEV/GT 运行时坐标协议核验",
        "",
        "本次使用作者真实 Slicer 和 nuScenes 标定，在 CPU 上传播合成 XY 坐标场；",
        "不执行训练或模型前向，不评估 mIoU。",
        "",
        f"- BEV 特征尺寸：{bev_h}×{bev_w}",
        f"- GT 栅格尺寸：{gt_x.shape[0]}×{gt_x.shape[1]}",
        "",
        "| 帧号 | 平均 ΔX (m) | 平均 ΔY (m) | 平均平面误差 (m) |",
        "|---|---:|---:|---:|",
    ]
    for sample in output["frames"]:
        err = sample["coordinate_error"]
        lines.append(
            f"| {sample['index']} | {err['signed_dx_mean_m']:+.4f} | "
            f"{err['signed_dy_mean_m']:+.4f} | "
            f"{err['mean_distance_m']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "上述误差是特定假设下坐标采样位置与 Ego 网格的物理差异，",
            "不能证明 checkpoint 训练时同样执行这份 Python 源码，",
            "也不能直接推出预测 IoU 或表示训练/推理空间已完全对齐。",
            "如有较大差异，应先检查源坐标 flatten、图像轴和 GT raster 的约定，",
            "在独立数据上核查后才决定是否修改任何训练或推理代码。",
            "",
        ]
    )
    report_md = args.output / "audit.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Step52] JSON: {result_json}")
    print(f"[Step52] 报告: {report_md}")
    print("[Step52] 完成；仅坐标诊断，不能据此宣称模型已修复。")


if __name__ == "__main__":
    main()

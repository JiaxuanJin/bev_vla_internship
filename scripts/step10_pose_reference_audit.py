#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对 DriveLM799 地图 GT 与相机模型 BEV 的参考坐标原点。

脚本只读取此前可信的本地 nuScenes PKL、nuScenes 元数据及 Step 9 结果。
本实验不执行 GPU 推理，不修改第三方仓库、权重或正式评估指标。
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

ROOT = Path(__file__).resolve().parents[1]
STEP9 = ROOT / "outputs/drivelm_check/root_cause/step9_drivelm799/results.json"
PKL = ROOT / "data/drivelm/nuscenes_infos_temporal_drivelm799_officialscene.pkl"
NUSC_ROOT = ROOT / "BEVFormer-seg/data/nuscenes"
OUT = ROOT / "outputs/drivelm_check/root_cause/step10_pose_reference_audit.json"
MAP_RESOLUTION_METERS = 0.15


def quaternion_angle_deg(a: Any, b: Any) -> float:
    """返回两个姿态四元数之间的最小夹角，单位为度。"""
    relative = Quaternion(a).inverse * Quaternion(b)
    w = min(1.0, abs(float(relative.normalised.w)))
    return float(np.degrees(2.0 * np.arccos(w)))


def main() -> None:
    """逐帧检查 ego 与 LiDAR 坐标基准，保存可复现诊断证据。"""
    for path in (STEP9, PKL, NUSC_ROOT):
        if not path.exists():
            raise FileNotFoundError(f"必需文件或目录不存在: {path}")

    step9 = json.loads(STEP9.read_text(encoding="utf-8"))
    rows = step9["rows"]
    if not rows:
        raise ValueError("Step 9 没有任何样本")

    # 仅加载本人此前在本地生成、可信来源的 nuScenes PKL。
    with PKL.open("rb") as stream:
        payload = pickle.load(stream)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    lookup = {str(info["token"]): info for info in infos}

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(NUSC_ROOT), verbose=False)
    records = []

    for row in rows:
        token = str(row["sample_token"])
        info = lookup.get(token)
        if info is None:
            raise KeyError(f"Step 9 样本未在官方场景 PKL 中找到: {token}")

        sample = nusc.get("sample", token)
        lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nusc.get("ego_pose", lidar["ego_pose_token"])
        calib = nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])

        map_pos = np.asarray(info["ego2global_translation"], dtype=float)
        map_rot = info["ego2global_rotation"]
        official_ego_pos = np.asarray(ego["translation"], dtype=float)
        lidar_in_ego = np.asarray(calib["translation"], dtype=float)
        lidar_in_global = official_ego_pos + Quaternion(ego["rotation"]).rotate(
            lidar_in_ego
        )

        record = {
            "index": row["index"],
            "sample_token": token,
            "scene_token": row["scene_token"],
            "location": str(info.get("location", "")),
            "pkl_map_ego_vs_nuscenes_ego_delta_m": round(
                float(np.linalg.norm(map_pos - official_ego_pos)), 6
            ),
            "pkl_map_ego_vs_nuscenes_ego_angle_deg": round(
                quaternion_angle_deg(map_rot, ego["rotation"]), 6
            ),
            "lidar_to_ego_translation_xyz_m": lidar_in_ego.tolist(),
            "lidar_to_ego_rotation_angle_deg": round(
                quaternion_angle_deg([1.0, 0.0, 0.0, 0.0], calib["rotation"]), 6
            ),
            "map_ego_to_lidar_global_origin_distance_m": round(
                float(np.linalg.norm(lidar_in_global - map_pos)), 6
            ),
            "lidar_to_ego_translation_in_map_pixels_xy": (
                lidar_in_ego[:2] / MAP_RESOLUTION_METERS
            ).round(4).tolist(),
            "step9_fg_miou_original": row["scores"]["original"]["foreground_miou"],
            "step9_fg_miou_fixed_shift": row["scores"]["fixed_flip_shift"]["foreground_miou"],
        }
        records.append(record)
        print(
            f"idx={record['index']:3d} | "
            f"地图PKL与官方ego原点差="
            f"{record['pkl_map_ego_vs_nuscenes_ego_delta_m']:.4f}m | "
            f"LiDAR相对ego平移={lidar_in_ego.round(3).tolist()}m | "
            f"旋转差={record['lidar_to_ego_rotation_angle_deg']:.4f}°",
            flush=True,
        )

    source = ROOT / "BEVFormer-seg/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py"
    source_text = source.read_text(encoding="utf-8") if source.exists() else ""
    flags = {
        "dataset_map_uses_ego2global": (
            "self.vector_map.gen_vectorized_samples" in source_text
            and "info['ego2global_translation']" in source_text
        ),
        "model_projection_uses_sensor2lidar": (
            "sensor2lidar_rotation" in source_text
            and "lidar2img" in source_text
        ),
    }

    result = {
        "type": "metadata_reference_frame_audit_only",
        "important_note": (
            "这里只比较参考坐标及元数据，不计算预测 mIoU；"
            "即使地图采用 ego 原点且投影采用 LiDAR 原点，也不能单独证明"
            "这就是低分根因，须进一步做物理一致的地图重栅格化对照。"
        ),
        "sample_count": len(records),
        "map_resolution_m_per_pixel": MAP_RESOLUTION_METERS,
        "source_flags": flags,
        "samples": records,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[保存] {OUT}")
    print("[说明] 仅验证坐标参考基准，不能直接判定根因，也不是新 mIoU。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 38：无需推理，核对全量 Val 元数据与评估公式。

在项目根目录执行：
    python scripts/step38_protocol_metadata_audit.py

仅读取数据和历史结果，不修改 checkpoint、GT 或已有实验记录。
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VAL_PKL = (
    ROOT
    / "outputs/drivelm_check/root_cause/step21_full_val"
    / "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
)
STEP34 = (
    ROOT
    / "outputs/drivelm_check/root_cause/step34_full_val_orientation_ab"
    / "results.json"
)
OUTPUT_DIR = ROOT / "outputs/drivelm_check/root_cause/step38_protocol_audit"
NUSCENES_ROOT = ROOT / "data/nuscenes"
SAMPLE_INDICES = (0, 300, 900, 1500, 2400, 3000, 3600, 3715, 3718, 3719, 4500, 6018)
CAMERA_KEYS = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)


def sha256_file(path: Path) -> str:
    """计算文件 SHA256，用于与 Step34 留存的实验身份比较。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def class_ious(confusion: np.ndarray) -> list[float]:
    """按作者 IoU 的 TP/(TP+FP+FN) 规则汇总类别指标。"""
    tp = np.diag(confusion)
    union = confusion.sum(0) + confusion.sum(1) - tp
    return [float(tp[i] / union[i]) if union[i] else 0.0 for i in range(4)]


def compare_step34(record: dict[str, Any]) -> dict[str, Any]:
    """由既有混淆矩阵独立验证 IoU，避免再次推理。"""
    audit: dict[str, Any] = {}
    for name in ("original", "physical_ccw_candidate"):
        branch = record[name]
        matrix = np.asarray(branch["confusion_gt_rows_pred_cols"], dtype=np.int64)
        if matrix.shape != (4, 4):
            raise ValueError(f"{name}: 混淆矩阵必须为 4x4")
        actual = class_ious(matrix)
        saved = [branch["class_iou"][key] for key in (
            "background", "divider", "ped_crossing", "boundary"
        )]
        max_delta = max(abs(a - b) for a, b in zip(actual, saved))
        fg = sum(actual[1:]) / 3.0
        fg_delta = abs(fg - branch["foreground_miou"])
        audit[name] = {
            "foreground_miou_recalculated": fg,
            "max_class_iou_abs_diff": max_delta,
            "foreground_miou_abs_diff": fg_delta,
            "pass": max_delta < 1e-6 and fg_delta < 1e-6,
        }
    return audit


def vector_delta(a: Any, b: Any) -> float:
    """计算两个向量的最大绝对差。"""
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right)))


def quaternion_delta(a: Any, b: Any) -> float:
    """通过旋转矩阵比较等价四元数（q 和 -q 视为相同）。"""
    from pyquaternion import Quaternion

    return vector_delta(
        Quaternion(a).rotation_matrix, Quaternion(b).rotation_matrix
    )


def check_nuscenes(infos: list[dict[str, Any]]) -> dict[str, Any]:
    """抽样比对官方 nuScenes 原始 sample 和 Ego Pose 元数据。"""
    from nuscenes.nuscenes import NuScenes  # 只在需要读取官方数据时导入

    nuscenes = NuScenes(version="v1.0-trainval", dataroot=str(NUSCENES_ROOT), verbose=False)
    rows: list[dict[str, Any]] = []
    for index in SAMPLE_INDICES:
        if index >= len(infos):
            continue
        info = infos[index]
        sample = nuscenes.get("sample", str(info["token"]))
        lidar = nuscenes.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego = nuscenes.get("ego_pose", lidar["ego_pose_token"])
        calibrated = nuscenes.get("calibrated_sensor", lidar["calibrated_sensor_token"])

        translation_error = vector_delta(
            info["ego2global_translation"], ego["translation"]
        )
        rotation_error = quaternion_delta(
            info["ego2global_rotation"], ego["rotation"]
        )
        lidar_translation_error = vector_delta(
            info["lidar2ego_translation"], calibrated["translation"]
        )
        lidar_rotation_error = quaternion_delta(
            info["lidar2ego_rotation"], calibrated["rotation"]
        )
        camera_token_errors = []
        for camera in CAMERA_KEYS:
            if camera not in info.get("cams", {}):
                camera_token_errors.append(f"{camera}:missing")
                continue
            camera_info = info["cams"][camera]
            # 某些版本的 PKL 并不保存 sample_data_token，此时明确跳过。
            if "sample_data_token" in camera_info:
                if str(camera_info["sample_data_token"]) != str(sample["data"][camera]):
                    camera_token_errors.append(f"{camera}:different_token")

        timestamp_ok = int(info["timestamp"]) == int(sample["timestamp"])
        scene_ok = str(info["scene_token"]) == str(sample["scene_token"])
        row = {
            "index": index,
            "sample_token": str(info["token"]),
            "scene_ok": scene_ok,
            "timestamp_ok": timestamp_ok,
            "ego_translation_max_abs_error_m": translation_error,
            "ego_rotation_matrix_max_abs_error": rotation_error,
            "lidar_translation_max_abs_error_m": lidar_translation_error,
            "lidar_rotation_matrix_max_abs_error": lidar_rotation_error,
            "camera_token_errors": camera_token_errors,
        }
        row["pass"] = (
            timestamp_ok
            and scene_ok
            and translation_error < 1e-4
            and rotation_error < 1e-5
            and lidar_translation_error < 1e-4
            and lidar_rotation_error < 1e-5
            and not camera_token_errors
        )
        rows.append(row)
    return {
        "status": "complete",
        "checked": len(rows),
        "passed": sum(row["pass"] for row in rows),
        "failed": sum(not row["pass"] for row in rows),
        "samples": rows,
    }


def generate_markdown(result: dict[str, Any]) -> str:
    """生成中文结果摘要，区分已确认和未确认的范围。"""
    data = result["dataset"]
    metric = result["metric"]
    identity = result["identity"]
    ref = result["nuscenes_reference"]
    lines = [
        "# Step 38：评估协议与 Val 元数据审计",
        "",
        "## 一、数据集身份",
        f"- PKL 样本数：{data['frames']}，场景数：{data['scenes']}",
        f"- Sample Token 重复数：{data['duplicate_tokens']}",
        f"- 时间戳是否非降序：{data['timestamps_nondecreasing']}",
        f"- Step 34 与当前 PKL 哈希一致：{identity['val_pkl_hash_match']}",
        "",
        "## 二、作者指标公式重算",
        "作者定义的前景 mIoU 为 divider、ped_crossing、boundary 三类 IoU 的平均，",
        "每类 IoU 按全局 TP/(TP+FP+FN) 计算，背景不参与平均。",
    ]
    for name, branch in metric.items():
        lines.append(
            f"- {name}: mIoU={branch['foreground_miou_recalculated']:.6%}，"
            f"重算一致={branch['pass']}，"
            f"最大类别差={branch['max_class_iou_abs_diff']:.3g}"
        )
    lines.extend(["", "## 三、与官方 nuScenes 元数据抽样对照"])
    if ref["status"] == "complete":
        lines.append(
            f"- 已核查 {ref['checked']} 帧：通过 {ref['passed']} 帧，"
            f"未通过 {ref['failed']} 帧。"
        )
        for item in ref["samples"]:
            if not item["pass"]:
                lines.append(
                    f"- 异常索引 {item['index']}: {json.dumps(item, ensure_ascii=False)}"
                )
    else:
        lines.append(f"- 未完成官方元数据比对：{ref.get('error', '未知原因')}")
    lines.extend([
        "",
        "## 四、结论边界",
        "- 已知 Step 25 与作者 IoU 核心定义一致，本文再次基于 Step 34 混淆矩阵校验计算结果。",
        "- 本检查不执行模型前向，不核实训练时的 forward 源码，也不证明预测/GT 几何完全一致。",
        "- 如果官方元数据抽查通过，应优先调查 GT 栅格方向、BEV 特征取样及训练源码对应关系。",
        "- 原始 Step 34 JSON 的 physical_ccw_candidate 为历史命名；实际高分候选是 Transpose。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    """审计已有实验结果与官方 Val 元数据，并输出 JSON/Markdown。"""
    if not VAL_PKL.is_file() or not STEP34.is_file():
        raise FileNotFoundError(f"找不到数据文件：{VAL_PKL} 或 {STEP34}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with VAL_PKL.open("rb") as stream:
        loaded = pickle.load(stream)
    infos = list(loaded["infos"])
    with STEP34.open("r", encoding="utf-8") as stream:
        step34 = json.load(stream)

    tokens = [str(info["token"]) for info in infos]
    timestamps = [int(info["timestamp"]) for info in infos]
    scenes = {str(info["scene_token"]) for info in infos}
    recorded_hash = step34.get("identity", {}).get("val_pkl_sha256")
    actual_hash = sha256_file(VAL_PKL)
    identity = {
        "val_pkl_sha256_current": actual_hash,
        "val_pkl_sha256_step34": recorded_hash,
        "val_pkl_hash_match": actual_hash == recorded_hash,
    }
    dataset = {
        "frames": len(infos),
        "scenes": len(scenes),
        "duplicate_tokens": len(tokens) - len(set(tokens)),
        "timestamps_nondecreasing": all(
            left <= right for left, right in zip(timestamps, timestamps[1:])
        ),
        "metadata_version": loaded.get("metadata", {}).get("version"),
    }
    try:
        reference = check_nuscenes(infos)
    except (ImportError, OSError, AssertionError, KeyError, ValueError, RuntimeError) as error:
        reference = {"status": "unavailable", "error": str(error)}

    result = {
        "experiment": "Step38 official metric and full Val metadata audit",
        "inference_performed": False,
        "identity": identity,
        "dataset": dataset,
        "metric": compare_step34(step34),
        "nuscenes_reference": reference,
    }
    json_path = OUTPUT_DIR / "results.json"
    md_path = OUTPUT_DIR / "audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(generate_markdown(result), encoding="utf-8")

    print("Step 38 审计完成（未执行 GPU 推理）")
    print(f"Val PKL：{dataset['frames']} 帧 / {dataset['scenes']} 场景")
    print(f"与 Step34 PKL 哈希匹配：{identity['val_pkl_hash_match']}")
    print("作者 IoU 公式重算：", all(x["pass"] for x in result["metric"].values()))
    if reference["status"] == "complete":
        print(f"官方 nuScenes 抽查：{reference['passed']}/{reference['checked']} 通过")
    else:
        print("官方 nuScenes 抽查不可用：", reference["error"])
    print(f"审计报告：{md_path}")
    print(f"结果 JSON：{json_path}")


if __name__ == "__main__":
    main()

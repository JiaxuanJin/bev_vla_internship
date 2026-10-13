#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DriveLM 799 帧推理条件复查（修复版）。

修复原脚本错误：nuScenes 原始 sample.json 通常不包含 sample['data']。
必须根据 sample_data.sample_token 反向查找，再经 calibrated_sensor 和
sensor.channel 确定六路相机及 LIDAR_TOP。全程只读数据，不进行模型推理。
仅覆盖旧版同名 JSON/CSV 检查结果，不修改数据集、模型或已有 PKL。
"""

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
REQUIRED_CHANNELS = (*CAMERAS, "LIDAR_TOP")


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rows_by_token(path):
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"nuScenes 元数据格式异常（预期列表）：{path}")
    return {row["token"]: row for row in rows}


def find_drivelm_scenes(root):
    """查找 DriveLM scene_token -> key_frames 结构。"""
    found = []
    seen = set()

    def walk(node, parent_key=""):
        if not isinstance(node, (dict, list)) or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, dict):
            if isinstance(node.get("key_frames"), dict):
                found.append((parent_key, node["key_frames"]))
                return
            for k, v in node.items():
                walk(v, str(k))
        else:
            for i, v in enumerate(node):
                walk(v, str(i))

    walk(root)
    if not found:
        raise ValueError("DriveLM JSON 中找不到 key_frames 字典")
    return found


def sample_data_index(raw_rows, wanted_tokens, calib, sensors, samples):
    """为所需样本建立 'sample_token -> channel -> sample_data' 反向索引。

    nuScenes 的原始 sample_data 表才存储样本和传感器数据的连接关系。
    只保留关键帧，避免混入非关键帧 sweeps；对于异常重复项取时间戳最近者。
    """
    index = defaultdict(dict)
    duplicates = 0
    missing_sensor_channel = 0
    selected = 0
    for sd in raw_rows:
        sample_token = sd.get("sample_token")
        if sample_token not in wanted_tokens:
            continue
        if sd.get("is_key_frame") is False:
            continue
        cs = calib.get(sd.get("calibrated_sensor_token"))
        sensor = sensors.get(cs.get("sensor_token")) if cs else None
        channel = sensor.get("channel") if sensor else None
        if channel not in REQUIRED_CHANNELS:
            missing_sensor_channel += 1
            continue
        prev = index[sample_token].get(channel)
        if prev is not None:
            duplicates += 1
            time = int(samples[sample_token].get("timestamp", 0))
            if abs(int(sd.get("timestamp", 0)) - time) >= abs(int(prev.get("timestamp", 0)) - time):
                continue
        index[sample_token][channel] = sd
        selected += 1
    return index, {"duplicate_channel_records": duplicates,
                   "unresolved_sensor_channels": missing_sensor_channel,
                   "selected_keyframe_records": selected}


def valid_pose(pose):
    return bool(pose and len(pose.get("translation", [])) == 3 and
                len(pose.get("rotation", [])) == 4)


def valid_calibration(calib_row, camera=False):
    if not calib_row:
        return False
    if len(calib_row.get("translation", [])) != 3 or len(calib_row.get("rotation", [])) != 4:
        return False
    if camera:
        intrinsic = calib_row.get("camera_intrinsic")
        return bool(isinstance(intrinsic, list) and len(intrinsic) == 3 and
                    all(isinstance(r, list) and len(r) == 3 for r in intrinsic))
    return True


def p90(numbers):
    if not numbers:
        return None
    s = sorted(numbers)
    return s[int(0.9 * (len(s) - 1))]


def inspect(root):
    qa = root / "data/drivelm/QA_dataset_nus/v1_1_val_nus_q_only.json"
    meta = root / "data/nuscenes_trainval_meta/v1.0-trainval"
    val_dir = root / "data/drivelm/val_data"
    out_dir = root / "outputs/drivelm_check"
    necessary = [qa, *(meta / (n + ".json") for n in
        ("sample", "sample_data", "sensor", "calibrated_sensor", "ego_pose", "scene"))]
    for file in necessary:
        if not file.is_file():
            raise FileNotFoundError(f"必需文件不存在：{file}")

    print("[1/4] 读取 DriveLM 的 799 个 token 和 nuScenes 原始元数据…", flush=True)
    groups = find_drivelm_scenes(load_json(qa))
    drive_scene_for_frame = {}
    duplicate_frame_tokens = 0
    for drive_scene_token, frames in groups:
        for frame_token in frames:
            if frame_token in drive_scene_for_frame:
                duplicate_frame_tokens += 1
            drive_scene_for_frame[frame_token] = drive_scene_token
    wanted = set(drive_scene_for_frame)
    samples = rows_by_token(meta / "sample.json")
    calib = rows_by_token(meta / "calibrated_sensor.json")
    sensors = rows_by_token(meta / "sensor.json")
    poses = rows_by_token(meta / "ego_pose.json")
    official_scenes = rows_by_token(meta / "scene.json")

    print("[2/4] 从 sample_data.sample_token 建立七路传感器反向索引…", flush=True)
    # 注意：原始 sample.json 没有 Devkit 动态补出的 sample['data'] 字段。
    # 这里不是用 sd['token'] 去找 sample['data']，而是由 sample_data 反查样本及通道。
    source_sd = load_json(meta / "sample_data.json")
    if not isinstance(source_sd, list):
        raise ValueError("sample_data.json 格式异常，预期列表")
    frame_sd, index_stats = sample_data_index(source_sd, wanted, calib, sensors, samples)
    del source_sd  # 释放完整 sample_data 列表占用的内存

    missing = Counter()
    examples = defaultdict(list)
    rows = []
    frame_times = defaultdict(list)
    image_found = 0
    calibration_good = 0
    lidar_good = 0
    complete = 0
    previous_in_subset = 0
    timestamp_offsets_ms = []

    print("[3/4] 验证六路图像、标定、车辆位姿和 LIDAR_TOP 参考系…", flush=True)
    for token in sorted(wanted):
        sample = samples.get(token)
        if sample is None:
            missing["sample_metadata"] += 1
            rows.append({"sample_token": token, "scene_name": "", "available_cameras": 0,
                "calibration_and_pose_ok": False, "lidar_reference_ok": False,
                "previous_frame_in_subset": False, "timestamp_us": "", "all_ready": False})
            continue
        scene_token = sample.get("scene_token")
        scene_name = official_scenes.get(scene_token, {}).get("name", "")
        expect = drive_scene_for_frame[token]
        all_ok = True
        if expect and scene_token != expect:
            missing["scene_mismatch"] += 1
            all_ok = False
        frame_channels = frame_sd.get(token, {})
        found = 0
        cam_calib_pose_ok = True
        for cam in CAMERAS:
            sd = frame_channels.get(cam)
            if not sd:
                missing["camera_sample_data"] += 1
                all_ok = cam_calib_pose_ok = False
                continue
            # DriveLM v1.1 图片目录名与 nuScenes 原始 samples/... 不同；按文件名对应。
            path = val_dir / cam / Path(sd.get("filename", "")).name
            if path.is_file() and path.stat().st_size > 0:
                found += 1
                image_found += 1
            else:
                missing["camera_image"] += 1
                if len(examples["camera_image"]) < 5:
                    examples["camera_image"].append(str(path))
                all_ok = False
            cs = calib.get(sd.get("calibrated_sensor_token"))
            pose = poses.get(sd.get("ego_pose_token"))
            if not valid_calibration(cs, camera=True) or not valid_pose(pose):
                missing["camera_calibration_or_pose"] += 1
                all_ok = cam_calib_pose_ok = False
            try:
                timestamp_offsets_ms.append(abs(int(sd["timestamp"]) - int(sample["timestamp"])) / 1000)
            except (ValueError, KeyError, TypeError):
                pass
        if cam_calib_pose_ok:
            calibration_good += 1
        lidar_sd = frame_channels.get("LIDAR_TOP")
        lidar_ok = bool(lidar_sd and
            valid_calibration(calib.get(lidar_sd.get("calibrated_sensor_token"))) and
            valid_pose(poses.get(lidar_sd.get("ego_pose_token"))))
        if lidar_ok:
            lidar_good += 1
        else:
            missing["lidar_reference_metadata"] += 1
            all_ok = False
        prev = bool(sample.get("prev") and sample["prev"] in wanted)
        previous_in_subset += int(prev)
        if scene_token:
            frame_times[scene_token].append(int(sample.get("timestamp", 0)))
        if all_ok:
            complete += 1
        rows.append({"sample_token": token, "scene_name": scene_name,
            "available_cameras": found, "calibration_and_pose_ok": cam_calib_pose_ok,
            "lidar_reference_ok": lidar_ok, "previous_frame_in_subset": prev,
            "timestamp_us": sample.get("timestamp", ""), "all_ready": all_ok})

    gaps_s = []
    for timestamps in frame_times.values():
        ordered = sorted(timestamps)
        gaps_s.extend((b - a) / 1e6 for a, b in zip(ordered, ordered[1:]))
    count = len(wanted)
    result = {
        "status": "data-only; no GPU inference or mAP/mIoU calculated",
        "fix": "reverse-index original sample_data by sample_token -> sensor.channel",
        "drivelm_val_scenes": len(groups),
        "drivelm_val_frames": count,
        "duplicate_frame_tokens": duplicate_frame_tokens,
        "metadata_frames_found": count - missing["sample_metadata"],
        "camera_images_found": image_found,
        "camera_images_expected": count * len(CAMERAS),
        "frames_with_camera_calibration_and_pose": calibration_good,
        "frames_with_lidar_reference": lidar_good,
        "fully_ready_frames": complete,
        "direct_previous_frame_in_drivelm_subset": previous_in_subset,
        "inter_frame_seconds_median": statistics.median(gaps_s) if gaps_s else None,
        "inter_frame_seconds_p90": p90(gaps_s),
        "camera_timestamp_offset_ms_median": statistics.median(timestamp_offsets_ms) if timestamp_offsets_ms else None,
        "index_stats": index_stats,
        "problems": dict(missing),
        "examples": dict(examples),
        "source_paths": {"qa_json": str(qa), "trainval_meta": str(meta), "drive_val_images": str(val_dir)},
        "caution": "799 帧稀疏分布，不能直接按连续场景沿用 prev_bev；该预检不证明模型推理成功。"
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "drivelm_799_inference_precheck.json"
    csv_path = out_dir / "drivelm_799_inference_precheck.csv"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else
            ["sample_token", "scene_name", "available_cameras", "calibration_and_pose_ok",
             "lidar_reference_ok", "previous_frame_in_subset", "timestamp_us", "all_ready"])
        writer.writeheader()
        writer.writerows(rows)
    print("[4/4] 数据检查结束，输出已覆盖旧版错误统计：")
    print(f"样本：{count}，官方样本匹配：{result['metadata_frames_found']}/{count}")
    print(f"六路相机图片：{image_found}/{count * 6}")
    print(f"相机内外参/位姿完整：{calibration_good}/{count}")
    print(f"LiDAR 参考元数据完整：{lidar_good}/{count}")
    print(f"满足静态元数据和图像条件：{complete}/{count}")
    print(f"前一关键帧在子集内：{previous_in_subset}/{count}（不能把稀疏帧当连续序列）")
    print(f"索引情况：{index_stats}")
    print(f"问题：{dict(missing)}")
    print(f"JSON：{report}\nCSV：{csv_path}")
    print("注意：仅做静态预检；未生成模型 PKL、未运行推理、未计算成绩。")
    return 0 if count > 0 and complete == count else 2


def main():
    parser = argparse.ArgumentParser(description="修复版 DriveLM 验证集数据关联检查")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    raise SystemExit(inspect(args.root.resolve()))


if __name__ == "__main__":
    main()

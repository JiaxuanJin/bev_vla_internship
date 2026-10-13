#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 DriveLM 799 帧的 BEVFormer 推理专用 PKL（不生成 Markdown）。

只使用本地已经通过静态校验的 DriveLM 六路图像和 nuScenes 原始标定元数据。
- officialscene：保留真实 scene_token，供数据核对和后续研究。
- independent：每帧使用独立的临时时序标识，禁止模型跨稀疏帧复用上一帧 BEV。
  真实的 scene_token 另存于 official_scene_token，不修改 sample token。
注意：生成 PKL != GPU 推理成功，也不等于已经得到 mAP/mIoU。
两种索引只供推理；不包含 3D 训练真值，也不依赖缺失的 LiDAR 点云文件。
"""
import argparse
import json
import os
import pickle
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

CAMERAS = (
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
)
CHANNELS = set(CAMERAS) | {'LIDAR_TOP'}


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def index_table(path):
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f'预期是 JSON 记录列表：{path}')
    return {row['token']: row for row in rows}


def get_drivelm_tokens(qa_file):
    data = load_json(qa_file)
    if not isinstance(data, dict):
        raise ValueError('DriveLM 验证集顶层应为 scene_token -> key_frames')
    frame_to_scene = {}
    for scene_token, scene in data.items():
        frames = scene.get('key_frames', {}) if isinstance(scene, dict) else {}
        for sample_token in frames:
            if sample_token in frame_to_scene:
                raise ValueError(f'DriveLM 出现重复帧：{sample_token}')
            frame_to_scene[sample_token] = scene_token
    return frame_to_scene


def lookup_sample_channels(sd_file, wanted, calibrations, sensors):
    """原始 sample.json 无 data 字段：必须通过 sample_data 反查通道。"""
    rows = load_json(sd_file)
    sample_channels = defaultdict(dict)
    for sd in rows:
        token = sd.get('sample_token')
        if token not in wanted or sd.get('is_key_frame') is False:
            continue
        calib = calibrations.get(sd.get('calibrated_sensor_token'))
        sensor = sensors.get(calib.get('sensor_token')) if calib else None
        channel = sensor.get('channel') if sensor else None
        if channel not in CHANNELS:
            continue  # 五路 RADAR 等无需加入相机版 BEVFormer
        if channel in sample_channels[token]:
            raise ValueError(f'同帧同通道重复: {token} {channel}')
        sample_channels[token][channel] = sd
    return sample_channels


def sensor_to_lidar(sd, cs, pose, l2e_t, l2e_R, e2g_t, e2g_R):
    """遵循 BEVFormer 官方 converter 的 sensor2lidar Rotation/Translation 约定。"""
    s2e_R = Quaternion(cs['rotation']).rotation_matrix
    s2e_t = np.asarray(cs['translation'], dtype=np.float64)
    se2g_R = Quaternion(pose['rotation']).rotation_matrix
    se2g_t = np.asarray(pose['translation'], dtype=np.float64)
    inv_global_rot = np.linalg.inv(e2g_R).T
    inv_lidar_rot = np.linalg.inv(l2e_R).T
    rot = (s2e_R.T @ se2g_R.T) @ (inv_global_rot @ inv_lidar_rot)
    trans = (s2e_t @ se2g_R.T + se2g_t) @ (inv_global_rot @ inv_lidar_rot)
    trans -= e2g_t @ (inv_global_rot @ inv_lidar_rot) + l2e_t @ inv_lidar_rot
    return np.asarray(rot.T, dtype=np.float64), np.asarray(trans, dtype=np.float64)


def camera_info(sd, calib, pose, l2e_t, l2e_R, e2g_t, e2g_R, image_path, channel):
    intr = np.asarray(calib['camera_intrinsic'], dtype=np.float64)
    if intr.shape != (3, 3) or not np.isfinite(intr).all():
        raise ValueError(f'{channel}: 相机内参形状/数值异常')
    R, t = sensor_to_lidar(sd, calib, pose, l2e_t, l2e_R, e2g_t, e2g_R)
    if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError(f'{channel}: 相机外参异常')
    return dict(
        data_path=str(image_path), type=channel, sample_data_token=sd['token'],
        sensor2ego_translation=calib['translation'], sensor2ego_rotation=calib['rotation'],
        ego2global_translation=pose['translation'], ego2global_rotation=pose['rotation'],
        timestamp=sd['timestamp'], sensor2lidar_rotation=R,
        sensor2lidar_translation=t, cam_intrinsic=intr,
    )


def build_infos(root, frame_to_scene, samples, channels, calibrations, poses, scenes):
    infos = []
    errors = []
    image_count = 0
    for sample_token in sorted(frame_to_scene, key=lambda t: (samples[t]['timestamp'], t)):
        sample = samples[sample_token]
        official_scene = sample['scene_token']
        if official_scene != frame_to_scene[sample_token]:
            raise ValueError(f'{sample_token}: DriveLM 与官方 scene_token 不一致')
        if official_scene not in scenes:
            raise ValueError(f'{sample_token}: nuScenes scene 记录缺失')
        chans = channels.get(sample_token, {})
        for chan in CHANNELS:
            if chan not in chans:
                raise ValueError(f'{sample_token}: 缺少 {chan} 的 sample_data')
        lidar = chans['LIDAR_TOP']
        l2e = calibrations[lidar['calibrated_sensor_token']]
        e2g = poses[lidar['ego_pose_token']]
        l2e_R = Quaternion(l2e['rotation']).rotation_matrix
        e2g_R = Quaternion(e2g['rotation']).rotation_matrix
        l2e_t = np.asarray(l2e['translation'], dtype=np.float64)
        e2g_t = np.asarray(e2g['translation'], dtype=np.float64)
        cameras = {}
        for cam in CAMERAS:
            sd = chans[cam]
            cs = calibrations[sd['calibrated_sensor_token']]
            pose = poses[sd['ego_pose_token']]
            image = root / 'data/drivelm/val_data' / cam / Path(sd['filename']).name
            if not image.is_file() or image.stat().st_size == 0:
                raise FileNotFoundError(f'缺少 {sample_token} {cam} 图像：{image}')
            cameras[cam] = camera_info(sd, cs, pose, l2e_t, l2e_R, e2g_t, e2g_R, image, cam)
            image_count += 1
        # 当前只有点云元数据，没有完整 trainval LiDAR .bin 文件。
        # 推理配置必须 use_camera=True, use_lidar=False，不可启用读取点云的 pipeline。
        point_path = root / 'data/bev_seg_nuscenes' / str(lidar['filename'])
        info = dict(
            token=sample_token, lidar_path=str(point_path),
            prev=sample.get('prev', ''), next=sample.get('next', ''),
            can_bus=np.zeros(18, dtype=np.float64), frame_idx=0,
            sweeps=[], cams=cameras,
            scene_token=official_scene,
            official_scene_token=official_scene,
            lidar2ego_translation=l2e['translation'], lidar2ego_rotation=l2e['rotation'],
            ego2global_translation=e2g['translation'], ego2global_rotation=e2g['rotation'],
            timestamp=sample['timestamp'],
        )
        infos.append(info)
    return infos, image_count


def atomic_pickle(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent, prefix='.tmp_', suffix='.pkl', delete=False) as temp:
        temp_name = temp.name
        pickle.dump(payload, temp, protocol=4)
    os.replace(temp_name, path)


def run(root, expected):
    qa = root / 'data/drivelm/QA_dataset_nus/v1_1_val_nus_q_only.json'
    meta = root / 'data/nuscenes_trainval_meta/v1.0-trainval'
    template = root / 'BEVFormer/data/nuscenes/nuscenes_infos_temporal_val.pkl'
    files = [qa, template] + [meta / (x + '.json') for x in
        ('sample', 'sample_data', 'sensor', 'calibrated_sensor', 'ego_pose', 'scene')]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f'必需文件不存在: {path}')
    print('[1/4] 检查本机已跑通的 BEVFormer Mini PKL 格式...', flush=True)
    # 只读取本机之前生成的可信 PKL；pickle 不能用来读取不可信来源。
    with open(template, 'rb') as f:
        mini = pickle.load(f)
    if not isinstance(mini, dict) or 'infos' not in mini or not mini['infos']:
        raise ValueError('Mini PKL 缺少 infos 模板')
    required = {'token', 'cams', 'can_bus', 'lidar2ego_rotation', 'ego2global_rotation',
                'scene_token', 'timestamp', 'prev', 'next', 'frame_idx'}
    missing = required - set(mini['infos'][0])
    if missing:
        raise ValueError(f'Mini 模板字段异常：{sorted(missing)}')
    expected_cams = set(mini['infos'][0]['cams'])
    if expected_cams != set(CAMERAS):
        raise ValueError(f'Mini 模板相机通道与预期不一致：{expected_cams}')
    print('[2/4] 读取 DriveLM 验证帧和 nuScenes Trainval 元数据...', flush=True)
    frame_to_scene = get_drivelm_tokens(qa)
    if len(frame_to_scene) != expected:
        raise ValueError(f'验证帧数异常：得到 {len(frame_to_scene)}，预期 {expected}')
    samples = index_table(meta / 'sample.json')
    calibrations = index_table(meta / 'calibrated_sensor.json')
    sensors = index_table(meta / 'sensor.json')
    poses = index_table(meta / 'ego_pose.json')
    scenes = index_table(meta / 'scene.json')
    channel_index = lookup_sample_channels(meta / 'sample_data.json', set(frame_to_scene), calibrations, sensors)
    print('[3/4] 生成 799 帧六路相机与坐标变换，并进行数值校验...', flush=True)
    official_infos, matched_images = build_infos(root, frame_to_scene, samples, channel_index, calibrations, poses, scenes)
    expected_images = expected * 6
    if len(official_infos) != expected or matched_images != expected_images:
        raise RuntimeError(f'结果未达标：{len(official_infos)}/{expected}, {matched_images}/{expected_images}')
    # model.forward_test 对相同 scene_token 可能保留 prev_bev；对于稀疏采样，安全做法是独立帧。
    independent_infos = []
    for src in official_infos:
        row = src.copy()
        row['scene_token'] = 'drive_independent_' + src['token']
        row['frame_idx'] = 0
        row['prev'], row['next'] = '', ''
        independent_infos.append(row)
    dest = root / 'data/drivelm'
    official_path = dest / 'nuscenes_infos_temporal_drivelm799_officialscene.pkl'
    independent_path = dest / 'nuscenes_infos_temporal_drivelm799_independent.pkl'
    payload_meta = {'version': 'v1.0-trainval'}
    print('[4/4] 保存官方场景索引和独立帧索引（不修改 Mini PKL）...', flush=True)
    atomic_pickle(official_path, {'infos': official_infos, 'metadata': payload_meta})
    atomic_pickle(independent_path, {'infos': independent_infos, 'metadata': payload_meta})
    # 写入精简 JSON，方便对照，不产生 Markdown。
    outputs = root / 'outputs/drivelm_check'
    outputs.mkdir(parents=True, exist_ok=True)
    result = dict(
        state='only inference PKL prepared; no GPU inference or metrics executed',
        samples=len(official_infos), camera_images=matched_images,
        official_scenes=len(set(x['official_scene_token'] for x in official_infos)),
        officialscene_pkl=str(official_path), independent_pkl=str(independent_path),
        has_3d_annotations=False, has_raw_lidar_points=False,
        independent_mode='sequence id is unique per frame to prevent reuse of previous BEV',
        caution='Independent frames are NOT the original temporal BEVFormer protocol. Evaluation requires a separate subset methodology.'
    )
    out = outputs / 'drivelm_799_pkl_build_summary.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[完成] 关键帧: {len(official_infos)}/{expected}，相机: {matched_images}/{expected_images}')
    print(f'[完成] 原始场景数量：{result["official_scenes"]}')
    print(f'[保存] 官方场景版：{official_path}')
    print(f'[保存] 独立帧推理版：{independent_path}')
    print(f'[保存] 核验摘要：{out}')
    print('[注意] 这一步只生成和检查 PKL，尚未进行数据加载或 GPU 推理。')


def main():
    parser = argparse.ArgumentParser(description='DriveLM 799 帧 BEVFormer PKL 构建')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--expected', type=int, default=799)
    args = parser.parse_args()
    run(args.root.resolve(), args.expected)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 DriveLM v1.1 帧 token 与本地 BEVFormer nuScenes Mini 信息索引的适配情况。

只读取 DriveLM JSON 和 BEVFormer 已有 PKL；只在 outputs/ 下输出 JSON/CSV。
不下载、不解压、不修改数据集，也不生成 Markdown。
"""

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

CAMERAS = (
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
)


def read_drivelm(path):
    """官方 DriveLM 格式：scene_token -> key_frames -> frame_token。"""
    with path.open('r', encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f'不是 DriveLM 的场景字典：{path}')
    frames = {}
    for scene_token, scene in raw.items():
        if not isinstance(scene, dict):
            continue
        key_frames = scene.get('key_frames', {})
        if not isinstance(key_frames, dict):
            continue
        for token, info in key_frames.items():
            paths = info.get('image_paths', {}) if isinstance(info, dict) else {}
            frames[str(token)] = {
                'scene_token': str(scene_token),
                'image_paths': paths if isinstance(paths, dict) else {},
            }
    return frames, len(raw)


def read_bevformer(path):
    """PKL 仅加载本机预处理产生且来源可信的文件。"""
    with path.open('rb') as f:
        raw = pickle.load(f)
    infos = raw['infos'] if isinstance(raw, dict) and 'infos' in raw else raw
    if not isinstance(infos, (list, tuple)):
        raise ValueError(f'不是 infos 列表：{path}')
    result = {}
    for sample in infos:
        if isinstance(sample, dict) and sample.get('token'):
            result[str(sample['token'])] = sample
    return result


def correct_shape(value, shape):
    try:
        arr = np.asarray(value, dtype=np.float64)
        return arr.shape == shape and bool(np.isfinite(arr).all())
    except (ValueError, TypeError):
        return False


def calib_check(info):
    """验证 BEVFormer 图像投影所需的内参和相机到 LiDAR 的外参。"""
    cams = info.get('cams', {}) if isinstance(info, dict) else {}
    if not isinstance(cams, dict):
        return (0, 0, 0)
    cam_found = intrinsics = extrinsics = 0
    for name in CAMERAS:
        cam = cams.get(name)
        if not isinstance(cam, dict):
            continue
        cam_found += 1
        if correct_shape(cam.get('cam_intrinsic'), (3, 3)):
            intrinsics += 1
        if (correct_shape(cam.get('sensor2lidar_rotation'), (3, 3))
                and correct_shape(cam.get('sensor2lidar_translation'), (3,))):
            extrinsics += 1
    return cam_found, intrinsics, extrinsics


def find_bev_image(project_root, raw_path):
    """兼容 info 中绝对路径和 data/nuscenes/samples 相对路径。"""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return False
    text = raw_path.replace('\\', '/').lstrip('./')
    path = Path(raw_path)
    if path.is_absolute() and path.is_file():
        return True
    bev = project_root / 'BEVFormer'
    rel = PurePosixPath(text)
    candidates = [bev / text, project_root / text, bev / 'data' / 'nuscenes' / text]
    if 'data/nuscenes/' in text:
        candidates.append(bev / 'data' / 'nuscenes' / text.split('data/nuscenes/', 1)[1])
    if 'samples/' in text:
        candidates.append(bev / 'data' / 'nuscenes' / 'samples' / text.split('samples/', 1)[1])
    return any(p.is_file() for p in candidates)


def assess_frame(root, drivelm_frame, bevformer_info):
    """检查六路相机标定、图片存在性以及两个数据源图片名是否一致。"""
    count, intrinsic, extrinsic = calib_check(bevformer_info)
    bev_cams = bevformer_info.get('cams', {}) if isinstance(bevformer_info, dict) else {}
    qa_paths = drivelm_frame.get('image_paths', {})
    found_image = name_agree = 0
    for cam in CAMERAS:
        meta = bev_cams.get(cam, {}) if isinstance(bev_cams, dict) else {}
        bev_img = meta.get('data_path', '') if isinstance(meta, dict) else ''
        qa_img = qa_paths.get(cam, '') if isinstance(qa_paths, dict) else ''
        if find_bev_image(root, bev_img):
            found_image += 1
        if (isinstance(bev_img, str) and isinstance(qa_img, str)
                and bev_img and qa_img
                and PurePosixPath(bev_img.replace('\\', '/')).name
                == PurePosixPath(qa_img.replace('\\', '/')).name):
            name_agree += 1
    return {
        '六路相机元信息数': count,
        '有效内参数': intrinsic,
        '有效外参数': extrinsic,
        'BEVFormer图片存在数': found_image,
        '两套图片文件名一致数': name_agree,
        '标定完整': count == intrinsic == extrinsic == 6,
        '六路图像存在': found_image == 6,
        '文件名完全一致': name_agree == 6,
    }


def run(root, output_dir):
    qa = root / 'data' / 'drivelm' / 'QA_dataset_nus'
    info_dir = root / 'BEVFormer' / 'data' / 'nuscenes'
    input_files = {
        'DriveLM训练': qa / 'v1_1_train_nus.json',
        'DriveLM验证': qa / 'v1_1_val_nus_q_only.json',
        'Mini训练': info_dir / 'nuscenes_infos_temporal_train.pkl',
        'Mini验证': info_dir / 'nuscenes_infos_temporal_val.pkl',
    }
    missing = [str(p) for p in input_files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError('缺少必要文件：\n' + '\n'.join(missing))

    train_frames, train_scenes = read_drivelm(input_files['DriveLM训练'])
    val_frames, val_scenes = read_drivelm(input_files['DriveLM验证'])
    mini_train = read_bevformer(input_files['Mini训练'])
    mini_val = read_bevformer(input_files['Mini验证'])
    dl_splits = {'DriveLM训练': train_frames, 'DriveLM验证': val_frames}
    mini_splits = {'Mini训练': mini_train, 'Mini验证': mini_val}

    summary = {
        '说明': '仅为数据交集和输入元数据适配检查；不是 DriveLM 官方3D检测mAP或分割mIoU',
        'DriveLM场景数': {'训练': train_scenes, '验证': val_scenes},
        'DriveLM关键帧数': {'训练': len(train_frames), '验证': len(val_frames)},
        'BEVFormer_Mini索引帧数': {'训练': len(mini_train), '验证': len(mini_val)},
        '交集统计': {},
        '标定及图像检查': {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    print('\n========== DriveLM × BEVFormer 数据适配检查 ==========')
    print(f'DriveLM 训练={len(train_frames)}，验证={len(val_frames)}')
    print(f'BEVFormer Mini 训练={len(mini_train)}，验证={len(mini_val)}')

    for dl_name, dl_frames in dl_splits.items():
        for mini_name, mini_infos in mini_splits.items():
            tokens = sorted(dl_frames.keys() & mini_infos.keys())
            key = f'{dl_name} × {mini_name}'
            summary['交集统计'][key] = len(tokens)
            health = Counter()
            for token in tokens:
                result = assess_frame(root, dl_frames[token], mini_infos[token])
                for flag in ('标定完整', '六路图像存在', '文件名完全一致'):
                    health[flag] += int(result[flag])
                rows.append({
                    'DriveLM划分': dl_name, 'BEVFormer划分': mini_name,
                    'sample_token': token,
                    'scene_token': dl_frames[token]['scene_token'],
                    **result,
                })
            summary['标定及图像检查'][key] = {
                '交集帧数': len(tokens),
                '六路内外参齐全': health['标定完整'],
                'Mini六路图片存在': health['六路图像存在'],
                '两套六路图片名一致': health['文件名完全一致'],
            }
            print(f'[{key}] 重合={len(tokens)}，标定完整={health["标定完整"]}，'
                  f'图片存在={health["六路图像存在"]}，文件名一致={health["文件名完全一致"]}')

    # CSV 只保存对应关系；不改动 PKL、JSON、图片和模型文件。
    csv_path = output_dir / 'drivelm_bevformer_token_matches.csv'
    columns = ['DriveLM划分', 'BEVFormer划分', 'sample_token', 'scene_token',
               '六路相机元信息数', '有效内参数', '有效外参数',
               'BEVFormer图片存在数', '两套图片文件名一致数',
               '标定完整', '六路图像存在', '文件名完全一致']
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / 'drivelm_bevformer_alignment_summary.json'
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n[关键提醒] DriveLM 验证帧与 Mini 训练帧的重合，不能当作 Mini 验证结果。')
    print('[关键提醒] 仅有 QA + JPG 不能直接生成官方 3D 检测 mAP / BEV 分割 mIoU。')
    print(f'[保存] {json_path}')
    print(f'[保存] {csv_path}')
    print('[完成] 本脚本不会生成 Markdown。')


def main():
    parser = argparse.ArgumentParser(description='检查 DriveLM / BEVFormer 数据对应关系（中文输出）')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        run(root, root / 'outputs' / 'drivelm_check')
    except (OSError, ValueError, KeyError, pickle.UnpicklingError) as exc:
        print(f'[错误] {exc}')
        raise SystemExit(1)


if __name__ == '__main__':
    main()

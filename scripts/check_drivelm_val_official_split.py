#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 DriveLM v1.1 验证集与完整 nuScenes Trainval 官方数据划分的关系。

只读取本地 JSON；将结果写入 outputs/drivelm_check/ 下的 JSON、CSV。
不下载、不修改原始数据，不生成 Markdown，不运行模型。
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRIVELM_JSON = ROOT / 'data/drivelm/QA_dataset_nus/v1_1_val_nus_q_only.json'
METADATA = ROOT / 'data/nuscenes_trainval_meta/v1.0-trainval'
OUT = ROOT / 'outputs/drivelm_check'


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f'文件不存在：{path}')
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def load_drivelm_frames():
    """DriveLM 格式：scene_token -> key_frames -> sample/frame_token。"""
    raw = load_json(DRIVELM_JSON)
    if not isinstance(raw, dict):
        raise ValueError('DriveLM 验证集 JSON 不是期望的场景字典')
    frames = {}
    for scene_token, record in raw.items():
        if not isinstance(record, dict):
            continue
        key_frames = record.get('key_frames', {})
        if not isinstance(key_frames, dict):
            continue
        for token, details in key_frames.items():
            if token in frames:
                raise ValueError(f'发现重复关键帧 token：{token}')
            image_paths = details.get('image_paths', {}) if isinstance(details, dict) else {}
            frames[str(token)] = {
                'DriveLM场景token': str(scene_token),
                '相机路径数量': len(image_paths) if isinstance(image_paths, dict) else 0,
            }
    return frames, len(raw)


def official_scene_splits():
    """使用 nuScenes-devkit 的官方 split 列表，不自行猜测 train / val。"""
    try:
        from nuscenes.utils.splits import create_splits_scenes
        splits = create_splits_scenes()
        return {name: set(names) for name, names in splits.items()}, None
    except Exception as exc:
        return {}, f'{type(exc).__name__}: {exc}'


def main():
    frames, scene_count = load_drivelm_frames()
    print('========== DriveLM 验证集 × nuScenes 官方划分核查 ==========')
    print(f'[DriveLM] 场景={scene_count}，关键帧={len(frames)}')

    # 仅加载 sample 和 scene 两张元数据表；暂不加载体积更大的 sample_data / annotation。
    samples = load_json(METADATA / 'sample.json')
    scenes = load_json(METADATA / 'scene.json')
    if not isinstance(samples, list) or not isinstance(scenes, list):
        raise ValueError('nuScenes 元数据格式与预期不符')
    wanted = set(frames)
    sample_by_token = {s['token']: s for s in samples if s.get('token') in wanted}
    scene_by_token = {s['token']: s for s in scenes if s.get('token')}
    official_splits, split_error = official_scene_splits()

    print(f'[Trainval 元数据] 总 sample={len(samples)}，总场景={len(scenes)}')
    print(f'[匹配] DriveLM 验证帧在 Trainval 元数据中={len(sample_by_token)}/{len(frames)}')
    if split_error:
        print(f'[提示] 官方 train/val 场景列表暂不可用：{split_error}')
        print('       本轮不推断样本属于哪个官方划分。')

    split_count = Counter()
    missing = []
    scene_mismatch = []
    rows = []
    for token, drivelm in frames.items():
        sample = sample_by_token.get(token)
        if sample is None:
            missing.append(token)
            rows.append({'sample_token': token, 'scene_token': drivelm['DriveLM场景token'],
                         'scene_name': '', '官方划分': 'metadata_missing', '场景token一致': '',
                         '六路图像路径数量': drivelm['相机路径数量']})
            continue
        scene_token = sample['scene_token']
        scene = scene_by_token.get(scene_token, {})
        scene_name = scene.get('name', '')
        hit = [name for name in ('train', 'val') if scene_name in official_splits.get(name, set())]
        split_name = hit[0] if len(hit) == 1 else ('未识别' if official_splits else '未检查')
        same_scene = scene_token == drivelm['DriveLM场景token']
        if not same_scene:
            scene_mismatch.append(token)
        split_count[split_name] += 1
        rows.append({'sample_token': token, 'scene_token': scene_token,
                     'scene_name': scene_name, '官方划分': split_name,
                     '场景token一致': same_scene,
                     '六路图像路径数量': drivelm['相机路径数量']})

    print(f'[一致性] 与 DriveLM 场景 token 不一致={len(scene_mismatch)}')
    print(f'[分布] {dict(split_count)}')
    print(f'[缺失] 元数据中缺少的 sample token={len(missing)}')
    print('[提醒] 这里核查的是样本身份和官方划分，尚未执行完整 799 帧模型推理，也未计算正式 mAP/mIoU。')
    print('[提醒] 如果 DriveLM 验证帧来自 nuScenes train，不能把这些帧宣称为 nuScenes 官方 val 成绩。')

    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        '实验性质': '数据对应关系与官方划分核查（非模型性能评估）',
        'DriveLM验证集场景数': scene_count,
        'DriveLM验证集关键帧数': len(frames),
        'Trainval元数据总样本数': len(samples),
        '匹配Trainval样本数': len(sample_by_token),
        '场景token不一致数量': len(scene_mismatch),
        '官方划分帧数': dict(split_count),
        '官方划分加载问题': split_error,
        '元数据缺失样本示例': missing[:10],
        '场景token不一致示例': scene_mismatch[:10],
        '是否完成DriveLM检测和分割评估': False,
    }
    report_path = OUT / 'drivelm_val_official_split_check.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    csv_path = OUT / 'drivelm_val_official_split_tokens.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_token','scene_token','scene_name',
                                                '官方划分','场景token一致','六路图像路径数量'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'[保存] {report_path.relative_to(ROOT)}')
    print(f'[保存] {csv_path.relative_to(ROOT)}')
    print('[完成] 只生成 JSON/CSV，不创建 Markdown。')

    # 数据不完整时退出为失败，避免遗漏关键提示。
    if missing or scene_mismatch:
        print('[注意] 存在数据缺失/场景不一致，需要先排查，暂不可进入正式验证。')
        return 2
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'[错误] {type(error).__name__}: {error}', file=sys.stderr)
        sys.exit(1)

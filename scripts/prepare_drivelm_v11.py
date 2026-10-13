#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DriveLM v1.1 数据准备与检查（不下载、不默认解压大文件）。

用法：
  python scripts/prepare_drivelm_v11.py --downloads /mnt/c/Users/<Windows用户名>/Downloads
  python scripts/prepare_drivelm_v11.py --downloads /mnt/c/.../Downloads --extract-val

功能：检查 Windows 下载文件、复制小型 JSON 到项目目录、检查 ZIP 结构；
对比 DriveLM 帧 token 与现有 nuScenes Mini，生成中文 Markdown 核验报告。
注意：DriveLM 的 q_only 标注不能替代 nuScenes 3D GT 或 BEV 分割标注。
"""

import argparse
import json
import shutil
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data' / 'drivelm'
QA_DIR = DATA_DIR / 'QA_dataset_nus'
DOCS_DIR = PROJECT_ROOT / 'docs'
JSON_FILES = ['v1_1_val_nus_q_only.json', 'v1_1_train_nus.json']
ZIP_FILES = ['drivelm_nus_imgs_val.zip', 'drivelm_nus_imgs_train.zip']


def mb(size):
    return f'{size / 1024**2:.1f} MiB'


def find_source(name, downloads):
    """优先检查数据目录，再检查用户指定的 Windows 下载目录。"""
    for folder in [QA_DIR, DATA_DIR, downloads]:
        if folder and (folder / name).is_file():
            return folder / name
    return None


def ensure_json(name, downloads):
    src = find_source(name, downloads)
    if not src:
        print(f' [缺失] {name}')
        return None
    dst = QA_DIR / name
    if src != dst:
        # 小型标注文件可以复制；跳过已有且大小相同的文件。
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            print(f' [复制] {src} -> {dst}', flush=True)
            shutil.copy2(src, dst)
    print(f' [已找到] {dst.name} | {mb(dst.stat().st_size)}')
    return dst


def summarize_json(path):
    """按官方 scene_token -> key_frames 结构统计，不修改原文件。"""
    try:
        with path.open('r', encoding='utf-8') as file:
            data = json.load(file)
    except (OSError, ValueError, MemoryError) as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}

    if not isinstance(data, dict):
        return {'error': f'顶层不是 dict，而是 {type(data).__name__}；请查看官方格式'}

    # 兼容不同版本可能使用的 wrapper。
    scenes = data
    if 'scenes' in data and isinstance(data['scenes'], dict):
        scenes = data['scenes']

    frame_tokens = set()
    images = Counter()
    qa_types = Counter()
    qa_total = 0
    valid_scenes = 0

    for scene in scenes.values():
        if not isinstance(scene, dict):
            continue
        frames = scene.get('key_frames')
        if not isinstance(frames, dict):
            continue
        valid_scenes += 1
        for token, frame in frames.items():
            frame_tokens.add(str(token))
            if not isinstance(frame, dict):
                continue
            image_paths = frame.get('image_paths', {})
            if isinstance(image_paths, dict):
                for cam, value in image_paths.items():
                    if isinstance(value, str) and value.strip():
                        images[str(cam)] += 1
            qa = frame.get('QA', {})
            if isinstance(qa, dict):
                for name, entries in qa.items():
                    if isinstance(entries, list):
                        qa_types[str(name)] += len(entries)
                        qa_total += len(entries)

    return dict(scene_count=valid_scenes,
                frame_count=len(frame_tokens),
                frame_tokens=frame_tokens,
                images=dict(images), qa_types=dict(qa_types), qa_total=qa_total,
                top_keys=list(data.keys())[:5])


def zip_info(path):
    try:
        with zipfile.ZipFile(path) as archive:
            files = [x for x in archive.infolist() if not x.is_dir()]
            cameras = Counter()
            for item in files:
                for cam in ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'):
                    if f'/{cam}/' in '/' + item.filename:
                        cameras[cam] += 1
                        break
            return {'count': len(files), 'uncompressed': sum(x.file_size for x in files),
                    'cameras': dict(cameras),
                    'samples': [x.filename for x in files[:3]]}
    except (zipfile.BadZipFile, OSError) as exc:
        return {'error': f'{type(exc).__name__}: {exc}'}


def safe_extract_val(path, target, uncompressed):
    """只在明确传入 --extract-val 时解压，并防止危险路径和磁盘空间不足。"""
    free = shutil.disk_usage(target).free
    if free < uncompressed + 512 * 1024**2:
        print(f' [暂停解压] 空闲 {mb(free)}，解压后需约 {mb(uncompressed)}，建议另留 512 MiB。')
        return
    root = target.resolve()
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            dest = (root / entry.filename).resolve()
            if not dest.is_relative_to(root):
                raise ValueError(f'ZIP 中包含非法路径：{entry.filename}')
        print(' [解压中] 将验证集图片解压到 ' + str(target), flush=True)
        archive.extractall(target)
    print(' [解压完成] ' + str(target), flush=True)


def mini_tokens():
    # nuScenes Mini 完整 metadata 中的 sample token 是可核实的对照对象。
    p = PROJECT_ROOT / 'BEVFormer' / 'data' / 'nuscenes' / 'v1.0-mini' / 'sample.json'
    if not p.exists():
        return None
    try:
        with p.open(encoding='utf-8') as f:
            rows = json.load(f)
        return {str(x['token']) for x in rows if 'token' in x}
    except (ValueError, OSError, KeyError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser(description='DriveLM v1.1 下载结果核验与数据结构分析')
    ap.add_argument('--downloads', type=Path, default=None,
                    help='Windows 下载目录在 WSL 中的路径，例如 /mnt/c/Users/XXX/Downloads')
    ap.add_argument('--extract-val', action='store_true', help='显式解压验证集 ZIP（默认不解压）')
    args = ap.parse_args()
    downloads = args.downloads.expanduser() if args.downloads else None
    QA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print('\n=== DriveLM v1.1 本地数据准备 ===', flush=True)
    print('项目数据目录：', DATA_DIR)
    print('Windows 下载目录：', downloads if downloads else '未指定（只检查项目目录）')
    print('\n[1] 检查语言标注 JSON')
    summaries = {}
    for name in JSON_FILES:
        p = ensure_json(name, downloads)
        if p:
            print(f' [分析中] {name}', flush=True)
            summaries[name] = summarize_json(p)
            summary = summaries[name]
            if 'error' in summary:
                print(' [格式问题]', summary['error'])
            else:
                print(f" [结构] 场景={summary['scene_count']}，关键帧={summary['frame_count']}，QA条目={summary['qa_total']}")

    print('\n[2] 检查图像 ZIP（不默认解压、不复制大文件）')
    archives = {}
    for name in ZIP_FILES:
        p = find_source(name, downloads)
        if not p:
            print(f' [缺失] {name}')
            continue
        info = zip_info(p)
        archives[name] = (p, info)
        if 'error' in info:
            print(f" [ZIP 不完整或格式错误] {name}: {info['error']}")
        else:
            print(f" [已找到] {name} | 压缩大小={mb(p.stat().st_size)} | 图片数={info['count']} | 解压大小={mb(info['uncompressed'])}")

    print('\n[3] 对比 nuScenes Mini 的 sample tokens')
    existing = mini_tokens()
    if existing is None:
        print(' [提示] 未找到 Mini metadata sample.json，暂不计算交集。')
    else:
        print(f' [Mini] {len(existing)} 个 sample tokens')
        for name, s in summaries.items():
            if 'frame_tokens' in s:
                matched = len(existing & s['frame_tokens'])
                print(f' [{name}] 与 Mini 交集：{matched} / {s["frame_count"]}')

    report = [
        '# DriveLM v1.1 数据准备与适配检查', '',
        f'- 检查时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '- 数据来源：OpenDriveLab/DriveLM 官方发布的 v1.1 链接；记录本地实际下载状态。',
        '- 注意：此脚本只检查数据，不运行 BEVFormer，不声称完成正式验证集的 mAP / mIoU。',
        '', '## 1. 语言标注', ''
    ]
    for name in JSON_FILES:
        s = summaries.get(name)
        if not s:
            report.append(f'- {name}：尚未找到。')
        elif 'error' in s:
            report.append(f'- {name}：无法解析，{s["error"]}。')
        else:
            report.append(f'- {name}：{s["scene_count"]} 个场景，{s["frame_count"]} 个关键帧；QA 条目 {s["qa_total"]}。')
            report.append(f'  - QA 类型：{s["qa_types"]}')
            report.append(f'  - 六路摄像头引用：{s["images"]}')
            if existing is not None:
                report.append(f'  - 与本地 nuScenes Mini 样本 token 交集：{len(existing & s["frame_tokens"])}。')
    report += ['', '## 2. 图像压缩包', '']
    for name in ZIP_FILES:
        item = archives.get(name)
        if not item:
            report.append(f'- {name}：尚未找到。')
        else:
            p, info = item
            if 'error' in info:
                report.append(f'- {name}：{info["error"]}')
            else:
                report.append(f'- {name}：压缩大小 {mb(p.stat().st_size)}；文件数 {info["count"]}；解压后约 {mb(info["uncompressed"])}。')
                report.append(f'  - 摄像头文件分布：{info["cameras"]}')
    report += ['', '## 3. 注意事项与下一步', '',
               '1. DriveLM v1.1 验证集的 `q_only` 是问题集，不含官方完整答案和 nuScenes 3D 检测 GT。',
               '2. 图像子集并不能替代完整 nuScenes 的 camera calibration、ego_pose、3D annotations 等元信息。',
               '3. 正式 BEVFormer 3D 检测 mAP 需要一致的 nuScenes 原始标注、数据划分和样本 token。',
               '4. 没有对应的 BEV 语义分割预测头与 GT，就不能计算 BEV 分割 mIoU。',
               '5. 下载后应核对官方文件来源，避免混用 DriveLM v1.0/v1.1 或误把训练集当验证集。', '']
    dest = DOCS_DIR / 'drivelm_v11_数据准备检查.md'
    dest.write_text('\n'.join(report), encoding='utf-8')
    print('\n[已保存中文报告]', dest)

    if args.extract_val:
        record = archives.get('drivelm_nus_imgs_val.zip')
        if not record or 'error' in record[1]:
            print('[未解压] 找不到有效验证集 ZIP，请先下载完成。')
            return
        safe_extract_val(record[0], DATA_DIR, record[1]['uncompressed'])

    print('\n完成。此脚本未下载任何内容，也没有修改已存在的 BEVFormer / nuScenes Mini 数据。')


if __name__ == '__main__':
    main()

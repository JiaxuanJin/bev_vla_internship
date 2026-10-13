#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段一交付本机验收：核实文档、相对图片、实测指标和训练 QA 转换。

用法（项目根目录）：python scripts/check_stage1_submission.py

仅检查已存在的文件，不重新训练、不推理、不修改原始数据；
结果写入 outputs/stage1_checks/stage1_audit.json。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

EXPECTED_DOCS = (
    '00_阶段一交付导航.md', '01_DriveLM数据集结构与标注说明.md',
    '02_BEVFormer原理与代码结构.md', '03_基线性能测试报告.md',
    '04_可视化成果索引.md', '05_复现和GitHub提交前检查.md',
)
REQUIRED_SCRIPTS = (
    'run_drivelm_official_preprocess.py', 'test_drivelm_799_bevformer_loader.py',
    'run_drivelm_799_det_all.py', 'evaluate_drivelm_799_det_subset.py',
    'run_drivelm_799_seg_all.py', 'extract_bev_features.py',
)
IMAGE_RE = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')


def get_json(path: Path):
    with path.open('r', encoding='utf-8') as stream:
        return json.load(stream)


def within(value, expected, tolerance):
    return isinstance(value, (int, float)) and math.isfinite(value) and abs(value - expected) < tolerance


def run(root: Path, strict: bool):
    """独立于 CUDA 的完整交付检查，输出详尽检查项和具体失败原因。"""
    root = root.resolve()
    checks = []

    def record(name, ok, info='', important=True):
        row = {'name': name, 'passed': bool(ok), 'important': important, 'detail': str(info)}
        checks.append(row)
        icon = '通过' if ok else ('失败' if important else '提醒')
        print(f'[{icon}] {name}' + (f'：{info}' if info else ''), flush=True)

    # 1. 文档齐备，图片路径从 Markdown 文件位置计算，杜绝 GitHub 图片失效。
    docs = root / 'docs' / 'stage1'
    for name in EXPECTED_DOCS:
        path = docs / name
        record(f'文档 {name}', path.is_file(), path.relative_to(root))

    logbook = root / 'docs/logbook_2026-10-12.md'
    record('10月12日中文 Logbook', logbook.is_file(), logbook.relative_to(root))
    record('项目概览草案', (root / 'STAGE1_OVERVIEW.md').is_file(), '不会覆盖原 README.md')

    missing_images = []
    linked_images = set()
    md_files = [docs / n for n in EXPECTED_DOCS] + [logbook, root / 'STAGE1_OVERVIEW.md']
    for md in md_files:
        if not md.is_file():
            continue
        body = md.read_text(encoding='utf-8')
        for raw_target in IMAGE_RE.findall(body):
            if re.match(r'^https?://', raw_target):
                continue
            target = (md.parent / raw_target.split(' ')[0]).resolve()
            linked_images.add(str(target))
            if not target.is_file():
                missing_images.append(f'{md.relative_to(root)} -> {raw_target}')
    record('Markdown 相对图片链接', not missing_images,
           f'引用 {len(linked_images)} 张，缺失 {len(missing_images)} 张' +
           ('；' + '；'.join(missing_images[:4]) if missing_images else ''))

    # 2. 检查核心可复现代码，不调用重型 CUDA 推理。
    for name in REQUIRED_SCRIPTS:
        path = root / 'scripts' / name
        record(f'实验脚本 {name}', path.is_file(), path.relative_to(root))

    qa_dir = root / 'outputs/drivelm_official_preprocess/full'
    qa_summary = qa_dir / 'summary.json'
    qa_file = qa_dir / 'test_llama.json'
    record('DriveLM 官方转换 QA 文件', qa_file.is_file(), qa_file.relative_to(root))
    if qa_summary.is_file():
        try:
            qa = get_json(qa_summary)
            record('DriveLM 转换场景/帧/QA 数量',
                   qa.get('scene_count') == 696 and qa.get('keyframe_count') == 4072 and
                   qa.get('qa_record_count') == 29450,
                   f"{qa.get('scene_count')} 场景，{qa.get('keyframe_count')} 帧，{qa.get('qa_record_count')} 条")
            record('DriveLM 六路图像路径和结构检查',
                   qa.get('unique_image_paths') == 24432 and
                   qa.get('missing_image_paths') == 0 and
                   qa.get('invalid_conversation_records') == 0,
                   f"唯一图片={qa.get('unique_image_paths')}，缺图={qa.get('missing_image_paths')}，结构异常={qa.get('invalid_conversation_records')}")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            record('DriveLM 转换汇总 JSON', False, repr(exc))
    else:
        record('DriveLM 转换汇总 JSON', False, qa_summary.relative_to(root))

    # 3. 一致性地检查检测推理存档与检测指标，不读取 799 份大体积 NPZ。
    frame_dir = root / 'outputs/drivelm_check/det799/frames'
    frame_files = list(frame_dir.glob('frame_????.npz')) if frame_dir.is_dir() else []
    record('799 帧检测 NPZ 预测文件', len(frame_files) == 799,
           f'找到 {len(frame_files)} / 799 个文件')

    det_path = root / 'outputs/drivelm_check/det799_subset_eval/metrics_summary.json'
    if det_path.is_file():
        try:
            det = get_json(det_path)
            record('真实子集 3D mAP', within(det.get('mean_ap'), 0.133870, 0.0005),
                   f"mAP={det.get('mean_ap')}")
            record('真实子集 NDS', within(det.get('nd_score'), 0.257912, 0.0005),
                   f"NDS={det.get('nd_score')}")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            record('3D 检测指标 JSON', False, repr(exc))
    else:
        record('3D 检测指标 JSON', False, det_path.relative_to(root))

    protocol_path = root / 'outputs/drivelm_check/det799_subset_eval/evaluation_protocol.json'
    record('检测子集协议 JSON', protocol_path.is_file(), protocol_path.relative_to(root))

    # 4. 检查分割指标以及完成标记，既防止读中间进度，又防止把含背景错写成前景。
    seg_path = root / 'outputs/drivelm_check/seg799_independent_eval/metrics_summary.json'
    if seg_path.is_file():
        try:
            seg = get_json(seg_path)
            record('分割完整样本数量', seg.get('evaluated_samples') == 799 and seg.get('complete') is True,
                   f"{seg.get('evaluated_samples')} / 799，complete={seg.get('complete')}")
            record('分割前景 mIoU', within(seg.get('foreground_mIoU'), 0.051211, 0.0001),
                   f"foreground_mIoU={seg.get('foreground_mIoU')}")
            record('分割含背景 mIoU', within(seg.get('mIoU_including_background'), 0.222220, 0.0001),
                   f"mIoU_including_background={seg.get('mIoU_including_background')}")
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            record('BEV 分割指标 JSON', False, repr(exc))
    else:
        record('BEV 分割指标 JSON', False, seg_path.relative_to(root))

    report = {
        'purpose': 'Stage 1 local deliverable verification (no training or inference)',
        'root': str(root), 'checked_files': len(checks),
        'passed': sum(x['passed'] for x in checks),
        'failed': sum(not x['passed'] for x in checks),
        'all_required_passed': all(x['passed'] for x in checks if x['important']),
        'checks': checks,
        'limitations': [
            '799-frame independent subset; not official full nuScenes val benchmark',
            'Low segmentation mIoU root cause unresolved',
            'Official QA conversion differs from full nuScenes BEVFormer train PKL generation',
            'GitHub repository creation/upload remains a separate task',
        ],
    }
    out = root / 'outputs/stage1_checks/stage1_audit.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n共检查 {len(checks)} 项，通过 {report["passed"]}，未通过 {report["failed"]}')
    print('验收报告：', out)
    print('提醒：自检可证明文件与主要数值存在，不代替对分割低精度根因的独立技术验证。')
    if strict and not report['all_required_passed']:
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description='第一阶段可交付文档与指标本机一致性检查')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1],
                        help='项目根目录，默认取 scripts 的上一级')
    parser.add_argument('--no-strict', action='store_true',
                        help='仅展示结果，不因缺失文件返回非零退出状态')
    args = parser.parse_args()
    raise SystemExit(run(args.root, strict=not args.no_strict))


if __name__ == '__main__':
    main()

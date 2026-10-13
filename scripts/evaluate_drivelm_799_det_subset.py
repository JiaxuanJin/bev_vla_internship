#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DriveLM 799帧独立帧检测：原始NPZ -> nuScenes框格式 -> 有GT支持的子集mAP/NDS。

说明：
- 不加载模型，不重复GPU推理；复用 run_drivelm_799_det_all.py 保存的全部799份预测。
- 用当前 BEVFormer / mmdet3d 的官方格式转换逻辑处理LiDAR框、坐标系和类别。
- 使用 nuScenes Devkit 的真实3D真值、距离过滤和官方检测指标算法。
- 指标仅适用于DriveLM的799帧子集、关闭历史BEV的独立帧实验，不是全量nuScenes val。
- 不对预测做显示用的0.3置信度阈值截断，也不向旧检测仓库写文件。

执行：
    cd ~/projects/bev_vla
    python -u scripts/evaluate_drivelm_799_det_subset.py \
        2>&1 | tee outputs/drivelm_check/det799_subset_eval.log
"""

import argparse
import copy
import hashlib
import importlib
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DETECTION_REPO = ROOT / 'BEVFormer'
CONFIG = DETECTION_REPO / 'projects/configs/bevformer/bevformer_small.py'
PKL = ROOT / 'data/drivelm/nuscenes_infos_temporal_drivelm799_independent.pkl'
PRED_DIR = ROOT / 'outputs/drivelm_check/det799'
META_DIR = ROOT / 'data/nuscenes_trainval_meta'
OUT_DIR = ROOT / 'outputs/drivelm_check/det799_subset_eval'
EXPECTED_COUNT = 799


def sha256_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def verify_prediction_files(dataset, pred_dir, expected):
    """核查全部NPZ确实来自同一轮799帧实验，绝不把缺失帧当空预测。"""
    identity_path = pred_dir / 'run_identity.json'
    summary_path = pred_dir / 'run_summary.json'
    if not identity_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError('缺少 run_identity.json / run_summary.json；先完成799帧推理')
    identity = json.loads(identity_path.read_text(encoding='utf-8'))
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if not summary.get('complete') or int(summary.get('finished_frames', -1)) != expected:
        raise RuntimeError('799帧推理尚未全部完成，拒绝评估不完整样本')
    if int(identity.get('requested_frames', -1)) != expected:
        raise RuntimeError('预测缓存所用帧数不等于799')
    if identity.get('pkl_sha256') != sha256_file(PKL):
        raise RuntimeError('当前PKL与GPU预测时的PKL不一致，不能直接配对')
    if identity.get('model_config_sha256') != sha256_file(CONFIG):
        raise RuntimeError('检测模型配置与GPU预测时不一致')

    frame_dir = pred_dir / 'frames'
    existing = sorted(frame_dir.glob('frame_*.npz'))
    if len(existing) != expected:
        raise RuntimeError(f'预测文件个数={len(existing)}，预期={expected}。请排查多余/缺失文件')
    names = getattr(dataset, 'CLASSES', None)
    if names is None or len(names) != 10:
        raise RuntimeError('检测类别不是nuScenes标准10类')
    total_boxes = 0
    for i in range(expected):
        file = frame_dir / f'frame_{i:04d}.npz'
        if not file.exists():
            raise FileNotFoundError(f'缺失 {file.name}')
        with np.load(file, allow_pickle=False) as z:
            required = {'sample_index', 'sample_token', 'boxes_3d', 'scores_3d', 'labels_3d'}
            if not required.issubset(set(z.files)):
                raise ValueError(f'{file.name} 缺失必要数组字段')
            if int(z['sample_index']) != i or str(z['sample_token']) != str(dataset.data_infos[i]['token']):
                raise ValueError(f'{file.name} 的token或索引与PKL不一致')
            boxes, scores, labels = z['boxes_3d'], z['scores_3d'], z['labels_3d']
            if boxes.ndim != 2 or boxes.shape[1] < 9 or not (len(boxes) == len(scores) == len(labels)):
                raise ValueError(f'{file.name} 预测框形状不符合9维 nuScenes检测格式')
            if len(scores) > 500:
                raise ValueError(f'{file.name} 超过nuScenes每帧500框上限')
            if not (np.isfinite(boxes).all() and np.isfinite(scores).all()):
                raise ValueError(f'{file.name} 有非有限预测值')
            if (scores < 0).any() or (scores > 1).any() or (labels < 0).any() or (labels >= 10).any():
                raise ValueError(f'{file.name} 置信度/类别值非法')
            total_boxes += len(scores)
    if len(dataset) != expected:
        raise RuntimeError(f'数据集帧数不是799：{len(dataset)}')
    if len({str(r['token']) for r in dataset.data_infos}) != expected:
        raise RuntimeError('同一sample_token在PKL中重复')
    print(f'[通过] 799/799份预测校验通过，原始候选框总数={total_boxes}', flush=True)
    return identity, total_boxes


def convert_npz_to_nuscenes_json(dataset, pred_dir, out_dir):
    """使用已安装的mmdet3d NuScenesDataset._format_bbox转换，避免自写yaw/size变换。"""
    import torch
    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    converted = []
    for i in range(EXPECTED_COUNT):
        file = pred_dir / 'frames' / f'frame_{i:04d}.npz'
        with np.load(file, allow_pickle=False) as z:
            b = np.array(z['boxes_3d'], dtype=np.float32, copy=True)
            s = np.array(z['scores_3d'], dtype=np.float32, copy=True)
            l = np.array(z['labels_3d'], dtype=np.int64, copy=True)
        # 默认LiDAR框以底部中心为原点；gravity_center自动映射到框中心。
        box_obj = LiDARInstance3DBoxes(torch.from_numpy(b), box_dim=b.shape[1],
                                       with_yaw=True, origin=(0.5, 0.5, 0.0))
        converted.append(dict(boxes_3d=box_obj,
                              scores_3d=torch.from_numpy(s), labels_3d=torch.from_numpy(l)))
        if (i + 1) % 200 == 0 or (i + 1) == EXPECTED_COUNT:
            print(f'[转换] 已读取 {i+1}/{EXPECTED_COUNT} 帧原始预测', flush=True)
    print('[转换] 复用 mmdet3d/BEVFormer 数据集的 _format_bbox（LiDAR坐标→世界坐标）', flush=True)
    result_path = Path(dataset._format_bbox(converted, jsonfile_prefix=str(out_dir))).resolve()
    del converted
    if not result_path.is_file():
        raise RuntimeError('nuScenes格式化没有生成 results_nusc.json')
    with result_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    actual_tokens = set(data['results'])
    expected_tokens = {str(x['token']) for x in dataset.data_infos}
    if actual_tokens != expected_tokens:
        raise RuntimeError(f'nuScenes输出token不匹配：多={len(actual_tokens-expected_tokens)},缺={len(expected_tokens-actual_tokens)}')
    if any(len(v) > 500 for v in data['results'].values()):
        raise RuntimeError('转换结果超过500框/帧')
    print(f'[转换通过] {len(actual_tokens)} 帧预测已写入 {result_path}', flush=True)
    return result_path


def load_subset_boxes(nusc, cfg, result_path, tokens):
    """使用官方nuScenes val真值，严格截取799个sample token，并沿用官方过滤函数。"""
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
    from nuscenes.eval.detection.data_classes import DetectionBox

    print('[真值] 载入完整nuScenes官方val标注并提取DriveLM的799帧...', flush=True)
    all_val_gt = load_gt(nusc, 'val', DetectionBox, verbose=False)
    pred_boxes, meta = load_prediction(str(result_path), cfg.max_boxes_per_sample,
                                       DetectionBox, verbose=False)
    if set(pred_boxes.sample_tokens) != set(tokens):
        raise RuntimeError('预测 JSON 不能与799帧 token 一一对应')
    missing = set(tokens) - set(all_val_gt.sample_tokens)
    if missing:
        raise RuntimeError(f'{len(missing)}帧不属于官方nuScenes val，不能执行子集评估')
    gt_boxes = EvalBoxes()
    for token in tokens:
        gt_boxes.add_boxes(token, all_val_gt[token])
    if set(gt_boxes.sample_tokens) != set(pred_boxes.sample_tokens):
        raise RuntimeError('预测和真值的token集合不一致')
    del all_val_gt

    # 官方nuScenes评估的标准预处理：ego中心距离、类别量程、点数、车架过滤。
    pred_boxes = filter_eval_boxes(nusc, add_center_dist(nusc, pred_boxes),
                                   cfg.class_range, verbose=False)
    gt_boxes = filter_eval_boxes(nusc, add_center_dist(nusc, gt_boxes),
                                 cfg.class_range, verbose=False)
    gt_count = sum(len(gt_boxes[t]) for t in tokens)
    pred_count = sum(len(pred_boxes[t]) for t in tokens)
    if gt_count == 0:
        raise RuntimeError('799帧GT数量为0，元数据或标注读取异常')
    print(f'[真值通过] 子集GT目标框={gt_count}，评估过滤后预测框={pred_count}', flush=True)
    return gt_boxes, pred_boxes, meta, gt_count, pred_count


def main():
    parser = argparse.ArgumentParser(description='DriveLM799独立帧BEVFormer检测：真实GT子集mAP/NDS评估')
    parser.add_argument('--output', type=Path, default=OUT_DIR)
    parser.add_argument('--pred-dir', type=Path, default=PRED_DIR)
    args = parser.parse_args()
    begin = time.monotonic()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for need in (CONFIG, PKL, META_DIR / 'v1.0-trainval' / 'sample_annotation.json'):
        if not need.is_file():
            raise FileNotFoundError(f'缺少必需输入 {need}')
    os.chdir(DETECTION_REPO)
    sys.path.insert(0, str(DETECTION_REPO))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from nuscenes import NuScenes
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.common.config import config_factory

    print('[1/5] 导入原BEVFormer插件、构建799帧检测数据集...', flush=True)
    plugin = importlib.import_module('projects.mmdet3d_plugin')
    if not Path(plugin.__file__).resolve().is_relative_to(DETECTION_REPO.resolve()):
        raise RuntimeError(f'导入错误的mmdet3d_plugin：{plugin.__file__}')
    cfg = Config.fromfile(str(CONFIG))
    ds_cfg = copy.deepcopy(cfg.data.test)
    ds_cfg['ann_file'] = str(PKL)
    ds_cfg['test_mode'] = True
    ds_cfg.pop('samples_per_gpu', None)
    ds_cfg.pop('workers_per_gpu', None)
    dataset = build_dataset(ds_cfg)
    identity, raw_total = verify_prediction_files(dataset, args.pred_dir.resolve(), EXPECTED_COUNT)
    result_path = out_dir / 'results_nusc.json'

    print('[2/5] 将原始NPZ转换为nuScenes官方检测提交格式...', flush=True)
    result_path = convert_npz_to_nuscenes_json(dataset, args.pred_dir.resolve(), out_dir)

    print('[3/5] 创建nuScenes数据库（仅元数据，无须重新下载点云）...', flush=True)
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(META_DIR), verbose=False)
    eval_cfg = getattr(dataset, 'eval_detection_configs', None)
    if eval_cfg is None:
        eval_cfg = config_factory('detection_cvpr_2019')
    tokens = [str(r['token']) for r in dataset.data_infos]
    gt_boxes, pred_boxes, meta, n_gt, n_pred = load_subset_boxes(nusc, eval_cfg, result_path, tokens)

    print('[4/5] 调用nuScenes官方 mAP / NDS 计算方法，仅在799帧交集上评估...', flush=True)
    # 复用官方 DetectionEval.evaluate/main，唯一差异是传入精确子集GT与预测，不更改AP/NDS算法。
    evaluator = DetectionEval.__new__(DetectionEval)
    evaluator.nusc = nusc
    evaluator.result_path = str(result_path)
    evaluator.eval_set = 'val_subset_drivelm799'
    evaluator.output_dir = str(out_dir)
    evaluator.plot_dir = str(out_dir / 'plots')
    (out_dir / 'plots').mkdir(parents=True, exist_ok=True)
    evaluator.verbose = False
    evaluator.cfg = eval_cfg
    evaluator.pred_boxes = pred_boxes
    evaluator.gt_boxes = gt_boxes
    evaluator.meta = meta
    evaluator.sample_tokens = tokens
    metrics = evaluator.main(plot_examples=0, render_curves=False)

    print('[5/5] 保存评估协议及实验摘要（不生成Markdown）...', flush=True)
    result = {
        'experiment': 'BEVFormer-Small / DriveLM validation 799 subset / independent frame',
        'dataset': 'DriveLM-v1.1 validation frames within nuScenes official val',
        'protocol': 'nuScenes CVPR2019 detection matching/AP/NDS; 799-token subset, not official complete val',
        'uses_previous_bev': False,
        'sample_count': EXPECTED_COUNT, 'gt_boxes_after_nuscenes_filter': n_gt,
        'prediction_boxes_after_nuscenes_filter': n_pred,
        'raw_prediction_boxes': raw_total,
        'mean_ap': metrics['mean_ap'], 'nd_score': metrics['nd_score'],
        'per_class_ap': metrics['mean_dist_aps'],
        'input_prediction_path': str(args.pred_dir.resolve()),
        'result_file': str(result_path),
        'metrics_summary_file': str(out_dir / 'metrics_summary.json'),
        'identity': identity,
        'elapsed_seconds': round(time.monotonic() - begin, 2),
        'caution': 'Non-temporal, sparse 799-frame subset. Not directly comparable with full NuScenes val or temporal baseline.',
    }
    save_json(out_dir / 'evaluation_protocol.json', result)
    print('\n========== 799帧真实3D检测子集评估完成 ==========', flush=True)
    print(f'样本={EXPECTED_COUNT}，评估GT框={n_gt}，预测框={n_pred}', flush=True)
    print(f"799帧子集3D mAP={metrics['mean_ap']:.6f} ({metrics['mean_ap']*100:.2f}%)", flush=True)
    print(f"799帧子集NDS={metrics['nd_score']:.6f}", flush=True)
    print(f'指标文件：{out_dir / "metrics_summary.json"}', flush=True)
    print(f'协议文件：{out_dir / "evaluation_protocol.json"}', flush=True)
    print('注意：这是独立帧799帧子集真实GT评估，不是完整nuScenes官方val成绩。', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'\n[失败] {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)
        traceback.print_exc(limit=18)
        sys.exit(1)

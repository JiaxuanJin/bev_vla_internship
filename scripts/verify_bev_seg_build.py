#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核查独立 BEVFormer 分割分支的模型构建与权重匹配。

特点：不训练、不推理、不修改代码仓库、不生成 Markdown；
      只在 CPU 构建模型并用 weights_only=True 加载现有分割权重。
用法：在 ~/projects/bev_vla 下执行 python scripts/verify_bev_seg_build.py
"""
import argparse
import importlib
import os
import sys
import traceback
from pathlib import Path


def run(project_root: Path, checkpoint_path: Path) -> int:
    repo = project_root / 'BEVFormer-seg'
    config_path = repo / 'projects/configs/bevformer/bevformer_small_seg.py'
    if not config_path.is_file():
        print(f'[错误] 未找到分割配置: {config_path}')
        return 2
    if not checkpoint_path.is_file():
        print(f'[错误] 未找到分割权重: {checkpoint_path}')
        return 2

    # 避免误用已安装/缓存的旧 BEVFormer 插件，强制先搜索独立分割仓库。
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    print(f'[路径] 分割仓库: {repo}', flush=True)
    print(f'[路径] 模型配置: {config_path}', flush=True)
    print(f'[路径] 权重文件: {checkpoint_path}', flush=True)

    import torch
    import mmcv
    print(f'[环境] torch={torch.__version__}, mmcv={mmcv.__version__}, CUDA可用={torch.cuda.is_available()}', flush=True)
    from mmcv import Config
    from mmdet3d.models import build_model

    config = Config.fromfile(str(config_path))
    task = config.model.pts_bbox_head.get('task', {})
    seg_cfg = config.model.pts_bbox_head.get('seg_encoder', {})
    print(f'[配置] task={dict(task)}, 分割输出通道={seg_cfg.get("outC")}', flush=True)
    if not task.get('seg', False):
        print('[错误] 当前配置没有启用分割任务！', flush=True)
        return 3

    # 第三方注册组件必须从分割仓库导入，而不是从 BEVFormer 检测仓库导入。
    plugin = importlib.import_module('projects.mmdet3d_plugin')
    print(f'[插件] 成功导入: {plugin.__file__}', flush=True)
    if not Path(plugin.__file__).resolve().is_relative_to(repo.resolve()):
        raise RuntimeError('误导入了非分割仓库的插件，停止以避免模型不匹配')

    config.model.pretrained = None
    config.model.train_cfg = None
    print('[阶段 1/3] CPU 构建 BEV 分割模型中……', flush=True)
    model = build_model(config.model, test_cfg=config.get('test_cfg'))
    model.eval()
    expected = model.state_dict()
    print(f'[模型] 构建成功，状态项={len(expected)}', flush=True)

    print('[阶段 2/3] 安全读取分割权重（可能需要几十秒）……', flush=True)
    package = torch.load(str(checkpoint_path), map_location='cpu', weights_only=True)
    if not isinstance(package, dict):
        raise TypeError('checkpoint 顶层数据不是 dict')
    weights = package.get('state_dict', package)
    if not isinstance(weights, dict):
        raise TypeError('checkpoint 的 state_dict 不是 dict')

    # 仅当当前字典中的所有键都含有 module. 前缀时，才剥离 DDP 前缀。
    if weights and all(str(k).startswith('module.') for k in weights):
        weights = {k[len('module.'):]: v for k, v in weights.items()}

    shared = expected.keys() & weights.keys()
    mismatched = [(k, tuple(expected[k].shape), tuple(weights[k].shape))
                  for k in shared if tuple(expected[k].shape) != tuple(weights[k].shape)]
    unexpected = sorted(weights.keys() - expected.keys())
    missing = sorted(expected.keys() - weights.keys())
    matched = len(shared) - len(mismatched)
    print(f'[对照] 权重文件项={len(weights)}，模型项={len(expected)}，名称与形状一致={matched}', flush=True)
    print(f'[对照] 模型缺少的项={len(missing)}，权重多出的项={len(unexpected)}，形状不一致={len(mismatched)}', flush=True)
    for label, values in [('缺失示例', missing), ('多余示例', unexpected), ('形状不一致示例', mismatched)]:
        if values:
            print(f'  {label}:', flush=True)
            for item in values[:8]:
                print(f'    {item}', flush=True)

    seg_keys = [k for k in expected if 'seg_' in k or 'seg.' in k or 'map_' in k]
    seg_matched = [k for k in seg_keys if k in weights and expected[k].shape == weights[k].shape]
    print(f'[分割分支] 模型分割相关项={len(seg_keys)}，匹配={len(seg_matched)}', flush=True)

    if mismatched:
        print('[中止] 存在权重形状冲突：暂不进行权重加载，避免产生误导。', flush=True)
        return 4
    if not seg_matched:
        print('[中止] 没有匹配到分割分支权重，暂不加载。', flush=True)
        return 5

    print('[阶段 3/3] 加载匹配的权重参数（CPU，不进行推理）……', flush=True)
    result = model.load_state_dict(weights, strict=False)
    print(f'[加载] 完成，missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}', flush=True)
    if result.missing_keys or result.unexpected_keys:
        print('[提醒] 非严格加载包含未匹配字段，后续需要确认是否影响推理。', flush=True)
    else:
        print('[成功] 模型构建与权重完全匹配！下一步可准备地图标注与推理。', flush=True)
    print('[安全] 未修改原检测仓库，未生成 Markdown，未开始训练。', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description='BEVFormer-Small BEV 分割模型和权重匹配检查')
    parser.add_argument('--project-root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--checkpoint', default='/mnt/g/nus/bevformer_small_seg.pth')
    args = parser.parse_args()
    try:
        return run(Path(args.project_root).resolve(), Path(args.checkpoint).resolve())
    except Exception as exc:
        print(f'\n[错误] 检查停止：{type(exc).__name__}: {exc}', flush=True)
        traceback.print_exc(limit=8)
        print('[提示] 把以上错误截图发我，先不要安装/升级依赖。', flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

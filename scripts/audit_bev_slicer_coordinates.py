"""BEVFormer-Seg 特征裁剪坐标单元测试（不运行模型、不修改原文件）。

目的：使用人工构建的物理坐标通道，验证 BevFeatureSlicer 的
x/y 网格采样数值，以及网络里 rot90 + flip 的等效变换。
注意：此检查只验证采样器数学行为，不等于验证真实分割预测的坐标方向。
"""

import ast
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py'
HEAD = ROOT / 'BEVFormer-seg/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py'
OUTPUT = ROOT / 'outputs/bev_seg/seg_slicer_coordinates_check.json'


def read_constant_assignments(path, names):
    """只解析配置文件中的字面量，不执行仓库配置和其他代码。"""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = ast.literal_eval(node.value)
    missing = set(names) - set(found)
    if missing:
        raise RuntimeError(f'缺少配置字段：{sorted(missing)}')
    return found


def load_slicer_implementation(path):
    """只提取原仓库的计算函数和裁剪类，绕开 mmdet 插件的其他导入。"""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    chosen = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'calculate_birds_eye_view_parameters':
            chosen.append(node)
        if isinstance(node, ast.ClassDef) and node.name == 'BevFeatureSlicer':
            chosen.append(node)
    if len(chosen) != 2:
        raise RuntimeError('无法从原仓库定位 BevFeatureSlicer 和坐标参数函数')
    module = ast.fix_missing_locations(ast.Module(body=chosen, type_ignores=[]))
    env = {'torch': torch, 'nn': nn, 'F': F}
    exec(compile(module, str(path), 'exec'), env)
    return env['BevFeatureSlicer']


def main():
    print('===== BEVFormer-Seg 坐标映射单元测试 =====', flush=True)
    cfg = read_constant_assignments(CONFIG, ('det_grid_conf', 'map_grid_conf', 'bev_h_', 'bev_w_'))
    det, seg = cfg['det_grid_conf'], cfg['map_grid_conf']
    h, w = cfg['bev_h_'], cfg['bev_w_']

    Slicer = load_slicer_implementation(HEAD)
    slicer = Slicer(det, seg)

    # 对已经经过网络 rot90+flip 的特征布局构造坐标通道。
    # 通道0标注列方向物理x，通道1标注行方向物理y。
    det_x0 = det['xbound'][0] + det['xbound'][2] / 2
    det_y0 = det['ybound'][0] + det['ybound'][2] / 2
    x_axis = torch.linspace(det_x0, -det_x0, w)
    y_axis = torch.linspace(det_y0, -det_y0, h)
    xx = x_axis.unsqueeze(0).expand(h, -1)
    yy = y_axis.unsqueeze(1).expand(-1, w)
    synthetic = torch.stack((xx, yy), dim=0).unsqueeze(0)

    with torch.no_grad():
        out = slicer(synthetic).cpu().numpy()

    target_x = np.arange(seg['xbound'][0] + seg['xbound'][2] / 2,
                         seg['xbound'][1], seg['xbound'][2])
    target_y = np.arange(seg['ybound'][0] + seg['ybound'][2] / 2,
                         seg['ybound'][1], seg['ybound'][2])
    if out.shape != (1, 2, len(target_y), len(target_x)):
        raise RuntimeError(f'裁剪尺寸异常：{out.shape}，目标200×400请以实际配置为准')

    err_x = float(np.abs(out[0, 0] - target_x[None, :]).mean())
    err_y = float(np.abs(out[0, 1] - target_y[:, None]).mean())
    center_row, center_col = len(target_y) // 2, len(target_x) // 2

    # 检验模型在分割头中执行的旋转 + 翻转是否等价于空间轴转置。
    example = torch.arange(2 * 3).reshape(1, 1, 2, 3)
    rotated_flipped = torch.flip(torch.rot90(example, k=-1, dims=(2, 3)), dims=[3])
    transpose_equal = bool(torch.equal(rotated_flipped, example.transpose(2, 3)))

    result = {
        '用途': '仅用于网格采样坐标数学检查，不是模型 mIoU 或预测结果',
        'BEV网格': [h, w],
        '分割输出形状': list(out.shape),
        '原始网格定义': det,
        '地图网格定义': seg,
        '平均绝对坐标误差_米': {'x': err_x, 'y': err_y},
        '中心坐标_米': {
            '采样输出': [float(out[0, 0, center_row, center_col]), float(out[0, 1, center_row, center_col])],
            '期望值': [float(target_x[center_col]), float(target_y[center_row])],
        },
        'rot90再flip等价于空间转置': transpose_equal,
        '该实验不能排除的问题': ['BEV原始特征的物理轴定义', '地图标签局部坐标定义', '模型预处理', '预训练权重泛化差异'],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    print('模型网格:', (h, w))
    print('分割输出形状:', out.shape)
    print('det x/y bounds:', det['xbound'], det['ybound'])
    print('map x/y bounds:', seg['xbound'], seg['ybound'])
    print(f'x/y 坐标平均误差（米）: {err_x:.5f}, {err_y:.5f}')
    print('中心采样坐标:', result['中心坐标_米'])
    print('rot90+flip 等于空间转置:', transpose_equal)
    print('结果已保存:', OUTPUT)
    print('注意：此测试不能证明模型与地图真值方向完全一致。')


if __name__ == '__main__':
    main()

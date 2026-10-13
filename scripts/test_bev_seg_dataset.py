#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BEVFormer-Seg 数据加载与 BEV 地图语义真值冒烟测试。

只读取一条 nuScenes Mini 验证样本，执行项目官方的 test_pipeline，
检查 RasterizeMapVectors 生成的 semantic_indices，保存检查结果。
不会训练模型、更新权重或创建 Markdown。
"""

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEG_REPO = ROOT / "BEVFormer-seg"
OUTPUT_DIR = ROOT / "outputs" / "bev_seg"
CONFIG = SEG_REPO / "projects/configs/bevformer/bevformer_small_seg.py"


def unwrap(value):
    """移除 MMCV DataContainer 和测试增强流程生成的单元素列表。"""
    for _ in range(8):
        if hasattr(value, "data") and value.__class__.__name__ == "DataContainer":
            value = value.data
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        else:
            break
    return value


def as_numpy(value):
    """把标签张量转换为 NumPy，避免修改模型数据。"""
    import torch
    value = unwrap(value)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)) and len(value):
        if all(np.isscalar(x) for x in value):
            return np.asarray(value)
        return as_numpy(value[0])
    raise TypeError(f"无法转换标签类型: {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description="检查 BEV 分割验证集的数据加载和语义标签")
    parser.add_argument("--index", type=int, default=0, help="Mini 验证集样本索引，默认 0")
    args = parser.parse_args()

    if not CONFIG.is_file():
        raise FileNotFoundError(f"分割配置不存在: {CONFIG}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 这一分割仓库使用相对数据路径，必须以它自己的根目录为工作目录。
    os.chdir(SEG_REPO)
    sys.path.insert(0, str(SEG_REPO))

    from mmcv import Config
    from mmdet3d.datasets import build_dataset

    cfg = Config.fromfile(str(CONFIG))
    print("[1/4] 导入分割扩展模块……", flush=True)
    importlib.import_module("projects.mmdet3d_plugin")
    print("[2/4] 构建 Mini 验证集数据加载器……", flush=True)
    val_cfg = cfg.data.val.copy()
    # samples_per_gpu / workers_per_gpu 属于 DataLoader 配置，
    # 不能传给 NuScenesDataset.__init__（否则会出现 unexpected keyword argument）。
    for key in ("samples_per_gpu", "workers_per_gpu"):
        if key in val_cfg:
            print(f"[兼容修复] 从数据集配置移除 DataLoader 参数: {key}", flush=True)
            val_cfg.pop(key)
    val_cfg.test_mode = True  # 验证时使用官方测试分支，避免触发训练队列逻辑
    ds = build_dataset(val_cfg)
    print(f"[数据集] 样本数: {len(ds)}", flush=True)

    if not 0 <= args.index < len(ds):
        raise IndexError(f"索引 {args.index} 超出范围 [0, {len(ds)-1}]")

    print(f"[3/4] 加载第 {args.index} 帧，执行 RasterizeMapVectors……", flush=True)
    sample = ds[args.index]
    if sample is None:
        raise RuntimeError("数据管线返回 None")
    if not isinstance(sample, dict):
        raise TypeError(f"数据管线输出不是 dict: {type(sample).__name__}")

    print("[管线输出字段]", list(sample.keys()), flush=True)
    if "semantic_indices" not in sample:
        raise KeyError("输出未包含 semantic_indices：地图语义标签尚未生成")

    labels = as_numpy(sample["semantic_indices"])
    if labels.size == 0 or not np.isfinite(labels).all():
        raise ValueError("地图语义标签为空或包含 NaN/Inf")

    # 保存原始语义标签，不对多通道结果擅自推断语义类别。
    np.save(OUTPUT_DIR / "seg_gt_sample0.npy", labels)
    print(f"[标签] shape={labels.shape}, dtype={labels.dtype}", flush=True)
    uniq = np.unique(labels)
    print(f"[标签] 唯一值（前20个）={uniq[:20].tolist()}, 唯一值数={len(uniq)}", flush=True)

    # 仅当标签布局便于展示时，保存真值可视化（不是模型预测）。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.squeeze(labels)
    if image.ndim == 2:
        view = image
        description = "2D label indices"
    elif image.ndim == 3 and image.shape[0] <= 16:
        view = image[0]
        description = "channel 0 of multi-channel GT (not argmax)"
    elif image.ndim == 3 and image.shape[-1] <= 16:
        view = image[..., 0]
        description = "channel 0 of multi-channel GT (not argmax)"
    else:
        view = None
        description = "unrecognized layout; raw .npy saved"

    if view is not None:
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(view, interpolation="nearest", origin="lower")
        ax.set_title("BEV map ground truth | " + description)
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        figure_path = OUTPUT_DIR / "seg_gt_sample0.png"
        fig.savefig(figure_path, dpi=160)
        plt.close(fig)
        print(f"[保存] 标签可视化: {figure_path}", flush=True)

    meta = {
        "dataset_length": len(ds),
        "sample_index": args.index,
        "pipeline_keys": sorted(sample.keys()),
        "semantic_indices_shape": list(labels.shape),
        "semantic_indices_dtype": str(labels.dtype),
        "unique_values_count": int(len(uniq)),
        "unique_values_preview": uniq[:20].tolist(),
        "note": "本测试仅验证语义真值生成；尚未完成模型推理及 mIoU 计算。",
    }
    report_path = OUTPUT_DIR / "seg_dataset_smoke_test.json"
    report_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[保存] 检查结果: {report_path}", flush=True)
    print("[成功] 分割验证集加载及地图语义标签生成通过！", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[失败] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)

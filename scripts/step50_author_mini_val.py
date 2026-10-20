"""在隔离环境下调用作者原始 Mini Val 分割评估函数。

此脚本只负责加载作者模型、mini_val 数据和 checkpoint，
不修改 BEVFormerHead、栅格标注、权重或原始测试指标实现。
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析独立复现实验的输入路径。"""
    parser = argparse.ArgumentParser(description="Author clean Mini Val evaluation")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / "projects/bevformer_author_clean_5080",
        help="隔离作者复现工程根目录",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/g/nus/bevformer_small_seg.pth"),
        help="作者原始分割 checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    """复用作者的单 GPU 评估函数进行完整 81 帧评估。"""
    args = parse_args()
    root = args.root.expanduser().resolve()
    source = root / "BEVFormer-seg"
    config_path = source / "projects/configs/bevformer/bevformer_small_seg.py"
    checkpoint = args.checkpoint.expanduser().resolve()

    for path in (source, config_path, checkpoint):
        if not path.exists():
            raise FileNotFoundError(f"必需路径不存在：{path}")

    # 使用相对 data/nuscenes 路径的作者原版 Config。
    os.chdir(source)
    sys.path.insert(0, str(source))

    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    # 显式注册项目中的模型与数据管线；源码保持只读。
    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.apis.test import single_gpu_test
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    cfg = Config.fromfile(str(config_path))
    ann_file = Path(cfg.data.test.ann_file)
    if not ann_file.exists():
        raise FileNotFoundError(
            f"作者配置中的测试 PKL 不存在：{ann_file.resolve()}。"
            "先确认新工程的 data/nuscenes 已正确链接到 mini 数据。"
        )

    with ann_file.open("rb") as file:
        dataset_info = pickle.load(file)
    if not isinstance(dataset_info, dict) or "infos" not in dataset_info:
        raise ValueError("测试 PKL 格式不符合作者 nuScenes infos 约定")

    frame_count = len(dataset_info["infos"])
    print(f"[Check] test PKL: {ann_file.resolve()}", flush=True)
    print(f"[Check] samples: {frame_count}", flush=True)
    print(f"[Check] checkpoint: {checkpoint}", flush=True)
    print(f"[Check] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[Check] video_test_mode: {cfg.model.video_test_mode}", flush=True)
    if frame_count != 81:
        raise ValueError(
            f"当前测试 PKL 有 {frame_count} 帧，并不是 81 帧 Mini Val。"
            "停止运行，避免错把正式 Val 当成 Mini。"
        )
    if not cfg.model.video_test_mode:
        raise ValueError("配置意外关闭了连续时序测试，停止评估")

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    samples_per_gpu = cfg.data.test.pop("samples_per_gpu", 1)
    cfg.data.workers_per_gpu = 0  # 避免调试阶段多进程吞掉真正错误。

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    print(f"[Check] built dataset: {len(dataset)} frames", flush=True)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint_data = load_checkpoint(model, str(checkpoint), map_location="cpu")
    model.CLASSES = checkpoint_data.get("meta", {}).get("CLASSES", dataset.CLASSES)
    if "PALETTE" in checkpoint_data.get("meta", {}):
        model.PALETTE = checkpoint_data["meta"]["PALETTE"]
    model = MMDataParallel(model.cuda(), device_ids=[0])

    print("[Run] Author single_gpu_test; 81-frame temporal ON", flush=True)
    single_gpu_test(model, loader, show=False, out_dir=None)
    print("[Done] 作者原始评估函数运行结束。请记录上方 mIoU 表格。", flush=True)


if __name__ == "__main__":
    main()

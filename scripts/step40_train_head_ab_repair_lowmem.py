#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step40：低内存版，使用训练集微调两种 BEV 空间约定的分割 Decoder。

只读作者原始权重和 nuScenes 数据；冻结 Encoder，将相同 BEV 特征分别输入
原始 CCW90 与 Transpose 两套独立 Decoder，在官方 train 样本上做配对训练。
使用独立 Val 少量样本比较是否改善（无时序），不覆盖既有 Step34 基线。

运行示例：
    python -u scripts/step40_train_head_ab_repair_lowmem.py --steps 20
    python -u scripts/step40_train_head_ab_repair_lowmem.py --steps 120

前提：存在作者配置中的 nuscenes_infos_temporal_train.pkl 及其引用的六路影像。
这不是 6019 帧官方评估，也不保证达到作者发布的 mIoU。
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer-seg"
CONFIG = REPO / "projects/configs/bevformer/bevformer_small_seg.py"
CHECKPOINT = Path("/mnt/g/nus/bevformer_small_seg.pth")
VAL_PKL = (
    ROOT / "outputs/drivelm_check/root_cause/step21_full_val/"
    "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
)
OUTPUT = ROOT / "outputs/drivelm_check/root_cause/step40_train_head_ab_repair"
CLASS_NAMES = ("background", "divider", "ped_crossing", "boundary")
VAL_INDICES = (73, 411, 817, 1311, 1733, 2261, 2877, 3449, 4027, 4631, 5239, 5831)
REGRESSION_INDICES = (3715, 3718, 3719)


def get_unwrapped(value: Any) -> Any:
    """只取单次测试增强对应的 DataContainer 数据。"""
    from mmcv.parallel import DataContainer

    while True:
        if isinstance(value, DataContainer):
            value = value.data
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        else:
            return value


def load_frame(dataset: Any, index: int) -> tuple[torch.Tensor, dict, torch.Tensor]:
    """加载单帧六路相机、元信息及 200x400 分割标签。"""
    item = dataset[index]
    image = get_unwrapped(item["img"])
    metadata = get_unwrapped(item["img_metas"])
    target = get_unwrapped(item["semantic_indices"])
    if not torch.is_tensor(image) or not isinstance(metadata, dict):
        raise TypeError(f"索引 {index} 的图像或元信息格式不正确")
    if image.ndim == 4:
        image = image.unsqueeze(0)
    if tuple(image.shape[:2]) != (1, 6) or image.ndim != 5:
        raise ValueError(f"索引 {index}: 图像维度异常 {tuple(image.shape)}")
    target = torch.as_tensor(target).squeeze().long()
    if tuple(target.shape) != (200, 400):
        raise ValueError(f"索引 {index}: GT 维度异常 {tuple(target.shape)}")
    if not torch.logical_and(target >= 0, target < 4).all().item():
        raise ValueError(f"索引 {index}: GT 含无效类别")
    return image, metadata, target


def confusion_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """GT 是行，预测是列，四类累计混淆矩阵。"""
    return np.bincount((gt * 4 + pred).reshape(-1), minlength=16).reshape(4, 4)


def fg_miou(cm: np.ndarray) -> float:
    """与作者口径一致：排除背景，对三类前景的全局 IoU 求均值。"""
    diag = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - diag
    return float(np.divide(diag, union, out=np.zeros(4), where=union > 0)[1:].mean())


def main() -> None:
    """执行同特征、同标签的双 Decoder 训练与 Val A/B。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=120, help="训练帧数，不是 Epoch 数")
    parser.add_argument("--lr", type=float, default=1e-4, help="Decoder 学习率")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--save-only-report", action="store_true", help="仅记录输入可用性，不运行训练")
    args = parser.parse_args()
    if args.steps < 1 or args.lr <= 0:
        parser.error("--steps 必须 >= 1，--lr 必须 > 0")

    from mmcv import Config

    cfg = Config.fromfile(str(CONFIG))
    train_ann = REPO / cfg.data.train.ann_file
    data_dir = REPO / cfg.data.train.data_root
    report = {
        "purpose": "train-split supervised repair A/B; not official full Val result",
        "train_ann_file": str(train_ann),
        "train_ann_exists": train_ann.is_file(),
        "val_ann_file": str(VAL_PKL),
        "val_ann_exists": VAL_PKL.is_file(),
        "checkpoint_file": str(CHECKPOINT),
        "checkpoint_exists": CHECKPOINT.is_file(),
        "data_root_exists": data_dir.is_dir(),
        "steps": args.steps,
        "lr": args.lr,
        "seed": args.seed,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUT / "results.json"
    if not train_ann.is_file():
        report["status"] = "blocked_missing_train_ann"
        report["next_action"] = (
            "缺少官方 train 信息文件，不能用 Val 标注代替训练。"
            "请准备 nuscenes_infos_temporal_train.pkl 以及其引用的训练图像，"
            "不要下载或覆盖已有 Val 和权重。"
        )
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[阻塞] 未找到官方训练集信息文件：", train_ann, flush=True)
        print(report["next_action"], flush=True)
        print("报告：", report_file, flush=True)
        return
    if not VAL_PKL.is_file() or not CHECKPOINT.is_file() or not data_dir.is_dir():
        raise FileNotFoundError("缺少已确认的 Val PKL、checkpoint 或数据目录")
    if args.save_only_report:
        report["status"] = "ready"
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("数据清单已记录：", report_file, flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("需要可用的 CUDA 环境")
    # 限制 CPU 工作线程，避免 WSL 内存较小时预处理阶段发生资源争抢。
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    importlib.import_module("projects.mmdet3d_plugin")
    from mmdet3d.datasets import build_dataset  # pylint: disable=import-outside-toplevel
    from mmdet3d.models import build_model  # pylint: disable=import-outside-toplevel

    # 只使用训练分区样本，采用与验证一致的单帧预处理以避免队列和增强干扰。
    train_cfg = copy.deepcopy(cfg.data.test)
    train_cfg.ann_file = str(train_ann)
    train_cfg.test_mode = True
    train_cfg.pop("samples_per_gpu", None)
    train_cfg.pop("workers_per_gpu", None)
    print("[1/5] 创建训练数据集（仅创建一个数据集实例）", flush=True)
    train_data = build_dataset(train_cfg)
    print(f"[1/5] 训练数据集已创建：{len(train_data)} 帧", flush=True)
    if len(train_data) < 2:
        raise RuntimeError("官方 train 分区没有足够样本")

    # 延迟到训练完且释放 train_data 后，再创建 val_data；避免重复加载 NuScenes + 地图元数据。
    val_cfg = copy.deepcopy(cfg.data.test)
    val_cfg.ann_file = str(VAL_PKL)
    val_cfg.test_mode = True
    val_cfg.pop("samples_per_gpu", None)
    val_cfg.pop("workers_per_gpu", None)
    report["train_sample_count"] = len(train_data)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    print("[2/5] 创建并加载冻结的 BEVFormer 模型", flush=True)
    base_model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    weights = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True)
    loaded = base_model.load_state_dict(weights["state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("权重严格加载失败")
    del weights
    base_model.to(device).eval()
    base_model.requires_grad_(False)
    gc.collect()
    print("[2/5] 模型权重加载完成", flush=True)
    slicer = base_model.pts_bbox_head.feat_cropper
    decoder = base_model.pts_bbox_head.seg_decoder

    # 保存两份互不影响的训练副本，不修改模型本体或官方 checkpoint。
    decoders = nn.ModuleDict({
        "original_ccw90": copy.deepcopy(decoder),
        "transpose_candidate": copy.deepcopy(decoder),
    }).to(device)
    decoders.requires_grad_(True)
    print("[3/5] 两个分割 Decoder 副本准备完成", flush=True)
    optimizers = {
        name: torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=1e-4)
        for name, module in decoders.items()
    }
    loss_weight = torch.tensor([0.5, 2.0, 2.0, 2.0], device=device)
    captured: dict[str, torch.Tensor] = {}

    def record_features(_module: nn.Module, inputs: tuple) -> None:
        """取得原始 head 输出给 Slicer 的特征，保持一个 Encoder 前向。"""
        captured["seg_bev"] = inputs[0].detach()

    hook = slicer.register_forward_pre_hook(record_features)

    def features_for(image: torch.Tensor, metadata: dict) -> torch.Tensor:
        """冻结模型提取原始 CCW90 特征，不传播 Encoder 梯度。"""
        captured.clear()
        with torch.no_grad():
            base_model(
                return_loss=False,
                rescale=True,
                img=[image.to(device).clone()],
                img_metas=[[copy.deepcopy(metadata)]],
            )
        if "seg_bev" not in captured:
            raise RuntimeError("未捕获到 Slicer 前的 BEV 特征")
        return captured["seg_bev"]

    def forward_variant(feature: torch.Tensor, variant: str) -> torch.Tensor:
        """原始或取消原始 CCW90 行翻转后的 Transpose。"""
        if variant == "transpose_candidate":
            feature = torch.flip(feature, dims=(2,))
        return slicer(feature)

    # 训练样本均匀遍布官方 train 分区，避免只取连续单个场景。
    rng = np.random.default_rng(args.seed)
    ordering = rng.permutation(len(train_data)).tolist()
    train_log: list[dict] = []
    start = time.monotonic()
    print(f"===== Step40：训练 {args.steps} 帧；train={len(train_data)} =====", flush=True)
    try:
        for step in range(args.steps):
            index = ordering[step % len(ordering)]
            try:
                image, metadata, target = load_frame(train_data, index)
            except (FileNotFoundError, OSError) as error:
                raise RuntimeError(
                    f"训练数据第 {index} 帧的图像文件缺失。需要真实 train 影像，"
                    f"不能用 Val 数据替代。原始错误: {error}"
                ) from error
            feature = features_for(image, metadata)
            gt = target.unsqueeze(0).to(device)
            item: dict[str, Any] = {"step": step + 1, "train_index": index}
            for name, head in decoders.items():
                head.train()
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                seg_input = forward_variant(feature, name)
                logits = head(seg_input)
                loss = F.cross_entropy(logits, gt, weight=loss_weight)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"{name} 出现非有限损失")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                item[f"loss_{name}"] = round(loss.detach().item(), 6)
            train_log.append(item)
            if (step + 1) % 10 == 0 or step + 1 == args.steps:
                print(
                    f"[{step+1}/{args.steps}] "
                    f"原始 loss={item['loss_original_ccw90']:.4f} "
                    f"Transpose loss={item['loss_transpose_candidate']:.4f} "
                    f"已耗时={(time.monotonic()-start)/60:.1f}min",
                    flush=True,
                )

        # 验证阶段不再保留训练数据集及其 nuScenes/map 引用。
        del train_data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[4/5] 已释放训练集，开始加载 Val 数据集", flush=True)
        val_data = build_dataset(val_cfg)
        if len(val_data) <= max((*VAL_INDICES, *REGRESSION_INDICES)):
            raise RuntimeError("Val 样本数量与已确认的 6019 帧不一致")
        report["val_sample_count"] = len(val_data)
        print(f"[4/5] Val 已加载：{len(val_data)} 帧，开始独立评估", flush=True)
        # 验证集绝不进行参数更新；包含此前已知的异常场景。
        for head in decoders.values():
            head.eval()
        cms = {
            name: np.zeros((4, 4), dtype=np.int64)
            for name in ("pretrained_original", "pretrained_transpose", *decoders.keys())
        }
        group_cms = {
            group: {name: np.zeros((4, 4), dtype=np.int64) for name in cms}
            for group in ("holdout", "regression")
        }
        per_frame: list[dict] = []
        for count, index in enumerate((*VAL_INDICES, *REGRESSION_INDICES), 1):
            image, metadata, target = load_frame(val_data, index)
            feature = features_for(image, metadata)
            gt = target.numpy().astype(np.int64)
            entry: dict[str, Any] = {
                "index": index,
                "group": "regression" if index in REGRESSION_INDICES else "holdout",
            }
            with torch.no_grad():
                for orientation in ("original_ccw90", "transpose_candidate"):
                    x = forward_variant(feature, orientation)
                    before = decoder(x).argmax(dim=1)[0].cpu().numpy().astype(np.int64)
                    after = decoders[orientation](x).argmax(dim=1)[0].cpu().numpy().astype(np.int64)
                    for key, pred in (
                        ("pretrained_original" if orientation == "original_ccw90" else "pretrained_transpose", before),
                        (orientation, after),
                    ):
                        cm = confusion_matrix(gt, pred)
                        cms[key] += cm
                        group_cms[entry["group"]][key] += cm
                        entry[key] = fg_miou(cm)
            per_frame.append(entry)
            print(
                f"[Val {count}/{len(VAL_INDICES) + len(REGRESSION_INDICES)}] "
                f"index={index}",
                flush=True,
            )
    finally:
        hook.remove()

    # 保存的是单独训练的 Decoder 权重，不可当作整模型 checkpoint 直接加载。
    save_ckpt = OUTPUT / "seg_decoder_ab_train_split.pt"
    digest = hashlib.sha256()
    with CHECKPOINT.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    torch.save({
        "model_source_sha256": digest.hexdigest(),
        "training": {"steps": args.steps, "learning_rate": args.lr, "seed": args.seed},
        "decoder_state_dicts": {
            key: module.cpu().state_dict() for key, module in decoders.items()
        },
    }, save_ckpt)
    summary = {
        **report,
        "status": "complete",
        "scope": "no-temporal sparse Val; not formal full Val mIoU",
        "protocol": "official train supervised fine-tuning; encoder frozen; two independent decoders; identical extracted BEV per sample",
        "val_indices": list(VAL_INDICES),
        "regression_indices": list(REGRESSION_INDICES),
        "foreground_miou_by_variant": {name: fg_miou(cm) for name, cm in cms.items()},
        "group_foreground_miou": {
            group: {name: fg_miou(cm) for name, cm in scores.items()}
            for group, scores in group_cms.items()
        },
        "per_class_confusion": {name: cm.tolist() for name, cm in cms.items()},
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "decoder_checkpoint": str(save_ckpt),
        "conclusion_limit": "属于训练集修复可行性实验；若无法在独立验证集稳健改善，不得覆盖原权重或声称根因解决",
    }
    report_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUTPUT / "train_loss.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(train_log[0]))
        writer.writeheader()
        writer.writerows(train_log)
    with (OUTPUT / "per_frame_val.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_frame[0]))
        writer.writeheader()
        writer.writerows(per_frame)
    print("\n===== Step40 低内存版完成 =====")
    for name, score in summary["foreground_miou_by_variant"].items():
        print(f"{name}: {score:.4%}")
    print("报告：", report_file)
    print("Decoder 权重：", save_ckpt)
    print("结论：训练修复候选，不是正式 6019 帧新基线。")


if __name__ == "__main__":
    main()

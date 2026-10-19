#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nuScenes Mini 81 帧 BEVFormer-Seg 历史 BEV 配对消融。

运行命令：
    python -u scripts/step15_temporal_bev_ab.py 2>&1 | tee \
        outputs/drivelm_check/root_cause/step15_temporal_ab.log

特点：
    - 同一模型和相同输入，逐帧配对比较历史 BEV 关闭/开启。
    - 对两组保留彼此独立的时序状态，禁止跨组串帧。
    - 克隆每次模型输入，避免第三方源码的 squeeze_ 原地修改。
    - 保存累计混淆矩阵、分类 IoU、逐帧记录；不覆盖已有实验。
    - 仅评估 Mini 81 帧，不能作为 DriveLM799 正式指标。
"""

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from run_bev_seg_minival import (
    CLASS_NAMES,
    NUM_CLASSES,
    confusion_of_frame,
    stats_from_confusion,
)
from run_bev_seg_one_sample import (
    CONFIG,
    ROOT,
    SEG_REPO,
    segmentation_classes,
    to_numpy,
    unwrap_single_aug,
)

OUT_DIR = ROOT / "outputs/drivelm_check/root_cause/step15_temporal_ab"
MINI_METRICS = ROOT / "outputs/bev_seg/seg_minival_metrics.json"
DEFAULT_CKPT = Path("/mnt/g/nus/bevformer_small_seg.pth")
MODES = ("no_history", "with_history")


def sha256_file(path: Path) -> str:
    """计算文件哈希，方便师傅准确复现本次实验。

    Args:
        path: 待检查文件。

    Returns:
        文件的 SHA256 十六进制字符串。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_temporal_state() -> dict:
    """初始化一个完整、彼此独立的模型时序状态。"""
    return {
        "prev_bev": None,
        "scene_token": None,
        "prev_pos": 0,
        "prev_angle": 0,
    }


def summarize(confusion: np.ndarray) -> dict:
    """用累计像素的交并比计算指标，不平均逐帧 mIoU。"""
    classes, foreground_miou, overall_miou = stats_from_confusion(confusion)
    return {
        "foreground_miou": foreground_miou,
        "including_background_miou": overall_miou,
        "classes": classes,
        "confusion_matrix_gt_rows_pred_cols": confusion.tolist(),
    }


def write_results(
    output_dir: Path,
    confusion: dict,
    per_frame: list,
    meta: dict,
    *,
    complete: bool,
) -> dict:
    """将独立评估结果写入专属目录，避免覆盖之前的原始成绩。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {mode: summarize(confusion[mode]) for mode in MODES}
    original = metrics["no_history"]["foreground_miou"]
    temporal = metrics["with_history"]["foreground_miou"]
    result = {
        **meta,
        "complete": complete,
        "processed_samples": len(per_frame),
        "metrics": metrics,
        "difference_percentage_points": (
            100.0 * (temporal - original)
            if original is not None and temporal is not None
            else None
        ),
        "per_frame": per_frame,
    }
    temp = output_dir / "results.json.tmp"
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(output_dir / "results.json")
    for mode in MODES:
        np.save(output_dir / f"confusion_{mode}.npy", confusion[mode])
    return result


def main() -> None:
    """执行严格配对的时序开关实验。"""
    parser = argparse.ArgumentParser(description="nuScenes Mini 时序 BEV 配对消融")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--limit", type=int, default=81)
    args = parser.parse_args()

    if not (1 <= args.limit <= 81):
        raise ValueError("--limit 应为 1..81；完整对照请使用默认值 81")
    for path in (CONFIG, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"找不到必要文件：{path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请激活项目 .venv-bevformer")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(SEG_REPO)
    sys.path.insert(0, str(SEG_REPO))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    importlib.import_module("projects.mmdet3d_plugin")
    config = Config.fromfile(str(CONFIG))
    dataset_config = copy.deepcopy(config.data.test)
    dataset_config.test_mode = True
    dataset_config.pop("samples_per_gpu", None)
    dataset_config.pop("workers_per_gpu", None)
    dataset = build_dataset(dataset_config)
    if len(dataset) != 81:
        raise RuntimeError(f"期望 Mini 验证集为 81 帧，实际为 {len(dataset)}")

    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_model(config.model, test_cfg=config.get("test_cfg"))
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    weights = ckpt.get("state_dict", ckpt)
    matched = model.load_state_dict(weights, strict=True)
    if matched.missing_keys or matched.unexpected_keys:
        raise RuntimeError("Checkpoint 与模型权重不匹配")
    del ckpt, weights
    gc.collect()
    model = model.cuda().eval()

    confusions = {
        mode: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for mode in MODES
    }
    states = {mode: new_temporal_state() for mode in MODES}
    per_frame = []
    prior_miou = None
    if MINI_METRICS.exists():
        prior_miou = json.loads(MINI_METRICS.read_text(encoding="utf-8")).get(
            "foreground_mIoU"
        )
    metadata = {
        "experiment": "nuScenes Mini 81-frame paired temporal BEV ablation",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "config_sha256": sha256_file(CONFIG),
        "dataset_samples": len(dataset),
        "requested_samples": args.limit,
        "modes": {
            "no_history": "video_test_mode=False; prev_bev reset each frame",
            "with_history": "video_test_mode=True; native prev_bev per scene",
        },
        "prior_mini_foreground_miou": prior_miou,
        "aggregation": "Global confusion matrix; absent classes excluded",
        "warning": (
            "仅 Mini 验证集配对消融；不得直接将此结果写成 DriveLM799 正式成绩。"
        ),
    }

    print("===== Step15 Mini 时序 BEV 配对消融 =====", flush=True)
    print("总样本：", len(dataset), "本次：", args.limit, flush=True)
    print("权重 SHA256：", metadata["checkpoint_sha256"], flush=True)
    print("每帧共用原始输入，模型内部单独维护两组 prev_bev", flush=True)

    started = time.monotonic()
    temporal_used = 0
    try:
        for index in range(args.limit):
            item = dataset[index]
            raw_image = unwrap_single_aug(item["img"])
            raw_meta = unwrap_single_aug(item["img_metas"])
            gt = np.squeeze(
                to_numpy(unwrap_single_aug(item["semantic_indices"]))
            ).astype(np.int64)
            if not torch.is_tensor(raw_image) or not isinstance(raw_meta, dict):
                raise TypeError(f"样本 {index} 的输入格式错误")
            if raw_image.ndim == 4:
                raw_image = raw_image.unsqueeze(0)
            if raw_image.ndim != 5 or tuple(raw_image.shape[:2]) != (1, 6):
                raise ValueError(f"样本 {index} 输入图像维度错误：{raw_image.shape}")
            if gt.shape != (200, 400):
                raise ValueError(f"样本 {index} GT 不是 200×400")
            if np.any((gt < 0) | (gt >= NUM_CLASSES)):
                raise ValueError(f"样本 {index} GT 类别编号异常")

            predictions = {}
            scores = {}
            image_gpu = raw_image.to("cuda:0")
            history_present = (
                states["with_history"]["prev_bev"] is not None
                and states["with_history"]["scene_token"]
                == raw_meta.get("scene_token")
            )
            for mode in MODES:
                model.video_test_mode = mode == "with_history"
                model.prev_frame_info = states[mode]
                # 第三方 extract_img_feat 会 squeeze_() 原地改变张量形状。
                # 每次推理必须单独 clone，避免第二组把摄像头维当成 Batch。
                independent_input = image_gpu.clone()
                independent_meta = copy.deepcopy(raw_meta)
                with torch.no_grad():
                    outputs = model(
                        return_loss=False,
                        rescale=True,
                        img=[independent_input],
                        img_metas=[[independent_meta]],
                    )
                prediction, shape = segmentation_classes(outputs[0]["seg_preds"])
                if prediction.shape != gt.shape:
                    raise ValueError(f"样本 {index} 输出尺寸错误：{shape}")
                predictions[mode] = prediction
                matrix = confusion_of_frame(gt, prediction)
                confusions[mode] += matrix
                _, fg_miou, _ = stats_from_confusion(matrix)
                scores[mode] = fg_miou
                states[mode] = model.prev_frame_info
                del outputs, independent_input

            if history_present:
                temporal_used += 1
            # 每个场景首帧两组都没有历史：理论上应得到完全相同的预测。
            if not history_present and not np.array_equal(
                predictions["no_history"], predictions["with_history"]
            ):
                raise RuntimeError(
                    f"样本 {index} 在无历史时两组预测不一致；请先检查实验有效性"
                )

            same = bool(np.array_equal(
                predictions["no_history"], predictions["with_history"]
            ))
            record = {
                "index": index,
                "sample_token": str(dataset.data_infos[index].get("token", "")),
                "scene_token": str(raw_meta.get("scene_token", "")),
                "history_available_before_forward": bool(history_present),
                "predictions_identical": same,
                "no_history_fg_miou": scores["no_history"],
                "with_history_fg_miou": scores["with_history"],
                "pixels_changed": int(np.count_nonzero(
                    predictions["no_history"] != predictions["with_history"]
                )),
            }
            per_frame.append(record)
            if index in (0, 1, 40, 80) or (index + 1) % 10 == 0:
                print(
                    f"[{index + 1:02d}/{args.limit}] "
                    f"has_history={history_present}, "
                    f"off={scores['no_history']:.4%}, "
                    f"on={scores['with_history']:.4%}, "
                    f"changed={record['pixels_changed']}",
                    flush=True,
                )
            if (index + 1) % 10 == 0 or index + 1 == args.limit:
                write_results(
                    OUT_DIR, confusions, per_frame, metadata, complete=False
                )
            del image_gpu, predictions
    except Exception:
        write_results(OUT_DIR, confusions, per_frame, metadata, complete=False)
        raise

    metadata["frames_with_actual_history"] = temporal_used
    result = write_results(
        OUT_DIR, confusions, per_frame, metadata, complete=True
    )
    print("\n===== 完整配对消融汇总（累计像素交并比） =====", flush=True)
    for mode in MODES:
        current = result["metrics"][mode]
        print(f"{mode}: 前景 mIoU = {current['foreground_miou']:.4%}")
        for row in current["classes"][1:]:
            value = row["iou"]
            name = row["class_name"]
            print(f"  {name}: {value:.4%}" if value is not None else f"  {name}: N/A")
    print("提升百分点：", f"{result['difference_percentage_points']:+.3f}")
    print("本次实际使用历史帧数：", temporal_used)
    if prior_miou is not None:
        difference = abs(prior_miou - result["metrics"]["with_history"]["foreground_miou"])
        print("此前 Mini mIoU：", prior_miou)
        print("与此前 Mini 汇总差值：", f"{difference:.8f}")
        if difference > 1e-4:
            print("[注意] 时序组与旧汇总不一致，应检查数据协议/模型状态后再解释", flush=True)
    print(f"总耗时：{time.monotonic() - started:.1f}s")
    print("结果文件：", OUT_DIR / "results.json", flush=True)
    print("实验只用于 Mini，未修改 DriveLM799 5.12% 正式指标。", flush=True)


if __name__ == "__main__":
    main()

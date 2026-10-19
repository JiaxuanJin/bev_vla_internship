"""在 DriveLM799 独立帧上验证固定空间变换是否跨场景有效。

仅做诊断：不修改 BEVFormer-Seg 源码、checkpoint 或原始 799 帧评估结果。
固定变换来自 Mini 第 0 帧，不能重新针对 DriveLM 样本调参。
运行：python -u scripts/step9_drivelm799_alignment.py --indices 0,100,200,300,399,500,600,700,798
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from run_bev_seg_minival import confusion_of_frame, stats_from_confusion
from run_bev_seg_one_sample import (
    CONFIG,
    ROOT,
    SEG_REPO,
    segmentation_classes,
    to_numpy,
    unwrap_single_aug,
)

DATA_PKL = ROOT / "data/drivelm/nuscenes_infos_temporal_drivelm799_officialscene.pkl"
OUTPUT = ROOT / "outputs/drivelm_check/root_cause/step9_drivelm799"
DEFAULT_CHECKPOINT = Path("/mnt/g/nus/bevformer_small_seg.pth")
DEFAULT_INDICES = "0,100,200,300,399,500,600,700,798"
CLASS_NAMES = ("background", "divider", "ped_crossing", "boundary")


def sha256_file(path: Path) -> str:
    """记录模型与配置文件的哈希，保障结果可追溯。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zero_fill_shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """二维标签执行零填充平移，不做循环位移。"""
    height, width = arr.shape
    result = np.zeros_like(arr)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    if src_y0 < src_y1 and src_x0 < src_x1:
        result[
            src_y0 + dy:src_y1 + dy,
            src_x0 + dx:src_x1 + dx,
        ] = arr[src_y0:src_y1, src_x0:src_x1]
    return result


def binary_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    """把 1、2、3 类合并为前景，只评价空间重叠。"""
    true_fg = gt > 0
    pred_fg = pred > 0
    union = np.count_nonzero(true_fg | pred_fg)
    return float(np.count_nonzero(true_fg & pred_fg) / union) if union else 1.0


def compute_scores(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, dict]:
    confusion = confusion_of_frame(gt, pred)
    classes, foreground, including_background = stats_from_confusion(confusion)
    return confusion, {
        "foreground_miou": foreground,
        "binary_foreground_iou": binary_iou(gt, pred),
        "including_background_miou": including_background,
        "classes": classes,
    }


def parse_indices(value: str) -> list[int]:
    """不依赖结果动态挑选样本；只检查用户预先确定的索引。"""
    indices = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("样本索引不能为空或重复")
    if any(index < 0 or index >= 799 for index in indices):
        raise ValueError("样本索引必须位于 0..798")
    return indices


def make_preview(gt: np.ndarray, raw: np.ndarray, shifted: np.ndarray, index: int) -> None:
    """仅展示诊断变换，不覆盖官方预测图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, array, title in zip(
        axes,
        (gt, raw, shifted),
        ("Ground Truth", "Original Prediction", "Diagnostic Flip+Shift"),
    ):
        ax.imshow(array, origin="lower", vmin=0, vmax=3, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("X pixels")
        ax.set_ylabel("Y pixels")
    fig.savefig(OUTPUT / f"sample_{index:03d}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveLM799 固定变换跨场景诊断")
    parser.add_argument("--indices", default=DEFAULT_INDICES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    indices = parse_indices(args.indices)
    checkpoint = args.checkpoint.expanduser().resolve()

    for path in (CONFIG, DATA_PKL, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"缺少实验文件：{path}")
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA，请检查 .venv-bevformer 环境")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # 与原始 799 帧的身份信息对照，避免使用另一套权重/配置误判。
    original_progress = ROOT / "outputs/drivelm_check/seg799_independent_eval/progress.json"
    if original_progress.is_file():
        original = json.loads(original_progress.read_text(encoding="utf-8"))
        identity = original.get("identity", {})
        checks = {
            "checkpoint": str(checkpoint) == identity.get("checkpoint"),
            "checkpoint_size": checkpoint.stat().st_size == identity.get("checkpoint_size"),
            "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns == identity.get("checkpoint_mtime_ns"),
            "config_sha256": sha256_file(CONFIG) == identity.get("config_sha256"),
            "dataset_sha256": sha256_file(DATA_PKL) == identity.get("dataset_sha256"),
        }
        print(f"[799帧原始实验身份核对] {checks}", flush=True)
        if not all(checks.values()):
            raise RuntimeError(
                "当前 checkpoint/配置/PKL 与原始 799 帧记录不一致，已停止。"
                "请核查原始 progress.json 后再运行，不能混合不同实验。"
            )
    else:
        raise FileNotFoundError(
            "缺少原始 799 帧 progress.json，无法确认对照实验身份一致"
        )

    # 第三方项目的数据路径是相对路径，需从 BEVFormer-Seg 根目录加载。
    os.chdir(SEG_REPO)
    sys.path.insert(0, str(SEG_REPO))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    importlib.import_module("projects.mmdet3d_plugin")
    cfg = Config.fromfile(str(CONFIG))
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.ann_file = str(DATA_PKL.resolve())
    dataset_cfg.test_mode = True
    dataset_cfg.pop("samples_per_gpu", None)
    dataset_cfg.pop("workers_per_gpu", None)
    dataset = build_dataset(dataset_cfg)
    if len(dataset) != 799:
        raise RuntimeError(f"期望 799 帧，实际 {len(dataset)}")

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    saved = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    state = saved["state_dict"] if "state_dict" in saved else saved
    model.load_state_dict(state, strict=True)
    del saved, state
    gc.collect()
    model = model.cuda().eval()
    if model.video_test_mode is not False:
        raise RuntimeError("独立帧模式没有成功关闭历史 BEV")

    print(f"[设置] 固定索引：{indices}", flush=True)
    print("[设置] 与原始 DriveLM799 相同的官方场景 PKL、独立帧协议", flush=True)
    print("[诊断] 变换预先固定：上下翻转 + dy=+6, dx=+22", flush=True)
    print("[限制] 不是正式评估指标，也不能直接当作修复结果", flush=True)

    accumulated = {
        key: np.zeros((4, 4), dtype=np.int64)
        for key in ("original", "flip_only", "fixed_flip_shift")
    }
    records = []

    for index in indices:
        item = dataset[index]
        image = unwrap_single_aug(item["img"])
        meta = unwrap_single_aug(item["img_metas"])
        gt = np.squeeze(to_numpy(unwrap_single_aug(item["semantic_indices"])))
        gt = np.asarray(gt, dtype=np.int64)
        if not isinstance(meta, dict) or not torch.is_tensor(image):
            raise TypeError(f"第 {index} 帧的数据格式异常")
        if image.ndim == 4:
            image = image.unsqueeze(0)
        if tuple(image.shape[:2]) != (1, 6) or gt.shape != (200, 400):
            raise ValueError(f"第 {index} 帧的图像或 GT 形状异常")
        if not np.isin(gt, (0, 1, 2, 3)).all():
            raise ValueError(f"第 {index} 帧存在未知 GT 类别")

        # 每帧显式清空历史 BEV，输入深拷贝避免前向原地修改影响缓存。
        model.prev_frame_info["prev_bev"] = None
        with torch.no_grad():
            output = model(
                return_loss=False,
                rescale=True,
                img=[image.to("cuda:0").clone()],
                img_metas=[[copy.deepcopy(meta)]],
            )
        pred, _ = segmentation_classes(output[0]["seg_preds"])
        if pred.shape != gt.shape:
            raise ValueError(f"第 {index} 帧预测形状不匹配")
        pred = pred.astype(np.int64, copy=False)

        variants = {
            "original": pred,
            "flip_only": np.flipud(pred),
            "fixed_flip_shift": zero_fill_shift(np.flipud(pred), 6, 22),
        }
        scores = {}
        for name, candidate in variants.items():
            confusion, score = compute_scores(gt, candidate)
            accumulated[name] += confusion
            scores[name] = score

        scene = str(dataset.data_infos[index].get("official_scene_token", ""))
        token = str(dataset.data_infos[index].get("token", ""))
        records.append(
            {"index": index, "scene_token": scene, "sample_token": token, "scores": scores}
        )
        # 只保存在新的诊断目录，不修改既有 Mini/DriveLM 原始预测。
        np.savez_compressed(OUTPUT / f"raw_sample_{index:03d}.npz", gt=gt, pred=pred)
        if index in (0, 399, 798):
            make_preview(gt, pred, variants["fixed_flip_shift"], index)
        raw_fg = scores["original"]["foreground_miou"]
        fixed_fg = scores["fixed_flip_shift"]["foreground_miou"]
        print(
            f"[{index:03d}] scene={scene[:12]} original={raw_fg:.4%} "
            f"fixed={fixed_fg:.4%} delta={(fixed_fg - raw_fg) * 100:+.2f}pp",
            flush=True,
        )
        del output, image

    aggregate = {}
    for name, matrix in accumulated.items():
        classes, foreground, with_background = stats_from_confusion(matrix)
        all_fg_gt = int(matrix[1:, :].sum())
        all_fg_pred = int(matrix[:, 1:].sum())
        common_fg = int(matrix[1:, 1:].sum())
        fg_union = all_fg_gt + all_fg_pred - common_fg
        aggregate[name] = {
            "foreground_miou": foreground,
            "including_background_miou": with_background,
            "binary_foreground_iou": common_fg / fg_union if fg_union else 1.0,
            "classes": classes,
            "confusion_matrix": matrix.tolist(),
        }

    report = {
        "scope": "DriveLM799 predefined sparse independent-frame diagnostic subset",
        "not_official_metric": True,
        "warning": "只评估固定抽样索引；空间变换不可替代原始 799 帧 mIoU。",
        "config_sha256": sha256_file(CONFIG),
        "checkpoint_sha256": sha256_file(checkpoint),
        "pkl_sha256": sha256_file(DATA_PKL),
        "checkpoint_path": str(checkpoint),
        "sample_count": len(records),
        "unique_scene_count": len({row["scene_token"] for row in records}),
        "fixed_shift_dy_dx": [6, 22],
        "rows": records,
        "aggregate": aggregate,
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUTPUT / "per_sample.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["index", "scene_token", "original_miou", "flip_only_miou",
             "fixed_flip_shift_miou", "original_binary_iou", "fixed_binary_iou"]
        )
        for row in records:
            scores = row["scores"]
            writer.writerow(
                [row["index"], row["scene_token"],
                 scores["original"]["foreground_miou"],
                 scores["flip_only"]["foreground_miou"],
                 scores["fixed_flip_shift"]["foreground_miou"],
                 scores["original"]["binary_foreground_iou"],
                 scores["fixed_flip_shift"]["binary_foreground_iou"]]
            )

    print("\n===== 预定样本的累计结果（诊断性） =====")
    for key, value in aggregate.items():
        print(f"{key}: fg mIoU={value['foreground_miou']:.4%}, "
              f"binary IoU={value['binary_foreground_iou']:.4%}")
    print(f"[样本] {len(records)} 帧；不同场景 {report['unique_scene_count']}")
    print(f"[保存] {OUTPUT / 'results.json'}")
    print(f"[保存] {OUTPUT / 'per_sample.csv'}")
    print("[提醒] 正式 799 帧 mIoU 仍以原始独立帧评估为准")


if __name__ == "__main__":
    main()

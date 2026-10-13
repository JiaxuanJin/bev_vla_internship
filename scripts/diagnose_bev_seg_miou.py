"""BEVFormer-Small 分割低 mIoU 的非破坏性诊断工具。

只读取已有的 .npy/.json/.csv 与代码配置；不重新运行模型、修改模型或更改评估结果。
检查类别比例、已有累计指标、单帧标签一致性及可能的方向/平移偏差。
注意：方向/平移扫描仅用于定位问题，其结果不能作为正式 mIoU。
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "bev_seg"
CLASSES = ["background", "divider", "ped_crossing", "boundary"]


def read_labels(path: Path, name: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    print(f"[{name}] 原始 shape={arr.shape}, dtype={arr.dtype}")
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[1] == 4:
        arr = arr.argmax(axis=1)[0]
    elif arr.ndim == 3 and arr.shape[0] == 4:
        arr = arr.argmax(axis=0)
    elif arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{name} 必须是二维标签，实际 {arr.shape}")
    if not np.isfinite(arr).all() or np.any(arr != np.round(arr)):
        raise ValueError(f"{name} 存在非有限或非整数标签")
    arr = arr.astype(np.int16)
    if arr.min() < 0 or arr.max() > 3:
        raise ValueError(f"{name} 类别范围错误: [{arr.min()}, {arr.max()}]")
    return arr


def compute_confusion(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if gt.shape != pred.shape:
        raise ValueError(f"标签形状不一致: {gt.shape} vs {pred.shape}")
    return np.bincount((gt.astype(np.int64) * 4 + pred).ravel(), minlength=16).reshape(4, 4)


def class_metrics(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(np.float64)
    gt_count = cm.sum(axis=1).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)
    union = gt_count + pred_count - tp
    iou = np.divide(tp, union, out=np.full(4, np.nan), where=union > 0)
    return {
        "class_iou": {name: None if not np.isfinite(iou[i]) else float(iou[i]) for i, name in enumerate(CLASSES)},
        "gt_pixels": {name: int(gt_count[i]) for i, name in enumerate(CLASSES)},
        "pred_pixels": {name: int(pred_count[i]) for i, name in enumerate(CLASSES)},
        "foreground_miou": float(np.nanmean(iou[1:])),
        "all_class_miou": float(np.nanmean(iou)),
        "confusion_gt_rows_pred_columns": cm.tolist(),
    }


def shift_zero(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """零填充平移，不环绕；仅用于探索潜在坐标偏移。"""
    h, w = image.shape
    result = np.zeros_like(image)
    sy0, sy1 = max(0, -dy), min(h, h - dy)
    sx0, sx1 = max(0, -dx), min(w, w - dx)
    if sy0 >= sy1 or sx0 >= sx1:
        return result
    result[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = image[sy0:sy1, sx0:sx1]
    return result


def binary_iou(gt: np.ndarray, pred: np.ndarray) -> float:
    a, b = gt > 0, pred > 0
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def scan_alignment(gt: np.ndarray, pred: np.ndarray) -> dict:
    orientations = {
        "原始方向": pred,
        "左右翻转": np.fliplr(pred),
        "上下翻转": np.flipud(pred),
        "上下左右翻转": np.flipud(np.fliplr(pred)),
    }
    raw = binary_iou(gt, pred)
    best = {"orientation": "原始方向", "dy": 0, "dx": 0, "foreground_binary_iou": raw}
    # 先粗扫，再在最佳点周围细扫；只用于诊断对齐，不修改官方评估结果。
    for name, candidate in orientations.items():
        local = (binary_iou(gt, candidate), 0, 0)
        for dy in range(-30, 31, 6):
            for dx in range(-30, 31, 6):
                score = binary_iou(gt, shift_zero(candidate, dy, dx))
                if score > local[0]:
                    local = (score, dy, dx)
        cy, cx = local[1:]
        for dy in range(cy - 5, cy + 6):
            for dx in range(cx - 5, cx + 6):
                score = binary_iou(gt, shift_zero(candidate, dy, dx))
                if score > local[0]:
                    local = (score, dy, dx)
        if local[0] > best["foreground_binary_iou"]:
            best = {"orientation": name, "dy": local[1], "dx": local[2], "foreground_binary_iou": local[0]}
    best["raw_foreground_binary_iou"] = raw
    return best


def collect_sources() -> dict:
    records = {}
    config = ROOT / "BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py"
    if config.exists():
        source = config.read_text(encoding="utf-8")
        for key in ("video_test_mode", "queue_length", "rotate_prev_bev", "use_shift", "use_can_bus", "map_grid_conf"):
            matches = re.findall(rf"^\s*{re.escape(key)}\s*=\s*[^\n]+", source, flags=re.MULTILINE)
            records[key] = matches[:3]
        print("[配置] 已读取分割配置的时序与地图参数。")
    for filename in ("run_bev_seg_one_sample.py", "run_bev_seg_minival.py"):
        p = ROOT / "scripts" / filename
        if p.exists():
            t = p.read_text(encoding="utf-8")
            records[filename] = {
                "uses_prev_bev_text": "prev_bev" in t,
                "uses_scene_token_text": "scene_token" in t,
                "uses_video_test_mode_text": "video_test_mode" in t,
                "uses_forward_test_text": "forward_test" in t,
                "uses_return_loss_text": "return_loss" in t,
            }
    return records


def main() -> None:
    print("========== BEV 分割 mIoU 非破坏性诊断 ==========")
    gt_path = OUT / "seg_gt_sample0.npy"
    pred_path = OUT / "seg_pred_sample0.npy"
    if not gt_path.is_file() or not pred_path.is_file():
        print("[缺文件] 请确认以下文件存在：")
        print(gt_path)
        print(pred_path)
        raise SystemExit(1)

    gt = read_labels(gt_path, "单帧真值")
    pred = read_labels(pred_path, "单帧预测")
    cm = compute_confusion(gt, pred)
    metrics = class_metrics(cm)
    alignment = scan_alignment(gt, pred)
    print(f"[单帧] 前景 mIoU={metrics['foreground_miou']:.6f}")
    for name in CLASSES:
        print(f"  {name:14s} IoU={metrics['class_iou'][name]} | GT={metrics['gt_pixels'][name]} | 预测={metrics['pred_pixels'][name]}")
    print(f"[空间诊断] 原始前景二值 IoU={alignment['raw_foreground_binary_iou']:.4f}")
    print(f"[空间诊断] 最佳测试方向={alignment['orientation']} 平移(dy,dx)=({alignment['dy']},{alignment['dx']})，二值 IoU={alignment['foreground_binary_iou']:.4f}")
    print("[提醒] 翻转/平移结果只是故障定位线索，不得当作模型 mIoU 成绩。")

    other_gt = OUT / "seg_gt_gpu_sample0.npy"
    extra = {}
    if other_gt.is_file():
        gt2 = read_labels(other_gt, "GPU 推理时使用的真值")
        extra["gt_between_scripts_equal"] = bool(np.array_equal(gt, gt2))
        extra["gt_between_scripts_different_pixels"] = int(np.count_nonzero(gt != gt2)) if gt.shape == gt2.shape else None
        print(f"[真值一致性] 两次脚本的标签完全相同：{extra['gt_between_scripts_equal']}")

    accumulated = OUT / "seg_minival_confusion.npy"
    if accumulated.is_file():
        arr = np.load(accumulated, allow_pickle=False)
        if arr.shape == (4, 4) and np.issubdtype(arr.dtype, np.integer):
            extra["minival_81_aggregate"] = class_metrics(arr)
            print(f"[81帧累计混淆矩阵] 前景 mIoU={extra['minival_81_aggregate']['foreground_miou']:.6f}")
        else:
            print(f"[提醒] 累计混淆矩阵形状或类型不符合预期: {arr.shape}, {arr.dtype}")

    summary_path = OUT / "seg_minival_metrics.json"
    if summary_path.is_file():
        saved = json.loads(summary_path.read_text(encoding="utf-8"))
        extra["existing_metrics_keys"] = list(saved.keys()) if isinstance(saved, dict) else []
        print(f"[原始指标] JSON 字段={extra['existing_metrics_keys']}")

    csv_path = OUT / "seg_minival_per_frame.csv"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        extra["per_frame_csv_rows"] = len(rows)
        print(f"[每帧指标] CSV 行数={len(rows)}")

    sources = collect_sources()
    for key, value in sources.items():
        if key.endswith(".py"):
            print(f"[时序线索] {key}: {value}")

    # 生成诊断图：仅展示原始预测、标签以及分歧，不篡改预测。
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(gt, cmap="viridis", vmin=0, vmax=3, origin="lower")
    axes[0].set_title("Ground Truth")
    axes[1].imshow(pred, cmap="viridis", vmin=0, vmax=3, origin="lower")
    axes[1].set_title("Raw Prediction")
    axes[2].imshow(gt != pred, cmap="gray_r", vmin=0, vmax=1, origin="lower")
    axes[2].set_title("Pixel Disagreement (white=match)")
    for ax in axes:
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
    fig.tight_layout()
    figure = OUT / "seg_diagnostic_sample0.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)

    report = {
        "dataset": "nuScenes Mini 81 samples; not DriveLM official validation",
        "note": "Do not substitute diagnostic alignment scores for official mIoU",
        "sample0": metrics,
        "alignment_diagnostic": alignment,
        "additional_checks": extra,
        "source_code_inspection": sources,
    }
    json_path = OUT / "seg_miou_diagnostic.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[完成] 诊断结果：{json_path}")
    print(f"[完成] 诊断图片：{figure}")
    print("[下一步] 根据结果判断需要检查类别映射、几何坐标还是时序调用。")


if __name__ == "__main__":
    main()

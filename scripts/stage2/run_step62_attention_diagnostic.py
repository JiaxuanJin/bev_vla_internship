"""Step62：Step61 训练后语言注意力质量与阈值敏感性诊断。

使用 Step61 相同的 Mini Val 6 帧，直接加载冻结作者 BEVFormer、
训练后的 LanguageBEVFusion 和三类 DistilBERT 文本向量。
仅只读推理、可视化及诊断，不改 checkpoint，不调参，不训练。
阈值扫描属于分析，禁止将扫描到的最优阈值作为独立测试集成绩。
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STEP61 = ROOT / "outputs/stage2_step61"
OUT = ROOT / "outputs/stage2_step62"
AUTHOR = Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg"
CHECKPOINT = Path("/mnt/g/nus/bevformer_small_seg.pth")
NAMES = ("Divider", "Ped Crossing", "Boundary")
THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def parse_args() -> argparse.Namespace:
    """解析验证输入，默认与 Step61 使用完全相同的六帧。"""
    parser = argparse.ArgumentParser(description="Step62 attention AUPRC / threshold / maps")
    parser.add_argument("--author", type=Path, default=AUTHOR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--step61", type=Path, default=STEP61)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--skip-figures", action="store_true", help="仅输出指标")
    return parser.parse_args()


def precision_recall_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """逐像素排序计算 Average Precision（PR 曲线阶梯面积）。"""
    x = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return None
    if not np.isfinite(x).all():
        raise ValueError("注意力出现 NaN / Inf")
    # 将相同注意力值视为并列，避免常数预测受像素排列顺序影响。
    order = np.argsort(-x, kind="stable")
    sorted_x = x[order]
    sorted_y = y[order]
    group_end = np.r_[np.flatnonzero(np.diff(sorted_x) != 0), len(x) - 1]
    total_tp = np.cumsum(sorted_y, dtype=np.float64)[group_end]
    total_pred = group_end.astype(np.float64) + 1.0
    precision = total_tp / total_pred
    delta_recall = np.diff(np.r_[0.0, total_tp / n_pos])
    return float(np.sum(precision * delta_recall))


def binary_stats(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, Any]:
    """累计 TP/FP/FN 计算 IoU、Precision 与 Recall。"""
    x = np.asarray(scores).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    p = x >= threshold
    tp = int(np.count_nonzero(p & y))
    fp = int(np.count_nonzero(p & ~y))
    fn = int(np.count_nonzero(~p & y))
    union = tp + fp + fn
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_fraction": float(np.mean(p)),
        "iou": float(tp / union) if union else None,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
    }


def top_fraction_stats(scores: np.ndarray, labels: np.ndarray, fraction: float) -> dict[str, Any]:
    """无阈值的 Top-K 评价；fraction 预先固定，不用 GT 选 K。"""
    x = np.asarray(scores).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    k = max(1, int(round(len(x) * fraction)))
    picks = np.argpartition(x, -k)[-k:]
    tp = int(np.count_nonzero(y[picks]))
    positives = int(np.count_nonzero(y))
    union = k + positives - tp
    return {
        "top_fraction": float(fraction),
        "top_pixels": k,
        "positive_pixels": positives,
        "precision_at_k": tp / k,
        "recall_at_k": tp / positives if positives else None,
        "iou": tp / union if union else None,
    }


def assess(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """计算概率分布、排序质量与各阈值表现，避免仅看 IoU@0.5。"""
    x = np.asarray(scores, dtype=np.float32).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    if not np.isfinite(x).all() or x.min() < -1e-4 or x.max() > 1.0001:
        raise ValueError("注意力必须为有限 [0,1] 概率")
    n_pos = int(np.count_nonzero(y))
    n = int(x.size)
    thresholds = [binary_stats(x, y, t) for t in THRESHOLDS]
    valid = [v for v in thresholds if v["iou"] is not None]
    best = max(valid, key=lambda v: v["iou"]) if valid else None
    positive_mean = float(x[y].mean()) if n_pos else None
    negative_mean = float(x[~y].mean()) if n_pos < n else None
    return {
        "pixels": n,
        "gt_positive_pixels": n_pos,
        "gt_prevalence": n_pos / n,
        "prediction_mean": float(x.mean()),
        "prediction_std": float(x.std()),
        "prediction_quantiles": {
            str(q): float(np.quantile(x, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "positive_minus_negative_mean": (
            positive_mean - negative_mean
            if positive_mean is not None and negative_mean is not None else None
        ),
        "average_precision": precision_recall_auc(x, y),
        "prevalence_random_baseline": n_pos / n,
        "thresholds": thresholds,
        "best_threshold_on_same_eval_set_diagnostic_only": best,
        "top_fraction_diagnostics": [
            top_fraction_stats(x, y, fraction) for fraction in (0.05, 0.1, 0.2)
        ],
    }


def save_figures(
    records: dict[str, list[dict[str, Any]]], output: Path, full: dict[str, Any]
) -> list[str]:
    """生成 GT/热图/阈值掩码和阈值曲线，仅诊断而非训练。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for name in NAMES:
        values = full[name]["thresholds"]
        ax.plot([v["threshold"] for v in values], [v["iou"] for v in values],
                marker="o", label=name)
    ax.set(xlabel="Attention probability threshold", ylabel="Attention IoU",
           title="Step62 / held-out Mini Val (diagnostic only)")
    ax.grid(alpha=0.2)
    ax.legend()
    path = output / "step62_threshold_scan.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path.name)

    for name in NAMES:
        # 只根据真实 GT 正像素数选展示帧，避免按模型预测结果挑好看的图。
        samples = records[name]
        candidates = [r for r in samples if int(r["mask"].sum()) > 0]
        if not candidates:
            continue
        record = sorted(candidates, key=lambda r: (-int(r["mask"].sum()), r["token"]))[0]
        heat = record["score"]
        mask = record["mask"]
        fig, axs = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
        axs[0].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axs[0].set_title("GT mask")
        hm = axs[1].imshow(heat, cmap="magma", vmin=0, vmax=1)
        axs[1].set_title("Learned attention")
        fig.colorbar(hm, ax=axs[1], fraction=0.047)
        axs[2].imshow(heat >= 0.5, cmap="gray", vmin=0, vmax=1)
        axs[2].set_title("Predicted >= 0.5")
        axs[3].imshow(heat, cmap="magma", vmin=0, vmax=1)
        axs[3].contour(mask.astype(np.float32), levels=[0.5], colors="cyan", linewidths=0.75)
        axs[3].set_title("Attention + GT outline")
        for axis in axs:
            axis.set_xticks([])
            axis.set_yticks([])
        fig.suptitle(f"{name}  |  token={record['token'][:12]}  |  Mini Val")
        path = output / f"step62_{name.lower().replace(' ', '_')}_overlay.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path.name)
        # 保存同一帧原始概率和 GT，方便进一步校验，避免只能看图。
        np.savez_compressed(
            output / f"step62_{name.lower().replace(' ', '_')}_sample.npz",
            attention=heat.astype(np.float32), gt_mask=mask.astype(np.uint8),
            token=np.array(record["token"]),
        )
    return paths


def self_test() -> None:
    """纯 NumPy 测试：无需 nuScenes、torch、作者权重。"""
    x = np.array([0.9, 0.7, 0.3, 0.1])
    y = np.array([True, True, False, False])
    stats = assess(x, y)
    assert abs(stats["average_precision"] - 1) < 1e-12
    assert abs(binary_stats(x, y, 0.5)["iou"] - 1) < 1e-12
    assert top_fraction_stats(x, y, 0.5)["iou"] == 1.0
    tied = assess(np.array([0.5, 0.5, 0.5, 0.5]), y)
    assert abs(tied["average_precision"] - 0.5) < 1e-12
    empty = assess(np.array([0.1, 0.2]), np.array([False, False]))
    assert empty["average_precision"] is None
    print("[Step62] CPU 自检 PASS：IoU、AP、TopK、空 GT", flush=True)


def main() -> None:
    """在 Step61 同一组 Mini Val 样本上执行只读注意力诊断。"""
    if "--self-test" in sys.argv:
        self_test()
        return
    args = parse_args()
    args.author = args.author.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.step61 = args.step61.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    summary_path = args.step61 / "step61_supervised_summary.json"
    weights_path = args.step61 / "fusion_step61.pt"
    text_path = args.step61 / "class_prompts.npy"
    config_path = args.author / "projects/configs/bevformer/bevformer_small_seg.py"
    for path in (summary_path, weights_path, text_path, args.checkpoint, config_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少 Step62 输入文件：{path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    val_indices = summary.get("val_indices")
    val_tokens = summary.get("val_tokens")
    if not isinstance(val_indices, list) or len(val_indices) != 6:
        raise ValueError("Step61 验证索引不是六帧，拒绝改变 A/B 样本")
    if len(set(val_indices)) != 6 or not all(isinstance(i, int) and 0 <= i < 81 for i in val_indices):
        raise ValueError("Step61 Mini Val 索引异常")
    if not isinstance(val_tokens, list) or len(val_tokens) != 6:
        raise ValueError("Step61 val_tokens 未记录")
    if summary.get("temporal_mode") != "OFF, independent frames":
        raise ValueError("Step61 不是独立帧实验，停止")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(args.author))
    os.chdir(args.author)
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from scripts.stage2.bev_fusion import LanguageBEVFusion
    from scripts.stage2.run_step59_train import build_mini_dataset, capture_samples, sha256
    from scripts.stage2.run_step60_extended_ab import IndexedDataset, move_to_cpu, cuda_sample, print_memory
    from scripts.stage2.run_step61_supervised_train import project_attention

    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    if not torch.cuda.is_available():
        raise RuntimeError("需要作者环境 CUDA 来提取真实 BEV 特征")
    if sha256(args.checkpoint) != summary["checkpoint_sha256"]:
        raise RuntimeError("作者 checkpoint 哈希与 Step61 不一致")
    if sha256(weights_path) != summary["weights_sha256"]:
        raise RuntimeError("Step61 融合权重 SHA256 与 JSON 不一致")
    if sha256(text_path) != summary["text_sha256"]:
        raise RuntimeError("Step61 文本向量 SHA256 与 JSON 不一致")
    started = time.perf_counter()
    print("[Step62] 已核对作者权重、语言文本和融合权重 SHA256", flush=True)
    text_vectors = np.load(text_path, allow_pickle=False)
    if text_vectors.shape != (3, 768) or not np.isfinite(text_vectors).all():
        raise ValueError("Step61 文本向量必须是 (3,768) 且有限")
    text = torch.tensor(text_vectors, dtype=torch.float32, device="cuda")
    cfg = Config.fromfile(str(config_path))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    cfg.data.workers_per_gpu = 0
    print("[Step62] 构建冻结作者模型，只抽取 Step61 的 6 帧 Mini Val", flush=True)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.requires_grad_(False)
    model.eval()
    wrapped = MMDataParallel(model.cuda(), device_ids=[0])
    head = wrapped.module.pts_bbox_head
    transformer = head.transformer
    print_memory("作者模型加载完成")
    dataset = build_mini_dataset(cfg, build_dataset, "val")
    subset = IndexedDataset(dataset, val_indices)
    samples = capture_samples(
        wrapped, transformer, head, subset, list(range(len(subset))),
        build_dataloader, cfg.data.nonshuffler_sampler, torch, "mini_val"
    )
    samples = move_to_cpu(samples)
    actual = [s.token for s in samples]
    if actual != val_tokens:
        raise RuntimeError(f"Step61 与 Step62 验证 token 或顺序不同：{actual} != {val_tokens}")
    del dataset, subset
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("六帧特征缓存完成")

    fusion = LanguageBEVFusion(text_dim=768, alpha_init=float(summary["alpha_fixed"]))
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("Step61 权重结构异常")
    fusion.load_state_dict(payload["state_dict"], strict=True)
    fusion.eval().cuda().requires_grad_(False)
    grouped: dict[str, list[dict[str, Any]]] = {n: [] for n in NAMES}
    print("[Step62] 计算注意力：3 类指令 × 6 帧", flush=True)
    with torch.inference_mode():
        for index, item in enumerate(samples):
            bev, gt, _ = cuda_sample(item, torch)
            gt_np = gt[0].cpu().numpy()
            for class_id, name in enumerate(NAMES, 1):
                _, raw_attention = fusion(bev, text[class_id-1:class_id])
                projected = project_attention(head, raw_attention, torch)
                score = projected[0, 0].detach().cpu().numpy().astype(np.float32)
                gt_mask = np.equal(gt_np, class_id)
                if score.shape != (200, 400) or gt_mask.shape != (200, 400):
                    raise ValueError("预测 / GT 尺寸异常")
                grouped[name].append({
                    "token": item.token,
                    "score": score,
                    "mask": gt_mask,
                })
            print(f"[Step62] {index + 1}/6 token={item.token[:12]}", flush=True)
            del bev, gt
    results: dict[str, Any] = {}
    for name in NAMES:
        stack_score = np.stack([r["score"] for r in grouped[name]])
        stack_mask = np.stack([r["mask"] for r in grouped[name]])
        stats = assess(stack_score, stack_mask)
        stats["frames_with_gt"] = int(sum(r["mask"].any() for r in grouped[name]))
        stats["empty_gt_frames"] = [
            r["token"] for r in grouped[name] if not r["mask"].any()
        ]
        results[name] = stats
        fixed = next(x for x in stats["thresholds"] if x["threshold"] == 0.5)
        previous_iou = summary["after"]["per_prompt"][name]["attention"]["attention_iou_at_0_5"]
        if previous_iou is None or fixed["iou"] is None or abs(fixed["iou"] - previous_iou) > 1e-5:
            raise RuntimeError(
                f"Step62 {name} IoU@0.5 与 Step61 不一致："
                f"{fixed['iou']} vs {previous_iou}；停止结论"
            )
        print(
            f"[Step62] {name}: GT={stats['gt_prevalence']:.2%}; "
            f"IoU@0.5={fixed['iou']:.2%}; AP={stats['average_precision']:.3f}; "
            f"random AP≈{stats['prevalence_random_baseline']:.3f}; "
            f"最佳阈值(仅诊断)={stats['best_threshold_on_same_eval_set_diagnostic_only']['threshold']}",
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    figure_paths: list[str] = []
    if not args.skip_figures:
        try:
            figure_paths = save_figures(grouped, args.output, results)
        except ImportError as exc:
            raise ImportError("作者环境缺 matplotlib，安装后重试；或指定 --skip-figures") from exc
    import csv

    with (args.output / "step62_threshold_scan.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "class", "threshold", "iou", "precision", "recall", "predicted_fraction", "tp", "fp", "fn"
        ])
        writer.writeheader()
        for name in NAMES:
            for row in results[name]["thresholds"]:
                writer.writerow({"class": name, **row})
    report: dict[str, Any] = {
        "experiment": "step62_step61_supervised_attention_diagnostic",
        "scope": "same 6 held-out Mini Val frames from Step61; temporal OFF",
        "val_indices": val_indices,
        "val_tokens": actual,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "step61_fusion_sha256": summary["weights_sha256"],
        "text_sha256": summary["text_sha256"],
        "alignment": "same author seg head flip/transpose and feat_cropper, replay verified",
        "thresholds_predefined": list(THRESHOLDS),
        "scores": results,
        "figures": figure_paths,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "notes": [
            "No training, no tuning. Threshold scans on validation frames are exploratory only.",
            "Average Precision is compared with class prevalence, a no-skill reference.",
            "Top-k uses predefined area percentages for diagnostic ranking, not inference.",
            "Segmentation Head geometry replay is verified; true physical GT alignment is not proven.",
            "Step61 ground-truth class prompts are not open-vocabulary language grounding.",
        ],
    }
    json_path = args.output / "step62_diagnostics.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Step62：语言注意力阈值与空间质量诊断", "",
        "完全复用 Step61 Mini Val 6 帧与训练后的语言融合权重，不训练、不修改原始权重。", "",
        "| 类别 | GT面积% | Attention IoU@0.5 | AP | 随机参考AP | 验证内最优阈值（仅诊断） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        x = results[name]
        iou5 = next(a for a in x["thresholds"] if a["threshold"] == 0.5)["iou"]
        lines.append(
            f"| {name} | {100*x['gt_prevalence']:.3f} | "
            f"{100*iou5:.3f}% | {x['average_precision']:.5f} | "
            f"{x['prevalence_random_baseline']:.5f} | "
            f"{x['best_threshold_on_same_eval_set_diagnostic_only']['threshold']:.2f} |"
        )
    lines.extend([
        "", "## 判断规则", "",
        "- 若 AP 明显高于 GT 占比、但 IoU@0.5 很低，优先检查概率校准/阈值，而不是先断言空间错位。",
        "- 若 AP 约等于 GT 占比或更差，模型基本没有有效像素排名，需要检查监督、几何协议和特征。",
        "- 正负样本注意力均值差若为负，说明在该类别上方向可能反了；仍需结合图像判断。",
        "- 本报告的最佳阈值是在同一组 Mini Val 上回看得到，不能充当独立验证提升或最终部署阈值。",
        "- 数据规模仅六帧、时序 OFF，且类别指令不是开放自然语言定位监督，不能与作者 81 帧 mIoU 混用。",
        "", "## 输出文件", "",
        "- `step62_diagnostics.json`：全量诊断与哈希。",
        "- `step62_threshold_scan.csv`：固定阈值下 TP/FP/FN/IoU/Precision/Recall。",
        "- `step62_threshold_scan.png`：阈值敏感性曲线。",
        "- `step62_*_overlay.png` 与 `step62_*_sample.npz`：同一帧 GT、热图、预测及数值证据。",
        "",
    ])
    md = args.output / "step62_review.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Step62] 完成：{json_path}", flush=True)
    print(f"[Step62] 中文报告：{md}", flush=True)
    print("[Step62] 下一步根据 AP vs GT 占比判断重训方向", flush=True)


if __name__ == "__main__":
    main()

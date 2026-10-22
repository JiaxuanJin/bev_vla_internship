"""Step67：一次完成轻量续训、Mini Train 校准、锁定后的 Mini Val A/B。

设计约束：
- 冻结原作者 BEVFormer、分割 Decoder、DistilBERT；原始 checkpoint 不写入。
- 训练只读 Mini Train GT；Mini Val 的 GT 只用于最后一次报告。
- Mini Train 的校准帧与 Step67 参数更新帧互斥，且在 Mini Val 加载前锁定方案。
- 严格复用 Step65/66 的 BEV 提取、feat_cropper 与四类混淆矩阵协议。
- 不假定任何配置必然胜过原模型；若训练校准不满足约束，回退原始分支。

文献启发（并非复现）：CVPR 2023 BLV 类别长尾 logit 调整；Tversky 的 FP/FN 权衡；
LAVT 的文本-目标监督思想。项目中使用的是原创的有限候选推理残差校准。
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DEFAULT = ROOT / "outputs/stage2_step67"
STEP65_SUMMARY = ROOT / "outputs/stage2_step65/step65_ab_summary.json"
STEP66_SUMMARY = ROOT / "outputs/stage2_step66/step66_ab_summary.json"
STEP65_WEIGHT = ROOT / "outputs/stage2_step65/adapter_step65.pt"
STEP65_BIAS = ROOT / "outputs/stage2_step65/bias_step65.pt"
STEP66_WEIGHT = ROOT / "outputs/stage2_step66/adapter_step66.pt"
SEED = 20261021 + 67
LABELS = ("Background", "Divider", "Ped Crossing", "Boundary")
LAMBDA_GRID = (0.15, 0.30, 0.50, 0.70, 1.0)
GUARDS = ("none", "soft", "strong")


@dataclass(frozen=True)
class Candidate:
    """有限候选空间；只允许在 Mini Train 校准子集搜索。"""

    source: str
    strength: float
    guard: str

    @property
    def name(self) -> str:
        """可读候选名。"""
        if self.source == "original":
            return "original"
        return f"{self.source}_lambda{self.strength:.2f}_{self.guard}"


def candidate_grid() -> list[Candidate]:
    """在加载任何 Mini Val 数据前固定校准候选。"""
    return [Candidate("original", 0.0, "none")] + [
        Candidate(source, strength, guard)
        for source in ("step65", "step67")
        for guard in GUARDS
        for strength in LAMBDA_GRID
    ]


def conditional_logits(
    reference: torch.Tensor,
    proposal: torch.Tensor,
    candidate: Candidate,
) -> torch.Tensor:
    """可控残差组合，绝不使用 GT 决定门控。"""
    if candidate.source == "original":
        return reference
    if reference.shape != proposal.shape:
        raise ValueError("原始 logits 和候选 logits 尺寸不一致")
    if candidate.guard == "none":
        gate: torch.Tensor | float = 1.0
    else:
        p_bg = reference.detach().softmax(dim=1)[:, :1]
        if candidate.guard == "soft":
            gate = 0.25 + 0.75 * torch.sigmoid(4 * (0.82 - p_bg))
        elif candidate.guard == "strong":
            gate = 0.10 + 0.90 * torch.sigmoid(8 * (0.78 - p_bg))
        else:
            raise ValueError(f"不支持的 guard：{candidate.guard}")
    return reference + candidate.strength * gate * (proposal - reference)


def losses_for_refinement(
    logits: torch.Tensor,
    reference: torch.Tensor,
    gt: torch.Tensor,
    attention: torch.Tensor,
    step65: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """兼顾背景误检与前景漏检的损失；仅在 Mini Train 上反传。"""
    if gt.ndim != 3 or logits.shape[0] != gt.shape[0]:
        raise ValueError("GT 与 logits 维度不一致")
    weights = logits.new_tensor([0.6, 1.05, 1.2, 1.05])
    ce = F.cross_entropy(logits, gt.long(), weight=weights)
    soft_iou = step65.soft_iou_loss(logits, gt)
    log_probs = logits.log_softmax(dim=1)
    probs = log_probs.exp()
    bg = gt == 0
    fg = (gt > 0) & (gt < 4)
    bg_ce = -log_probs[:, 0][bg].mean() if bg.any() else logits.sum() * 0
    # 显式抑制前景被吞进背景，修复 Step66 在校准前背景过强的问题。
    fg_recall = (-torch.log1p(-probs[:, 0].clamp(max=1 - 1e-6)))[fg].mean() if fg.any() else logits.sum() * 0
    targets = F.one_hot(gt.long(), num_classes=4).permute(0, 3, 1, 2)[:, 1:].float()
    predicted = probs[:, 1:]
    tp = (predicted * targets).sum((0, 2, 3))
    fp = (predicted * (1 - targets)).sum((0, 2, 3))
    fn = ((1 - predicted) * targets).sum((0, 2, 3))
    # FP/FN 各 0.5，避免 Step66 的 0.75 FP 权重造成前景召回崩塌。
    balanced_tversky = (1 - (tp + 1) / (tp + 0.5 * fp + 0.5 * fn + 1)).mean()
    att = step65.supervised_attention_loss(attention, gt)
    if fg.any():
        attention_logits = torch.logit(attention.clamp(1e-5, 1 - 1e-5))
        prompt_target = (gt.long() - 1).clamp(0, 2)
        contrast = F.cross_entropy(attention_logits, prompt_target, reduction="none")[fg].mean()
    else:
        contrast = logits.sum() * 0
    residual = (logits - reference.detach()).square().mean()
    loss = (
        0.42 * ce + 0.35 * soft_iou + 0.08 * balanced_tversky
        + 0.04 * bg_ce + 0.08 * fg_recall + 0.05 * att
        + 0.02 * contrast + 0.003 * residual
    )
    return loss, {
        "ce": float(ce.detach()),
        "soft_iou": float(soft_iou.detach()),
        "tversky": float(balanced_tversky.detach()),
        "background_ce": float(bg_ce.detach()),
        "foreground_recall_loss": float(fg_recall.detach()),
        "attention": float(att.detach()),
        "prompt_contrast": float(contrast.detach()),
        "residual_penalty": float(residual.detach()),
    }


def metrics_from_confusion(matrix: torch.Tensor) -> dict[str, Any]:
    """严格按整组像素的累计混淆矩阵计算四类 IoU。"""
    m = matrix.cpu().numpy().astype(np.int64)
    per: dict[str, float | None] = {}
    for k, label in enumerate(LABELS):
        denominator = int(m[k, :].sum() + m[:, k].sum() - m[k, k])
        per[label] = float(m[k, k] / denominator) if denominator else None
    if any(per[k] is None for k in LABELS):
        raise ValueError("某一类别在汇总样本中未出现；需调整评估抽样，而不是默认为 0")
    gt_bg = int(m[0, :].sum())
    total = int(m.sum())
    return {
        "foreground_miou": float(np.mean([per[k] for k in LABELS[1:]])),
        "four_class_miou": float(np.mean([per[k] for k in LABELS])),
        "background_iou": float(per["Background"]),
        "bg_to_fg_false_positive_rate": float(m[0, 1:].sum() / gt_bg),
        "predicted_foreground_ratio": float(m[:, 1:].sum() / total),
        "gt_foreground_ratio": float(m[1:, :].sum() / total),
        "per_class_iou": per,
        "confusion_gt_rows_pred_cols": m.tolist(),
        "frame_count": total // (200 * 400),
    }


def update_confusion(matrix: torch.Tensor, gt: torch.Tensor, logits: torch.Tensor) -> None:
    """四类 argmax 统计；仅计算结果，不对标签做变换。"""
    g = gt.long().reshape(-1)
    p = logits.argmax(dim=1).long().reshape(-1)
    matrix += torch.bincount(4 * g + p, minlength=16).reshape(4, 4).cpu()


def evaluate_calibration(
    samples: list[Any],
    head: Any,
    text: torch.Tensor,
    model65: nn.Module,
    model67: nn.Module,
    candidates: list[Candidate],
) -> dict[str, dict[str, Any]]:
    """Mini Train 校准，全部候选同样本、同 GT；禁止碰 Mini Val。"""
    from scripts.stage2.run_step60_extended_ab import cuda_sample

    matrix = {c.name: torch.zeros(4, 4, dtype=torch.int64) for c in candidates}
    model65.eval()
    model67.eval()
    with torch.inference_mode():
        for sample in samples:
            bev, gt, ref = cuda_sample(sample, torch)
            old, _ = model65(bev, ref, text, head.feat_cropper)
            new, _ = model67(bev, ref, text, head.feat_cropper)
            proposals = {"step65": old, "step67": new}
            for c in candidates:
                pred = conditional_logits(ref, proposals[c.source], c) if c.source != "original" else ref
                update_confusion(matrix[c.name], gt, pred)
            del bev, gt, ref, old, new
    return {name: metrics_from_confusion(cm) for name, cm in matrix.items()}


def select_locked_candidate(
    results: dict[str, dict[str, Any]],
    candidates: list[Candidate],
) -> tuple[Candidate, list[dict[str, Any]], str]:
    """固定约束 + 统一排序；只依赖 Mini Train 校准集指标。"""
    baseline = results["original"]
    scored = []
    passed: list[Candidate] = []
    for candidate in candidates:
        m = results[candidate.name]
        valid = (
            m["foreground_miou"] >= baseline["foreground_miou"]
            and m["four_class_miou"] > baseline["four_class_miou"] + 1e-9
            and m["bg_to_fg_false_positive_rate"]
            <= baseline["bg_to_fg_false_positive_rate"] + 0.05
        )
        scored.append({"candidate": asdict(candidate), "name": candidate.name,
                       "eligible": valid, "metrics": m})
        if valid:
            passed.append(candidate)
    if not passed:
        return Candidate("original", 0.0, "none"), scored, "无满足三项约束的候选，安全回退原始模型"
    best = max(
        passed,
        key=lambda c: (results[c.name]["four_class_miou"],
                       results[c.name]["foreground_miou"]),
    )
    return best, scored, "依据 Mini Train 校准四类 IoU 选定，且满足前景和背景 FP 约束"


def evaluate_validation(
    samples: list[Any],
    head: Any,
    text: torch.Tensor,
    model65: nn.Module,
    bias65: nn.Module,
    model66: nn.Module,
    model67: nn.Module,
    chosen: Candidate,
) -> dict[str, Any]:
    """Mini Val 只评估锁定后的分支，绝不重选参数。"""
    from scripts.stage2.run_step60_extended_ab import cuda_sample

    names = (
        "original", "bias_step65", "adapter_step65", "adapter_step66",
        "adapter_step67_raw", "step67_locked", "step67_locked_text_shuffled",
        "step67_locked_text_constant",
    )
    confusion = {n: torch.zeros(4, 4, dtype=torch.int64) for n in names}
    shuffled = text[[2, 0, 1]]
    constant = text.mean(dim=0, keepdim=True).expand_as(text).contiguous()
    for model in (model65, bias65, model66, model67):
        model.eval()
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            bev, gt, ref = cuda_sample(sample, torch)
            old, _ = model65(bev, ref, text, head.feat_cropper)
            new, _ = model67(bev, ref, text, head.feat_cropper)
            guarded, _ = model66(bev, ref, text, head.feat_cropper)
            active_model = model65 if chosen.source == "step65" else model67
            if chosen.source == "original":
                selected = ref
                selected_shuffled = ref
                selected_constant = ref
            else:
                proposal = old if chosen.source == "step65" else new
                shuffled_proposal, _ = active_model(bev, ref, shuffled, head.feat_cropper)
                constant_proposal, _ = active_model(bev, ref, constant, head.feat_cropper)
                selected = conditional_logits(ref, proposal, chosen)
                selected_shuffled = conditional_logits(ref, shuffled_proposal, chosen)
                selected_constant = conditional_logits(ref, constant_proposal, chosen)
            preds = {
                "original": ref,
                "bias_step65": bias65(ref),
                "adapter_step65": old,
                "adapter_step66": guarded,
                "adapter_step67_raw": new,
                "step67_locked": selected,
                "step67_locked_text_shuffled": selected_shuffled,
                "step67_locked_text_constant": selected_constant,
            }
            for name, pred in preds.items():
                update_confusion(confusion[name], gt, pred)
            if i + 1 == len(samples) or (i + 1) % 6 == 0:
                print(f"[Step67] 验证 {i+1}/{len(samples)} 帧", flush=True)
            del bev, gt, ref, old, new, guarded, selected
    return {name: metrics_from_confusion(m) for name, m in confusion.items()}


def report_text(report: dict[str, Any]) -> str:
    """中文报告，严格区分校准结果与 Mini Val 单次验证。"""
    chosen = report["locked_candidate"]
    lines = [
        "# Step67：Mini Train 受控校准、轻量续训及一次性验证", "",
        "## 实验说明", "",
        "- 原始 BEVFormer、Decoder 和 DistilBERT 均冻结；不修改作者 checkpoint。",
        "- mini_train 更新帧与校准帧互斥；校准完成锁定后才加载 mini_val。",
        "- 固定 argmax 四类分割，无 GT 推理泄漏、无 Mini Val 阈值选择。",
        f"- 校准选定：`{chosen['name']}`；{report['selection_reason']}。",
        "- Step65/66 已查看过的 30 帧单列，新增加 12 帧属于同一 Mini Val 数据集。",
        "- 空间 GT 与模型权重的物理协议问题尚未闭环。", "",
        "## 校准集（Mini Train 独立于本轮训练）", "",
        "| 指标 | Original | 锁定候选 |", "|---|---:|---:|",
    ]
    reference = report["calibration"]["original"]
    decided = report["calibration"][chosen["name"]]
    for title, key in (("前景 mIoU", "foreground_miou"), ("四类 mIoU", "four_class_miou"),
                       ("背景 IoU", "background_iou"), ("BG→前景误检率", "bg_to_fg_false_positive_rate")):
        lines.append(f"| {title} | {reference[key]:.4%} | {decided[key]:.4%} |")
    lines.extend(["", "## 锁定后 Mini Val A/B（仅测试，不调参）", ""])
    for split, result in report["metrics"].items():
        lines.extend([f"### {split}", "", "| 分支 | 前景 mIoU | 四类 mIoU | BG IoU | BG→前景 FP |",
                      "|---|---:|---:|---:|---:|"])
        for name, m in result.items():
            lines.append(f"| {name} | {m['foreground_miou']:.4%} | {m['four_class_miou']:.4%} | "
                         f"{m['background_iou']:.4%} | {m['bg_to_fg_false_positive_rate']:.4%} |")
        lines.append("")
    lines.extend([
        "## 解释边界", "",
        "- 某验证组上偶然变好并不保证多场景或完整 81 帧时序泛化。",
        "- 如果置乱/常量文本与正常指令接近，不能宣称语言语义有效。",
        "- 如果 Mini Train 校准没有有效候选，程序主动选 original；这不是训练错误。",
        "- 新增 12 帧也与前期实验属于同一 Mini Val，不是外部独立数据集。",
        "- 论文为架构/损失启发，不是完整方法复现。", "",
        "## 参考研究及代码", "",
        "- BLV（CVPR 2023）：https://github.com/grantword8/BLV",
        "- Tversky 实现：https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/losses/tversky.py",
        "- LAVT（CVPR 2022）：https://github.com/yz93/LAVT-RIS", "",
    ])
    return "\n".join(lines)


def safe_load(path: Path) -> dict[str, Any]:
    """只读取本地、项目已生成的权重；不加载来源不明的 pickle。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def cpu_self_test() -> None:
    """快速 CPU 自检：门控方向、选优约束、混淆矩阵、梯度。"""
    torch.manual_seed(67)
    ref = torch.randn((1, 4, 8, 8))
    proposed = ref + torch.ones_like(ref) * 0.3
    for c in candidate_grid():
        result = conditional_logits(ref, proposed, c)
        assert result.shape == ref.shape
        if c.source == "original":
            assert torch.equal(result, ref)
    gt = torch.randint(0, 4, (1, 8, 8))
    fake_pred = ref.detach().requires_grad_(True)
    attention = torch.sigmoid(torch.randn(1, 3, 8, 8, requires_grad=True))

    class FakeStep65:
        """只用于 loss 张量测试，不依赖作者环境。"""

        @staticmethod
        def soft_iou_loss(z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            p = z.softmax(1)[:, 1:]
            y = F.one_hot(target, 4).permute(0, 3, 1, 2)[:, 1:].float()
            return (1 - (p * y).sum((0, 2, 3)).add(1) /
                    (p + y - p * y).sum((0, 2, 3)).add(1)).mean()

        @staticmethod
        def supervised_attention_loss(a: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            y = F.one_hot(target, 4).permute(0, 3, 1, 2)[:, 1:].float()
            return F.binary_cross_entropy(a, y)

    loss, _ = losses_for_refinement(fake_pred, ref, gt, attention, FakeStep65())
    loss.backward()
    assert torch.isfinite(loss) and fake_pred.grad is not None
    cm = torch.zeros(4, 4, dtype=torch.long)
    update_confusion(cm, gt, fake_pred.detach())
    m = metrics_from_confusion(cm)
    assert 0 <= m["bg_to_fg_false_positive_rate"] <= 1
    candidates = candidate_grid()
    baseline = {"foreground_miou": 0.04, "four_class_miou": 0.23,
                "bg_to_fg_false_positive_rate": 0.14}
    fake = {c.name: dict(baseline) for c in candidates}
    winner = candidates[-1]
    fake[winner.name] = {"foreground_miou": 0.05, "four_class_miou": 0.25,
                         "bg_to_fg_false_positive_rate": 0.17}
    selected, _, _ = select_locked_candidate(fake, candidates)
    assert selected == winner
    fake[winner.name]["bg_to_fg_false_positive_rate"] = 0.30
    selected, _, _ = select_locked_candidate(fake, candidates)
    assert selected.source == "original"
    print("[Step67] CPU 自检 PASS：门控、Loss 梯度、校准约束、四类混淆矩阵")


def cli() -> argparse.Namespace:
    """默认仅一条命令完成完整流程。"""
    p = argparse.ArgumentParser(description="Step67 一键训练→Train 校准→锁定 Mini Val A/B")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--calibration-frames", type=int, default=8)
    p.add_argument("--fresh-val-frames", type=int, default=12)
    p.add_argument("--lr", type=float, default=7e-5)
    p.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = p.parse_args()
    if not 16 <= args.steps <= 180:
        p.error("--steps 必须在 16～180")
    if not 4 <= args.calibration_frames <= 16:
        p.error("--calibration-frames 必须在 4～16")
    if not 4 <= args.fresh_val_frames <= 20:
        p.error("--fresh-val-frames 必须在 4～20")
    if not 0 < args.lr <= 0.001:
        p.error("--lr 必须在 (0,0.001]")
    return args


def main() -> None:
    """统一处理训练、Mini Train 校准、锁定 Mini Val 与落盘。"""
    args = cli()
    if args.self_test:
        cpu_self_test()
        return
    # 重复执行不覆盖已有实验，避免把未确认的回退结果丢失。
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，拒绝覆盖：{output}")
    for path in (STEP65_SUMMARY, STEP66_SUMMARY, STEP65_WEIGHT, STEP65_BIAS,
                 STEP66_WEIGHT):
        if not path.is_file():
            raise FileNotFoundError(f"Step67 输入缺失：{path}")
    if not torch.cuda.is_available():
        raise RuntimeError("请先激活 .venv-author CUDA 环境")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    step65 = importlib.import_module("scripts.stage2.run_step65_literature_adapter")
    step66 = importlib.import_module("scripts.stage2.step66_background_safe_train")
    from scripts.stage2.run_step59_train import build_mini_dataset, capture_samples, sha256
    from scripts.stage2.run_step60_extended_ab import (
        IndexedDataset, choose_scene_balanced, cuda_sample, move_to_cpu, print_memory,
    )
    report65 = json.loads(STEP65_SUMMARY.read_text(encoding="utf-8"))
    report66 = json.loads(STEP66_SUMMARY.read_text(encoding="utf-8"))
    paths_and_hashes = (
        (step65.CHECKPOINT, report65["checkpoint_sha256"]),
        (step65.TEXT, report65["text_sha256"]),
        (STEP65_WEIGHT, report65["adapter_weights_sha256"]),
        (STEP65_BIAS, report65["bias_weights_sha256"]),
        (STEP66_WEIGHT, report66["step66_adapter_sha256"]),
    )
    for path, digest in paths_and_hashes:
        if sha256(path) != digest:
            raise RuntimeError(f"哈希校验失败：{path}")
    output.mkdir(parents=True, exist_ok=False) if not output.exists() else None
    (output / "step67_method.json").write_text(json.dumps({
        "seed": SEED, "steps": args.steps, "lr": args.lr,
        "candidates": [asdict(c) for c in candidate_grid()],
        "selection_rule": (
            "Mini Train calibration only; maximize four-class mIoU, foreground mIoU "
            ">= original, bg FP <= original + 5pp; else fallback original"
        ),
        "source_hashes": {str(p): digest for p, digest in paths_and_hashes},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    start = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    sys.path.insert(0, str(step65.AUTHOR))
    os.chdir(step65.AUTHOR)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    cfg = Config.fromfile(str(step65.AUTHOR / "projects/configs/bevformer/bevformer_small_seg.py"))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    cfg.data.workers_per_gpu = 0
    print("[Step67] 初始化冻结的作者 BEVFormer；时序 OFF", flush=True)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(step65.CHECKPOINT), map_location="cpu")
    model.requires_grad_(False).eval()
    wrapped = MMDataParallel(model.cuda(), device_ids=[0])
    head = wrapped.module.pts_bbox_head
    text = torch.from_numpy(np.load(step65.TEXT, allow_pickle=False)).float().cuda()
    assert text.shape == (3, 768) and bool(torch.isfinite(text).all())
    print_memory("作者权重加载")
    weights65 = safe_load(STEP65_WEIGHT)["state_dict"]
    weights66 = safe_load(STEP66_WEIGHT)["state_dict"]
    bias_state = safe_load(STEP65_BIAS)["state_dict"]

    model65 = step65.TextConditionedLogitAdapter().cuda()
    model65.load_state_dict(weights65, strict=True)
    model65.eval().requires_grad_(False)
    bias65 = step65.BiasCalibration().cuda()
    bias65.load_state_dict(bias_state, strict=True)
    bias65.eval().requires_grad_(False)
    model66 = step66.BackgroundSafeAdapter(step65.TextConditionedLogitAdapter().cuda())
    model66.load_state_dict(weights66, strict=True)
    model66.eval().requires_grad_(False)
    model67 = step65.TextConditionedLogitAdapter().cuda()
    model67.load_state_dict(weights65, strict=True)
    model67.train()
    optimizer = torch.optim.AdamW(model67.parameters(), lr=args.lr, weight_decay=1e-4)

    # ---- 1. Mini Train 参数更新。额外校准帧从未参与 Step65/66 的 Train 索引选取。 ----
    train_ds = build_mini_dataset(cfg, build_dataset, "train")
    train_indices = list(report66["mini_train_indices"])
    old_training = set(train_indices)
    available = [i for i in range(len(train_ds)) if i not in old_training]
    local_calib = choose_scene_balanced(
        [train_ds.data_infos[i] for i in available], args.calibration_frames, SEED + 1,
    )
    calib_indices = [available[i] for i in local_calib]
    assert not set(calib_indices) & old_training
    selected_indices = train_indices + calib_indices
    samples_ds = IndexedDataset(train_ds, selected_indices)
    captured = move_to_cpu(capture_samples(
        wrapped, head.transformer, head, samples_ds,
        list(range(len(samples_ds))), build_dataloader,
        cfg.data.nonshuffler_sampler, torch, "mini_train",
    ))
    train_samples = captured[:len(train_indices)]
    calibration_samples = captured[len(train_indices):]
    assert len(calibration_samples) == args.calibration_frames
    del train_ds, samples_ds, captured
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("Mini Train 与 Train Calibration 缓存完成")
    (output / "step67_train_calib_split.json").write_text(json.dumps({
        "seed": SEED, "train_indices": train_indices,
        "train_tokens": [s.token for s in train_samples],
        "calibration_indices": calib_indices,
        "calibration_tokens": [s.token for s in calibration_samples],
        "train_calib_disjoint": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rng = random.Random(SEED + 2)
    history: list[dict[str, Any]] = []
    print(f"[Step67] 阶段 1/3：Train={len(train_samples)} 校准={len(calibration_samples)} "
          f"Steps={args.steps}", flush=True)
    for step in range(args.steps):
        index = rng.randrange(len(train_samples))
        bev, gt, reference = cuda_sample(train_samples[index], torch)
        optimizer.zero_grad(set_to_none=True)
        logits, attention = model67(bev, reference, text, head.feat_cropper)
        loss, parts = losses_for_refinement(logits, reference, gt, attention, step65)
        if not torch.isfinite(loss):
            raise RuntimeError(f"第 {step + 1} 次 loss 非有限")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model67.parameters(), 1.0))
        if not np.isfinite(norm):
            raise RuntimeError("梯度非有限")
        optimizer.step()
        history.append({"step": step + 1, "mini_train_index": train_indices[index],
                        "loss": float(loss.detach()), "gradient_norm_before_clip": norm,
                        **parts})
        if (step + 1) % 16 == 0 or (step + 1) == args.steps:
            path = output / f"step67_interim_{step + 1:04d}.pt"
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model67.state_dict().items()},
                        "step": step + 1, "step65_parent_sha256": sha256(STEP65_WEIGHT)}, path)
            (output / "step67_train_history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if step == 0 or (step + 1) % 8 == 0 or step + 1 == args.steps:
            print(f"[Step67] {step+1:03d}/{args.steps} loss={float(loss.detach()):.4f} "
                  f"grad={norm:.3f} fg_recall_loss={parts['foreground_recall_loss']:.4f}",
                  flush=True)
        del bev, gt, reference, logits, attention, loss
    saved = output / "adapter_step67.pt"
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model67.state_dict().items()},
                "step65_parent_sha256": sha256(STEP65_WEIGHT), "step": args.steps}, saved)
    print("[Step67] 训练权重已保存", flush=True)
    del train_samples
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 2. 只用 Mini Train 留出的校准帧选推理残差强度及门控。 ----
    print("[Step67] 阶段 2/3：Mini Train 校准，禁止查看 Mini Val 指标", flush=True)
    grid = candidate_grid()
    calib_result = evaluate_calibration(calibration_samples, head, text, model65, model67, grid)
    chosen, all_scores, selection_reason = select_locked_candidate(calib_result, grid)
    lock = {
        "locked_before_loading_mini_val": True,
        "selected": asdict(chosen), "name": chosen.name,
        "reason": selection_reason,
        "mini_train_calibration_indices": calib_indices,
        "selection_grid_scores": all_scores,
        "baseline": calib_result["original"],
    }
    (output / "step67_selection_lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[Step67] 校准方案锁定：{chosen.name}；{selection_reason}", flush=True)
    del calibration_samples
    gc.collect()
    torch.cuda.empty_cache()

    # ---- 3. 锁定配置后才加载、统计 Mini Val 旧/新增 token。 ----
    print("[Step67] 阶段 3/3：锁定后评估 Mini Val，不再改候选", flush=True)
    val_ds = build_mini_dataset(cfg, build_dataset, "val")
    dev_indices = list(report66["split_indices"]["development_6"])
    seen_indices = list(report66["split_indices"]["step65_seen_12"])
    reserved_indices = list(report66["split_indices"]["reserved_12"])
    previous_ids = set(dev_indices + seen_indices + reserved_indices)
    assert len(previous_ids) == len(dev_indices) + len(seen_indices) + len(reserved_indices)
    remaining = [i for i in range(len(val_ds)) if i not in previous_ids]
    new_local = choose_scene_balanced(
        [val_ds.data_infos[i] for i in remaining], args.fresh_val_frames, SEED + 3,
    )
    new_indices = [remaining[i] for i in new_local]
    val_indices = dev_indices + seen_indices + reserved_indices + new_indices
    val_subset = IndexedDataset(val_ds, val_indices)
    val_samples = move_to_cpu(capture_samples(
        wrapped, head.transformer, head, val_subset,
        list(range(len(val_subset))), build_dataloader,
        cfg.data.nonshuffler_sampler, torch, "mini_val",
    ))
    assert not set(s.token for s in val_samples) & set(report66["mini_train_tokens"])
    assert [s.token for s in val_samples[:6]] == report65["dev_tokens"]
    assert [s.token for s in val_samples[6:18]] == report65["fresh_token_holdout_tokens"]
    assert [s.token for s in val_samples[18:30]] == report66["split_tokens"]["reserved_12"]
    (output / "step67_val_manifest.json").write_text(json.dumps({
        "previous_dev_6": dev_indices, "previous_seen_12": seen_indices,
        "step66_seen_12": reserved_indices, "new_mini_val_12": new_indices,
        "new_tokens": [s.token for s in val_samples[30:]],
        "note": "All from same mini_val; new tokens not a separate dataset.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del val_ds, val_subset
    gc.collect()
    torch.cuda.empty_cache()
    splits = {
        "development_6": val_samples[:6],
        "step65_seen_12": val_samples[6:18],
        "step66_seen_12": val_samples[18:30],
        "new_mini_val_12": val_samples[30:],
    }
    measured: dict[str, Any] = {}
    for name, samples in splits.items():
        print(f"[Step67] 验证 {name}，帧数={len(samples)}", flush=True)
        measured[name] = evaluate_validation(
            samples, head, text, model65, bias65, model66, model67, chosen,
        )
    # 只核对复现，绝不按下面的 Mini Val 结果重新挑选候选。
    expected = (
        ("development_6", report66["metrics"]["development_6"]["original"]),
        ("step65_seen_12", report66["metrics"]["step65_seen_12"]["original"]),
        ("step66_seen_12", report66["metrics"]["reserved_12"]["original"]),
    )
    for split, source in expected:
        baseline = measured[split]["original"]["foreground_miou"]
        if abs(baseline - source["foreground_miou"]) > 1e-6:
            raise RuntimeError(f"基线复现失败：{split} {baseline} != {source['foreground_miou']}")
    result = {
        "experiment": "step67_three_in_one_train_calibrate_locked_evaluation",
        "train_steps": args.steps,
        "train_indices": train_indices,
        "calibration_indices": calib_indices,
        "locked_candidate": {**asdict(chosen), "name": chosen.name},
        "selection_reason": selection_reason,
        "calibration": calib_result,
        "metrics": measured,
        "new_mini_val_indices": new_indices,
        "original_checkpoint_sha256": report65["checkpoint_sha256"],
        "adapter_step67_sha256": sha256(saved),
        "time_seconds": round(time.perf_counter() - start, 2),
        "protocol": "nuScenes mini; frozen BEVFormer+decoder; temporal OFF; argmax; train-only calibration",
        "limitations": [
            "Mini Val 包含已在 Step65/66 被观察的 token；new12 仍在同一迷你场景体系",
            "如选择 original 是符合预定安全规则的回退，不代表新模型有效提升",
            "正常文本和置乱/常量文本差别较小时不得宣称语言语义贡献",
            "原始分割 81 帧时序 6.38% 根因仍未闭环",
            "本代码借鉴 BLV、Tversky、LAVT 的思想，并非这些论文完整复现",
        ],
    }
    summary = output / "step67_ab_summary.json"
    summary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review = output / "step67_review.md"
    review.write_text(report_text(result), encoding="utf-8")
    for split, data in measured.items():
        print(f"\n[Step67] === {split} ===", flush=True)
        for name in ("original", "adapter_step65", "adapter_step66", "adapter_step67_raw",
                     "step67_locked", "step67_locked_text_shuffled",
                     "step67_locked_text_constant"):
            m = data[name]
            print(f"  {name:29s} FG={m['foreground_miou']:.4%} "
                  f"ALL={m['four_class_miou']:.4%} BG={m['background_iou']:.4%} "
                  f"BG_FP={m['bg_to_fg_false_positive_rate']:.4%}", flush=True)
    print("[Step67] JSON:", summary, flush=True)
    print("[Step67] 中文报告:", review, flush=True)
    print("[Step67] 权重:", saved, flush=True)
    print("[Step67] 注意：参数只由 Mini Train 校准；所有 Mini Val 结果仅报告，不参与选优。")


if __name__ == "__main__":
    main()

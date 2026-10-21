"""Step66：高置信背景保护 + 非对称 Tversky + 文本消融验证。

真实 BEVFormer/DistilBERT 权重冻结，在 Step65 已训练适配器基础上继续训练。
旧 6 帧开发集、Step65 曾查看的 12 帧和新保留的 Mini Val token 分开报告。
注意：新保留 token 仍来自相同 mini_val，非外部测试集，时序保持 OFF。
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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

ROOT = Path(__file__).resolve().parents[2]
STEP65_REPORT = ROOT / "outputs/stage2_step65/step65_ab_summary.json"
STEP65_WEIGHT = ROOT / "outputs/stage2_step65/adapter_step65.pt"
STEP65_BIAS = ROOT / "outputs/stage2_step65/bias_step65.pt"
STEP65_SCRIPT = ROOT / "scripts/stage2/run_step65_literature_adapter.py"
OUTPUT = ROOT / "outputs/stage2_step66"
LABELS = ("Background", "Divider", "Ped Crossing", "Boundary")
SEED = 20261021


def import_step65() -> Any:
    """仅复用已经验证过的 BEV 路径、结构和指标实现。"""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return importlib.import_module("scripts.stage2.run_step65_literature_adapter")


class BackgroundSafeAdapter(nn.Module):
    """基于原始模型背景置信度抑制残差，保留 Step65 的所有训练参数。"""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.background_guard_midpoint = 0.75
        self.background_guard_slope = 8.0

    def forward(
        self,
        bev: torch.Tensor,
        reference: torch.Tensor,
        prompts: torch.Tensor,
        cropper: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """保护原始高置信背景；仅用原始 logits 定义门控，推理不使用 GT。"""
        proposed, attention = self.base(bev, reference, prompts, cropper)
        orig_bg_prob = reference.detach().softmax(dim=1)[:, :1]
        # 原始 P(background) 越大，允许的残差幅度越小；没有硬阈值。
        guard = torch.sigmoid(
            self.background_guard_slope * (
                self.background_guard_midpoint - orig_bg_prob
            )
        )
        logits = reference.detach() + guard * (proposed - reference.detach())
        return logits, attention


def foreground_tversky(
    logits: torch.Tensor,
    gt: torch.Tensor,
    fp_weight: float = 0.75,
) -> torch.Tensor:
    """相较 FN，更惩罚将背景误报为三类前景（训练端）。"""
    pred = logits.softmax(dim=1)[:, 1:]
    target = functional.one_hot(gt.long(), 4).permute(0, 3, 1, 2)[:, 1:].float()
    tp = (pred * target).sum(dim=(0, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(0, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(0, 2, 3))
    return (1 - (tp + 1) / (tp + fp_weight * fp + (1 - fp_weight) * fn + 1)).mean()


def background_guard_loss(
    logits: torch.Tensor,
    reference: torch.Tensor,
    gt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """通过真背景 CE 与可信教师背景约束，压制背景到前景的误检。"""
    bg_mask = gt == 0
    if bg_mask.any():
        background_ce = -functional.log_softmax(logits, dim=1)[:, 0][bg_mask].mean()
    else:
        background_ce = logits.sum() * 0
    teacher = reference.detach().softmax(dim=1)[:, 0]
    student = logits.softmax(dim=1)[:, 0]
    confident_background = (teacher > 0.7) & bg_mask
    if confident_background.any():
        decrease = functional.relu(teacher - student).square()
        preserve = decrease[confident_background].mean()
    else:
        preserve = logits.sum() * 0
    return background_ce, preserve


def prompt_contrastive_loss(attention: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """训练时用类别 GT 约束三条文本的相对排序，仅在前景像素监督。"""
    mask = (gt > 0) & (gt < 4)
    if not mask.any():
        return attention.sum() * 0
    scores = torch.logit(attention.clamp(1e-5, 1 - 1e-5))
    targets = (gt.long() - 1).clamp(min=0, max=2)
    per_pixel = functional.cross_entropy(scores, targets, reduction="none")
    return per_pixel[mask].mean()


def step66_loss(
    logits: torch.Tensor,
    reference: torch.Tensor,
    gt: torch.Tensor,
    attention: torch.Tensor,
    step65: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """保持分割前景监督，同时显式约束背景与语义配对。"""
    # 与 Step65 不同，提高背景 CE 相对重要性。
    class_weights = logits.new_tensor([0.85, 1.05, 1.20, 1.05])
    ce = functional.cross_entropy(logits, gt.long(), weight=class_weights)
    tversky = foreground_tversky(logits, gt)
    bg_ce, preserve = background_guard_loss(logits, reference, gt)
    att = step65.supervised_attention_loss(attention, gt)
    contrast = prompt_contrastive_loss(attention, gt)
    residual = (logits - reference.detach()).square().mean()
    loss = (
        0.45 * ce + 0.25 * tversky + 0.18 * bg_ce + 0.20 * preserve
        + 0.12 * att + 0.04 * contrast + 0.005 * residual
    )
    components = {
        "weighted_ce": ce,
        "foreground_tversky": tversky,
        "background_ce": bg_ce,
        "background_preservation": preserve,
        "attention_bce_dice": att,
        "text_contrastive": contrast,
        "logit_residual_l2": residual,
    }
    return loss, {key: float(value.detach()) for key, value in components.items()}


def compute_diagnostics(metrics: dict[str, Any]) -> dict[str, Any]:
    """将四类混淆矩阵转换为包含背景误检的可审计指标。"""
    m = np.asarray(metrics["confusion_gt_rows_pred_cols"], dtype=np.int64)
    gt_background = int(m[0].sum())
    bg_fp = int(m[0, 1:].sum())
    class_ious = metrics["per_class_iou"]
    all_iou = [class_ious[name] for name in LABELS]
    metrics["four_class_miou"] = (
        float(np.mean(all_iou)) if all(x is not None for x in all_iou) else None
    )
    metrics["background_to_foreground_fp_rate"] = (
        bg_fp / gt_background if gt_background else None
    )
    metrics["predicted_foreground_ratio"] = (
        int(m[:, 1:].sum()) / int(m.sum()) if m.sum() else None
    )
    metrics["ground_truth_foreground_ratio"] = (
        int(m[1:, :].sum()) / int(m.sum()) if m.sum() else None
    )
    return metrics


def evaluate(
    samples: list[Any],
    head: Any,
    text: torch.Tensor,
    step65_adapter: nn.Module,
    step65_bias: nn.Module,
    step66_init: nn.Module,
    step66_adapter: nn.Module,
    step65: Any,
) -> dict[str, Any]:
    """同批独立帧评估 7 条路径，不允许查询 GT 调整推理参数。"""
    from scripts.stage2.run_step60_extended_ab import cuda_sample

    names = (
        "original", "bias_step65", "adapter_step65", "step65_text_shuffled",
        "step66_guard_only", "adapter_step66", "step66_text_shuffled",
    )
    confusion = {name: torch.zeros(4, 4, dtype=torch.int64) for name in names}
    loss_values: dict[str, list[float]] = {name: [] for name in names}
    shuffled_prompts = text[[2, 0, 1]]
    for module in (step65_adapter, step65_bias, step66_init, step66_adapter):
        module.eval()
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            bev, gt, ref = cuda_sample(sample, torch)
            old_logits, _ = step65_adapter(bev, ref, text, head.feat_cropper)
            old_shuffled, _ = step65_adapter(bev, ref, shuffled_prompts, head.feat_cropper)
            guard_only, _ = step66_init(bev, ref, text, head.feat_cropper)
            guarded, _ = step66_adapter(bev, ref, text, head.feat_cropper)
            guarded_shuffled, _ = step66_adapter(
                bev, ref, shuffled_prompts, head.feat_cropper
            )
            outputs = {
                "original": ref,
                "bias_step65": step65_bias(ref),
                "adapter_step65": old_logits,
                "step65_text_shuffled": old_shuffled,
                "step66_guard_only": guard_only,
                "adapter_step66": guarded,
                "step66_text_shuffled": guarded_shuffled,
            }
            for name, prediction in outputs.items():
                step65.confusion_add(confusion[name], gt, prediction)
                loss_values[name].append(float(head.loss_seg(prediction.float(), gt)))
            if (i + 1) % 6 == 0 or i + 1 == len(samples):
                print(f"[Step66] A/B {i+1}/{len(samples)} 帧", flush=True)
            del bev, gt, ref, old_logits, old_shuffled, guarded, guarded_shuffled
    return {
        name: compute_diagnostics({
            **step65.confusion_metrics(confusion[name]),
            "mean_author_seg_loss": float(np.mean(loss_values[name])),
            "frames": len(samples),
        }) for name in names
    }


def build_report_text(result: dict[str, Any]) -> str:
    """输出中文报告；包含背景误检、四类平均和文本置乱对照。"""
    lines = [
        "# Step66：背景保护与语言对照实验", "",
        "冻结作者 BEVFormer、Decoder、DistilBERT；在 Step65 权重上继续训练。",
        "采用原始四类 argmax，不对 GT 做推理时处理，不根据验证集调阈值。", "",
    ]
    for split, group in result["metrics"].items():
        lines += [f"## {split}", "",
                  "| 分支 | 前景 mIoU | Background IoU | 四类 mIoU | BG→前景误检率 |",
                  "|---|---:|---:|---:|---:|"]
        for name, metrics in group.items():
            lines.append(
                f"| {name} | {metrics['foreground_miou']:.4%} | "
                f"{metrics['per_class_iou']['Background']:.4%} | "
                f"{metrics['four_class_miou']:.4%} | "
                f"{metrics['background_to_foreground_fp_rate']:.4%} |"
            )
        lines.append("")
    lines += [
        "## 解释边界", "",
        "- `development_6` 是反复观察过的开发集，不属于独立盲测。",
        "- `step65_seen_12` 在 Step65 已经查看，不能用作盲测。",
        "- `reserved_12` 是本轮前保留的其他 Mini Val token，仍属同一 mini_val 场景体系。",
        "- 只记录固定单帧、时序关闭协议，不能与作者 81 帧时序评估直接比较。",
        "- `step66_guard_only` 可区分推理期背景门控和继续训练的收益。",
        "- `step66_text_shuffled` 若不明显差于正常语言输入，则语言贡献仍未证实。",
        "- 不能仅凭前景 mIoU 提升宣称模型整体增强，需联合 Background IoU / 四类 mIoU。",
        "- Step65 与 Step66 均依赖尚未完全闭环的地图数据和权重评估协议。",
        "",
        "## 参考思路（非论文完整复现）", "",
        "- Learning without Forgetting (ECCV 2016): https://arxiv.org/abs/1606.09282",
        "- Generalised Dice (2017): https://arxiv.org/abs/1707.03237",
        "- Unified Focal loss / Tversky (2021): https://arxiv.org/abs/2102.04525",
        "- Boundary loss (2019): https://proceedings.mlr.press/v102/kervadec19a.html",
        "",
    ]
    return "\n".join(lines)


def self_test() -> None:
    """CPU 测试：尺寸、保留背景的门控方向、两类损失反传、指标。"""
    torch.manual_seed(12)
    step65 = import_step65()

    class FakeCropper:
        """只在 CPU 自测用于张量检查，不代表真实标定。"""

        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            return functional.interpolate(
                tensor, size=(200, 400), mode="bilinear", align_corners=True
            )

    base = step65.TextConditionedLogitAdapter(channels=16)
    guarded = BackgroundSafeAdapter(base)
    bev = torch.randn(1, 256, 150, 150)
    reference = torch.randn(1, 4, 200, 400)
    prompts = torch.randn(3, 768)
    gt = torch.randint(0, 4, (1, 200, 400))
    logits, att = guarded(bev, reference, prompts, FakeCropper())
    assert torch.equal(logits, reference), "零初始化的背景门控不能改变原始输出"
    assert att.shape == (1, 3, 200, 400)
    # 让非零残差通过门控，不依赖随机初始化能否产生差异。
    with torch.no_grad():
        base.residual.bias.fill_(0.5)
    high = torch.full((1, 4, 200, 400), -3.0)
    high[:, 0] = 6.0
    low = torch.full((1, 4, 200, 400), 0.0)
    high_out, _ = guarded(bev, high, prompts, FakeCropper())
    low_out, _ = guarded(bev, low, prompts, FakeCropper())
    assert (high_out - high).abs().mean() < (low_out - low).abs().mean()
    logits, att = guarded(bev, reference, prompts, FakeCropper())
    loss, parts = step66_loss(logits, reference, gt, att, step65)
    assert len(parts) == 7 and torch.isfinite(loss).item()
    loss.backward()
    grads = [p.grad for p in guarded.parameters() if p.grad is not None]
    assert grads and any(float(g.abs().sum()) > 0 for g in grads)
    confusion = torch.zeros(4, 4, dtype=torch.int64)
    step65.confusion_add(confusion, gt, logits)
    metrics = compute_diagnostics(step65.confusion_metrics(confusion))
    assert 0 <= metrics["background_to_foreground_fp_rate"] <= 1
    print("[Step66] CPU 自测 PASS：零初始化、背景门控、Loss 梯度、四类 IoU")


def parse_args() -> argparse.Namespace:
    """限制 GPU 与 WSL 内存规模；不允许改写既有 Step65 结果。"""
    parser = argparse.ArgumentParser(description="Step66 背景保护 + 非对称损失实验")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--train-frames", type=int, default=32)
    parser.add_argument("--reserve-frames", type=int, default=12)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    args = parser.parse_args()
    if not 24 <= args.train_frames <= 40:
        parser.error("--train-frames 必须为 24～40")
    if not 4 <= args.reserve_frames <= 24:
        parser.error("--reserve-frames 必须为 4～24")
    if not 8 <= args.steps <= 240:
        parser.error("--steps 必须为 8～240")
    if not 0 < args.lr <= 0.002:
        parser.error("--lr 必须在 (0, 0.002]")
    return args


def main() -> None:
    """继续训练 Step65 适配器，以冻结数据协议做原始模型对照。"""
    args = parse_args()
    if args.self_test:
        self_test()
        return
    step65 = import_step65()
    from scripts.stage2.run_step59_train import build_mini_dataset, capture_samples, sha256
    from scripts.stage2.run_step60_extended_ab import (
        IndexedDataset,
        choose_scene_balanced,
        cuda_sample,
        move_to_cpu,
        print_memory,
    )

    for path in (STEP65_REPORT, STEP65_WEIGHT, STEP65_BIAS, STEP65_SCRIPT):
        if not path.is_file():
            raise FileNotFoundError(f"必要输入缺失：{path}")
    report65 = json.loads(STEP65_REPORT.read_text(encoding="utf-8"))
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Step66 输出目录非空，拒绝覆盖：{output}")
    author = step65.AUTHOR.expanduser().resolve()
    ckpt = step65.CHECKPOINT
    text_path = step65.TEXT
    if report65["checkpoint_sha256"] != sha256(ckpt):
        raise RuntimeError("Step65 原始 checkpoint SHA256 不匹配")
    if report65["text_sha256"] != sha256(text_path):
        raise RuntimeError("Step65 三类语言向量 SHA256 不匹配")
    if report65["adapter_weights_sha256"] != sha256(STEP65_WEIGHT):
        raise RuntimeError("Step65 adapter 权重 SHA256 不匹配")
    if report65["bias_weights_sha256"] != sha256(STEP65_BIAS):
        raise RuntimeError("Step65 bias 权重 SHA256 不匹配")
    if not torch.cuda.is_available():
        raise RuntimeError("请切换到作者 CUDA 环境")

    sys.path.insert(0, str(author))
    os.chdir(author)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    torch.manual_seed(SEED + 66)
    np.random.seed(SEED + 66)
    rng = random.Random(SEED + 66)
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    text = torch.from_numpy(np.load(text_path, allow_pickle=False)).float().cuda()
    if text.shape != (3, 768) or not torch.isfinite(text).all():
        raise RuntimeError("三条语言向量必须是有效的 (3,768)")
    cfg = Config.fromfile(str(author / "projects/configs/bevformer/bevformer_small_seg.py"))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    cfg.data.workers_per_gpu = 0
    print("[Step66] 构建冻结作者模型和分割头...", flush=True)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(ckpt), map_location="cpu")
    model.requires_grad_(False).eval()
    wrapped = MMDataParallel(model.cuda(), device_ids=[0])
    head = wrapped.module.pts_bbox_head
    print_memory("作者权重加载完成")

    # Step65 的所有训练 token 保留，新增样本从 Mini Train 独立选取。
    train_ds = build_mini_dataset(cfg, build_dataset, "train")
    previous = list(report65["train_indices"])
    rest = [i for i in range(len(train_ds)) if i not in set(previous)]
    extra_count = args.train_frames - len(previous)
    if extra_count < 0:
        raise ValueError("本实验不支持比 Step65 更少的训练帧")
    chosen_extra = (
        [rest[k] for k in choose_scene_balanced(
            [train_ds.data_infos[i] for i in rest], extra_count, SEED + 6600
        )] if extra_count else []
    )
    training_indices = sorted(previous + chosen_extra)
    train_sub = IndexedDataset(train_ds, training_indices)
    train = move_to_cpu(capture_samples(
        wrapped, head.transformer, head, train_sub,
        list(range(len(train_sub))), build_dataloader,
        cfg.data.nonshuffler_sampler, torch, "mini_train",
    ))
    del train_ds, train_sub
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("Step66 Train 特征提取完成")

    # 前 6 帧和 Step65 后 12 帧仅用于透明对照；另取未曾用过的 12 帧。
    val_ds = build_mini_dataset(cfg, build_dataset, "val")
    dev_ids = list(report65["dev_indices"])
    seen_ids = list(report65["fresh_token_holdout_indices"])
    excluded = set(dev_ids + seen_ids)
    if len(excluded) != len(dev_ids) + len(seen_ids):
        raise RuntimeError("Step65 开发集/旧 holdout 索引重叠")
    remaining = [i for i in range(len(val_ds)) if i not in excluded]
    reserve_ids = [remaining[k] for k in choose_scene_balanced(
        [val_ds.data_infos[i] for i in remaining],
        args.reserve_frames, SEED + 6601,
    )]
    val_ids = sorted(excluded | set(reserve_ids))
    val_sub = IndexedDataset(val_ds, val_ids)
    cached_val = move_to_cpu(capture_samples(
        wrapped, head.transformer, head, val_sub,
        list(range(len(val_sub))), build_dataloader,
        cfg.data.nonshuffler_sampler, torch, "mini_val",
    ))
    mapped = {val_ids[k]: v for k, v in enumerate(cached_val)}
    dev = [mapped[i] for i in dev_ids]
    seen = [mapped[i] for i in seen_ids]
    reserved = [mapped[i] for i in reserve_ids]
    if [s.token for s in dev] != list(report65["dev_tokens"]):
        raise RuntimeError("六帧开发集 token 不一致")
    if [s.token for s in seen] != list(report65["fresh_token_holdout_tokens"]):
        raise RuntimeError("Step65 旧 holdout token 不一致")
    if set(s.token for s in train) & set(s.token for s in cached_val):
        raise RuntimeError("Mini Train/Val token 泄漏")
    del cached_val, val_sub, val_ds
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("验证集特征提取完成")
    # 先锁定验证划分，再进入训练；后续不会依照分数修改 reserve。
    split_manifest = {
        "seed": SEED,
        "dev_indices": dev_ids,
        "previous_holdout_indices": seen_ids,
        "reserved_indices": reserve_ids,
        "reserved_tokens": [s.token for s in reserved],
        "warning": "same mini_val scenes; no validation-driven selection",
    }
    (output / "step66_fixed_splits.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    old_base = step65.TextConditionedLogitAdapter().cuda()
    bias65 = step65.BiasCalibration().cuda()
    try:
        old_sd = torch.load(STEP65_WEIGHT, map_location="cpu", weights_only=True)
        bias_sd = torch.load(STEP65_BIAS, map_location="cpu", weights_only=True)
    except TypeError:  # 某些旧版本 Torch 不支持 weights_only。
        old_sd = torch.load(STEP65_WEIGHT, map_location="cpu")
        bias_sd = torch.load(STEP65_BIAS, map_location="cpu")
    old_base.load_state_dict(old_sd["state_dict"], strict=True)
    bias65.load_state_dict(bias_sd["state_dict"], strict=True)
    old_base.eval().requires_grad_(False)
    bias65.eval().requires_grad_(False)
    guard_initial = BackgroundSafeAdapter(step65.TextConditionedLogitAdapter().cuda())
    guard_initial.base.load_state_dict(old_sd["state_dict"], strict=True)
    guard_initial.eval().requires_grad_(False)
    guarded = BackgroundSafeAdapter(step65.TextConditionedLogitAdapter().cuda())
    guarded.base.load_state_dict(old_sd["state_dict"], strict=True)
    guarded.train()
    optim = torch.optim.AdamW(guarded.parameters(), lr=args.lr, weight_decay=1e-4)
    history: list[dict[str, Any]] = []
    print(
        f"[Step66] Train={len(train)} Dev={len(dev)} OldHoldout={len(seen)} "
        f"Reserved={len(reserved)} Steps={args.steps}", flush=True,
    )
    for i in range(args.steps):
        idx = rng.randrange(len(train))
        bev, gt, reference = cuda_sample(train[idx], torch)
        optim.zero_grad(set_to_none=True)
        logits, att = guarded(bev, reference, text, head.feat_cropper)
        loss, components = step66_loss(logits, reference, gt, att, step65)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Step66 第 {i + 1} 次 Loss 非有限")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(guarded.parameters(), 1.0))
        if not np.isfinite(grad_norm):
            raise RuntimeError("Step66 梯度不是有限数")
        optim.step()
        history.append({
            "step": i + 1,
            "index": training_indices[idx],
            "loss": float(loss.detach()),
            "grad_norm": grad_norm,
            **components,
        })
        if (i + 1) % 24 == 0:
            # 定期保存可恢复权重，不需要从头重复所有训练。
            interim = output / f"step66_interim_{i+1:04d}.pt"
            torch.save({
                "state_dict": {k: v.detach().cpu() for k, v in guarded.state_dict().items()},
                "step": i + 1,
                "parent_adapter_sha256": sha256(STEP65_WEIGHT),
            }, interim)
            (output / "step66_partial_history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        if i == 0 or (i + 1) % 10 == 0 or i == args.steps - 1:
            print(
                f"[Step66] {i+1:03d}/{args.steps} "
                f"loss={float(loss.detach()):.4f} "
                f"bg_ce={components['background_ce']:.4f} "
                f"grad={grad_norm:.3f}", flush=True,
            )
        del bev, gt, reference, logits, att, loss

    # 保存发生在大规模验证前：若评估出错仍能恢复权重。
    saved = output / "adapter_step66.pt"
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in guarded.state_dict().items()},
        "parent_adapter_sha256": sha256(STEP65_WEIGHT),
        "foreground_tversky_fp_weight": 0.75,
    }, saved)
    print("[Step66] 训练结束；权重已落盘，开始严格同条件 A/B。", flush=True)
    split_info = {
        "development_6": (dev, dev_ids),
        "step65_seen_12": (seen, seen_ids),
        "reserved_12": (reserved, reserve_ids),
    }
    measured: dict[str, Any] = {}
    for split, (samples, _) in split_info.items():
        print(f"[Step66] 正在评估 {split}: {len(samples)} 帧", flush=True)
        measured[split] = evaluate(
            samples, head, text, old_base, bias65, guard_initial, guarded, step65
        )
    old_baseline = float(report65["metrics"]["development_6"]["original"]["foreground_miou"])
    cur_baseline = float(measured["development_6"]["original"]["foreground_miou"])
    if abs(old_baseline - cur_baseline) > 1e-6:
        raise RuntimeError(f"开发集基线不一致 {old_baseline} != {cur_baseline}")
    old_holdout = float(report65["metrics"]["fresh_tokens"]["original"]["foreground_miou"])
    cur_holdout = float(measured["step65_seen_12"]["original"]["foreground_miou"])
    if abs(old_holdout - cur_holdout) > 1e-6:
        raise RuntimeError(f"旧 12 帧基线不一致 {old_holdout} != {cur_holdout}")
    metadata = {
        "experiment": "step66_confident_background_guard_asymmetric_tversky",
        "checkpoint_sha256": sha256(ckpt),
        "text_sha256": sha256(text_path),
        "step65_adapter_sha256": sha256(STEP65_WEIGHT),
        "step66_adapter_sha256": sha256(saved),
        "mini_train_indices": training_indices,
        "mini_train_tokens": [s.token for s in train],
        "split_indices": {name: ids for name, (_, ids) in split_info.items()},
        "split_tokens": {name: [s.token for s in samples] for name, (samples, _) in split_info.items()},
        "steps": args.steps,
        "lr": args.lr,
        "temporal_mode": "OFF: isolated frames",
        "replay_verified": True,
        "history": history,
        "metrics": measured,
        "training_loss": (
            "0.45*weighted_CE+0.25*foreground_tversky_fp0.75+"
            "0.18*gt_background_CE+0.20*teacher_bg_preserve+"
            "0.12*class_attention+0.04*prompt_contrast+0.005*residual_L2"
        ),
        "inference_bg_guard": "sigmoid(8*(0.75-original_background_probability))",
        "checkpoint_selection": "final step only; no val threshold tuning",
        "limitations": [
            "同一 Mini Val 场景，reserved_token 仅为未被早期试验查看的帧，不是独立数据集",
            "训练时使用真实 Mini Train GT；推理使用冻结作者输出，不读取 GT",
            "只做独立帧推理，不能与 81 帧时序官方指标直接比较",
            "GT / Map 物理坐标协议与作者发布权重仍可能存在未闭环差异",
            "类别文本模板不是开放词表 DriveLM 自然语言定位监督",
            "若文本置乱性能接近正常，不能认定分割提升来自语言语义",
            "只报告预先设定的全部分支与固定 argmax，不按验证集挑 checkpoint",
        ],
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }
    summary = output / "step66_ab_summary.json"
    summary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review = output / "step66_review.md"
    review.write_text(build_report_text(metadata), encoding="utf-8")
    for split, group in measured.items():
        print(f"\n[Step66] {split}", flush=True)
        for name, value in group.items():
            print(
                f"  {name:23s} "
                f"FG={value['foreground_miou']:.4%} "
                f"BG={value['per_class_iou']['Background']:.4%} "
                f"ALL={value['four_class_miou']:.4%} "
                f"BG_FP={value['background_to_foreground_fp_rate']:.4%}",
                flush=True,
            )
    print("[Step66] JSON:", summary, flush=True)
    print("[Step66] 中文报告:", review, flush=True)
    print("[Step66] 权重:", saved, flush=True)


if __name__ == "__main__":
    main()

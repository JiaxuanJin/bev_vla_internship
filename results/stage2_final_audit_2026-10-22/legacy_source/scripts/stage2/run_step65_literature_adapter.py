"""Step65：参考轻量适配/零初始化残差/IoU 优化的语言条件分割实验。

冻结作者 BEVFormer、分割 Decoder、DistilBERT，仅使用真实 Mini Train
缓存 BEV 与地图 GT 训练新增 logit adapter，并与四类 bias-only 校准对照。
所有评估使用不经阈值调整的四类 argmax mIoU，并单独报告开发验证帧和
未用于前期 Step61~64 决策的新 token 验证帧。

借鉴思想：ControlNet zero convolution、AdaptFormer PEFT、Focal Loss、
Lovasz-Softmax 对 IoU 优化的启发；这里不是那些论文实现的完整复现。
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
AUTHOR = Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg"
CHECKPOINT = Path("/mnt/g/nus/bevformer_small_seg.pth")
TEXT = ROOT / "outputs/stage2_step61/class_prompts.npy"
STEP61 = ROOT / "outputs/stage2_step61/step61_supervised_summary.json"
OUTPUT = ROOT / "outputs/stage2_step65"
NAMES = ("Background", "Divider", "Ped Crossing", "Boundary")
PREVIOUS_MIOU = 0.04225996280498306


class BiasCalibration(nn.Module):
    """四个可训练的全局类别偏置：区分后处理校准和语言模块收益。"""

    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, 4, 1, 1))

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        """对现有 logits 进行恒等初始化的类别偏置校准。"""
        return reference + self.bias


class TextConditionedLogitAdapter(nn.Module):
    """冻结 BEV 特征上的轻量语言条件 logit 残差校正。

    先在 150x150 网格压缩通道，再通过作者自身裁剪路径到 200x400。
    使用三条固定类别模板 DistilBERT 向量形成三个目标区域图，
    零初始化卷积确保未经训练的四类输出与原始分割严格一致。
    """

    def __init__(self, channels: int = 32, gain: float = 0.8) -> None:
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(256, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.text_projection = nn.Linear(768, channels, bias=False)
        self.attention_bias = nn.Parameter(torch.zeros(3))
        self.residual = nn.Conv2d(channels, 4, kernel_size=1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.class_bias = nn.Parameter(torch.zeros(1, 4, 1, 1))
        self.gain = gain

    def forward(
        self,
        bev: torch.Tensor,
        reference: torch.Tensor,
        prompts: torch.Tensor,
        cropper: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """输出四类分割 logits 与三条指令对应的 Attention 概率。"""
        if bev.shape != (1, 256, 150, 150):
            raise ValueError(f"BEV 张量尺寸异常：{tuple(bev.shape)}")
        if reference.shape != (1, 4, 200, 400):
            raise ValueError(f"作者分割 logits 尺寸异常：{tuple(reference.shape)}")
        if prompts.shape != (3, 768):
            raise ValueError(f"类别文本特征异常：{tuple(prompts.shape)}")
        visual = self.visual(bev.float())
        visual_map = cropper(torch.flip(visual.transpose(2, 3), dims=[2]))
        if visual_map.shape != (1, visual.shape[1], 200, 400):
            raise RuntimeError("特征未通过作者分割采样协议投影到地图网格")
        q = functional.normalize(self.text_projection(prompts.float()), dim=-1)
        v = functional.normalize(visual_map, dim=1)
        scores = 5.0 * torch.einsum("bchw,kc->bkhw", v, q)
        attention = torch.sigmoid(scores + self.attention_bias[None, :, None, None])
        raw_delta = self.residual(visual_map)
        # 对前三类残差分别使用其文字注意力加权；背景保持独立校正。
        gated = torch.cat(
            (raw_delta[:, :1], raw_delta[:, 1:] * (0.5 + attention)), dim=1
        )
        logits = reference.detach() + self.gain * gated + self.class_bias
        return logits, attention


def soft_iou_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """前景三类的 soft-IoU 代理损失，直接对齐主要评价指标。"""
    probs = logits.softmax(dim=1)[:, 1:]
    targets = functional.one_hot(gt.long(), 4).permute(0, 3, 1, 2)[:, 1:].float()
    intersection = (probs * targets).sum(dim=(0, 2, 3))
    union = (probs + targets - probs * targets).sum(dim=(0, 2, 3))
    return (1.0 - (intersection + 1.0) / (union + 1.0)).mean()


def focal_ce_loss(logits: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Focal CE 缓解背景大量易分类像素挤占学习信号的问题。"""
    weights = logits.new_tensor([0.30, 1.20, 1.70, 1.15])
    ce = functional.cross_entropy(logits, gt.long(), reduction="none")
    pt = torch.exp(-ce)
    pix_weight = weights[gt.long()]
    return ((1 - pt) ** 1.5 * pix_weight * ce).mean() / pix_weight.mean()


def supervised_attention_loss(
    attention: torch.Tensor, gt: torch.Tensor
) -> torch.Tensor:
    """三类文字注意力与各类 GT 直接配对，不对验证 GT 做形态学修改。"""
    y = functional.one_hot(gt.long(), 4).permute(0, 3, 1, 2)[:, 1:].float()
    a = attention.clamp(1e-6, 1 - 1e-6)
    n_pos = y.sum(dim=(0, 2, 3))
    n_neg = y.shape[0] * y.shape[-2] * y.shape[-1] - n_pos
    w = (n_neg / n_pos.clamp_min(1)).clamp(max=12)
    loss_bce = -(
        w[None, :, None, None] * y * torch.log(a)
        + (1 - y) * torch.log1p(-a)
    ).mean()
    intersection = (a * y).sum(dim=(0, 2, 3))
    dice = 1 - (2 * intersection + 1) / (
        a.sum(dim=(0, 2, 3)) + n_pos + 1
    )
    return 0.5 * loss_bce + 0.5 * dice.mean()


def train_loss(
    logits: torch.Tensor,
    reference: torch.Tensor,
    gt: torch.Tensor,
    attention: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """监督损失仅作用于 Mini Train；不从 Mini Val 反传梯度。"""
    focal = focal_ce_loss(logits, gt)
    iou = soft_iou_loss(logits, gt)
    # 轻微输出约束，防止小训练集把冻住的原模型校正得过激。
    residual_penalty = (logits - reference).square().mean()
    att = supervised_attention_loss(attention, gt) if attention is not None else None
    loss = 0.45 * focal + 0.55 * iou + 0.002 * residual_penalty
    if att is not None:
        loss = loss + 0.18 * att
    components = {
        "focal": float(focal.detach()),
        "soft_iou": float(iou.detach()),
        "attention": float(att.detach()) if att is not None else 0.0,
        "logit_regularizer": float(residual_penalty.detach()),
    }
    return loss, components


def confusion_add(matrix: torch.Tensor, gt: torch.Tensor, logits: torch.Tensor) -> None:
    """四类混淆矩阵按全部像素累计，非逐帧简单平均。"""
    true = gt.reshape(-1).long()
    predicted = logits.argmax(1).reshape(-1).long()
    matrix += torch.bincount(4 * true + predicted, minlength=16).reshape(4, 4).cpu()


def confusion_metrics(matrix: torch.Tensor) -> dict[str, Any]:
    """计算背景及三类前景 IoU，前景宏平均与原始协议一致。"""
    m = matrix.numpy()
    ious: dict[str, float | None] = {}
    for index, name in enumerate(NAMES):
        inter = int(m[index, index])
        union = int(m[index, :].sum() + m[:, index].sum() - inter)
        ious[name] = inter / union if union else None
    vals = [ious[name] for name in NAMES[1:] if ious[name] is not None]
    return {
        "foreground_miou": float(np.mean(vals)) if vals else None,
        "per_class_iou": ious,
        "confusion_gt_rows_pred_cols": m.tolist(),
    }


def evaluate_models(
    head: Any,
    samples: list[Any],
    text: torch.Tensor,
    bias: BiasCalibration,
    adapter: TextConditionedLogitAdapter,
    device: torch.device,
) -> dict[str, Any]:
    """同一批样本分别评估原始、Bias-only 与文字残差适配器。"""
    from scripts.stage2.run_step60_extended_ab import cuda_sample
    from scripts.stage2.run_step62_attention_diagnostic import precision_recall_auc

    matrices = {name: torch.zeros(4, 4, dtype=torch.long) for name in (
        "original", "bias_only", "language_adapter", "text_shuffled"
    )}
    losses: dict[str, list[float]] = {name: [] for name in matrices}
    attention_values = {k: [] for k in range(3)}
    attention_targets = {k: [] for k in range(3)}
    bias.eval()
    adapter.eval()
    with torch.no_grad():
        for sample in samples:
            bev, gt, reference = cuda_sample(sample, torch)
            pred_bias = bias(reference)
            pred_adapter, attention = adapter(bev, reference, text, head.feat_cropper)
            shuffled, _ = adapter(
                bev, reference, text[[2, 0, 1]], head.feat_cropper
            )
            for name, pred in (
                ("original", reference),
                ("bias_only", pred_bias),
                ("language_adapter", pred_adapter),
                ("text_shuffled", shuffled),
            ):
                confusion_add(matrices[name], gt, pred)
                losses[name].append(float(head.loss_seg(pred.float(), gt)))
            for index in range(3):
                attention_values[index].append(attention[0, index].float().cpu().numpy().ravel())
                attention_targets[index].append((gt[0] == index + 1).cpu().numpy().ravel())
            del bev, gt, reference, pred_bias, pred_adapter, shuffled, attention
    result: dict[str, Any] = {}
    for name in matrices:
        result[name] = {
            **confusion_metrics(matrices[name]),
            "mean_original_seg_loss": float(np.mean(losses[name])),
            "frames": len(samples),
        }
    for index, name in enumerate(NAMES[1:]):
        scores = np.concatenate(attention_values[index])
        y = np.concatenate(attention_targets[index])
        ap = precision_recall_auc(scores, y)
        result["language_adapter"].setdefault("attention_ap", {})[name] = {
            "ap": float(ap) if ap is not None else None,
            "gt_prevalence": float(y.mean()),
        }
    return result


def parse_args() -> argparse.Namespace:
    """设置资源上限与记录范围，避免错误覆盖既有实验。"""
    parser = argparse.ArgumentParser(description="Step65 零初始化语言残差对照")
    parser.add_argument("--author", type=Path, default=AUTHOR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--text", type=Path, default=TEXT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--train-frames", type=int, default=24)
    parser.add_argument("--holdout-frames", type=int, default=12)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    if not 8 <= args.train_frames <= 48:
        parser.error("--train-frames 必须在 8～48")
    if not 4 <= args.holdout_frames <= 30:
        parser.error("--holdout-frames 必须在 4～30")
    if not 16 <= args.steps <= 300:
        parser.error("--steps 必须在 16～300")
    if not 0 < args.lr <= 0.002:
        parser.error("--lr 必须在 (0,0.002]")
    return args


def self_test() -> None:
    """CPU 自检：严格零初始化、loss 梯度、类 IoU、三条语言提示。"""
    torch.manual_seed(7)

    class FakeCropper:
        """仅用于 CPU 尺寸/梯度检测，不代表真实作者地图采样。"""

        def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
            return functional.interpolate(
                tensor, size=(200, 400), mode="bilinear", align_corners=True
            )

    bev = torch.randn(1, 256, 150, 150)
    reference = torch.randn(1, 4, 200, 400)
    prompts = torch.randn(3, 768)
    gt = torch.randint(0, 4, (1, 200, 400))
    adapter = TextConditionedLogitAdapter(channels=16)
    logits, attention = adapter(bev, reference, prompts, FakeCropper())
    assert logits.shape == reference.shape
    assert attention.shape == (1, 3, 200, 400)
    assert torch.equal(logits, reference), "训练前必须零扰动原始 logits"
    loss, _ = train_loss(logits, reference, gt, attention)
    loss.backward()
    grads = [p.grad for p in adapter.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all().item() for g in grads)
    assert any(g.abs().sum().item() > 0 for g in grads)
    original = torch.zeros((4, 4), dtype=torch.long)
    confusion_add(original, gt, reference)
    assert confusion_metrics(original)["foreground_miou"] is not None
    bias = BiasCalibration()
    assert torch.equal(bias(reference), reference)
    print("[Step65] CPU 自测 PASS：零初始化回退、注意力、Loss、梯度、mIoU")


def main() -> None:
    """冻结作者模型后完成两个可训练分支和严格三组 A/B 评估。"""
    if "--self-test" in sys.argv:
        self_test()
        return
    args = parse_args()
    for path in (args.author, args.checkpoint, args.text, STEP61):
        if not path.is_file() and not path.is_dir():
            raise FileNotFoundError(f"缺少输入文件：{path}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"实验目录非空，禁止覆盖已有结果：{output}")
    source = args.author.expanduser().resolve()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(source))
    os.chdir(source)

    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from scripts.stage2.run_step59_train import build_mini_dataset, capture_samples, sha256
    from scripts.stage2.run_step60_extended_ab import (
        IndexedDataset, choose_scene_balanced, cuda_sample, move_to_cpu, print_memory,
    )

    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    if not torch.cuda.is_available():
        raise RuntimeError("Step65 需要作者环境 CUDA")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda")
    ref = json.loads(STEP61.read_text(encoding="utf-8"))
    if ref.get("checkpoint_sha256") != sha256(args.checkpoint):
        raise RuntimeError("作者 checkpoint 与 Step61 SHA256 不一致")
    if ref.get("text_sha256") != sha256(args.text):
        raise RuntimeError("三类文本向量与 Step61 SHA256 不一致")
    text = torch.tensor(np.load(args.text, allow_pickle=False), device=device).float()
    if text.shape != (3, 768) or not torch.isfinite(text).all():
        raise RuntimeError("文本向量必须是 (3,768) 且数值有限")
    cfg = Config.fromfile(str(source / "projects/configs/bevformer/bevformer_small_seg.py"))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    cfg.data.workers_per_gpu = 0
    print("[Step65] 加载并冻结作者真实 BEVFormer...", flush=True)
    started = time.perf_counter()
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.requires_grad_(False)
    model.eval()
    wrapped = MMDataParallel(model.cuda(), device_ids=[0])
    head = wrapped.module.pts_bbox_head
    print_memory("作者权重载入完成")

    # Mini Train 逐步读取后释放 Dataset，避免此前 WSL OOM。
    train_dataset = build_mini_dataset(cfg, build_dataset, "train")
    train_indices = choose_scene_balanced(
        train_dataset.data_infos, args.train_frames, args.seed
    )
    train_source = IndexedDataset(train_dataset, train_indices)
    train = move_to_cpu(capture_samples(
        wrapped, head.transformer, head, train_source,
        list(range(len(train_source))), build_dataloader,
        cfg.data.nonshuffler_sampler, torch, "mini_train",
    ))
    del train_dataset, train_source
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("Mini Train 特征完成")

    val_dataset = build_mini_dataset(cfg, build_dataset, "val")
    dev_indices = ref.get("val_indices")
    if not isinstance(dev_indices, list) or len(dev_indices) != 6:
        raise RuntimeError("Step61 开发验证集缺少固定六帧索引")
    other_indices = [i for i in range(len(val_dataset)) if i not in set(dev_indices)]
    available_infos = [val_dataset.data_infos[i] for i in other_indices]
    local_holdout_indices = choose_scene_balanced(
        available_infos, args.holdout_frames, args.seed + 6500
    )
    holdout_indices = [other_indices[i] for i in local_holdout_indices]
    if set(dev_indices) & set(holdout_indices):
        raise RuntimeError("新的 holdout token 与前 6 帧交叉")
    val_indices = sorted(set(dev_indices + holdout_indices))
    val_source = IndexedDataset(val_dataset, val_indices)
    cached_val = move_to_cpu(capture_samples(
        wrapped, head.transformer, head, val_source,
        list(range(len(val_source))), build_dataloader,
        cfg.data.nonshuffler_sampler, torch, "mini_val",
    ))
    mapping = {val_indices[i]: s for i, s in enumerate(cached_val)}
    dev = [mapping[i] for i in dev_indices]
    holdout = [mapping[i] for i in holdout_indices]
    if [s.token for s in dev] != ref["val_tokens"]:
        raise RuntimeError("原始六帧 Mini Val token 不一致")
    if set(s.token for s in train) & set(s.token for s in cached_val):
        raise RuntimeError("Train/Val token 泄漏")
    del cached_val, val_dataset, val_source
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("Mini Val 特征完成")

    bias = BiasCalibration().to(device)
    adapter = TextConditionedLogitAdapter().to(device)
    with torch.no_grad():
        for item in dev[:1]:
            bev, gt, reference = cuda_sample(item, torch)
            before, _ = adapter(bev, reference, text, head.feat_cropper)
            if not torch.equal(before, reference):
                raise RuntimeError("零初始化回退不等于作者输出")
    train_tokens = [s.token for s in train]
    if args.train_frames == 16 and train_tokens != ref.get("train_tokens"):
        raise RuntimeError("同条件训练 token 不一致")

    # 所有优化仅来自 Mini Train，验证样本不会参与梯度或 checkpoint 选择。
    optimizer_bias = torch.optim.AdamW(bias.parameters(), lr=args.lr, weight_decay=0)
    optimizer_adapter = torch.optim.AdamW(
        adapter.parameters(), lr=args.lr, weight_decay=1e-4
    )
    history: list[dict[str, Any]] = []
    print(
        f"[Step65] Train={len(train)} Dev={len(dev)} NewHoldout={len(holdout)} "
        f"Steps={args.steps} (two controlled branches)", flush=True,
    )
    for step in range(args.steps):
        choice = rng.randrange(len(train))
        bev, gt, reference = cuda_sample(train[choice], torch)
        # A: 全局 bias-only 校准组。
        optimizer_bias.zero_grad(set_to_none=True)
        corrected_bias = bias(reference)
        bias_loss, bias_parts = train_loss(corrected_bias, reference, gt)
        bias_loss.backward()
        optimizer_bias.step()
        # B: 零初始化语言条件残差适配器，注意力与分割联合监督。
        optimizer_adapter.zero_grad(set_to_none=True)
        logits, attention = adapter(bev, reference, text, head.feat_cropper)
        loss, parts = train_loss(logits, reference, gt, attention)
        if not torch.isfinite(loss):
            raise RuntimeError("Step65 出现 NaN/Inf")
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0))
        if not np.isfinite(norm):
            raise RuntimeError("Step65 梯度不合法")
        optimizer_adapter.step()
        history.append({
            "step": step + 1,
            "train_index": train_indices[choice],
            "bias_total_loss": round(float(bias_loss.item()), 6),
            "adapter_total_loss": round(float(loss.item()), 6),
            "adapter_grad_norm": round(norm, 6),
            **{f"adapter_{k}": round(v, 6) for k, v in parts.items()},
        })
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(
                f"[Step65] {step+1:03d}/{args.steps} "
                f"bias={bias_loss.item():.4f} adapter={loss.item():.4f} "
                f"grad={norm:.3f}", flush=True,
            )
        del bev, gt, reference, logits, attention, loss, bias_loss

    print("[Step65] 训练完成，分别计算开发验证和新 token 验证", flush=True)
    dev_eval = evaluate_models(head, dev, text, bias, adapter, device)
    holdout_eval = evaluate_models(head, holdout, text, bias, adapter, device)
    observed = dev_eval["original"]["foreground_miou"]
    if abs(observed - PREVIOUS_MIOU) > 1e-6:
        raise RuntimeError(f"六帧原始基线未复现：{observed:.8f} != {PREVIOUS_MIOU:.8f}")
    weights = {
        "bias": output / "bias_step65.pt",
        "adapter": output / "adapter_step65.pt",
    }
    torch.save({"state_dict": bias.cpu().state_dict()}, weights["bias"])
    torch.save({"state_dict": adapter.cpu().state_dict()}, weights["adapter"])
    report = {
        "experiment": "step65_zero_init_text_logit_residual_with_bias_control",
        "method": "ControlNet-inspired zero init residual + lightweight visual adapter + focal CE + soft IoU",
        "note": "Not a faithful reproduction of cited papers; architecture is project-specific",
        "data": "nuScenes v1.0-mini mini_train / mini_val, temporal OFF",
        "train_indices": train_indices,
        "train_tokens": train_tokens,
        "dev_indices": dev_indices,
        "dev_tokens": [s.token for s in dev],
        "fresh_token_holdout_indices": holdout_indices,
        "fresh_token_holdout_tokens": [s.token for s in holdout],
        "fresh_holdout_is_same_mini_val_not_independent_dataset": True,
        "steps_per_branch": args.steps,
        "train_frames": len(train),
        "checkpoint_sha256": sha256(args.checkpoint),
        "text_sha256": sha256(args.text),
        "bias_weights_sha256": sha256(weights["bias"]),
        "adapter_weights_sha256": sha256(weights["adapter"]),
        "metrics": {"development_6": dev_eval, "fresh_tokens": holdout_eval},
        "history": history,
        "weights": {k: str(p) for k, p in weights.items()},
        "training_objective": "0.45*class_weighted_focal + 0.55*soft_fg_IoU + 0.18*class_attention + 0.002*logit_reg",
        "validation_tuning": "no threshold or checkpoint chosen from dev / fresh token holdout",
        "limitations": [
            "官方 Mini Val 仅 81 帧，新增验证 token 与既有开发帧可能属于同一 scene",
            "只训练小 adapter，不等于 BEVFormer 作者完整 81 帧时序评估",
            "GT 由人工类别模板监督，不代表 DriveLM 自由文本定位",
            "如果只有 bias-only 有收益，不能归因到语言语义",
            "置乱三条文本提示后若分割无变化，也不能认定语言条件有效",
            "6 帧开发验证集已反复用于前期模型决策，不属于盲测",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (output / "step65_ab_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = ["# Step65：论文启发的零初始化语言分割适配器", "",
             "使用冻结作者 BEVFormer 输出、真实 Mini Train GT，单帧时序 OFF。",
             "新增 Bias-only 与 Text-shuffled 对照，避免将常数校准/视觉适配收益归因于语言模块。", ""]
    for group_name, group in (("development_6", dev_eval), ("fresh_tokens", holdout_eval)):
        lines += [f"## {group_name}（{group['original']['frames']} 帧）", "",
                  "| 分支 | Divider IoU | Ped Crossing IoU | Boundary IoU | 前景 mIoU |",
                  "|---|---:|---:|---:|---:|"]
        for name in ("original", "bias_only", "language_adapter", "text_shuffled"):
            m = group[name]
            row = [
                f"{m['per_class_iou'][c]*100:.4f}%"
                if m["per_class_iou"][c] is not None else "N/A"
                for c in NAMES[1:]
            ]
            lines.append(f"| {name} | {' | '.join(row)} | {m['foreground_miou']*100:.4f}% |")
        lines.append("")
    lines += [
        "## 结论边界", "",
        "- 输出始终为 4 类 argmax 分割 IoU，不用 GT 掩码指导推理或调阈值。",
        "- 新 token 验证仍属于同一 v1.0-mini，不是外部独立数据集。",
        "- 冻结的作者分割模型与现有 GT/坐标协议可能有未解决不一致。",
        "- 训练后结果若不能稳定超过原始模型，必须如实保留负结果。",
        "- 文本类别模板不等于自然驾驶语言理解；若 bias-only 提升更大不可宣称语言增强胜出。",
        "- 若 prompt 置乱效果与正常 prompt 近似，则不能宣称模型有效使用文本语义。",
        "",
        "## 参考文献与代码", "",
        "- ControlNet (ICCV 2023): https://github.com/lllyasviel/ControlNet",
        "- AdaptFormer (NeurIPS 2022): https://github.com/ShoufaChen/AdaptFormer",
        "- Lovász-Softmax (CVPR 2018): https://github.com/bermanmaxim/LovaszSoftmax",
        "- Focal Loss (ICCV 2017): https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html",
        "", "说明：Step65 仅借鉴适配与损失思想，未照搬完整开源实现。", "",
    ]
    (output / "step65_review.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n[Step65] 完成。开发集与新验证的 IoU 比较：")
    for group_name, group in (("development_6", dev_eval), ("fresh_tokens", holdout_eval)):
        print(f"  [{group_name}]")
        for name in ("original", "bias_only", "language_adapter", "text_shuffled"):
            print(f"    {name:16s}: {group[name]['foreground_miou']:.4%}")
    print("[Step65] 报告:", output / "step65_review.md", flush=True)
    print("[Step65] JSON:", output / "step65_ab_summary.json", flush=True)
    print("[Step65] 权重:", weights["adapter"], flush=True)


if __name__ == "__main__":
    main()

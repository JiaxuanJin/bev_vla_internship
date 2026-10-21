"""Step61：基于真实 nuScenes Map GT 的语言类别注意力监督训练。

使用已经验证过的 Step59/60 作者分割 Encoder + Decoder replay；
对注意力应用作者分割头自身的几何变换后，在 GT 200x400 栅格上监督。
冻结 BEVFormer、DistilBERT、Seg Decoder，仅更新 LanguageBEVFusion 的
文本投影与空间注意力参数；融合系数 alpha 暂时固定，避免退化为直接关闭模块。

注意：这里的语义监督是三个地图类别的人工文本模板，并非 DriveLM 自然语言
定位标注，不代表已学会开放词表或真实语义目标定位。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg"
CHECKPOINT = Path("/mnt/g/nus/bevformer_small_seg.pth")
TEXT = ROOT / "outputs/stage2_step61/class_prompts.npy"
OUTPUT = ROOT / "outputs/stage2_step61"
PREVIOUS = ROOT / "outputs/stage2_step60/step60_ab_summary.json"
NAMES = ("Background", "Divider", "Ped Crossing", "Boundary")


def parse_args() -> argparse.Namespace:
    """支持先做少量样本测试，再增加样本数量与更新次数。"""
    parser = argparse.ArgumentParser(description="Step61 类别文本→地图 GT 注意力监督")
    parser.add_argument("--author", type=Path, default=AUTHOR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--text", type=Path, default=TEXT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--previous", type=Path, default=PREVIOUS)
    parser.add_argument("--train-frames", type=int, default=16)
    parser.add_argument("--val-frames", type=int, default=6)
    parser.add_argument("--steps", type=int, default=72)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-att", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    if not 4 <= args.train_frames <= 48:
        parser.error("--train-frames 支持 4~48")
    if not 4 <= args.val_frames <= 16:
        parser.error("--val-frames 支持 4~16")
    if not 3 <= args.steps <= 300:
        parser.error("--steps 支持 3~300")
    if not 0 < args.lr <= 0.002:
        parser.error("--lr 必须在 (0,0.002] 范围内")
    if not 0 <= args.lambda_att <= 20:
        parser.error("--lambda-att 必须在 [0,20] 范围内")
    if not 0 < args.alpha <= 0.1:
        parser.error("--alpha 必须在 (0,0.1] 范围内")
    return args


def project_attention(head: Any, attention: Any, torch: Any) -> Any:
    """严格重用分割头的 BEV 旋转/翻转/裁剪协议，不直接拉伸 GT。"""
    if tuple(attention.shape) != (1, 1, 150, 150):
        raise ValueError(f"注意力必须为 (1,1,150,150)，得到 {tuple(attention.shape)}")
    # 与 run_step59_train.replay_decoder 完全一致：交换 H/W，后翻转 dim=2。
    spatial = torch.flip(attention.transpose(2, 3), dims=[2])
    projected = head.feat_cropper(spatial)
    if tuple(projected.shape) != (1, 1, 200, 400):
        raise RuntimeError(f"注意力的 GT 坐标投影异常：{tuple(projected.shape)}")
    if not torch.isfinite(projected).all():
        raise RuntimeError("投影注意力存在 NaN/Inf")
    return projected.clamp(min=1e-6, max=1 - 1e-6)


def supervision_loss(attention_map: Any, gt: Any, class_id: int, torch: Any) -> Any:
    """类别不平衡加权 BCE + Soft Dice；空类掩码不会假装存在目标。"""
    if class_id not in (1, 2, 3):
        raise ValueError("只能使用分割前景类 1/2/3")
    if (
        tuple(attention_map.shape) != (1, 1, 200, 400)
        or tuple(gt.shape) != (1, 200, 400)
    ):
        raise ValueError("注意力与 GT 的空间尺寸不一致")
    positive = (gt == class_id).unsqueeze(1).to(dtype=attention_map.dtype)
    n_pos = positive.sum()
    n_neg = positive.numel() - n_pos
    # 训练时类别正像素可能稀疏，最大权重固定封顶以稳定梯度。
    positive_weight = (n_neg / n_pos.clamp_min(1)).clamp(min=1.0, max=20.0)
    loss_bce = -(
        positive_weight * positive * torch.log(attention_map)
        + (1 - positive) * torch.log1p(-attention_map)
    ).mean()
    intersection = (attention_map * positive).sum()
    smooth = 1.0
    soft_dice_loss = 1 - (2 * intersection + smooth) / (
        attention_map.sum() + positive.sum() + smooth
    )
    # 正例为空的样本，鼓励输出 0；不人为构造不存在的目标。
    empty_loss = attention_map.mean()
    dice_component = torch.where(n_pos > 0, soft_dice_loss, empty_loss)
    return 0.5 * loss_bce + 0.5 * dice_component


def binary_counts(
    prediction: Any, truth: Any, torch: Any
) -> dict[str, float | int | None]:
    """固定 0.5 阈值的 Attention IoU，以及对阈值更稳定的 Soft Dice。"""
    p = prediction.detach().float().reshape(-1)
    t = truth.detach().bool().reshape(-1)
    binary = p >= 0.5
    tp = int((binary & t).sum().item())
    fp = int((binary & ~t).sum().item())
    fn = int((~binary & t).sum().item())
    union = tp + fp + fn
    n_pos = int(t.sum().item())
    smooth = 1.0
    soft_dice = float((2 * p[t].sum() + smooth) / (p.sum() + n_pos + smooth))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_positive_pixels": n_pos,
        "attention_iou_threshold_0_5": tp / union if union else None,
        "attention_soft_dice": soft_dice,
        "positive_minus_negative_mean": (
            float(p[t].mean() - p[~t].mean()) if n_pos and (~t).any() else None
        ),
    }


def add_confusion(confusion: Any, gt: Any, logits: Any, torch: Any) -> None:
    """累积四类混淆矩阵，防止平均逐帧 IoU 带来分母偏差。"""
    truth = gt.reshape(-1).long()
    label = logits.argmax(dim=1).reshape(-1).long()
    if truth.min().item() < 0 or truth.max().item() > 3:
        raise ValueError("GT 不是 0~3 的四类标签")
    confusion += torch.bincount(truth * 4 + label, minlength=16).reshape(4, 4).cpu()


def metric_confusion(confusion: Any) -> dict[str, Any]:
    """前景 mIoU 为三类按累计交并比计算的宏平均。"""
    matrix = confusion.numpy()
    ious = {}
    for k, name in enumerate(NAMES):
        tp = int(matrix[k, k])
        union = int(matrix[k, :].sum() + matrix[:, k].sum() - tp)
        ious[name] = tp / union if union else None
    fg = [ious[n] for n in NAMES[1:] if ious[n] is not None]
    return {
        "foreground_miou_argmax": float(np.mean(fg)) if fg else None,
        "per_class_iou": ious,
        "confusion_gt_rows_pred_columns": matrix.tolist(),
    }


def evaluate(
    fusion: Any, head: Any, samples: list[Any], text: Any, torch: Any, *,
    original: bool = False,
) -> dict[str, Any]:
    """按三个类别指令分别评估注意力定位与分割；原始分割只算一遍。"""
    from scripts.stage2.run_step59_train import replay_decoder
    from scripts.stage2.run_step60_extended_ab import cuda_sample

    fusion.eval()
    # 基线仅原始分割，三个指令不影响原始 logits。
    per_prompt = {}
    with torch.no_grad():
        for class_id, name in enumerate(NAMES[1:], 1):
            confusion = torch.zeros((4, 4), dtype=torch.int64)
            stats: list[dict[str, Any]] = []
            att_losses: list[float] = []
            seg_losses: list[float] = []
            diffs: list[float] = []
            for item in samples:
                bev, gt, reference = cuda_sample(item, torch)
                if original:
                    logits = reference
                else:
                    enhanced, att = fusion(
                        bev, text[class_id - 1 : class_id], enabled=True
                    )
                    projected = project_attention(head, att, torch)
                    mask = gt == class_id
                    stats.append(binary_counts(projected, mask.unsqueeze(1), torch))
                    att_losses.append(
                        float(supervision_loss(projected, gt, class_id, torch))
                    )
                    logits = replay_decoder(head, enhanced, torch).float()
                add_confusion(confusion, gt, logits, torch)
                seg_losses.append(float(head.loss_seg(logits.float(), gt)))
                diffs.append(float((logits - reference).abs().mean()))
                del bev, gt, reference, logits
            seg_metrics = metric_confusion(confusion)
            sub = {
                "segmentation": seg_metrics,
                "mean_seg_loss": float(np.mean(seg_losses)),
                "mean_abs_logits_vs_original": float(np.mean(diffs)),
            }
            if not original:
                # 微平均的注意力 IoU：缺少类别样本参与 FN/FP，不人为记作 perfect。
                tp = sum(int(s["tp"]) for s in stats)
                fp = sum(int(s["fp"]) for s in stats)
                fn = sum(int(s["fn"]) for s in stats)
                union = tp + fp + fn
                positive_frames = sum(s["gt_positive_pixels"] > 0 for s in stats)
                soft_dices = [
                    s["attention_soft_dice"]
                    for s in stats
                    if s["gt_positive_pixels"] > 0
                ]
                gaps = [
                    s["positive_minus_negative_mean"]
                    for s in stats
                    if s["positive_minus_negative_mean"] is not None
                ]
                sub["attention"] = {
                    "attention_iou_at_0_5": tp / union if union else None,
                    "soft_dice_positive_frames_mean": (
                        float(np.mean(soft_dices)) if soft_dices else None
                    ),
                    "positive_minus_negative_mean": (
                        float(np.mean(gaps)) if gaps else None
                    ),
                    "mean_attention_loss": float(np.mean(att_losses)),
                    "frames_with_class": positive_frames,
                    "gt_positive_pixels": sum(
                        int(s["gt_positive_pixels"]) for s in stats
                    ),
                    "counts": {"tp": tp, "fp": fp, "fn": fn},
                }
            per_prompt[name] = sub
    return {"per_prompt": per_prompt, "frame_count": len(samples)}


def fmt_pct(value: float | None) -> str:
    """人类可读的百分比格式。"""
    return "N/A" if value is None else f"{value * 100:.4f}%"


def self_test() -> None:
    """只检查注意力 GT 投影、监督损失、梯度与混淆矩阵，不加载作者模型。"""
    import torch
    import torch.nn.functional as functional

    class FakeHead:
        """仅用于测试尺寸/梯度；不是作者真实几何算子。"""

        def feat_cropper(self, feature: Any) -> Any:
            return functional.interpolate(
                feature, size=(200, 400), mode="bilinear", align_corners=True
            )

    torch.manual_seed(61)
    x = torch.randn(1, 1, 150, 150, requires_grad=True)
    att = torch.sigmoid(x)
    projected = project_attention(FakeHead(), att, torch)
    gt = torch.zeros((1, 200, 400), dtype=torch.long)
    gt[:, 30:55, 35:75] = 1
    gt[:, 70:84, 130:145] = 2
    gt[:, 150:180, 220:245] = 3
    losses = [supervision_loss(projected, gt, cls, torch) for cls in (1, 2, 3)]
    sum(losses).backward()
    if x.grad is None or not torch.isfinite(x.grad).all() or x.grad.abs().sum() <= 0:
        raise AssertionError("Step61 Attention Loss 梯度检查失败")
    stats = binary_counts(projected.detach(), (gt == 1).unsqueeze(1), torch)
    if stats["gt_positive_pixels"] != 1000:
        raise AssertionError("GT 类别面积不正确")
    assert tuple(projected.shape) == (1, 1, 200, 400)
    print("[Step61] CPU 自测 PASS：映射尺寸、类别标签、BCE+Dice、梯度", flush=True)
    print("[Step61] 注意：该自测不代表真实作者模型或实测准确率", flush=True)


def main() -> None:
    """以完全相同的留出样本进行训练前后 A/B 对照。"""
    if "--self-test" in sys.argv:
        self_test()
        return
    args = parse_args()
    source = args.author.expanduser().resolve()
    ckpt = args.checkpoint.expanduser().resolve()
    text_path = args.text.expanduser().resolve()
    config_path = source / "projects/configs/bevformer/bevformer_small_seg.py"
    for path in (source, ckpt, text_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f"必需路径不存在：{path}")
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(source))
    os.chdir(source)

    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from scripts.stage2.bev_fusion import LanguageBEVFusion
    from scripts.stage2.run_step59_train import (
        build_mini_dataset, capture_samples, replay_decoder, sha256,
    )
    from scripts.stage2.run_step60_extended_ab import (
        choose_scene_balanced, IndexedDataset, move_to_cpu, cuda_sample,
        print_memory,
    )

    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    if not torch.cuda.is_available():
        raise RuntimeError("Step61 作者模型需要 CUDA")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    text_arr = np.load(text_path, allow_pickle=False)
    if tuple(text_arr.shape) != (3, 768) or not np.isfinite(text_arr).all():
        raise ValueError(f"需要三个类别文本向量 (3,768)，实际 {text_arr.shape}")
    text = torch.from_numpy(text_arr).float().cuda()
    cfg = Config.fromfile(str(config_path))
    if (cfg.model.pts_bbox_head.bev_h, cfg.model.pts_bbox_head.bev_w) != (150, 150):
        raise ValueError("不支持非 150×150 的 BEV 网格")
    if not bool(cfg.model.pts_bbox_head.task.get("seg")):
        raise ValueError("作者配置没有分割头")
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    cfg.data.workers_per_gpu = 0

    print("[Step61] 构建原始冻结模型并加载 checkpoint", flush=True)
    started = time.perf_counter()
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(ckpt), map_location="cpu")
    model.requires_grad_(False)
    model.eval()
    model = MMDataParallel(model.cuda(), device_ids=[0])
    head = model.module.pts_bbox_head
    transformer = head.transformer
    print_memory("冻结模型就绪")

    print("[Step61] 顺序提取 Mini Train 特征+GT...", flush=True)
    train_dataset = build_mini_dataset(cfg, build_dataset, "train")
    train_scene_set = {str(i["scene_token"]) for i in train_dataset.data_infos}
    indices = choose_scene_balanced(
        train_dataset.data_infos, args.train_frames, args.seed
    )
    selected = IndexedDataset(train_dataset, indices)
    train = capture_samples(
        model, transformer, head, selected, list(range(len(selected))),
        build_dataloader, cfg.data.nonshuffler_sampler, torch, "mini_train"
    )
    train = move_to_cpu(train)
    del train_dataset, selected
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("Train 特征缓存完成")

    print("[Step61] 顺序提取 Mini Val 特征+GT...", flush=True)
    val_dataset = build_mini_dataset(cfg, build_dataset, "val")
    val_scene_set = {str(i["scene_token"]) for i in val_dataset.data_infos}
    if train_scene_set & val_scene_set:
        raise RuntimeError("训练/验证场景重叠")
    if args.val_frames == 6 and args.previous.is_file():
        previous = json.loads(args.previous.read_text(encoding="utf-8"))
        val_indices = previous.get("val_indices", [])
        if len(val_indices) != 6 or not all(isinstance(i, int) for i in val_indices):
            raise ValueError("Step60 val_indices 格式异常")
        if len(set(val_indices)) != 6 or any(
            i < 0 or i >= len(val_dataset) for i in val_indices
        ):
            raise ValueError("Step60 val_indices 超出当前 val 范围")
        print("[Step61] 固定使用 Step60 的六帧 Mini Val 作相同样本对照", flush=True)
    else:
        val_indices = choose_scene_balanced(
            val_dataset.data_infos, args.val_frames, args.seed + 1
        )
    selected_val = IndexedDataset(val_dataset, val_indices)
    val = capture_samples(
        model, transformer, head, selected_val,
        list(range(len(selected_val))), build_dataloader,
        cfg.data.nonshuffler_sampler, torch, "mini_val"
    )
    val = move_to_cpu(val)
    del val_dataset, selected_val, train_scene_set, val_scene_set
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("Val 特征缓存完成")
    if {x.token for x in train} & {x.token for x in val}:
        raise RuntimeError("Train 与 Val token 重复")
    if {x.scene for x in train} & {x.scene for x in val}:
        raise RuntimeError("Train 与 Val scene 重复")

    fusion = LanguageBEVFusion(text_dim=768, alpha_init=args.alpha).float().cuda()
    # 避免上一个实验的模式：alpha 被优化到 0，掩盖注意力本身的学习。
    fusion.alpha.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        (p for p in fusion.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=1e-3,
    )
    print("[Step61] 训练前同帧验证：注意力指标、分割指标", flush=True)
    with torch.no_grad():
        # 针对不同文本，显式验证关闭增强仍完全回退。
        for item in val[:1]:
            bev, _, reference = cuda_sample(item, torch)
            bypass, _ = fusion(bev, text[:1], enabled=False)
            logits = replay_decoder(head, bypass, torch).float()
            error = float((logits - reference).abs().max())
            if error > 1e-4:
                raise RuntimeError(f"Baseline 回退失败：误差 {error}")
    before = evaluate(fusion, head, val, text, torch)
    original = evaluate(fusion, head, val, text, torch, original=True)

    # 预计算每类正像素帧，避免每步都拿一个不存在目标的帧做正监督。
    positives: dict[int, list[int]] = {
        c: [i for i, item in enumerate(train) if (item.gt == c).any().item()]
        for c in (1, 2, 3)
    }
    missing = [NAMES[c] for c, choices in positives.items() if not choices]
    if missing:
        raise RuntimeError(
            "训练样本没有覆盖以下前景类，不能监督：" + ", ".join(missing)
            + "。建议将 --train-frames 增加至 24 或 32。"
        )
    rng = random.Random(args.seed + 17)
    history: list[dict[str, Any]] = []
    print(
        f"[Step61] 类别正例样本数：{ {NAMES[c]: len(v) for c, v in positives.items()} }",
        flush=True,
    )
    print(
        f"[Step61] 开始训练：{len(train)} 帧 / {len(val)} 留出帧 / "
        f"{args.steps} 次更新 / λatt={args.lambda_att} / 固定alpha={args.alpha}",
        flush=True,
    )
    for step in range(args.steps):
        class_id = step % 3 + 1
        # 80% 使用该类的真实正像素帧，另外 20% 从所有帧采样（可能负例）。
        choice = rng.choice(positives[class_id] if step % 5 else range(len(train)))
        sample = train[choice]
        bev, gt, _ = cuda_sample(sample, torch)
        fusion.train()
        optimizer.zero_grad(set_to_none=True)
        enhanced, attention = fusion(bev, text[class_id - 1:class_id], enabled=True)
        projected = project_attention(head, attention, torch)
        att_loss = supervision_loss(projected, gt, class_id, torch)
        logits = replay_decoder(head, enhanced, torch).float()
        seg_loss = head.loss_seg(logits, gt)
        loss = seg_loss + args.lambda_att * att_loss
        if not torch.isfinite(loss):
            raise RuntimeError("训练损失含 NaN/Inf")
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_norm=1.0))
        if not np.isfinite(grad) or grad <= 0:
            raise RuntimeError(f"融合层梯度异常：{grad}")
        optimizer.step()
        history.append({
            "step": step + 1,
            "class_name": NAMES[class_id],
            "sample_index": indices[choice],
            "gt_positive_pixels": int((gt == class_id).sum().item()),
            "attention_loss": round(float(att_loss.item()), 7),
            "segmentation_loss": round(float(seg_loss.item()), 7),
            "grad_before_clip": round(grad, 7),
            "alpha_fixed": args.alpha,
            "attention_mean": round(float(attention.mean().item()), 7),
        })
        if step < 5 or (step + 1) % 6 == 0 or step == args.steps - 1:
            print(
                f"[Step61] {step+1:03d}/{args.steps} {NAMES[class_id]} "
                f"att={att_loss.item():.5f} seg={seg_loss.item():.5f} "
                f"grad={grad:.4f}",
                flush=True,
            )
        del bev, gt, enhanced, attention, projected, att_loss, logits, seg_loss, loss

    print("[Step61] 训练完成，计算同条件验证 A/B...", flush=True)
    after = evaluate(fusion, head, val, text, torch)
    weights = args.output / "fusion_step61.pt"
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in fusion.state_dict().items()},
        "train_frames": len(train), "steps": args.steps, "alpha_fixed": args.alpha,
        "class_prompts": ["Divider", "Ped Crossing", "Boundary"],
        "note": "仅保存语言融合参数，原始 BEVFormer/DistilBERT 权重未修改",
    }, weights)
    result = {
        "experiment": "step61_supervised_class_conditioned_attention",
        "train_indices": indices,
        "val_indices": val_indices,
        "train_tokens": [s.token for s in train],
        "val_tokens": [s.token for s in val],
        "train_scenes": len({s.scene for s in train}),
        "val_scenes": len({s.scene for s in val}),
        "source": str(source),
        "checkpoint_sha256": sha256(ckpt),
        "text_sha256": sha256(text_path),
        "weights_sha256": sha256(weights),
        "steps": args.steps,
        "train_frames": len(train),
        "val_frames": len(val),
        "lambda_attention": args.lambda_att,
        "alpha_fixed": args.alpha,
        "training_loss": (
            "author segmentation CE + lambda * "
            "(class_weighted_BCE + soft_Dice)/2"
        ),
        "attention_alignment": (
            "original segmentation Head flip/transposition "
            "+ original feat_cropper"
        ),
        "geometry_assertion": "per-sample author decoder replay max_diff <= 1e-4",
        "language_supervision": (
            "class name templates from GT; NOT open-vocabulary grounding"
        ),
        "baseline_recovery_checked": True,
        "temporal_mode": "OFF, independent frames",
        "full_81frame_official_eval": False,
        "history": history,
        "original_segmentation": original,
        "before": before,
        "after": after,
        "limitations": [
            "Mini Train/Val 按数据集官方 mini 划分；Mini Val 只抽样独立帧且关闭时序",
            "语言指令和 GT 来自人工类别映射，不是 DriveLM 目标定位真值",
            "用同一组冻结原始 BEVFormer 权重、图像、标签比较原始、训练前和训练后",
            "类别文本并不提供额外目标位置标注，不能验证任意自由文本语义定位",
            "注意力 IoU 的阈值固定为 0.5，需同时参考 Soft Dice 和分割 mIoU",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    report = args.output / "step61_supervised_summary.json"
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = []
    for cls in NAMES[1:]:
        pre = before["per_prompt"][cls]
        post = after["per_prompt"][cls]
        base = original["per_prompt"][cls]
        rows.append(
            f"| {cls} | "
            f"{fmt_pct(pre['attention']['attention_iou_at_0_5'])} | "
            f"{fmt_pct(post['attention']['attention_iou_at_0_5'])} | "
            f"{fmt_pct(base['segmentation']['foreground_miou_argmax'])} | "
            f"{fmt_pct(post['segmentation']['foreground_miou_argmax'])} |"
        )
    lines = [
        "# Step61：类别引导 BEV 注意力监督训练", "",
        f"Mini Train 不同帧：{len(train)}；Mini Val 独立帧：{len(val)}；更新：{args.steps}。",
        f"监督：GT 生成类别二值掩码，投影使用作者原始 feat_cropper；固定 alpha={args.alpha}。",
        "",
        (
            "| 指令类别 | 未训练 Attention IoU@0.5 | 训练后 Attention IoU@0.5 "
            "| 原始分割前景 mIoU | 训练后分割前景 mIoU |"
        ),
        "|---|---:|---:|---:|---:|", *rows,
        "", "## 结论边界", "",
        "- 三类文本指令为人工映射的标签类别监督，不是 DriveLM 场景语言对齐监督。",
        "- Attention IoU 只反映注意力与 GT 的重合，不能单独代表模型分割性能。",
        "- 使用 frozen backbone/decoder，GT 仅用于训练和验证，不在推理中喂给融合模块。",
        "- 最终对照时序 OFF、仅 Mini Val 抽样，不能与作者 81 帧时序 mIoU 比较。",
        "- 请对照原始 BEVFormer，而不是仅对照未训练融合。", "",
    ]
    (args.output / "step61_review.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n[Step61] 同条件验证 A/B：", flush=True)
    for row in rows:
        print(row, flush=True)
    print("[Step61] JSON：", report, flush=True)
    print("[Step61] 中文报告：", args.output / "step61_review.md", flush=True)
    print("[Step61] 训练权重：", weights, flush=True)
    print("[Step61] 注：类别指令监督 ≠ 任意自然语言定位，Mini Val ≠ 81 帧时序", flush=True)


if __name__ == "__main__":
    main()

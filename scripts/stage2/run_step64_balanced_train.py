"""Step64：细线类别难样本训练与固定对照。

使用已经验证过的 Step59/60 作者分割 Encoder + Decoder replay；
沿用作者分割头特征采样，采用正样本平衡/困难负例/细线边缘容忍训练。
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
OUTPUT = ROOT / "outputs/stage2_step64"
PREVIOUS = ROOT / "outputs/stage2_step61/step61_supervised_summary.json"
NAMES = ("Background", "Divider", "Ped Crossing", "Boundary")


def parse_args() -> argparse.Namespace:
    """支持先做少量样本测试，再增加样本数量与更新次数。"""
    parser = argparse.ArgumentParser(description="Step64 类别文本→地图 GT 注意力监督")
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
    """细线友好损失：正例 BCE、难负例 BCE、Dice、正负排序。

    - 远离 GT 的错误高亮区域通过 hard-negative mining 增强惩罚。
    - GT 2 像素邻域内降低负例罚分，缓解细线边缘亚像素偏差。
    - 真实 GT 正像素的权重不由区域总大小稀释。
    - 无目标帧只进行负例抑制，不能假造正样本。

    注意：这是训练损失设计假设，尚未证明 GT 与 BEV 物理标定无误。
    """
    import torch.nn.functional as functional

    if class_id not in (1, 2, 3):
        raise ValueError("类别 id 必须为 1/2/3")
    if tuple(attention_map.shape) != (1, 1, 200, 400):
        raise ValueError("注意力输入形状不符合 (1,1,200,400)")
    if tuple(gt.shape) != (1, 200, 400):
        raise ValueError("GT 输入形状不符合 (1,200,400)")
    if gt.min().item() < 0 or gt.max().item() > 3:
        raise ValueError("地图 GT 必须包含 0~3 类别")

    p = attention_map.float().clamp(1e-6, 1 - 1e-6)
    y = (gt == class_id).unsqueeze(1)
    n_pos = int(y.sum().item())
    negatives = ~y
    neg_bce = -torch.log1p(-p[negatives])
    if not n_pos:
        k = min(2048, neg_bce.numel())
        return 0.5 * neg_bce.mean() + 0.5 * neg_bce.topk(k).values.mean()

    # 细线标注可能只占很少像素，使用两个像素宽的邻域作为边缘容忍带。
    # 这是降低训练惩罚，不是修改 GT，也不是在评估阶段膨胀目标。
    near = functional.max_pool2d(y.float(), kernel_size=5, stride=1, padding=2).bool()
    border = near & negatives
    weighted_neg = neg_bce * torch.where(
        border[negatives], neg_bce.new_tensor(0.35), neg_bce.new_tensor(1.0)
    )
    # 固定比例难负例采样，避免无限强化本就占绝大多数的背景。
    k = min(weighted_neg.numel(), max(512, 3 * n_pos))
    hard = weighted_neg.topk(k, sorted=False).values.mean()
    pos_bce = -torch.log(p[y]).mean()
    bce = 0.5 * pos_bce + 0.5 * hard

    y_float = y.float()
    overlap = (p * y_float).sum()
    dice = 1 - (2 * overlap + 1) / (p.sum() + y_float.sum() + 1)
    # 目标区域注意力应至少比非目标区域高出一段 margin。
    # 针对难负样本的均值进行对比，且没有使用额外目标位置标注。
    neg_high = p[negatives].topk(k, sorted=False).values.mean()
    ranking = functional.softplus(3.0 * (0.10 + neg_high - p[y].mean())) / 3.0
    return 0.55 * bce + 0.25 * dice + 0.20 * ranking


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


def attention_ap_by_class(
    fusion: Any, head: Any, val: list[Any], text: Any, torch: Any
) -> dict[str, Any]:
    """不调阈值，在整批验证帧上计算与 Step62 同口径的 AP。"""
    from scripts.stage2.run_step60_extended_ab import cuda_sample
    from scripts.stage2.run_step62_attention_diagnostic import precision_recall_auc

    output: dict[str, Any] = {}
    fusion.eval()
    with torch.no_grad():
        for class_id, name in enumerate(NAMES[1:], start=1):
            all_scores = []
            all_labels = []
            for item in val:
                bev, gt, _ = cuda_sample(item, torch)
                _, attention = fusion(bev, text[class_id - 1 : class_id], enabled=True)
                projected = project_attention(head, attention, torch)
                all_scores.append(projected.float().cpu().numpy().ravel())
                all_labels.append((gt == class_id).cpu().numpy().ravel())
                del bev, gt, attention, projected
            probabilities = np.concatenate(all_scores)
            labels = np.concatenate(all_labels)
            output[name] = {
                "average_precision": precision_recall_auc(probabilities, labels),
                "positive_prevalence": float(labels.mean()),
                "positive_pixels": int(labels.sum()),
                "pixels": int(labels.size),
            }
    return output


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
        raise AssertionError("Step64 Attention Loss 梯度检查失败")
    stats = binary_counts(projected.detach(), (gt == 1).unsqueeze(1), torch)
    if stats["gt_positive_pixels"] != 1000:
        raise AssertionError("GT 类别面积不正确")
    assert tuple(projected.shape) == (1, 1, 200, 400)
    print("[Step64] CPU 自测 PASS：投影尺寸、困难负样本损失、梯度", flush=True)
    print("[Step64] 注意：该自测不代表真实作者模型或实测准确率", flush=True)


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
    if (args.output / "step64_supervised_summary.json").exists():
        raise FileExistsError(
            "本输出目录已有 Step64 结果，禁止覆盖。请通过 --output 指定新目录。"
        )
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
        raise RuntimeError("Step64 作者模型需要 CUDA")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    text_arr = np.load(text_path, allow_pickle=False)
    if tuple(text_arr.shape) != (3, 768) or not np.isfinite(text_arr).all():
        raise ValueError(f"需要三个类别文本向量 (3,768)，实际 {text_arr.shape}")
    text = torch.from_numpy(text_arr).float().cuda()
    diag_path = ROOT / "outputs/stage2_step62/step62_diagnostics.json"
    if not diag_path.is_file() or not args.previous.is_file():
        raise FileNotFoundError("缺少 Step61 / Step62 参考报告，禁止无基线比较")
    diag = json.loads(diag_path.read_text(encoding="utf-8"))
    prev = json.loads(args.previous.read_text(encoding="utf-8"))
    if diag.get("val_indices") != prev.get("val_indices"):
        raise RuntimeError("Step61/62 验证样本不一致")
    if diag.get("checkpoint_sha256") != sha256(ckpt):
        raise RuntimeError("Step62 与本次模型权重 hash 不一致")
    if diag.get("text_sha256") != sha256(text_path):
        raise RuntimeError("Step62 与本次文本向量 hash 不一致")
    reference_ap = {name: diag["scores"][name]["average_precision"] for name in NAMES[1:]}
    cfg = Config.fromfile(str(config_path))
    if (cfg.model.pts_bbox_head.bev_h, cfg.model.pts_bbox_head.bev_w) != (150, 150):
        raise ValueError("不支持非 150×150 的 BEV 网格")
    if not bool(cfg.model.pts_bbox_head.task.get("seg")):
        raise ValueError("作者配置没有分割头")
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    cfg.data.workers_per_gpu = 0

    print("[Step64] 构建原始冻结模型并加载 checkpoint", flush=True)
    started = time.perf_counter()
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(ckpt), map_location="cpu")
    model.requires_grad_(False)
    model.eval()
    model = MMDataParallel(model.cuda(), device_ids=[0])
    head = model.module.pts_bbox_head
    transformer = head.transformer
    print_memory("冻结模型就绪")

    print("[Step64] 顺序提取 Mini Train 特征+GT...", flush=True)
    train_dataset = build_mini_dataset(cfg, build_dataset, "train")
    train_scene_set = {str(i["scene_token"]) for i in train_dataset.data_infos}
    indices = choose_scene_balanced(
        train_dataset.data_infos, args.train_frames, args.seed
    )
    if args.train_frames == 16 and args.previous.is_file():
        ref_data = json.loads(args.previous.read_text(encoding="utf-8"))
        if indices != ref_data.get("train_indices"):
            raise RuntimeError("Step64 与 Step61 训练帧不一致，禁止宣称同条件比较")
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

    print("[Step64] 顺序提取 Mini Val 特征+GT...", flush=True)
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
        print("[Step64] 固定使用 Step61 的六帧 Mini Val 作相同样本对照", flush=True)
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
    if args.val_frames == 6 and [x.token for x in val] != prev.get("val_tokens"):
        raise RuntimeError("Step64 与 Step61 验证 sample token 不一致，停止实验")
    if args.train_frames == 16 and [x.token for x in train] != prev.get("train_tokens"):
        raise RuntimeError("Step64 与 Step61 训练 sample token 不一致，停止实验")
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
    print("[Step64] 训练前同帧验证：注意力指标、分割指标", flush=True)
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
    before_ap = attention_ap_by_class(fusion, head, val, text, torch)

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
        f"[Step64] 类别正例样本数：{ {NAMES[c]: len(v) for c, v in positives.items()} }",
        flush=True,
    )
    print(
        f"[Step64] 开始训练：{len(train)} 帧 / {len(val)} 留出帧 / "
        f"{args.steps} 次更新 / λatt={args.lambda_att} / 固定alpha={args.alpha}",
        flush=True,
    )
    for step in range(args.steps):
        class_id = step % 3 + 1
        # 90% 采正例，10% 随机负例/正例混合，避免稀疏类别被空掩码稀释。
        choice = rng.choice(positives[class_id] if step % 10 else range(len(train)))
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
                f"[Step64] {step+1:03d}/{args.steps} {NAMES[class_id]} "
                f"att={att_loss.item():.5f} seg={seg_loss.item():.5f} "
                f"grad={grad:.4f}",
                flush=True,
            )
        del bev, gt, enhanced, attention, projected, att_loss, logits, seg_loss, loss

    print("[Step64] 训练完成，计算同条件验证 A/B...", flush=True)
    after = evaluate(fusion, head, val, text, torch)
    after_ap = attention_ap_by_class(fusion, head, val, text, torch)
    weights = args.output / "fusion_step64.pt"
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in fusion.state_dict().items()},
        "train_frames": len(train), "steps": args.steps, "alpha_fixed": args.alpha,
        "class_prompts": ["Divider", "Ped Crossing", "Boundary"],
        "note": "仅保存语言融合参数，原始 BEVFormer/DistilBERT 权重未修改",
    }, weights)
    result = {
        "experiment": "step64_balanced_hard_negative_finetune",
        "train_indices": indices,
        "val_indices": val_indices,
        "train_tokens": [s.token for s in train],
        "val_tokens": [s.token for s in val],
        "train_scenes": len({s.scene for s in train}),
        "val_scenes": len({s.scene for s in val}),
        "source": str(source),
        "step61_reference": str(args.previous),
        "step62_reference": str(diag_path),
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
            "(balanced_positive_BCE + hard_negative_BCE + soft_Dice + ranking)"
        ),
        "hard_negative_config": {"negative_topk_multiplier": 3, "min_negative_k": 512,
                                 "border_radius_px": 2, "border_negative_weight": 0.35,
                                 "positive_sampling_fraction": 0.9},
        "attention_alignment": (
            "original segmentation Head flip/transposition "
            "+ original feat_cropper"
        ),
        "geometry_assertion": "per-sample author decoder replay max_diff <= 1e-4",
        "language_supervision": (
            "class name templates from GT; NOT open-vocabulary grounding"
        ),
        "step61_comparison_controls": "same 16 train + 6 val + 72 steps + seed when defaults used",
        "evaluation_previously_used": True,
        "baseline_recovery_checked": True,
        "temporal_mode": "OFF, independent frames",
        "full_81frame_official_eval": False,
        "history": history,
        "original_segmentation": original,
        "before": before,
        "after": after,
        "attention_ap_before": before_ap,
        "attention_ap_after": after_ap,
        "step61_reference_att_ap": reference_ap,
        "limitations": [
            "同一 6 帧 Mini Val 已参与前期诊断与模型决策，属于开发验证集，不能作为最终盲测提升证据",
            "更改 Loss 及采样比例，不得仅凭 Step64 对 Step61 的差异作唯一因果归因",
            "保留真实 GT 的四类别监督而非额外语言空间标注；不支持开放词表定位结论",
            "Mini Train/Val 按数据集官方 mini 划分；Mini Val 只抽样独立帧且关闭时序",
            "语言指令和 GT 来自人工类别映射，不是 DriveLM 目标定位真值",
            "用同一组冻结原始 BEVFormer 权重、图像、标签比较原始、训练前和训练后",
            "类别文本并不提供额外目标位置标注，不能验证任意自由文本语义定位",
            "注意力 IoU 的阈值固定为 0.5，需同时参考 Soft Dice 和分割 mIoU",
        ],
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    report = args.output / "step64_supervised_summary.json"
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
        "# Step64：细线与困难负例平衡训练 A/B", "",
        f"Mini Train 不同帧：{len(train)}；Mini Val 独立帧：{len(val)}；更新：{args.steps}。",
        f"监督：GT 生成类别二值掩码，投影使用作者原始 feat_cropper；固定 alpha={args.alpha}。",
        "",
        (
            "| 指令类别 | 未训练 Attention IoU@0.5 | 训练后 Attention IoU@0.5 "
            "| 原始分割前景 mIoU | 训练后分割前景 mIoU |"
        ),
        "|---|---:|---:|---:|---:|", *rows,
        "", "## 注意力 AP（与 Step62 同口径）", "",
        "| 类别 | Step61 AP | Step64 训练前 AP | Step64 训练后 AP | GT 像素占比 |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {name} | {fmt_pct(reference_ap[name])} | "
            f"{fmt_pct(before_ap[name]['average_precision'])} | "
            f"{fmt_pct(after_ap[name]['average_precision'])} | "
            f"{fmt_pct(after_ap[name]['positive_prevalence'])} |"
            for name in NAMES[1:]
        ],
        "", "## 结论边界", "",
        "- 监督掩码仍为原始地图 GT，训练时只降低邻域负例罚分，不修改 GT。",
        "- 三类文本指令为人工映射的标签类别监督，不是 DriveLM 场景语言对齐监督。",
        "- Attention IoU 只反映注意力与 GT 的重合，不能单独代表模型分割性能。",
        "- 使用 frozen backbone/decoder，GT 仅用于训练和验证，不在推理中喂给融合模块。",
        "- 最终对照时序 OFF、仅 Mini Val 抽样，不能与作者 81 帧时序 mIoU 比较。",
        "- 请对照原始 BEVFormer，而不是仅对照未训练融合。",
        "- 同一 Mini Val 六帧已用于 Step62 开发诊断，不能当作全新独立盲测。", "",
    ]
    (args.output / "step64_review.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n[Step64] 同条件验证 A/B：", flush=True)
    for row in rows:
        print(row, flush=True)
    print("[Step64] Attention AP：", flush=True)
    for name in NAMES[1:]:
        print(f"  {name}: Step61 {reference_ap[name]:.5f}; "
              f"Step64 {after_ap[name]['average_precision']:.5f}; "
              f"Random prevalence {after_ap[name]['positive_prevalence']:.5f}", flush=True)
    print("[Step64] JSON：", report, flush=True)
    print("[Step64] 中文报告：", args.output / "step64_review.md", flush=True)
    print("[Step64] 训练权重：", weights, flush=True)
    print("[Step64] 注：类别指令监督 ≠ 任意自然语言定位，Mini Val ≠ 81 帧时序", flush=True)


if __name__ == "__main__":
    main()

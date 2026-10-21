#!/usr/bin/env python3
"""安装 Step60 多场景训练与同条件 Mini Val A/B 脚本。"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts/stage2/run_step60_extended_ab.py"

SOURCE = r'''"""Step60：多场景 Mini Train 训练与 Mini Val 同条件分割 A/B。

使用 Step59 已验证的真实 BEV/GT 提取及 Decoder 重放逻辑；
作者 BEVFormer、分割 Decoder、DistilBERT 全程冻结。
所有对照均关闭时序，不能与 81 帧时序官方 mIoU 直接比较。
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
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTHOR = Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg"
DEFAULT_CKPT = Path("/mnt/g/nus/bevformer_small_seg.pth")
DEFAULT_TEXT = ROOT / "outputs/stage2_step58/distilbert_prompts.npy"
DEFAULT_OUT = ROOT / "outputs/stage2_step60"
CLASS_NAMES = ("Background", "Divider", "Ped Crossing", "Boundary")


def parse_args() -> argparse.Namespace:
    """限制样本与迭代规模，降低 WSL / GPU 的内存峰值。"""
    parser = argparse.ArgumentParser(description="Step60 多场景真实 GT 训练与 A/B")
    parser.add_argument("--author", type=Path, default=DEFAULT_AUTHOR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-frames", type=int, default=8)
    parser.add_argument("--val-frames", type=int, default=6)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    if not 4 <= args.train_frames <= 16:
        parser.error("--train-frames 必须在 4~16 之间")
    if not 4 <= args.val_frames <= 16:
        parser.error("--val-frames 必须在 4~16 之间")
    if not 8 <= args.steps <= 80:
        parser.error("--steps 必须在 8~80 之间")
    if not 0 < args.lr <= 0.002:
        parser.error("--lr 必须大于 0 且不超过 0.002")
    return args


def print_memory(tag: str) -> None:
    """显示 WSL 进程和可用内存，方便排查 OOM。"""
    values = {}
    for file, keys in (
        ("/proc/self/status", ("VmHWM:", "VmRSS:")),
        ("/proc/meminfo", ("MemAvailable:", "SwapFree:")),
    ):
        try:
            for line in Path(file).read_text(encoding="utf-8").splitlines():
                if line.startswith(keys):
                    key, value = line.split(":", 1)
                    values[key] = value.strip()
        except OSError:
            pass
    print(f"[Step60][内存] {tag}: {values}", flush=True)


def choose_scene_balanced(infos: list[dict[str, Any]], count: int, seed: int) -> list[int]:
    """按 scene 轮流抽样，避免默认取前 N 帧而只覆盖一个场景。"""
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, info in enumerate(infos):
        grouped[str(info["scene_token"])].append(i)
    rng = random.Random(seed)
    scene_names = sorted(grouped)
    rng.shuffle(scene_names)
    for scene in scene_names:
        rng.shuffle(grouped[scene])
    picked: list[int] = []
    while len(picked) < count:
        progress = False
        for scene in scene_names:
            if grouped[scene]:
                picked.append(grouped[scene].pop())
                progress = True
                if len(picked) >= count:
                    break
        if not progress:
            break
    if len(picked) != count:
        raise RuntimeError(f"需要 {count} 帧，实际找到 {len(picked)} 帧")
    return sorted(picked)


class IndexedDataset:
    """只访问指定原始索引，避免扫描前面几百帧浪费时间。"""

    def __init__(self, dataset: Any, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.data_infos = [dataset.data_infos[i] for i in indices]
        flags = getattr(dataset, "flag", None)
        self.flag = (
            np.asarray(flags)[indices]
            if flags is not None else np.zeros(len(indices), dtype=np.uint8)
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[self.indices[index]]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def move_to_cpu(samples: list[Any]) -> list[Any]:
    """GPU 上只保留模型，缓存特征/GT 移到 CPU，限制显存峰值。"""
    from scripts.stage2.run_step59_train import CachedSample

    return [
        CachedSample(
            item.token,
            item.scene,
            item.bev.detach().cpu().contiguous(),
            item.gt.detach().cpu().contiguous(),
            item.reference_logits.detach().cpu().contiguous(),
        )
        for item in samples
    ]


def cuda_sample(sample: Any, torch: Any) -> tuple[Any, Any, Any]:
    """仅移动当前样本到 CUDA，避免同时保存全部特征。"""
    return (
        sample.bev.to("cuda", non_blocking=True),
        sample.gt.to("cuda", non_blocking=True),
        sample.reference_logits.to("cuda", non_blocking=True),
    )


def prompt_index(token: str, length: int) -> int:
    """稳定的文本索引；并不代表存在逐帧语言标签。"""
    return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % length


def evaluate_branch(
    fusion: Any,
    head: Any,
    samples: list[Any],
    text: Any,
    torch: Any,
    branch: str,
) -> dict[str, Any]:
    """使用同一批独立帧 GT 和原始分割输出，统计四类 IoU。"""
    from scripts.stage2.run_step59_train import replay_decoder

    if branch not in ("original", "before", "after"):
        raise ValueError(f"无效分支：{branch}")
    fusion.eval()
    confusion = torch.zeros((4, 4), dtype=torch.long)
    losses: list[float] = []
    differences: list[float] = []
    same_sample_details: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in samples:
            bev, gt, reference = cuda_sample(sample, torch)
            if branch == "original":
                logits = reference
                # 关闭融合必须和真正的作者模型一致。
                bypass, _ = fusion(bev, text[0:1], enabled=False)
                replay = replay_decoder(head, bypass, torch).float()
                max_diff = float((replay - reference).abs().max().item())
                if max_diff > 1e-4:
                    raise RuntimeError(
                        f"Baseline 回退不一致：{sample.token} / {max_diff}"
                    )
            else:
                index = prompt_index(sample.token, int(text.shape[0]))
                enhanced, _ = fusion(bev, text[index:index + 1], enabled=True)
                logits = replay_decoder(head, enhanced, torch).float()
                max_diff = float((logits - reference).abs().max().item())
            if tuple(logits.shape) != (1, 4, 200, 400):
                raise RuntimeError("分割输出尺寸错误")
            loss = head.loss_seg(logits.float(), gt)
            if not torch.isfinite(loss):
                raise ValueError("验证损失不是有限值")
            labels = logits.argmax(1).reshape(-1).long()
            truth = gt.reshape(-1).long()
            counts = torch.bincount(truth * 4 + labels, minlength=16)
            confusion += counts.reshape(4, 4).cpu()
            differences.append(float((logits - reference).abs().mean().item()))
            losses.append(float(loss.item()))
            same_sample_details.append({
                "token": sample.token,
                "scene": sample.scene,
                "prompt_index": prompt_index(sample.token, int(text.shape[0])),
                "loss": round(float(loss.item()), 7),
                "max_logit_difference_vs_original": round(max_diff, 8),
            })
            del bev, gt, reference, logits
    matrix = confusion.numpy()
    ious: dict[str, float | None] = {}
    class_support = {}
    for idx, name in enumerate(CLASS_NAMES):
        tp = int(matrix[idx, idx])
        union = int(matrix[idx, :].sum() + matrix[:, idx].sum() - tp)
        ious[name] = round(tp / union, 8) if union else None
        class_support[name] = int(matrix[idx, :].sum())
    valid = [ious[name] for name in CLASS_NAMES[1:] if ious[name] is not None]
    foreground_miou = float(np.mean(valid)) if valid else None
    correct = int(np.trace(matrix))
    total = int(matrix.sum())
    return {
        "frame_count": len(samples),
        "mean_seg_loss": round(float(np.mean(losses)), 8),
        "foreground_miou_argmax": (
            round(foreground_miou, 8) if foreground_miou is not None else None
        ),
        "per_class_iou": ious,
        "gt_pixel_counts": class_support,
        "pixel_accuracy_including_background": round(correct / total, 8),
        "confusion_gt_rows_pred_columns": matrix.tolist(),
        "mean_absolute_output_difference_vs_original": round(
            float(np.mean(differences)), 8
        ),
        "per_frame": same_sample_details,
    }


def main() -> None:
    """顺序缓存 Train/Val，冻结基座，训练融合层并生成 A/B 报告。"""
    args = parse_args()
    source = args.author.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    text_path = args.text.expanduser().resolve()
    config_path = source / "projects/configs/bevformer/bevformer_small_seg.py"
    for path in (source, checkpoint, text_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f"输入不存在：{path}")
    args.output = args.output.expanduser().resolve()
    if args.output == ROOT:
        raise ValueError("拒绝把工程根目录作为输出目录")
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
        build_mini_dataset,
        capture_samples,
        sha256,
    )

    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA；真实作者推理不能在 CPU 上继续")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    text_np = np.load(text_path, allow_pickle=False)
    if text_np.shape != (2, 768) or not np.isfinite(text_np).all():
        raise ValueError("Step58 文本特征必须是 (2,768) 的有限实数")
    text = torch.from_numpy(text_np).float().cuda()
    cfg = Config.fromfile(str(config_path))
    if (int(cfg.model.pts_bbox_head.bev_h), int(cfg.model.pts_bbox_head.bev_w)) != (
        150, 150
    ):
        raise ValueError("BEV 网格并非 150×150，停止运行")
    if not bool(cfg.model.pts_bbox_head.task.get("seg")):
        raise ValueError("当前配置没有分割任务")
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    cfg.data.workers_per_gpu = 0

    started = time.perf_counter()
    print("[Step60] 加载作者冻结模型和权重...", flush=True)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(checkpoint), map_location="cpu")
    model.requires_grad_(False)
    model.eval()
    model = MMDataParallel(model.cuda(), device_ids=[0])
    head = model.module.pts_bbox_head
    transformer = head.transformer
    print(f"[Step60] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print_memory("冻结模型准备完成")

    print("[Step60] 构建 Mini Train，按场景选择真实训练帧...", flush=True)
    train_ds = build_mini_dataset(cfg, build_dataset, "train")
    train_scenes = {str(i["scene_token"]) for i in train_ds.data_infos}
    train_indices = choose_scene_balanced(
        train_ds.data_infos, args.train_frames, args.seed
    )
    selected_train = IndexedDataset(train_ds, train_indices)
    print(f"[Step60] Train 原始索引: {train_indices}", flush=True)
    train_items = capture_samples(
        model, transformer, head, selected_train,
        list(range(len(selected_train))), build_dataloader,
        cfg.data.nonshuffler_sampler, torch, "mini_train",
    )
    train_items = move_to_cpu(train_items)
    del selected_train, train_ds
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("Train 特征缓存到 CPU，释放 Train Dataset")

    print("[Step60] 构建 Mini Val，按场景选择独立验证帧...", flush=True)
    val_ds = build_mini_dataset(cfg, build_dataset, "val")
    val_scenes = {str(i["scene_token"]) for i in val_ds.data_infos}
    if train_scenes & val_scenes:
        raise RuntimeError("Mini Train/Val 场景重叠，拒绝训练")
    val_indices = choose_scene_balanced(
        val_ds.data_infos, args.val_frames, args.seed + 1
    )
    selected_val = IndexedDataset(val_ds, val_indices)
    print(f"[Step60] Val 原始索引: {val_indices}", flush=True)
    val_items = capture_samples(
        model, transformer, head, selected_val,
        list(range(len(selected_val))), build_dataloader,
        cfg.data.nonshuffler_sampler, torch, "mini_val",
    )
    val_items = move_to_cpu(val_items)
    del selected_val, val_ds, train_scenes, val_scenes
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("Val 特征缓存到 CPU，释放 Val Dataset")
    if {s.token for s in train_items} & {s.token for s in val_items}:
        raise RuntimeError("Train 和 Val 的 sample_token 重叠")
    if {s.scene for s in train_items} & {s.scene for s in val_items}:
        raise RuntimeError("Train 和 Val 的 scene_token 重叠")

    fusion = LanguageBEVFusion(text_dim=768, alpha_init=0.01).cuda().float()
    optimizer = torch.optim.AdamW(
        fusion.parameters(), lr=args.lr, weight_decay=1e-3
    )
    if not any(p.requires_grad for p in fusion.parameters()):
        raise RuntimeError("融合层没有可训练参数")
    print("[Step60] 测试基线回退，并评估训练前相同验证集...", flush=True)
    original = evaluate_branch(fusion, head, val_items, text, torch, "original")
    before = evaluate_branch(fusion, head, val_items, text, torch, "before")

    rng = random.Random(args.seed)
    history: list[dict[str, Any]] = []
    order = list(range(len(train_items)))
    print(
        f"[Step60] 开始训练：不同训练帧 {len(train_items)} / "
        f"验证帧 {len(val_items)} / 迭代 {args.steps}",
        flush=True,
    )
    for step in range(args.steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        item_index = order[step % len(order)]
        sample = train_items[item_index]
        bev, gt, _ = cuda_sample(sample, torch)
        pidx = prompt_index(sample.token, int(text.shape[0]))
        fusion.train()
        optimizer.zero_grad(set_to_none=True)
        enhanced, attention = fusion(bev, text[pidx:pidx + 1], enabled=True)
        from scripts.stage2.run_step59_train import replay_decoder

        logits = replay_decoder(head, enhanced, torch).float()
        seg_loss = head.loss_seg(logits, gt)
        total_loss = seg_loss + 0.01 * fusion.alpha.square()
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"第 {step + 1} 次训练 Loss 出现 NaN/Inf")
        total_loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0))
        if not np.isfinite(grad) or grad <= 0:
            raise RuntimeError(f"第 {step + 1} 次训练梯度错误：{grad}")
        optimizer.step()
        with torch.no_grad():
            fusion.alpha.clamp_(-0.3, 0.3)
        history.append({
            "step": step + 1,
            "train_index_original": train_indices[item_index],
            "prompt_index": pidx,
            "seg_loss": round(float(seg_loss.item()), 7),
            "gradient_norm_before_clip": round(grad, 7),
            "alpha": round(float(fusion.alpha.item()), 8),
            "attention_mean": round(float(attention.mean().item()), 7),
        })
        print(
            f"[Step60] {step + 1:02d}/{args.steps} "
            f"frame={train_indices[item_index]} "
            f"seg_loss={seg_loss.item():.6f} grad={grad:.4f}",
            flush=True,
        )
        del bev, gt, logits, enhanced, attention, seg_loss, total_loss

    print("[Step60] 训练结束，评估训练后相同 Mini Val 帧...", flush=True)
    after = evaluate_branch(fusion, head, val_items, text, torch, "after")
    state_path = args.output / "fusion_step60.pt"
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in fusion.state_dict().items()},
        "text_dim": 768,
        "bev_channels": 256,
        "optimizer_steps": args.steps,
        "seed": args.seed,
        "note": "仅 LanguageBEVFusion；未包含 BEVFormer 或 DistilBERT 权重",
    }, state_path)
    result = {
        "experiment": "step60_scene_balanced_mini_train_heldout_ab",
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_tokens": [s.token for s in train_items],
        "val_tokens": [s.token for s in val_items],
        "train_distinct_scenes": len({s.scene for s in train_items}),
        "val_distinct_scenes": len({s.scene for s in val_items}),
        "optimizer_steps": args.steps,
        "language_assignment": "deterministic token hash, NOT language-grounding labels",
        "checkpoint_sha256": sha256(checkpoint),
        "text_embedding_sha256": sha256(text_path),
        "fusion_weights": str(state_path),
        "fusion_weights_sha256": sha256(state_path),
        "split": "nuScenes v1.0-mini: mini_train vs mini_val",
        "temporal": False,
        "segmentation_metric": "aggregated argmax confusion IoU, foreground macro",
        "official_81_frame_miou": False,
        "language_grounding_supervised": False,
        "frozen": ["BEVFormer", "segmentation decoder", "DistilBERT"],
        "initialization": "new random fusion parameters; Step59 weights not reused",
        "history": history,
        "heldout_original": original,
        "heldout_before": before,
        "heldout_after": after,
        "delta_foreground_miou_vs_original": (
            round(after["foreground_miou_argmax"] - original["foreground_miou_argmax"], 8)
            if after["foreground_miou_argmax"] is not None
            and original["foreground_miou_argmax"] is not None else None
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "limitations": [
            "Mini Val 仅抽样独立帧、时序关闭，不是完整 81 帧时序官方指标",
            "训练使用两条固定且无逐帧目标标注的文本向量，不能验证语言定位",
            "冻结真实 BEVFormer 和原作者分割头，仅训练小型语言融合层",
            "IoU 来源相同 GT 和 argmax 混淆矩阵，不等同于作者可能采用的其他评估口径",
        ],
    }
    output_json = args.output / "step60_ab_summary.json"
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fmt = lambda x: f"{x:.4%}" if x is not None else "N/A"
    rows = []
    for name, r in (("原始 BEVFormer", original),
                    ("未训练融合", before), ("训练后融合", after)):
        cls = r["per_class_iou"]
        rows.append(
            f"| {name} | {r['mean_seg_loss']:.6f} | "
            f"{fmt(cls['Divider'])} | {fmt(cls['Ped Crossing'])} | "
            f"{fmt(cls['Boundary'])} | {fmt(r['foreground_miou_argmax'])} |"
        )
    md = [
        "# Step60：跨场景训练与独立 Mini Val A/B", "",
        f"- Mini Train：{len(train_items)} 帧、"
        f"{len({s.scene for s in train_items})} 个场景。",
        f"- Mini Val：{len(val_items)} 帧、"
        f"{len({s.scene for s in val_items})} 个场景。",
        f"- 训练迭代：{args.steps}，时序关闭。", "",
        "| 分支 | 平均分割 Loss | Divider IoU | Ped Crossing IoU | "
        "Boundary IoU | 前景 mIoU |",
        "|---|---:|---:|---:|---:|---:|",
        *rows,
        "", "## 实验边界", "",
        "- 按 scene_token 抽取不同帧，训练与验证划分不重叠。",
        "- 冻结 BEVFormer / Decoder / DistilBERT；仅优化语言融合模块。",
        "- 真实地图 GT，同一验证帧、同一分割 Decoder 对照三分支。",
        "- 文本仅按 token 哈希分配，没有语义空间标注，不证明语言定位。",
        "- 本实验关闭时序、不是 Mini Val 81 帧官方 IoU，不能直接和 6.38% 比较。",
        "- 使用像素 argmax 混淆矩阵聚合 IoU，缺失类别按 N/A 处理。",
        "- 原始模型才是参照；训练后好于未训练并不等于好于原始模型。", "",
    ]
    (args.output / "step60_review.md").write_text("\n".join(md), encoding="utf-8")
    print("\n[Step60] 同条件验证集结果：", flush=True)
    for line in rows:
        print(line, flush=True)
    print("[Step60] 报告：", output_json, flush=True)
    print("[Step60] 中文摘要：", args.output / "step60_review.md", flush=True)
    print("[Step60] 融合权重：", state_path, flush=True)
    print("[Step60] 注意：非 81 帧时序官方 mIoU，语言定位能力未验证", flush=True)


if __name__ == "__main__":
    main()
'''


def main() -> None:
    """验证代码与目录并写入，不覆盖用户已经修改的版本。"""
    required = [
        ROOT / "scripts/stage2/run_step59_train.py",
        ROOT / "scripts/stage2/bev_fusion.py",
        ROOT / "outputs/stage2_step58/distilbert_prompts.npy",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少 Step59/Step58 必需文件：\n" + "\n".join(missing))
    ast.parse(SOURCE, filename=str(TARGET))
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        existing = TARGET.read_text(encoding="utf-8")
        if existing != SOURCE:
            raise FileExistsError(
                f"已存在内容不同的 Step60 文件，请先手动核对：{TARGET}"
            )
        print("[Step60] 脚本已安装，无需覆盖：", TARGET)
        return
    TARGET.write_text(SOURCE, encoding="utf-8")
    print("[Step60] 多场景训练和 IoU A/B 脚本安装完成")
    print("[Step60] 目标文件：", TARGET)
    print("[Step60] 默认：Mini Train 8 帧 + Mini Val 6 帧 + 24 次更新")
    print("[Step60] 真实训练需使用作者 .venv-author 环境")


if __name__ == "__main__":
    main()

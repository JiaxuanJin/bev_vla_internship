"""Step59：冻结真实 BEVFormer/DistilBERT，使用 nuScenes Mini Train GT 微调融合层。

过程：
1. 从作者原始 checkpoint 对 Mini Train/Mini Val 进行真实单帧推理。
2. 缓存 transformer 的真实 BEV 输出、原始分割 logits 和同帧 Map GT。
3. 重放作者原始分割 Decoder，严格检验与完整前向输出一致。
4. 仅训练 LanguageBEVFusion，并在独立 Mini Val 场景核对 Loss。

此任务不是训练整个 BEVFormer，也不代表语言目标定位或官方 mIoU 改善。
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTHOR = Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg"
DEFAULT_CKPT = Path("/mnt/g/nus/bevformer_small_seg.pth")
DEFAULT_TEXT = ROOT / "outputs/stage2_step58/distilbert_prompts.npy"
DEFAULT_OUTPUT = ROOT / "outputs/stage2_step59"


@dataclass
class CachedSample:
    """单帧的真实 BEV 特征与地图监督；不缓存原始相机图片。"""

    token: str
    scene: str
    bev: Any  # Tensor (1,256,150,150)
    gt: Any  # Tensor (1,200,400), long
    reference_logits: Any  # Tensor (1,4,200,400)


def parse_args() -> argparse.Namespace:
    """读取小样本训练参数；默认 15 次迭代、2 帧训练 + 2 帧留出测试。"""
    parser = argparse.ArgumentParser(description="Step59 冻结 Backbone 小样本监督训练")
    parser.add_argument("--author", type=Path, default=DEFAULT_AUTHOR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--val-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    if not (1 <= args.steps <= 100):
        parser.error("--steps 限制在 1 到 100；正式训练请另建配置")
    if not (0 < args.lr <= 1e-2):
        parser.error("--lr 必须介于 0 和 1e-2 之间")
    for name, indices in (("train", args.train_indices), ("val", args.val_indices)):
        if any(v < 0 for v in indices) or len(indices) != len(set(indices)):
            parser.error(f"{name} 索引必须非负且不重复")
    return args


def sha256(path: Path) -> str:
    """哈希权重、文本和结果文件，保证复现时可回溯。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack_first_tensor(value: Any, torch: Any) -> Any:
    """处理 MMDetection 的 DataContainer、list 和 tensor 嵌套。"""
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if hasattr(value, "data") and not isinstance(value, (str, bytes, dict)):
        return unpack_first_tensor(value.data, torch)
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return unpack_first_tensor(item, torch)
            except (TypeError, ValueError):
                pass
    raise TypeError(f"没有找到地图 GT 张量：{type(value)!r}")


def normalize_gt(data: dict[str, Any], torch: Any) -> Any:
    """校验作者生成的 semantic_indices，类别必须为 0、1、2、3。"""
    if "semantic_indices" not in data:
        raise KeyError("DataLoader 未生成 semantic_indices：停止，禁止使用伪造 GT")
    gt = unpack_first_tensor(data["semantic_indices"], torch)
    if gt.numel() != 200 * 400:
        raise ValueError(f"地图 GT 像素数异常：{tuple(gt.shape)}")
    gt = gt.reshape(1, 200, 400).cuda(non_blocking=True)
    if not torch.isfinite(gt.float()).all():
        raise ValueError("GT 含 NaN/Inf")
    if gt.min().item() < 0 or gt.max().item() > 3:
        raise ValueError(f"GT 类别超出 [0,3]：min={gt.min()}, max={gt.max()}")
    if not torch.equal(gt.float(), gt.long().float()):
        raise ValueError("GT 包含非整数类别")
    return gt.long().contiguous()


def to_spatial(bev_seq: Any, torch: Any) -> Any:
    """按照作者 BEV 查询顺序还原 (B,C,H,W)，不做任何空间变换。"""
    if tuple(bev_seq.shape) != (22500, 1, 256):
        raise ValueError(f"transformer 的 BEV 形状异常：{tuple(bev_seq.shape)}")
    batch = bev_seq.permute(1, 0, 2).reshape(1, 150, 150, 256)
    return batch.permute(0, 3, 1, 2).contiguous()


def replay_decoder(head: Any, bev: Any, torch: Any) -> Any:
    """重放作者 BEV 排列 → feat_cropper → seg_decoder 路径。

    严格和完整模型 logits 比较。如果本地 Head 空间实现不同，直接停止。
    """
    if tuple(bev.shape) != (1, 256, 150, 150):
        raise ValueError(f"缓存的 BEV 特征形状不正确：{tuple(bev.shape)}")
    # 作者当前 bevformer_head.py 使用 reshape(...).permute(2,3,1,0)，之后 flip dim=2。
    # 此处等价于在 (B,C,H,W) 形式上交换 H/W，再翻转新的 dim=2。
    seg_bev = torch.flip(bev.transpose(2, 3), dims=[2])
    logits = head.seg_decoder(head.feat_cropper(seg_bev))
    if tuple(logits.shape) != (1, 4, 200, 400):
        raise ValueError(f"作者分割 Decoder 输出异常：{tuple(logits.shape)}")
    return logits


def extract_logits(preds: Any, torch: Any) -> Any:
    """保留作者分割预测的原始数值，避免 argmax 后不能训练。"""
    if not isinstance(preds, (list, tuple)) or len(preds) != 1:
        raise TypeError("预期模型输出 1 帧的结果列表")
    obj = preds[0]
    if not isinstance(obj, dict) or obj.get("seg_preds") is None:
        raise KeyError("作者推理输出缺少 seg_preds")
    output = obj["seg_preds"]
    if isinstance(output, (list, tuple)) and len(output) == 1:
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"预测不是 Tensor：{type(output)}")
    if tuple(output.shape) != (1, 4, 200, 400):
        raise ValueError(f"分割输出形状异常：{tuple(output.shape)}")
    return output.detach().float().clone()


def build_mini_dataset(cfg: Any, build_dataset: Any, split: str) -> Any:
    """训练与验证统一用测试预处理，但分别读取 mini_train / mini_val PKL。"""
    dcfg = copy.deepcopy(cfg.data.test)
    dcfg.pop("samples_per_gpu", None)  # 这是 DataLoader 参数，不属于 Dataset。
    dcfg.test_mode = True
    dcfg.ann_file = (
        cfg.data.train.ann_file if split == "train" else cfg.data.val.ann_file
    )
    dataset = build_dataset(dcfg)
    target = 323 if split == "train" else 81
    if len(dataset) != target:
        raise RuntimeError(
            f"{split} 集帧数 {len(dataset)}，预期 Mini={target}。"
            "为避免错用完整 Trainval，停止运行。"
        )
    return dataset


def capture_samples(
    model: Any,
    transformer: Any,
    head: Any,
    dataset: Any,
    indices: list[int],
    build_dataloader: Any,
    sampler_cfg: Any,
    torch: Any,
    split: str,
) -> list[CachedSample]:
    """原始作者模型提取真实 BEV，校验 GT、Token 和重放一致性。"""
    if max(indices) >= len(dataset):
        raise IndexError(f"{split} 样本索引越界")
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False, nonshuffler_sampler=sampler_cfg,
    )
    result: list[CachedSample] = []
    selected = set(indices)
    max_index = max(indices)
    for idx, data in enumerate(loader):
        if idx > max_index:
            break
        if idx not in selected:
            continue
        token = str(dataset.data_infos[idx]["token"])
        scene = str(dataset.data_infos[idx]["scene_token"])
        gt = normalize_gt(data, torch)
        model.module.prev_frame_info = {
            "prev_bev": None, "scene_token": None,
            "prev_pos": 0, "prev_angle": 0,
        }
        captures: list[Any] = []

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            """只读捕获本次模型前向得到的真实 BEV，绝不替换输出。"""
            if not isinstance(output, (tuple, list)) or not torch.is_tensor(output[0]):
                raise TypeError("作者 transformer 输出格式异常")
            captures.append(output[0].detach().float().clone())

        handle = transformer.register_forward_hook(hook)
        model_data = dict(data)
        model_data.pop("semantic_indices", None)
        try:
            with torch.no_grad():
                prediction = model(
                    return_loss=False, rescale=True,
                    **copy.deepcopy(model_data),
                )
        finally:
            handle.remove()
        if len(captures) != 1:
            raise RuntimeError(f"{split} {idx} transformer 调用次数={len(captures)}")
        reference = extract_logits(prediction, torch)
        bev = to_spatial(captures[0], torch).detach().clone()
        with torch.no_grad():
            replay = replay_decoder(head, bev, torch).float()
        delta = float((replay - reference).abs().max().item())
        if not np.isfinite(delta) or delta > 1e-4:
            raise RuntimeError(
                f"{split} idx={idx} 原始模型与重放 Decoder 不一致："
                f" max_abs_diff={delta:.7g}，超过 1e-4；"
                "禁止训练！请提供输出排查空间排列。"
            )
        bincount = torch.bincount(gt.flatten(), minlength=4).cpu().tolist()
        print(
            f"[Step59] {split}[{idx}] token={token[:12]} "
            f"GT={bincount} replay_error={delta:.3g}",
            flush=True,
        )
        result.append(CachedSample(token, scene, bev, gt, reference))
        if len(result) == len(indices):
            break
    if len(result) != len(indices):
        raise RuntimeError(f"{split} 请求 {len(indices)} 帧，实际捕获 {len(result)} 帧")
    return result


def evaluate(
    fusion: Any,
    head: Any,
    samples: list[CachedSample],
    text: Any,
    torch: Any,
    enabled: bool = True,
) -> dict[str, Any]:
    """冻结模型上的受限样本损失，可明确区分原始和增强分支。"""
    fusion.eval()
    losses: list[float] = []
    pixel_correct = 0
    pixel_total = 0
    with torch.no_grad():
        for i, sample in enumerate(samples):
            prompt = text[i % text.shape[0]: i % text.shape[0] + 1]
            enhanced, _ = fusion(sample.bev, prompt, enabled=enabled)
            logits = replay_decoder(head, enhanced, torch)
            loss = head.loss_seg(logits.float(), sample.gt)
            if loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError("验证 Loss 异常")
            losses.append(float(loss.item()))
            pixel_correct += int((logits.argmax(1) == sample.gt).sum().item())
            pixel_total += sample.gt.numel()
    return {
        "mean_loss": float(np.mean(losses)),
        "per_frame_loss": losses,
        "pixel_accuracy_including_background": pixel_correct / pixel_total,
        "frame_count": len(samples),
    }


def print_memory(stage: str) -> None:
    """记录当前进程 RAM 与系统可用内存，定位意外 OOM 阶段。"""
    fields = {}
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith(("MemAvailable:", "SwapFree:")):
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    print(
        f"[Step59][内存] {stage}: "
        + ", ".join(f"{key}={value}" for key, value in fields.items()),
        flush=True,
    )


def main() -> None:
    """冻结主模型、复用已验证作者分割头，仅优化语言融合模块。"""
    args = parse_args()
    source = args.author.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    text_path = args.text.expanduser().resolve()
    config_path = source / "projects/configs/bevformer/bevformer_small_seg.py"
    for path in (source, checkpoint, text_path, config_path):
        if not path.exists():
            raise FileNotFoundError(f"必需输入不存在：{path}")
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

    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 GPU，不能继续作者模型真实特征训练")
    torch.manual_seed(20261021)
    np.random.seed(20261021)
    embed = np.load(text_path, allow_pickle=False)
    if embed.shape != (2, 768) or not np.isfinite(embed).all():
        raise ValueError(f"未找到 Step58 的两条真实 DistilBERT 向量：{embed.shape}")
    text = torch.from_numpy(embed).float().cuda()

    cfg = Config.fromfile(str(config_path))
    if (
        int(cfg.model.pts_bbox_head.bev_h) != 150
        or int(cfg.model.pts_bbox_head.bev_w) != 150
    ):
        raise ValueError("只支持当前 150×150 的 BEVFormer-Small")
    if not bool(cfg.model.pts_bbox_head.task.get("seg")):
        raise ValueError("当前模型未启用地图分割")
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    cfg.data.workers_per_gpu = 0
    print_memory("启动配置完成")
    start = time.perf_counter()

    # 先加载一次冻结模型，再逐个构建 Train/Val Dataset。
    # CustomNuScenesDataset 初始化会读 nuScenes 元数据与地图，禁止同时持有两份。
    print("[Step59] 正在构建冻结的作者 BEVFormer 模型...", flush=True)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(checkpoint), map_location="cpu")
    model.requires_grad_(False)
    model.eval()
    model = MMDataParallel(model.cuda(), device_ids=[0])
    head = model.module.pts_bbox_head
    transformer = head.transformer
    print("[Step59] GPU:", torch.cuda.get_device_name(0), flush=True)
    print("[Step59] 作者 BEVFormer 和文本向量冻结，仅训练融合层", flush=True)
    print_memory("作者模型加载完成")

    # Mini Train 使用真实 Map GT，只在数据提取期间保留数据集对象。
    print("[Step59] 正在加载 Mini Train Dataset...", flush=True)
    train_dataset = build_mini_dataset(cfg, build_dataset, "train")
    train_scenes = {str(item["scene_token"]) for item in train_dataset.data_infos}
    print("[Step59] Mini Train 帧数:", len(train_dataset), flush=True)
    print_memory("Mini Train Dataset 加载完成")
    train_samples = capture_samples(
        model, transformer, head, train_dataset, args.train_indices,
        build_dataloader, cfg.data.nonshuffler_sampler, torch, "mini_train",
    )
    del train_dataset
    gc.collect()
    print_memory("Train 特征提取完成、释放 Dataset")

    # Mini Val 与 Train 严格分开构建，减少双地图对象同时驻留的 RAM 峰值。
    print("[Step59] 正在加载 Mini Val Dataset...", flush=True)
    val_dataset = build_mini_dataset(cfg, build_dataset, "val")
    val_scenes = {str(item["scene_token"]) for item in val_dataset.data_infos}
    if train_scenes & val_scenes:
        raise RuntimeError("Mini Train 与 Mini Val scene_token 重叠，拒绝训练")
    print("[Step59] Mini Val 帧数:", len(val_dataset), flush=True)
    print_memory("Mini Val Dataset 加载完成")
    val_samples = capture_samples(
        model, transformer, head, val_dataset, args.val_indices,
        build_dataloader, cfg.data.nonshuffler_sampler, torch, "mini_val",
    )
    del val_dataset
    del train_scenes, val_scenes
    gc.collect()
    print_memory("Val 特征提取完成、释放 Dataset")
    fusion = LanguageBEVFusion(text_dim=768, alpha_init=0.01).cuda().float()
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=args.lr, weight_decay=1e-3)
    if sum(p.numel() for p in fusion.parameters() if p.requires_grad) == 0:
        raise RuntimeError("融合层没有可训练参数")

    original_baseline = evaluate(
        fusion, head, val_samples, text, torch, enabled=False
    )
    baseline = evaluate(fusion, head, val_samples, text, torch)
    history: list[dict[str, float | int]] = []
    for step in range(args.steps):
        sample = train_samples[step % len(train_samples)]
        prompt = text[step % text.shape[0]: step % text.shape[0] + 1]
        fusion.train()
        optimizer.zero_grad(set_to_none=True)
        enhanced, attention = fusion(sample.bev, prompt, enabled=True)
        logits = replay_decoder(head, enhanced, torch)
        seg_loss = head.loss_seg(logits.float(), sample.gt)
        # alpha 轻度约束，限制短训中无监督空间门控的幅度漂移。
        loss = seg_loss + 0.01 * fusion.alpha.square()
        if not torch.isfinite(loss):
            raise RuntimeError(f"step={step + 1} 的训练 loss 非有限值")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0))
        if not np.isfinite(grad_norm) or grad_norm <= 0:
            raise RuntimeError(f"step={step + 1} 语言融合层梯度异常：{grad_norm}")
        optimizer.step()
        with torch.no_grad():
            fusion.alpha.clamp_(-0.3, 0.3)
        item = {
            "step": step + 1,
            "mini_train_index": args.train_indices[step % len(train_samples)],
            "seg_loss": round(float(seg_loss.item()), 7),
            "total_loss": round(float(loss.item()), 7),
            "gradient_norm_before_clip": round(grad_norm, 7),
            "alpha": round(float(fusion.alpha.item()), 8),
            "attention_mean": round(float(attention.mean().item()), 7),
        }
        history.append(item)
        print(
            f"[Step59] {step + 1:02d}/{args.steps} "
            f"seg_loss={item['seg_loss']:.6f} grad={grad_norm:.4g} "
            f"alpha={item['alpha']:+.5f}",
            flush=True,
        )

    after = evaluate(fusion, head, val_samples, text, torch)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    weights_path = output / "fusion_step59.pt"
    torch.save({
        "state_dict": fusion.cpu().state_dict(),
        "text_dim": 768,
        "bev_channels": 256,
        "steps": args.steps,
        "note": "仅语言投影和融合层，作者 BEVFormer/DistilBERT 权重均未写入",
    }, weights_path)
    # 记录训练集重复使用情况，避免误把迭代数当成独立样本数量。
    result = {
        "experiment": "step59_frozen_bevformer_real_gt_fusion_training",
        "training_type": "frozen real BEV / frozen decoder / real GT",
        "author_source": str(source),
        "author_checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "text_embedding_path": str(text_path),
        "text_embedding_sha256": sha256(text_path),
        "dataset": {"train": "nuScenes v1.0-mini mini_train", "val": "mini_val"},
        "mini_train_index": args.train_indices,
        "mini_val_index": args.val_indices,
        "train_tokens": [x.token for x in train_samples],
        "val_tokens": [x.token for x in val_samples],
        "train_distinct_frames": len(train_samples),
        "val_distinct_frames": len(val_samples),
        "optimizer_steps": args.steps,
        "frozen_modules": ["BEVFormer", "segmentation decoder", "DistilBERT"],
        "trainable_module": "LanguageBEVFusion only",
        "loss": "author segmentation cross entropy + 0.01 * alpha^2",
        "history": history,
        "heldout_original_frozen_author": original_baseline,
        "heldout_before": baseline,
        "heldout_after": after,
        "heldout_loss_delta": after["mean_loss"] - baseline["mean_loss"],
        "weights": str(weights_path),
        "weights_sha256": sha256(weights_path),
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "official_miou": False,
        "full_mini_val_81": False,
        "temporally_consistent": False,
        "language_grounding_supervised": False,
        "limitations": (
            "仅两条冻结驾驶文本特征；小样本重复训练以验证梯度与数据监督。"
            "语义文本与道路 GT 未严格配对，不证明定位能力。"
            "Mini Train/Val 分离但评估样本很少，不代表官方 81 帧时序或指标提升。"
        ),
    }
    report = output / "step59_training_summary.json"
    report.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    md = [
        "# Step59：冻结真实 BEVFormer 的语言融合小样本监督训练", "",
        f"- 训练独立帧：{len(train_samples)}；Mini Val 留出独立帧：{len(val_samples)}。",
        f"- 优化器迭代：{args.steps}，重复使用训练帧；不开启时序。",
        f"- 留出样本原始模型 Loss：{original_baseline['mean_loss']:.6f}。",
        f"- 加入融合层前/后训练的留出 Loss：{baseline['mean_loss']:.6f} → {after['mean_loss']:.6f}。",
        (
            f"- 训练 Loss（首轮/末轮）：{history[0]['seg_loss']:.6f} / "
            f"{history[-1]['seg_loss']:.6f}。"
        ),
        "", "## 实验限制", "",
        "- 冻结作者原始 BEVFormer、作者分割 Decoder 和 DistilBERT，仅训练 LanguageBEVFusion。",
        "- GT 来自真实 nuScenes Map，训练和验证场景分开；但验证仅极少量帧。",
        "- 两条提示词没有逐像素语言监督，无法证明正确的语言目标定位。",
        "- 这些 Loss 不可替代作者完整 81 帧时序前景 mIoU，也不是正式模型性能结论。",
        "- 训练前对每个真实样本都进行了 Decoder 重放一致性检查。", "",
    ]
    (output / "step59_review.md").write_text("\n".join(md), encoding="utf-8")
    print("[Step59] 作者原始留出样本 Loss：", f"{original_baseline['mean_loss']:.6f}")
    print(
        "[Step59] 融合层训练前/后留出样本 Loss：",
        f"{baseline['mean_loss']:.6f} -> {after['mean_loss']:.6f}",
    )
    print("[Step59] 小样本训练结束，报告：", report)
    print("[Step59] 融合权重：", weights_path)
    print("[Step59] 非正式 Mini Val 指标，尚未完成语言空间对齐监督", flush=True)


if __name__ == "__main__":
    main()

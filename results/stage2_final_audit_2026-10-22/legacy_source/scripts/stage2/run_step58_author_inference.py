"""Step58 第二阶段：运行时接入作者 BEVFormer 分割推理链路。

不修改第三方源码、不写入 checkpoint，不将单帧结果冒充 Mini Val mIoU。
在 transformer 输出 BEV 和分割头之间插入语言条件残差融合。
三次对照：原始 / 注入但关闭 / 注入且开启（独立帧，时序关闭）。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTHOR = Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg"
DEFAULT_CHECKPOINT = Path("/mnt/g/nus/bevformer_small_seg.pth")
DEFAULT_OUT = ROOT / "outputs/stage2_step58"


def file_sha256(path: Path) -> str:
    """为实验输入生成可靠的文件哈希。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    """读取作者隔离工程与真实文本特征路径。"""
    parser = argparse.ArgumentParser(description="Step58 真正模型链路融合 A/B")
    parser.add_argument("--author", type=Path, default=DEFAULT_AUTHOR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


def get_segmentation_tensor(result: Any, torch: Any) -> Any:
    """只提取作者真实分割结果，不擅自转换或重排类别。"""
    if not isinstance(result, (list, tuple)) or len(result) != 1:
        raise TypeError(f"作者返回应为单样本列表：{type(result)}")
    obj = result[0]
    if not isinstance(obj, dict) or obj.get("seg_preds") is None:
        raise ValueError("作者模型没有返回 seg_preds")
    pred = obj["seg_preds"]
    if torch.is_tensor(pred):
        return pred.detach().float().cpu().clone()
    if isinstance(pred, (list, tuple)) and len(pred) == 1 and torch.is_tensor(pred[0]):
        return pred[0].detach().float().cpu().clone()
    if isinstance(pred, np.ndarray):
        return torch.from_numpy(pred).float().clone()
    raise TypeError(f"无法对比当前分割结果：{type(pred)}")


def main() -> None:
    """真实单帧 BEVFormer 推理：baseline、回退、语言增强。"""
    args = parse_args()
    source = args.author.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = source / "projects/configs/bevformer/bevformer_small_seg.py"
    text_path = args.out.expanduser().resolve() / "distilbert_prompts.npy"
    for path in (source, checkpoint, config_path, text_path):
        if not path.exists():
            raise FileNotFoundError(f"必需输入不存在：{path}")
    if args.index < 0:
        raise ValueError("--index 必须是非负数")

    # 外部项目与本项目分别通过固定路径导入，不写入第三方工程。
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
        raise RuntimeError("独立作者环境 CUDA 不可用")
    pooled_array = np.load(text_path, allow_pickle=False)
    if pooled_array.shape != (2, 768) or not np.isfinite(pooled_array).all():
        raise ValueError(f"缓存真实文本向量错误：{pooled_array.shape}")

    cfg = Config.fromfile(str(config_path))
    if (
        int(cfg.model.pts_bbox_head.bev_h) != 150
        or int(cfg.model.pts_bbox_head.bev_w) != 150
    ):
        raise ValueError("当前代码只适用于 150×150 BEV")
    if not cfg.model.pts_bbox_head.task.get("seg"):
        raise ValueError("当前模型不是分割任务")
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False  # 三组采用相同独立帧，禁止混入时序记忆。
    cfg.data.test.test_mode = True
    cfg.data.workers_per_gpu = 0
    cfg.data.test.pop("samples_per_gpu", None)
    dataset = build_dataset(cfg.data.test)
    if len(dataset) != 81:
        raise ValueError(f"要求使用 Mini Val 81 帧，而非 {len(dataset)} 帧")
    if args.index >= len(dataset):
        raise IndexError(f"帧索引 {args.index} 超过 {len(dataset)}")

    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    iterator = iter(loader)
    data = next(iterator)
    for _ in range(args.index):
        data = next(iterator)
    token = str(dataset.data_infos[args.index]["token"])
    data.pop("semantic_indices", None)  # GT不传入推理模型

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    metadata = load_checkpoint(model, str(checkpoint), map_location="cpu")
    model.CLASSES = metadata.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model.eval()
    model = MMDataParallel(model.cuda(), device_ids=[0])
    base = model.module
    head = base.pts_bbox_head
    transformer = head.transformer
    # 尚未训练的融合头保持随机初始化；固定随机种子以保证复现。
    torch.manual_seed(20261021)
    fusion = LanguageBEVFusion(text_dim=768, alpha_init=0.01).cuda().float().eval()
    pooled = torch.from_numpy(pooled_array).cuda().float()
    state: dict[str, Any] = {
        "mode": "bypass",
        "calls": 0,
        "maps": [],
        "bev_shape": None,
        "prompt_index": 0,
    }

    def transformer_output_hook(_module: Any, _inputs: Any, output: Any) -> Any:
        """只替换 BEV 张量，保持作者 Seg Decoder 和其他分支不变。"""
        if not isinstance(output, (tuple, list)) or not torch.is_tensor(output[0]):
            raise TypeError("无法定位作者 transformer.forward 输出的 BEV")
        original = output[0]
        if original.ndim != 3 or tuple(original.shape) != (22500, 1, 256):
            raise ValueError(f"意外的作者 BEV 特征形状：{tuple(original.shape)}")
        state["calls"] += 1
        state["bev_shape"] = list(original.shape)
        if state["mode"] == "bypass":
            return output  # 完全保留原对象，检查关闭功能时不发生更改。
        if state["mode"] != "enhanced":
            raise ValueError(f"非法模式：{state['mode']}")
        # 作者 reshape 约定：(H*W,B,C) -> (B,C,H,W)。
        spatial = original.permute(1, 0, 2).reshape(1, 150, 150, 256)
        spatial = spatial.permute(0, 3, 1, 2).float()
        enhanced, attention = fusion(
            spatial, pooled[state["prompt_index"]:state["prompt_index"] + 1],
            enabled=True,
        )
        state["maps"].append((
            int(state["prompt_index"]),
            attention[0, 0].detach().float().cpu().numpy(),
        ))
        restored = enhanced.to(original.dtype).permute(0, 2, 3, 1)
        restored = restored.reshape(1, 22500, 256).permute(1, 0, 2).contiguous()
        if not torch.isfinite(restored).all():
            raise ValueError("增强 BEV 包含 NaN/Inf")
        return (restored, *output[1:])

    def reset_state() -> None:
        """所有分支都独立清零 prev_bev 与 ego motion 状态。"""
        base.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }

    def infer(label: str, injected: bool) -> Any:
        """对同一真实样本运行作者原始 forward_test。"""
        reset_state()
        state["calls"] = 0
        if not injected:
            with torch.inference_mode():
                result = model(return_loss=False, rescale=True, **copy.deepcopy(data))
        else:
            handle = transformer.register_forward_hook(transformer_output_hook)
            try:
                with torch.inference_mode():
                    result = model(
                        return_loss=False, rescale=True, **copy.deepcopy(data)
                    )
            finally:
                handle.remove()
            if state["calls"] != 1:
                raise RuntimeError(f"{label} 预期一次 transformer 调用，实测 {state['calls']}")
        pred = get_segmentation_tensor(result, torch)
        if not torch.isfinite(pred).all():
            raise ValueError(f"{label} 分割输出出现 NaN/Inf")
        print(f"[Step58] {label}: pred_shape={list(pred.shape)}", flush=True)
        return pred

    print("[Step58] GPU:", torch.cuda.get_device_name(0), flush=True)
    print("[Step58] sample_token:", token, flush=True)
    print("[Step58] 真正作者模型，固定权重，关闭时序、单帧 A/B", flush=True)
    start = time.perf_counter()
    original_pred = infer("original", injected=False)
    state["mode"] = "bypass"
    bypass_pred = infer("bypass", injected=True)
    state["mode"] = "enhanced"
    state["prompt_index"] = 0
    enhanced_a = infer("enhanced_prompt_a", injected=True)
    state["prompt_index"] = 1
    enhanced_b = infer("enhanced_prompt_b", injected=True)
    if original_pred.shape != enhanced_a.shape or enhanced_a.shape != enhanced_b.shape:
        raise ValueError("各分支输出形状不一致")
    baseline_delta = (original_pred - bypass_pred).abs()
    enhanced_delta = (original_pred - enhanced_a).abs()
    prompt_delta = (enhanced_a - enhanced_b).abs()
    max_bypass = float(baseline_delta.max().item())
    mean_enhanced = float(enhanced_delta.mean().item())
    mean_prompt_delta = float(prompt_delta.mean().item())
    print(f"[Step58] 关闭模块与原始模型最大差异：{max_bypass:.8g}")
    print(f"[Step58] 增强后原始输出平均绝对差：{mean_enhanced:.8g}")
    print(f"[Step58] 不同驾驶指令预测平均绝对差：{mean_prompt_delta:.8g}")
    # CPU 混合精度/非确定性算子允许极小偏差，但应记录而非隐瞒。
    bypass_same = max_bypass <= 1e-4
    if not bypass_same:
        print("[Step58] 警告：bypass 不一致，需先排查确定性/输入状态！")
    if len(state["maps"]) != 2 or [x[0] for x in state["maps"]] != [0, 1]:
        raise RuntimeError("真实模型链路没有按两条指令各生成一张注意力图")
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "attention_live_prompt_a.npy", state["maps"][0][1])
    np.save(args.out / "attention_live_prompt_b.npy", state["maps"][1][1])
    attention_delta = float(np.abs(state["maps"][0][1] - state["maps"][1][1]).mean())
    report = {
        "experiment": "step58_live_author_bevformer_seg_inference_hook",
        "sample_index": args.index,
        "sample_token": token,
        "model_checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "author_config": str(config_path),
        "text_manifest": str(args.out / "text_manifest.json"),
        "real_pretrained_text_embeddings": True,
        "architecture_insertion": (
            "transformer.forward BEV output before segmentation head"
        ),
        "temporal_mode": "OFF (four isolated single-frame passes)",
        "bev_shape": state["bev_shape"],
        "segmentation_output_shape": list(original_pred.shape),
        "bypass_max_absolute_difference": max_bypass,
        "bypass_equivalent_tolerance_1e-4": bypass_same,
        "enhanced_output_mean_absolute_difference": mean_enhanced,
        "two_prompt_prediction_mean_absolute_difference": mean_prompt_delta,
        "two_prompt_attention_mean_absolute_difference": attention_delta,
        "attention_shape": list(state["maps"][0][1].shape),
        "alpha_init": 0.01,
        "fusion_weights_trained": False,
        "full_81_frame_temporal_eval": False,
        "miou_evaluated": False,
        "note": (
            "实际作者模型运行时 BEV 插入实验。输入文本来自真实 DistilBERT，"
            "融合头随机初始化；本实验不具备监督训练或 IoU 性能提升含义。"
        ),
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }
    target = args.out / "step58_author_live_smoke.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(f"[Step58] 两条指令 Attention MAE: {attention_delta:.8g}")
    print("[Step58] 融合链路执行完成，报告：", target)
    if bypass_same:
        print("[Step58] Baseline 回退：PASS（容差 1e-4）")
    else:
        raise RuntimeError("Baseline 回退未通过；已保留诊断结果，不能进入增强评估")


if __name__ == "__main__":
    main()

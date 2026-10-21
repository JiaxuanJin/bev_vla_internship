#!/usr/bin/env python3
"""Step64 报告恢复：从已保存的训练权重重建验证结果，不再次训练。

要求先完成 Step64 72 次训练，且保留报错前生成的 fusion_step61.pt。
恢复后的报告明确标注原始逐步 history 未保存，不补造训练日志。
同时对原 Step64 脚本进行有备份、精确匹配的缺失变量修复。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUTHOR = Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg"
MODEL_WEIGHT = Path("/mnt/g/nus/bevformer_small_seg.pth")
OUTPUT = ROOT / "outputs/stage2_step64"
PROMPTS = ROOT / "outputs/stage2_step61/class_prompts.npy"
REFERENCE = ROOT / "outputs/stage2_step61/step61_supervised_summary.json"
DIAGNOSTIC = ROOT / "outputs/stage2_step62/step62_diagnostics.json"
TRAIN_SCRIPT = ROOT / "scripts/stage2/run_step64_balanced_train.py"
CLASSES = ("Divider", "Ped Crossing", "Boundary")


def digest(path: Path) -> str:
    """计算文件 SHA256。"""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def apply_future_fix(code: str) -> str:
    """为将来的 Step64 新实验补充 AP 计算并纠正权重文件名。"""
    before = (
        '    after = evaluate(fusion, head, val, text, torch)\n'
        '    weights = args.output / "fusion_step61.pt"\n'
    )
    after = (
        '    after = evaluate(fusion, head, val, text, torch)\n'
        '    after_ap = attention_ap_by_class(fusion, head, val, text, torch)\n'
        '    weights = args.output / "fusion_step64.pt"\n'
    )
    if code.count(before) == 1:
        return code.replace(before, after, 1)
    if code.count(after) == 1:
        return code
    raise RuntimeError(
        "Step64 代码不符合已验证的原始版本，拒绝自动修改。"
        "请先检查第 562～565 行。"
    )


def patch_script() -> None:
    """备份后精确修复未来训练入口，不触发训练。"""
    code = TRAIN_SCRIPT.read_text(encoding="utf-8")
    corrected = apply_future_fix(code)
    compile(corrected, str(TRAIN_SCRIPT), "exec")
    if corrected == code:
        print("[Step64恢复] 未来训练入口此前已修正。", flush=True)
        return
    backup = OUTPUT / "run_step64_before_report_fix.py"
    if backup.exists():
        if backup.read_text(encoding="utf-8") != code:
            raise FileExistsError("已有不同的备份，停止以避免覆盖：" + str(backup))
    else:
        backup.write_text(code, encoding="utf-8")
    TRAIN_SCRIPT.write_text(corrected, encoding="utf-8")
    print(f"[Step64恢复] 代码已备份：{backup}", flush=True)
    print("[Step64恢复] 已补算 after_ap，并修正未来权重文件名。", flush=True)


def output_table(original: dict, enhanced: dict, ap: dict, reference: dict) -> list[str]:
    """组装指标表格；不使用丢失的训练前初始化指标。"""
    lines = [
        "| 指令 | 原始分割前景 mIoU | Step64 分割前景 mIoU | "
        "Step61 Attention AP | Step64 Attention AP | GT 占比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in CLASSES:
        o = original["per_prompt"][name]["segmentation"]["foreground_miou_argmax"]
        e = enhanced["per_prompt"][name]["segmentation"]["foreground_miou_argmax"]
        s = reference[name]
        a = ap[name]["average_precision"]
        prevalence = ap[name]["positive_prevalence"]
        lines.append(
            f"| {name} | {o:.4%} | {e:.4%} | {s:.4%} | "
            f"{a:.4%} | {prevalence:.4%} |"
        )
    return lines


def self_test() -> None:
    """不加载模型，仅测试精确代码补丁与输出结构。"""
    snippet = (
        '    after = evaluate(fusion, head, val, text, torch)\n'
        '    weights = args.output / "fusion_step61.pt"\n'
    )
    fixed = apply_future_fix(snippet)
    assert "after_ap = attention_ap_by_class" in fixed
    assert 'weights = args.output / "fusion_step64.pt"' in fixed
    assert apply_future_fix(fixed) == fixed
    compile("def inspect():\n" + fixed, "<patch-test>", "exec")
    print("[Step64恢复] 代码补丁自检 PASS（不包含作者模型运行）")


def main() -> None:
    """重新读取 Val 样本、加载训练权重并计算同条件指标。"""
    parser = argparse.ArgumentParser(description="Step64 报错后免训练恢复报告")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    required = [
        AUTHOR, MODEL_WEIGHT, PROMPTS, REFERENCE, DIAGNOSTIC, TRAIN_SCRIPT,
        OUTPUT / "fusion_step61.pt",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"缺少恢复依赖：{path}")
    old_weight = OUTPUT / "fusion_step61.pt"
    new_weight = OUTPUT / "fusion_step64.pt"
    json_path = OUTPUT / "step64_recovered_summary.json"
    md_path = OUTPUT / "step64_recovered_review.md"
    for file in (json_path, md_path):
        if file.exists():
            raise FileExistsError(f"恢复报告已存在，禁止覆盖：{file}")
    if new_weight.exists() and digest(new_weight) != digest(old_weight):
        raise RuntimeError("新旧 Step64 权重不一致，拒绝覆盖。")

    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    # 作者仓库的项目注册依赖其自身当前工作目录。
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(AUTHOR))
    os.chdir(AUTHOR)
    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    from scripts.stage2.bev_fusion import LanguageBEVFusion
    from scripts.stage2.run_step59_train import build_mini_dataset, capture_samples
    from scripts.stage2.run_step60_extended_ab import IndexedDataset, move_to_cpu
    from scripts.stage2.run_step64_balanced_train import (
        attention_ap_by_class,
        evaluate,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("当前作者环境没有 CUDA，不能执行真实验证")
    step61 = json.loads(REFERENCE.read_text(encoding="utf-8"))
    step62 = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    if step61.get("val_indices") != step62.get("val_indices"):
        raise RuntimeError("Step61 与 Step62 的留出帧不一致")
    if step62.get("checkpoint_sha256") != digest(MODEL_WEIGHT):
        raise RuntimeError("BEVFormer checkpoint 与 Step62 不一致")
    if step62.get("text_sha256") != digest(PROMPTS):
        raise RuntimeError("文本特征与 Step62 不一致")

    data = torch.load(str(old_weight), map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "state_dict" not in data:
        raise RuntimeError("已保存的 Step64 权重格式异常")
    if int(data.get("steps", -1)) != 72 or int(data.get("train_frames", -1)) != 16:
        raise RuntimeError("本恢复脚本仅针对当前 16 帧、72 步实验")
    if abs(float(data.get("alpha_fixed", -1)) - 0.02) > 1e-9:
        raise RuntimeError("权重 alpha 不符合原 Step64 设置")

    config_path = AUTHOR / "projects/configs/bevformer/bevformer_small_seg.py"
    cfg = Config.fromfile(str(config_path))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    cfg.data.workers_per_gpu = 0
    started = time.perf_counter()
    print("[Step64恢复] 构建冻结 BEVFormer（不启动训练）...", flush=True)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(MODEL_WEIGHT), map_location="cpu")
    model.requires_grad_(False)
    model.eval()
    model = MMDataParallel(model.cuda(), device_ids=[0])
    head = model.module.pts_bbox_head
    transformer = head.transformer

    dataset = build_mini_dataset(cfg, build_dataset, "val")
    val_indices = step61["val_indices"]
    if len(val_indices) != 6 or len(set(val_indices)) != 6:
        raise RuntimeError("Step61 验证帧索引不符合预期")
    selected = IndexedDataset(dataset, val_indices)
    samples = capture_samples(
        model, transformer, head, selected, list(range(len(selected))),
        build_dataloader, cfg.data.nonshuffler_sampler, torch, "mini_val",
    )
    samples = move_to_cpu(samples)
    if [s.token for s in samples] != step61["val_tokens"]:
        raise RuntimeError("当前 Val token 与 Step61 不一致，停止恢复")
    del dataset, selected
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Step64恢复] 已缓存验证帧：{len(samples)}", flush=True)

    text_array = np.load(PROMPTS, allow_pickle=False)
    if text_array.shape != (3, 768) or not np.isfinite(text_array).all():
        raise RuntimeError("DistilBERT 文本向量形状或数值异常")
    text = torch.from_numpy(text_array).float().cuda()
    fusion = LanguageBEVFusion(text_dim=768, alpha_init=0.02).float().cuda()
    fusion.load_state_dict(data["state_dict"], strict=True)
    fusion.eval()
    for param in fusion.parameters():
        param.requires_grad_(False)

    print("[Step64恢复] 计算原始与 Step64 增强模型分割指标及 AP...", flush=True)
    original = evaluate(fusion, head, samples, text, torch, original=True)
    enhanced = evaluate(fusion, head, samples, text, torch)
    after_ap = attention_ap_by_class(fusion, head, samples, text, torch)
    ref_ap = {
        name: step62["scores"][name]["average_precision"] for name in CLASSES
    }

    if not new_weight.exists():
        shutil.copy2(old_weight, new_weight)
    if digest(new_weight) != digest(old_weight):
        raise RuntimeError("规范化文件名后权重哈希改变")
    result: dict[str, Any] = {
        "experiment": "step64_balanced_hard_negative_report_recovery",
        "recovery_mode": "frozen trained fusion weights; validation only; no training",
        "training_completed_before_crash": True,
        "training_history_available": False,
        "training_history_note": (
            "72 次迭代完成但进程报错时历史尚未写入 JSON；"
            "权重仍已保存，本文件只重新计算验证指标，不编造 history。"
        ),
        "train_indices_reference_step61": step61["train_indices"],
        "val_indices": val_indices,
        "val_tokens": [s.token for s in samples],
        "val_frames": len(samples),
        "optimizer_steps_recorded_in_weights": int(data["steps"]),
        "alpha_fixed": float(data["alpha_fixed"]),
        "checkpoint_sha256": digest(MODEL_WEIGHT),
        "text_sha256": digest(PROMPTS),
        "step64_weight_sha256": digest(new_weight),
        "step64_weight_path": str(new_weight),
        "legacy_saved_weight_path": str(old_weight),
        "original_segmentation": original,
        "after": enhanced,
        "attention_ap_after": after_ap,
        "step61_attention_ap": ref_ap,
        "temporal": False,
        "full_81_frame_eval": False,
        "eval_set_previously_used_for_development": True,
        "notes": [
            "Mini Val 仅 6 帧单帧推理，不等于 81 帧时序成绩",
            "Attention AP 来自阈值无关排名指标；IoU@0.5 仍需单独检查",
            "GT 和文本指令是类别模板配对，不代表开放词表定位",
            "训练前随机融合模型的 Step64 指标未持久化，不能伪造",
        ],
        "recovery_seconds": round(time.perf_counter() - started, 2),
    }
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Step64：已训练权重恢复验证报告", "",
        "Step64 已完成 72 次更新；原报告在引用未定义 after_ap 时出错。",
        "本次直接加载崩溃前保存的融合权重，只重新评估，不重新训练。", "",
        *output_table(original, enhanced, after_ap, ref_ap), "",
        "## 结果限制", "",
        "- 训练过程逐步 history 未落盘，本恢复报告不包含该历史。",
        "- 开发期反复使用的 Mini Val 6 帧，不能视作新的独立盲测。",
        "- 时序关闭，不与 81 帧时序 mIoU 直接比较。",
        "- GT 类别模板不等于开放词表语言定位。", "",
        f"融合权重 SHA256：`{digest(new_weight)}`。", "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    patch_script()
    print("[Step64恢复] 完成，同条件验证数据：", json_path, flush=True)
    print("[Step64恢复] 中文报告：", md_path, flush=True)
    print("[Step64恢复] Step64 权重：", new_weight, flush=True)
    print("[Step64恢复] 结果汇总：", flush=True)
    for row in output_table(original, enhanced, after_ap, ref_ap):
        print(row, flush=True)
    print("[Step64恢复] 原训练历史未保存，未虚构数据。", flush=True)


if __name__ == "__main__":
    main()

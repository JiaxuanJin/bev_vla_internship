"""Step57：冻结真实预训练 DistilBERT，验证双文本对 BEV 门控的影响。

在工程根目录运行：
    python3 -m scripts.stage2.run_step57_distilbert

说明：这是模块级 smoke test，并非检测/分割端到端训练或 mIoU 评估。
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .bev_fusion import LanguageBEVFusion
from .language_encoder import DistilBERTLanguageEncoder
from .run_step56_smoke import load_bev_feature

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BEV = PROJECT_ROOT / "outputs/bev_features/bev_feature.npy"
DEFAULT_OUT = PROJECT_ROOT / "outputs/stage2_step57"
DEFAULT_PROMPTS = (
    "Pay attention to pedestrians in the front right.",
    "Pay attention to vehicles behind the car.",
)


def file_sha256(path: Path) -> str:
    """流式计算输入特征文件的 SHA256，以固定后续 A/B 的来源。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """加载两条真实文本和同一 BEV，验证输出差异、回退与梯度。"""
    parser = argparse.ArgumentParser(description="Step57 真实 DistilBERT + BEV")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_BEV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if not args.feature_path.is_file():
        raise FileNotFoundError(
            f"真实 BEV 特征不存在：{args.feature_path}；"
            "请先检查 Step56 的真实特征路径"
        )
    try:
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "缺少 transformers；可先执行："
            "python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
            "'transformers>=4.46,<5'"
        ) from exc

    torch.manual_seed(20261021)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20261021)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bev, source = load_bev_feature(args.feature_path, "real", device)
    # 两条不同指令使用完全相同的 BEV 特征，隔离语言变量。
    same_bev_twice = bev.expand(2, -1, -1, -1).contiguous()

    encoder = DistilBERTLanguageEncoder(
        model_name=args.model,
        freeze=True,
        local_files_only=args.local_files_only,
    ).to(device)
    encoder.eval()
    pooled = encoder(list(DEFAULT_PROMPTS), str(device)).detach()
    if pooled.ndim != 2 or pooled.shape[0] != 2:
        raise AssertionError(f"真实语言向量维度异常：{list(pooled.shape)}")
    if not torch.isfinite(pooled).all():
        raise AssertionError("DistilBERT 输出包含 NaN/Inf")

    fusion = LanguageBEVFusion(text_dim=pooled.shape[-1], alpha_init=0.01).to(device)
    fusion.train()
    baseline, _ = fusion(same_bev_twice, pooled, enabled=False)
    if not torch.equal(baseline, same_bev_twice):
        raise AssertionError("Baseline 回退必须逐元素精确一致")

    enhanced, attention = fusion(same_bev_twice, pooled, enabled=True)
    if enhanced.shape != (2, 256, 150, 150):
        raise AssertionError(f"增强 BEV 形状异常：{list(enhanced.shape)}")
    if attention.shape != (2, 1, 150, 150):
        raise AssertionError(f"注意力形状异常：{list(attention.shape)}")
    if not torch.isfinite(enhanced).all() or not torch.isfinite(attention).all():
        raise AssertionError("输出包含 NaN/Inf")

    attention_delta = (attention[0] - attention[1]).abs().mean().item()
    pooled_delta = (pooled[0] - pooled[1]).abs().mean().item()
    # 不做语义定位正确性的断言，未训练的映射头尚没有这方面保证。
    if pooled_delta <= 1e-8 or attention_delta <= 1e-8:
        raise AssertionError("改变指令后语言向量或注意力没有可检测的变化")

    # 仅为检查融合模块是否可反向传播；不是分割任务损失。
    smoke_loss = enhanced.square().mean()
    smoke_loss.backward()
    gradient_norms = {}
    for name, parameter in fusion.named_parameters():
        grad = parameter.grad
        value = None if grad is None else float(grad.norm().item())
        if value is None or not np.isfinite(value) or value <= 0.0:
            raise AssertionError(f"融合模块梯度无效：{name}={value}")
        gradient_norms[name] = value
    if any(parameter.grad is not None for parameter in encoder.parameters()):
        raise AssertionError("预期冻结的 DistilBERT 仍然产生了梯度")

    args.out.mkdir(parents=True, exist_ok=True)
    maps = attention.detach().cpu().float().numpy()[:, 0]
    np.save(args.out / "attention_prompt_a.npy", maps[0])
    np.save(args.out / "attention_prompt_b.npy", maps[1])
    result = {
        "experiment": "step57_real_distilbert_frozen_text_bev_smoke",
        "model_name_or_path": args.model,
        "model_pretrained_loaded": True,
        "language_model_frozen": True,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "device": str(device),
        "prompts": list(DEFAULT_PROMPTS),
        "bev_source": source,
        "bev_path": str(args.feature_path),
        "bev_sha256": file_sha256(args.feature_path),
        "bev_shape": list(same_bev_twice.shape),
        "pooled_shape": list(pooled.shape),
        "attention_shape": list(attention.shape),
        "enhanced_shape": list(enhanced.shape),
        "pooled_text_mean_absolute_difference": pooled_delta,
        "attention_mean_absolute_difference": attention_delta,
        "baseline_exact_identity": True,
        "fusion_gradient_norms": gradient_norms,
        "smoke_test_loss": float(smoke_loss.detach().item()),
        "task_supervision_used": False,
        "training_completed": False,
        "segmentation_evaluation_completed": False,
        "note": (
            "真实 DistilBERT 已加载并冻结；仅验证语义编码可改变未训练注意力输出，"
            "不能说明方向语义已学到，也不能视为提升分割 mIoU。"
        ),
    }
    report = args.out / "step57_distilbert_smoke.json"
    report.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("===== Step57 真实 DistilBERT + BEV 验证 =====")
    print(f"设备：{device} | BEV：{source} | 模型：{args.model}")
    print(f"文本向量：{tuple(pooled.shape)} | 注意力：{tuple(attention.shape)}")
    print(f"文本向量平均差异：{pooled_delta:.8f}")
    print(f"注意力平均差异：{attention_delta:.8f}")
    print("真实文本编码：PASS；Baseline 回退：PASS；融合梯度：PASS")
    print(f"报告：{report}")
    print("说明：模块级测试，未完成语言监督训练或分割任务评估。")


if __name__ == "__main__":
    main()

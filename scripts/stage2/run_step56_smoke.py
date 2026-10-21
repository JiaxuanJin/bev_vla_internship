"""Step56：语言引导 BEV 模块的形状、回退与反向传播测试。

示例：
    python3 -m scripts.stage2.run_step56_smoke --text-mode synthetic
    python3 -m scripts.stage2.run_step56_smoke --text-mode distilbert

注意：synthetic 只用于接口与梯度验证，不代表预训练语言理解或真实训练。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .bev_fusion import LanguageBEVFusion
from .language_encoder import DistilBERTLanguageEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE = PROJECT_ROOT / "outputs/bev_features/bev_feature.npy"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/stage2_step56"


def load_bev_feature(
    path: Path,
    source: str,
    device: torch.device,
) -> tuple[torch.Tensor, str]:
    """读取曾经提取的真实 BEV 特征，或生成确定性测试张量。"""
    if source == "real" and not path.is_file():
        raise FileNotFoundError(f"没有找到真实特征：{path}")
    if source == "real" or (source == "auto" and path.is_file()):
        loaded = np.load(path, allow_pickle=False)
        if loaded.shape == (150, 150, 256):
            loaded = np.transpose(loaded, (2, 0, 1))[None]
        elif loaded.shape == (1, 150, 150, 256):
            loaded = np.transpose(loaded, (0, 3, 1, 2))
        elif loaded.shape != (1, 256, 150, 150):
            raise ValueError(f"不支持的 BEV 特征形状：{loaded.shape}")
        if not np.isfinite(loaded).all():
            raise ValueError("读取的真实特征包含 NaN 或 Inf")
        return torch.from_numpy(loaded.copy()).to(device=device).float(), "real_npy"

    # 生成固定随机种子的模拟特征，便于不同环境复现测试。
    rng = torch.Generator(device="cpu").manual_seed(20261021)
    synthetic = torch.randn(1, 256, 150, 150, generator=rng)
    return synthetic.to(device=device), "synthetic_tensor"


def make_text_feature(
    text_mode: str,
    device: torch.device,
    model_name: str,
    local_files_only: bool,
) -> tuple[torch.Tensor, bool]:
    """返回语言特征；真实 DistilBERT 与合成特征严格区分。"""
    if text_mode == "distilbert":
        encoder = DistilBERTLanguageEncoder(
            model_name=model_name,
            freeze=True,
            local_files_only=local_files_only,
        ).to(device)
        pooled = encoder(
            ["Pay attention to pedestrians in the front right."], str(device)
        )
        return pooled.detach(), True

    # 只有合成向量，不调用 tokenizer 或预训练权重。
    rng = torch.Generator(device="cpu").manual_seed(1021)
    return torch.randn(1, 768, generator=rng).to(device), False


def main() -> None:
    """执行 forward/backward、基线回退、梯度与输出文件检查。"""
    parser = argparse.ArgumentParser(description="Step56 语言引导 BEV 冒烟测试")
    parser.add_argument(
        "--text-mode", choices=("synthetic", "distilbert"), default="synthetic"
    )
    parser.add_argument(
        "--feature-source", choices=("auto", "real", "synthetic"), default="auto"
    )
    parser.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bev, feature_kind = load_bev_feature(
        args.feature_path, args.feature_source, device
    )
    pooled, pretrained_loaded = make_text_feature(
        args.text_mode, device, args.model, args.local_files_only
    )
    fusion = LanguageBEVFusion(text_dim=pooled.shape[1]).to(device)
    fusion.train()

    baseline, _ = fusion(bev, pooled, enabled=False)
    if not torch.equal(baseline, bev):
        raise AssertionError("Baseline 回退应与输入完全一致")

    enhanced, attention = fusion(bev, pooled, enabled=True)
    if enhanced.shape != bev.shape or attention.shape != (1, 1, 150, 150):
        raise AssertionError("输出维度与设计接口不一致")
    if not torch.isfinite(enhanced).all() or not torch.isfinite(attention).all():
        raise AssertionError("生成结果包含 NaN/Inf")
    if attention.min().item() < 0 or attention.max().item() > 1:
        raise AssertionError("注意力应处于 [0, 1] 区间")

    # 该损失仅用于测试梯度流向，不是分割/检测训练损失。
    gradient_test_loss = enhanced.square().mean()
    gradient_test_loss.backward()
    grad_info = {}
    for name, parameter in fusion.named_parameters():
        grad = parameter.grad
        grad_norm = 0.0 if grad is None else float(grad.norm().item())
        grad_info[name] = grad_norm
        if not np.isfinite(grad_norm) or grad_norm <= 0:
            raise AssertionError(f"梯度检查失败：{name} = {grad_norm}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output_dir / f"attention_{args.text_mode}.npy",
        attention.detach().float().cpu().numpy()[0, 0],
    )
    result = {
        "experiment": "step56_language_bev_forward_backward_smoke",
        "text_mode": args.text_mode,
        "pretrained_language_model_loaded": pretrained_loaded,
        "bev_source": feature_kind,
        "bev_shape": list(bev.shape),
        "pooled_text_shape": list(pooled.shape),
        "attention_shape": list(attention.shape),
        "enhanced_shape": list(enhanced.shape),
        "baseline_exact_identity": True,
        "attention_min": float(attention.min().item()),
        "attention_max": float(attention.max().item()),
        "initial_alpha": float(fusion.alpha.detach().item()),
        "mean_absolute_feature_delta": float((enhanced - bev).abs().mean().item()),
        "gradient_test_loss": float(gradient_test_loss.item()),
        "gradient_norms": grad_info,
        "all_nonzero_grads": True,
        "task_training_completed": False,
        "segmentation_evaluation_completed": False,
        "official_score": False,
        "notes": "仅验证模块接口与梯度；真实 BEV NPY 也不代表已接入模型分割头。",
    }
    report_path = args.output_dir / f"step56_smoke_{args.text_mode}.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("===== Step56 语言引导 BEV 模块测试 =====")
    print(f"Device: {device} | BEV: {feature_kind} | Text: {args.text_mode}")
    print(f"BEV: {tuple(bev.shape)} | Attention: {tuple(attention.shape)}")
    print(f"增强差值均值：{result['mean_absolute_feature_delta']:.8f}")
    print("Baseline 回退：PASS；张量检查：PASS；梯度流：PASS")
    print(f"报告：{report_path}")
    print("注意：未进行真实训练或分割 mIoU 评估。")


if __name__ == "__main__":
    main()

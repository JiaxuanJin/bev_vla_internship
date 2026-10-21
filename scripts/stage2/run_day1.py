"""阶段二 Day 1：真实 DistilBERT + 语义投影的最小运行实验。"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage2.language_encoder import (  # noqa: E402
    LanguageEncoder,
    load_local_tokenizer,
    prepare_text_batch,
)
from stage2.semantic_projection import SemanticProjection  # noqa: E402


MODEL_DIR = PROJECT_ROOT / "models" / "distilbert-base-uncased"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stage2" / "2026-10-20"
CONFIG_PATH = PROJECT_ROOT / "stage2" / "tokenizer_config.json"


def main() -> None:
    """运行两条指令的批量编码，并验证投影头具有非零梯度。"""
    parser = argparse.ArgumentParser(description="阶段二 Day 1 真实模型冒烟测试")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    torch.manual_seed(20261020)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    tokenizer = load_local_tokenizer(MODEL_DIR)
    language_encoder = LanguageEncoder.from_local_model(
        MODEL_DIR, freeze_backbone=bool(config["freeze_backbone"])
    ).to(device)
    semantic_projection = SemanticProjection(
        text_dim=language_encoder.output_dim,
        bev_dim=int(config["bev_feature_dim"]),
    ).to(device)

    texts = [
        "Focus on the pedestrian in front of the ego vehicle.",
        "Watch for vehicles approaching from the front right, and avoid obstacles.",
    ]
    inputs = prepare_text_batch(
        texts, tokenizer, max_length=int(config["max_length"])
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    language_features = language_encoder(**inputs)
    bev_guidance = semantic_projection(language_features)

    # 用占位的 BEV 特征规格核对通道，不把它冒充实际 BEVFormer 输出。
    expected_bev_shape = (len(texts), int(config["bev_feature_dim"]), 150, 150)
    assert bev_guidance.shape == (len(texts), expected_bev_shape[1])
    assert torch.isfinite(bev_guidance).all().item()
    assert (inputs["attention_mask"] == 0).any().item(), "应出现 padding token"

    # 这只是梯度链连通性检测，随机投影头没有经过任务训练。
    semantic_projection.zero_grad(set_to_none=True)
    loss = bev_guidance.square().mean()
    loss.backward()
    grad = semantic_projection.projection.weight.grad
    if grad is None or grad.abs().sum().item() <= 0:
        raise RuntimeError("语义投影层未收到有效梯度")

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "stage2_day1_language_encoding_semantic_projection",
        "real_pretrained_distilbert": True,
        "model_id": config["model_id"],
        "model_revision": config["revision"],
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sample_texts": texts,
        "token_tensor_shape": list(inputs["input_ids"].shape),
        "valid_token_counts": inputs["attention_mask"].sum(dim=1).tolist(),
        "padding_tokens_total": int((inputs["attention_mask"] == 0).sum().item()),
        "language_features_shape": list(language_features.shape),
        "semantic_projection_shape": list(bev_guidance.shape),
        "future_bev_interface_shape": list(expected_bev_shape),
        "backbone_frozen": bool(config["freeze_backbone"]),
        "projection_gradient_l1": float(grad.abs().sum().item()),
        "projection_is_randomly_initialized": True,
        "bevformer_real_features_used": False,
        "status": "PASS",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "day1_smoke_test.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": semantic_projection.state_dict(),
            "text_dim": language_encoder.output_dim,
            "bev_dim": int(config["bev_feature_dim"]),
            "trained": False,
            "note": "随机初始化：仅做接口与梯度测试，尚未学习视觉语言对应关系",
        },
        OUTPUT_DIR / "semantic_projection_init.pt",
    )
    print("========== 第二阶段 Day 1 ==========")
    print("运行设备:", device)
    print("Token shape:", tuple(inputs["input_ids"].shape))
    print("Mask shape:", tuple(inputs["attention_mask"].shape))
    print("有效 token 数:", result["valid_token_counts"])
    print("文本向量:", tuple(language_features.shape))
    print("语义投影:", tuple(bev_guidance.shape))
    print("明日 BEV 对接规格:", expected_bev_shape)
    print("投影层梯度 L1:", result["projection_gradient_l1"])
    print("[PASS] 预训练语言编码、padding、投影与梯度链均通过")
    print("结果文件:", result_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()

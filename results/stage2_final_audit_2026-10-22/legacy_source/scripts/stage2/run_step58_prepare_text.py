"""Step58 第一阶段：在工作环境中离线编码真实驾驶指令。

语言主干使用 Step57 验证过的本地 DistilBERT，保持冻结。
单独缓存文本向量，避免向作者隔离环境重新安装 transformers。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.stage2.language_encoder import DistilBERTLanguageEncoder

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models/distilbert-base-uncased"
DEFAULT_OUT = ROOT / "outputs/stage2_step58"
PROMPTS = (
    "Pay attention to pedestrians in the front right.",
    "Pay attention to vehicles behind the car.",
)


def sha256(path: Path) -> str:
    """流式计算模型权重及其配置的校验和。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """生成供独立作者环境消费的文本向量和可追溯元数据。"""
    parser = argparse.ArgumentParser(description="Step58 冻结文本特征导出")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    model_dir = args.model.expanduser().resolve()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"本地 DistilBERT 配置不存在：{model_dir}")

    torch.manual_seed(20261021)
    encoder = DistilBERTLanguageEncoder(
        model_name=str(model_dir), freeze=True, local_files_only=True
    ).cpu()
    encoder.eval()
    with torch.inference_mode():
        pooled = encoder(list(PROMPTS), "cpu").float().numpy()
    if pooled.shape != (2, 768) or not np.isfinite(pooled).all():
        raise ValueError(f"真实文本向量错误：{pooled.shape}")
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "distilbert_prompts.npy", pooled, allow_pickle=False)

    weights = next(
        (model_dir / item for item in ("model.safetensors", "pytorch_model.bin")
         if (model_dir / item).is_file()),
        None,
    )
    if weights is None:
        raise FileNotFoundError("找不到模型权重文件")
    metadata = {
        "experiment": "step58_frozen_real_distilbert_precomputed_embeddings",
        "source_model": str(model_dir),
        "weight_file": weights.name,
        "weight_sha256": sha256(weights),
        "config_sha256": sha256(model_dir / "config.json"),
        "prompts": list(PROMPTS),
        "text_shape": list(pooled.shape),
        "embeddings_sha256": sha256(args.out / "distilbert_prompts.npy"),
        "note": "真实预训练模型、冻结推理；只缓存文本，非端到端可训练语言主干",
    }
    report = args.out / "text_manifest.json"
    report.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print("[Step58-1] 真实 DistilBERT 文本向量已缓存：", args.out)
    print("[Step58-1] shape:", pooled.shape)
    print("[Step58-1] report:", report)


if __name__ == "__main__":
    main()

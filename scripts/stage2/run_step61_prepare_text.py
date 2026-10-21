"""Step61：从本地 DistilBERT 导出三个类别指令的冻结文本特征。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.stage2.language_encoder import DistilBERTLanguageEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = PROJECT_ROOT / "models/distilbert-base-uncased"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/stage2_step61"
PROMPTS = (
    "Focus on the road dividers and lane dividers.",
    "Focus on the pedestrian crossings.",
    "Focus on the road boundaries.",
)


def sha256(path: Path) -> str:
    """分块计算文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """只进行一次文本预处理，不训练语言主干。"""
    parser = argparse.ArgumentParser(description="Step61 DistilBERT class prompts")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    model_path = args.model.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"本地 DistilBERT 模型缺少 config.json：{model_path}")

    torch.manual_seed(20261021)
    encoder = DistilBERTLanguageEncoder(
        model_name=str(model_path), freeze=True, local_files_only=True
    ).cpu().eval()
    with torch.inference_mode():
        vectors = encoder(list(PROMPTS), device="cpu")
    embeddings = vectors.detach().numpy().astype(np.float32, copy=False)
    if embeddings.shape != (3, 768) or not np.isfinite(embeddings).all():
        raise RuntimeError(f"文本特征异常：{embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1)
    cosine = embeddings @ embeddings.T / np.maximum(np.outer(norms, norms), 1e-8)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "class_prompts.npy"
    np.save(output, embeddings)
    manifest = {
        "step": 61,
        "prompts": list(PROMPTS),
        "class_ids": [1, 2, 3],
        "model_local_dir": str(model_path),
        "embedding_shape": list(embeddings.shape),
        "embedding_sha256": sha256(output),
        "cosine_similarity": cosine.tolist(),
        "note": "冻结 DistilBERT：仅用于人工指定类别词与 nuScenes Map 类别的监督配对",
    }
    report = args.output / "class_prompts_manifest.json"
    report.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("[Step61-1] DistilBERT 3 个类别指令加载成功", flush=True)
    print("[Step61-1] 文本形状:", embeddings.shape, flush=True)
    print("[Step61-1] 向量 SHA256:", manifest["embedding_sha256"], flush=True)
    print("[Step61-1] 文件:", output, flush=True)
    print("[Step61-1] 配置:", report, flush=True)


if __name__ == "__main__":
    main()

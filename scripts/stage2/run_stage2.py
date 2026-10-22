#!/usr/bin/env python3
"""阶段二统一入口：CPU 回归、真实离线文本/BEV、真实联合双 Head。"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from scripts.stage2.pipeline import Stage2Pipeline
from scripts.stage2.run_step56_smoke import load_bev_feature


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def offline_bev(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pipeline = Stage2Pipeline.from_local_model(args.model).to(device).eval()
    bev, source = load_bev_feature(args.bev, "real", device)
    texts = args.text
    with torch.inference_mode():
        batch_bev = bev.expand(len(texts), -1, -1, -1)
        result = pipeline(batch_bev, texts)
        bypass = pipeline(batch_bev, enabled=False)
        if bypass["enhanced_bev"] is not batch_bev:
            raise AssertionError("旁路未返回原始 BEV 对象")
        # 批次置乱只核对样本独立性，不声称正确文本优于置乱文本。
        embeddings = result["text_embedding"]
        reversed_embedding, _ = pipeline.encode(list(reversed(texts)), device)
        torch.testing.assert_close(
            reversed_embedding.flip(0), embeddings, atol=1e-6, rtol=1e-5
        )
        alone, _ = pipeline.encode([texts[0]], device)
        torch.testing.assert_close(alone[0], embeddings[0], atol=1e-6, rtol=1e-5)
        shuffled_text = " ".join(reversed(texts[0].split()))
        shuffled_embedding, parsed = pipeline.encode(
            [shuffled_text], device, strict=False
        )
        # 相同 BEV，置乱配对：仅记录数值，未设置语义成功阈值。
        paired = pipeline.fuse_encoded(batch_bev, embeddings.flip(0))
        info = {
            "status": "PASS_PRETRAINED_ENCODING_AND_REAL_CACHED_BEV_FUNCTION_ONLY",
            "feature_source": source,
            "text": texts,
            "instructions": result["instructions"],
            "shapes": {
                key: list(value.shape)
                for key, value in result.items()
                if isinstance(value, torch.Tensor)
            },
            "bypass_exact_identity": True,
            "batch_permutation_equivariant": True,
            "single_vs_padded_batch_consistent": True,
            "word_shuffle": {
                "text": shuffled_text,
                "rule_parse": parsed[0],
                "embedding_mae": float(
                    (embeddings[:1] - shuffled_embedding).abs().mean()
                ),
            },
            "text_bev_pair_shuffle_attention_mae": float(
                (result["attention"] - paired["attention"]).abs().mean()
            ),
            "semantic_localization": "NOT_VERIFIED",
            "accuracy_improvement": "NOT_VERIFIED",
            "bev_gt_physical_alignment": "NOT_VERIFIED",
            "fusion_weights_trained": False,
            "optimizer_step_executed": False,
        }
    config = json.loads((ROOT / "stage2/tokenizer_config.json").read_text())
    actual = file_digest(args.model / "model.safetensors")
    if actual != config["expected_model_sha256"]:
        raise AssertionError("本地 DistilBERT safetensors 与已记录 SHA256 不一致")
    return info | {
        "distilbert_sha256": actual,
        "expected_distilbert_sha256": config["expected_model_sha256"],
        "model_id": config["model_id"],
        "recorded_revision": config["revision"],
        "bev_sha256": file_digest(args.bev),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("cpu-tests", "offline-bev", "joint", "joint-backward", "micro-train"),
        required=True,
    )
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/distilbert-base-uncased"
    )
    parser.add_argument(
        "--bev", type=Path, default=ROOT / "outputs/bev_features/bev_feature.npy"
    )
    parser.add_argument(
        "--text", action="append", help="可重复指定；joint 单帧只接受一条指令"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--author-root",
        type=Path,
        default=Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg",
    )
    parser.add_argument(
        "--joint-checkpoint",
        type=Path,
        default=Path("/mnt/g/nus/bevformer_small_joint_author.pth"),
    )
    parser.add_argument("--joint-config", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--micro-phase",
        choices=("step", "protocol", "continue", "validate", "smoke"),
        default="step",
    )
    parser.add_argument(
        "--micro-worker",
        choices=("protocol", "train", "validate", "smoke"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--micro-steps",
        type=int,
        default=1,
        help="Additional updates; fresh run is restricted to one",
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--fusion-checkpoint", type=Path, help="smoke 加载独立融合权重，不创建优化器"
    )
    parser.add_argument(
        "--attention-tv-weight",
        type=float,
        default=0.0,
        help="仅用于新协议的 TV 权重；默认 0 保持历史损失",
    )
    parser.add_argument("--min-host-available-mib", type=int, default=2048)
    parser.add_argument("--min-cuda-free-mib", type=int, default=768)
    parser.add_argument("--max-worker-rss-mib", type=int, default=11264)
    args = parser.parse_args()
    if not math.isfinite(args.attention_tv_weight) or args.attention_tv_weight < 0:
        parser.error("attention-tv-weight must be finite and nonnegative")
    if args.protocol and args.attention_tv_weight != 0:
        parser.error("已有协议的损失参数不可覆盖；新配置需生成新协议")
    if (
        args.fusion_checkpoint
        and args.micro_phase != "smoke"
        and args.micro_worker != "smoke"
    ):
        parser.error("fusion-checkpoint 仅用于不更新参数的 smoke")
    if args.mode == "micro-train" and args.output is None:
        parser.error("micro-train requires --output in a new directory")
    if (
        min(
            args.min_host_available_mib, args.min_cuda_free_mib, args.max_worker_rss_mib
        )
        <= 0
    ):
        parser.error("memory thresholds must be positive")
    if args.text is None:
        args.text = (
            ["Find lane markings"]
            if args.mode == "joint-backward"
            else (
                ["Find lane markings ahead"]
                if args.mode == "joint"
                else [
                    "Find lane markings ahead",
                    "Locate lane dividers in front",
                    "Locate pedestrian crossings ahead",
                    "Find the curb on the rear right",
                    "Find road boundaries and pedestrian crossings on the left",
                    "Highlight road boundaries but ignore pedestrian crossings",
                    "Show the zebra crossing on the front right",
                ]
            )
        )
    if args.mode in ("joint", "joint-backward") and len(args.text) != 1:
        parser.error("joint 单帧只接受一条 --text，禁止静默丢弃其余指令")
    args.model, args.bev = args.model.resolve(), args.bev.resolve()
    output = (
        args.output or ROOT / f"outputs/stage2_completion/{args.mode}.json"
    ).resolve()
    if output.exists() or (
        args.mode == "cpu-tests"
        and (output.parent / (output.stem + "_regression.json")).exists()
    ):
        parser.error(f"结果已存在，拒绝覆盖历史记录：{output}")
    args.output = output
    if args.protocol:
        args.protocol = args.protocol.resolve()
    if args.resume:
        args.resume = args.resume.resolve()
    if args.fusion_checkpoint:
        args.fusion_checkpoint = args.fusion_checkpoint.resolve()
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "seed": args.seed,
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    record["command"] = [sys.executable, *sys.argv]
    record["random_seed_scope"] = (
        "Python random, NumPy and torch before dataset construction; fusion torch seed reset"
        if args.mode in ("joint", "joint-backward", "micro-train")
        else "torch; CPU regression has per-test fixed seeds"
    )
    record["python_version"] = sys.version.split()[0]
    record["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    record["versions"] = {}
    for package in (
        "torch",
        "transformers",
        "numpy",
        "mmcv-full",
        "mmdet",
        "mmdet3d",
        "pytest",
    ):
        try:
            record["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            record["versions"][package] = "NOT_INSTALLED"
    record["source_sha256"] = {
        str(path.relative_to(ROOT)): file_digest(path)
        for path in sorted(
            {*ROOT.glob("scripts/stage2/*.py"), *ROOT.glob("tests/test_stage2*.py")}
        )
    }
    try:
        if args.mode == "cpu-tests":
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/stage2/run_regression.py"),
                    "--output",
                    str(output.parent / (output.stem + "_regression.json")),
                ],
                cwd=ROOT,
            )
            if proc.returncode:
                raise RuntimeError(f"CPU 回归退出码 {proc.returncode}")
            record.update(status="PASS_CPU_REGRESSION")
        elif args.mode == "offline-bev":
            record.update(offline_bev(args))
        elif args.mode == "micro-train" and not args.micro_worker:
            from scripts.stage2.micro_process import run_micro_stages

            record.update(run_micro_stages(args))
        else:
            from scripts.stage2.joint_runtime import run_joint

            record.update(run_joint(args))
        exit_code = 0
    except Exception as exc:
        record.update(
            status="FAIL",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
            semantic_localization="NOT_VERIFIED",
            accuracy_improvement="NOT_VERIFIED",
        )
        exit_code = 1
    output.parent.mkdir(parents=True, exist_ok=True)
    # x 模式防止同名进程覆盖既有记录。
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"{args.mode}: {record['status']} -> {output}", flush=True)
    if exit_code:
        print(record["error"], file=sys.stderr, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

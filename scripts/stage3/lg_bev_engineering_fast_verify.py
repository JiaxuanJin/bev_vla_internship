"""只跳过不进入模型的GT重建，32帧验证原图像处理/原生时序/梯度逐项不变。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.stage2.joint_runtime import _first_tensor
from scripts.stage2.micro_training import get_batch
from scripts.stage3.lg_bev_common import ROOT, read, sha, write
from scripts.stage3.lg_bev_engineering_audit import Reference, maximum_delta
from scripts.stage3.lg_bev_engineering_cache import RollingCache
from scripts.stage3.lg_bev_engineering_data import SceneData, LabelStore, formal_frames
from scripts.stage3.lg_bev_engineering_native import NativeRuntime, NativeCacheProvider
from scripts.stage3.lg_bev_engineering_train import Trainer
from scripts.stage3.lg_bev_native import dual_head, module_sha


def verify(out: Path) -> None:
    """先登记完全一致门槛再运行，不修改任何已学习算子或模型GT协议。"""
    plan = read(out / "engineering_supplement.json")
    source_hashes = {
        str(p): sha(p)
        for p in (ROOT / "scripts/stage3").glob("lg_bev_engineering_*.py")
    }
    run = Path(plan["run"])
    write(
        out / "image_pipeline_supplement.json",
        dict(
            purpose="remove unused GT raster/map-vector generation only from input preparation; all author image transforms, native backbone/history/heads unchanged",
            expected="32frame image tensors, FP32 BEV and native logits bitwise; gradient/optimizer use existing preregistered thresholds; no loss/target/budget changes",
        ),
    )
    selected = plan["rows"]
    scene = selected[0]["scene_token"]
    data = SceneData([r for r in formal_frames() if r["scene_token"] == scene])
    reference = Reference(read(out / "native_temporal_audit.json")["reference"])
    labels = LabelStore(read(out / "data_report.json"))
    runtime = NativeRuntime(fast_images=True)
    cache = RollingCache()
    records = []
    before = module_sha(runtime.model)
    original_hash = {}
    for index, row in enumerate(selected):
        token = row["sample_token"]
        slow = get_batch(data.dataset, index, 20261027)
        with torch.no_grad():
            bev, pred, trace = runtime.forward(data, index)
            golden = reference.get(token)
            if not torch.equal(bev, golden):
                raise RuntimeError("image-only改变了原生BEV")
            replay = dual_head(runtime.model, golden.cuda())
            delta = {k: float((pred[k] - replay[k].cpu()).abs().max()) for k in pred}
            if any(delta.values()):
                raise RuntimeError("image-only改变原生双Head预测")
            original_hash[token] = {
                k: __import__("hashlib")
                .sha256(replay[k].cpu().numpy().tobytes())
                .hexdigest()
                for k in pred
            }
        actual = (
            _first_tensor(slow["semantic_indices"])
            .reshape(200, 400)
            .cpu()
            .to(torch.uint8)
        )
        if not torch.equal(actual, labels.get(token)["gt"]):
            raise RuntimeError("真实原作者raster与独立GT不符")
        old = data.dataset.pipeline
        data.dataset.pipeline = data.image_pipeline
        data.dataset.get_map_ann_info = lambda _: []
        try:
            fast = get_batch(data.dataset, index, 20261027)
        finally:
            data.dataset.pipeline = old
            del data.dataset.get_map_ann_info
        if not torch.equal(_first_tensor(slow["img"]), _first_tensor(fast["img"])):
            raise RuntimeError("原生图像预处理不一致")
        records.append(
            {
                **trace,
                "images_bitwise": True,
                "map_gt_bitwise": True,
                "native_heads_max_abs": delta,
            }
        )
        del slow, fast
    from scripts.stage2.language_encoder import DistilBERTLanguageEncoder
    from scripts.stage2.spatial_v3_model import InstanceReadout

    encoder = (
        InstanceReadout(
            DistilBERTLanguageEncoder(
                str(ROOT / "models/distilbert-base-uncased"), local_files_only=True
            )
        )
        .cuda()
        .eval()
    )
    encoder.requires_grad_(False)
    tokens = [r["sample_token"] for r in selected]
    fingerprint = sha(out / "engineering_supplement.json")
    baseline = Trainer(
        runtime.model,
        encoder,
        labels,
        tokens,
        reference.get,
        "B",
        8,
        fingerprint,
        retain_audit_tensors=True,
    )
    raw_probe = Trainer(
        runtime.model,
        encoder,
        labels,
        tokens,
        reference.get,
        "B",
        8,
        fingerprint,
        retain_audit_tensors=True,
    )
    raw_probe.microstep()
    stable_probe = Trainer(
        runtime.model,
        encoder,
        labels,
        tokens,
        reference.get,
        "B",
        8,
        fingerprint,
        retain_audit_tensors=True,
    )
    stable_probe.run(1)
    ga = {
        n: p.grad.cpu()
        for n, p in raw_probe.adapter.named_parameters()
        if p.grad is not None
    }
    gb = {
        n: p.grad.cpu()
        for n, p in stable_probe.adapter.named_parameters()
        if p.grad is not None
    }
    gradient_abs, gradient_rel = maximum_delta(ga, gb)
    write(
        out / "gradient_implementation_comparison.json",
        dict(max_abs=gradient_abs, relative_l2=gradient_rel),
    )
    if (
        gradient_abs > plan["tolerances_registered_before_gpu"]["gradient_max_abs"]
        or gradient_rel
        > plan["tolerances_registered_before_gpu"]["gradient_relative_l2"]
    ):
        raise RuntimeError("确定性卷积adjoint偏离原生CUDA梯度")
    baseline.run(32, log=run / "fast_verify_reference_training.jsonl")
    provider = NativeCacheProvider(cache, runtime, {scene: data}, admit_disk=False)
    streamed = Trainer(
        runtime.model,
        encoder,
        labels,
        tokens,
        provider.get,
        "B",
        8,
        fingerprint,
        retain_audit_tensors=True,
    )
    streamed.run(32, log=run / "fast_verify_native_stream_training.jsonl")
    absolute, relative = maximum_delta(
        baseline.update_records[-1]["state"], streamed.update_records[-1]["state"]
    )
    write(
        out / "fast_training_comparison.json",
        dict(
            max_abs=absolute,
            relative_l2=relative,
            per_parameter_max_abs={
                n: float(
                    (
                        baseline.update_records[-1]["state"][n]
                        - streamed.update_records[-1]["state"][n]
                    )
                    .abs()
                    .max()
                )
                for n in baseline.update_records[-1]["state"]
            },
        ),
    )
    if absolute > plan["tolerances_registered_before_gpu"]["parameter_max_abs"]:
        raise RuntimeError("优化数据通路的训练参数不一致")
    for token in tokens:
        with torch.no_grad():
            pred = dual_head(runtime.model, reference.get(token).cuda())
        for key, digest in original_hash[token].items():
            if (
                __import__("hashlib")
                .sha256(pred[key].cpu().numpy().tobytes())
                .hexdigest()
                != digest
            ):
                raise RuntimeError("训练后原始预测不变量失败")
    if module_sha(runtime.model) != before:
        raise RuntimeError("原作者权重或BN变化")
    if any(sha(Path(p)) != digest for p, digest in source_hashes.items()):
        raise RuntimeError("验收过程中工程源码变化，禁止封存通过报告")
    write(
        out / "fast_pipeline_equivalence.json",
        dict(
            status="PASS",
            frames=32,
            records=records,
            images_bitwise=True,
            BEV_bitwise=True,
            native_heads_bitwise=True,
            original_predictions_after_updates_bitwise=True,
            optimizer_updates_per_case=8,
            formal_updates=0,
            parameter_max_abs=absolute,
            parameter_relative_l2=relative,
            reference_cached_B32_seconds=baseline.wall_seconds,
            native_RAM_stream_B32_seconds=streamed.wall_seconds,
            mean_input_preprocessing_seconds=sum(
                r["preprocessing_seconds"] for r in records
            )
            / 32,
            mean_gpu_native_seconds=sum(r["seconds"] for r in records) / 32,
            frozen_sha256=before,
            source_sha256=source_hashes,
        ),
    )
    print("FAST_NATIVE32_EQUIVALENCE_PASS_NO_FORMAL_TRAINING", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    verify(parser.parse_args().out.resolve())

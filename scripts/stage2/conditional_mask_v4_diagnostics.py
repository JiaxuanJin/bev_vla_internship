"""V4 因果梯度、真实六相机等价校验与同条件同步计时。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.stage2.conditional_mask_v4_data import OUT, SEED, V3, write
from scripts.stage2.conditional_mask_v4_experiment import (
    frame_inputs,
    load_label,
    load_runtime,
)
from scripts.stage2.conditional_mask_v4_model import (
    LanguageConditionedTargetMaskHead,
    ZeroChannelResidual,
    mask_loss,
)
from scripts.stage2.micro_training import forward_payload, get_batch, seed_all
from scripts.stage2.spatial_annotations import AUTHOR
from scripts.stage2.spatial_v3_experiment import actual_losses, decode, head_forward
from scripts.stage2.spatial_v3_range_correction import filter_ann
from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.runtime_memory import release_memory


def gradient(
    value: torch.Tensor, inputs: list[torch.Tensor]
) -> list[torch.Tensor | None]:
    """允许不连通路径，显式返回 None，不把未训练误称成梯度更新。"""
    return list(
        torch.autograd.grad(value, inputs, retain_graph=True, allow_unused=True)
    )


def grad_norm(values: list[torch.Tensor | None]) -> float:
    """计算可核验的总 L2 范数。"""
    return (
        float(sum(v.detach().square().sum() for v in values if v is not None).sqrt())
        if any(v is not None for v in values)
        else 0.0
    )


def diagnose(
    model: Any,
    pipeline: Any,
    dataset: Any,
    manifest: dict,
    features: dict,
    text: dict,
    head: torch.nn.Module,
) -> dict:
    """同一真实输入比较零更新的无融合、原标量融合、零初始化通道残差。"""
    directory = OUT / "fusion_diagnosis"
    directory.mkdir()
    train = [r for r in manifest["annotations"] if r["split"] == "train"]
    dev = [r for r in manifest["annotations"] if r["split"] == "dev"]
    probes = []
    for rows, key in ((train, "scene_token"), (dev, "index")):
        for group in sorted({r[key] for r in rows}):
            candidates = [r for r in rows if r[key] == group]
            probes.append(
                next((r for r in candidates if r["counterfactual_text"]), candidates[0])
            )
    write(
        directory / "protocol.json",
        dict(
            annotations=probes,
            branches=["no_fusion", "scalar_v3", "zero_channel_residual"],
            optimizer_updates_per_branch=0,
            selection="metadata-only first paired per train scene and dev frame; no optimizer, no model choice",
            alpha=0.0129582463,
            pretrained_cropper_changed=False,
        ),
    )
    pipeline.fusion.requires_grad_(True)
    pipeline.fusion.alpha.requires_grad_(False)
    scalar_params = [p for p in pipeline.fusion.parameters() if p.requires_grad]
    records = []
    for row in probes:
        with np.load(row["source_file"]) as f:
            gt = torch.tensor(f["gt"], device="cuda", dtype=torch.long)
        label = load_label(row)
        grid, geometry = frame_inputs(row)
        bev = features[row["index"]].detach().requires_grad_(True)
        pipeline.fusion.set_calibration(row)
        with torch.no_grad():
            baseline = head_forward(model, bev)
        seed_all(SEED)
        channel = ZeroChannelResidual().cuda().eval()
        for branch in ("no_fusion", "scalar_v3", "zero_channel_residual"):
            detail = pipeline.fusion.forward_with_details(bev, text[row["text"]])
            if branch == "no_fusion":
                enhanced, params = bev, []
            elif branch == "scalar_v3":
                enhanced, params = detail["enhanced_bev"], scalar_params
            else:
                enhanced, params = channel(
                    bev, text[row["text"]], detail["attention"].detach()
                ), list(channel.parameters())
            pred = head_forward(model, enhanced)
            ann, filter_stats = filter_ann(
                dataset.get_ann_info(row["index"]), model.pts_bbox_head.pc_range
            )
            detection = actual_losses(model, pred, ann)
            semantic = F.cross_entropy(pred["seg_preds"], gt[None])
            mask_logits = head(bev, text[row["text"]], grid, geometry)
            grounding, parts = mask_loss(
                mask_logits[0, 0], label["target"], label["same_class_other"]
            )
            grads = {
                name: gradient(value, [bev])[0].detach()
                for name, value in (
                    ("detection", detection),
                    ("segmentation", semantic),
                    ("grounding", grounding),
                )
            }
            values = dict(
                branch=branch,
                instruction_id=row["instruction_id"],
                split=row["split"],
                optimizer_updates=0,
                relative_bev_l2=float(((enhanced - bev).norm() / bev.norm()).detach()),
                segmentation_logits_rms_delta=float(
                    (pred["seg_preds"] - baseline["seg_preds"])
                    .square()
                    .mean()
                    .sqrt()
                    .detach()
                ),
                detection_logits_rms_delta=float(
                    (pred["all_cls_scores"] - baseline["all_cls_scores"])
                    .square()
                    .mean()
                    .sqrt()
                    .detach()
                ),
                bbox_rms_delta=float(
                    (pred["all_bbox_preds"] - baseline["all_bbox_preds"])
                    .square()
                    .mean()
                    .sqrt()
                    .detach()
                ),
                changed_semantic_pixels=int(
                    (
                        pred["seg_preds"].argmax(1) != baseline["seg_preds"].argmax(1)
                    ).sum()
                ),
                bev_grad_l2={k: float(v.norm()) for k, v in grads.items()},
                feature_gradient_cosine={
                    f"{a}__{b}": float(
                        F.cosine_similarity(
                            grads[a].reshape(1, -1), grads[b].reshape(1, -1)
                        )
                    )
                    for a, b in (
                        ("detection", "segmentation"),
                        ("detection", "grounding"),
                        ("segmentation", "grounding"),
                    )
                },
                fusion_parameter_grad_l2={
                    "detection": (
                        grad_norm(gradient(detection, params)) if params else None
                    ),
                    "segmentation": (
                        grad_norm(gradient(semantic, params)) if params else None
                    ),
                },
                native_task_gradient_to_mask_head={
                    "detection": grad_norm(
                        gradient(detection, list(head.parameters()))
                    ),
                    "segmentation": grad_norm(
                        gradient(semantic, list(head.parameters()))
                    ),
                },
                grounding_gradient_to_mask_head=grad_norm(
                    gradient(grounding, list(head.parameters()))
                ),
                filtered_gt=filter_stats,
            )
            if branch == "zero_channel_residual":
                values["per_parameter_segmentation_gradient_l2"] = {
                    name: float(v.norm()) if v is not None else None
                    for (name, p), v in zip(
                        channel.named_parameters(),
                        gradient(semantic, list(channel.parameters())),
                    )
                }
            records.append(values)
            print(
                "V4_FUSION",
                branch,
                row["split"],
                values["relative_bev_l2"],
                values["native_task_gradient_to_mask_head"],
                flush=True,
            )
    pipeline.fusion.requires_grad_(False)
    result = dict(
        records=records,
        n_train_probes=5,
        n_dev_probes=8,
        optimizer_updates=0,
        interpretation="Independent grounding Head is disconnected from native perception losses. Zero channel residual has zero initial perturbation; up-layer task gradient is a feasibility diagnosis, not measured trained benefit. No alpha tuning, no cropper/GT change.",
    )
    write(directory / "metrics.json", result)
    return result


def timing(fn: Callable, warmup: int, iterations: int, cuda: bool = True) -> dict:
    """真实同步壁钟计时，记录每次样本与峰值显存。"""
    for _ in range(warmup):
        with torch.inference_mode():
            fn()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    values = []
    for _ in range(iterations):
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            fn()
        if cuda:
            torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return dict(
        warmup=warmup,
        iterations=iterations,
        raw_ms=values,
        mean_ms=float(np.mean(values)),
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20 if cuda else None,
    )


def benchmark(
    model: Any,
    wrapped: Any,
    pipeline: Any,
    manifest: dict,
    features: dict,
    text: dict,
    head: torch.nn.Module,
) -> dict:
    """同一六相机输入测原模型、冻结 V3 和 V4 分支；区分缓存与完整前向。"""
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from scripts.stage2.low_memory_dataset import bounded_metadata
    from scripts.stage2.joint_runtime import _first_tensor

    directory = OUT / "latency"
    directory.mkdir()
    row = next(r for r in manifest["annotations"] if r["split"] == "dev")
    for index in list(features):
        if index != row["index"]:
            del features[index]
    bev = features[row["index"]]
    grid, geometry = frame_inputs(row)
    with (V3 / "dataset_infos.pkl").open("rb") as f:
        info = next(
            i for i in pickle.load(f)["infos"] if i["token"] == row["sample_token"]
        )
    with (directory / "timing_info.pkl").open("xb") as f:
        pickle.dump(dict(infos=[info], metadata={"version": "v1.0-trainval"}), f)
    cfg = Config.fromfile(
        str(AUTHOR / "projects/configs/bevformer/bevformer_small_seg_det.py")
    )
    dc = copy.deepcopy(cfg.data.test)
    dc.ann_file = str(directory / "timing_info.pkl")
    dc.test_mode = True
    dc.pop("samples_per_gpu", None)
    with bounded_metadata({row["sample_token"]}):
        dataset = build_dataset(dc)
    data = get_batch(dataset, 0, SEED)
    # 同一真实 test-pipeline 图像捕获 BEV，和缓存双 Head 逐项比较。
    transformer = model.pts_bbox_head.transformer
    original_method = transformer.get_bev_features
    captured = {}

    def capture(*args: Any, **kwargs: Any) -> torch.Tensor:
        value = original_method(*args, **kwargs)
        captured["bev"] = BEVGrid().to_map(value).detach()
        return value

    transformer.get_bev_features = capture
    model.prev_frame_info = dict(
        prev_bev=None, scene_token=None, prev_pos=0, prev_angle=0
    )
    try:
        with torch.no_grad():
            direct = wrapped(
                return_loss=False, rescale=True, **forward_payload(data, inference=True)
            )[0]["pts_bbox"]
    finally:
        del transformer.get_bev_features
    with torch.no_grad():
        cached = decode(model, head_forward(model, captured["bev"]))
    errors = {}
    for k, key in (
        ("boxes", "boxes_3d"),
        ("scores", "scores_3d"),
        ("labels", "labels_3d"),
    ):
        expected = (
            direct[key].tensor.cpu().numpy()
            if k == "boxes"
            else direct[key].cpu().numpy()
        )
        errors[k] = float(np.abs(cached[k] - expected).max())
    assert max(errors.values()) < 1e-4
    with torch.no_grad():
        target_logits = head(captured["bev"], text[row["text"]], grid, geometry)
    np.savez_compressed(
        directory / "full_image_equivalence.npz",
        bev=captured["bev"].cpu().numpy(),
        target_logits=target_logits.cpu().numpy(),
        **cached,
    )
    write(
        directory / "full_image_equivalence.json",
        dict(
            sample_token=row["sample_token"],
            errors=errors,
            scope="real six-camera test-pipeline native boxes/scores/labels exactly agree with same captured FP32 BEV native Heads; V4 logits computed on this image BEV",
            training_cache_difference="training/dev48frames use frozen historical train-pipeline augmentation + float16 BEV cache; equivalence check uses fresh FP32 test image and is not a new accuracy dataset",
        ),
    )
    del captured["bev"], target_logits, direct, cached
    parameters = dict(
        author=sum(p.numel() for p in model.parameters()),
        frozen_v3_adapter=sum(p.numel() for p in pipeline.fusion.parameters()),
        frozen_text=sum(p.numel() for p in pipeline.encoder.parameters()),
        new_mask_head=sum(p.numel() for p in head.parameters()),
    )
    protocol = dict(
        sample_token=row["sample_token"],
        text=row["text"],
        image_shape=list(_first_tensor(data["img"]).shape),
        bev_shape=list(bev.shape),
        dtype=str(bev.dtype),
        cpu="geometry/calibration+single native Head CPU timing; no full six-camera CPU run",
        cached_gpu="original: native Heads; frozenV3: adapter+nativeHeads; V4: original nativeHeads+independentMask; cached genuineBEV and frozen text embedding, excludes backbone/BERT/IO",
        full_gpu="same real6camera image scatter+backbone/BEV+both nativeHeads/decode; V3 and V4 include full BERT; V4 adds maskHead on original capturedBEV, V3 retains historical scalar fusion; excludes IO, GT and fixed calibration setup",
        synchronization="CUDA synchronize before/after every measurement; perf_counter, inference_mode, batch1, temporal reset",
        warmup_cached=5,
        iterations_cached=30,
        warmup_full=3,
        iterations_full=10,
        parameter_count=parameters,
        gflops="NOT_VERIFIED",
        memory="one cache resident; move unused adapter/BERT/MaskHead to CPU per branch; retained allocator reservations reported separately",
    )
    write(directory / "protocol.json", protocol)
    result = {}
    for branch in ("original", "v3_frozen", "v4_independent_head"):
        pipeline.cpu()
        head.cpu()
        if branch != "original":
            pipeline.encoder.cuda()
        if branch == "v3_frozen":
            pipeline.fusion.cuda()
            pipeline.fusion.set_calibration(row)
        if branch == "v4_independent_head":
            head.cuda()
        release_memory()

        def cached_fn() -> Any:
            if branch == "v3_frozen":
                enhanced = pipeline.fusion.forward_with_details(bev, text[row["text"]])[
                    "enhanced_bev"
                ]
            else:
                enhanced = bev
            pred = head_forward(model, enhanced)
            mask = (
                head(bev, text[row["text"]], grid, geometry)
                if branch == "v4_independent_head"
                else None
            )
            return pred, mask

        def full_fn() -> Any:
            model.prev_frame_info = dict(
                prev_bev=None, scene_token=None, prev_pos=0, prev_angle=0
            )
            payload = forward_payload(data, inference=True)
            if branch == "v3_frozen":
                with pipeline.inject_shared_bev(
                    transformer, [row["text"]], strict=False
                ):
                    return wrapped(return_loss=False, rescale=True, **payload)
            if branch == "v4_independent_head":
                original = transformer.get_bev_features
                holder = {}

                def target_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
                    value = original(*args, **kwargs)
                    embedding = pipeline.encoder([row["text"]], "cuda")
                    holder["mask"] = head(
                        BEVGrid().to_map(value), embedding, grid, geometry
                    )
                    return value

                transformer.get_bev_features = target_forward
                try:
                    return wrapped(return_loss=False, rescale=True, **payload), holder
                finally:
                    del transformer.get_bev_features
            return wrapped(return_loss=False, rescale=True, **payload)

        result[branch] = dict(
            cached_gpu=timing(cached_fn, 5, 30),
            full_six_camera_gpu=timing(full_fn, 3, 10),
        )
        print(
            "V4_LATENCY",
            branch,
            {s: v["mean_ms"] for s, v in result[branch].items()},
            flush=True,
        )
    # CPU 原生双 Head 与几何生成、条件 Head 均实测；CPU 图像主干不估算。
    pipeline.cpu()
    head.cpu()
    model.cpu()
    cpu_bev, cpu_grid, cpu_geo = bev.cpu(), grid.cpu(), geometry.cpu()
    embedding = text[row["text"]].cpu()
    torch.set_num_threads(4)
    pipeline.fusion.set_calibration(row)
    cpu_result = {}
    for branch in ("original", "v3_frozen", "v4_independent_head"):

        def cpu_fn() -> Any:
            enhanced = (
                pipeline.fusion.forward_with_details(cpu_bev, embedding)["enhanced_bev"]
                if branch == "v3_frozen"
                else cpu_bev
            )
            prediction = head_forward(model, enhanced)
            mask = (
                head(cpu_bev, embedding, cpu_grid, cpu_geo)
                if branch == "v4_independent_head"
                else None
            )
            return prediction, mask

        cpu_result[branch] = timing(cpu_fn, 2, 10, cuda=False)
    write(
        directory / "metrics.json",
        dict(
            gpu=result,
            cpu_cached_native_heads=cpu_result,
            full_cpu="NOT_MEASURED",
            parameters=parameters,
            gflops="NOT_VERIFIED",
            protocol=protocol,
        ),
    )
    return result


def main() -> None:
    """诊断不更新权重，计时后不继续用开发指标改模型。"""
    argparse.ArgumentParser(description=__doc__).parse_args()
    model, wrapped, pipeline, dataset, manifest, features, text = load_runtime()
    training = json.loads((OUT / "training_summary.json").read_text())
    selected = training["branches"].get("full_instruction", training["diagnostic"])
    head = LanguageConditionedTargetMaskHead().cuda().eval()
    head.load_state_dict(
        torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)[
            "state_dict"
        ]
    )
    diagnose(model, pipeline, dataset, manifest, features, text, head)
    benchmark(model, wrapped, pipeline, manifest, features, text, head)


if __name__ == "__main__":
    main()

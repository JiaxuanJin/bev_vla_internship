"""V5 停训后的根因诊断、真实六相机推理及同步时延；禁止新优化。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.stage2.coupled_v5_data import OUT, SEED, sha, write
from scripts.stage2.coupled_v5_train import position, runtime, flat_gradient, objective
from scripts.stage2.coupled_v5_evaluate import adapters, enhanced
from scripts.stage2.conditional_mask_v4_experiment import frame_inputs, load_label
from scripts.stage2.conditional_mask_v4_diagnostics import timing
from scripts.stage2.spatial_annotations import AUTHOR
from scripts.stage2.spatial_v3_data import OUT as V3
from scripts.stage2.spatial_v3_experiment import decode, head_forward
from scripts.stage2.spatial_v3_range_correction import filter_ann
from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.micro_training import forward_payload, get_batch
from scripts.stage2.runtime_memory import release_memory
from scripts.stage2.stage3_metric_review import neighborhood


def diagnosis(
    model: Any,
    pipeline: Any,
    dataset: Any,
    manifest: dict,
    features: dict,
    text: dict,
    models: dict,
) -> None:
    """固定训练探针与全帧GT过滤核验，不从开发选择新超参数。"""
    from mmdet3d.datasets.pipelines import ObjectRangeFilter, ObjectNameFilter

    folder = OUT / "diagnosis"
    folder.mkdir()
    filters = []
    for frame in manifest["frames"]:
        raw = dataset.get_ann_info(frame["index"])
        actual, stats = filter_ann(raw, model.pts_bbox_head.pc_range)
        expected = ObjectNameFilter(list(model.CLASSES))(
            ObjectRangeFilter(model.pts_bbox_head.pc_range)(
                dict(
                    gt_bboxes_3d=raw["gt_bboxes_3d"].clone(),
                    gt_labels_3d=raw["gt_labels_3d"].copy(),
                )
            )
        )
        np.testing.assert_array_equal(
            actual["gt_bboxes_3d"].tensor.numpy(),
            expected["gt_bboxes_3d"].tensor.numpy(),
        )
        np.testing.assert_array_equal(actual["gt_labels_3d"], expected["gt_labels_3d"])
        filters.append(
            dict(index=frame["index"], sample_token=frame["sample_token"], **stats)
        )
    write(
        folder / "gt_filter_equivalence.json",
        dict(frames=filters, n_frames=len(filters), exact_author_pipeline=True),
    )
    objects = [
        r
        for r in json.loads((OUT / "objects_protocol.json").read_text())["annotations"]
        if r["split"] == "train"
    ]
    probes = []
    for scene in sorted(
        {r["scene_token"] for r in manifest["annotations"] if r["split"] == "train"}
    ):
        maps = [
            r
            for r in manifest["annotations"]
            if r["scene_token"] == scene and r["counterfactual_text"]
        ]
        if maps:
            probes.append(("map", maps[0]))
        candidates = [
            r for r in objects if r["scene_token"] == scene and r["relevant_gt_indices"]
        ]
        if candidates:
            probes.append(("object", candidates[0]))
    write(
        folder / "protocol.json",
        dict(
            probes=[
                dict(task=t, instruction_id=r["instruction_id"]) for t, r in probes
            ],
            optimizer_updates=0,
            selection="metadata-only first samequadrant paired map and nonempty object per train scene",
            heldout_used=False,
        ),
    )
    records = []
    adapter = models["targeted_language"]
    params = list(adapter.parameters())
    for task, row in probes:
        bev = features[row["index"]]
        pos = position(pipeline, row)
        with np.load(row["source_file"]) as f:
            gt = torch.tensor(f["gt"], device="cuda", dtype=torch.long)
            xy = torch.tensor(f["ego_xy"], device="cuda")
        ann, stats = filter_ann(
            dataset.get_ann_info(row["index"]), model.pts_bbox_head.pc_range
        )
        with torch.no_grad():
            base = head_forward(model, bev)
        frame = dict(
            gt=gt, xy=xy, teacher=base["seg_preds"].softmax(1).detach(), ann=ann
        )
        label = load_label(row) if task == "map" else None
        if label is not None:
            label["roi"] = (
                torch.tensor(neighborhood(label["target"].cpu().numpy()), device="cuda")
                & label["region"]
            )
        detail = adapter(bev, text[row["text"]], pos)
        pred = head_forward(model, detail["enhanced_bev"])
        loss, parts, trace = objective(model, pred, row, frame, label, True)
        grads = {
            k: flat_gradient(parts[k], params) for k in ("detection", "segmentation")
        }
        cf = row.get("counterfactual_text") if task == "map" else None
        if not cf:
            from scripts.stage2.spatial_v2_experiment import swap_direction

            cf = swap_direction(row["text"])
        with torch.no_grad():
            wrong = adapter(bev, text[cf], pos)
            wp = head_forward(model, wrong["enhanced_bev"])
        records.append(
            dict(
                task=task,
                instruction_id=row["instruction_id"],
                optimizer_updates=0,
                losses={k: float(v.detach()) for k, v in parts.items()},
                gradient_l2={k: float(v.norm()) for k, v in grads.items()},
                gradient_cosine=float(
                    F.cosine_similarity(
                        grads["detection"][None], grads["segmentation"][None]
                    )
                ),
                text_embedding_cosine=float(
                    F.cosine_similarity(text[row["text"]], text[cf])
                ),
                relative_bev_l2=float(
                    (detail["residual"].norm() / bev.norm()).detach()
                ),
                correct_wrong_bev_relative_l2=float(
                    (
                        (detail["enhanced_bev"] - wrong["enhanced_bev"]).norm()
                        / bev.norm()
                    ).detach()
                ),
                seg_logits_rms_delta=float(
                    (pred["seg_preds"] - base["seg_preds"])
                    .square()
                    .mean()
                    .sqrt()
                    .detach()
                ),
                det_logits_rms_delta=float(
                    (pred["all_cls_scores"] - base["all_cls_scores"])
                    .square()
                    .mean()
                    .sqrt()
                    .detach()
                ),
                bbox_rms_delta=float(
                    (pred["all_bbox_preds"] - base["all_bbox_preds"])
                    .square()
                    .mean()
                    .sqrt()
                    .detach()
                ),
                semantic_changed_pixels=int(
                    (pred["seg_preds"].argmax(1) != base["seg_preds"].argmax(1)).sum()
                ),
                correct_wrong_semantic_changed_pixels=int(
                    (pred["seg_preds"].argmax(1) != wp["seg_preds"].argmax(1)).sum()
                ),
                native_gt_filter=stats,
                weighted_assignment_trace=trace,
            )
        )
        print(
            "V5_FAILURE_DIAGNOSIS",
            task,
            row["instruction_id"],
            records[-1]["gradient_cosine"],
            flush=True,
        )
    write(
        folder / "metrics.json",
        dict(
            records=records,
            optimizer_updates=0,
            scope="training metadata probes only; no tuning after failed dev gate",
            conclusion="nonzero native task gradients and actual feature changes; detection loss/Seg loss conflict and weak correct-vs-wrong changes documented. Grounding and language-specific perception gain remain unproved.",
        ),
    )


def build_test_input(manifest: dict, sample_token: str | None, folder: Path) -> tuple:
    """固定已有训练/开发token构造真实六相机test输入，不新增留出评价。"""
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from scripts.stage2.low_memory_dataset import bounded_metadata
    from scripts.stage2.joint_runtime import _first_tensor

    rows = [r for r in manifest["annotations"] if r["split"] == "dev"]
    row = (
        next((r for r in rows if r["sample_token"] == sample_token), None)
        if sample_token
        else rows[0]
    )
    if row is None:
        raise ValueError("only frozen V5 development tokens allowed")
    with (V3 / "dataset_infos.pkl").open("rb") as f:
        info = next(
            r for r in pickle.load(f)["infos"] if r["token"] == row["sample_token"]
        )
    with (folder / "timing_info.pkl").open("xb") as f:
        pickle.dump(dict(infos=[info], metadata={"version": "v1.0-trainval"}), f)
    cfg = Config.fromfile(
        str(AUTHOR / "projects/configs/bevformer/bevformer_small_seg_det.py")
    )
    dc = copy.deepcopy(cfg.data.test)
    dc.ann_file = str(folder / "timing_info.pkl")
    dc.test_mode = True
    dc.pop("samples_per_gpu", None)
    with bounded_metadata({row["sample_token"]}):
        dataset = build_dataset(dc)
    data = get_batch(dataset, 0, SEED)
    return row, data, list(_first_tensor(data["img"]).shape)


def full_inference(
    model: Any,
    wrapped: Any,
    pipeline: Any,
    mask_head: Any,
    adapter: Any,
    branch: str,
    data: dict,
    row: dict,
    grid: torch.Tensor,
    geo: torch.Tensor,
    pos: torch.Tensor,
    capture: bool = False,
) -> tuple:
    """真实图像主干产生BEV后注入条件残差，再由原生Head推理。

    Args:
        model: 原作者权重冻结模型。
        wrapped: 实际MMCV单GPU包装。
        pipeline: 冻结BERT及V3对照。
        mask_head: 仅V4/V5辅助Grounding输出。
        adapter: 已训练的V5参数，无优化器。
        branch: original、v3_historical、v4_independent或v5_coupled。
        data: 同一真实六相机test-pipeline样本。
        row: 公开开发指令及标定，不传GT。
        grid: 校准的物理采样网格。
        geo: 解析Ego坐标。
        pos: 原生解析位置特征。
        capture: 是否保存真实原生logits及BEV。

    Returns:
        原作者decode结果及新增Mask/特征追踪。
    """
    transformer = model.pts_bbox_head.transformer
    original = transformer.get_bev_features
    captured = {}
    handle = None
    if capture:

        def capture_head(module: Any, args: tuple, output: dict) -> None:
            for k in ("seg_preds", "all_cls_scores", "all_bbox_preds"):
                captured[k] = output[k].detach().clone()

        handle = model.pts_bbox_head.register_forward_hook(capture_head)

    def inject(*args: Any, **kwargs: Any) -> torch.Tensor:
        native = original(*args, **kwargs)
        bev = BEVGrid().to_map(native)
        if capture:
            captured["original_bev"] = bev.detach().clone()
        embedding = (
            None if branch == "original" else pipeline.encoder([row["text"]], "cuda")
        )
        if branch == "v3_historical":
            value = pipeline.fusion.forward_with_details(bev, embedding)["enhanced_bev"]
        elif branch == "v5_coupled":
            value = adapter(bev, embedding, pos)["enhanced_bev"]
        else:
            value = bev
        if branch in ("v4_independent", "v5_coupled"):
            captured["target_mask_logits"] = mask_head(value, embedding, grid, geo)
        if capture:
            captured["enhanced_bev"] = value.detach().clone()
        return BEVGrid().to_sequence(value)

    transformer.get_bev_features = inject
    model.prev_frame_info = dict(
        prev_bev=None, scene_token=None, prev_pos=0, prev_angle=0
    )
    try:
        result = wrapped(
            return_loss=False, rescale=True, **forward_payload(data, inference=True)
        )
    finally:
        del transformer.get_bev_features
        if handle is not None:
            handle.remove()
    return result, captured


def benchmark(
    model: Any,
    wrapped: Any,
    pipeline: Any,
    mask_head: Any,
    manifest: dict,
    features: dict,
    text: dict,
    models: dict,
    mode: str,
    output: Path | None,
    sample_token: str | None,
) -> None:
    """同条件Original/V3/V4/V5计时并保存真正耦合的完整图像输出。"""
    folder = output.resolve() if output else OUT / "runtime"
    folder.mkdir(parents=True, exist_ok=False)
    row, data, shape = build_test_input(manifest, sample_token, folder)
    features.clear()
    pipeline.fusion.last_bev = None
    grid, geo = frame_inputs(row)
    pos = position(pipeline, row)
    adapter = models["targeted_language"]
    with torch.no_grad():
        full, original = full_inference(
            model,
            wrapped,
            pipeline,
            mask_head,
            adapter,
            "original",
            data,
            row,
            grid,
            geo,
            pos,
            True,
        )
        _, v4 = full_inference(
            model,
            wrapped,
            pipeline,
            mask_head,
            adapter,
            "v4_independent",
            data,
            row,
            grid,
            geo,
            pos,
            True,
        )
        result, v5 = full_inference(
            model,
            wrapped,
            pipeline,
            mask_head,
            adapter,
            "v5_coupled",
            data,
            row,
            grid,
            geo,
            pos,
            True,
        )
        cached = head_forward(model, v5["enhanced_bev"])
    errors = {
        k: float((cached[k] - v5[k]).abs().max())
        for k in ("seg_preds", "all_cls_scores", "all_bbox_preds")
    }
    original_v4 = {k: float((original[k] - v4[k]).abs().max()) for k in errors}
    assert max(errors.values()) == 0 and max(original_v4.values()) == 0
    with torch.no_grad():
        decoded = decode(model, cached)
    np.savez_compressed(
        folder / "real_six_camera_prediction.npz",
        **decoded,
        semantic_logits=v5["seg_preds"].cpu().numpy(),
        target_mask_logits=v5["target_mask_logits"].cpu().numpy(),
        original_semantic_logits=original["seg_preds"].cpu().numpy(),
    )
    selected = json.loads((OUT / "pilot_training_summary.json").read_text())[
        "branches"
    ]["targeted_language"]["selected_terminal"]
    write(
        folder / "inference.json",
        dict(
            sample_token=row["sample_token"],
            text=row["text"],
            checkpoint=selected,
            scope="new V5 adapter runs BETWEEN real imageBEV backbone and native jointHeads; GT unused in inference",
            v5_full_vs_cached_head_max_errors=errors,
            original_v4_native_logit_max_errors=original_v4,
            release_status="research-only failed dev gate; not improved production model",
        ),
    )
    if mode == "infer":
        return
    bev = original["original_bev"]
    del full, result, original, v4, v5, cached
    for name in list(models):
        if name != "targeted_language":
            del models[name]
    parameters = dict(
        author=sum(p.numel() for p in model.parameters()),
        frozen_text=sum(p.numel() for p in pipeline.encoder.parameters()),
        v3_adapter=sum(p.numel() for p in pipeline.fusion.parameters()),
        v4_mask=sum(p.numel() for p in mask_head.parameters()),
        v5_adapter=sum(p.numel() for p in adapter.parameters()),
    )
    protocol = dict(
        sample_token=row["sample_token"],
        text=row["text"],
        image_shape=shape,
        bev_shape=list(bev.shape),
        dtype=str(bev.dtype),
        cached="same fresh real-image FP32 BEV, cached text +native doubleHead; V4 addsfrozenMask,V5 addsadapter+frozenMask,V3 addsoldfusion; no IO/backbone/BERT",
        full="same sixcamera test image scatter+backbone/BEV+native doubleHead/decode; includes full BERT for V3/V4/V5 and auxiliary Mask for V4/V5; no IO/GT/fixed calibration setup",
        warmup_cached=5,
        iterations_cached=30,
        warmup_full=3,
        iterations_full=10,
        synchronization="CUDA synchronize eachstart/end; perf_counter,inference_mode,temporal reset; fixed branchorder",
        memory="unused BERT/adapters/Mask movedCPU; one FP32 BEV resident, report allocated/reserved peaks; not a fresh process per branch",
        parameters=parameters,
        gflops="NOT_VERIFIED",
        cpu="same cached native-head scopes, fourthreads,2warm/10runs; full CPU image path NOT_MEASURED",
    )
    write(folder / "protocol.json", protocol)
    metrics = {}
    for branch in ("original", "v3_historical", "v4_independent", "v5_coupled"):
        pipeline.cpu()
        mask_head.cpu()
        adapter.cpu()
        if branch != "original":
            pipeline.encoder.cuda()
        if branch == "v3_historical":
            pipeline.fusion.cuda()
            pipeline.fusion.set_calibration(row)
        if branch in ("v4_independent", "v5_coupled"):
            mask_head.cuda()
        if branch == "v5_coupled":
            adapter.cuda()
        release_memory()

        def cached_fn() -> Any:
            if branch == "v3_historical":
                value = pipeline.fusion.forward_with_details(bev, text[row["text"]])[
                    "enhanced_bev"
                ]
            elif branch == "v5_coupled":
                value = adapter(bev, text[row["text"]], pos)["enhanced_bev"]
            else:
                value = bev
            pred = head_forward(model, value)
            mask = (
                mask_head(value, text[row["text"]], grid, geo)
                if branch in ("v4_independent", "v5_coupled")
                else None
            )
            return pred, mask

        def full_fn() -> Any:
            return full_inference(
                model,
                wrapped,
                pipeline,
                mask_head,
                adapter,
                branch,
                data,
                row,
                grid,
                geo,
                pos,
            )

        metrics[branch] = dict(
            cached_gpu=timing(cached_fn, 5, 30),
            full_six_camera_gpu=timing(full_fn, 3, 10),
        )
        print(
            "V5_TIMING",
            branch,
            {k: v["mean_ms"] for k, v in metrics[branch].items()},
            flush=True,
        )
    model.cpu()
    pipeline.cpu()
    mask_head.cpu()
    adapter.cpu()
    torch.set_num_threads(4)
    cb, ce, cg, cp = bev.cpu(), text[row["text"]].cpu(), grid.cpu(), pos.cpu()
    cgf = geo.cpu()
    pipeline.fusion.set_calibration(row)
    cpu = {}
    for branch in ("original", "v3_historical", "v4_independent", "v5_coupled"):

        def cpu_fn() -> Any:
            if branch == "v3_historical":
                value = pipeline.fusion.forward_with_details(cb, ce)["enhanced_bev"]
            elif branch == "v5_coupled":
                value = adapter(cb, ce, cp)["enhanced_bev"]
            else:
                value = cb
            pred = head_forward(model, value)
            mask = (
                mask_head(value, ce, cg, cgf)
                if branch in ("v4_independent", "v5_coupled")
                else None
            )
            return pred, mask

        cpu[branch] = timing(cpu_fn, 2, 10, cuda=False)
    write(
        folder / "metrics.json",
        dict(
            gpu=metrics,
            cpu_cached=cpu,
            full_cpu="NOT_MEASURED",
            gflops="NOT_VERIFIED",
            protocol=protocol,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "infer"), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-token")
    args = parser.parse_args()
    model, wrapped, pipeline, dataset, manifest, features, text, mask_head = runtime()
    models, _ = adapters("pilot")
    if args.mode == "all":
        diagnosis(model, pipeline, dataset, manifest, features, text, models)
    benchmark(
        model,
        wrapped,
        pipeline,
        mask_head,
        manifest,
        features,
        text,
        models,
        args.mode,
        args.output_dir,
        args.sample_token,
    )

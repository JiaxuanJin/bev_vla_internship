"""真实训练场景的运行时坐标链诊断与最小标定桥接对照，不更新权重。"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.spatial_annotations import AUTHOR, DATA, load_module
from scripts.stage2.spatial_report import confusion, summary
from scripts.stage3.lg_bev_common import DEFAULT_OUT, ROOT, read, sha, write
from scripts.stage3.lg_bev_coordinate_core import (
    COORD_OUT,
    calibrated_cropper,
    legal_frames,
    native_to_segment,
    physical_ramp,
    ramp_error,
    require_runtime_head,
)
from scripts.stage3.lg_bev_native import dual_head, load_native, module_sha
from scripts.stage3.lg_bev_train import frame_input


def trace_ramp(
    model: torch.nn.Module, frame: dict, calibrated: bool
) -> tuple[dict, Tensor]:
    """在真实 Head 入口注入合成 ramp，hook 实际 cropper 和 decoder 张量。"""
    from contextlib import nullcontext

    head = model.pts_bbox_head
    ramp = physical_ramp(frame, device="cuda")
    captured = {}

    def hook(name):
        def save(_module, args, value):
            captured[name] = (args[0].detach().clone(), value.detach().clone())

        return save

    handles = [
        head.feat_cropper.register_forward_hook(hook("cropper")),
        head.seg_decoder.register_forward_hook(hook("decoder")),
    ]
    context = calibrated_cropper(head, frame) if calibrated else nullcontext()
    try:
        with context, torch.no_grad():
            prediction = dual_head(model, ramp)
    finally:
        for handle in handles:
            handle.remove()
    expected = native_to_segment(ramp)
    assert torch.equal(captured["cropper"][0], expected)
    assert torch.equal(captured["cropper"][1], captured["decoder"][0])
    assert torch.equal(captured["decoder"][1], prediction["seg_preds"])
    measured = captured["cropper"][1]
    result = dict(
        **ramp_error(measured, frame),
        input_matches_active_transform_bitwise=True,
        cropper_to_decoder_bitwise=True,
        decoder_to_final_logits_bitwise=True,
        cropper_shape=list(measured.shape),
        final_shape=list(prediction["seg_preds"].shape),
        calibration_only=calibrated,
    )
    return result, measured[0, :3].detach().cpu()


def decoder_sensitivity(head: torch.nn.Module, cropped: Tensor) -> dict:
    """局部 logit 对 cropper 输入的梯度中心，描述 learned 路径而不宣称因果补偿。"""
    value = cropped.detach().requires_grad_(True)
    logits = head.seg_decoder(value)
    centres = []
    for row, col in ((50, 100), (100, 200), (150, 300)):
        scalar = logits[0, 1, row, col] - logits[0, 0, row, col]
        gradient = torch.autograd.grad(scalar, value, retain_graph=True)[0]
        mass = gradient.detach().abs().sum(1)[0]
        yy, xx = torch.meshgrid(
            torch.arange(200, device=value.device),
            torch.arange(400, device=value.device),
            indexing="ij",
        )
        denominator = mass.sum().clamp_min(1e-20)
        centres.append(
            dict(
                output_pixel=[row, col],
                absolute_gradient_centroid=[
                    float((yy * mass).sum() / denominator),
                    float((xx * mass).sum() / denominator),
                ],
                gradient_l1=float(denominator),
            )
        )
    return dict(
        samples=centres,
        interpretation="内容相关卷积感受野可以包含偏移；此诊断不证明或排除 learned calibration compensation",
    )


def run(out: Path, fresh_cameras: bool) -> None:
    """冻结权重，固定四个训练场景首帧；先登记，无基于结果的结构选择。"""
    from scripts.stage3.lg_bev_coordinate_core import raw_map_gt

    frames = legal_frames()
    selected = [
        next(f for f in frames if f["scene_name"] == scene)
        for scene in sorted({f["scene_name"] for f in frames})
    ]
    config = AUTHOR / "projects/configs/bevformer/bevformer_small_seg_det.py"
    write(
        out / "rca_protocol.json",
        dict(
            scope="four original train scenes first frame; no internal_dev or sealed scene",
            selected_frames=selected,
            original_lg_protocol_sha256=sha(DEFAULT_OUT / "protocol.json"),
            cfg_sha256=sha(config),
            optimizer_updates=0,
            candidate="one analytic calibration-only parameterless cropper bridge; no fitted transform/threshold",
            checks="actual runtime class/hooks, raw map+pose+calibration, synthetic line/ramp, raw camera forward, frozen state, native detection invariant, calibrated ramp<1e-4m",
            no_performance_selection="Report both legacy and analytic bridge on all fixed train frames; no alternative geometry trials",
            source_sha256={
                str(p.relative_to(ROOT)): sha(p)
                for p in (ROOT / "scripts/stage3").glob("lg_bev_coordinate*.py")
            },
        ),
    )
    model, wrapped, encoder, dataset = load_native()
    head = model.pts_bbox_head
    runtime = require_runtime_head(head)
    runtime["configured_head_type"] = "BEVFormerHead"
    write(out / "runtime_chain.json", runtime)
    before = module_sha(model)
    tools = AUTHOR / "projects/mmdet3d_plugin/datasets/utils"
    vector = load_module("coordinate_raw_vector", tools / "vector_map.py")
    raster = load_module("coordinate_raw_raster", tools / "rasterize.py")
    mapper = vector.VectorizedLocalMap(str(DATA), (30.0, 60.0), (200, 400))
    records, arrays = [], {}
    aggregate = {
        name: np.zeros((4, 4), dtype=np.int64) for name in ("native", "calibrated")
    }
    for index, frame in enumerate(selected):
        info = dataset.data_infos[frame["index"]]
        assert info["token"] == frame["sample_token"]
        gt, lineage = raw_map_gt(frame, info, mapper, raster)
        with np.load(frame["source_file"]) as cached:
            np.testing.assert_array_equal(gt, cached["gt"])
        # 原始 table 再读取逐帧标定与位姿，不只信任缓存的元数据。
        sample_data = [
            r
            for r in read(DATA / "v1.0-mini/sample_data.json")
            if r["sample_token"] == frame["sample_token"] and r["is_key_frame"]
        ]
        calib = {r["token"]: r for r in read(DATA / "v1.0-mini/calibrated_sensor.json")}
        sensors = {r["token"]: r for r in read(DATA / "v1.0-mini/sensor.json")}
        lidar_record = next(
            r
            for r in sample_data
            if sensors[calib[r["calibrated_sensor_token"]]["sensor_token"]]["channel"]
            == "LIDAR_TOP"
        )
        original_calib = calib[lidar_record["calibrated_sensor_token"]]
        poses = {r["token"]: r for r in read(DATA / "v1.0-mini/ego_pose.json")}
        pose = poses[lidar_record["ego_pose_token"]]
        for key in ("rotation", "translation"):
            np.testing.assert_allclose(original_calib[key], frame[f"lidar2ego_{key}"])
            np.testing.assert_allclose(pose[key], frame[f"ego2global_{key}"])
        native_ramp, native_values = trace_ramp(model, frame, False)
        calibrated_ramp, calibrated_values = trace_ramp(model, frame, True)
        bev, _ = frame_input(frame)
        crop_inputs = []
        handle = head.seg_decoder.register_forward_pre_hook(
            lambda _m, args: crop_inputs.append(args[0].detach())
        )
        with torch.no_grad():
            native = dual_head(model, bev)
        handle.remove()
        with calibrated_cropper(head, frame), torch.no_grad():
            corrected = dual_head(model, bev)
        for key in ("all_cls_scores", "all_bbox_preds"):
            assert torch.equal(native[key], corrected[key]), key
        with torch.no_grad():
            restored = dual_head(model, bev)
        for key in ("all_cls_scores", "all_bbox_preds", "seg_preds"):
            assert torch.equal(native[key], restored[key]), key
        gt_tensor = torch.as_tensor(gt.astype(np.int64), device="cuda")[None]
        metrics = {}
        for name, prediction in (("native", native), ("calibrated", corrected)):
            semantic = prediction["seg_preds"].argmax(1)[0].cpu().numpy()
            matrix = confusion(gt, semantic, np.ones_like(gt, dtype=bool))
            aggregate[name] += matrix
            metrics[name] = dict(
                **summary(matrix),
                native_seg_loss=float(head.seg_loss(gt_tensor, prediction)["loss_seg"]),
                plain_ce=float(F.cross_entropy(prediction["seg_preds"], gt_tensor)),
            )
            arrays[f"frame{index}_{name}_semantic"] = semantic
        arrays.update(
            {
                f"frame{index}_gt": gt,
                f"frame{index}_native_ramp": native_values.numpy(),
                f"frame{index}_calibrated_ramp": calibrated_values.numpy(),
            }
        )
        record = dict(
            sample_token=frame["sample_token"],
            scene_name=frame["scene_name"],
            native_ramp=native_ramp,
            calibrated_ramp=calibrated_ramp,
            raw_map_lineage=lineage,
            semantic_metrics_training_only=metrics,
            raw_gt_matches_cache_bitwise=True,
            detection_outputs_unchanged=True,
            native_restored_bitwise=True,
            cached_bev_path=frame["cache_file"],
            cache_reuse=True,
        )
        if index == 0:
            record["decoder_sensitivity"] = decoder_sensitivity(head, crop_inputs[0])
        records.append(record)
        print(
            "COORD_FRAME",
            frame["scene_name"],
            native_ramp["mean_lidar_xy_m"],
            calibrated_ramp["max_lidar_xy_m"],
            flush=True,
        )
    assert module_sha(model) == before
    write(
        out / "coordinate_evidence.json",
        dict(
            status="RUNTIME_HEAD_MISMATCH_CONFIRMED",
            records=records,
            aggregate_training_only={
                name: summary(matrix) for name, matrix in aggregate.items()
            },
            frozen_before=before,
            frozen_after=module_sha(model),
            optimizer_updates=0,
            previous_15m_report="INVALID_FOR_ACTIVE_MODEL: it manually applied inactive BEVFormerHead_seg transform, not the runtime BEVFormerHead transform",
            explicit_downstream_compensation="No coordinate warp/flip/calibration in active SegEncode; hook proves decoder output is final logits unchanged",
            learned_compensation="possible; native vs single analytic bridge train-only metrics are descriptive, no fitted weights/coordinates",
        ),
    )
    with (out / "coordinate_arrays.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    if fresh_cameras:
        fresh_forward(out, model, wrapped, dataset, selected[0])
    draw(out, records, arrays)
    print("COORDINATE_RCA_COMPLETE", flush=True)


def fresh_forward(
    out: Path, model: torch.nn.Module, wrapped: Any, dataset: Any, frame: dict
) -> None:
    """执行单帧真实六相机主干，不把读取相机或 cached BEV 称为主干前向。"""
    from scripts.stage2.micro_training import forward_payload, get_batch
    from scripts.stage2.low_memory_dataset import bounded_metadata
    from scripts.stage2.joint_runtime import _first_tensor
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    import pickle

    captured = []
    transformer = model.pts_bbox_head.transformer
    original = transformer.get_bev_features

    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        captured.append(value.detach().clone())
        return value

    # inference 需要 test pipeline 的 list/meta 结构，不能直接传 train queue dict。
    info_path = out / "fresh_training_frame_info.pkl"
    if info_path.exists():
        with info_path.open("rb") as stream:
            recorded_info = pickle.load(stream)
        if [r["token"] for r in recorded_info["infos"]] != [frame["sample_token"]]:
            raise ValueError("已有单帧元数据不匹配，拒绝覆盖")
    else:
        with info_path.open("xb") as stream:
            pickle.dump(
                dict(
                    infos=[dataset.data_infos[frame["index"]]],
                    metadata={"version": "v1.0-mini"},
                ),
                stream,
            )
    config = Config.fromfile(
        str(AUTHOR / "projects/configs/bevformer/bevformer_small_seg_det.py")
    )
    dataset_config = copy.deepcopy(config.data.test)
    dataset_config.ann_file = str(info_path)
    dataset_config.test_mode = True
    dataset_config.pop("samples_per_gpu", None)
    with bounded_metadata({frame["sample_token"]}):
        test_dataset = build_dataset(dataset_config)
    data = get_batch(test_dataset, 0, 20261027)
    image_shape = list(_first_tensor(data["img"]).shape)
    transformer.get_bev_features = capture
    model.prev_frame_info = dict(
        prev_bev=None, scene_token=None, prev_pos=0, prev_angle=0
    )
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            result = wrapped(
                return_loss=False, rescale=True, **forward_payload(data, inference=True)
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        transformer.get_bev_features = original
    assert len(captured) == 1
    fresh = BEVGrid().to_map(captured[0])
    cached, _ = frame_input(frame)
    # 测试流水线重新生成的原始地图 GT，与独立重建记录逐像素核对。
    actual_gt = _first_tensor(data["semantic_indices"]).reshape(200, 400).cpu().numpy()
    with np.load(frame["source_file"]) as source:
        np.testing.assert_array_equal(actual_gt, source["gt"])
    write(
        out / "raw_camera_forward.json",
        dict(
            status="PASS",
            scene_name=frame["scene_name"],
            sample_token=frame["sample_token"],
            six_cameras_forwarded=True,
            actual_model_input_images=image_shape,
            test_pipeline_raw_gt_matches_frozen_gt=True,
            bev_shape=list(fresh.shape),
            cached_bev_reused_for_forward=False,
            cached_comparison_only=True,
            seconds=elapsed,
            fresh_cached_relative_l2=float((fresh - cached).norm() / fresh.norm()),
            temporal_history=False,
            checkpoint_loaded_from="/mnt/g/nus/bevformer_small_joint_author.pth",
            decoded_boxes=int(len(result[0]["pts_bbox"]["scores_3d"])),
            warning="single frame, fresh backbone forward includes preprocessing convention differences and float16 cache quantization; not performance evaluation",
        ),
    )


def draw(out: Path, records: list[dict], arrays: dict) -> None:
    """四个固定训练帧展示原始 GT、原生与单一标定桥接输出。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for k, record in enumerate(records):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for ax, name in zip(axes, ("gt", "native_semantic", "calibrated_semantic")):
            ax.imshow(
                arrays[f"frame{k}_{name}"],
                origin="lower",
                extent=(-30, 30, -15, 15),
                vmin=0,
                vmax=3,
                interpolation="nearest",
            )
            ax.set_title(name)
            ax.set_xlabel("Ego x / m")
            ax.set_ylabel("Ego y / m")
        fig.suptitle(record["scene_name"] + " / " + record["sample_token"])
        path = out / "figures" / f"training_comparison_{k}.png"
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=COORD_OUT)
    parser.add_argument("--fresh-cameras", action="store_true")
    parser.add_argument("--fresh-only", action="store_true")
    args = parser.parse_args()
    if args.fresh_only:
        model, wrapped, _encoder, dataset = load_native()
        fresh_forward(args.out.resolve(), model, wrapped, dataset, legal_frames()[0])
    else:
        run(args.out.resolve(), args.fresh_cameras)

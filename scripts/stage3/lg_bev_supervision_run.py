"""保留原生双 Head 的监督空间验收及唯一16步 LG-BEV pilot。"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage3.lg_bev_common import (
    DEFAULT_OUT,
    ROOT,
    SEED,
    read,
    sha,
    verify_protocol,
    write,
)
from scripts.stage3.lg_bev_coordinate_core import (
    COORD_OUT,
    legal_frames,
    physical_ramp,
    require_runtime_head,
)
from scripts.stage3.lg_bev_data import load_supervision
from scripts.stage3.lg_bev_native import (
    dual_head,
    gradient_audit,
    load_native,
    module_sha,
)
from scripts.stage3.lg_bev_prototype import LanguageGroundedBEVAdapter
from scripts.stage3.lg_bev_supervision_space import (
    AttentionSupervisionSpace,
    bilinear_adjoint_check,
    objective,
)
from scripts.stage3.lg_bev_train import encode, frame_input, make_schedule

SPACE_OUT = ROOT / "results/lg_bev_supervision_space_20261027"


def prepare(out: Path) -> None:
    """在新训练/开发评价前固定同一模型、split、seed、权重及16步预算。"""
    old = verify_protocol(DEFAULT_OUT)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(DEFAULT_OUT / "manifest.json", out / "manifest.json")
    (out / "labels").symlink_to(DEFAULT_OUT / "labels", target_is_directory=True)
    protocol = copy.deepcopy(old)
    protocol["split_id"] = (
        "lg_bev_20261027_same_split_native_heads_attention_space_audit"
    )
    protocol["parent_protocol_sha256"] = sha(DEFAULT_OUT / "protocol.json")
    protocol["registration_note"] = (
        "用户要求只适配独立 Attention 监督空间；不替换 cropper，不修改作者源码/权重/原生双Head；数值监督投影与旧正确标定公式相同，修复错误的审计对象及门槛。"
    )
    protocol["mapping"] = dict(
        native_bev="LiDAR XY row=y,col=x;150x150;+/-51.2m;pixel centres",
        attention="same native LiDAR grid; map supervision only uses calibration inverse at Ego GT centres",
        detection="unchanged native head, LiDAR boxes, gravity centre through calibration for instruction partition",
        segmentation="unchanged active BEVFormerHead transpose+flip2 -> native cropper -> learned SegEncode -> original Ego-raster GT; do not assume cropper is pointwise output",
        explicit_separation="GT transforms Attention logits only; no GT/cropper/decoder/baseline warp",
    )
    protocol["gate"] = (
        "Raw map/annotation/calibration PASS; raw train-camera replay equals historical float16 cache; "
        "runtime active Head identified; calibrated Attention ramp<1e-4m; bilinear adjoint<1e-10; "
        "fresh full native head outputs equal cache-boundary replay bitwise; original pre/post outputs bitwise; "
        "frozen states/grad invariants; real per-task backward finite; tests PASS. "
        "No requirement that pre-decoder feature sampler equal post-decoder GT pointwise."
    )
    for name in (
        "runtime_chain.json",
        "independent_gt.json",
        "cache_replay.json",
        "raw_camera_forward.json",
    ):
        protocol["lineage_sha256"][str(COORD_OUT / name)] = sha(COORD_OUT / name)
    sources = [
        *sorted((ROOT / "scripts/stage3").glob("lg_bev_supervision*.py")),
        ROOT / "scripts/stage3/lg_bev_coordinate_core.py",
    ]
    protocol["source_sha256"].update(
        {str(p.relative_to(ROOT)): sha(p) for p in sources}
    )
    protocol["budget"]["smoke_updates_this_run"] = 0
    protocol["pilot_policy"] = (
        "one terminal16-step run; no expansion or architecture/threshold/lr retries; same cached augmentation for all paired branches"
    )
    write(out / "protocol.json", protocol)
    rows = read(out / "manifest.json")["annotations"]
    write(out / "schedule.json", [r["instruction_id"] for r in make_schedule(rows, 16)])
    print("SUPERVISION_PROTOCOL", sha(out / "protocol.json"), flush=True)


def audit(out: Path) -> None:
    """真实训练数据验收独立监督，原始模型完整路径及预测保持逐位一致。"""
    from scripts.stage2.micro_training import forward_payload, get_batch
    from scripts.stage2.spatial_annotations import metric_grid

    protocol = verify_protocol(out)
    model, wrapped, encoder, dataset = load_native()
    head = model.pts_bbox_head
    runtime = require_runtime_head(head)
    assert "forward" not in head.feat_cropper.__dict__
    cropper_method = head.feat_cropper.forward.__func__
    before = dict(author=module_sha(model), text=module_sha(encoder))
    frames = [
        next(f for f in legal_frames() if f["scene_name"] == scene)
        for scene in sorted({f["scene_name"] for f in legal_frames()})
    ]
    rows = read(out / "manifest.json")["annotations"]
    map_checks = []
    for frame in frames:
        row = next(
            r
            for r in rows
            if r["sample_token"] == frame["sample_token"] and r["task"] == "map"
        )
        label = load_supervision(row, out, training=True)
        space = AttentionSupervisionSpace.from_calibration("map", frame)
        projected = (
            space.project(physical_ramp(frame, channels=3).double())[0]
            .permute(1, 2, 0)
            .numpy()
        )
        _, _, expected = metric_grid(frame)
        error = float(np.linalg.norm(projected - expected, axis=-1).max())
        assert error < 1e-4
        pixels = np.argwhere(label["target"][0, 0].numpy())
        selected_pixel = tuple(int(k) for k in pixels[len(pixels) // 2])
        adjoint = bilinear_adjoint_check(space, selected_pixel)
        assert adjoint["pass_check"]
        # 独立标定映射数值兼容原先正确的 Attention 公式；不做额外平移/翻转。
        old_grid = metric_grid(frame)[0]
        assert torch.equal(old_grid, space.sampling_grid)
        map_checks.append(
            dict(
                scene_name=frame["scene_name"],
                sample_token=frame["sample_token"],
                max_xyz_ramp_error_m=error,
                bilinear_adjoint=adjoint,
                target_pixels=int(label["target"].sum()),
            )
        )
    frame = frames[0]
    obj = next(
        r
        for r in rows
        if r["sample_token"] == frame["sample_token"]
        and r["task"] == "object"
        and r["qualifier"] == "nearest"
        and r["same_direction_distractors"] > 0
    )
    # 从原始6相机重新前向，捕获实际 native BEV 与原生双 Head 输出。
    data = get_batch(dataset, frame["index"], SEED + frame["index"])
    captured = {}
    transformer = head.transformer
    original = transformer.get_bev_features

    def capture_bev(*args, **kwargs):
        value = original(*args, **kwargs)
        captured["bev"] = BEVGrid().to_map(value).detach().clone()
        return value

    def capture_head(_module, _args, prediction):
        captured["prediction"] = {
            k: prediction[k].detach().clone()
            for k in ("all_cls_scores", "all_bbox_preds", "seg_preds")
        }

    transformer.get_bev_features = capture_bev
    handle = head.register_forward_hook(capture_head)
    try:
        with torch.no_grad():
            wrapped(return_loss=True, **forward_payload(data))
    finally:
        transformer.get_bev_features = original
        handle.remove()
    fresh = captured["bev"]
    with torch.no_grad():
        replay = dual_head(model, fresh)
    for key, value in captured["prediction"].items():
        assert torch.equal(value, replay[key]), key
    cached, position = frame_input(frame)
    assert torch.equal(fresh.half().float(), cached)
    with torch.no_grad():
        original_prediction = {
            k: v.detach().clone()
            for k, v in dual_head(model, cached).items()
            if k in captured["prediction"]
        }
    text = encode(encoder, [obj])
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    adapter = LanguageGroundedBEVAdapter(**protocol["model_config"]).cuda()
    detail = adapter(cached, text[obj["text"]], position)
    prediction = dual_head(model, detail["enhanced_bev"])
    assert torch.equal(detail["enhanced_bev"], cached)
    for key, value in original_prediction.items():
        assert torch.equal(value, prediction[key]), key
    total, parts, _ = objective(
        model,
        prediction,
        detail,
        obj,
        out,
        dataset.get_ann_info(obj["index"]),
        original_prediction["seg_preds"].softmax(1),
        protocol,
    )
    gradients = {
        key: gradient_audit(parts[key], adapter)
        for key in ("detection", "segmentation", "mask")
    }
    assert all(v["all_finite"] and v["total_l2"] > 0 for v in gradients.values())
    total.backward()
    with torch.no_grad():
        wrong = adapter(cached, text[obj["wrong_text"]], position)
    language_delta = float(
        (detail["attention_logits"].detach() - wrong["attention_logits"]).abs().max()
    )
    assert language_delta > 1e-8
    with torch.no_grad():
        after_prediction = dual_head(model, cached)
    assert all(
        torch.equal(after_prediction[k], value)
        for k, value in original_prediction.items()
    )
    assert before == dict(author=module_sha(model), text=module_sha(encoder))
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    assert head.feat_cropper.forward.__func__ is cropper_method
    assert "forward" not in head.feat_cropper.__dict__
    assert read(COORD_OUT / "cache_replay.json")["rounded_fresh_matches_cache_bitwise"]
    assert (
        read(COORD_OUT / "independent_gt.json")["status"]
        == "PASS_COORDINATE_AND_ANNOTATION_CHAIN"
    )
    assert read(COORD_OUT / "raw_camera_forward.json")[
        "test_pipeline_raw_gt_matches_frozen_gt"
    ]
    report = dict(
        status="PASS",
        pilot_allowed=True,
        optimizer_updates=0,
        runtime_head=runtime["head_class"],
        decoder=runtime["decoder_class"],
        attention_mapping_checks=map_checks,
        per_task_gradients=gradients,
        frozen_before=before,
        frozen_after=before,
        native_cropper_forward_untouched=True,
        baseline_identity_bitwise=True,
        fresh_full_model_to_native_replay_bitwise=True,
        baseline_pre_post_backward_bitwise=True,
        raw_train_cameras_float16_reproduce_cache_bitwise=True,
        language_logit_max_delta=language_delta,
        corrected_gate="Attention physical projection + independent output GT + actual full-head replay invariants; does not require learned decoder to be spatial identity",
        previous_15m_gate="superseded diagnostic, old artifact preserved; inactive Head transform was audited",
        native_feature_0_9m_difference="documented sampling convention; not shifted or fitted in supervision; decoder receives original inputs",
    )
    write(out / "supervision_space_audit.json", report)
    write(
        out / "spatial_gate.json",
        dict(
            status="PASS",
            pilot_allowed=True,
            evidence_sha256=sha(out / "supervision_space_audit.json"),
            reason="运行时Head/监督空间独立验证通过；原始双Head及预测逐位不变；无cropper替换",
        ),
    )
    print("SUPERVISION_SPACE_AUDIT_PASS no_optimizer_updates", flush=True)


def pilot(out: Path) -> None:
    """只运行一次16步终点，无结构/阈值/lr搜索，原生双Head从不替换。"""
    protocol = verify_protocol(out)
    assert read(out / "spatial_gate.json")["pilot_allowed"]
    folder = out / "pilot"
    folder.mkdir(exist_ok=False)
    model, _, encoder, dataset = load_native()
    assert "forward" not in model.pts_bbox_head.feat_cropper.__dict__
    manifest = read(out / "manifest.json")
    rows = [r for r in manifest["annotations"] if r["split"] == "train"]
    lookup = {r["instruction_id"]: r for r in rows}
    schedule = [lookup[key] for key in read(out / "schedule.json")]
    assert len(schedule) == protocol["budget"]["pilot_updates"] == 16
    text = encode(encoder, rows)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    adapter = LanguageGroundedBEVAdapter(**protocol["model_config"]).cuda().train()
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=protocol["budget"]["lr"],
        weight_decay=protocol["budget"]["weight_decay"],
    )
    frozen_before = dict(author=module_sha(model), text=module_sha(encoder))
    first = schedule[0]
    original_bev, _ = frame_input(first)
    with torch.no_grad():
        original_prediction = {
            k: v.detach().clone()
            for k, v in dual_head(model, original_bev).items()
            if k in ("all_cls_scores", "all_bbox_preds", "seg_preds")
        }
    logs, audits = [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (folder / "training.jsonl").open("x") as stream:
        for step, row in enumerate(schedule, 1):
            assert row["split"] == "train"
            bev, position = frame_input(row)
            with torch.no_grad():
                teacher = dual_head(model, bev)["seg_preds"].softmax(1).detach()
            optimizer.zero_grad(set_to_none=True)
            detail = adapter(bev, text[row["text"]], position)
            prediction = dual_head(model, detail["enhanced_bev"])
            total, parts, stats = objective(
                model,
                prediction,
                detail,
                row,
                out,
                dataset.get_ann_info(row["index"]),
                teacher,
                protocol,
            )
            grads = {
                key: gradient_audit(parts[key], adapter)
                for key in ("detection", "segmentation", "mask")
            }
            assert all(v["all_finite"] and v["total_l2"] > 0 for v in grads.values())
            total.backward()
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(),
                    protocol["budget"]["clip"],
                    error_if_nonfinite=True,
                )
            )
            before = [p.detach().clone() for p in adapter.parameters()]
            optimizer.step()
            delta = float(
                sum(
                    (p.detach() - b).square().sum()
                    for p, b in zip(adapter.parameters(), before)
                ).sqrt()
            )
            assert norm > 0 and delta > 0
            record = dict(
                step=step,
                instruction_id=row["instruction_id"],
                task=row["task"],
                sample_token=row["sample_token"],
                target_present=row["target_present"],
                losses={k: float(v.detach()) for k, v in parts.items()},
                total_loss=float(total.detach()),
                gradient_l2=norm,
                parameter_update_l2=delta,
                task_gradient_l2={k: v["total_l2"] for k, v in grads.items()},
                relative_residual_l2=float(
                    (detail["residual"].norm() / bev.norm()).detach()
                ),
                native_gt_filter=stats,
                cached_real_bev=True,
                native_heads_unchanged=True,
                optimizer_state_steps=sorted(
                    {int(v["step"]) for v in optimizer.state.values()}
                ),
            )
            stream.write(__import__("json").dumps(record) + "\n")
            stream.flush()
            logs.append(record)
            if step in (1, 16):
                audits.append(dict(step=step, tasks=grads))
            print("SUPERVISION_PILOT", step, record["total_loss"], delta, flush=True)
            del total, detail, prediction, parts
    with torch.no_grad():
        restored = dual_head(model, original_bev)
    assert all(torch.equal(restored[k], v) for k, v in original_prediction.items())
    frozen_after = dict(author=module_sha(model), text=module_sha(encoder))
    assert frozen_after == frozen_before
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    assert "forward" not in model.pts_bbox_head.feat_cropper.__dict__
    path = out / "adapter_step016.pt"
    with path.open("xb") as stream:
        torch.save(
            dict(
                state_dict=adapter.state_dict(),
                optimizer=optimizer.state_dict(),
                optimizer_updates=16,
                protocol_sha256=sha(out / "protocol.json"),
            ),
            stream,
        )
    write(
        out / "pilot_training.json",
        dict(
            status="PASS",
            optimizer_updates=16,
            checkpoint=dict(path=str(path), sha256=sha(path)),
            steps=logs,
            frozen_before=frozen_before,
            frozen_after=frozen_after,
            native_original_outputs_unchanged_bitwise=True,
            seconds=time.perf_counter() - started,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,
        ),
    )
    write(out / "grad_audit.json", dict(per_step=audits, status="PASS"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SPACE_OUT)
    parser.add_argument("--phase", choices=("prepare", "audit", "pilot"), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "audit": audit, "pilot": pilot}[args.phase](args.out.resolve())

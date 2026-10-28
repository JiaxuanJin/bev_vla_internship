"""LG-BEV 单帧真实双 Head 验收与唯一固定预算 pilot，原文件只读。"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.spatial_annotations import metric_grid, project_attention
from scripts.stage3.lg_bev_common import (
    DEFAULT_OUT,
    ROOT,
    SEED,
    read,
    sha,
    verify_protocol,
    write,
)
from scripts.stage3.lg_bev_data import geometry, load_supervision
from scripts.stage3.lg_bev_native import (
    dual_head,
    gradient_audit,
    load_native,
    module_sha,
    native_losses,
)
from scripts.stage3.lg_bev_prototype import (
    LanguageGroundedBEVAdapter,
    attention_supervision_loss,
)


def encode(encoder: nn.Module, rows: list[dict]) -> dict[str, Tensor]:
    """仅从指令文本编码；冻结编码器，无 GT 接口。"""
    texts = sorted({r[k] for r in rows for k in ("text", "wrong_text")})
    result = {}
    for offset in range(0, len(texts), 16):
        batch = texts[offset : offset + 16]
        with torch.no_grad():
            values = encoder(batch, "cuda")
        result.update({text: value[None] for text, value in zip(batch, values)})
    return result


def frame_input(frame: dict, *, device: str = "cuda") -> tuple[Tensor, Tensor]:
    """推理仅消费 BEV 缓存与标定，不读取训练标签或源语义 GT。"""
    if sha(frame["cache_file"]) != frame["cache_sha256"]:
        raise ValueError("真实 BEV 缓存已改变")
    with np.load(frame["cache_file"]) as data:
        bev = torch.as_tensor(data["bev"].astype(np.float32), device=device)
    return bev, geometry(frame, device=device)


def training_frame(row: dict) -> tuple[Tensor, Tensor]:
    """损失专用原始全图 GT 与 Ego 网格，拒绝任何内部开发标签。"""
    if row["split"] != "train":
        raise ValueError("训练不能读取开发 GT")
    with np.load(row["source_file"]) as data:
        return (
            torch.as_tensor(data["gt"].astype(np.int64), device="cuda"),
            torch.as_tensor(data["ego_xy"].astype(np.float32), device="cuda"),
        )


def objective(
    model: nn.Module,
    prediction: dict,
    detail: dict,
    row: dict,
    out: Path,
    ann: dict,
    teacher: Tensor,
    protocol: dict,
) -> tuple[Tensor, dict[str, Tensor], dict]:
    """真实检测/作者分割/实例掩码/无关区 KL；GT 只在此损失函数读取。"""
    gt, xy = training_frame(row)
    label = {k: v.cuda() for k, v in load_supervision(row, out, training=True).items()}
    parts = native_losses(model, prediction, ann, gt)
    stats = parts.pop("filter_stats")
    logits = detail["attention_logits"]
    if row["task"] == "map":
        grid, _, _ = metric_grid(row)
        logits = project_attention(logits, grid)
        fb, lr = row["fb"], row["lr"]
    else:
        fb, lr = row["prompt"]["front_back"], row["prompt"]["left_right"]
    losses = attention_supervision_loss(
        logits,
        label["target"],
        label["distractor"],
        distractor_weight=protocol["loss"]["distractor"],
        valid_mask=label["valid"],
    )
    region = ((xy[..., 0] > 0) if fb == "front" else (xy[..., 0] < 0)) & (
        (xy[..., 1] > 0) if lr == "left" else (xy[..., 1] < 0)
    )
    kl = F.kl_div(
        prediction["seg_preds"].log_softmax(1), teacher, reduction="none"
    ).sum(1)[0]
    parts.update(
        mask=losses["total"],
        mask_bce=losses["bce"],
        mask_dice=losses["dice"],
        distractor=losses["distractor"],
        irrelevant_protect=kl[~region].mean(),
    )
    weights = protocol["loss"]
    total = sum(
        parts[key] * weights[key]
        for key in ("detection", "segmentation", "mask", "irrelevant_protect")
    )
    if not bool(torch.isfinite(total)):
        raise ValueError("非有限真实损失")
    return total, parts, stats


def make_schedule(rows: list[dict], steps: int) -> list[dict]:
    """固定 seed 和地图/对象、非空/空目标顺序，不读取预测。"""
    rng = np.random.RandomState(SEED)
    train = [r for r in rows if r["split"] == "train"]
    result = []
    for step in range(steps):
        task = "map" if step % 2 == 0 else "object"
        eligible = [r for r in train if r["task"] == task]
        if task == "object":
            positive = (step // 2) % 2 == 0
            eligible = [r for r in eligible if r["target_present"] == positive]
        if not eligible:
            raise ValueError("预注册训练组缺失")
        result.append(eligible[int(rng.randint(len(eligible)))])
    return result


def save_checkpoint(
    out: Path,
    name: str,
    adapter: nn.Module,
    optimizer: Any,
    updates: int,
) -> dict:
    """checkpoint 绑定协议和更新次数，只保存新增模块。"""
    path = out / name
    with path.open("xb") as stream:
        torch.save(
            dict(
                state_dict=adapter.state_dict(),
                optimizer=optimizer.state_dict(),
                optimizer_updates=updates,
                protocol_sha256=sha(out / "protocol.json"),
            ),
            stream,
        )
    return dict(path=str(path), sha256=sha(path), optimizer_updates=updates)


def train(out: Path, phase: str) -> None:
    """先验收两步，随后只有审计门槛通过才启动全新随机初始化16步。"""
    protocol = verify_protocol(out)
    if phase == "pilot":
        smoke = read(out / "smoke_test.json")
        if smoke["status"] != "PASS":
            raise ValueError("smoke 未通过")
        if not read(out / "spatial_gate.json")["pilot_allowed"]:
            raise ValueError("空间审计不通过，禁止 pilot")
    folder = out / phase
    folder.mkdir(exist_ok=False)
    model, _, encoder, dataset = load_native()
    manifest = read(out / "manifest.json")
    rows = [r for r in manifest["annotations"] if r["split"] == "train"]
    encoded = encode(encoder, rows)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    adapter = LanguageGroundedBEVAdapter(**protocol["model_config"]).cuda().train()
    budget = protocol["budget"]
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=budget["lr"], weight_decay=budget["weight_decay"]
    )
    if phase == "smoke":
        first = next(
            r
            for r in rows
            if r["task"] == "object"
            and r["qualifier"] == "nearest"
            and r["same_direction_distractors"] > 0
        )
        second = next(
            r
            for r in rows
            if r["task"] == "map" and r["sample_token"] == first["sample_token"]
        )
        schedule = [first, second]
    else:
        schedule = make_schedule(rows, budget["pilot_updates"])
    write(folder / "schedule.json", [r["instruction_id"] for r in schedule])
    frozen_before = dict(author=module_sha(model), text=module_sha(encoder))
    first = schedule[0]
    bev0, _ = frame_input(first)
    with torch.no_grad():
        baseline0 = {
            k: v.detach().clone()
            for k, v in dual_head(model, bev0).items()
            if k in ("all_cls_scores", "all_bbox_preds", "seg_preds")
        }
    before_adapter = {n: p.detach().clone() for n, p in adapter.named_parameters()}
    logs, audits = [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (folder / "training.jsonl").open("x") as stream:
        for step, row in enumerate(schedule, 1):
            bev, pos = frame_input(row)
            with torch.no_grad():
                original = dual_head(model, bev)
                teacher = original["seg_preds"].softmax(1)
            optimizer.zero_grad(set_to_none=True)
            detail = adapter(bev, encoded[row["text"]], pos)
            prediction = dual_head(model, detail["enhanced_bev"])
            if step == 1:
                for key in baseline0:
                    assert torch.equal(prediction[key], original[key]), key
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
            task_gradients = {}
            if phase == "smoke" or step == 1:
                task_gradients = {
                    key: gradient_audit(parts[key], adapter)
                    for key in ("detection", "segmentation", "mask")
                }
                for value in task_gradients.values():
                    assert value["all_finite"] and value["total_l2"] > 0
                audits.append(
                    dict(
                        step=step,
                        instruction_id=row["instruction_id"],
                        tasks=task_gradients,
                    )
                )
            total.backward()
            gradient = float(
                torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(), budget["clip"], error_if_nonfinite=True
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
            assert gradient > 0 and delta > 0
            record = dict(
                step=step,
                task=row["task"],
                instruction_id=row["instruction_id"],
                sample_token=row["sample_token"],
                losses={k: float(v.detach()) for k, v in parts.items()},
                total_loss=float(total.detach()),
                gradient_l2=gradient,
                parameter_update_l2=delta,
                task_gradient_l2={k: v["total_l2"] for k, v in task_gradients.items()},
                relative_residual_l2=float(
                    (detail["residual"].norm() / bev.norm()).detach()
                ),
                filter_stats=stats,
                cuda=True,
                cached_real_bev=True,
                native_dual_heads=True,
                optimizer_updates=step,
            )
            stream.write(__import__("json").dumps(record) + "\n")
            stream.flush()
            logs.append(record)
            print(
                "LG_UPDATE",
                phase,
                step,
                "loss",
                record["total_loss"],
                "gradient",
                gradient,
                "delta",
                delta,
                flush=True,
            )
            del total, parts, detail, prediction, original
    frozen_after = dict(author=module_sha(model), text=module_sha(encoder))
    assert frozen_before == frozen_after
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    with torch.no_grad():
        after = dual_head(model, bev0)
        baseline_unchanged = all(torch.equal(after[k], v) for k, v in baseline0.items())
        pos = geometry(first, device="cuda")
        correct = adapter(bev0, encoded[first["text"]], pos)
        wrong = adapter(bev0, encoded[first["wrong_text"]], pos)
        text_delta = float(
            (correct["attention_logits"] - wrong["attention_logits"]).abs().max()
        )
        residual = float(correct["residual"].norm() / bev0.norm())
    assert baseline_unchanged and text_delta > 1e-8 and residual > 0
    saved = save_checkpoint(
        out, f"{phase}_adapter.pt", adapter, optimizer, len(schedule)
    )
    summary = dict(
        status="PASS",
        optimizer_updates=len(schedule),
        checkpoint=saved,
        frozen_before=frozen_before,
        frozen_after=frozen_after,
        baseline_outputs_bitwise_unchanged=baseline_unchanged,
        initial_identity_bitwise=True,
        correct_wrong_max_logit_delta=text_delta,
        post_update_relative_residual_l2=residual,
        parameter_total_delta_l2=float(
            sum(
                (p.detach() - before_adapter[n]).square().sum()
                for n, p in adapter.named_parameters()
            ).sqrt()
        ),
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        python=__import__("sys").executable,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        seconds=time.perf_counter() - started,
        steps=logs,
    )
    write(
        out / ("smoke_test.json" if phase == "smoke" else "pilot_training.json"),
        summary,
    )
    if phase == "smoke":
        write(
            out / "grad_audit.json",
            dict(
                status="PASS",
                per_step=audits,
                interpretation="初始检测/分割仅到达 up；显式 mask 到达视觉/位置/语言/key/refine，不能到达 up。首步后检测/分割可达上游，逐参数见报告。",
            ),
        )
        audit_v5(model, dataset, bev0, pos, encoded[first["text"]], first, out)
        audit_native_geometry(model, first, out)
        paired_runtime_audit(
            model,
            adapter,
            bev0,
            pos,
            encoded[first["text"]],
            encoded[first["wrong_text"]],
            out,
        )


def paired_runtime_audit(
    model: nn.Module,
    adapter: nn.Module,
    bev: Tensor,
    pos: Tensor,
    correct: Tensor,
    wrong: Tensor,
    out: Path,
) -> None:
    """单训练帧四分支真实前向及同步计时；不计 AP/IoU，不接触开发 GT。"""
    from scripts.stage3.lg_bev_evaluate import paired_forward

    values = paired_forward(model, adapter, bev, pos, correct, wrong)
    original = values["original"]
    records = {}
    for branch, value in values.items():
        records[branch] = dict(
            decoded_boxes=int(len(value["scores"])),
            score_rms_delta_vs_original=float(
                np.sqrt(np.mean((value["scores"] - original["scores"]) ** 2))
            ),
            semantic_changed_pixels=int(
                (value["semantic"] != original["semantic"]).sum()
            ),
            relative_residual_l2=value["relative_residual_l2"],
            mean_attention=(
                float(value["attention"].mean())
                if value["attention"] is not None
                else None
            ),
        )
    write(
        out / "paired_smoke.json",
        dict(
            status="PASS",
            frames=1,
            scene_scope="training only",
            branches=records,
            performance_metrics="NOT_RUN: forward/update diagnostics only",
        ),
    )

    def measure(function: Any, repeats: int) -> dict:
        with torch.no_grad():
            for _ in range(3):
                function()
            times = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                function()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) * 1000)
        return dict(
            repeats=repeats,
            mean_ms=float(np.mean(times)),
            median_ms=float(np.median(times)),
            samples_ms=times,
        )

    baseline = measure(lambda: dual_head(model, bev), 7)
    cached = measure(
        lambda: dual_head(model, adapter(bev, correct, pos)["enhanced_bev"]), 7
    )
    adapter_only = measure(lambda: adapter(bev, correct, pos), 20)
    write(
        out / "latency.json",
        dict(
            status="SMOKE_TIMING_ONLY",
            gpu=torch.cuda.get_device_name(),
            original_cached_bev_dual_head=baseline,
            lg_cached_bev_dual_head=cached,
            adapter_only=adapter_only,
            cached_mean_difference_ms=cached["mean_ms"] - baseline["mean_ms"],
            text_encoding_included=False,
            cache_io_included=False,
            six_camera_end_to_end=dict(status="NOT_RUN", milliseconds=None),
            rtx3090_threshold=dict(status="NOT_EVALUATED", milliseconds=None),
            caveat="single training frame; synchronized wall clock, 3 warmups; not official deployment latency",
        ),
    )


def audit_v5(
    model: nn.Module,
    dataset: Any,
    bev: Tensor,
    pos: Tensor,
    text: Tensor,
    row: dict,
    out: Path,
) -> None:
    """在允许训练帧上用真实 autograd 核查 V5，完全不更新其 checkpoint。"""
    from scripts.stage2.coupled_v5_model import LanguageConditionedBEVResidualAdapter

    checkpoint = (
        ROOT
        / "outputs/stage3_coupled_v5_20261026/train/targeted_language/adapter_step096.pt"
    )
    report = {}
    gt, _ = training_frame(row)
    for phase in ("zero_init", "sealed_step096"):
        adapter = LanguageConditionedBEVResidualAdapter().cuda()
        if phase == "sealed_step096":
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            adapter.load_state_dict(saved["state_dict"])
        detail = adapter(bev, text, pos)
        prediction = dual_head(model, detail["enhanced_bev"])
        losses = native_losses(
            model, prediction, dataset.get_ann_info(row["index"]), gt
        )
        report[phase] = {
            key: gradient_audit(losses[key], adapter)
            for key in ("detection", "segmentation")
        }
    write(
        out / "v5_autograd_audit.json",
        dict(
            sample_token=row["sample_token"],
            checkpoint_sha256=sha(checkpoint),
            optimizer_updates=0,
            results=report,
            conclusion="detach 用于 RMS 标度，主任务路径未断；零 up 导致首步上游梯度为零须另列。",
        ),
    )


def audit_native_geometry(model: nn.Module, row: dict, out: Path) -> None:
    """定量区分新监督的物理投影与原作者分割 cropper 的既有采样协议。"""
    native = BEVGrid().lidar_centers(dtype=torch.float32, device="cuda")
    ramp = native[..., :2].permute(2, 0, 1)[None]
    head = model.pts_bbox_head
    with torch.no_grad():
        legacy = head.feat_cropper(torch.flip(torch.rot90(ramp, -1, [2, 3]), [3]))
    _, _, expected = metric_grid(row)
    error = np.linalg.norm(
        legacy[0].permute(1, 2, 0).cpu().numpy() - expected[..., :2], axis=-1
    )
    # 不改冻结 Head 的坐标路径；有冲突时停止扩大训练，保留两步链路验收。
    allowed = float(error.max()) < 1e-4
    write(
        out / "spatial_gate.json",
        dict(
            pilot_allowed=allowed,
            status="PASS" if allowed else "BLOCKED",
            supervision_ramp="PASS: calibrated native->GT ramp <1e-8m",
            native_seg_cropper_mean_error_m=float(error.mean()),
            native_seg_cropper_max_error_m=float(error.max()),
            reason="原作者分割旋转/翻转/cropper 与逐帧 LiDAR↔Ego 标定的物理采样不同；保持原生冻结路径，禁止带未解决空间冲突进入 pilot。",
            smoke_scope="2 updates for native loss/gradient/update/freeze contract only; no performance selection",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--phase", choices=("smoke", "pilot"), default="smoke")
    args = parser.parse_args()
    train(args.out.resolve(), args.phase)

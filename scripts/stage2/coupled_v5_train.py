"""V5 有限预算真实 CUDA 双 Head 训练与逐步耦合梯度记录。"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.stage2.coupled_v5_data import BRANCHES, OUT, SEED, V4, sha, write
from scripts.stage2.coupled_v5_model import LanguageConditionedBEVResidualAdapter
from scripts.stage2.conditional_mask_v4_experiment import (
    frame_inputs,
    load_label,
    load_runtime,
)
from scripts.stage2.conditional_mask_v4_model import LanguageConditionedTargetMaskHead
from scripts.stage2.micro_training import seed_all
from scripts.stage2.spatial_v3_evaluate import segmentation
from scripts.stage2.spatial_v3_experiment import actual_losses, head_forward
from scripts.stage2.spatial_v3_objects import object_rows, relevant_query_weights
from scripts.stage2.spatial_v3_range_correction import filter_ann
from scripts.stage2.stage3_metric_review import neighborhood


def runtime() -> tuple:
    """只读复用 V4 的真实48帧缓存及独立实例标签，不载入其最终集。"""
    model, wrapped, pipeline, dataset, manifest, features, text = load_runtime()
    saved = json.loads((V4 / "training_summary.json").read_text())["branches"][
        "full_instruction"
    ]
    assert sha(saved["checkpoint"]) == saved["sha256"]
    mask_head = LanguageConditionedTargetMaskHead().cuda().eval()
    mask_head.load_state_dict(
        torch.load(saved["checkpoint"], map_location="cpu", weights_only=False)[
            "state_dict"
        ]
    )
    mask_head.requires_grad_(False)
    return model, wrapped, pipeline, dataset, manifest, features, text, mask_head


def position(pipeline: Any, row: dict) -> torch.Tensor:
    """标定生成原生米制几何，训练与推理完全相同。"""
    pipeline.fusion.set_calibration(row)
    return pipeline.fusion.position_features.detach().clone()


def load_training_inputs(
    model: Any, pipeline: Any, dataset: Any, manifest: dict, features: dict
) -> tuple:
    """真实对象先核验ID；全图GT保留所有正确类别，不按指令改写。"""
    objects = object_rows(manifest["frames"], dataset)
    object_protocol = dict(
        annotations=objects,
        n_train=sum(r["split"] == "train" for r in objects),
        n_dev=sum(r["split"] == "dev" for r in objects),
        policy="real raw3Dannotation/instance IDs, center matching<1mm; plural car/pedestrian in quadrant; no mapMask-to-box mapping",
    )
    object_path = OUT / "objects_protocol.json"
    if object_path.exists():
        assert json.loads(object_path.read_text()) == object_protocol
    else:
        write(object_path, object_protocol)
    frames = {}
    for row in manifest["frames"]:
        with np.load(row["source_file"]) as f:
            gt = torch.tensor(f["gt"], device="cuda", dtype=torch.long)
            xy = torch.tensor(f["ego_xy"], device="cuda", dtype=torch.float32)
        with torch.no_grad():
            logits = segmentation(model.pts_bbox_head, features[row["index"]])
        filtered, stats = filter_ann(
            dataset.get_ann_info(row["index"]), model.pts_bbox_head.pc_range
        )
        frames[row["index"]] = dict(
            gt=gt,
            xy=xy,
            teacher=logits.softmax(1).detach(),
            position=position(pipeline, row),
            ann=filtered,
            filter_stats=stats,
        )
    maps = {}
    for row in manifest["annotations"]:
        label = load_label(row)
        roi = (
            torch.tensor(neighborhood(label["target"].cpu().numpy()), device="cuda")
            & label["region"]
        )
        maps[row["instruction_id"]] = dict(**label, roi=roi)
    return objects, frames, maps


def make_schedule(
    rows: list[dict], objects: list[dict], total: int = 256
) -> list[dict]:
    """固定目标mini-batch，平衡对象类别及存在/空目标；不读取预测。"""
    rng = np.random.RandomState(SEED)
    maps = [r for r in rows if r["split"] == "train"]
    objects = [r for r in objects if r["split"] == "train"]
    schedule = []
    for i in range(total):
        if i % 2 == 0:
            row = maps[int(rng.randint(len(maps)))]
            schedule.append(dict(task="map", instruction_id=row["instruction_id"]))
        else:
            j = i // 2
            cls = 0 if j % 2 == 0 else 8
            nonempty = (j // 2) % 2 == 0
            eligible = [
                r
                for r in objects
                if r["prompt"]["class_id"] == cls
                and bool(r["relevant_gt_indices"]) == nonempty
            ]
            if not eligible:
                raise RuntimeError("missing preregistered positive/empty object group")
            row = eligible[int(rng.randint(len(eligible)))]
            schedule.append(dict(task="object", instruction_id=row["instruction_id"]))
    return schedule


def objective(
    model: Any, pred: dict, row: dict, frame: dict, label: dict | None, targeted: bool
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict]]:
    """真实检测损失及相关/无关语义保护，原GT和Hungarian分配不变。

    Args:
        model: 权重冻结的原作者模型。
        pred: 增强BEV经真实原生双Head的输出。
        row: 真实地图或对象指令。
        frame: 全图语义、原模型teacher和过滤后的3DGT。
        label: 地图实例监督；对象任务为None。
        targeted: 两个目标加权分支使用相同规则。

    Returns:
        总损失、各真实损失分量和匹配正例权重追踪。
    """
    traces = []
    weighted = (
        relevant_query_weights(model.pts_bbox_head, row, traces)
        if targeted and label is None
        else nullcontext()
    )
    with weighted:
        detection = actual_losses(model, pred, frame["ann"])
    pixels = F.cross_entropy(pred["seg_preds"], frame["gt"][None], reduction="none")[0]
    global_ce = pixels.mean()
    if label is not None:
        region = label["region"]
        local_ce = pixels[label["roi"]].mean()
    else:
        p = row["prompt"]
        xy = frame["xy"]
        region = (
            (xy[..., 0] > 0) if p["front_back"] == "front" else (xy[..., 0] < 0)
        ) & ((xy[..., 1] > 0) if p["left_right"] == "left" else (xy[..., 1] < 0))
        local_ce = global_ce * 0
    irrelevant_ce = pixels[~region].mean()
    kl_pixels = F.kl_div(
        pred["seg_preds"].log_softmax(1), frame["teacher"], reduction="none"
    ).sum(1)[0]
    irrelevant_kl = kl_pixels[~region].mean()
    semantic = global_ce
    if targeted:
        semantic = semantic + 0.1 * irrelevant_kl
        if label is not None:
            semantic = semantic + local_ce + 0.5 * irrelevant_ce
    parts = dict(
        detection=detection,
        segmentation=semantic,
        global_ce=global_ce,
        target_neighborhood_ce=local_ce,
        irrelevant_ce=irrelevant_ce,
        irrelevant_kl=irrelevant_kl,
    )
    return detection + semantic, parts, traces


def flat_gradient(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
    """完整梯度允许初始上投影之外的参数为零；不能将整个任务断路。"""
    values = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat(
        [
            (torch.zeros_like(p) if g is None else g.detach()).reshape(-1)
            for p, g in zip(params, values)
        ]
    )


def checkpoint(
    adapter: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    folder: Path,
    step: int,
    branch: str,
) -> dict:
    """封存真实权重、优化器、更新次数与协议SHA，拒绝覆盖。"""
    path = folder / f"adapter_step{step:03d}.pt"
    with path.open("xb") as f:
        torch.save(
            dict(
                state_dict=adapter.state_dict(),
                optimizer=optimizer.state_dict(),
                optimizer_updates=step,
                branch=branch,
                protocol_sha256=sha(OUT / "protocol.json"),
            ),
            f,
        )
    return dict(checkpoint=str(path), sha256=sha(path), optimizer_updates=step)


def train(phase: str) -> None:
    """小试96步；只有合法门槛通过才允许继续至256，不能无限增加。"""
    protocol = json.loads((OUT / "protocol.json").read_text())
    for p, digest in protocol["source_sha256"].items():
        assert sha(ROOT / p) == digest, p
    if phase == "expand":
        decision = json.loads((OUT / "pilot_decision.json").read_text())
        assert decision["expand"]
    model, wrapped, pipeline, dataset, manifest, features, text, mask_head = runtime()
    if phase == "pilot":
        write(
            OUT / "hardware.json",
            dict(
                torch=torch.__version__,
                cuda_available=torch.cuda.is_available(),
                gpu=torch.cuda.get_device_name(),
                capability=torch.cuda.get_device_capability(),
                python=sys.executable,
                nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True),
                host_memory=subprocess.check_output(["free", "-h"], text=True),
            ),
        )
    objects, frames, maps = load_training_inputs(
        model, pipeline, dataset, manifest, features
    )
    rows = {r["instruction_id"]: r for r in [*manifest["annotations"], *objects]}
    schedule = make_schedule(manifest["annotations"], objects)
    if phase == "pilot":
        write(OUT / "schedule.json", dict(steps=schedule, maximum=256, pilot=96))
    start_step, end_step = (0, 96) if phase == "pilot" else (96, 256)
    summaries = {}
    for branch in BRANCHES:
        seed_all(SEED)
        adapter = LanguageConditionedBEVResidualAdapter().cuda().train()
        optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=0.0003, weight_decay=0.01
        )
        if phase == "pilot":
            folder = OUT / "train" / branch
            folder.mkdir(parents=True)
        else:
            folder = OUT / "train" / branch
            old = torch.load(
                folder / "adapter_step096.pt", map_location="cpu", weights_only=False
            )
            adapter.load_state_dict(old["state_dict"])
            optimizer.load_state_dict(old["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.cuda()
        params = list(adapter.parameters())
        initial = {n: p.detach().clone() for n, p in adapter.named_parameters()}
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        saved = []
        with (folder / f"{phase}_training.jsonl").open("x") as log:
            for k in range(start_step, end_step):
                item = schedule[k]
                row = rows[item["instruction_id"]]
                frame = frames[row["index"]]
                bev = features[row["index"]]
                embedding = (
                    text[row["text"]]
                    if branch == "targeted_language"
                    else torch.zeros_like(text[row["text"]])
                )
                optimizer.zero_grad(set_to_none=True)
                detail = adapter(bev, embedding, frame["position"])
                pred = head_forward(model, detail["enhanced_bev"])
                label = maps[row["instruction_id"]] if item["task"] == "map" else None
                loss, parts, trace = objective(
                    model, pred, row, frame, label, branch != "global_control"
                )
                grads = {
                    name: flat_gradient(parts[name], params)
                    for name in ("detection", "segmentation")
                }
                norms = {name: float(g.norm()) for name, g in grads.items()}
                assert all(np.isfinite(v) and v > 0 for v in norms.values())
                cosine = float(
                    F.cosine_similarity(
                        grads["detection"][None], grads["segmentation"][None]
                    )
                )
                loss.backward()
                gn = float(
                    torch.nn.utils.clip_grad_norm_(params, 5.0, error_if_nonfinite=True)
                )
                before = [p.detach().clone() for p in params]
                optimizer.step()
                delta = float(
                    sum(
                        (p.detach() - b).square().sum() for p, b in zip(params, before)
                    ).sqrt()
                )
                assert delta > 0
                record = dict(
                    step=k + 1,
                    branch=branch,
                    task=item["task"],
                    instruction_id=row["instruction_id"],
                    sample_token=row["sample_token"],
                    text=(
                        row["text"]
                        if branch == "targeted_language"
                        else "<NO_LANGUAGE>"
                    ),
                    losses={
                        name: float(value.detach()) for name, value in parts.items()
                    },
                    total_loss=float(loss.detach()),
                    task_parameter_gradient_l2=norms,
                    task_gradient_cosine=cosine,
                    gradient_l2_preclip=gn,
                    parameter_step_delta_l2=delta,
                    relative_bev_l2=float(
                        (detail["residual"].norm() / bev.norm()).detach()
                    ),
                    optimizer_state_steps=sorted(
                        {int(s["step"]) for s in optimizer.state.values()}
                    ),
                    native_gt_filter=frame["filter_stats"],
                    relevant_positive_assignments=trace,
                    cuda=True,
                    cached_real_bev=True,
                    native_heads_executed=True,
                )
                log.write(json.dumps(record) + "\n")
                log.flush()
                if k == 0 or (k + 1) % 16 == 0:
                    print(
                        "V5_CUDA_UPDATE",
                        branch,
                        k + 1,
                        item["task"],
                        norms,
                        "delta",
                        delta,
                        "feature",
                        record["relative_bev_l2"],
                        flush=True,
                    )
                if k + 1 in (1, 48, 96, 256):
                    saved.append(checkpoint(adapter, optimizer, folder, k + 1, branch))
        summaries[branch] = dict(
            selected_terminal=saved[-1],
            checkpoints=saved,
            optimizer_updates_this_phase=end_step - start_step,
            total_adapter_updates=end_step,
            map_updates_this_phase=(end_step - start_step) // 2,
            object_updates_this_phase=(end_step - start_step) // 2,
            seconds=time.perf_counter() - started,
            parameter_delta_l2_this_phase=float(
                sum(
                    (p.detach() - initial[n]).square().sum()
                    for n, p in adapter.named_parameters()
                ).sqrt()
            ),
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            parameter_count=sum(p.numel() for p in params),
        )
        del pred, detail, loss, parts, grads, adapter, optimizer
    assert not any(p.grad is not None for p in model.parameters())
    assert not any(p.grad is not None for p in pipeline.parameters())
    assert not any(p.grad is not None for p in mask_head.parameters())
    write(
        OUT / f"{phase}_training_summary.json",
        dict(
            branches=summaries,
            total_updates_this_phase=3 * (end_step - start_step),
            frozen_native_head_updates=0,
            frozen_backbone_updates=0,
            frozen_bert_updates=0,
            frozen_v4mask_updates=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "expand"), default="pilot")
    train(parser.parse_args().phase)

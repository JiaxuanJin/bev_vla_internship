"""V4 真 CUDA 单对诊断、等预算训练；缓存边界是原作者真实 BEV。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.stage2.conditional_mask_v4_data import OUT, SEED, V3, sha, write
from scripts.stage2.conditional_mask_v4_model import (
    LanguageConditionedTargetMaskHead,
    mask_loss,
)
from scripts.stage2.micro_training import seed_all
from scripts.stage2.spatial_experiment import INITIAL, initialize
from scripts.stage2.spatial_v3_model import InstanceFusion, InstanceReadout, VARIANTS
from scripts.stage2.spatial_v3_experiment import SELECTED


def load_runtime() -> (
    tuple[Any, Any, Any, Any, dict, dict[int, Tensor], dict[str, Tensor]]
):
    """加载冻结原作者模型；仅载入新划分的 48 帧，不执行旧开发/留出前向。"""
    model, wrapped, pipeline, dataset = initialize("train")
    pipeline.encoder = InstanceReadout(pipeline.encoder).cuda().eval()
    initial = torch.load(INITIAL, map_location="cpu", weights_only=False)["state_dict"]
    v2 = torch.load(SELECTED, map_location="cpu", weights_only=False)["state_dict"]
    pipeline.fusion = InstanceFusion(initial, v2, VARIANTS["combined"]).cuda().eval()
    checkpoint = V3 / "train/combined/fusion_step176.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    pipeline.fusion.load_state_dict(saved["state_dict"])
    pipeline.requires_grad_(False).eval()
    manifest = json.loads((OUT / "labels/manifest.json").read_text())
    features = {}
    for frame in manifest["frames"]:
        assert frame["scene_name"] not in ("scene-0061", "scene-0553", "scene-0001")
        assert sha(frame["cache_file"]) == frame["cache_sha256"]
        with np.load(frame["cache_file"]) as f:
            features[frame["index"]] = torch.tensor(
                f["bev"], device="cuda", dtype=torch.float32
            )
    texts = set()
    for row in manifest["annotations"]:
        texts.update((row["text"], class_text(row)))
        if row["counterfactual_text"]:
            texts.add(row["counterfactual_text"])
        texts.add(shuffle_words(row["text"]))
    texts.update(
        f"Find {name} on the {fb} {lr}"
        for name in ("cars", "pedestrians")
        for fb in ("front", "rear")
        for lr in ("left", "right")
    )
    encoded = {}
    ordered = sorted(texts)
    for i in range(0, len(ordered), 16):
        with torch.no_grad():
            embedding = pipeline.encoder(ordered[i : i + 16], "cuda")
        encoded.update({t: v[None] for t, v in zip(ordered[i : i + 16], embedding)})
    return model, wrapped, pipeline, dataset, manifest, features, encoded


def class_text(row: dict) -> str:
    """类别文本不含方位或实例限定词。"""
    return "Find lane markings" if row["target_class"] == 1 else "Find crosswalk"


def shuffle_words(text: str) -> str:
    """固定逐词置乱，属于语序破坏测试，不等价于随机标签训练。"""
    words = text.split()
    order = np.random.RandomState(SEED).permutation(len(words))
    return " ".join(words[i] for i in order)


def frame_inputs(row: dict) -> tuple[Tensor, Tensor]:
    """独立从标定生成网格并核对历史物理网格；不使用实例位置。"""
    from scripts.stage2.spatial_annotations import metric_grid

    grid, ego, _ = metric_grid(row)
    with np.load(row["source_file"]) as f:
        np.testing.assert_allclose(grid.numpy(), f["sampling_grid"], atol=1e-10)
    x, y = ego[..., 0] / 30.0, ego[..., 1] / 15.0
    geometry = np.stack((x, y, np.sqrt(x * x + y * y), x * y, np.ones_like(x)))
    return grid.cuda().float(), torch.tensor(
        geometry[None], device="cuda", dtype=torch.float32
    )


def load_label(row: dict) -> dict[str, Tensor]:
    """GT 仅传至损失与评估，推理 forward 没有标签接口。"""
    assert sha(OUT / "labels" / row["label_file"]) == row["label_sha256"]
    with np.load(OUT / "labels" / row["label_file"]) as f:
        return {k: torch.tensor(f[k], device="cuda", dtype=torch.bool) for k in f.files}


def counts(target: np.ndarray, pred: np.ndarray) -> dict[str, int | float]:
    """保存原始像素计数；空预测 precision 定义为零。"""
    tp = int((target & pred).sum())
    gt, active = int(target.sum()), int(pred.sum())
    union = gt + active - tp
    return dict(
        tp=tp,
        gt=gt,
        pred=active,
        union=union,
        iou=tp / union if union else 0.0,
        precision=tp / active if active else 0.0,
        recall=tp / gt if gt else 0.0,
        f1=2 * tp / (gt + active) if gt + active else 0.0,
    )


def save_checkpoint(
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    directory: Path,
    step: int,
    branch: str,
) -> dict:
    """保存真实参数、AdamW 状态及更新次数。"""
    path = directory / f"target_mask_step{step:04d}.pt"
    with path.open("xb") as f:
        torch.save(
            dict(
                state_dict=head.state_dict(),
                optimizer=optimizer.state_dict(),
                optimizer_updates=step,
                branch=branch,
                protocol_sha256=sha(OUT / "protocol.json"),
            ),
            f,
        )
    return dict(checkpoint=str(path), sha256=sha(path), optimizer_updates=step)


def train_steps(
    head: torch.nn.Module,
    rows: list[dict],
    features: dict[int, Tensor],
    text: dict[str, Tensor],
    steps: int,
    branch: str,
    directory: Path,
    paired_diagnostic: bool = False,
) -> dict:
    """只更新新增 Head；每步记录非零梯度及实际参数位移。"""
    directory.mkdir()
    seed_all(SEED)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
    rng = np.random.RandomState(SEED)
    schedule = (
        [[0, 1]] * steps
        if paired_diagnostic
        else [rng.randint(0, len(rows), size=2).tolist() for _ in range(steps)]
    )
    initial = {n: p.detach().clone() for n, p in head.named_parameters()}
    frames = {r["index"]: frame_inputs(r) for r in rows}
    labels = {r["instruction_id"]: load_label(r) for r in rows}
    parameters = [p for p in head.parameters()]
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (directory / "training.jsonl").open("x") as log:
        for step, indices in enumerate(schedule, 1):
            optimizer.zero_grad(set_to_none=True)
            chosen = [rows[n] for n in indices]
            if branch == "no_language":
                embeddings = torch.cat(
                    [torch.zeros_like(text[r["text"]]) for r in chosen]
                )
                used_text = ["<NO_LANGUAGE>"] * len(chosen)
            else:
                if branch == "class_only":
                    used_text = [class_text(r) for r in chosen]
                elif branch == "shuffled_text":
                    used_text = [
                        rows[int(rng.randint(len(rows)))]["text"] for _ in chosen
                    ]
                else:
                    used_text = [r["text"] for r in chosen]
                embeddings = torch.cat([text[t] for t in used_text])
            bev = torch.cat([features[r["index"]] for r in chosen])
            grid = torch.cat([frames[r["index"]][0] for r in chosen])
            geometry = torch.cat([frames[r["index"]][1] for r in chosen])
            pred = head(bev, embeddings, grid, geometry)
            targets = [labels[r["instruction_id"]] for r in chosen]
            losses = [
                mask_loss(pred[i, 0], t["target"], t["same_class_other"])
                for i, t in enumerate(targets)
            ]
            loss = torch.stack([v[0] for v in losses]).mean()
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(parameters, 5.0))
            if not np.isfinite(grad) or grad == 0:
                raise RuntimeError("invalid CUDA gradient")
            before = [p.detach().clone() for p in parameters]
            optimizer.step()
            delta = float(
                sum(
                    (p.detach() - b).square().sum() for p, b in zip(parameters, before)
                ).sqrt()
            )
            assert delta > 0
            record = dict(
                step=step,
                branch=branch,
                instruction_ids=[r["instruction_id"] for r in chosen],
                target_sample_tokens=[r["sample_token"] for r in chosen],
                used_text=used_text,
                loss=float(loss.detach()),
                loss_parts={
                    k: float(torch.stack([v[1][k] for v in losses]).mean().detach())
                    for k in losses[0][1]
                },
                gradient_l2_preclip=grad,
                parameter_step_delta_l2=delta,
                optimizer_state_steps=sorted(
                    {int(s["step"]) for s in optimizer.state.values()}
                ),
                cuda=True,
                cached_bev=True,
            )
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step == 1 or step % 64 == 0:
                print(
                    "V4_TRAIN",
                    branch,
                    step,
                    "loss",
                    record["loss"],
                    "gradient",
                    grad,
                    "update",
                    delta,
                    flush=True,
                )
    checkpoint = save_checkpoint(head, optimizer, directory, steps, branch)
    return dict(
        **checkpoint,
        seconds=time.perf_counter() - start,
        parameter_delta_l2=float(
            sum(
                (p.detach() - initial[n]).square().sum()
                for n, p in head.named_parameters()
            ).sqrt()
        ),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
        parameter_count=sum(p.numel() for p in head.parameters()),
        trainable_parameters=[n for n, p in head.named_parameters() if p.requires_grad],
    )


def main() -> None:
    """先诊断可学性，成功后执行预注册的四个消融分支。"""
    argparse.ArgumentParser(description=__doc__).parse_args()
    for p, digest in json.loads((OUT / "protocol.json").read_text())[
        "source_sha256"
    ].items():
        assert sha(ROOT / p) == digest, p
    model, wrapped, pipeline, dataset, manifest, features, text = load_runtime()
    hardware = dict(
        torch=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        python=sys.executable,
        nvidia_smi=subprocess.check_output(["nvidia-smi"], text=True),
        host_memory=subprocess.check_output(["free", "-h"], text=True),
    )
    from mmcv.ops import MultiScaleDeformableAttention
    import mmcv._ext

    hardware["mmcv_custom_ops_import"] = str(mmcv._ext.__file__)
    write(OUT / "hardware.json", hardware)
    train = sorted(
        [r for r in manifest["annotations"] if r["split"] == "train"],
        key=lambda r: r["instruction_id"],
    )
    first = next(r for r in train if r["counterfactual_text"])
    partner = next(
        r
        for r in train
        if r["sample_token"] == first["sample_token"]
        and r["text"] == first["counterfactual_text"]
    )
    paired = [first, partner]
    seed_all(SEED)
    head = LanguageConditionedTargetMaskHead().cuda().train()
    diagnostic = train_steps(
        head,
        paired,
        features,
        text,
        256,
        "diagnostic_pair",
        OUT / "diagnostic_pair",
        True,
    )
    values, wrong = [], []
    grid, geometry = frame_inputs(first)
    payload = {}
    with torch.no_grad():
        for i, row in enumerate(paired):
            logits = (
                head(features[row["index"]], text[row["text"]], grid, geometry)[0, 0]
                .cpu()
                .numpy()
            )
            cf = (
                head(
                    features[row["index"]],
                    text[row["counterfactual_text"]],
                    grid,
                    geometry,
                )[0, 0]
                .cpu()
                .numpy()
            )
            target = load_label(row)["target"].cpu().numpy()
            values.append(counts(target, logits >= 0))
            wrong.append(counts(target, cf >= 0))
            payload.update(
                {
                    f"target_{i}": target,
                    f"correct_logits_{i}": logits,
                    f"wrong_logits_{i}": cf,
                }
            )
    correct_iou = sum(v["tp"] for v in values) / sum(v["union"] for v in values)
    wrong_iou = sum(v["tp"] for v in wrong) / sum(v["union"] for v in wrong)
    passed = correct_iou >= 0.25 and correct_iou - wrong_iou > 0.05
    diagnostic.update(
        annotations=paired,
        correct=values,
        wrong=wrong,
        pooled_correct_iou=correct_iou,
        pooled_wrong_iou=wrong_iou,
        success=passed,
        interpretation="train-pair memorization only; never generalization evidence",
    )
    np.savez_compressed(OUT / "diagnostic_pair/predictions.npz", **payload)
    write(OUT / "diagnostic_pair/summary.json", diagnostic)
    print("V4_DIAGNOSTIC", correct_iou, wrong_iou, passed, flush=True)
    if not passed:
        write(
            OUT / "training_summary.json",
            dict(
                status="DIAGNOSTIC_FAILED",
                optimizer_updates=256,
                diagnostic=diagnostic,
                branches={},
                reason="pre-registered learnability threshold not reached; equal-budget expansion stopped",
            ),
        )
        return
    summaries = {}
    for branch in ("no_language", "shuffled_text", "class_only", "full_instruction"):
        seed_all(SEED)
        head = LanguageConditionedTargetMaskHead().cuda().train()
        summaries[branch] = train_steps(
            head, train, features, text, 384, branch, OUT / branch
        )
    assert not any(p.grad is not None for p in model.parameters())
    assert not any(p.grad is not None for p in pipeline.parameters())
    write(
        OUT / "training_summary.json",
        dict(
            status="COMPLETED",
            optimizer_updates=256 + 4 * 384,
            diagnostic=diagnostic,
            branches=summaries,
            frozen_author_gradient_tensors=0,
            frozen_pipeline_gradient_tensors=0,
            image_backbone_optimizer_updates=0,
            native_head_optimizer_updates=0,
            text_optimizer_updates=0,
        ),
    )


if __name__ == "__main__":
    main()

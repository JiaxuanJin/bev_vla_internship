"""LG-BEV 严格配对推理与内部开发计分；不打开封存场景。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch
from torch import Tensor, nn

from scripts.stage3.lg_bev_common import DEFAULT_OUT, read, sha, verify_protocol, write
from scripts.stage3.lg_bev_native import decode, dual_head, load_native
from scripts.stage3.lg_bev_prototype import LanguageGroundedBEVAdapter
from scripts.stage3.lg_bev_train import encode, frame_input

BRANCHES = ("original", "zero_text", "wrong_text", "correct_text")


@torch.no_grad()
def paired_forward(
    model: nn.Module,
    adapter: nn.Module,
    bev: Tensor,
    position: Tensor,
    correct_text: Tensor,
    wrong_text: Tensor,
) -> dict:
    """四分支共享帧、权重和标定；推理签名无 GT、标签路径或监督对象。"""
    predictions = {}
    for branch in BRANCHES:
        detail = None
        value = bev
        if branch != "original":
            embedding = (
                torch.zeros_like(correct_text)
                if branch == "zero_text"
                else wrong_text if branch == "wrong_text" else correct_text
            )
            detail = adapter(bev, embedding, position)
            value = detail["enhanced_bev"]
        prediction = dual_head(model, value)
        predictions[branch] = dict(
            **decode(model, prediction),
            semantic=prediction["seg_preds"].argmax(1)[0].cpu().numpy(),
            attention=(
                detail["attention_mask"].cpu().numpy() if detail is not None else None
            ),
            relative_residual_l2=float((value - bev).norm() / bev.norm()),
        )
    return predictions


def blocked_report(out: Path) -> None:
    """未过门槛时写明未运行原因，绝不补造性能值。"""
    protocol = verify_protocol(out)
    gate = read(out / "spatial_gate.json")
    if gate["pilot_allowed"]:
        raise ValueError("审计通过时应执行固定 pilot，不生成 BLOCKED 报告")
    metrics = dict(
        status="NOT_RUN",
        reason=gate["reason"],
        pilot="BLOCKED",
        n_evaluated_frames=0,
        n_evaluated_instructions=0,
        optimizer_updates_pilot=0,
        optimizer_updates_smoke=read(out / "smoke_test.json")["optimizer_updates"],
        branches={
            branch: dict(
                relevant_ap_2m_xy=None,
                relevant_segmentation_iou=None,
                irrelevant_segmentation_miou=None,
                attention_iou=None,
                distractor_fp_rate=None,
                background_fp_rate=None,
                status="NOT_RUN",
            )
            for branch in BRANCHES
        },
        definition=protocol["metrics"],
        final_holdout="BLOCKED",
        no_performance_claim=True,
    )
    write(out / "metrics.json", metrics)
    with (out / "metrics.csv").open("x") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["branch", "status", "n_frames", "relevant_AP@2m_xy", "relevant_seg_IoU"]
        )
        for branch in BRANCHES:
            writer.writerow([branch, "NOT_RUN", 0, "", ""])


def evaluate(out: Path) -> None:
    """只评价唯一 pilot 终点；先持久化无 GT 预测，再由计分器读取内部开发 GT。"""
    protocol = verify_protocol(out)
    if not read(out / "spatial_gate.json")["pilot_allowed"]:
        raise ValueError("空间审计 BLOCKED，禁止性能对照")
    pilot = read(out / "pilot_training.json")
    path = Path(pilot["checkpoint"]["path"])
    if sha(path) != pilot["checkpoint"]["sha256"]:
        raise ValueError("pilot checkpoint 已改变")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["protocol_sha256"] != sha(out / "protocol.json"):
        raise ValueError("checkpoint 协议错误")
    assert saved["optimizer_updates"] == protocol["budget"]["pilot_updates"]
    model, _, encoder, _ = load_native()
    rows = [
        r
        for r in read(out / "manifest.json")["annotations"]
        if r["split"] == "internal_dev"
    ]
    encoded = encode(encoder, rows)
    adapter = LanguageGroundedBEVAdapter(**protocol["model_config"]).cuda().eval()
    adapter.load_state_dict(saved["state_dict"])
    folder = out / "internal_dev_predictions"
    folder.mkdir(exist_ok=False)
    records, timings = [], []
    for index, row in enumerate(rows):
        bev, position = frame_input(row)
        torch.cuda.synchronize()
        started = time.perf_counter()
        predictions = paired_forward(
            model,
            adapter,
            bev,
            position,
            encoded[row["text"]],
            encoded[row["wrong_text"]],
        )
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        payload = {}
        for branch, prediction in predictions.items():
            payload.update(
                {
                    branch + "_" + k: v
                    for k, v in prediction.items()
                    if isinstance(v, np.ndarray)
                }
            )
        path = folder / f"instruction_{index:04d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **payload)
        records.append(
            dict(
                instruction_id=row["instruction_id"],
                prediction_file=str(path.relative_to(out)),
                sha256=sha(path),
            )
        )
    write(
        out / "prediction_manifest.json",
        dict(
            annotations=records,
            cached_four_branch_seconds=timings,
            reads_training_gt=False,
        ),
    )
    score(out)


def score(out: Path) -> None:
    """独立计分仅允许 internal_dev 标签；AP 保留空指令，限定词不混入集合 AP。"""
    from scripts.stage2.spatial_annotations import metric_grid, project_attention
    from scripts.stage2.spatial_detection import ego_centers, in_region
    from scripts.stage2.spatial_report import confusion, summary
    from scripts.stage2.stage3_metric_review import neighborhood
    from scripts.stage3.lg_bev_data import load_supervision

    rows = {
        r["instruction_id"]: r
        for r in read(out / "manifest.json")["annotations"]
        if r["split"] == "internal_dev"
    }
    counts = {
        b: dict(
            tp=0,
            union=0,
            distractor_fp=0,
            distractor_pixels=0,
            background_fp=0,
            background_pixels=0,
            seg_tp=0,
            seg_union=0,
        )
        for b in BRANCHES
    }
    matrices = {b: np.zeros((4, 4), dtype=np.int64) for b in BRANCHES}
    pools = {
        (b, cls, region): []
        for b in BRANCHES
        for cls in (0, 8)
        for region in ("relevant", "irrelevant")
    }
    gt_counts = {key: 0 for key in pools}
    for record in read(out / "prediction_manifest.json")["annotations"]:
        row = rows[record["instruction_id"]]
        assert row["split"] == "internal_dev"
        label = load_supervision(row, out, training=False)
        target = label["target"][0, 0].numpy()
        distractor = label["distractor"][0, 0].numpy()
        path = out / record["prediction_file"]
        assert sha(path) == record["sha256"]
        with np.load(path) as predictions:
            for branch in BRANCHES:
                c = counts[branch]
                if branch != "original":
                    attention = torch.from_numpy(predictions[branch + "_attention"])
                    if row["task"] == "map":
                        attention = project_attention(attention, metric_grid(row)[0])
                    active = attention[0, 0].numpy() >= 0.5
                    c["tp"] += int((active & target).sum())
                    c["union"] += int((active | target).sum())
                    c["distractor_fp"] += int((active & distractor).sum())
                    c["distractor_pixels"] += int(distractor.sum())
                    background = ~target & ~distractor
                    c["background_fp"] += int((active & background).sum())
                    c["background_pixels"] += int(background.sum())
                if row["task"] == "map":
                    with np.load(row["source_file"]) as source:
                        gt, xy = source["gt"], source["ego_xy"]
                    semantic = predictions[branch + "_semantic"]
                    region = (
                        (xy[..., 0] > 0) if row["fb"] == "front" else (xy[..., 0] < 0)
                    ) & ((xy[..., 1] > 0) if row["lr"] == "left" else (xy[..., 1] < 0))
                    roi = neighborhood(target) & region
                    truth, active = (
                        gt == row["target_class"],
                        semantic == row["target_class"],
                    )
                    c["seg_tp"] += int((truth & active & roi).sum())
                    c["seg_union"] += int(((truth | active) & roi).sum())
                    matrices[branch] += confusion(gt, semantic, ~region)
                elif row["qualifier"] == "all":
                    boxes = label["boxes"].numpy()
                    labels = label["labels"].numpy()
                    prompt = row["prompt"]
                    pred_boxes = predictions[branch + "_boxes"]
                    scores, classes = (
                        predictions[branch + "_scores"],
                        predictions[branch + "_labels"],
                    )
                    inside = in_region(ego_centers(boxes, row), prompt)
                    pred_inside = in_region(ego_centers(pred_boxes, row), prompt)
                    for region, is_inside in (
                        ("relevant", True),
                        ("irrelevant", False),
                    ):
                        key = (branch, prompt["class_id"], region)
                        valid = (labels == prompt["class_id"]) & (inside == is_inside)
                        indices = np.flatnonzero(valid)
                        gt_counts[key] += len(indices)
                        used = set()
                        chosen = np.flatnonzero(
                            (classes == prompt["class_id"]) & (pred_inside == is_inside)
                        )
                        for j in chosen[np.argsort(-scores[chosen], kind="stable")]:
                            eligible = [k for k in indices if k not in used]
                            positive = False
                            if eligible:
                                distances = np.linalg.norm(
                                    boxes[eligible, :2] - pred_boxes[j, :2], axis=1
                                )
                                nearest = int(np.argmin(distances))
                                if distances[nearest] <= 2.0:
                                    positive = True
                                    used.add(eligible[nearest])
                            pools[key].append((float(scores[j]), positive))
    result = {}
    for branch in BRANCHES:
        c = counts[branch]
        ap = {}
        for region in ("relevant", "irrelevant"):
            per_class = []
            for cls in (0, 8):
                key = (branch, cls, region)
                tp, numerator = 0, 0.0
                for k, (_, positive) in enumerate(
                    sorted(pools[key], key=lambda v: -v[0]), 1
                ):
                    tp += int(positive)
                    numerator += tp / k if positive else 0
                if gt_counts[key]:
                    per_class.append(numerator / gt_counts[key])
            ap[region] = float(np.mean(per_class)) if per_class else None
        result[branch] = dict(
            relevant_ap_2m_xy=ap["relevant"],
            irrelevant_ap_2m_xy=ap["irrelevant"],
            relevant_segmentation_iou=(
                c["seg_tp"] / c["seg_union"] if c["seg_union"] else None
            ),
            irrelevant_segmentation_miou=summary(matrices[branch])["foreground_miou"],
            attention_iou=c["tp"] / c["union"] if c["union"] else None,
            distractor_fp_rate=(
                c["distractor_fp"] / c["distractor_pixels"]
                if c["distractor_pixels"]
                else None
            ),
            background_fp_rate=(
                c["background_fp"] / c["background_pixels"]
                if c["background_pixels"]
                else None
            ),
            raw_counts=c,
        )
    write(
        out / "metrics.json",
        dict(
            status="INTERNAL_DEVELOPMENT_ONLY",
            branches=result,
            n_instructions=len(rows),
            n_frames=len({r["sample_token"] for r in rows.values()}),
            final_holdout="BLOCKED",
            metric="nonofficial AP@2m_xy; no threshold selection",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--blocked-report", action="store_true")
    args = parser.parse_args()
    (blocked_report if args.blocked_report else evaluate)(args.out.resolve())

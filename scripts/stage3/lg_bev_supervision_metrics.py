"""原生双 Head 不变的固定配对计分，Attention 只在独立标定空间评价。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import time

import numpy as np
import torch

from scripts.stage2.spatial_detection import ego_centers, in_region
from scripts.stage2.spatial_report import confusion, summary
from scripts.stage2.stage3_metric_review import neighborhood
from scripts.stage3.lg_bev_common import read, sha, verify_protocol, write
from scripts.stage3.lg_bev_data import box_masks, load_supervision
from scripts.stage3.lg_bev_native import decode, dual_head, load_native, module_sha
from scripts.stage3.lg_bev_prototype import LanguageGroundedBEVAdapter
from scripts.stage3.lg_bev_supervision_space import AttentionSupervisionSpace
from scripts.stage3.lg_bev_supervision_run import SPACE_OUT
from scripts.stage3.lg_bev_train import encode, frame_input

BRANCHES = ("original", "zero_text", "wrong_text", "correct_text")


def average_precision(predictions: list[tuple[float, bool]], gt_count: int) -> dict:
    """不插值 AP；空目标预测留在池中，零 GT 类别的 AP 为 None。"""
    tp, numerator = 0, 0.0
    for rank, (_, positive) in enumerate(sorted(predictions, key=lambda r: -r[0]), 1):
        tp += int(positive)
        if positive:
            numerator += tp / rank
    return dict(
        ap=numerator / gt_count if gt_count else None,
        gt_count=gt_count,
        predictions=len(predictions),
        tp=tp,
        fp=len(predictions) - tp,
        fn=gt_count - tp,
    )


def match_boxes(
    gt_boxes: np.ndarray,
    gt_indices: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    pred_indices: np.ndarray,
) -> list[tuple[float, bool]]:
    """同类/区域分区后的置信度贪心唯一匹配；固定2m XY，不拟合阈值。"""
    used = set()
    result = []
    order = pred_indices[np.argsort(-scores[pred_indices], kind="stable")]
    for index in order:
        candidates = [k for k in gt_indices if k not in used]
        positive = False
        if candidates:
            distances = np.linalg.norm(
                gt_boxes[candidates, :2] - boxes[index, :2], axis=1
            )
            nearest = int(np.argmin(distances))
            if distances[nearest] <= 2.0:
                used.add(candidates[nearest])
                positive = True
        result.append((float(scores[index]), positive))
    return result


def mask_counts(
    active: np.ndarray,
    target: np.ndarray,
    distractor: np.ndarray,
    background: np.ndarray,
) -> dict:
    """背景仅原始 GT 背景，避免把其他类别前景混叫背景。"""
    assert not (target & distractor).any()
    return dict(
        tp=int((active & target).sum()),
        union=int((active | target).sum()),
        distractor_fp=int((active & distractor).sum()),
        distractor_pixels=int(distractor.sum()),
        background_fp=int((active & background).sum()),
        background_pixels=int(background.sum()),
        empty_prompts=int(not target.any()),
        empty_prompt_active_pixels=int(active.sum()) if not target.any() else 0,
    )


def mask_summary(counts: dict) -> dict:
    """保留计数及空指令分母；不同像素尺度任务分别汇总。"""
    return dict(
        counts,
        iou=counts["tp"] / counts["union"] if counts["union"] else None,
        distractor_fp_rate=(
            counts["distractor_fp"] / counts["distractor_pixels"]
            if counts["distractor_pixels"]
            else None
        ),
        background_fp_rate=(
            counts["background_fp"] / counts["background_pixels"]
            if counts["background_pixels"]
            else None
        ),
    )


@torch.no_grad()
def paired_inference(
    model, adapter, bev, position, correct_text, wrong_text
) -> tuple[dict, dict]:
    """无 GT/label 路径接口；同一原生BEV、同一权重、同一未改动双Head。"""
    output, timings = {}, {}
    for branch in BRANCHES:
        torch.cuda.synchronize()
        start = time.perf_counter()
        detail = None
        value = bev
        if branch != "original":
            text = (
                torch.zeros_like(correct_text)
                if branch == "zero_text"
                else wrong_text if branch == "wrong_text" else correct_text
            )
            detail = adapter(bev, text, position)
            value = detail["enhanced_bev"]
        prediction = dual_head(model, value)
        output[branch] = dict(
            **decode(model, prediction),
            semantic=prediction["seg_preds"]
            .argmax(1)[0]
            .cpu()
            .numpy()
            .astype(np.uint8),
        )
        if detail is not None:
            output[branch]["attention_logits"] = (
                detail["attention_logits"].cpu().numpy()
            )
        torch.cuda.synchronize()
        timings[branch] = (time.perf_counter() - start) * 1000
    return output, timings


def infer(out: Path) -> None:
    """唯一16步终点，内部开发只打开一次；不访问任何封存集。"""
    protocol = verify_protocol(out)
    if not read(out / "spatial_gate.json")["pilot_allowed"]:
        raise ValueError("监督空间门槛未通过")
    saved_record = read(out / "pilot_training.json")["checkpoint"]
    path = Path(saved_record["path"])
    assert sha(path) == saved_record["sha256"]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["optimizer_updates"] == 16
    assert saved["protocol_sha256"] == sha(out / "protocol.json")
    folder = out / "internal_dev_predictions"
    folder.mkdir(exist_ok=False)
    model, _, encoder, _ = load_native()
    frozen = module_sha(model)
    adapter = LanguageGroundedBEVAdapter(**protocol["model_config"]).cuda().eval()
    adapter.load_state_dict(saved["state_dict"])
    rows = [
        r
        for r in read(out / "manifest.json")["annotations"]
        if r["split"] == "internal_dev"
    ]
    text = encode(encoder, rows)
    frame_inputs, originals = {}, {}
    records, timings = [], []
    for index, row in enumerate(rows):
        token = row["sample_token"]
        if token not in frame_inputs:
            frame_inputs[token] = frame_input(row)
        bev, position = frame_inputs[token]
        prediction, times = paired_inference(
            model, adapter, bev, position, text[row["text"]], text[row["wrong_text"]]
        )
        if token not in originals:
            originals[token] = prediction["original"]
        else:
            assert all(
                np.array_equal(v, originals[token][k])
                for k, v in prediction["original"].items()
            )
        payload = {
            f"{branch}_{key}": value
            for branch, item in prediction.items()
            for key, value in item.items()
        }
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
        timings.append(times)
        if index % 24 == 0:
            print("PAIRED_INTERNAL_DEV", index, len(rows), flush=True)
    assert module_sha(model) == frozen
    assert "forward" not in model.pts_bbox_head.feat_cropper.__dict__
    write(
        out / "prediction_manifest.json",
        dict(
            annotations=records,
            cached_paired_ms=timings,
            original_per_frame_bitwise_consistent=True,
            frozen_author_sha256=frozen,
            reads_training_gt=False,
        ),
    )
    write(
        out / "latency.json",
        dict(
            scope="cached BEV native dual heads + decode + CPU output copy, including adapter for LG; excludes camera backbone/text/cache I/O; untrimmed all189 pairs",
            gpu=torch.cuda.get_device_name(),
            cached_ms={
                b: dict(
                    mean=float(np.mean([r[b] for r in timings])),
                    median=float(np.median([r[b] for r in timings])),
                    samples=len(timings),
                )
                for b in BRANCHES
            },
            lg_six_camera_end_to_end="NOT_RUN",
            RTX3090="NOT_EVALUATED",
        ),
    )
    score(out)


def score(out: Path) -> None:
    """只读已保存预测后再读取 internal_dev GT；绝不读取 train GT 计分。"""
    protocol = verify_protocol(out)
    rows = {
        r["instruction_id"]: r
        for r in read(out / "manifest.json")["annotations"]
        if r["split"] == "internal_dev"
    }
    counts = {(b, t): defaultdict(int) for b in BRANCHES for t in ("map", "object")}
    matrices = {b: np.zeros((4, 4), dtype=np.int64) for b in BRANCHES}
    seg_counts = {b: defaultdict(int) for b in BRANCHES}
    pools = {
        (b, cls, region): []
        for b in BRANCHES
        for cls in (0, 8)
        for region in ("relevant", "irrelevant")
    }
    gt_counts = defaultdict(int)
    saved = read(out / "prediction_manifest.json")["annotations"]
    per_instruction = []
    for record in saved:
        row = rows[record["instruction_id"]]
        assert row["split"] == "internal_dev"
        path = out / record["prediction_file"]
        assert sha(path) == record["sha256"]
        label = load_supervision(row, out, training=False)
        target = label["target"][0, 0].numpy()
        distractor = label["distractor"][0, 0].numpy()
        if row["task"] == "map":
            with np.load(row["source_file"]) as source:
                gt, xy = source["gt"], source["ego_xy"]
            background = gt == 0
            region = (
                (xy[..., 0] > 0) if row["fb"] == "front" else (xy[..., 0] < 0)
            ) & ((xy[..., 1] > 0) if row["lr"] == "left" else (xy[..., 1] < 0))
            roi = neighborhood(target) & region
        else:
            gt_boxes, gt_labels = label["boxes"].numpy(), label["labels"].numpy()
            all_masks, _ = box_masks(gt_boxes)
            background = ~all_masks.any(0)
        with np.load(path) as prediction:
            for branch in BRANCHES:
                if branch != "original":
                    space = AttentionSupervisionSpace.from_calibration(row["task"], row)
                    # 与训练相同：先投影 logits，再 sigmoid/固定0.5阈值。
                    logits = space.project(
                        torch.from_numpy(prediction[branch + "_attention_logits"])
                    )
                    active = logits[0, 0].numpy() >= 0
                    values = mask_counts(active, target, distractor, background)
                    for key, value in values.items():
                        counts[(branch, row["task"])][key] += value
                    per_instruction.append(
                        dict(
                            branch=branch,
                            instruction_id=row["instruction_id"],
                            task=row["task"],
                            **values,
                        )
                    )
                if row["task"] == "map":
                    semantic = prediction[branch + "_semantic"]
                    truth, active = (
                        gt == row["target_class"],
                        semantic == row["target_class"],
                    )
                    seg_counts[branch]["tp"] += int((truth & active & roi).sum())
                    seg_counts[branch]["union"] += int(((truth | active) & roi).sum())
                    matrices[branch] += confusion(gt, semantic, ~region)
                elif row["qualifier"] == "all":
                    prompt = row["prompt"]
                    boxes, scores, labels = (
                        prediction[branch + "_" + key]
                        for key in ("boxes", "scores", "labels")
                    )
                    gt_inside = in_region(ego_centers(gt_boxes, row), prompt)
                    pred_inside = in_region(ego_centers(boxes, row), prompt)
                    for region_name, inside in (
                        ("relevant", True),
                        ("irrelevant", False),
                    ):
                        key = (branch, prompt["class_id"], region_name)
                        gi = np.flatnonzero(
                            (gt_labels == prompt["class_id"]) & (gt_inside == inside)
                        )
                        pi = np.flatnonzero(
                            (labels == prompt["class_id"]) & (pred_inside == inside)
                        )
                        pools[key].extend(match_boxes(gt_boxes, gi, boxes, scores, pi))
                        gt_counts[key] += len(gi)
    result = {}
    for branch in BRANCHES:
        detection = {}
        for region_name in ("relevant", "irrelevant"):
            classes = {
                str(cls): average_precision(
                    pools[(branch, cls, region_name)],
                    gt_counts[(branch, cls, region_name)],
                )
                for cls in (0, 8)
            }
            aps = [v["ap"] for v in classes.values() if v["ap"] is not None]
            detection[region_name] = dict(
                macro_ap=float(np.mean(aps)) if aps else None, classes=classes
            )
        result[branch] = dict(
            relevant_ap_2m_xy=detection["relevant"]["macro_ap"],
            irrelevant_ap_2m_xy=detection["irrelevant"]["macro_ap"],
            detection=detection,
            relevant_segmentation_iou=seg_counts[branch]["tp"]
            / seg_counts[branch]["union"],
            relevant_segmentation_counts=dict(seg_counts[branch]),
            irrelevant_segmentation=summary(matrices[branch]),
            attention=(
                {
                    task: mask_summary(counts[(branch, task)])
                    for task in ("map", "object")
                }
                if branch != "original"
                else None
            ),
        )
    original = result["original"]
    for branch, value in result.items():
        value["changes_pp_vs_original"] = {
            key: 100 * (value[key] - original[key])
            for key in (
                "relevant_ap_2m_xy",
                "irrelevant_ap_2m_xy",
                "relevant_segmentation_iou",
            )
        }
        value["changes_pp_vs_original"]["irrelevant_segmentation_miou"] = 100 * (
            value["irrelevant_segmentation"]["foreground_miou"]
            - original["irrelevant_segmentation"]["foreground_miou"]
        )
    checks = {}
    for control in ("original", "zero_text", "wrong_text"):
        for key in ("relevant_ap_2m_xy", "relevant_segmentation_iou"):
            checks[f"correct_exceeds_{control}_{key}"] = (
                result["correct_text"][key] > result[control][key]
            )
    checks["irrelevant_ap_protected"] = (
        result["correct_text"]["irrelevant_ap_2m_xy"]
        >= original["irrelevant_ap_2m_xy"] - 0.001
    )
    checks["irrelevant_segmentation_protected"] = (
        result["correct_text"]["irrelevant_segmentation"]["foreground_miou"]
        >= original["irrelevant_segmentation"]["foreground_miou"] - 0.001
    )
    write(
        out / "metrics.json",
        dict(
            status="INTERNAL_DEVELOPMENT_ONLY",
            branches=result,
            n_frames=len({r["sample_token"] for r in rows.values()}),
            n_scenes=1,
            n_map_instructions=sum(r["task"] == "map" for r in rows.values()),
            n_object_instructions=sum(r["task"] == "object" for r in rows.values()),
            n_plural_ap_instructions=sum(
                r["task"] == "object" and r["qualifier"] == "all" for r in rows.values()
            ),
            metric=protocol["metrics"],
            exposure=protocol["exposure"],
            final_holdout="BLOCKED",
            controls="zero/wrong are inference ablations of the same trained checkpoint, not separately trained controls",
        ),
    )
    write(
        out / "pilot_decision.json",
        dict(
            checks=checks,
            useful_on_this_internal_dev=all(checks.values()),
            expand=False,
            budget_exhausted=True,
            decision="SEAL_FIXED16_TERMINAL_NO_RETRIES",
            no_threshold_lr_architecture_selection=True,
            final_holdout="NOT_RUN",
        ),
    )
    with (out / "metrics.csv").open("x") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "branch",
                "relevant_ap_2m_xy",
                "irrelevant_ap_2m_xy",
                "relevant_segmentation_iou",
                "irrelevant_segmentation_miou",
                "map_attention_iou",
                "object_attention_iou",
            ],
        )
        writer.writeheader()
        for branch, value in result.items():
            writer.writerow(
                dict(
                    branch=branch,
                    relevant_ap_2m_xy=value["relevant_ap_2m_xy"],
                    irrelevant_ap_2m_xy=value["irrelevant_ap_2m_xy"],
                    relevant_segmentation_iou=value["relevant_segmentation_iou"],
                    irrelevant_segmentation_miou=value["irrelevant_segmentation"][
                        "foreground_miou"
                    ],
                    map_attention_iou=(
                        value["attention"]["map"]["iou"] if value["attention"] else None
                    ),
                    object_attention_iou=(
                        value["attention"]["object"]["iou"]
                        if value["attention"]
                        else None
                    ),
                )
            )
    write(out / "attention_per_instruction.json", per_instruction)
    print(
        "PILOT_METRICS_WRITTEN",
        {
            b: (v["relevant_ap_2m_xy"], v["relevant_segmentation_iou"])
            for b, v in result.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SPACE_OUT)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    (score if args.score_only else infer)(args.out.resolve())

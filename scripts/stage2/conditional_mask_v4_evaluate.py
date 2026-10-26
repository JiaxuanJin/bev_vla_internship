"""V4 三套独立指标、真实对象 AP 迁移诊断与失败图；不选择模型。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.stage2.conditional_mask_v4_data import OUT, sha, write
from scripts.stage2.conditional_mask_v4_experiment import (
    class_text,
    counts,
    frame_inputs,
    load_label,
    load_runtime,
    shuffle_words,
)
from scripts.stage2.conditional_mask_v4_model import LanguageConditionedTargetMaskHead
from scripts.stage2.spatial_annotations import project_attention
from scripts.stage2.spatial_report import confusion, summary
from scripts.stage2.spatial_v2_experiment import swap_direction
from scripts.stage2.spatial_v3_evaluate import ap_metrics, segmentation
from scripts.stage2.spatial_v3_experiment import decode, head_forward
from scripts.stage2.spatial_v3_objects import object_rows
from scripts.stage2.stage3_metric_review import neighborhood


def csv_write(path: Path, records: list[dict]) -> None:
    """原始逐指令数值与汇总表均保持可直接审阅。"""
    with path.open("x") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in records for k in r}))
        writer.writeheader()
        writer.writerows(records)


def pooled(records: list[dict]) -> dict:
    """汇总交并集而非平均逐指令 IoU；保留 FP 分母。"""
    values = Counter()
    for r in records:
        for key in (
            "tp",
            "gt",
            "pred",
            "union",
            "background_fp",
            "background_pixels",
            "distractor_fp",
            "distractor_pixels",
            "outside_roi_fp",
            "outside_roi_negative",
        ):
            values[key] += r[key]
    tp, gt, pred, union = (values[k] for k in ("tp", "gt", "pred", "union"))
    return dict(
        **values,
        iou=tp / union if union else None,
        precision=tp / pred if pred else 0.0,
        recall=tp / gt if gt else None,
        f1=2 * tp / (gt + pred) if gt + pred else None,
        background_fp_rate=(
            values["background_fp"] / values["background_pixels"]
            if values["background_pixels"]
            else None
        ),
        distractor_fp_rate=(
            values["distractor_fp"] / values["distractor_pixels"]
            if values["distractor_pixels"]
            else None
        ),
        n_instructions=len(records),
    )


def measure(
    target: np.ndarray,
    logits: np.ndarray,
    label: dict,
    gt: np.ndarray,
    threshold: float,
    roi: np.ndarray,
) -> dict:
    """完整物理窗口统计，不能通过只裁剪目标 ROI 虚增定位精度。"""
    probability = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    pred = probability >= threshold
    result = counts(target, pred)
    other = label["quadrant_other"]
    result.update(
        background_fp=int((pred & (gt == 0)).sum()),
        background_pixels=int((gt == 0).sum()),
        distractor_fp=int((pred & other).sum()),
        distractor_pixels=int(other.sum()),
        outside_roi_fp=int((pred & ~target & ~roi).sum()),
        outside_roi_negative=int((~target & ~roi).sum()),
    )
    return result


def evaluate_maps(
    model: Any,
    pipeline: Any,
    manifest: dict,
    features: dict,
    text: dict,
    training: dict,
    evaluation_split: str = "dev",
) -> dict:
    """所有终点模型在固定新开发集上评价；原双 Head 与新分支互相独立。"""
    directory = OUT / f"{evaluation_split}_predictions"
    directory.mkdir()
    rows = [r for r in manifest["annotations"] if r["split"] == evaluation_split]
    frame_predictions = {}
    for f in manifest["frames"]:
        if f["split"] != evaluation_split:
            continue
        with torch.no_grad():
            value = segmentation(model.pts_bbox_head, features[f["index"]])
        frame_predictions[f["index"]] = value.argmax(1)[0].cpu().numpy()
    mask_records, semantic_records, metadata = [], [], []
    matrices = {
        b: {
            k: np.zeros((4, 4), dtype=np.int64)
            for k in ("irrelevant", "global", "quadrant")
        }
        for b in ("original", "v3_frozen", "v4_independent_head")
    }
    checkpoints = training["branches"] or {
        "diagnostic_pair_only": training["diagnostic"]
    }
    heads = {}
    for name, record in checkpoints.items():
        assert sha(record["checkpoint"]) == record["sha256"]
        head = LanguageConditionedTargetMaskHead().cuda().eval()
        head.load_state_dict(
            torch.load(record["checkpoint"], map_location="cpu", weights_only=False)[
                "state_dict"
            ]
        )
        heads[name] = head
    thresholds = json.loads((OUT / "protocol.json").read_text())["thresholds"]
    for n, row in enumerate(rows):
        label = {k: v.cpu().numpy() for k, v in load_label(row).items()}
        target, region = label["target"], label["region"]
        with np.load(row["source_file"]) as f:
            gt = f["gt"]
        roi = neighborhood(target) & region
        grid, geometry = frame_inputs(row)
        bev = features[row["index"]]
        pipeline.fusion.set_calibration(row)
        with torch.no_grad():
            detail = pipeline.fusion.forward_with_details(bev, text[row["text"]])
            v3 = (
                segmentation(model.pts_bbox_head, detail["enhanced_bev"])
                .argmax(1)[0]
                .cpu()
                .numpy()
            )
            attention = project_attention(detail["attention"], grid)[0, 0].cpu().numpy()
        original = frame_predictions[row["index"]]
        payload = dict(
            gt=gt,
            target=target,
            region=region,
            target_neighborhood=roi,
            same_class_other=label["same_class_other"],
            quadrant_other=label["quadrant_other"],
            original_semantic=original,
            v3_semantic=v3,
            v4_semantic=original,
            v3_attention=attention,
        )
        for branch, pred in (
            ("original", original),
            ("v3_frozen", v3),
            ("v4_independent_head", original),
        ):
            cls = row["target_class"]
            truth, predicted = gt == cls, pred == cls
            inter = int((truth & predicted & roi).sum())
            union = int(((truth | predicted) & roi).sum())
            semantic_records.append(
                dict(
                    branch=branch,
                    instruction_id=row["instruction_id"],
                    sample_token=row["sample_token"],
                    instance_id=row["instance_id"],
                    target_class=cls,
                    intersection=inter,
                    union=union,
                    gt_pixels=int((truth & roi).sum()),
                    predicted_pixels=int((predicted & roi).sum()),
                    roi_pixels=int(roi.sum()),
                    outside_target_neighborhood_fp=int(
                        (predicted & ~truth & ~roi).sum()
                    ),
                    semantic_changed_pixels_vs_original=int((pred != original).sum()),
                )
            )
            matrices[branch]["irrelevant"] += confusion(gt, pred, ~region)
            matrices[branch]["quadrant"] += confusion(gt, pred, region)
            matrices[branch]["global"] += confusion(gt, pred, np.ones_like(region))
        for branch, head in heads.items():
            cf_text = row["counterfactual_text"] or swap_direction(row["text"])
            evaluation_texts = dict(
                correct=row["text"],
                wrong=cf_text,
                word_shuffled=shuffle_words(row["text"]),
            )
            for control, sentence in evaluation_texts.items():
                if branch == "no_language":
                    embedding = torch.zeros_like(text[row["text"]])
                elif branch == "class_only":
                    embedding = text[class_text(row)]
                else:
                    embedding = text[sentence]
                with torch.no_grad():
                    logits = head(bev, embedding, grid, geometry)[0, 0].cpu().numpy()
                payload[f"{branch}_{control}_logits"] = logits
                payload[f"{branch}_{control}_mask"] = logits >= 0
                for threshold in thresholds:
                    values = measure(target, logits, label, gt, threshold, roi)
                    mask_records.append(
                        dict(
                            branch=branch,
                            text_control=control,
                            threshold=threshold,
                            instruction_id=row["instruction_id"],
                            sample_token=row["sample_token"],
                            instance_id=row["instance_id"],
                            actual_text=(
                                sentence
                                if branch not in ("no_language", "class_only")
                                else (
                                    "<NO_LANGUAGE>"
                                    if branch == "no_language"
                                    else class_text(row)
                                )
                            ),
                            paired_same_quadrant=bool(row["counterfactual_text"]),
                            **values,
                        )
                    )
        attention_logits = np.log(
            np.clip(attention, 1e-7, 1 - 1e-7) / np.clip(1 - attention, 1e-7, 1)
        )
        mask_records.append(
            dict(
                branch="v3_attention",
                text_control="correct",
                threshold=0.5,
                instruction_id=row["instruction_id"],
                sample_token=row["sample_token"],
                instance_id=row["instance_id"],
                actual_text=row["text"],
                paired_same_quadrant=bool(row["counterfactual_text"]),
                **measure(target, attention_logits, label, gt, 0.5, roi),
            )
        )
        filename = f"instruction_{n:03d}.npz"
        np.savez_compressed(directory / filename, **payload)
        metadata.append(
            {
                **row,
                "prediction_file": filename,
                "prediction_sha256": sha(directory / filename),
            }
        )
        if n % 16 == 0:
            print("V4_DEV_MASK", n, len(rows), flush=True)
    semantic = {}
    for branch in matrices:
        records = [r for r in semantic_records if r["branch"] == branch]
        tp, union = sum(r["intersection"] for r in records), sum(
            r["union"] for r in records
        )
        semantic[branch] = dict(
            target_neighborhood_semantic_iou=tp / union,
            intersection=tp,
            union=union,
            gt_pixels=sum(r["gt_pixels"] for r in records),
            predicted_pixels=sum(r["predicted_pixels"] for r in records),
            outside_target_neighborhood_fp=sum(
                r["outside_target_neighborhood_fp"] for r in records
            ),
            **{k: summary(m) for k, m in matrices[branch].items()},
        )
    metrics = {}
    for branch in (*heads, "v3_attention"):
        primary = [
            r for r in mask_records if r["branch"] == branch and r["threshold"] == 0.5
        ]
        metrics[branch] = {
            c: pooled([r for r in primary if r["text_control"] == c])
            for c in sorted({r["text_control"] for r in primary})
        }
        paired = [r for r in primary if r["paired_same_quadrant"]]
        if branch != "v3_attention":
            correct = pooled([r for r in paired if r["text_control"] == "correct"])
            wrong = pooled([r for r in paired if r["text_control"] == "wrong"])
            lookup = {(r["instruction_id"], r["text_control"]): r for r in paired}
            ids = {r["instruction_id"] for r in paired}
            metrics[branch]["same_direction_counterfactual"] = dict(
                correct=correct,
                wrong=wrong,
                pooled_iou_gap=(
                    correct["iou"] - wrong["iou"]
                    if correct["iou"] is not None
                    else None
                ),
                directed_pairs=len(ids),
                unordered_pairs=len(ids) // 2,
                wins=sum(
                    lookup[(i, "correct")]["iou"] > lookup[(i, "wrong")]["iou"]
                    for i in ids
                ),
                ties=sum(
                    lookup[(i, "correct")]["iou"] == lookup[(i, "wrong")]["iou"]
                    for i in ids
                ),
            )
        metrics[branch]["threshold_sensitivity"] = {
            str(t): pooled(
                [
                    r
                    for r in mask_records
                    if r["branch"] == branch
                    and r["threshold"] == t
                    and r["text_control"] == "correct"
                ]
            )
            for t in thresholds
            if any(r["branch"] == branch and r["threshold"] == t for r in mask_records)
        }
    write(
        directory / "predictions.json",
        dict(annotations=metadata, checkpoints=checkpoints),
    )
    write(
        directory / "metrics.json",
        dict(
            grounding=metrics,
            segmentation=semantic,
            n_instructions=len(rows),
            n_frames=len(frame_predictions),
            n_scenes=len({r["scene_token"] for r in rows}),
            generalization="V4 scratch-head dev; final supplementary adaptation-scene evaluation if preregistered separately; pretraining-unseen BLOCKED",
            threshold_selection="fixed 0.5 before training; sensitivity only, no dev threshold choice",
        ),
    )
    csv_write(directory / "mask_per_instruction.csv", mask_records)
    csv_write(directory / "semantic_per_instruction.csv", semantic_records)
    plot_examples(directory, metadata, checkpoints, mask_records)
    return dict(
        grounding=metrics,
        segmentation=semantic,
        n_instructions=len(rows),
        n_frames=len(frame_predictions),
        n_scenes=len({r["scene_token"] for r in rows}),
    )


def plot_examples(
    directory: Path, rows: list[dict], checkpoints: dict, records: list[dict]
) -> None:
    """每帧第一对指令及全开发集最差样本，展示真实失败。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branch = (
        "full_instruction"
        if "full_instruction" in checkpoints
        else next(iter(checkpoints))
    )
    selected = []
    for index in sorted({r["index"] for r in rows}):
        candidates = [
            r for r in rows if r["index"] == index and r["counterfactual_text"]
        ]
        if candidates:
            selected.append(candidates[0])
    worst = min(
        [
            r
            for r in records
            if r["branch"] == branch
            and r["threshold"] == 0.5
            and r["text_control"] == "correct"
        ],
        key=lambda r: r["iou"],
    )
    failure = next(r for r in rows if r["instruction_id"] == worst["instruction_id"])
    if failure not in selected:
        selected.append(failure)
    for n, row in enumerate(selected):
        with np.load(directory / row["prediction_file"]) as f:
            a = {k: f[k] for k in f.files}
        panels = [
            ("Original map GT", a["gt"], "tab10", 3),
            ("Selected real instance", a["target"], "gray_r", 1),
            ("Same-quadrant distractors", a["quadrant_other"], "gray_r", 1),
            ("Frozen V3 Attention", a["v3_attention"], "magma", 1),
            ("Correct conditional Mask", a[branch + "_correct_mask"], "gray_r", 1),
            ("Wrong nearest/farthest Mask", a[branch + "_wrong_mask"], "gray_r", 1),
            ("Word-shuffled Mask", a[branch + "_word_shuffled_mask"], "gray_r", 1),
            ("Original semantic", a["original_semantic"], "tab10", 3),
            ("V4 original semantic Head", a["v4_semantic"], "tab10", 3),
            (
                "Instance FP / FN",
                (a[branch + "_correct_mask"] & ~a["target"]).astype(int)
                + 2 * (a["target"] & ~a[branch + "_correct_mask"]),
                "viridis",
                2,
            ),
        ]
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
        for ax, (title, value, cmap, hi) in zip(axes.flat, panels):
            ax.imshow(
                value,
                origin="lower",
                extent=(-30, 30, -15, 15),
                cmap=cmap,
                vmin=0,
                vmax=hi,
                interpolation="nearest",
            )
            ax.set_title(title, fontsize=9)
        fig.suptitle(
            f"{row['split'].upper()} sample={row['sample_token']} | {row['text']}\ninstance={row['instance_id']} | {branch} ckpt={checkpoints[branch]['sha256']}\nwrong={row['counterfactual_text'] or 'direction swap; no same-quadrant instance pair'} | frozen original Heads; no segmentation gain implied",
            fontsize=8,
        )
        fig.savefig(directory / f"comparison_{n:02d}.png", dpi=120)
        plt.close(fig)


def evaluate_objects(
    model: Any,
    pipeline: Any,
    dataset: Any,
    manifest: dict,
    features: dict,
    text: dict,
    evaluation_split: str = "dev",
    supplied_rows: list[dict] | None = None,
) -> dict:
    """真实对象匹配与原生 top300 AP；新增地图 Mask 分支不参与检测。"""
    directory = OUT / f"{evaluation_split}_objects"
    directory.mkdir()
    frames = [f for f in manifest["frames"] if f["split"] == evaluation_split]
    rows = object_rows(frames, dataset) if supplied_rows is None else supplied_rows
    saved, original_by_frame = [], {}
    for n, row in enumerate(rows):
        index = row["index"]
        bev = features[index]
        ann = dataset.get_ann_info(index)
        if index not in original_by_frame:
            with torch.no_grad():
                original_by_frame[index] = decode(model, head_forward(model, bev))
        original = original_by_frame[index]
        pipeline.fusion.set_calibration(row)
        with torch.no_grad():
            enhanced = pipeline.fusion.forward_with_details(bev, text[row["text"]])[
                "enhanced_bev"
            ]
            v3 = decode(model, head_forward(model, enhanced))
        payload = dict(
            gt_boxes=ann["gt_bboxes_3d"].tensor.numpy(), gt_labels=ann["gt_labels_3d"]
        )
        for branch, result in (
            ("original", original),
            ("v3_frozen", v3),
            ("v4_independent_head", original),
        ):
            payload.update({branch + "_" + k: v for k, v in result.items()})
        filename = f"objects_{n:03d}.npz"
        np.savez_compressed(directory / filename, **payload)
        saved.append({**row, "prediction_file": filename})
    metrics, matches = ap_metrics(
        saved, directory, ("original", "v3_frozen", "v4_independent_head")
    )
    result = dict(
        metrics=metrics,
        n_instructions=len(rows),
        n_frames=len(frames),
        n_scenes=len({r["scene_token"] for r in rows}),
        n_empty_prompts=sum(not r["relevant_gt_indices"] for r in rows),
        dedicated_object_conditioned_training="NOT_EVALUABLE: V4 map instance Mask Head has no object-conditioned detection training or path to original detector; measured AP is transfer/control only",
        definition=json.loads((OUT / "protocol.json").read_text())["detection"],
    )
    write(directory / "metrics.json", result)
    write(directory / "predictions.json", dict(annotations=saved))
    write(directory / "matches.json", matches)
    csv_write(
        directory / "summary.csv",
        [
            dict(
                branch=b,
                relevant_ap_2m_xy=m["relevant"]["macro_ap"],
                irrelevant_ap_2m_xy=m["irrelevant"]["macro_ap"],
            )
            for b, m in metrics.items()
        ],
    )
    return result


def main() -> None:
    """终点冻结后一次性打开新开发集，不据结果重训。"""
    argparse.ArgumentParser(description=__doc__).parse_args()
    model, wrapped, pipeline, dataset, manifest, features, text = load_runtime()
    training = json.loads((OUT / "training_summary.json").read_text())
    maps = evaluate_maps(model, pipeline, manifest, features, text, training)
    objects = evaluate_objects(model, pipeline, dataset, manifest, features, text)
    report = OUT / "report"
    report.mkdir()
    write(
        report / "metrics.json",
        dict(
            **maps,
            detection=objects,
            optimizer_updates=training["optimizer_updates"],
            final_generalization="BLOCKED",
            semantic_detection_conclusion="Independent new instance Head leaves original semantic/detection outputs unchanged. Grounding gains do not establish perception gains.",
        ),
    )


if __name__ == "__main__":
    main()

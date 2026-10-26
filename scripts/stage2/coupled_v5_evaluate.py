"""V5 固定开发协议：原生语义、真实对象AP和冻结实例读出分别评价。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.stage2.coupled_v5_data import BRANCHES, OUT, V4, sha, write
from scripts.stage2.coupled_v5_model import LanguageConditionedBEVResidualAdapter
from scripts.stage2.coupled_v5_train import position, runtime
from scripts.stage2.conditional_mask_v4_experiment import frame_inputs, load_label
from scripts.stage2.conditional_mask_v4_evaluate import csv_write, measure, pooled
from scripts.stage2.spatial_annotations import project_attention
from scripts.stage2.spatial_report import confusion, summary
from scripts.stage2.spatial_v2_experiment import swap_direction
from scripts.stage2.spatial_v3_evaluate import ap_metrics, segmentation
from scripts.stage2.spatial_v3_experiment import decode, head_forward
from scripts.stage2.stage3_metric_review import neighborhood

SYSTEMS = (
    "original",
    "v3_historical",
    "v4_independent",
    *BRANCHES,
    "targeted_language_wrong",
)


def adapters(phase: str) -> tuple[dict, dict]:
    """所有终点都报告，不用Mask或开发阈值选择模型。"""
    records = json.loads((OUT / f"{phase}_training_summary.json").read_text())[
        "branches"
    ]
    models = {}
    for branch, row in records.items():
        selected = row["selected_terminal"]
        assert sha(selected["checkpoint"]) == selected["sha256"]
        model = LanguageConditionedBEVResidualAdapter().cuda().eval()
        model.load_state_dict(
            torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)[
                "state_dict"
            ]
        )
        models[branch] = model
    return models, records


def enhanced(
    branch: str,
    bev: torch.Tensor,
    sentence: str,
    position_features: torch.Tensor,
    pipeline: Any,
    models: dict,
    text: dict,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """所有前向均不接受GT，地图与对象使用各自真实文本。"""
    if branch in ("original", "v4_independent"):
        return bev, None
    if branch == "v3_historical":
        detail = pipeline.fusion.forward_with_details(bev, text[sentence])
        return detail["enhanced_bev"], detail["attention"]
    name = "targeted_language" if branch == "targeted_language_wrong" else branch
    embedding = (
        text[sentence]
        if name == "targeted_language"
        else torch.zeros_like(text[sentence])
    )
    return models[name](bev, embedding, position_features)["enhanced_bev"], None


def maps(
    model: Any,
    pipeline: Any,
    mask_head: Any,
    manifest: dict,
    features: dict,
    text: dict,
    models: dict,
    folder: Path,
) -> dict:
    """相同独立GT/ROI下真正重新运行每个分支的原生语义Head。"""
    directory = folder / "maps"
    directory.mkdir()
    rows = [r for r in manifest["annotations"] if r["split"] == "dev"]
    semantics, masks, metadata, perturbations = [], [], [], []
    matrices = {
        b: {
            k: np.zeros((4, 4), dtype=np.int64)
            for k in ("irrelevant", "quadrant", "global")
        }
        for b in SYSTEMS
    }
    for n, row in enumerate(rows):
        label = {k: v.cpu().numpy() for k, v in load_label(row).items()}
        with np.load(row["source_file"]) as f:
            gt, all_masks, classes = f["gt"], f["masks"], f["classes"]
        region, target = label["region"], label["target"]
        roi = neighborhood(target) & region
        others = [
            m & region & ~target
            for i, m in enumerate(all_masks)
            if classes[i] == row["target_class"]
            and i != row["selected_index"]
            and (m & region & ~target).any()
        ]
        distractor_roi = (
            np.logical_or.reduce([neighborhood(m) & region for m in others])
            if others
            else np.zeros_like(region)
        )
        grid, ego_features = frame_inputs(row)
        pos = position(pipeline, row)
        bev = features[row["index"]]
        payload = dict(
            gt=gt,
            target=target,
            region=region,
            target_neighborhood=roi,
            other=label["quadrant_other"],
            distractor_neighborhood=distractor_roi,
        )
        for branch in SYSTEMS:
            sentence = (
                row["counterfactual_text"] or swap_direction(row["text"])
                if branch == "targeted_language_wrong"
                else row["text"]
            )
            with torch.no_grad():
                value, attention = enhanced(
                    branch, bev, sentence, pos, pipeline, models, text
                )
                prediction = (
                    segmentation(model.pts_bbox_head, value).argmax(1)[0].cpu().numpy()
                )
                payload[branch + "_semantic"] = prediction
                if attention is not None:
                    payload[branch + "_attention"] = (
                        project_attention(attention, grid)[0, 0].cpu().numpy()
                    )
                if branch not in ("original", "v3_historical"):
                    logits = (
                        mask_head(value, text[sentence], grid, ego_features)[0, 0]
                        .cpu()
                        .numpy()
                    )
                    payload[branch + "_mask_logits"] = logits
                    payload[branch + "_mask"] = logits >= 0
                    masks.append(
                        dict(
                            branch=branch,
                            instruction_id=row["instruction_id"],
                            sample_token=row["sample_token"],
                            instance_id=row["instance_id"],
                            paired=bool(row["counterfactual_text"]),
                            **measure(target, logits, label, gt, 0.5, roi),
                        )
                    )
                relative = float((value - bev).norm() / bev.norm())
            truth, active = gt == row["target_class"], prediction == row["target_class"]
            tc = dict(
                intersection=int((truth & active & roi).sum()),
                union=int(((truth | active) & roi).sum()),
                gt_pixels=int((truth & roi).sum()),
                predicted_pixels=int((active & roi).sum()),
                roi_pixels=int(roi.sum()),
            )
            dc = dict(
                distractor_intersection=int((truth & active & distractor_roi).sum()),
                distractor_union=int(((truth | active) & distractor_roi).sum()),
                distractor_roi_pixels=int(distractor_roi.sum()),
            )
            changed = int((prediction != payload["original_semantic"]).sum())
            semantics.append(
                dict(
                    branch=branch,
                    instruction_id=row["instruction_id"],
                    sample_token=row["sample_token"],
                    instance_id=row["instance_id"],
                    paired=bool(row["counterfactual_text"]),
                    **tc,
                    **dc,
                    outside_roi_semantic_fp=int((active & ~truth & ~roi).sum()),
                    semantic_changed_pixels_vs_original=changed,
                )
            )
            for key, valid in (
                ("irrelevant", ~region),
                ("quadrant", region),
                ("global", np.ones_like(region)),
            ):
                matrices[branch][key] += confusion(gt, prediction, valid)
            perturbations.append(
                dict(
                    branch=branch,
                    instruction_id=row["instruction_id"],
                    relative_bev_l2=relative,
                    semantic_changed_pixels_vs_original=changed,
                )
            )
        np.testing.assert_array_equal(
            payload["original_semantic"], payload["v4_independent_semantic"]
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
            print("V5_MAP_METRICS", n, len(rows), flush=True)
    result = {}
    for branch in SYSTEMS:
        records = [r for r in semantics if r["branch"] == branch]
        tp, union = sum(r["intersection"] for r in records), sum(
            r["union"] for r in records
        )
        dtp, du = sum(r["distractor_intersection"] for r in records), sum(
            r["distractor_union"] for r in records
        )
        mask_rows = [r for r in masks if r["branch"] == branch]
        result[branch] = dict(
            relevant_segmentation_iou=tp / union,
            intersection=tp,
            union=union,
            gt_pixels=sum(r["gt_pixels"] for r in records),
            predicted_pixels=sum(r["predicted_pixels"] for r in records),
            irrelevant_segmentation=summary(matrices[branch]["irrelevant"]),
            global_segmentation=summary(matrices[branch]["global"]),
            quadrant_segmentation=summary(matrices[branch]["quadrant"]),
            distractor_semantic_iou=dtp / du if du else None,
            distractor_intersection=dtp,
            distractor_union=du,
            outside_roi_semantic_fp=sum(r["outside_roi_semantic_fp"] for r in records),
            target_mask=(
                pooled(mask_rows)
                if mask_rows
                else dict(status="NOT_EVALUABLE: model has no V4 Mask readout")
            ),
            mean_relative_bev_l2=float(
                np.mean(
                    [
                        p["relative_bev_l2"]
                        for p in perturbations
                        if p["branch"] == branch
                    ]
                )
            ),
            semantic_changed_pixels=sum(
                r["semantic_changed_pixels_vs_original"] for r in records
            ),
        )
    paired_correct = pooled(
        [r for r in masks if r["branch"] == "targeted_language" and r["paired"]]
    )
    paired_wrong = pooled(
        [r for r in masks if r["branch"] == "targeted_language_wrong" and r["paired"]]
    )
    cf_semantic = {}
    for b in ("targeted_language", "targeted_language_wrong"):
        values = [r for r in semantics if r["branch"] == b and r["paired"]]
        tp, union = sum(r["intersection"] for r in values), sum(
            r["union"] for r in values
        )
        cf_semantic[b] = dict(intersection=tp, union=union, iou=tp / union)
    output = dict(
        metrics=result,
        n_instructions=len(rows),
        n_frames=len({r["index"] for r in rows}),
        n_scenes=1,
        mask_readout="same FROZEN V4 MaskHead on each enhancedBEV; not retrained or used in optimizer loss",
        counterfactual=dict(
            directed_pairs=42,
            correct_mask=paired_correct,
            wrong_mask=paired_wrong,
            mask_iou_gap=paired_correct["iou"] - paired_wrong["iou"],
            relevant_semantic=cf_semantic,
        ),
        distractor_definition="union of2m sameclass-other-real-record neighborhoods within quadrant per instruction; original semanticGT remains positive for all correct records, no per-record duplicate union",
    )
    write(directory / "metrics.json", output)
    write(directory / "predictions.json", dict(annotations=metadata))
    csv_write(directory / "semantic_per_instruction.csv", semantics)
    csv_write(directory / "mask_per_instruction.csv", masks)
    csv_write(directory / "feature_perturbations.csv", perturbations)
    plot(directory, metadata)
    return output


def plot(directory: Path, rows: list[dict]) -> None:
    """每开发帧首条及最差V5样本出图，不只展示成功。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        next(r for r in rows if r["index"] == index)
        for index in sorted({r["index"] for r in rows})
    ]
    for n, row in enumerate(selected):
        with np.load(directory / row["prediction_file"]) as f:
            a = {k: f[k] for k in f.files}
        panels = [
            ("Original map GT", a["gt"], "tab10", 3),
            ("Real selected instance", a["target"], "gray_r", 1),
            ("Sameclass distractors", a["other"], "gray_r", 1),
            ("V4 frozen Mask", a["v4_independent_mask"], "gray_r", 1),
            ("V5 frozen Mask readout", a["targeted_language_mask"], "gray_r", 1),
            ("Original native semantic", a["original_semantic"], "tab10", 3),
            ("V3 native semantic", a["v3_historical_semantic"], "tab10", 3),
            ("V5 target-no-language", a["targeted_no_language_semantic"], "tab10", 3),
            ("V5 target-language", a["targeted_language_semantic"], "tab10", 3),
            (
                "V5 minus Original error",
                (
                    (a["targeted_language_semantic"] != a["gt"]).astype(int)
                    - (a["original_semantic"] != a["gt"]).astype(int)
                ),
                "coolwarm",
                1,
            ),
        ]
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
        for ax, (title, value, cmap, hi) in zip(axes.flat, panels):
            ax.imshow(
                value,
                origin="lower",
                extent=(-30, 30, -15, 15),
                cmap=cmap,
                vmin=-1 if cmap == "coolwarm" else 0,
                vmax=hi,
                interpolation="nearest",
            )
            ax.set_title(title, fontsize=9)
        ckpt = json.loads(
            (
                directory.parent.parent
                / f"{directory.parent.name}_training_summary.json"
            ).read_text()
        )["branches"]["targeted_language"]["selected_terminal"]["sha256"]
        fig.suptitle(
            f"DEVELOPMENT sample={row['sample_token']} | {row['text']}\nV5={ckpt} | instance={row['instance_id']} | same GT/ROI for all branches",
            fontsize=8,
        )
        fig.savefig(directory / f"comparison_{n:02d}.png", dpi=120)
        plt.close(fig)


def objects(
    model: Any,
    pipeline: Any,
    dataset: Any,
    features: dict,
    text: dict,
    models: dict,
    folder: Path,
) -> dict:
    """真实对象指令与3DGT独立计算AP，不借用地图实例指标。"""
    directory = folder / "objects"
    directory.mkdir()
    rows = [
        r
        for r in json.loads((OUT / "objects_protocol.json").read_text())["annotations"]
        if r["split"] == "dev"
    ]
    saved = []
    for n, row in enumerate(rows):
        ann = dataset.get_ann_info(row["index"])
        payload = dict(
            gt_boxes=ann["gt_bboxes_3d"].tensor.numpy(), gt_labels=ann["gt_labels_3d"]
        )
        pos = position(pipeline, row)
        bev = features[row["index"]]
        for branch in SYSTEMS:
            sentence = (
                swap_direction(row["text"])
                if branch == "targeted_language_wrong"
                else row["text"]
            )
            with torch.no_grad():
                value, _ = enhanced(branch, bev, sentence, pos, pipeline, models, text)
                result = decode(model, head_forward(model, value))
            payload.update({branch + "_" + k: v for k, v in result.items()})
        for k in ("boxes", "scores", "labels"):
            np.testing.assert_array_equal(
                payload["original_" + k], payload["v4_independent_" + k]
            )
        filename = f"objects_{n:03d}.npz"
        np.savez_compressed(directory / filename, **payload)
        saved.append({**row, "prediction_file": filename})
        if n % 16 == 0:
            print("V5_REAL_OBJECT_AP", n, len(rows), flush=True)
    metrics, matches = ap_metrics(saved, directory, SYSTEMS)
    result = dict(
        metrics=metrics,
        n_instructions=len(rows),
        n_frames=len({r["index"] for r in rows}),
        n_scenes=1,
        empty_prompts=sum(not r["relevant_gt_indices"] for r in rows),
        definition=json.loads((OUT / "protocol.json").read_text())["metrics"][
            "detection"
        ],
        training="targeted_language/no_language each genuine object-instruction positive matching supervision on TRAIN only; raw AP GT protocol and all negative predictions retained",
    )
    write(directory / "metrics.json", result)
    write(directory / "predictions.json", dict(annotations=saved))
    write(directory / "matches.json", matches)
    csv_write(
        directory / "summary.csv",
        [
            dict(
                branch=b,
                relevant_ap=v["relevant"]["macro_ap"],
                irrelevant_ap=v["irrelevant"]["macro_ap"],
            )
            for b, v in metrics.items()
        ],
    )
    return result


def decide(metrics: dict, phase: str) -> dict:
    """门槛在训练前固定，失败则停止增加预算。"""
    m, o = metrics["maps"]["metrics"], metrics["objects"]["metrics"]
    branch = "targeted_language"
    checks = {}
    for control in ("global_control", "targeted_no_language"):
        checks[f"relevant_segmentation_vs_{control}"] = (
            m[branch]["relevant_segmentation_iou"]
            - m[control]["relevant_segmentation_iou"]
            >= 0.0001
        )
        checks[f"relevant_ap_vs_{control}"] = (
            o[branch]["relevant"]["macro_ap"] - o[control]["relevant"]["macro_ap"]
            >= 0.0001
        )
    checks["relevant_segmentation_vs_original"] = (
        m[branch]["relevant_segmentation_iou"]
        > m["original"]["relevant_segmentation_iou"]
    )
    checks["relevant_ap_vs_original"] = (
        o[branch]["relevant"]["macro_ap"] > o["original"]["relevant"]["macro_ap"]
    )
    checks["irrelevant_segmentation_protected"] = (
        m[branch]["irrelevant_segmentation"]["foreground_miou"]
        >= m["original"]["irrelevant_segmentation"]["foreground_miou"] - 0.001
    )
    checks["irrelevant_ap_protected"] = (
        o[branch]["irrelevant"]["macro_ap"]
        >= o["original"]["irrelevant"]["macro_ap"] - 0.001
    )
    passed = all(checks.values())
    return dict(
        phase=phase,
        checks=checks,
        expand=passed and phase == "pilot",
        useful_model=passed,
        decision=(
            "EXPAND_ALL3_TO256"
            if passed and phase == "pilot"
            else (
                "SEAL_USEFUL_TERMINAL"
                if passed
                else "STOP_ARCHITECTURE_EXPLORATION_AND_CLOSE"
            )
        ),
        no_new_threshold_or_validation_selection=True,
    )


def main(phase: str) -> None:
    """一次固定开发评价，记录可审阅的停止决定。"""
    model, wrapped, pipeline, dataset, manifest, features, text, mask_head = runtime()
    models, records = adapters(phase)
    folder = OUT / phase
    folder.mkdir()
    map_metrics = maps(
        model, pipeline, mask_head, manifest, features, text, models, folder
    )
    object_metrics = objects(model, pipeline, dataset, features, text, models, folder)
    result = dict(
        maps=map_metrics,
        objects=object_metrics,
        phase=phase,
        checkpoints=records,
        final_generalization="BLOCKED: development evidence only; old heldouts excluded",
    )
    write(folder / "metrics.json", result)
    decision = decide(result, phase)
    write(OUT / f"{phase}_decision.json", decision)
    print("V5_BUDGET_DECISION", json.dumps(decision), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "expand"), default="pilot")
    main(parser.parse_args().phase)

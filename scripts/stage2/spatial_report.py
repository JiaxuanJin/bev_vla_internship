"""从真实 GPU A/B 数组汇总指令指标、像素计数及可导出的对照图。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BRANCHES = ("original", "historical", "spatial", "incorrect", "shuffled")


def ratio(numerator, denominator):
    """无有效分母时返回不可评价。"""
    return float(numerator / denominator) if denominator else None


def confusion(gt, pred, mask):
    """汇总真实有效像素，GT 行、预测列。"""
    return np.bincount((4 * gt[mask].astype(int) + pred[mask]).ravel(),
                       minlength=16).reshape(4, 4)


def summary(matrix):
    """同时保存交集、GT 像素、预测像素和并集，绝不平均逐帧 IoU。"""
    classes = {}
    for cls in range(4):
        intersection = int(matrix[cls, cls])
        truth, prediction = int(matrix[cls].sum()), int(matrix[:, cls].sum())
        union = truth + prediction - intersection
        classes[str(cls)] = dict(intersection=intersection, union=union,
                                 gt_pixels=truth, predicted_pixels=prediction,
                                 iou=ratio(intersection, union))
    foreground = [classes[str(c)]["iou"] for c in (1, 2, 3)
                  if classes[str(c)]["iou"] is not None]
    return {"classes": classes, "foreground_miou": float(np.mean(foreground)) if foreground else None,
            "pixels": int(matrix.sum()), "confusion_gt_rows_pred_cols": matrix.tolist()}


def visualize(row, data, directory, metadata, number):
    """每条已冻结验证指令均出图，不筛选成功样本。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    colors = ListedColormap(["#eeeeee", "#e59c24", "#3c9ed6", "#7f62b5"])
    fig, axes = plt.subplots(2, 4, figsize=(18, 7), constrained_layout=True)
    gt, target, region = data["gt"], data["target"], data["region"]
    panels = [
        ("Original GT", gt, colors, 0, 3),
        ("Target instance (2) / instruction ROI (1)", region.astype(int) + target.astype(int), "viridis", 0, 2),
        ("Historical class Attention", data["historical_attention"], "magma", 0, 1),
        ("Spatial Attention", data["spatial_attention"], "magma", 0, 1),
        ("Other same-class map instances", data["other_instances"], "gray_r", 0, 1),
        ("Original segmentation", data["original"], colors, 0, 3),
        ("Spatial segmentation", data["spatial"], colors, 0, 3),
        ("ROI errors: spatial minus original", ((data["spatial"] != gt).astype(int) -
                                                (data["original"] != gt).astype(int)) * region,
         "coolwarm", -1, 1),
    ]
    for axis, (title, image, cmap, lower, upper) in zip(axes.flat, panels):
        axis.imshow(image, origin="lower", extent=(-30, 30, -15, 15),
                    cmap=cmap, vmin=lower, vmax=upper, interpolation="nearest")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Ego x (m), front positive")
        axis.set_ylabel("Ego y (m), left positive")
    fig.suptitle(f"sample={row['sample_token']} | {row['text']}\n"
                 f"author={metadata['author_sha256'][:12]}  historical={metadata['historical_sha256'][:12]}  "
                 f"spatial(step20)={metadata['spatial_sha256'][:12]}", fontsize=11)
    path = directory / f"comparison_{number:03d}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def direction_pairs(metadata, directory):
    """同帧同类别前左/前右的物理注意力重心对照。"""
    pairs = []
    for left in metadata["annotations"]:
        if not left["text"].endswith("front left"):
            continue
        right = next((r for r in metadata["annotations"]
                      if r["sample_token"] == left["sample_token"]
                      and r["target_class"] == left["target_class"]
                      and r["text"].endswith("front right")), None)
        if right is None:
            continue
        record = {"sample_token": left["sample_token"],
                  "left_instruction": left["text"], "right_instruction": right["text"],
                  "expected_centroid_delta_y": "negative (left to right)"}
        with np.load(directory / left["prediction_file"]) as a, np.load(directory / right["prediction_file"]) as b:
            yy = -15 + (np.arange(200) + .5) * .15
            for branch in ("historical", "spatial"):
                al, ar = a[branch + "_attention"].astype(float), b[branch + "_attention"].astype(float)
                cy_l = float((al * yy[:, None]).sum() / al.sum())
                cy_r = float((ar * yy[:, None]).sum() / ar.sum())
                record[branch] = {
                    "left_centroid_y_m": cy_l, "right_centroid_y_m": cy_r,
                    "delta_y_m": cy_r - cy_l, "direction_correct": bool(cy_r < cy_l),
                    "mean_abs_attention_change": float(np.abs(ar - al).mean()),
                    "own_target_left_mean": float(al[a["target"]].mean()),
                    "wrong_text_left_target_mean": float(ar[a["target"]].mean()),
                    "own_target_right_mean": float(ar[b["target"]].mean()),
                    "wrong_text_right_target_mean": float(al[b["target"]].mean()),
                }
        pairs.append(record)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(args.predictions.read_text())
    accumulators = {b: {"related": np.zeros((4, 4), dtype=np.int64),
                         "unrelated": np.zeros((4, 4), dtype=np.int64),
                         "target_class_intersection": 0, "target_class_union": 0,
                         "attention_intersection": 0, "attention_union": 0,
                         "attention_target_sum": 0., "attention_total_sum": 0.,
                         "other_instance_attention_sum": 0., "non_target_same_class_sum": 0.,
                         "outside_roi_attention_sum": 0., "outside_roi_activation_pixels": 0,
                         "outside_roi_foreground_fp": 0, "outside_roi_background_pixels": 0}
                    for b in BRANCHES}
    per_instruction, visualizations, directions = [], [], []
    for index, row in enumerate(metadata["annotations"]):
        with np.load(args.predictions.parent / row["prediction_file"]) as file:
            data = {k: file[k] for k in file.files}
        gt, target, region = data["gt"], data["target"], data["region"]
        for branch in BRANCHES:
            pred = data[branch]
            relevant = confusion(gt, pred, region)
            irrelevant = confusion(gt, pred, ~region)
            accumulator = accumulators[branch]
            accumulator["related"] += relevant
            accumulator["unrelated"] += irrelevant
            cls = row["target_class"]
            intersection = int(((gt == cls) & (pred == cls)).sum())
            union = int(((gt == cls) | (pred == cls)).sum())
            accumulator["target_class_intersection"] += intersection
            accumulator["target_class_union"] += union
            fp = int(((gt == 0) & (pred != 0) & ~region).sum())
            accumulator["outside_roi_foreground_fp"] += fp
            accumulator["outside_roi_background_pixels"] += int(((gt == 0) & ~region).sum())
            item = {"instruction_id": row["instruction_id"], "branch": branch,
                    "sample_token": row["sample_token"], "target_class": cls,
                    "target_gt_pixels": int(target.sum()), "roi_pixels": int(region.sum()),
                    "related_foreground_miou": summary(relevant)["foreground_miou"],
                    "irrelevant_foreground_miou": summary(irrelevant)["foreground_miou"],
                    "target_class_intersection": intersection, "target_class_union": union,
                    "outside_roi_foreground_fp": fp}
            if branch != "original":
                att = data[branch + "_attention"].astype(np.float64)
                activation = att >= .5
                tp, au = int((activation & target).sum()), int((activation | target).sum())
                accumulator["attention_intersection"] += tp
                accumulator["attention_union"] += au
                accumulator["attention_target_sum"] += float(att[target].sum())
                accumulator["attention_total_sum"] += float(att.sum())
                accumulator["other_instance_attention_sum"] += float(att[data["other_instances"]].sum())
                accumulator["non_target_same_class_sum"] += float(att[(gt == cls) & ~target].sum())
                accumulator["outside_roi_attention_sum"] += float(att[~region].sum())
                accumulator["outside_roi_activation_pixels"] += int((activation & ~region).sum())
                item.update(attention_intersection=tp, attention_union=au,
                            target_attention_mass=ratio(att[target].sum(), att.sum()),
                            target_mean_attention=float(att[target].mean()),
                            other_instance_attention_mass=ratio(att[data["other_instances"]].sum(), att.sum()))
            per_instruction.append(item)
        normal = data["spatial_attention"]
        wrong = data["incorrect_attention"]
        directions.append({"instruction_id": row["instruction_id"],
                           "correct_text_target_mean": float(normal[target].mean()),
                           "incorrect_text_target_mean": float(wrong[target].mean()),
                           "correct_text_has_more_target_attention": bool(normal[target].mean() > wrong[target].mean()),
                           "attention_mean_abs_change": float(np.abs(normal - wrong).mean()),
                           "note": "same BEV and class; swap left/right only; target fixed"})
        visualizations.append(visualize(row, data, args.output_dir, metadata, index))
    metrics = {}
    for branch, acc in accumulators.items():
        item = {"relevant_segmentation": summary(acc.pop("related")),
                "irrelevant_segmentation": summary(acc.pop("unrelated")), **acc}
        item["target_class_iou"] = ratio(item["target_class_intersection"], item["target_class_union"])
        item["attention_target_iou"] = ratio(item["attention_intersection"], item["attention_union"]) if branch != "original" else None
        for field, numerator in (("target_attention_mass", "attention_target_sum"),
                                  ("same_class_other_instance_mass", "other_instance_attention_sum"),
                                  ("same_class_non_target_mass", "non_target_same_class_sum"),
                                  ("outside_roi_attention_mass", "outside_roi_attention_sum")):
            item[field] = ratio(item[numerator], item["attention_total_sum"]) if branch != "original" else None
        item["outside_roi_foreground_fp_rate"] = ratio(item["outside_roi_foreground_fp"], item["outside_roi_background_pixels"])
        item["relevant_detection_ap"] = None
        metrics[branch] = item
    result = {"num_instructions": len(metadata["annotations"]),
              "num_distinct_frames": len({r["sample_token"] for r in metadata["annotations"]}),
              "num_scenes": len({r["scene_token"] for r in metadata["annotations"]}),
              "metrics": metrics, "per_instruction": per_instruction,
              "direction_controls": directions,
              "direction_correct_count": sum(r["correct_text_has_more_target_attention"] for r in directions),
              "detection": {"status": "NOT_EVALUABLE_FOR_MAP_INSTRUCTIONS",
                  "reason": "divider/crosswalk map records have no correspondence to 10-class 3D boxes; no map-object AP is invented",
                  "missing": ["object-language target schema and frozen instance matching protocol", "object-conditioned inference outputs on same protocol"]},
              "protocol": "pooled valid pixel confusion across all fixed instruction-frame pairs; macro IoU of present foreground classes; repeats are instruction-weighted, not independent frames; 0.5 attention threshold fixed before evaluation",
              "visualizations": visualizations, "checkpoint_metadata": metadata}
    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "direction_pairs.json").write_text(json.dumps(
        direction_pairs(metadata, args.predictions.parent), indent=2) + "\n")
    columns = sorted({k for row in per_instruction for k in row})
    with (args.output_dir / "per_instruction.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(per_instruction)
    table = []
    for label, getter in (
        ("Relevant Segmentation IoU", lambda m: m["relevant_segmentation"]["foreground_miou"]),
        ("Irrelevant Segmentation IoU", lambda m: m["irrelevant_segmentation"]["foreground_miou"]),
        ("Target Class IoU", lambda m: m["target_class_iou"]),
        ("Relevant Detection AP", lambda m: m["relevant_detection_ap"]),
        ("Attention Target IoU", lambda m: m["attention_target_iou"]),
        ("Target Attention Mass", lambda m: m["target_attention_mass"]),
        ("Same-Class Other-Instance Mass", lambda m: m["same_class_other_instance_mass"]),
        ("Same-Class Non-Target Mass", lambda m: m["same_class_non_target_mass"]),
        ("Outside-ROI Foreground FP Rate", lambda m: m["outside_roi_foreground_fp_rate"]),
    ):
        values = {b: getter(metrics[b]) for b in BRANCHES}
        original, spatial, historical = values["original"], values["spatial"], values["historical"]
        table.append({"metric": label, **values,
                      "spatial_minus_original_pp": (spatial - original) * 100 if spatial is not None and original is not None else None,
                      "spatial_minus_historical_pp": (spatial - historical) * 100 if spatial is not None and historical is not None else None,
                      "relative_change_vs_original_pct": (spatial / original - 1) * 100 if spatial is not None and original else None})
    with (args.output_dir / "ab_table.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    lines = ["| Metric | Original | Class | Spatial | Incorrect | Shuffled | Spatial-Original pp | Spatial-Class pp |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        values = [row[b] * 100 if row[b] is not None else None for b in BRANCHES]
        values += [row["spatial_minus_original_pp"], row["spatial_minus_historical_pp"]]
        lines.append("| " + row["metric"] + " | " + " | ".join("N/A" if v is None else f"{v:.8f}" for v in values) + " |")
    (args.output_dir / "ab_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

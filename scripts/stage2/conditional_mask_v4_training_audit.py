"""V4 训练侧根因诊断：校准、视觉依赖、数据重叠；不再更新或选择权重。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.stage2.conditional_mask_v4_data import OUT, write
from scripts.stage2.conditional_mask_v4_experiment import (
    frame_inputs,
    load_label,
    load_runtime,
)
from scripts.stage2.conditional_mask_v4_model import LanguageConditionedTargetMaskHead
from scripts.stage2.conditional_mask_v4_evaluate import csv_write, measure, pooled
from scripts.stage2.stage3_metric_review import neighborhood


def main() -> None:
    """对全部训练指令评价固定终点，视觉屏蔽仅作诊断，绝不据开发挑阈值。"""
    argparse.ArgumentParser(description=__doc__).parse_args()
    directory = OUT / "train_diagnosis"
    directory.mkdir()
    model, wrapped, pipeline, dataset, manifest, features, text = load_runtime()
    training = json.loads((OUT / "training_summary.json").read_text())
    chosen = training["branches"].get("full_instruction", training["diagnostic"])
    head = LanguageConditionedTargetMaskHead().cuda().eval()
    head.load_state_dict(
        torch.load(chosen["checkpoint"], map_location="cpu", weights_only=False)[
            "state_dict"
        ]
    )
    rows = [r for r in manifest["annotations"] if r["split"] == "train"]
    records, distributions = [], []
    for n, row in enumerate(rows):
        label = {k: v.cpu().numpy() for k, v in load_label(row).items()}
        with np.load(row["source_file"]) as f:
            gt = f["gt"]
        grid, geometry = frame_inputs(row)
        bev = features[row["index"]]
        roi = neighborhood(label["target"]) & label["region"]
        for branch in ("actual_bev", "zero_bev", "wrong_qualifier"):
            if branch == "wrong_qualifier" and not row["counterfactual_text"]:
                continue
            source = torch.zeros_like(bev) if branch == "zero_bev" else bev
            embedding = (
                text[row["counterfactual_text"]]
                if branch == "wrong_qualifier"
                else text[row["text"]]
            )
            with torch.no_grad():
                logits = head(source, embedding, grid, geometry)[0, 0].cpu().numpy()
            record = measure(label["target"], logits, label, gt, 0.5, roi)
            records.append(
                dict(
                    branch=branch,
                    instruction_id=row["instruction_id"],
                    paired=bool(row["counterfactual_text"]),
                    **record,
                )
            )
            if branch == "actual_bev":
                probability = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
                for name, mask in (
                    ("target", label["target"]),
                    ("same_class_negative", label["same_class_other"]),
                    ("background", gt == 0),
                ):
                    if mask.any():
                        values = probability[mask]
                        distributions.append(
                            dict(
                                instruction_id=row["instruction_id"],
                                group=name,
                                pixels=int(mask.sum()),
                                mean=float(values.mean()),
                                p10=float(np.percentile(values, 10)),
                                median=float(np.median(values)),
                                p90=float(np.percentile(values, 90)),
                            )
                        )
        if n % 64 == 0:
            print("V4_TRAIN_AUDIT", n, len(rows), flush=True)
    actual = pooled([r for r in records if r["branch"] == "actual_bev"])
    zero = pooled([r for r in records if r["branch"] == "zero_bev"])
    correct = pooled(
        [r for r in records if r["branch"] == "actual_bev" and r["paired"]]
    )
    wrong = pooled([r for r in records if r["branch"] == "wrong_qualifier"])
    result = dict(
        n_instructions=len(rows),
        n_frames=len({r["index"] for r in rows}),
        n_scenes=len({r["scene_token"] for r in rows}),
        actual_bev=actual,
        zero_bev=zero,
        same_quadrant_pair=dict(
            correct=correct, wrong=wrong, pooled_iou_gap=correct["iou"] - wrong["iou"]
        ),
        optimizer_updates=0,
        policy="training-side terminal diagnostics only; no threshold/architecture changes after observed dev; zeroBEV is off-distribution necessity probe, cannot prove sufficient visual grounding",
        decision="No additional tuning on old or newly opened development scenes. Conditional capacity shown by pair overfit; low train precision plus background activation documented as underfitting/objective/calibration bottleneck, no perception benefit claimed.",
    )
    write(directory / "metrics.json", result)
    csv_write(directory / "per_instruction.csv", records)
    csv_write(directory / "probability_distributions.csv", distributions)


if __name__ == "__main__":
    main()

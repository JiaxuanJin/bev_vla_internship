"""第二轮同留出集数值/图像报告，明确区分 V1 与 V2。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2 import spatial_report as shared


def visualize(row, data, directory, metadata, number):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors = ListedColormap(["#eeeeee", "#e59c24", "#3c9ed6", "#7f62b5"])
    fig, axes = plt.subplots(2, 5, figsize=(21, 8), constrained_layout=True)
    panels = [
        ("Original GT", data["gt"], colors, 0, 3),
        ("Target (2) / fixed ROI (1)", data["region"].astype(int) + data["target"].astype(int), "viridis", 0, 2),
        ("Other true same-class instances", data["other_instances"], "gray_r", 0, 1),
        ("Round1 Attention", data["historical_attention"], "magma", 0, 1),
        ("Round2 correct Attention", data["spatial_attention"], "magma", 0, 1),
        ("Original segmentation", data["original"], colors, 0, 3),
        ("Round1 segmentation", data["historical"], colors, 0, 3),
        ("Round2 segmentation", data["spatial"], colors, 0, 3),
        ("Round2 wrong-direction Attention", data["incorrect_attention"], "magma", 0, 1),
        ("ROI errors: Round2 minus Round1", ((data["spatial"] != data["gt"]).astype(int)
                                           - (data["historical"] != data["gt"]).astype(int)) * data["region"], "coolwarm", -1, 1),
    ]
    for ax, (title, value, cmap, lo, hi) in zip(axes.flat, panels):
        ax.imshow(value, origin="lower", extent=(-30, 30, -15, 15), cmap=cmap,
                  vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title, fontsize=10); ax.set_xlabel("Ego x (m)"); ax.set_ylabel("Ego y (m)")
    fig.suptitle(f"FRESH HOLDOUT sample={row['sample_token']} | {row['text']}\n"
                 f"round1={metadata['historical_sha256'][:12]} | round2(step{metadata['selected_step']})={metadata['spatial_sha256'][:12]}", fontsize=11)
    name = f"comparison_{number:03d}.png"
    fig.savefig(directory / name, dpi=140); plt.close(fig)
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.predictions = args.predictions.resolve()
    args.controls = args.controls.resolve()
    args.output_dir = args.output_dir.resolve()
    # 仅复用统计公式；旧脚本、旧数据和旧图不写入。
    shared.visualize = visualize
    sys.argv = [sys.argv[0], "--predictions", str(args.predictions), "--output-dir", str(args.output_dir)]
    shared.main()
    metrics = json.loads((args.output_dir / "metrics.json").read_text())
    metadata = json.loads(args.predictions.read_text())
    controls = {r["instruction_id"]: r for r in json.loads(args.controls.read_text())["records"]}
    gaps = {"round1": [], "round2": []}
    logit_gaps = {"round1": [], "round2": []}
    wins = {"round1": 0, "round2": 0}
    per_instruction = []
    for row in metadata["annotations"]:
        record = controls[row["instruction_id"]]
        with np.load(args.predictions.parent / row["prediction_file"]) as data, np.load(args.controls.parent / record["file"]) as control:
            target = data["target"]
            for branch, correct, wrong in (("round1", data["historical_attention"], control["v1_incorrect_attention"]),
                                           ("round2", data["spatial_attention"], data["incorrect_attention"])):
                gap = float((correct[target] - wrong[target]).mean())
                a, b = np.clip(correct[target], 1e-5, 1 - 1e-5), np.clip(wrong[target], 1e-5, 1 - 1e-5)
                logit_gap = float((np.log(a / (1 - a)) - np.log(b / (1 - b))).mean())
                gaps[branch].append(gap); logit_gaps[branch].append(logit_gap); wins[branch] += int(gap > 0)
                per_instruction.append({"instruction_id": row["instruction_id"], "model": branch,
                                        "target_probability_gap": gap, "target_logit_gap": logit_gap})
    metrics["round_roles"] = {"historical": "round1 spatial step20", "spatial": f"round2 contrast step{metadata['selected_step']}"}
    metrics["correct_wrong_comparison"] = {key: {"mean_target_probability_gap": float(np.mean(gaps[key])),
        "mean_target_logit_gap": float(np.mean(logit_gaps[key])), "wins": wins[key], "count": len(gaps[key])} for key in gaps}
    metrics["correct_wrong_per_instruction"] = per_instruction
    import csv
    with (args.output_dir / "correct_wrong.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_instruction[0]))
        writer.writeheader()
        writer.writerows(per_instruction)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    table = (args.output_dir / "ab_table.md").read_text().replace("| Class |", "| Round1 | ").replace("| Spatial |", "| Round2 | ")
    table = table.replace("Spatial-Class", "V2-V1").replace("Spatial-Original", "V2-Original")
    lines = table.splitlines()
    lines[0] = "| Metric | Original | Round1 | Round2 | Incorrect | Shuffled | V2-Original pp | V2-V1 pp |"
    table = "\n".join(lines) + "\n"
    (args.output_dir / "ab_table.md").write_text(table)
    print("CORRECT_WRONG", json.dumps(metrics["correct_wrong_comparison"]))


if __name__ == "__main__":
    main()

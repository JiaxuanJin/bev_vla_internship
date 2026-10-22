#!/usr/bin/env python3
"""从既有 JSONL/NPZ 离线生成交付图表；不导入感知模型、不推理或训练。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ("original", "trained", "shuffled")
LABELS = ("Original", "Trained", "Shuffled")
CLASSES = ("Background", "Divider", "Ped Crossing", "Boundary")
COLORS = ("#202936", "#f4bc41", "#49add3", "#ab77df")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def detection_counts(data, mode):
    boxes, labels, scores = (
        data[mode + "_" + k] for k in ("boxes", "labels", "scores")
    )
    truth, classes = data["gt_boxes"], data["gt_labels"]
    valid = (classes >= 0) & (np.abs(truth[:, :2]) <= 51.2).all(1)
    truth, classes = truth[valid], classes[valid]
    selected = np.flatnonzero((scores >= 0.1) & (np.abs(boxes[:, :2]) <= 51.2).all(1))
    unused = set(range(len(truth)))
    tp = 0
    for index in selected[np.argsort(-scores[selected], kind="stable")]:
        candidates = sorted(i for i in unused if classes[i] == labels[index])
        if candidates:
            distance = np.linalg.norm(truth[candidates, :2] - boxes[index, :2], axis=1)
            best = int(distance.argmin())
            if distance[best] <= 2:
                unused.remove(candidates[best])
                tp += 1
    return np.array([tp, len(selected) - tp, len(unused)], dtype=np.int64)


def save(fig, path):
    with path.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def generate(output):
    first = ROOT / "outputs/stage2_oom_fix_20261022_step1"
    continuation = ROOT / "outputs/stage2_oom_fix_20261022_continue/continue"
    validation = ROOT / "outputs/stage2_oom_fix_20261022_validation/validate"
    protocol_path = first / "protocol/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    assert protocol["loss"].get("attention_tv_weight", 0) == 0
    logs = (first / "train/training.jsonl", continuation / "training.jsonl")
    rows = [json.loads(line) for path in logs for line in path.read_text().splitlines()]
    assert [row["step"] for row in rows] == list(range(1, 21))
    assert all(row["optimizer_step_executed"] for row in rows)
    assert [r["token"] for r in rows] == [r["token"] for r in protocol["train"]]
    for row in rows:
        loss = row["losses"]
        assert all(np.isfinite(v) for v in loss.values())
        assert np.isclose(
            loss["total_loss"],
            loss["detection_loss"]
            + loss["segmentation_loss"]
            + 0.1 * loss["attention_loss"],
            atol=1e-6,
        )
    ab_path = validation / "ab_results.json"
    ab = json.loads(ab_path.read_text())
    assert [f["token"] for f in ab["frames"]] == [
        f["token"] for f in protocol["validation"]
    ]
    matrices = {mode: np.zeros((4, 4), dtype=np.int64) for mode in MODES}
    counts = {mode: np.zeros(3, dtype=np.int64) for mode in MODES}
    frames = []
    sources = {
        str(p.relative_to(ROOT)): digest(p) for p in (protocol_path, *logs, ab_path)
    }
    for record in ab["frames"]:
        path = validation / record["artifact"]
        assert digest(path) == record["sha256"]
        sources[str(path.relative_to(ROOT))] = record["sha256"]
        with np.load(path, allow_pickle=False) as data:
            gt = data["gt"]
            assert gt.shape == (200, 400)
            assert (
                hashlib.sha256(gt.astype(np.int64).tobytes()).hexdigest()
                == record["gt_sha256"]
            )
            assert np.array_equal(data["original_logits"], data["untrained_logits"])
            frame = {"index": record["index"], "text": record["text"], "gt": gt.copy()}
            for mode in MODES:
                assert np.array_equal(data[mode], data[mode + "_logits"].argmax(0))
                matrices[mode] += np.bincount(
                    4 * gt.astype(np.int64).ravel()
                    + data[mode].astype(np.int64).ravel(),
                    minlength=16,
                ).reshape(4, 4)
                counts[mode] += detection_counts(data, mode)
                frame[mode] = data[mode].copy()
            frame["trained_attention"] = data["trained_attention"].copy()
            frame["shuffled_attention"] = data["shuffled_attention"].copy()
            frames.append(frame)
    metrics = {}
    for mode, matrix in matrices.items():
        tp = matrix.diagonal().astype(float)
        union = matrix.sum(0) + matrix.sum(1) - tp
        iou = np.divide(tp, union, out=np.full(4, np.nan), where=union > 0)
        metrics[mode] = {
            "foreground_miou": float(np.nanmean(iou[1:])),
            "class_iou": iou.tolist(),
            "background_false_positive_rate": float(
                matrix[0, 1:].sum() / matrix[0].sum()
            ),
            "tp_fp_fn": counts[mode].tolist(),
        }
        expected = ab["metrics"][mode]
        assert np.isclose(
            metrics[mode]["foreground_miou"],
            expected["segmentation"]["foreground_miou"],
            atol=1e-14,
            rtol=0,
        )
        assert counts[mode].tolist() == [
            expected["detection"][k] for k in ("tp", "fp", "fn")
        ]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "training_20steps.csv").open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "step",
                "token",
                "text",
                "detection_loss",
                "segmentation_loss",
                "attention_supervision_loss",
                "total_loss",
                "tv_weight",
            ]
        )
        for row in rows:
            loss = row["losses"]
            writer.writerow(
                [
                    row["step"],
                    row["token"],
                    row["text"],
                    loss["detection_loss"],
                    loss["segmentation_loss"],
                    loss["attention_loss"],
                    loss["total_loss"],
                    0,
                ]
            )
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for ax, key, label in zip(
        axes.flat,
        ("total_loss", "detection_loss", "segmentation_loss", "attention_loss"),
        (
            "Total",
            "Detection",
            "Segmentation",
            "Attention supervision (BCE + 0.25 Dice)",
        ),
    ):
        ax.plot(range(1, 21), [r["losses"][key] for r in rows], "o-", markersize=3)
        ax.set(
            title=label,
            xlabel="Actual optimizer step",
            ylabel="Raw loss",
            xticks=[1, 5, 10, 15, 20],
        )
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Real 20-step training | fixed samples | TV=0 | no smoothing", fontsize=12
    )
    fig.text(
        0.5,
        0.005,
        "Different frame/text at each step. These curves do not establish convergence or validation gains.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.95))
    save(fig, output / "training_loss_20steps.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    x = np.arange(3)
    for n, mode in enumerate(MODES):
        axes[0].bar(
            x + (n - 1) * 0.25,
            metrics[mode]["class_iou"][1:],
            width=0.25,
            label=LABELS[n],
        )
    axes[0].set(
        xticks=x, xticklabels=CLASSES[1:], ylabel="Pooled class IoU", ylim=(0, 0.42)
    )
    axes[0].legend(fontsize=8)
    axes[1].bar(
        LABELS,
        [metrics[m]["foreground_miou"] for m in MODES],
        color=["#64748b", "#0284c7", "#d97706"],
    )
    axes[1].set(ylabel="Foreground mIoU", ylim=(0, 0.4))
    for i, m in enumerate(MODES):
        axes[1].text(
            i,
            metrics[m]["foreground_miou"] + 0.008,
            f"{metrics[m]['foreground_miou']:.10f}",
            ha="center",
            fontsize=8,
        )
    axes[2].bar(
        LABELS,
        [
            1e5
            * (metrics[m]["foreground_miou"] - metrics["original"]["foreground_miou"])
            for m in MODES
        ],
    )
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[2].set(
        ylabel="mIoU delta vs Original (units of 1e-5)",
        title="Tiny changes; no improvement shown",
    )
    fig.suptitle(
        "Same fixed Mini Val 6 frames | TV=0 trained weights | TP/FP/FN = 207/599/35 in all modes",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, output / "ab_metrics_6frames.png")
    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(6, 4, figsize=(14, 12))
    for n, frame in enumerate(frames):
        for col, key in enumerate(("gt", *MODES)):
            ax = axes[n, col]
            ax.imshow(frame[key], cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if n == 0:
                ax.set_title(("Author raster GT", *LABELS)[col])
            if col == 0:
                ax.set_ylabel(f"index {frame['index']}\n{frame['text']}", fontsize=8)
    fig.suptitle(
        "Fixed 6 frames: GT / Original / Trained / Shuffled | native raster order; physical alignment NOT_VERIFIED",
        fontsize=10,
    )
    fig.legend(
        handles=[Patch(color=c, label=label) for c, label in zip(COLORS, CLASSES)],
        loc="lower center",
        ncol=4,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    save(fig, output / "ab_segmentation_6frames.png")
    max_diff = max(
        float(np.abs(f["trained_attention"] - f["shuffled_attention"]).max())
        for f in frames
    )
    fig, axes = plt.subplots(6, 3, figsize=(10, 14))
    for n, frame in enumerate(frames):
        for col, key in enumerate(("trained_attention", "shuffled_attention")):
            im = axes[n, col].imshow(frame[key], vmin=0, vmax=1, cmap="viridis")
            axes[n, col].set_xticks([])
            axes[n, col].set_yticks([])
        diff = np.abs(frame["trained_attention"] - frame["shuffled_attention"])
        delta = axes[n, 2].imshow(diff, vmin=0, vmax=max(max_diff, 1e-8), cmap="magma")
        axes[n, 2].set(xticks=[], yticks=[], xlabel=f"MAE={float(diff.mean()):.6f}")
        axes[n, 0].set_ylabel(f"index {frame['index']}")
        if n == 0:
            for col, label in enumerate(
                (
                    "Trained normal text",
                    "Same words reversed",
                    "Absolute attention difference",
                )
            ):
                axes[n, col].set_title(label, fontsize=9)
    fig.suptitle(
        "Existing 6-frame attention | response difference is not correct semantic localization",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.06, right=0.97, top=0.95, bottom=0.09, hspace=0.33, wspace=0.28
    )
    first_bar = fig.add_axes([0.10, 0.035, 0.47, 0.012])
    second_bar = fig.add_axes([0.70, 0.035, 0.22, 0.012])
    fig.colorbar(
        im, cax=first_bar, orientation="horizontal", label="Attention (fixed 0-1)"
    )
    fig.colorbar(
        delta,
        cax=second_bar,
        orientation="horizontal",
        label="Difference (shared scale)",
    )
    save(fig, output / "attention_control_6frames.png")
    provenance = {
        "status": "PASS_OFFLINE_FIGURES_FROM_EXISTING_ARRAYS",
        "inference_calls": 0,
        "optimizer_steps": 0,
        "historical_training_steps": 20,
        "historical_tv_weight": 0,
        "new_tv_smoke_included": False,
        "sources_sha256": sources,
        "protocol_sha256": digest(protocol_path),
        "metrics": metrics,
        "frames": [
            {
                "index": f["index"],
                "text": f["text"],
                "changed_pixels_trained_vs_original": int(
                    (f["trained"] != f["original"]).sum()
                ),
                "changed_pixels_shuffled_vs_trained": int(
                    (f["shuffled"] != f["trained"]).sum()
                ),
            }
            for f in frames
        ],
        "figures": {
            p.name: {"sha256": digest(p), "size_bytes": p.stat().st_size}
            for p in sorted(output.glob("*.png"))
        },
        "limits": [
            "No stable accuracy gain established",
            "Semantic localization NOT_VERIFIED",
            "Physical alignment NOT_VERIFIED",
        ],
    }
    with (output / "figure_provenance.json").open("x") as stream:
        json.dump(provenance, stream, ensure_ascii=False, indent=2)
    print(
        json.dumps(
            {
                "status": provenance["status"],
                "figures": list(provenance["figures"]),
                "optimizer_steps": 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    generate(parser.parse_args().output_dir.resolve())

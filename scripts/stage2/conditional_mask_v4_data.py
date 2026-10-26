"""V4：只依赖真实地图几何的冻结划分、标签与历史完整性清单。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from scripts.stage2.spatial_v3_data import V2, OUT as V3, sha, write

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(
    os.environ.get(
        "BEV_V4_OUTPUT", ROOT / "outputs/stage3_conditional_mask_v4_20261026"
    )
).resolve()
SEED = 20261030
HISTORY = [
    ROOT / "outputs" / name
    for name in (
        "stage2_spatial_20261026",
        "stage2_spatial_v2_20261026",
        "stage2_spatial_v3_20261026",
        "stage3_metric_review_20261026",
    )
]


def region_mask(xy: np.ndarray, fb: str, lr: str) -> np.ndarray:
    """按照 Ego 像素中心构造评估方位，仅供标注和评估使用。

    Args:
        xy: Ego 米制坐标，形状 H×W×2。
        fb: front 或 rear。
        lr: left 或 right。

    Returns:
        严格正负半平面交集。
    """
    return ((xy[..., 0] > 0) if fb == "front" else (xy[..., 0] < 0)) & (
        (xy[..., 1] > 0) if lr == "left" else (xy[..., 1] < 0)
    )


def inventory(directories: list[Path]) -> list[dict]:
    """记录文件内容，完成时以同一列表逐一验证。"""
    return [
        dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
        for d in directories
        for p in sorted(d.rglob("*"))
        if p.is_file() and not p.is_symlink()
    ]


def prepare() -> None:
    """先冻结协议，再读取预测；不访问旧开发或留出场景标签。"""
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / "frozen_history.json", dict(files=inventory(HISTORY)))
    (OUT / "initial_git_status.txt").write_text(
        subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    )
    manifest = json.loads((V3 / "annotations/manifest.json").read_text())
    old_frames = [f for f in manifest["frames"] if f["split"] == "train"]
    scenes = sorted({f["scene_name"] for f in old_frames})
    dev_scene = str(np.random.RandomState(SEED).permutation(scenes)[0])
    protocol = dict(
        seed=SEED,
        dev_scene=dev_scene,
        train_scenes=[s for s in scenes if s != dev_scene],
        forbidden_scenes=["scene-0061", "scene-0553", "scene-0001"],
        prior_exposure="These six scenes trained historical adapters. V4 Head starts from scratch; inputs are original frozen author BEV, not trained adapter BEV/Attention. New dev is V4-Head-unseen only; no claim of foundation-model-unseen data.",
        final_generalization=dict(
            status="BLOCKED",
            reason="No traceable scene independent of author pretraining and all historical evaluation. No new final opened; old final/dev scenes excluded from V4 execution.",
        ),
        geometry="150x150 native LiDAR XY centers within +/-51.2m; calibration Ego z=0 -> LiDAR. GT200x400 Ego x[-30,30],y[-15,15],pixel0.15m. grid_sample bilinear align_corners=False; no legacy cropper modifications.",
        instance_selection="ALL visible record masks >=1px participate in ordering by minimum visible Ego radial distance. Adjacent-order gap>=1m. Selected targets <8px rejected after ordering, never excluded before ordering. Unique means exactly one visible record.",
        overlap="Target pixels win; overlapping same-class record pixels excluded only from Mask distractor negatives. All same-class positives retained in global semantic GT.",
        loss="0.5 positive BCE +0.25 negative BCE +0.25 same-class non-target BCE +0.25 Dice +0.1 topK negative BCE (K=max32,4*positive pixels); positive target empty/invalid => reject; absent-object instructions NOT_EVALUABLE, not trained.",
        training=dict(
            diagnostic_updates=256,
            diagnostic_batch="first metadata-ordered train nearest/farthest pair",
            diagnostic_success="pooled train-pair IoU>=0.25 AND correct-minus-wrong paired IoU>0.05 at fixed threshold0.5",
            steps_per_branch=384,
            branches=["no_language", "shuffled_text", "class_only", "full_instruction"],
            batch_size=2,
            learning_rate=0.001,
            weight_decay=0.01,
            gradient_clip=5.0,
            initialization="identical scratch Head; frozen author/BERT; original BEV; no adapter Attention input to avoid language leakage into no-language control",
            shuffle="fresh deterministic train-text permutation each step independent of target; evaluation full text, plus wrong and word-order shuffled controls",
            selection="terminal step384 for all branches; no early stopping or dev threshold/model selection",
        ),
        thresholds=[0.2, 0.35, 0.5, 0.65, 0.8],
        primary_threshold=0.5,
        semantic="2m Euclidean target neighborhood intersected with quadrant; target-class binary IoU with ALL correct same-class GT; irrelevant foreground mIoU outside quadrant; global native Head GT unchanged",
        detection="object-instruction transfer diagnostic only; no V4 object-conditioned training. Noninterpolated AP@2m_xy with all native top300 scores, real car/pedestrian boxes and IDs, quadrant partitions; NOT nuScenes official AP",
        source_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / "scripts/stage2").glob("conditional_mask_v4*.py"))
        },
    )
    write(OUT / "protocol.json", protocol)
    directory = OUT / "labels"
    directory.mkdir()
    rows, frames, rejected, audits = [], [], [], []
    cache = {
        f["index"]: f
        for f in json.loads((V2 / "cache/cache.json").read_text())["frames"]
    }
    for old in old_frames:
        f = dict(old, split="dev" if old["scene_name"] == dev_scene else "train")
        source = V3 / "annotations" / f["frame_file"]
        with np.load(source) as a:
            gt, masks, classes, xy, grid = (
                a[k] for k in ("gt", "masks", "classes", "ego_xy", "sampling_grid")
            )
        assert masks.shape[0] == len(f["instance_ids"])
        for k, cls in enumerate(classes):
            assert not (masks[k] & (gt != cls)).any()
        cf = cache[f["index"]]
        cache_path = V2 / "cache" / cf["cache_file"]
        assert sha(cache_path) == cf["sha256"]
        f.update(
            source_file=str(source),
            source_sha256=sha(source),
            cache_file=str(cache_path),
            cache_sha256=sha(cache_path),
        )
        frames.append(f)
        for cls, name in ((1, "lane markings"), (2, "crosswalk")):
            for fb, lr in (
                ("front", "left"),
                ("front", "right"),
                ("rear", "left"),
                ("rear", "right"),
            ):
                region = region_mask(xy, fb, lr)
                candidates = [
                    k
                    for k, c in enumerate(classes)
                    if c == cls and (masks[k] & region).any()
                ]
                ordered = sorted(
                    (float(np.linalg.norm(xy[masks[k] & region], axis=-1).min()), k)
                    for k in candidates
                )
                selections = []
                if len(ordered) == 1:
                    selections = [("unique", ordered[0])]
                elif len(ordered) > 1:
                    if ordered[1][0] - ordered[0][0] >= 1.0:
                        selections.append(("nearest", ordered[0]))
                    if ordered[-1][0] - ordered[-2][0] >= 1.0:
                        selections.append(("farthest", ordered[-1]))
                eligible = []
                for qualifier, (distance, k) in selections:
                    pixels = int((masks[k] & region).sum())
                    if pixels < 8:
                        rejected.append(
                            dict(
                                sample_token=f["sample_token"],
                                qualifier=qualifier,
                                instance_id=f["instance_ids"][k],
                                pixels=pixels,
                                reason="true selected instance too small; no alternate instance substitution",
                            )
                        )
                    else:
                        eligible.append((qualifier, distance, k))
                for qualifier, distance, k in eligible:
                    target = masks[k] & region
                    same = masks[classes == cls].any(0) & ~target
                    others = [masks[j] & region & ~target for j in candidates if j != k]
                    quadrant_other = (
                        np.logical_or.reduce(others)
                        if others
                        else np.zeros_like(target)
                    )
                    prefix = "" if qualifier == "unique" else qualifier + " "
                    text = f"Find the {prefix}{name} on the {fb} {lr}"
                    opposite = "farthest" if qualifier == "nearest" else "nearest"
                    paired = len(eligible) == 2
                    instruction_id = f"{f['sample_token']}_c{cls}_{fb}_{lr}_{qualifier}"
                    filename = instruction_id + ".npz"
                    np.savez_compressed(
                        directory / filename,
                        target=target,
                        same_class_other=same,
                        quadrant_other=quadrant_other,
                        region=region,
                    )
                    rows.append(
                        {
                            **f,
                            "instruction_id": instruction_id,
                            "text": text,
                            "target_class": cls,
                            "qualifier": qualifier,
                            "fb": fb,
                            "lr": lr,
                            "instance_id": f["instance_ids"][k],
                            "selected_index": k,
                            "distance_m": distance,
                            "label_file": filename,
                            "label_sha256": sha(directory / filename),
                            "counterfactual_text": (
                                f"Find the {opposite} {name} on the {fb} {lr}"
                                if paired
                                else None
                            ),
                            "target_pixels": int(target.sum()),
                            "visible_candidate_ids": [
                                f["instance_ids"][j] for _, j in ordered
                            ],
                        }
                    )
                    audits.append(
                        dict(
                            instruction_id=instruction_id,
                            n_all_visible=len(ordered),
                            min_selected_pixels=int(target.sum()),
                            same_class_overlap_pixels=sum(
                                int((target & masks[j]).sum())
                                for j in candidates
                                if j != k
                            ),
                            order=[
                                dict(distance_m=d, instance_id=f["instance_ids"][j])
                                for d, j in ordered
                            ],
                        )
                    )
    train_ids = {r["instance_id"] for r in rows if r["split"] == "train"}
    dev_ids = {r["instance_id"] for r in rows if r["split"] == "dev"}
    write(
        directory / "manifest.json",
        dict(
            frames=frames,
            annotations=rows,
            counts=dict(Counter(r["split"] for r in rows)),
            paired_counts=dict(
                Counter(r["split"] for r in rows if r["counterfactual_text"])
            ),
            shared_record_ids=sorted(train_ids & dev_ids),
            split_method="scene-level, no frame overlap; shared real map records explicitly audited",
        ),
    )
    write(directory / "audit.json", dict(records=audits, rejected=rejected))
    print(
        "V4_FROZEN_SPLIT",
        dev_scene,
        Counter(r["split"] for r in rows),
        "paired",
        Counter(r["split"] for r in rows if r["counterfactual_text"]),
        flush=True,
    )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()

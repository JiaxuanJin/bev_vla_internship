"""第二轮按场景隔离标注：不读取第一轮已观察的验证预测。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2.spatial_annotations import (
    AUTHOR, DATA, digest_array, load_module, metric_grid, tagged_vectors,
)


def main() -> None:
    """先以场景定义训练/开发/最终集，再按几何拒绝歧义指令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nearest", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    first = json.loads((ROOT / "outputs/stage2_spatial_20261026/protocol/protocol.json").read_text())
    first_scenes = {r["scene_token"] for r in first["train"]}
    with (DATA / "nuscenes_infos_temporal_train.pkl").open("rb") as stream:
        infos = sorted(pickle.load(stream)["infos"], key=lambda r: r["timestamp"])
    scenes = {r["token"]: r for r in json.loads((DATA / "v1.0-mini/scene.json").read_text())}
    logs = {r["token"]: r for r in json.loads((DATA / "v1.0-mini/log.json").read_text())}
    all_scenes = sorted({r["scene_token"] for r in infos}, key=lambda s: scenes[s]["name"])
    fresh = [s for s in all_scenes if s not in first_scenes]
    if len(fresh) != 1:
        raise ValueError("需重新审查独立留出场景，不能猜测")
    final_scene = fresh[0]
    dev_scene = next(s for s in all_scenes if s != final_scene)
    split_scenes = {"final": [final_scene], "dev": [dev_scene],
                    "train": [s for s in all_scenes if s not in (final_scene, dev_scene)]}
    tools = AUTHOR / "projects/mmdet3d_plugin/datasets/utils"
    vector = load_module("v2_author_vector", tools / "vector_map.py")
    raster = load_module("v2_author_raster", tools / "rasterize.py")
    mapper = vector.VectorizedLocalMap(str(DATA), (30., 60.), (200, 400))
    annotations, rejected, frames = [], [], []
    for split in ("train", "dev", "final"):
        for scene in split_scenes[split]:
            indices = [i for i, info in enumerate(infos) if info["scene_token"] == scene]
            chosen = indices if split == "final" else [indices[i] for i in np.linspace(0, len(indices) - 1, 8, dtype=int)]
            for index in chosen:
                info = infos[index]
                location = logs[scenes[scene]["log_token"]]["location"]
                vectors, tagged = tagged_vectors(mapper, info, location)
                semantic, *_ = raster.preprocess_map(vectors, (30., 60.), (200, 400), 3, 5, 36)
                gt = (semantic.numpy() * np.arange(1, 4)[:, None, None]).sum(0).astype(np.uint8)
                groups = {}
                for v, identity in tagged:
                    groups.setdefault(identity, []).append(v)
                masks = {}
                for identity, vv in groups.items():
                    single, *_ = raster.preprocess_map(vv, (30., 60.), (200, 400), 3, 5, 36)
                    cls = vv[0]["type"] + 1
                    masks[identity] = (cls, single[cls - 1].numpy() & (gt == cls))
                sampling, ego, _ = metric_grid(info)
                provenance = {"sample_token": info["token"], "scene_token": scene,
                    "scene_name": scenes[scene]["name"], "index": index, "split": split,
                    "location": location, "lidar2ego_rotation": info["lidar2ego_rotation"],
                    "lidar2ego_translation": info["lidar2ego_translation"],
                    "ego2global_rotation": info["ego2global_rotation"],
                    "ego2global_translation": info["ego2global_translation"]}
                frame_file = f"frame_{index:03d}.npz"
                np.savez_compressed(output / frame_file, gt=gt, sampling_grid=sampling.numpy())
                frames.append({**provenance, "frame_file": frame_file})
                for cls, name in ((1, "lane markings"), (2, "crosswalk")):
                    for fb, lr in (("front", "left"), ("front", "right"), ("rear", "left"), ("rear", "right")):
                        region = (ego[..., 0] > 0) if fb == "front" else (ego[..., 0] < 0)
                        region &= (ego[..., 1] > 0) if lr == "left" else (ego[..., 1] < 0)
                        ids = [k for k, (c, mask) in masks.items() if c == cls and (mask & region).any()]
                        key = f"{info['token']}_c{cls}_{fb}_{lr}"
                        row = {**provenance, "instruction_id": key, "target_class": cls,
                               "text": f"Find {name} on the {fb} {lr}", "candidate_instance_ids": ids}
                        nearest = None
                        if len(ids) > 1 and args.allow_nearest:
                            distances = sorted((float(np.linalg.norm(ego[..., :2][masks[k][1] & region], axis=-1).min()), k) for k in ids)
                            gap = distances[1][0] - distances[0][0]
                            if gap >= 1.0:
                                nearest = distances[0][1]
                                row["text"] = f"Find the nearest {name} on the {fb} {lr}"
                                row["instance_selection"] = "explicit nearest visible GT-raster point in Ego quadrant; gap >=1m"
                                row["nearest_distance_m"] = distances[0][0]
                                row["next_nearest_distance_m"] = distances[1][0]
                                row["nearest_gap_m"] = gap
                        if nearest is not None:
                            ids = [nearest]
                        if len(ids) != 1:
                            rejected.append({**row, "reason": "AMBIGUOUS" if ids else "ABSENT"})
                            continue
                        identity = ids[0]
                        target = masks[identity][1] & region
                        if target.sum() < 8:
                            rejected.append({**row, "reason": "TOO_SMALL"})
                            continue
                        other = np.zeros_like(target)
                        for other_id, (c, mask) in masks.items():
                            if other_id != identity and c == cls:
                                other |= mask
                        other &= ~masks[identity][1]
                        filename = key + ".npz"
                        np.savez_compressed(output / filename, gt=gt, target=target,
                            selected_full_instance=masks[identity][1], other_instances=other,
                            region=region, sampling_grid=sampling.numpy())
                        annotations.append({**row, "mask_file": filename, "instance_id": identity,
                            "annotation_source": "nuscenes_map_record_exact_author_raster",
                            "gt_sha256": digest_array(gt), "target_sha256": digest_array(target),
                            "target_pixels": int(target.sum()), "other_instance_pixels": int(other.sum())})
                print("ANNOTATED_V2", split, index, flush=True)
    pairs = []
    for left in annotations:
        partners = [r for r in annotations if r["sample_token"] == left["sample_token"]
                    and r["target_class"] == left["target_class"]
                    and r["instruction_id"] != left["instruction_id"]]
        for right in partners:
            if left["instruction_id"] < right["instruction_id"]:
                pairs.append({"split": left["split"], "first": left["instruction_id"], "second": right["instruction_id"]})
    manifest = {"seed": 20261027, "split_scenes": split_scenes,
        "scene_names": {s: scenes[s]["name"] for s in all_scenes},
        "first_round_train_scenes": sorted(first_scenes),
        "final_scene_never_used_in_round1_optimizer": final_scene not in first_scenes,
        "selection": "scene-0553 untouched by round1 optimizer; first other scene dev; 8 linspace frames per train/dev scene; all41 final frames; unique direction or explicit nearest with >=1m gap; no predictions read",
        "annotations": annotations, "rejected": rejected, "frames": frames, "pairs": pairs,
        "counts": dict(Counter(r["split"] for r in annotations)),
        "pair_counts": dict(Counter(r["split"] for r in pairs)),
        "rejection_counts": dict(Counter(r["reason"] for r in rejected))}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("counts", "pair_counts", "rejection_counts", "scene_names")}, indent=2))


if __name__ == "__main__":
    main()

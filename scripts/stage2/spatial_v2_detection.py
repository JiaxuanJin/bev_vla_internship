"""在第二轮新留出场景运行对象语言 AP；复用已审查的非官方 AP@2m。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2.micro_training import forward_payload, get_batch
from scripts.stage2.pipeline import Stage2Pipeline
from scripts.stage2.run_step70_joint_integration import sha256
from scripts.stage2.runtime_memory import release_memory
from scripts.stage2.spatial_annotations import DATA
from scripts.stage2.spatial_detection import ap_metrics, ego_centers, in_region, parse_object_instruction
from scripts.stage2.spatial_experiment import write_json
from scripts.stage2.spatial_v2_experiment import V1, configure_v2, final_dataset, swap_direction


def prepare(args):
    from pyquaternion import Quaternion
    metadata = json.loads(args.predictions.read_text())
    source_rows = {}
    for row in metadata["annotations"]:
        source_rows.setdefault(row["index"], row)
    raw = json.loads((DATA / "v1.0-mini/sample_annotation.json").read_text())
    rows = []
    for index, info in sorted(source_rows.items()):
        source = args.predictions.parent / info["prediction_file"]
        with np.load(source) as cached:
            boxes, labels = cached["gt_boxes"], cached["gt_labels"]
        ego = ego_centers(boxes, info)
        world = ego @ Quaternion(info["ego2global_rotation"]).rotation_matrix.T + np.asarray(info["ego2global_translation"])
        annotations = [r for r in raw if r["sample_token"] == info["sample_token"]]
        distances = np.linalg.norm(world[:, None] - np.asarray([a["translation"] for a in annotations])[None], axis=-1)
        nearest = distances.argmin(1)
        assert distances.min(1).max() < .001 and len(set(nearest)) == len(nearest)
        ids = [annotations[n]["token"] for n in nearest]
        instances = [annotations[n]["instance_token"] for n in nearest]
        for name in ("cars", "pedestrians"):
            for fb, lr in (("front", "left"), ("front", "right"), ("rear", "left"), ("rear", "right")):
                text = f"Find {name} on the {fb} {lr}"
                prompt = parse_object_instruction(text)
                relevant = (labels == prompt["class_id"]) & in_region(ego, prompt)
                irrelevant = (labels == prompt["class_id"]) & ~in_region(ego, prompt)
                rows.append({**{k: info[k] for k in ("sample_token", "scene_token", "index", "lidar2ego_rotation", "lidar2ego_translation")},
                    "instruction_id": f"{info['sample_token']}_{name}_{fb}_{lr}",
                    "text": text, "prompt": prompt, "gt_annotation_ids": ids, "gt_instance_ids": instances,
                    "relevant_gt_indices": np.flatnonzero(relevant).tolist(),
                    "irrelevant_gt_indices": np.flatnonzero(irrelevant).tolist(),
                    "source_npz": str(source), "source_sha256": sha256(source),
                    "gt_geometry_max_error_m": float(distances.min(1).max())})
    write_json(args.output_dir / "protocol.json", {"annotations": rows,
        "definition": "noninterpolated AP@2m_xy; same-class unique nearest GT within2m, descending confidence, full decoded top300; partition by fixed Ego quadrant before matching; macro over car/pedestrian with GT; NOT official nuScenes AP/mAP",
        "selection": "same6 preregistered fresh scene0553 frames; all2 classes x4quadrants, including absent prompts",
        "training": "object-language transfer only; no object-pair supervised training"})
    print("FRESH_OBJECT_PROTOCOL", len(rows), flush=True)


def infer(args):
    protocol = json.loads(args.protocol.read_text())
    selection = json.loads(args.selection.read_text())["selected"]
    assert sha256(Path(selection["checkpoint"])) == selection["sha256"]
    model, wrapped, baseline, dataset = final_dataset()
    pipeline = configure_v2(Stage2Pipeline(baseline.encoder, baseline.grid).cuda())
    pipeline.fusion.load_state_dict(torch.load(selection["checkpoint"], map_location="cpu", weights_only=False)["state_dict"])
    pipeline.eval()
    baseline.fusion.load_state_dict(torch.load(V1, map_location="cpu", weights_only=False)["state_dict"])
    baseline.eval()
    rows = []
    for number, row in enumerate(protocol["annotations"]):
        with np.load(row["source_npz"]) as cached:
            payload = {k: cached[k] for k in ("gt_boxes", "gt_labels", "original_boxes", "original_scores", "original_labels")}
        data = get_batch(dataset, row["index"], 20261027 + row["index"])
        pipeline.fusion.set_calibration(row)
        for branch, system, text in (("historical", baseline, row["text"]),
                                     ("spatial", pipeline, row["text"]),
                                     ("incorrect", pipeline, swap_direction(row["text"]))):
            model.prev_frame_info = {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}
            with torch.inference_mode(), system.inject_shared_bev(
                model.pts_bbox_head.transformer, [text], strict=False,
            ):
                result = wrapped(return_loss=False, rescale=True, **forward_payload(data, inference=True))[0]["pts_bbox"]
            payload[branch + "_boxes"] = result["boxes_3d"].tensor.cpu().numpy()
            payload[branch + "_scores"] = result["scores_3d"].cpu().numpy()
            payload[branch + "_labels"] = result["labels_3d"].cpu().numpy()
        filename = f"objects_{number:03d}.npz"
        np.savez_compressed(args.output_dir / filename, **payload)
        rows.append({**row, "prediction_file": filename})
        del data, payload, result
        release_memory()
        print("FRESH_OBJECT_AB", number + 1, len(protocol["annotations"]), flush=True)
    write_json(args.output_dir / "predictions.json", {**protocol, "annotations": rows,
        "selected_checkpoint_sha256": selection["sha256"], "v1_checkpoint_sha256": sha256(V1)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("protocol", "infer", "metrics"), required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("predictions", "protocol", "selection", "output_dir"):
        value = getattr(args, name)
        if value:
            setattr(args, name, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.phase == "protocol":
        prepare(args)
    elif args.phase == "infer":
        infer(args)
    else:
        metadata = json.loads(args.predictions.read_text())
        metrics, matches = ap_metrics(metadata["annotations"], args.predictions.parent)
        write_json(args.output_dir / "detection_ap.json", {"metrics": metrics, "matches": matches,
            "definition": metadata["definition"], "n_instructions": len(metadata["annotations"]),
            "n_frames": len({r["sample_token"] for r in metadata["annotations"]})})
        import csv
        with (args.output_dir / "detection_ap.csv").open("x") as stream:
            writer = csv.writer(stream); writer.writerow(["branch", "relevant_ap", "irrelevant_ap"])
            for branch, value in metrics.items():
                writer.writerow([branch, value["relevant"]["macro_ap"], value["irrelevant"]["macro_ap"]])
                print(branch, value["relevant"]["macro_ap"], value["irrelevant"]["macro_ap"])


if __name__ == "__main__":
    main()

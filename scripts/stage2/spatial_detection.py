"""独立对象语言协议与真实 3D 中心距离 AP；不改动地图语言解析器。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import re
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2.micro_training import forward_payload, get_batch
from scripts.stage2.run_step70_joint_integration import sha256
from scripts.stage2.runtime_memory import release_memory
from scripts.stage2.spatial_experiment import HISTORICAL, initialize, write_json

DATA = Path("/mnt/g/BEV_VLA_Datasets/bev_seg_nuscenes")


def parse_object_instruction(text: str) -> dict:
    """仅接受明确定义的对象集合指令，与 1/2/3 地图类别完全隔离。"""
    match = re.fullmatch(r"Find (cars|pedestrians) on the (front|rear) (left|right)", text)
    if match is None:
        raise ValueError("不支持的对象指令；地图任务继续使用原地图解析器")
    name, fb, lr = match.groups()
    return {"class_id": 0 if name == "cars" else 8, "object": name,
            "front_back": fb, "left_right": lr}


def ego_centers(boxes, info):
    """真实 LiDAR boxes 的几何中心转到 Ego，使用逐帧 R/t。"""
    from pyquaternion import Quaternion
    centers = boxes[:, :3].copy()
    centers[:, 2] += boxes[:, 5] / 2
    return centers @ Quaternion(info["lidar2ego_rotation"]).rotation_matrix.T + np.asarray(info["lidar2ego_translation"])


def in_region(points, prompt):
    """方位区域是固定米制象限。"""
    front = points[:, 0] > 0 if prompt["front_back"] == "front" else points[:, 0] < 0
    side = points[:, 1] > 0 if prompt["left_right"] == "left" else points[:, 1] < 0
    return front & side


def prepare(args):
    """在对象条件推理前冻结全部 6 帧×2 类×4 方位，包括无目标指令。"""
    from pyquaternion import Quaternion
    with (DATA / "nuscenes_infos_temporal_val.pkl").open("rb") as stream:
        infos = sorted(pickle.load(stream)["infos"], key=lambda r: r["timestamp"])
    annotations = json.loads((DATA / "v1.0-mini/sample_annotation.json").read_text())
    by_sample = defaultdict(list)
    for annotation in annotations:
        by_sample[annotation["sample_token"]].append(annotation)
    rows = []
    for index in (0, 16, 32, 48, 64, 80):
        info = infos[index]
        source = ROOT / f"outputs/stage2_oom_fix_20261022_validation/validate/val_{index:03d}.npz"
        with np.load(source) as data:
            boxes, labels = data["gt_boxes"], data["gt_labels"]
        ego = ego_centers(boxes, info)
        world = ego @ Quaternion(info["ego2global_rotation"]).rotation_matrix.T + np.asarray(info["ego2global_translation"])
        raw = by_sample[info["token"]]
        distances = np.linalg.norm(world[:, None, :] - np.asarray([r["translation"] for r in raw])[None], axis=-1)
        nearest = distances.argmin(1)
        if distances.min(1).max() > .001 or len(set(nearest)) != len(nearest):
            raise ValueError("3D GT 无法唯一对应真实 sample_annotation，拒绝生成对象协议")
        identities = [raw[n]["token"] for n in nearest]
        for obj in ("cars", "pedestrians"):
            for fb, lr in (("front", "left"), ("front", "right"), ("rear", "left"), ("rear", "right")):
                text = f"Find {obj} on the {fb} {lr}"
                prompt = parse_object_instruction(text)
                relevant = (labels == prompt["class_id"]) & in_region(ego, prompt)
                irrelevant = (labels == prompt["class_id"]) & ~in_region(ego, prompt)
                rows.append({"instruction_id": f"{info['token']}_{obj}_{fb}_{lr}",
                             "index": index, "sample_token": info["token"],
                             "scene_token": info["scene_token"], "text": text,
                             "prompt": prompt, "gt_annotation_ids": identities,
                             "relevant_gt_indices": np.flatnonzero(relevant).tolist(),
                             "irrelevant_gt_indices": np.flatnonzero(irrelevant).tolist(),
                             "geometry_match_max_error_m": float(distances.min(1).max()),
                             "source_npz": str(source), "source_sha256": sha256(source),
                             "lidar2ego_rotation": info["lidar2ego_rotation"],
                             "lidar2ego_translation": info["lidar2ego_translation"]})
    write_json(args.output_dir / "object_protocol.json", {
        "annotations": rows,
        "matching": "same class; greedy descending score; unique GT; LiDAR xy center distance <=2m; relevant/irrelevant fixed Ego quadrant; match within each partition; all native decoded top300 scores; no fixed confidence threshold",
        "metric": "noninterpolated object-conditioned AP@2m_xy, macro over car/pedestrian; NOT official nuScenes mAP",
        "gt_source": "real sample_annotation tokens; LiDAR/ego/global geometric correspondence <1mm; plural instruction selects all objects in quadrant; absent prompts retained",
        "training": "no object-language training; map-spatial-trained fusion evaluated for object-language transfer only",
        "model_source_sha256": sha256(Path(__file__)),
    })
    print("OBJECT_PROTOCOL", len(rows), "prompts; GT mapping exact; frozen")


def infer(args):
    protocol = json.loads(args.protocol.read_text())
    model, wrapped, pipeline, dataset = initialize("evaluate")
    historical = torch.load(HISTORICAL, map_location="cpu", weights_only=False)["state_dict"]
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    assert checkpoint["steps"] == 20
    spatial = checkpoint["state_dict"]
    del checkpoint
    rows = []
    for number, row in enumerate(protocol["annotations"]):
        parse_object_instruction(row["text"])
        source = Path(row["source_npz"])
        assert sha256(source) == row["source_sha256"]
        with np.load(source) as old:
            payload = {"gt_boxes": old["gt_boxes"], "gt_labels": old["gt_labels"],
                       "original_boxes": old["original_boxes"],
                       "original_scores": old["original_scores"],
                       "original_labels": old["original_labels"]}
        data = get_batch(dataset, row["index"], 20261026 + 1000 + row["index"])
        actual = dataset.get_ann_info(row["index"])
        np.testing.assert_array_equal(actual["gt_labels_3d"], payload["gt_labels"])
        np.testing.assert_allclose(actual["gt_bboxes_3d"].tensor.numpy(), payload["gt_boxes"], atol=1e-5)
        swapped = row["text"].replace("left", "RIGHT").replace("right", "left").replace("RIGHT", "right")
        for branch, weights, text in (("historical", historical, row["text"]),
                                     ("spatial", spatial, row["text"]),
                                     ("incorrect", spatial, swapped)):
            pipeline.fusion.load_state_dict(weights)
            model.prev_frame_info = {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}
            with torch.inference_mode(), pipeline.inject_shared_bev(
                model.pts_bbox_head.transformer, [text], strict=False,
            ):
                # 对象指令已由独立 schema 校验；不交给地图解析器强行分类。
                result = wrapped(return_loss=False, rescale=True, **forward_payload(data, inference=True))[0]["pts_bbox"]
            payload[branch + "_boxes"] = result["boxes_3d"].tensor.cpu().numpy()
            payload[branch + "_scores"] = result["scores_3d"].cpu().numpy()
            payload[branch + "_labels"] = result["labels_3d"].cpu().numpy()
        filename = f"objects_{number:03d}.npz"
        np.savez_compressed(args.output_dir / filename, **payload)
        rows.append({**row, "prediction_file": filename})
        del data, actual, payload, result
        release_memory()
        print("OBJECT_GPU_AB", number + 1, "/48", row["text"], flush=True)
    write_json(args.output_dir / "predictions.json", {
        **protocol, "annotations": rows, "spatial_checkpoint_sha256": sha256(args.resume),
        "historical_checkpoint_sha256": sha256(HISTORICAL),
        "object_protocol_sha256": sha256(args.protocol), "optimizer_steps": 0})


def ap_metrics(rows, directory):
    """按置信度全排序累积 precision-recall/AP，保留匹配逐条证据。"""
    pools = defaultdict(list)
    gt_counts = defaultdict(int)
    matching_records = []
    for row in rows:
        with np.load(directory / row["prediction_file"]) as data:
            gt_boxes, gt_labels = data["gt_boxes"], data["gt_labels"]
            cls = row["prompt"]["class_id"]
            for branch in ("original", "historical", "spatial", "incorrect"):
                boxes, scores, labels = (data[branch + suffix] for suffix in ("_boxes", "_scores", "_labels"))
                regions = in_region(ego_centers(boxes, row), row["prompt"])
                for region_name, membership in (("relevant", True), ("irrelevant", False)):
                    gt_ids = row[region_name + "_gt_indices"]
                    key = (branch, cls, region_name)
                    gt_counts[key] += len(gt_ids)
                    matched = set()
                    for index in np.argsort(-scores, kind="stable"):
                        if labels[index] != cls or bool(regions[index]) != membership:
                            continue
                        candidates = [i for i in gt_ids if i not in matched]
                        chosen = None
                        distance = None
                        if candidates:
                            dd = np.linalg.norm(gt_boxes[candidates, :2] - boxes[index, :2], axis=1)
                            nearest = int(dd.argmin())
                            if dd[nearest] <= 2.:
                                chosen, distance = candidates[nearest], float(dd[nearest])
                                matched.add(chosen)
                        tp = chosen is not None
                        pools[key].append((float(scores[index]), tp))
                        matching_records.append({"instruction_id": row["instruction_id"], "branch": branch,
                            "class_id": cls, "region": region_name, "score": float(scores[index]),
                            "tp": tp, "matched_sample_annotation": row["gt_annotation_ids"][chosen] if tp else None,
                            "distance_xy_m": distance})
    metrics = {}
    for branch in ("original", "historical", "spatial", "incorrect"):
        metrics[branch] = {}
        for region in ("relevant", "irrelevant"):
            classes = {}
            for cls in (0, 8):
                key = (branch, cls, region)
                predictions = sorted(pools[key], key=lambda r: -r[0])
                tp = 0; ap_sum = 0.; curve = []
                for number, (score, positive) in enumerate(predictions, 1):
                    tp += int(positive)
                    if positive:
                        ap_sum += tp / number
                    curve.append({"score": score, "precision": tp / number,
                                  "recall": tp / gt_counts[key] if gt_counts[key] else None})
                classes[str(cls)] = {"ap": ap_sum / gt_counts[key] if gt_counts[key] else None,
                    "gt_count": gt_counts[key], "pred_count": len(predictions),
                    "tp": tp, "fp": len(predictions) - tp, "fn": gt_counts[key] - tp,
                    "precision_recall": curve}
            values = [r["ap"] for r in classes.values() if r["ap"] is not None]
            metrics[branch][region] = {"macro_ap": float(np.mean(values)) if values else None, "classes": classes}
    return metrics, matching_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("protocol", "infer", "metrics"), required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.protocol:
        args.protocol = args.protocol.resolve()
    if args.resume:
        args.resume = args.resume.resolve()
    if args.phase == "protocol":
        prepare(args)
    elif args.phase == "infer":
        infer(args)
    else:
        data = json.loads(args.predictions.read_text())
        metrics, matches = ap_metrics(data["annotations"], args.predictions.parent)
        write_json(args.output_dir / "detection_ap.json", {
            "metrics": metrics, "protocol": data["matching"],
            "metric_name": data["metric"], "num_instructions": len(data["annotations"]),
            "num_frames": len({r["sample_token"] for r in data["annotations"]}),
            "matches": matches, "transfer_only_no_object_training": True})
        import csv
        with (args.output_dir / "detection_ap.csv").open("x") as stream:
            writer = csv.writer(stream)
            writer.writerow(["branch", "relevant_macro_ap", "irrelevant_macro_ap"])
            for branch, value in metrics.items():
                writer.writerow([branch, value["relevant"]["macro_ap"], value["irrelevant"]["macro_ap"]])
                print(branch, "Relevant AP", value["relevant"]["macro_ap"], "Irrelevant AP", value["irrelevant"]["macro_ap"])


if __name__ == "__main__":
    main()

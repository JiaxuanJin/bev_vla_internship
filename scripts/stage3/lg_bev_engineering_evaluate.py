"""四分支正式计分：SDK四距离条件mAP/全类mAP及真实ROI宏IoU。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from pyquaternion import Quaternion
import torch
from torch import Tensor, nn

from scripts.stage3.lg_bev_data import geometry
from scripts.stage3.lg_bev_engineering_data import CLASSES, LabelStore, masks
from scripts.stage3.lg_bev_engineering_loss import map_roi
from scripts.stage3.lg_bev_engineering_train import choose_query, encode_texts
from scripts.stage3.lg_bev_native import decode, dual_head
from scripts.stage3.lg_bev_supervision_space import AttentionSupervisionSpace

BRANCHES = ("original", "correct_text", "wrong_text", "zero_text")
RANGES = dict(
    car=50.0,
    truck=50.0,
    bus=50.0,
    trailer=50.0,
    construction_vehicle=50.0,
    pedestrian=40.0,
    motorcycle=40.0,
    bicycle=40.0,
    barrier=30.0,
    traffic_cone=30.0,
)


def detection_boxes(
    boxes: np.ndarray,
    classes: np.ndarray,
    scores: np.ndarray,
    frame: dict,
    sample: str,
    num_points: np.ndarray | None = None,
    racks: list | None = None,
) -> list:
    """旧MMDet3D顺时针yaw→nuScenes/Ego，按SDK类别范围/零点/rack规则过滤。"""
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import points_in_box

    result = []
    for i, (b, c, score) in enumerate(zip(boxes, classes, scores)):
        if not 0 <= c < len(CLASSES) or num_points is not None and num_points[i] == 0:
            continue
        name = CLASSES[int(c)]
        center = np.array(b[:3], dtype=float)
        center[2] += b[5] / 2
        orientation = Quaternion(axis=[0, 0, 1], radians=float(-b[6] - np.pi / 2))
        box = Box(center, b[3:6], orientation, velocity=tuple(np.r_[b[7:9], 0.0]))
        if name in ("bicycle", "motorcycle") and racks:
            inside = False
            for rack in racks:
                rack_box = Box(
                    rack[:3],
                    rack[3:6],
                    Quaternion(axis=[0, 0, 1], radians=float(-rack[6] - np.pi / 2)),
                )
                if points_in_box(rack_box, center[:, None])[0]:
                    inside = True
                    break
            if inside:
                continue
        box.rotate(Quaternion(frame["lidar2ego_rotation"]))
        box.translate(np.array(frame["lidar2ego_translation"]))
        if np.linalg.norm(box.center[:2]) >= RANGES[name]:
            continue
        result.append(
            DetectionBox(
                sample_token=sample,
                translation=tuple(box.center),
                size=tuple(box.wlh),
                rotation=tuple(box.orientation.elements),
                velocity=tuple(box.velocity[:2]),
                ego_translation=tuple(box.center),
                num_pts=int(num_points[i]) if num_points is not None else -1,
                detection_name=name,
                detection_score=float(score),
                attribute_name="",
            )
        )
    return result


def sdk_map(ground_truth: dict, predictions: dict, classes: tuple[str, ...]) -> dict:
    """直接调用已安装SDK accumulate/calc_ap；空GT query的FP不丢弃。"""
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.utils import center_distance
    from nuscenes.eval.detection.algo import accumulate, calc_ap

    gt = EvalBoxes()
    pred = EvalBoxes()
    for token in ground_truth:
        gt.add_boxes(token, ground_truth[token])
        pred.add_boxes(token, predictions.get(token, []))
    values = {}
    missing = []
    for cls in classes:
        n = sum(
            b.detection_name == cls for boxes in ground_truth.values() for b in boxes
        )
        if not n:
            missing.append(cls)
            values[cls] = None
            continue
        values[cls] = {
            str(d): float(
                calc_ap(accumulate(gt, pred, cls, center_distance, d), 0.1, 0.1)
            )
            for d in (0.5, 1.0, 2.0, 4.0)
        }
    return dict(
        status="NOT_EVALUABLE" if missing else "OK",
        mAP=(
            None
            if missing
            else float(np.mean([v for c in values.values() for v in c.values()]))
        ),
        class_AP=values,
        missing_positive_classes=missing,
        virtual_samples=len(ground_truth),
        GT_boxes=sum(len(v) for v in ground_truth.values()),
        prediction_boxes=sum(len(v) for v in predictions.values()),
        official_full_validation_mAP=False,
    )


def confusion(gt: np.ndarray, prediction: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """保持原GT轴向与类别，不resize，不把其他同类真实实例当背景。"""
    return np.bincount(
        4 * gt[valid].astype(int) + prediction[valid].astype(int), minlength=16
    ).reshape(4, 4)


def macro_iou(matrix: np.ndarray, classes: tuple[int, ...]) -> dict:
    """缺正类/并集明确NOT_EVALUABLE，不删类伪造达标。"""
    values = {
        c: (
            float(matrix[c, c] / (matrix[c].sum() + matrix[:, c].sum() - matrix[c, c]))
            if matrix[c].sum() + matrix[:, c].sum() - matrix[c, c] > 0
            else None
        )
        for c in classes
    }
    return dict(
        mIoU=(
            float(np.mean(list(values.values())))
            if all(v is not None for v in values.values())
            else None
        ),
        class_IoU=values,
        confusion=matrix.tolist(),
    )


def functional_audit(
    model: nn.Module,
    adapter: nn.Module,
    encoder: nn.Module,
    labels: LabelStore,
    tokens: list[str],
    get_bev: Callable[[str], Tensor],
    *,
    all_queries: bool = False,
    spool=None,
    bootstrap: bool = False,
) -> dict:
    """在训练帧做真实功能验收；正式开发入口另行检查固定终点和一次性闸门。"""
    outputs = {
        b: dict(
            related_gt={},
            related_pred={},
            overall_gt={},
            overall_pred={},
            global_confusion=np.zeros((4, 4), dtype=np.int64),
            related_confusion={c: np.zeros((4, 4), dtype=np.int64) for c in (1, 2)},
            attention={},
            scenes={},
            related_scenes={},
            detection_fp=dict(
                same_class_distractor=0, background=0, outside_native_grid=0
            ),
        )
        for b in BRANCHES
    }
    adapter.eval()
    text_cache = {}
    instructions = 0
    for index, token in enumerate(tokens):
        label = labels.get(token)
        frame = label["frame"]
        bev = get_bev(token).cuda()
        position = geometry(frame, device="cuda")
        canonical = sorted(
            label["queries"],
            key=lambda q: __import__("hashlib")
            .sha256(q["instruction_id"].encode())
            .hexdigest(),
        )[0]
        queries = label["queries"] if all_queries else [choose_query(label, 0, index)]
        if canonical not in queries:
            queries = queries + [canonical]
        for query in queries:
            missing = [
                t for t in (query["text"], query["wrong_text"]) if t not in text_cache
            ]
            text_cache.update(encode_texts(encoder, missing))
            instructions += 1
            gt_masks = masks(label, query)
            target = gt_masks["target"][0, 0].numpy()
            distractor = gt_masks["distractor"][0, 0].numpy()
            for branch in BRANCHES:
                with torch.no_grad():
                    enhanced = bev
                    attention = None
                    if branch != "original":
                        text = (
                            torch.zeros_like(text_cache[query["text"]])
                            if branch == "zero_text"
                            else text_cache[
                                (
                                    query["wrong_text"]
                                    if branch == "wrong_text"
                                    else query["text"]
                                )
                            ]
                        )
                        detail = adapter(bev, text, position)
                        enhanced = detail["enhanced_bev"]
                        attention = (
                            AttentionSupervisionSpace.from_calibration(
                                query["task"], frame
                            )
                            .project(detail["attention_logits"])
                            .sigmoid()[0, 0]
                            .cpu()
                            .numpy()
                            >= 0.5
                        )
                    pred = dual_head(model, enhanced)
                    decoded = decode(model, pred)
                    semantic = pred["seg_preds"].argmax(1)[0].cpu().numpy()
                result = outputs[branch]
                virtual = token + "/" + query["instruction_id"]
                if query["task"] == "object":
                    selected = set(query["target_instance_ids"])
                    gt_keep = np.array(
                        [t in selected for t in label["eval_annotation_tokens"]]
                    )
                    gt_boxes = label["eval_boxes"].numpy()
                    gt_labels = label["eval_labels"].numpy()
                    points = label["eval_num_points"].numpy()
                    result["related_gt"][virtual] = detection_boxes(
                        gt_boxes[gt_keep],
                        gt_labels[gt_keep],
                        np.full(gt_keep.sum(), -1.0),
                        frame,
                        virtual,
                        points[gt_keep],
                        label.get("bicycle_racks"),
                    )
                    p = decoded["labels"] == query["target_class"]
                    result["related_pred"][virtual] = detection_boxes(
                        decoded["boxes"][p],
                        decoded["labels"][p],
                        decoded["scores"][p],
                        frame,
                        virtual,
                        racks=label.get("bicycle_racks"),
                    )
                    if spool is not None:
                        spool.add(
                            branch,
                            "related",
                            virtual,
                            frame["scene_token"],
                            result["related_gt"].pop(virtual),
                            result["related_pred"].pop(virtual),
                        )
                    xy = decoded["boxes"][p, :2]
                    cells = np.floor((xy + 51.2) / (102.4 / 150)).astype(int)
                    inside = (cells >= 0).all(1) & (cells < 150).all(1)
                    cells = cells[inside]
                    result["detection_fp"]["outside_native_grid"] += int(
                        (~inside).sum()
                    )
                    result["detection_fp"]["same_class_distractor"] += int(
                        distractor[cells[:, 1], cells[:, 0]].sum()
                    )
                    background = ~label["object_masks"].numpy().any(0)
                    result["detection_fp"]["background"] += int(
                        background[cells[:, 1], cells[:, 0]].sum()
                    )
                elif target.any():
                    c = query["target_class"]
                    roi = map_roi(target)
                    result["related_confusion"][c] += confusion(
                        label["gt"].numpy(), semantic, roi
                    )
                    scene = result["related_scenes"].setdefault(
                        frame["scene_token"],
                        {k: np.zeros((4, 4), dtype=np.int64) for k in (1, 2)},
                    )
                    scene[c] += confusion(label["gt"].numpy(), semantic, roi)
                if query is canonical:
                    result["overall_gt"][token] = detection_boxes(
                        label["eval_boxes"].numpy(),
                        label["eval_labels"].numpy(),
                        np.full(len(label["eval_boxes"]), -1.0),
                        frame,
                        token,
                        label["eval_num_points"].numpy(),
                        label.get("bicycle_racks"),
                    )
                    result["overall_pred"][token] = detection_boxes(
                        decoded["boxes"],
                        decoded["labels"],
                        decoded["scores"],
                        frame,
                        token,
                        racks=label.get("bicycle_racks"),
                    )
                    if spool is not None:
                        spool.add(
                            branch,
                            "overall",
                            token,
                            frame["scene_token"],
                            result["overall_gt"].pop(token),
                            result["overall_pred"].pop(token),
                        )
                    matrix = confusion(
                        label["gt"].numpy(), semantic, np.ones((200, 400), bool)
                    )
                    result["global_confusion"] += matrix
                    result["scenes"].setdefault(
                        frame["scene_token"], np.zeros((4, 4), dtype=np.int64)
                    )
                    result["scenes"][frame["scene_token"]] += matrix
                if attention is not None:
                    category = query["task"]
                    counts = result["attention"].setdefault(
                        category,
                        dict(
                            tp=0,
                            union=0,
                            background_fp=0,
                            distractor_fp=0,
                            empty_active=0,
                            empty_queries=0,
                        ),
                    )
                    counts["tp"] += int((attention & target).sum())
                    counts["union"] += int((attention | target).sum())
                    # 真正背景定义来自原始全类别GT，不是~target。
                    background = (
                        label["gt"].numpy() == 0
                        if category == "map"
                        else ~label["object_masks"].numpy().any(0)
                    )
                    counts["background_fp"] += int((attention & background).sum())
                    counts["distractor_fp"] += int((attention & distractor).sum())
                    if not target.any():
                        counts["empty_queries"] += 1
                        counts["empty_active"] += int(attention.sum())
    summary = {}
    for branch, value in outputs.items():
        related = {
            c: macro_iou(matrix, (c,))["class_IoU"][c]
            for c, matrix in value["related_confusion"].items()
        }
        summary[branch] = dict(
            related_conditional_mAP=(
                spool.metric(branch, "related", ("car", "pedestrian"))
                if spool
                else sdk_map(
                    value["related_gt"], value["related_pred"], ("car", "pedestrian")
                )
            ),
            overall_10class_mAP=(
                spool.metric(branch, "overall", CLASSES)
                if spool
                else sdk_map(value["overall_gt"], value["overall_pred"], CLASSES)
            ),
            related_macro_IoU=(
                float(np.mean(list(related.values())))
                if all(v is not None for v in related.values())
                else None
            ),
            related_class_IoU=related,
            overall_3class_mIoU=macro_iou(value["global_confusion"], (1, 2, 3)),
            attention={
                k: {**v, "IoU": v["tp"] / v["union"] if v["union"] else None}
                for k, v in value["attention"].items()
            },
            detection_center_footprint_FP=value["detection_fp"],
            per_scene={k: macro_iou(v, (1, 2, 3)) for k, v in value["scenes"].items()},
        )
        if spool:
            for scene in summary[branch]["per_scene"]:
                summary[branch]["per_scene"][scene].update(
                    related_conditional_mAP=spool.metric(
                        branch, "related", ("car", "pedestrian"), scene
                    ),
                    overall_10class_mAP=spool.metric(branch, "overall", CLASSES, scene),
                )
            if bootstrap:
                summary[branch]["scene_bootstrap"] = dict(
                    related_mAP=spool.bootstrap(
                        branch, "related", ("car", "pedestrian")
                    ),
                    overall_mAP=spool.bootstrap(branch, "overall", CLASSES),
                )
                summary[branch]["scene_bootstrap"].update(
                    related_mIoU=seg_bootstrap(value["related_scenes"], related=True),
                    overall_mIoU=seg_bootstrap(value["scenes"], related=False),
                )
    return dict(
        status="REAL_TRAINING_FUNCTIONAL_CHECK_NOT_GENERALIZATION",
        frames=len(tokens),
        instructions=instructions,
        formal_development_or_final_evaluations=0,
        branches=summary,
        controls="same trainedadapter inference controls, identical correctGT",
        SDK="installed nuscenes.eval.detection.algo accumulate/calc_ap; conditional metric is not official whole-val mAP",
    )


def seg_bootstrap(scenes: dict, *, related: bool) -> dict:
    """场景级1000次宏IoU区间，单场景/缺失类别不产生虚假置信区间。"""
    keys = sorted(scenes)
    if len(keys) < 2:
        return dict(status="NOT_MEANINGFUL_ONE_SCENE")
    rng = np.random.RandomState(20261027)
    values = []
    for _ in range(1000):
        selected = rng.choice(len(keys), len(keys), replace=True)
        if related:
            scores = []
            for c in (1, 2):
                matrix = sum(
                    (scenes[keys[i]][c] for i in selected),
                    np.zeros((4, 4), dtype=np.int64),
                )
                scores.append(macro_iou(matrix, (c,))["class_IoU"][c])
            if all(s is not None for s in scores):
                values.append(float(np.mean(scores)))
        else:
            score = macro_iou(
                sum(
                    (scenes[keys[i]] for i in selected),
                    np.zeros((4, 4), dtype=np.int64),
                ),
                (1, 2, 3),
            )["mIoU"]
            if score is not None:
                values.append(score)
    return dict(
        status="SCENE_CLUSTER_BOOTSTRAP",
        replicates=1000,
        valid_replicates=len(values),
        percentile95=np.quantile(values, [0.025, 0.975]).tolist() if values else None,
    )


def main() -> None:
    """只有正式B终点、封存摘要和一次性闸门齐全才允许打开development。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    args = parser.parse_args()
    from scripts.stage3.lg_bev_engineering_full import evaluate_terminal

    evaluate_terminal(args)


if __name__ == "__main__":
    main()

"""三组与A0/A9562的统一推理驱动；只有授权worker才调用真实模型。"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from core import freeze, group_bev, load_adapter, require
from metrics import DetectionAP, SegmentationIoU, UnifiedMetrics

CONTROLS = ("correct", "wrong", "zero")
CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)


def queries(label, index):
    """每帧固定两条指令，保留空GT，不根据预测选择类别或区域。"""
    result = []
    for task, cls in (("object", 0 if index % 2 == 0 else 8), ("map", 1 + index % 2)):
        pool = [
            q
            for q in label["queries"]
            if q["task"] == task
            and q["target_class"] == cls
            and (task != "object" or q["qualifier"] == "all")
        ]
        require(pool, "预登记指令不存在，禁止静默换帧")
        result.append(
            min(
                pool,
                key=lambda q: hashlib.sha256(
                    ("explore-eval-20261101:" + q["instruction_id"]).encode()
                ).hexdigest(),
            )
        )
    return result


def detection_records(label, decoded, query_id):
    """两组框共同沿用SDK过滤和LiDAR→Ego变换；GT保留实例身份。"""
    from scripts.stage3.lg_bev_engineering_evaluate import detection_boxes

    truth = []
    for i, identity in enumerate(label["eval_annotation_tokens"]):
        boxes = detection_boxes(
            label["eval_boxes"][i : i + 1].numpy(),
            label["eval_labels"][i : i + 1].numpy(),
            np.array([-1.0]),
            label["frame"],
            query_id,
            label["eval_num_points"][i : i + 1].numpy(),
            label.get("bicycle_racks"),
        )
        truth.extend(
            {"id": identity, "class": b.detection_name, "xy": list(b.translation[:2])}
            for b in boxes
        )
    forecasts = detection_boxes(
        decoded["boxes"],
        decoded["labels"],
        decoded["scores"],
        label["frame"],
        query_id,
        racks=label.get("bicycle_racks"),
    )
    predictions = [
        {
            "class": b.detection_name,
            "xy": list(b.translation[:2]),
            "score": b.detection_score,
        }
        for b in forecasts
    ]
    return truth, predictions


def head_equal(left, right):
    """原生检测分类/回归及分割三张量逐位一致。"""
    return all(
        torch.equal(left[k], right[k])
        for k in ("all_cls_scores", "all_bbox_preds", "seg_preds")
    )


def ground_truth_coverage(runtime, rows, stop=lambda: False):
    """在任何预测前检查固定清单的GT覆盖；只由已授权评价进程调用。"""
    from scripts.stage3.lg_bev_engineering_data import masks

    detection, segmentation, query_ids = DetectionAP(), SegmentationIoU(), []
    for index, row in enumerate(rows):
        if stop():
            return None
        token = row["sample_token"]
        label = runtime.validate_label(token)
        for query in queries(label, index):
            query_id = token + "/" + query["instruction_id"]
            query_ids.append(query_id)
            if query["task"] == "object":
                gt, _ = detection_records(
                    label, {"boxes": [], "labels": [], "scores": []}, query_id
                )
                detection.add(
                    query_id,
                    gt,
                    [],
                    "car" if query["target_class"] == 0 else "pedestrian",
                    query["fb"],
                    query["lr"],
                )
            else:
                supervision = masks(label, query)
                segmentation.add(
                    query_id,
                    label["gt"].numpy(),
                    np.zeros_like(label["gt"].numpy()),
                    supervision["target"][0, 0].numpy(),
                    supervision["valid"][0, 0].numpy(),
                    query["target_class"],
                )
    det, seg = detection.result(), segmentation.result()
    return {
        "query_ids": query_ids,
        "frames": len(rows),
        "detection_GT": {
            r: {c: ds["0.5"]["GT"] for c, ds in value["classes"].items()}
            for r, value in det.items()
        },
        "segmentation_GT": {
            r: {
                c: {k: values[k] for k in ("GT_positive_pixels", "pixels")}
                for c, values in value["classes"].items()
            }
            for r, value in seg.items()
        },
        "policy": "缺正类如实NOT_EVALUABLE；固定320帧，不按模型表现改样本或阈值",
    }


def evaluate_frame(runtime, adapters, token, index):
    """产生统一轻量计分行；每一组都使用同一正确query及GT。"""
    from scripts.stage3.lg_bev_engineering_data import masks
    from scripts.stage3.lg_bev_engineering_train import encode_texts
    from scripts.stage3.lg_bev_data import geometry
    from scripts.stage3.lg_bev_supervision_space import AttentionSupervisionSpace
    from scripts.stage3.lg_bev_native import dual_head, decode

    label = runtime.validate_label(token)
    bev = runtime.get(token).to(runtime.device)
    position = geometry(label["frame"], device=runtime.device)
    selected = queries(label, index)
    texts = encode_texts(
        runtime.encoder, [q[k] for q in selected for k in ("text", "wrong_text")]
    )
    records = []
    with torch.no_grad():
        baseline = dual_head(runtime.model, bev)
        # A0仅比较Attention，不伪造其独立训练结果。
        for query in selected:
            supervision = masks(label, query)
            target, valid = (supervision[k][0, 0].numpy() for k in ("target", "valid"))
            space = AttentionSupervisionSpace.from_calibration(
                query["task"], label["frame"]
            )
            paired = query["pair_id"] is not None
            common = {
                "query_id": token + "/" + query["instruction_id"],
                "task": query["task"],
                "target": target,
                "valid": valid,
                "target_class": query["target_class"],
                "fb": query["fb"],
                "lr": query["lr"],
                "paired": paired,
                "selected_ids": query["target_instance_ids"],
                "scene": label["frame"]["scene_token"],
                "log": runtime.rows[token]["log_token"],
                "gt": label["gt"].numpy(),
                "distractor": supervision["distractor"][0, 0].numpy(),
                "background": (
                    label["gt"].numpy() == 0
                    if query["task"] == "map"
                    else ~label["object_masks"].numpy().any(0)
                ),
            }
            for group in ("baseline", "a_only", "enhanced", "a0"):
                details, predictions = {}, {}
                for control, text in (
                    ("correct", texts[query["text"]]),
                    ("wrong", texts[query["wrong_text"]]),
                    ("zero", torch.zeros_like(texts[query["text"]])),
                ):
                    detail = group_bev(group, adapters.get(group), bev, text, position)
                    if group in ("baseline", "a0"):
                        prediction = baseline
                    else:
                        prediction = dual_head(runtime.model, detail["enhanced_bev"])
                        if group == "a_only":
                            require(
                                head_equal(prediction, baseline),
                                "AOnly不再与Baseline恒等",
                            )
                    require(
                        all(
                            torch.isfinite(prediction[k]).all()
                            for k in ("all_cls_scores", "all_bbox_preds", "seg_preds")
                        ),
                        "预测非有限",
                    )
                    details[control] = (
                        space.project(detail["attention_logits"])[0, 0].cpu().numpy()
                        if detail["attention_logits"] is not None
                        else None
                    )
                    if group != "a0":
                        decoded = decode(runtime.model, prediction)
                        truth, forecasts = detection_records(
                            label, decoded, common["query_id"]
                        )
                        predictions[control] = {
                            "gt_boxes": truth,
                            "pred_boxes": forecasts,
                            "semantic": prediction["seg_preds"]
                            .argmax(1)[0]
                            .cpu()
                            .numpy(),
                        }
                records.append(
                    {
                        **common,
                        "group": group,
                        "details": details,
                        "predictions": predictions,
                    }
                )
    return records


class EvaluationState:
    """流式累计状态，可在帧边界保存继续，不保存全部大张量预测。"""

    def __init__(self):
        self.accumulators = {}
        self.per_log = {}
        self.per_scene = {}
        self.supplemental = {}

    def add(self, row):
        """同一计分函数处理真实/合成输入及全部实验组。"""
        accumulators, per_log = self.accumulators, self.per_log
        group = row["group"]
        for control in CONTROLS:
            key = group + "/" + control
            target = accumulators.setdefault(key, UnifiedMetrics())
            log_key = (row["log"], key)
            cluster = per_log.setdefault(log_key, UnifiedMetrics())
            scene = self.per_scene.setdefault((row["scene"], key), UnifiedMetrics())
            for destination in (target, cluster, scene):
                if group != "a0":
                    prediction = row["predictions"][control]
                    if row["task"] == "object":
                        destination.detection.add(
                            row["query_id"],
                            prediction["gt_boxes"],
                            prediction["pred_boxes"],
                            "car" if row["target_class"] == 0 else "pedestrian",
                            row["fb"],
                            row["lr"],
                        )
                    else:
                        destination.segmentation.add(
                            row["query_id"],
                            row["gt"],
                            prediction["semantic"],
                            row["target"],
                            row["valid"],
                            row["target_class"],
                        )
                if group != "baseline":
                    destination.attention.add(
                        row["query_id"],
                        row["task"],
                        row["details"][control],
                        row["details"]["wrong" if control == "correct" else "correct"],
                        row["target"],
                        row["valid"],
                        paired=row["paired"],
                        distractor=row["distractor"],
                        background=row["background"],
                    )
            if group != "a0":
                extra = self.supplemental.setdefault(key, Supplemental())
                extra.add(row, row["predictions"][control])

    def result(self):
        """同时导出独立累计的全子集结果及驾驶日志结果。"""
        return {
            "groups": {key: value.result() for key, value in self.accumulators.items()},
            "per_log": {
                log + "/" + key: value.result()
                for (log, key), value in self.per_log.items()
            },
            "per_scene": {
                scene + "/" + key: value.result()
                for (scene, key), value in self.per_scene.items()
            },
            "supplemental": {
                key: value.result() for key, value in self.supplemental.items()
            },
            "A9562_minus_A0": self.attention_delta(),
            "paired_log_comparison": self.paired_logs(),
            "A0_provenance": "独立验收零更新参数候选，非正式A0回执",
            "heldout_scope": "仅Adapter训练日志留出，作者主干预训练覆盖未核实",
        }

    def attention_delta(self):
        """同一清单的前后Attention差值；无正例仍保留N/A。"""
        keys = ("a0/correct", "a_only/correct")
        if not all(k in self.accumulators for k in keys):
            return None
        before, after = [self.accumulators[k].attention.result() for k in keys]
        return {
            task: {
                key: (
                    after[task][key] - before[task][key]
                    if after[task][key] is not None and before[task][key] is not None
                    else None
                )
                for key in (
                    "positive_macro_IoU",
                    "mean_logit_gap",
                    "mean_probability_gap",
                    "empty_active_fraction",
                )
            }
            for task in ("map", "object")
        }

    def paired_logs(self):
        """固定种子驾驶日志配对bootstrap；估计量是日志宏平均差，不冒充pooled差CI。"""
        logs = sorted({log for log, _ in self.per_log})
        values = {key: value.result() for key, value in self.per_log.items()}
        result = {}
        for metric, field in (
            ("Relevant_Detection_AP", "mAP"),
            ("Irrelevant_Detection_AP", "mAP"),
            ("Relevant_Segmentation_IoU", "mIoU"),
            ("Irrelevant_Segmentation_IoU", "mIoU"),
        ):
            differences = []
            for log in logs:
                first, second = (
                    values.get((log, k), {}).get(metric, {}).get(field)
                    for k in ("baseline/correct", "enhanced/correct")
                )
                differences.append(
                    None if first is None or second is None else second - first
                )
            if not differences or any(v is None for v in differences):
                result[metric] = {
                    "status": "NOT_EVALUABLE",
                    "log_differences": differences,
                }
                continue
            rng = np.random.RandomState(20261101)
            replicates = rng.choice(differences, (1000, len(logs)), replace=True).mean(
                1
            )
            result[metric] = {
                "mean_log_delta": float(np.mean(differences)),
                "percentile_95": np.percentile(replicates, [2.5, 97.5]).tolist(),
                "logs": logs,
                "log_differences": differences,
                "说明": "仅日志宏平均差；最多5日志，不能承诺显著性",
            }
        return result


class Supplemental:
    """辅助全10类AP、全前景语义及旧实例conditional AP；与区域主指标分开。"""

    def __init__(self):
        self.overall = DetectionAP(CLASSES)
        self.conditional = DetectionAP()
        self.confusion = np.zeros((4, 4), np.int64)

    def add(self, row, prediction):
        for cls in CLASSES:
            self.overall.add(
                row["query_id"] + "/" + cls,
                prediction["gt_boxes"],
                prediction["pred_boxes"],
                cls,
                "front",
                "left",
                partition=False,
            )
        # 全图指标按固定两query/帧协议计分，不称唯一原始帧官方指标。
        gt, pred = row["gt"], prediction["semantic"]
        self.confusion += np.bincount(
            4 * gt.reshape(-1).astype(int) + pred.reshape(-1).astype(int), minlength=16
        ).reshape(4, 4)
        if row["task"] == "object":
            selected = set(row["selected_ids"])
            self.conditional.add(
                row["query_id"],
                [b for b in prediction["gt_boxes"] if b["id"] in selected],
                prediction["pred_boxes"],
                "car" if row["target_class"] == 0 else "pedestrian",
                row["fb"],
                row["lr"],
                partition=False,
            )

    def result(self):
        values = {}
        for cls in (1, 2, 3):
            matrix = self.confusion
            union = int(matrix[cls].sum() + matrix[:, cls].sum() - matrix[cls, cls])
            values[str(cls)] = (
                float(matrix[cls, cls] / union) if matrix[cls].sum() and union else None
            )
        return {
            "overall_10class_AP": self.overall.result()["relevant"],
            "instance_conditional_AP": self.conditional.result()["relevant"],
            "overall_foreground_IoU": {
                "classes": values,
                "mIoU": (
                    float(np.mean(list(values.values())))
                    if all(v is not None for v in values.values())
                    else None
                ),
                "confusion": self.confusion.tolist(),
            },
            "scope": "固定query级辅助指标，不替代相关/无关区域指标或官方全val",
        }


def aggregate(records):
    """合成验收和真实入口复用同一流式状态机。"""
    state = EvaluationState()
    for row in records:
        state.add(row)
    return state.result()


def adapters_for_evaluation(config, checkpoint):
    """只有自有独立B的固定终点允许进入完整开发评价。"""
    enhanced = load_adapter(config)
    enhanced.load_state_dict(checkpoint["adapter"], strict=True)
    return {
        "a_only": freeze(load_adapter(config)).cuda(),
        "enhanced": freeze(enhanced).cuda(),
        "a0": freeze(load_adapter(config, "a0")).cuda(),
    }

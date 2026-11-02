"""统一CPU计分：相关/无关区独立累计，不由全局指标相减。"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import distance_transform_edt

from core import require

DISTANCES = (0.5, 1.0, 2.0, 4.0)
REGIONS = ("relevant", "irrelevant")


def quadrant(xy, front_back, left_right):
    """Ego坐标轴零值归front/left；区域及其补集覆盖完整有效域。"""
    require(
        front_back in ("front", "back", "rear") and left_right in ("left", "right"),
        "方向必须明确",
    )
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    require(np.isfinite(xy).all(), "检测坐标非有限")
    front = xy[:, 0] >= 0
    left = xy[:, 1] >= 0
    return (front if front_back == "front" else ~front) & (
        left if left_right == "left" else ~left
    )


def segmentation_regions(target, valid):
    """目标地图mask按0.15m像素扩张2m，空目标的无关区为完整有效域。"""
    target, valid = np.asarray(target, bool), np.asarray(valid, bool)
    require(target.shape == valid.shape and target.ndim == 2, "区域形状不符")
    roi = (
        distance_transform_edt(~target, sampling=0.15) <= 2 if target.any() else target
    )
    return {"relevant": roi & valid, "irrelevant": ~roi & valid}


def sdk_ap(hits, positives):
    """SDK的101点插值与min_recall/min_precision=.1；无正例为N/A。"""
    if positives == 0:
        return None
    hits = np.asarray(hits, dtype=float)
    if not len(hits) or not hits.any():
        return 0.0
    recall = np.cumsum(hits) / positives
    precision = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    interpolated = np.interp(np.linspace(0, 1, 101), recall, precision, right=0)
    return float(np.maximum(interpolated[11:] - 0.1, 0).mean() / 0.9)


class DetectionAP:
    """按query独立贪心匹配，再按类别/距离跨query全局排序。"""

    def __init__(self, classes=("car", "pedestrian")):
        self.classes = classes
        self.counts = defaultdict(int)
        self.events = defaultdict(list)
        self.queries = set()
        self.sequence = 0
        self.empty_predictions = defaultdict(int)

    def add(self, query_id, gt, predictions, target_class, fb, lr, *, partition=True):
        """框必须已按统一native解码与SDK规则转到Ego/过滤，不做GT邻近筛选。"""
        require(query_id not in self.queries, "重复query会重复计分")
        require(target_class in self.classes, "请求类别不在预登记集合")
        require(len({b["id"] for b in gt}) == len(gt), "重复GT实例ID")
        self.queries.add(query_id)
        gt = [b for b in gt if b["class"] == target_class]
        predictions = [b for b in predictions if b["class"] == target_class]
        inside_gt = quadrant([b["xy"] for b in gt], fb, lr)
        inside_pred = quadrant([b["xy"] for b in predictions], fb, lr)
        if not partition:
            inside_gt = np.ones(len(gt), bool)
            inside_pred = np.ones(len(predictions), bool)
        require(all(np.isfinite(b["score"]) for b in predictions), "预测分数非有限")
        for region in REGIONS:
            truth = [
                b
                for b, inside in zip(gt, inside_gt)
                if inside == (region == "relevant")
            ]
            forecasts = [
                b
                for b, inside in zip(predictions, inside_pred)
                if inside == (region == "relevant")
            ]
            self.counts[region, target_class] += len(truth)
            if not truth:
                self.empty_predictions[region] += len(forecasts)
            order = sorted(
                range(len(forecasts)),
                key=lambda i: (forecasts[i]["score"], i),
                reverse=True,
            )
            sequences = list(range(self.sequence, self.sequence + len(forecasts)))
            self.sequence += len(forecasts)
            for distance in DISTANCES:
                matched = set()
                hits = {}
                for i in order:
                    candidates = [
                        (
                            float(
                                np.linalg.norm(
                                    np.array(forecasts[i]["xy"]) - target["xy"]
                                )
                            ),
                            j,
                        )
                        for j, target in enumerate(truth)
                        if j not in matched
                    ]
                    best = min(candidates, default=(float("inf"), -1))
                    hit = best[0] < distance
                    if hit:
                        matched.add(best[1])
                    hits[i] = int(hit)
                self.events[region, target_class, distance].extend(
                    (b["score"], sequences[i], hits[i]) for i, b in enumerate(forecasts)
                )

    def result(self):
        """保留每类GT/TP/FP及空GT预测数；缺正类不删除分母。"""
        result = {}
        for region in REGIONS:
            classes = {}
            for cls in self.classes:
                distances = {}
                for distance in DISTANCES:
                    events = sorted(self.events[region, cls, distance], reverse=True)
                    hits = [e[2] for e in events]
                    distances[str(distance)] = {
                        "AP": sdk_ap(hits, self.counts[region, cls]),
                        "TP": sum(hits),
                        "FP": len(hits) - sum(hits),
                        "GT": self.counts[region, cls],
                    }
                classes[cls] = distances
            aps = [v["AP"] for ds in classes.values() for v in ds.values()]
            result[region] = {
                "mAP": float(np.mean(aps)) if all(v is not None for v in aps) else None,
                "AP@2m_xy": (
                    float(np.mean([ds["2.0"]["AP"] for ds in classes.values()]))
                    if all(ds["2.0"]["AP"] is not None for ds in classes.values())
                    else None
                ),
                "status": "OK" if all(v is not None for v in aps) else "NOT_EVALUABLE",
                "classes": classes,
                "empty_GT_predictions": self.empty_predictions[region],
            }
        return result


class SegmentationIoU:
    """每请求类别在两个互补区域独立累积完整语义混淆矩阵。"""

    def __init__(self):
        self.matrices = {
            (r, c): np.zeros((4, 4), np.int64) for r in REGIONS for c in (1, 2)
        }
        self.queries = set()

    def add(self, query_id, gt, prediction, target, valid, target_class):
        require(query_id not in self.queries, "重复分割query")
        require(target_class in (1, 2), "地图请求类别不符")
        gt, prediction = np.asarray(gt), np.asarray(prediction)
        require(
            gt.shape == prediction.shape == np.asarray(target).shape, "分割尺寸错配"
        )
        require(
            np.isin(gt, np.arange(4)).all() and np.isin(prediction, np.arange(4)).all(),
            "语义类别无效",
        )
        self.queries.add(query_id)
        for region, mask in segmentation_regions(target, valid).items():
            self.matrices[region, target_class] += np.bincount(
                4 * gt[mask].astype(int) + prediction[mask].astype(int), minlength=16
            ).reshape(4, 4)

    def result(self):
        result = {}
        for region in REGIONS:
            classes = {}
            for cls in (1, 2):
                # 所有地图query的完整语义共同计分，不能只取请求类的query。
                matrix = sum(self.matrices[region, c] for c in (1, 2))
                positive = int(matrix[cls].sum())
                union = int(matrix[cls].sum() + matrix[:, cls].sum() - matrix[cls, cls])
                classes[str(cls)] = {
                    "IoU": (
                        float(matrix[cls, cls] / union) if positive and union else None
                    ),
                    "GT_positive_pixels": positive,
                    "union": union,
                    "intersection": int(matrix[cls, cls]),
                    "pixels": int(matrix.sum()),
                    "confusion": matrix.tolist(),
                }
            values = [v["IoU"] for v in classes.values()]
            result[region] = {
                "mIoU": (
                    float(np.mean(values))
                    if all(v is not None for v in values)
                    else None
                ),
                "status": (
                    "OK" if all(v is not None for v in values) else "NOT_EVALUABLE"
                ),
                "classes": classes,
            }
        return result


class AttentionMetrics:
    """正确/错误共享正确GT；macro与pooled独立命名，空目标单独累计。"""

    def __init__(self):
        self.rows = []

    def add(
        self,
        query_id,
        task,
        correct,
        wrong,
        target,
        valid,
        *,
        paired=True,
        distractor=None,
        background=None,
    ):
        require(task in ("map", "object"), "Attention任务错误")
        require(
            query_id not in {r["query_id"] for r in self.rows}, "重复Attention query"
        )
        correct, wrong = np.asarray(correct, float), np.asarray(wrong, float)
        target, valid = np.asarray(target, bool), np.asarray(valid, bool)
        require(
            correct.shape == wrong.shape == target.shape == valid.shape,
            "Attention尺寸错配",
        )
        require(
            np.isfinite(correct).all() and np.isfinite(wrong).all(), "Attention非有限"
        )
        target = target & valid
        require(valid.any(), "Attention有效域为空")
        active = (correct >= 0) & valid
        wrong_active = (wrong >= 0) & valid
        positive = bool(target.any())
        intersection = int((active & target).sum())
        union = int((active | target).sum())
        prob = 1 / (1 + np.exp(-np.clip(correct, -80, 80)))
        wrong_prob = 1 / (1 + np.exp(-np.clip(wrong, -80, 80)))
        target_size, active_size = int(target.sum()), int(active.sum())
        row = {
            "query_id": query_id,
            "task": task,
            "positive": positive,
            "intersection": intersection,
            "union": union,
            "IoU": intersection / union if positive else None,
            "wrong_IoU": (
                float((wrong_active & target).sum() / (wrong_active | target).sum())
                if positive and paired
                else None
            ),
            "Dice": (
                2 * intersection / (active_size + target_size) if positive else None
            ),
            "precision": intersection / active_size if active_size else None,
            "recall": intersection / target_size if positive else None,
            "logit_gap": (
                float((correct - wrong)[target].mean()) if positive and paired else None
            ),
            "probability_gap": (
                float((prob - wrong_prob)[target].mean())
                if positive and paired
                else None
            ),
            "empty_active_fraction": (
                active_size / int(valid.sum()) if not positive else None
            ),
            "target_probability_mass_fraction": float(
                prob[target].sum() / prob[valid].sum()
            ),
            "peak_in_target": bool(
                target.flat[np.argmax(np.where(valid, correct, -np.inf))]
            ),
            "distractor_FP": (
                int((active & np.asarray(distractor, bool)).sum())
                if distractor is not None
                else None
            ),
            "background_FP": (
                int((active & np.asarray(background, bool)).sum())
                if background is not None
                else None
            ),
            "counterfactual_valid": paired,
        }
        self.rows.append(row)

    def result(self):
        result = {}
        for task in ("map", "object"):
            rows = [r for r in self.rows if r["task"] == task]
            positive = [r for r in rows if r["positive"]]

            def mean(key):
                values = [r[key] for r in rows if r[key] is not None]
                return float(np.mean(values)) if values else None

            union = sum(r["union"] for r in positive)
            result[task] = {
                "positive_macro_IoU": mean("IoU"),
                "positive_pooled_IoU": (
                    sum(r["intersection"] for r in positive) / union if union else None
                ),
                "wrong_positive_macro_IoU": mean("wrong_IoU"),
                "mean_logit_gap": mean("logit_gap"),
                "mean_probability_gap": mean("probability_gap"),
                "empty_active_fraction": mean("empty_active_fraction"),
                "queries": len(rows),
                "positive_queries": len(positive),
                "empty_queries": len(rows) - len(positive),
                "valid_counterfactuals": sum(r["counterfactual_valid"] for r in rows),
                "rows": rows,
            }
        return result


class UnifiedMetrics:
    """所有模型组使用同一累计器，不允许专用分支改变评分目标。"""

    def __init__(self):
        self.detection = DetectionAP()
        self.segmentation = SegmentationIoU()
        self.attention = AttentionMetrics()

    def result(self):
        detection, segmentation = self.detection.result(), self.segmentation.result()
        return {
            "Relevant_Detection_AP": detection["relevant"],
            "Irrelevant_Detection_AP": detection["irrelevant"],
            "Relevant_Segmentation_IoU": segmentation["relevant"],
            "Irrelevant_Segmentation_IoU": segmentation["irrelevant"],
            "Attention": self.attention.result(),
            "scope": "独立探索子集区域指标；不是官方全val结果",
        }

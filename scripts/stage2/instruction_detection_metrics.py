"""指令相关检测 AP 的匹配结果聚合；匹配必须来自真实 3D 评估协议。"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PredictionMatch:
    """一条检测预测的独立匹配结果。

    score: 置信度，范围 [0,1]。
    matched_gt_id: 官方匹配器给出的 GT 实例 ID；未匹配时填 None。
    in_instruction_region: 预测几何中心是否在已验证的指令区域内。
    """

    score: float
    matched_gt_id: str | None
    in_instruction_region: bool


def region_average_precision(
    predictions: list[PredictionMatch],
    *,
    relevant_gt_ids: set[str],
    irrelevant_gt_ids: set[str],
) -> dict[str, object]:
    """针对单类别、单匹配阈值计算 AP，绝不称为官方 nuScenes mAP。

    匹配输入须由独立真实检测评估器按同一阈值、同一坐标系生成。
    对每个区域单独累计 TP/FP；重复匹配视为 FP。
    """
    if relevant_gt_ids & irrelevant_gt_ids:
        raise ValueError("相关/无关 GT 实例 ID 不能重叠")
    if any(not isinstance(x, str) or not x for x in relevant_gt_ids | irrelevant_gt_ids):
        raise ValueError("GT 实例 ID 必须是非空字符串")
    for prediction in predictions:
        if not math.isfinite(prediction.score) or not 0 <= prediction.score <= 1:
            raise ValueError("检测 score 必须在 [0,1]")
        if not isinstance(prediction.in_instruction_region, bool):
            raise TypeError("预测区域归属必须由独立几何标注确定")

    def calculate(ids: set[str], in_region: bool) -> dict[str, float | int | None]:
        pool = sorted(
            (x for x in predictions if x.in_instruction_region == in_region),
            key=lambda x: x.score,
            reverse=True,
        )
        matched: set[str] = set()
        tp = 0
        fp = 0
        ap_sum = 0.0
        for pred in pool:
            match = pred.matched_gt_id
            if match in ids and match not in matched:
                tp += 1
                matched.add(match)
                ap_sum += tp / (tp + fp)
            else:
                fp += 1
        n = len(ids)
        return {
            "gt_count": n,
            "pred_count": len(pool),
            "tp": tp,
            "fp": fp,
            "fn": n - tp,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / n if n else None,
            "noninterpolated_ap": ap_sum / n if n else None,
        }

    return {
        "relevant_detection": calculate(relevant_gt_ids, True),
        "irrelevant_detection": calculate(irrelevant_gt_ids, False),
        "metric_name": "noninterpolated_AP_after_external_3D_matching",
        "warning": (
            "不等于 nuScenes 官方 mAP；需要真正的 3D box/距离阈值匹配结果，"
            "不能用分割地图类别代替车辆/行人检测实例。"
        ),
    }

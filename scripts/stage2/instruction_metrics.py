"""指令相关/无关分项指标：只在同一已验证 GT 像素网格计算。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import torch
from torch import Tensor

from .spatial_supervision import SpatialTarget, validate_spatial_target


def _confusion(gt: Tensor, pred: Tensor, region: Tensor) -> Tensor:
    """region 中按 GT 行、预测列统计 4x4 混淆矩阵。"""
    codes = gt[region].long() * 4 + pred[region].long()
    return torch.bincount(codes, minlength=16).reshape(4, 4).cpu()


def _class_iou(matrix: Tensor, class_id: int) -> float | None:
    tp = int(matrix[class_id, class_id])
    union = int(matrix[class_id].sum() + matrix[:, class_id].sum() - tp)
    return tp / union if union else None


def _summary(matrix: Tensor) -> dict[str, Any]:
    ious = {str(c): _class_iou(matrix, c) for c in range(4)}
    fg = [v for c, v in ious.items() if c != "0" and v is not None]
    return {
        "per_class_iou": ious,
        "foreground_miou_present_classes": sum(fg) / len(fg) if fg else None,
        "confusion_gt_rows_pred_cols": matrix.tolist(),
        "pixels": int(matrix.sum()),
    }


@dataclass
class InstructionMetrics:
    """跨帧累积严格分区的混淆矩阵与注意力实例定位计数。"""

    relevant: Tensor | None = None
    irrelevant: Tensor | None = None
    attention_tp: int = 0
    attention_fp: int = 0
    attention_fn: int = 0
    attention_target_mass: float = 0.0
    attention_total_mass: float = 0.0
    same_class_distractor_mass: float = 0.0
    count: int = 0
    matched_keys: set[str] | None = None
    annotation_fingerprints: dict[str, str] | None = None

    def update(
        self,
        predicted_labels: Tensor,
        attention_prob: Tensor,
        gt: Tensor,
        target: SpatialTarget,
    ) -> None:
        """加入一帧：GT/预测/实例/ROI必须全部在同一像素网格。"""
        valid = validate_spatial_target(target, gt)
        if predicted_labels.shape != gt.shape or predicted_labels.dtype != torch.long:
            raise ValueError("预测分割必须是同尺寸 long 类别索引")
        if predicted_labels.device != gt.device or not bool(
            ((predicted_labels >= 0) & (predicted_labels <= 3)).all()
        ):
            raise ValueError("预测分割类别/设备错误")
        if attention_prob.shape != gt.shape or attention_prob.device != gt.device:
            raise ValueError("Attention/GT 网格不一致")
        if not attention_prob.is_floating_point() or not bool(
            torch.isfinite(attention_prob).all()
        ) or not bool(((attention_prob >= 0) & (attention_prob <= 1)).all()):
            raise ValueError("Attention 需要 [0,1] 有限概率，不能传 logits")
        key = target.instruction_id
        if self.matched_keys is None:
            self.matched_keys = set()
        if key in self.matched_keys:
            raise ValueError("重复指令 ID，统计结果可能重复计数")
        self.matched_keys.add(key)
        if self.annotation_fingerprints is None:
            self.annotation_fingerprints = {}
        digest = hashlib.sha256()
        digest.update(str(target.target_class).encode("utf-8"))
        digest.update(target.instance_id.encode("utf-8"))
        for tensor in (gt, target.instance_mask, target.related_region):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self.annotation_fingerprints[key] = digest.hexdigest()

        relevant = target.related_region & valid
        irrelevant = (~target.related_region) & valid
        r = _confusion(gt, predicted_labels, relevant)
        u = _confusion(gt, predicted_labels, irrelevant)
        self.relevant = r if self.relevant is None else self.relevant + r
        self.irrelevant = u if self.irrelevant is None else self.irrelevant + u

        truth = target.instance_mask & valid
        predicted = (attention_prob >= 0.5) & valid
        self.attention_tp += int((predicted & truth).sum())
        self.attention_fp += int((predicted & ~truth).sum())
        self.attention_fn += int((~predicted & truth).sum())
        prob = attention_prob.float() * valid
        self.attention_target_mass += float(prob[truth].sum())
        self.attention_total_mass += float(prob.sum())
        distractors = (gt == target.target_class) & ~truth & valid
        self.same_class_distractor_mass += float(prob[distractors].sum())
        self.count += 1

    def report(self) -> dict[str, Any]:
        """同时报告相关/无关区域的分割 IoU 与目标注意力定位指标。"""
        if self.count == 0 or self.relevant is None or self.irrelevant is None:
            raise ValueError("没有可评估的有效指令样本")
        iou_denom = self.attention_tp + self.attention_fp + self.attention_fn
        return {
            "num_instructions": self.count,
            "instruction_ids": sorted(self.matched_keys or set()),
            "annotation_fingerprints": self.annotation_fingerprints,
            "related_region_segmentation": _summary(self.relevant),
            "unrelated_region_segmentation": _summary(self.irrelevant),
            "attention_target_instance_iou_at_0_5": (
                self.attention_tp / iou_denom if iou_denom else None
            ),
            "attention_target_mass_fraction": (
                self.attention_target_mass / self.attention_total_mass
                if self.attention_total_mass > 0 else None
            ),
            "attention_same_class_distractor_mass_fraction": (
                self.same_class_distractor_mass / self.attention_total_mass
                if self.attention_total_mass > 0 else None
            ),
            "metric_note": (
                "related_region 为独立标注 ROI；注意力对目标实例计算，"
                "分割为类别 IoU。只有 A/B 指令/场景/标注一致才能比较。"
            ),
        }


def compare_ab(baseline: dict[str, Any], enhanced: dict[str, Any]) -> dict[str, Any]:
    """校验样本配对后计算分割分项提升；未验证的结果不能比较。"""
    if baseline["instruction_ids"] != enhanced["instruction_ids"]:
        raise ValueError("A/B 指令样本不一致，禁止比较")
    if baseline["annotation_fingerprints"] != enhanced["annotation_fingerprints"]:
        raise ValueError("A/B 的 GT/目标实例/相关 ROI 不一致，禁止比较")
    fields = ("related_region_segmentation", "unrelated_region_segmentation")
    result = {}
    for field in fields:
        before = baseline[field]["per_class_iou"]
        after = enhanced[field]["per_class_iou"]
        result[field + "_per_class_delta"] = {
            key: after[key] - before[key]
            if after[key] is not None and before[key] is not None
            else None
            for key in before
        }
    for key in (
        "attention_target_instance_iou_at_0_5",
        "attention_target_mass_fraction",
    ):
        x, y = baseline[key], enhanced[key]
        result[key + "_delta"] = y - x if x is not None and y is not None else None
    return result

"""正式合同的独立损失包装；原作者双Head推理路径与参数完全不变。"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from scripts.stage2.spatial_annotations import metric_grid
from scripts.stage3.lg_bev_engineering_data import masks
from scripts.stage3.lg_bev_native import dual_head
from scripts.stage3.lg_bev_prototype import attention_supervision_loss
from scripts.stage3.lg_bev_supervision_space import AttentionSupervisionSpace


def map_roi(target: np.ndarray) -> np.ndarray:
    """0.15m栅格按像素中心欧氏距离扩张2m；空目标不伪造邻域。"""
    if not target.any():
        return np.zeros_like(target, dtype=bool)
    return distance_transform_edt(~target, sampling=0.15) <= 2.0


def pair_ranking(
    correct: Tensor, wrong: Tensor, target: Tensor, pair_target: Tensor, valid: Tensor
) -> Tensor:
    """只对真实目标集合的两个差集做对称margin=.5排序。"""
    a = target.bool() & ~pair_target.bool() & valid.bool()
    b = pair_target.bool() & ~target.bool() & valid.bool()
    parts = []
    if a.any():
        parts.append(F.relu(0.5 - correct[a] + wrong[a]).mean())
    if b.any():
        parts.append(F.relu(0.5 - wrong[b] + correct[b]).mean())
    return torch.stack(parts).mean() if parts else correct.sum() * 0


@contextmanager
def target_weighting(head: nn.Module, selected_gt: Tensor) -> Iterator[None]:
    """仅包装训练target权重，恢复原方法；Hungarian及Head.forward保持原样。"""
    if "_get_target_single" in head.__dict__:
        raise RuntimeError("拒绝嵌套loss包装")
    original = head._get_target_single

    def weighted(*args, **kwargs):
        values = list(original(*args, **kwargs))
        target = values[2]
        positive = values[4]
        if selected_gt.numel() and positive.numel():
            hits = (
                (target[positive, None, :] == selected_gt[None, :, :]).all(-1).any(-1)
            )
            indices = positive[hits]
            values[1] = values[1].clone()
            values[3] = values[3].clone()
            values[1][indices] *= 3.0
            values[3][indices] *= 3.0
        return tuple(values)

    head._get_target_single = weighted
    try:
        yield
    finally:
        del head._get_target_single


@contextmanager
def empty_positive_shape(head: nn.Module) -> Iterator[None]:
    """修正旧2D SamplingResult空GT的[0,4]回退，保持真实3D维度和全部负例。

    原作者forward与Hungarian不变，只包装训练loss中的采样结果；不添加框。
    """
    sampler = head.sampler
    if "sample" in sampler.__dict__:
        raise RuntimeError("拒绝嵌套空GT采样包装")
    original = sampler.sample

    def sample(assign_result, boxes, gt_boxes, *args, **kwargs):
        result = original(assign_result, boxes, gt_boxes, *args, **kwargs)
        if result.pos_gt_bboxes.numel() == 0 and result.pos_gt_bboxes.shape != (
            0,
            gt_boxes.shape[-1],
        ):
            result.pos_gt_bboxes = gt_boxes.new_empty((0, gt_boxes.shape[-1]))
        return result

    sampler.sample = sample
    try:
        yield
    finally:
        del sampler.sample


def detection_loss(
    model: nn.Module,
    prediction: dict,
    label: dict,
    query: dict,
    *,
    weighted: bool = True,
) -> Tensor:
    """作者原生loss聚合及目标matched positives ×3，未知/越界GT按原协议过滤。"""
    from contextlib import nullcontext
    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    head = model.pts_bbox_head
    device = prediction["all_cls_scores"].device
    boxes = label["boxes"].to(device)
    classes = label["labels"].long().to(device)
    keep = (
        (boxes[:, 0] > -51.2)
        & (boxes[:, 0] < 51.2)
        & (boxes[:, 1] > -51.2)
        & (boxes[:, 1] < 51.2)
        & (classes >= 0)
        & (classes < 10)
    )
    native = LiDARInstance3DBoxes(boxes[keep].clone(), box_dim=9)
    native.limit_yaw(offset=0.5, period=2 * np.pi)
    chosen = torch.zeros(len(boxes), dtype=torch.bool, device=device)
    if query["task"] == "object":
        chosen[query["selected"]] = True
    chosen = chosen[keep]
    gt = torch.cat((native.gravity_center, native.tensor[:, 3:]), -1)[chosen]
    context = target_weighting(head, gt) if weighted and gt.numel() else nullcontext()
    with empty_positive_shape(head), context:
        parts = head.loss([native], [classes[keep]], prediction)
    value = sum(v for key, v in parts.items() if "loss" in key)
    if not torch.isfinite(value):
        raise FloatingPointError("检测损失非有限")
    return value


def objective(
    model: nn.Module | None,
    correct: dict,
    wrong: dict,
    label: dict,
    query: dict,
    stage: str,
) -> tuple[Tensor, dict]:
    """GT仅在此损失端读取；A严格恒等，B共享同一enhanced BEV进原生双Head。"""
    device = correct["enhanced_bev"].device
    space = AttentionSupervisionSpace.from_calibration(query["task"], label["frame"])
    logits = space.project(correct["attention_logits"])
    wrong_logits = space.project(wrong["attention_logits"])
    supervision = masks(label, query, str(device))
    mask = attention_supervision_loss(
        logits,
        supervision["target"],
        supervision["distractor"],
        0.5,
        supervision["valid"],
    )
    paired = next(
        (q for q in label["queries"] if q["instruction_id"] == query["pair_id"]), None
    )
    rank = (
        pair_ranking(
            logits,
            wrong_logits,
            supervision["target"],
            masks(label, paired, str(device))["target"],
            supervision["valid"],
        )
        if paired
        else logits.sum() * 0
    )
    parts = dict(attention=mask["total"], ranking=rank)
    if stage == "A":
        total = mask["total"] + 0.1 * rank
    else:
        if model is None:
            raise ValueError("B阶段必须接入真实原生双Head")
        prediction = dual_head(model, correct["enhanced_bev"])
        if not torch.stack(
            [
                torch.isfinite(prediction[k]).all()
                for k in ("all_cls_scores", "all_bbox_preds", "seg_preds")
            ]
        ).all():
            raise FloatingPointError("Head预测非有限，不能由nan_to_num掩盖")
        native_bev = correct["source_bev"]
        with torch.no_grad():
            teacher = dual_head(model, native_bev)["seg_preds"].softmax(1)
        detection = detection_loss(model, prediction, label, query)
        gt = label["gt"][None].long().to(device)
        segmentation = model.pts_bbox_head.seg_loss(gt, prediction)["loss_seg"]
        if query["task"] == "map":
            roi = torch.from_numpy(
                map_roi(supervision["target"][0, 0].cpu().numpy())
            ).to(device)
            extra = (
                F.cross_entropy(prediction["seg_preds"], gt, reduction="none")[0][
                    roi
                ].mean()
                if roi.any()
                else prediction["seg_preds"].sum() * 0
            )
        else:
            _, xy, _ = metric_grid(label["frame"])
            roi = torch.from_numpy(
                ((xy[..., 0] > 0) if query["fb"] == "front" else (xy[..., 0] < 0))
                & ((xy[..., 1] > 0) if query["lr"] == "left" else (xy[..., 1] < 0))
            ).to(device)
            extra = prediction["seg_preds"].sum() * 0
        kl = F.kl_div(
            prediction["seg_preds"].log_softmax(1), teacher, reduction="none"
        ).sum(1)[0]
        protect = kl[~roi].mean() if (~roi).any() else kl.sum() * 0
        parts.update(
            detection=detection,
            segmentation=segmentation,
            map_roi=extra,
            protect=protect,
        )
        total = (
            detection
            + segmentation
            + 0.2 * mask["total"]
            + 0.1 * rank
            + 0.5 * extra
            + 0.1 * protect
        )
    if not torch.isfinite(total):
        raise FloatingPointError("总损失非有限")
    parts["total"] = total
    return total, parts

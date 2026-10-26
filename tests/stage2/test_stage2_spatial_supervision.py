"""空间指令监督、分项指标及 3D 检测匹配摘要的 CPU 测试。"""

from __future__ import annotations

import pytest
import torch

from scripts.stage2.instruction_detection_metrics import (
    PredictionMatch,
    region_average_precision,
)
from scripts.stage2.instruction_metrics import InstructionMetrics, compare_ab
from scripts.stage2.spatial_supervision import (
    SpatialTarget,
    spatial_attention_loss,
    validate_spatial_target,
)


def example():
    """类别 1 有两个对象，但指令仅指定左上方那一个。"""
    gt = torch.tensor([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 2, 0]], dtype=torch.long)
    target_mask = torch.zeros_like(gt, dtype=torch.bool)
    target_mask[0, 0] = True
    region = torch.zeros_like(gt, dtype=torch.bool)
    region[:2, :2] = True
    target = SpatialTarget(
        instruction_id="sampleA_prompt1",
        target_class=1,
        instance_id="verified_divider_01",
        instance_mask=target_mask,
        related_region=region,
        annotation_source="manual_verified",
    )
    return gt, target


def test_target_cannot_fall_back_to_whole_class():
    gt, target = example()
    validate_spatial_target(target, gt)
    assert int(target.instance_mask.sum()) == 1
    assert int((gt == 1).sum()) == 2
    with pytest.raises(ValueError, match="类别掩码"):
        bad = SpatialTarget(
            target.instruction_id, 1, target.instance_id, gt == 1,
            target.related_region, "class_mask"
        )
        validate_spatial_target(bad, gt)


def test_spatial_loss_penalizes_same_class_distractor_and_backward():
    gt, target = example()
    logits = torch.zeros(1, 1, 3, 4, requires_grad=True)
    loss = spatial_attention_loss(logits, gt[None], [target])
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad[0, 0, 0, 0] < 0  # 目标像素应该被鼓励上升
    assert logits.grad[0, 0, 0, 3] > 0  # 同类干扰物必须被抑制
    with pytest.raises(ValueError, match="未对齐"):
        spatial_attention_loss(logits[:, :, :2], gt[None], [target])


def test_invalid_or_empty_annotation_rejected():
    gt, target = example()
    with pytest.raises(ValueError, match="目标实例"):
        empty = SpatialTarget(
            target.instruction_id, 1, "empty", torch.zeros_like(target.instance_mask),
            target.related_region, "manual_verified"
        )
        validate_spatial_target(empty, gt)
    with pytest.raises(ValueError, match="其他类别"):
        bad_instance = target.instance_mask.clone()
        bad_instance[2, 2] = True
        bad = SpatialTarget(
            target.instruction_id, 1, "wrong", bad_instance,
            target.related_region, "manual_verified"
        )
        validate_spatial_target(bad, gt)


def test_relevant_segmentation_and_instance_attention_split():
    gt, target = example()
    original = InstructionMetrics()
    improved = InstructionMetrics()
    seg_bad = gt.clone()
    seg_bad[0, 0] = 0  # 仅破坏指令相关 ROI
    seg_fixed = gt.clone()
    att_bad = torch.zeros_like(gt, dtype=torch.float32)
    att_bad[0, 3] = 0.9  # 注意力错误地放在同类干扰物上
    att_good = torch.zeros_like(gt, dtype=torch.float32)
    att_good[0, 0] = 0.9
    original.update(seg_bad, att_bad, gt, target)
    improved.update(seg_fixed, att_good, gt, target)
    a, b = original.report(), improved.report()
    assert a["related_region_segmentation"]["per_class_iou"]["1"] == 0.0
    assert b["related_region_segmentation"]["per_class_iou"]["1"] == 1.0
    assert a["unrelated_region_segmentation"] == b["unrelated_region_segmentation"]
    delta = compare_ab(a, b)
    assert delta["related_region_segmentation_per_class_delta"]["1"] == 1.0
    assert b["attention_target_instance_iou_at_0_5"] == 1.0
    assert a["attention_target_instance_iou_at_0_5"] == 0.0
    assert (
        a["attention_same_class_distractor_mass_fraction"]
        > b["attention_same_class_distractor_mass_fraction"]
    )
    with pytest.raises(ValueError, match="重复指令"):
        improved.update(seg_fixed, att_good, gt, target)


def test_detection_ap_requires_independent_matching():
    records = [
        PredictionMatch(0.95, "g1", True),
        PredictionMatch(0.90, "g1", True),  # 重复预测只能算 FP
        PredictionMatch(0.85, None, True),
        PredictionMatch(0.80, "g2", False),
    ]
    summary = region_average_precision(
        records, relevant_gt_ids={"g1"}, irrelevant_gt_ids={"g2"}
    )
    assert summary["relevant_detection"]["tp"] == 1
    assert summary["relevant_detection"]["fp"] == 2
    assert summary["irrelevant_detection"]["noninterpolated_ap"] == 1.0
    assert summary["metric_name"] != "nuscenes_official_mAP"


def test_opt_in_loss_integration_preserves_old_mode():
    """在用户本地已存在的模块基础上验证 opt-in，不需要作者模型。"""
    from scripts.stage2.multitask_training import LossConfig, supervised_attention_loss

    gt, target = example()
    logits = torch.zeros((1, 1, 3, 4), requires_grad=True)
    classes = torch.tensor([1], dtype=torch.long)
    old = supervised_attention_loss(logits, gt[None], classes, LossConfig())
    new = supervised_attention_loss(
        logits, gt[None], classes, LossConfig(), spatial_targets=[target]
    )
    assert old.isfinite() and new.isfinite()
    assert old.item() != new.item()
    with pytest.raises(ValueError, match="指令类别"):
        supervised_attention_loss(
            logits, gt[None], torch.tensor([2]), LossConfig(), spatial_targets=[target]
        )

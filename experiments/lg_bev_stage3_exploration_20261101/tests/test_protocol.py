"""统一计分端到端合成验收、坐标投影与新优化器暂停重放。"""

import copy
import io

import numpy as np
import pytest
import torch

from core import fresh_optimizer, load_adapter, seed_cpu, tensor_sha
from evaluation import EvaluationState, queries
from synthetic import report, records


def test_three_groups_same_protocol_synthetic():
    result = report()
    groups = result["metrics"]["groups"]
    for metric in (
        "Relevant_Detection_AP",
        "Irrelevant_Detection_AP",
        "Relevant_Segmentation_IoU",
        "Irrelevant_Segmentation_IoU",
    ):
        assert groups["baseline/correct"][metric] == groups["a_only/correct"][metric]
        assert groups["baseline/correct"][metric] == groups["enhanced/correct"][metric]
        field = "mAP" if "Detection" in metric else "mIoU"
        assert groups["baseline/correct"][metric][field] == pytest.approx(1)
    assert groups["baseline/correct"]["Attention"]["map"]["positive_macro_IoU"] is None
    assert groups["a_only/correct"]["Attention"]["map"]["mean_logit_gap"] == 2
    assert result["metrics"]["A9562_minus_A0"]["map"]["positive_macro_IoU"] == 0
    assert groups["a0/correct"]["Relevant_Detection_AP"]["status"] == "NOT_EVALUABLE"
    assert len(result["metrics"]["per_scene"]) == 24


def test_evaluation_save_continue_identical():
    rows = records()
    continuous, split = EvaluationState(), EvaluationState()
    for row in rows:
        continuous.add(row)
    for row in rows[:8]:
        split.add(row)
    stream = io.BytesIO()
    torch.save(split, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    for row in rows[8:]:
        restored.add(row)
    assert continuous.result() == restored.result()


@pytest.mark.parametrize(
    "rotation,translation",
    [
        ([1, 0, 0, 0], [0, 0, 0]),
        ([0.7071067811865476, 0, 0, 0.7071067811865476], [1, 2, 0]),
    ],
)
def test_metric_ramp_projection_axes_scale_and_alignment(rotation, translation):
    from scripts.stage2.spatial_annotations import metric_grid
    from scripts.stage3.lg_bev_supervision_space import (
        AttentionSupervisionSpace,
        bilinear_adjoint_check,
    )

    frame = {"lidar2ego_rotation": rotation, "lidar2ego_translation": translation}
    grid, ego, lidar = metric_grid(frame)
    assert grid.shape == (1, 200, 400, 2) and bool((grid.abs() < 1).all())
    assert ego[0, 0, 0] == pytest.approx(-29.925)
    assert ego[0, 0, 1] == pytest.approx(-14.925)
    y, x = torch.meshgrid(
        torch.arange(150, dtype=torch.float64),
        torch.arange(150, dtype=torch.float64),
        indexing="ij",
    )
    ramp = 3 * ((x + 0.5) * 102.4 / 150 - 51.2) + 7 * ((y + 0.5) * 102.4 / 150 - 51.2)
    space = AttentionSupervisionSpace.from_calibration("map", frame)
    output = space.project(ramp[None, None])[0, 0].numpy()
    np.testing.assert_allclose(
        output, 3 * lidar[..., 0] + 7 * lidar[..., 1], atol=1e-10
    )
    assert bilinear_adjoint_check(space, (70, 130))["pass_check"]
    assert AttentionSupervisionSpace.from_calibration("object", frame).project(
        ramp[None, None]
    ).shape == (1, 1, 150, 150)


def test_query_selection_is_fixed_and_keeps_empty_targets():
    label = {"queries": []}
    for task, classes in (("map", (1, 2)), ("object", (0, 8))):
        for cls in classes:
            for i in range(3):
                label["queries"].append(
                    {
                        "task": task,
                        "target_class": cls,
                        "qualifier": "all",
                        "instruction_id": f"{task}_{cls}_{i}",
                        "selected": [],
                    }
                )
    assert queries(label, 0) == queries(label, 2)
    assert queries(label, 1) == queries(label, 3)
    assert all(not q["selected"] for q in queries(label, 0))
    missing = copy.deepcopy(label)
    missing["queries"] = [q for q in missing["queries"] if q["task"] == "map"]
    with pytest.raises(ValueError):
        queries(missing, 0)


def test_cpu_adapter_optimizer_rng_roundtrip(config):
    """验证新状态自身重放，明确不是原GPU训练恢复等价性证据。"""
    adapter = load_adapter(config)
    optimizer, scheduler = fresh_optimizer(adapter, 64)
    seed_cpu(20261101)

    def step(module, opt, sched):
        bev, text, position = (
            torch.randn(1, 256, 4, 4),
            torch.randn(1, 3072),
            torch.randn(1, 5, 4, 4),
        )
        detail = module(bev, text, position)
        (
            detail["enhanced_bev"].square().mean()
            + detail["attention_logits"].square().mean()
        ).backward()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)

    step(adapter, optimizer, scheduler)
    stream = io.BytesIO()
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": torch.get_rng_state(),
        },
        stream,
    )
    step(adapter, optimizer, scheduler)
    expected = tensor_sha(adapter.state_dict())
    stream.seek(0)
    state = torch.load(stream, weights_only=False)
    restored = load_adapter(config)
    opt, sched = fresh_optimizer(restored, 64)
    restored.load_state_dict(state["adapter"])
    opt.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"])
    step(restored, opt, sched)
    assert tensor_sha(restored.state_dict()) == expected
    assert sched.state_dict() == scheduler.state_dict()


@pytest.mark.parametrize(
    "damage", [None, "budget", "partial", "cursor", "step", "missing", "nan", "frozen"]
)
def test_independent_training_boundary_validation(config, damage):
    from worker import validate_training_state

    adapter = load_adapter(config)
    optimizer, scheduler = fresh_optimizer(adapter, 64)
    for parameter in adapter.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler.step()
    state = {
        "frozen_sha256": ("head", "encoder"),
        "tokens": list(range(32)),
        "budget": 64,
        "accumulated": 0,
        "updates": 1,
        "cursor": 4,
        "epoch": 0,
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if damage == "budget":
        state["budget"] = 256
    elif damage == "partial":
        state["accumulated"] = 1
    elif damage == "cursor":
        state["cursor"] = 5
    elif damage == "step":
        next(iter(state["optimizer"]["state"].values()))["step"] = torch.tensor(2.0)
    elif damage == "missing":
        state["optimizer"]["state"].popitem()
    elif damage == "nan":
        next(iter(state["adapter"].values())).fill_(float("nan"))
    elif damage == "frozen":
        state["frozen_sha256"] = ("changed", "encoder")
    if damage:
        with pytest.raises(ValueError):
            validate_training_state(state, list(range(32)), 64, 8, ("head", "encoder"))
    else:
        validate_training_state(state, list(range(32)), 64, 8, ("head", "encoder"))


def test_preflight_32_frames_real_prefix_metadata(config):
    from core import selected_rows
    from worker import preflight_rows

    rows = preflight_rows(selected_rows(config, "train"))
    assert len(rows) == 32
    assert len({r["scene_token"] for r in rows}) == 3
    assert [
        sum(r["scene_token"] == rows[i]["scene_token"] for r in rows)
        for i in (0, 12, 22)
    ] == [12, 10, 10]


def test_real_evaluation_driver_with_cpu_stubs(monkeypatch):
    """真实驱动+真实投影/GT过滤，模型与文本编码仅用CPU替身，无GT文件读取。"""
    from types import SimpleNamespace
    from evaluation import evaluate_frame, aggregate, ground_truth_coverage
    from scripts.stage3 import lg_bev_native, lg_bev_engineering_train

    frame = {
        "lidar2ego_rotation": [1, 0, 0, 0],
        "lidar2ego_translation": [0, 0, 0],
        "scene_token": "cpu_scene",
    }
    label = {
        "frame": frame,
        "gt": torch.ones(200, 400, dtype=torch.long),
        "object_masks": torch.zeros(1, 150, 150, dtype=torch.bool),
        "map_masks": torch.ones(1, 200, 400, dtype=torch.bool),
        "labels": torch.tensor([0]),
        "map_classes": torch.tensor([1]),
        "eval_annotation_tokens": ["target"],
        "eval_boxes": torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]),
        "eval_labels": torch.tensor([0]),
        "eval_num_points": torch.tensor([1]),
        "queries": [],
    }
    label["object_masks"][0, 74:77, 74:77] = True
    for task, cls in (("object", 0), ("map", 1)):
        label["queries"].append(
            {
                "task": task,
                "target_class": cls,
                "qualifier": "all",
                "instruction_id": task,
                "text": task,
                "wrong_text": task + " wrong",
                "fb": "front",
                "lr": "left",
                "pair_id": "paired",
                "selected": [0],
                "target_instance_ids": ["target"],
            }
        )
    bev = torch.zeros(1, 256, 150, 150)
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        validate_label=lambda _: label,
        get=lambda _: bev,
        rows={"cpu": {"log_token": "cpu_log"}},
        model=None,
        encoder=None,
    )
    calls = []

    def head(model, value):
        calls.append(value.data_ptr())
        semantic = torch.zeros(1, 4, 200, 400)
        semantic[:, 1] = 1
        return {
            "all_cls_scores": torch.zeros(1, 1, 1, 10),
            "all_bbox_preds": torch.zeros(1, 1, 1, 10),
            "seg_preds": semantic,
        }

    def encode(encoder, texts):
        return {text: torch.ones(1, 3072) for text in texts}

    def adapter(value, text, position):
        return {"enhanced_bev": value, "attention_logits": position[:, :1]}

    monkeypatch.setattr(lg_bev_native, "dual_head", head)
    monkeypatch.setattr(
        lg_bev_native,
        "decode",
        lambda *_: {
            "boxes": label["eval_boxes"].numpy(),
            "labels": np.array([0]),
            "scores": np.array([0.9]),
        },
    )
    monkeypatch.setattr(lg_bev_engineering_train, "encode_texts", encode)
    coverage = ground_truth_coverage(runtime, [{"sample_token": "cpu"}])
    assert not calls and coverage["frames"] == 1
    assert coverage["detection_GT"]["relevant"]["car"] == 1
    assert (
        ground_truth_coverage(runtime, [{"sample_token": "cpu"}], lambda: True) is None
    )
    rows = evaluate_frame(
        runtime, {"a_only": adapter, "enhanced": adapter, "a0": adapter}, "cpu", 0
    )
    assert len(rows) == 8 and len(calls) == 13
    groups = aggregate(rows)["groups"]
    assert (
        groups["baseline/correct"]["Relevant_Detection_AP"]
        == groups["enhanced/wrong"]["Relevant_Detection_AP"]
    )
    assert groups["a_only/correct"]["Attention"]["map"]["queries"] == 1

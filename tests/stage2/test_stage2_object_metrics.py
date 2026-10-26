"""独立对象指令和 AP 的小型数学契约，不作为真实实验成绩。"""

import numpy as np
import pytest

from scripts.stage2.semantic_instruction import parse_instruction
from scripts.stage2.spatial_detection import ap_metrics, parse_object_instruction


def test_object_and_map_schemas_remain_separate():
    assert parse_object_instruction("Find cars on the front left")["class_id"] == 0
    assert parse_object_instruction("Find pedestrians on the rear right")["class_id"] == 8
    assert parse_instruction("Find crosswalk on the front left").targets == ("ped_crossing",)
    with pytest.raises(ValueError):
        parse_object_instruction("Find crosswalk on the front left")
    with pytest.raises(ValueError):
        parse_object_instruction("Find cars somewhere")


def test_ap_sorts_scores_and_counts_duplicate_predictions_as_fp(tmp_path):
    gt = np.array([[5., 5., 0., 1., 1., 1., 0.],
                   [10., 5., 0., 1., 1., 1., 0.],
                   [-5., 5., 0., 1., 1., 1., 0.]])
    boxes = np.array([[20., 20., 0., 1., 1., 1., 0.], *gt[:1], *gt[:1], *gt[1:]])
    arrays = {"gt_boxes": gt, "gt_labels": np.zeros(3, dtype=int)}
    for branch in ("original", "historical", "spatial", "incorrect"):
        arrays[branch + "_boxes"] = boxes
        arrays[branch + "_scores"] = np.array([.99, .9, .8, .7, .6])
        arrays[branch + "_labels"] = np.zeros(5, dtype=int)
    np.savez(tmp_path / "prediction.npz", **arrays)
    row = {"instruction_id": "synthetic_math_only", "prediction_file": "prediction.npz",
           "prompt": parse_object_instruction("Find cars on the front left"),
           "gt_annotation_ids": ["one", "two", "three"],
           "relevant_gt_indices": [0, 1], "irrelevant_gt_indices": [2],
           "lidar2ego_rotation": [1., 0., 0., 0.], "lidar2ego_translation": [0., 0., 0.]}
    metrics, matches = ap_metrics([row], tmp_path)
    assert metrics["spatial"]["relevant"]["macro_ap"] == .5
    assert metrics["spatial"]["irrelevant"]["macro_ap"] == 1.
    assert metrics["spatial"]["relevant"]["classes"]["0"]["fp"] == 2
    assert metrics["spatial"]["relevant"]["classes"]["8"]["ap"] is None
    assert any(not record["tp"] for record in matches)

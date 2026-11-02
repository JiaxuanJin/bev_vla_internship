"""合成区域计分、空目标、边界及与安装SDK的数值对照。"""

import numpy as np
import pytest

from eval_metrics import (
    AttentionMetrics,
    DetectionAP,
    SegmentationIoU,
    quadrant,
    sdk_ap,
    segmentation_regions,
)


def box(identity="a", xy=(1, 1), cls="car", score=None):
    result = {"id": identity, "xy": xy, "class": cls}
    if score is not None:
        result["score"] = score
    return result


@pytest.mark.parametrize(
    "xy,expected",
    [
        ((0, 0), True),
        ((0, -1), False),
        ((-1, 0), False),
        ((1, 1), True),
        ((-1, -1), False),
    ],
)
def test_boundary_ownership(xy, expected):
    assert bool(quadrant([xy], "front", "left")[0]) is expected
    assert (
        sum(
            int(quadrant([xy], fb, lr)[0])
            for fb in ("front", "back")
            for lr in ("left", "right")
        )
        == 1
    )


@pytest.mark.parametrize("distance,tp", [(0.499, 1), (0.5, 0), (0.501, 0)])
def test_strict_distance_threshold(distance, tp):
    metric = DetectionAP(("car",))
    metric.add(
        "q", [box()], [box(xy=(1 + distance, 1), score=0.9)], "car", "front", "left"
    )
    assert metric.result()["relevant"]["classes"]["car"]["0.5"]["TP"] == tp


def test_duplicate_predictions_match_once():
    metric = DetectionAP(("car",))
    metric.add("q", [box()], [box(score=0.9), box(score=0.8)], "car", "front", "left")
    value = metric.result()["relevant"]["classes"]["car"]["0.5"]
    assert (value["TP"], value["FP"], value["GT"]) == (1, 1, 1)


@pytest.mark.parametrize("kind", ["duplicate_gt", "duplicate_query", "nonfinite"])
def test_invalid_detection_rejected(kind):
    metric = DetectionAP(("car",))
    truth = [box(), box()] if kind == "duplicate_gt" else [box()]
    predictions = [box(score=float("nan"))] if kind == "nonfinite" else []
    if kind == "duplicate_query":
        metric.add("q", truth, [], "car", "front", "left")
    with pytest.raises(ValueError):
        metric.add("q", truth, predictions, "car", "front", "left")


def test_no_predictions_positive_gt_zero_ap():
    metric = DetectionAP(("car",))
    metric.add("q", [box()], [], "car", "front", "left")
    assert metric.result()["relevant"]["mAP"] == 0
    assert metric.result()["irrelevant"]["mAP"] is None


def test_empty_gt_fp_retained_and_reduces_global_ap():
    metric = DetectionAP(("car",))
    metric.add("empty", [], [box(score=0.99)], "car", "front", "left")
    assert metric.result()["relevant"]["mAP"] is None
    metric.add("positive", [box()], [box(score=0.9)], "car", "front", "left")
    assert 0 < metric.result()["relevant"]["mAP"] < 1
    assert metric.result()["relevant"]["empty_GT_predictions"] == 1


def test_irrelevant_independent_not_global_subtraction():
    truth = [box(), box("b", (-1, -1))]
    metric = DetectionAP(("car",))
    metric.add("q", truth, [box(score=0.9)], "car", "front", "left")
    result = metric.result()
    assert result["relevant"]["mAP"] == pytest.approx(1)
    assert result["irrelevant"]["mAP"] == 0
    for region in ("relevant", "irrelevant"):
        assert result[region]["classes"]["car"]["0.5"]["GT"] == 1


def test_cross_boundary_cannot_match_even_if_near():
    metric = DetectionAP(("car",))
    metric.add(
        "q",
        [box(xy=(0.01, 1))],
        [box(xy=(-0.01, 1), score=0.9)],
        "car",
        "front",
        "left",
    )
    result = metric.result()
    assert result["relevant"]["mAP"] == 0
    assert result["irrelevant"]["classes"]["car"]["0.5"]["FP"] == 1


def test_wrong_class_does_not_match():
    metric = DetectionAP()
    metric.add("q", [box()], [box(cls="pedestrian", score=0.9)], "car", "front", "left")
    assert metric.result()["relevant"]["classes"]["car"]["0.5"]["TP"] == 0
    assert metric.result()["relevant"]["mAP"] is None


@pytest.mark.parametrize("hits", [[1], [0, 1], [1, 0, 1], [0, 0], [1, 1, 0, 1]])
def test_ap_matches_installed_nuscenes_sdk(hits):
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.utils import center_distance
    from nuscenes.eval.detection.algo import accumulate, calc_ap
    from nuscenes.eval.detection.data_classes import DetectionBox

    gt, pred = EvalBoxes(), EvalBoxes()

    def native(token, xy, score=-1):
        return DetectionBox(
            sample_token=token,
            translation=(*xy, 0),
            size=(1, 1, 1),
            rotation=(1, 0, 0, 0),
            velocity=(0, 0),
            detection_name="car",
            detection_score=float(score),
            attribute_name="",
            num_pts=1,
        )

    for i, hit in enumerate(hits):
        token = str(i)
        gt.add_boxes(token, [native(token, (1, 1))])
        pred.add_boxes(
            token, [native(token, (1, 1) if hit else (30, 30), float(1 - i / 10))]
        )
    expected = calc_ap(accumulate(gt, pred, "car", center_distance, 0.5), 0.1, 0.1)
    assert sdk_ap(hits, len(hits)) == pytest.approx(expected, abs=1e-12)


def test_segmentation_regions_cover_valid_without_overlap():
    target = np.zeros((40, 40), bool)
    target[0, 0] = True
    valid = np.ones_like(target)
    valid[-1] = False
    regions = segmentation_regions(target, valid)
    assert np.array_equal(regions["relevant"] | regions["irrelevant"], valid)
    assert not (regions["relevant"] & regions["irrelevant"]).any()
    assert regions["relevant"][0, 13] and not regions["relevant"][0, 14]


def test_segmentation_independent_full_semantics():
    target = np.zeros((40, 40), bool)
    target[0, 0] = True
    gt = np.ones((40, 40), int)
    gt[:, 20:] = 2
    prediction = gt.copy()
    prediction[:, 20:] = 0
    metric = SegmentationIoU()
    metric.add("q", gt, prediction, target, np.ones_like(target), 1)
    result = metric.result()
    assert result["relevant"]["classes"]["1"]["IoU"] == 1
    assert result["irrelevant"]["classes"]["2"]["IoU"] == 0
    assert result["irrelevant"]["mIoU"] == 0.5
    assert sum(result[r]["classes"]["1"]["pixels"] for r in result) == gt.size


def test_empty_segmentation_target_and_missing_positive():
    metric = SegmentationIoU()
    gt, prediction = np.zeros((4, 4), int), np.ones((4, 4), int)
    metric.add("q", gt, prediction, gt.astype(bool), np.ones_like(gt, bool), 1)
    result = metric.result()
    assert result["relevant"]["classes"]["1"]["pixels"] == 0
    assert result["irrelevant"]["classes"]["1"]["union"] == 16
    assert result["irrelevant"]["classes"]["1"]["IoU"] is None
    assert result["irrelevant"]["status"] == "NOT_EVALUABLE"


@pytest.mark.parametrize("damage", ["shape", "class", "duplicate"])
def test_segmentation_invalid(damage):
    metric = SegmentationIoU()
    gt = np.ones((2, 2), int)
    pred = np.ones((3, 2), int) if damage == "shape" else gt.copy()
    if damage == "class":
        pred[0, 0] = 4
    if damage == "duplicate":
        metric.add("q", gt, pred, gt, gt, 1)
    with pytest.raises(ValueError):
        metric.add("q", gt, pred, gt, gt, 1)


def test_attention_fixed_gt_threshold_and_pairing():
    metric = AttentionMetrics()
    target = np.array([[1, 0], [0, 0]], bool)
    correct = np.array([[0.0, -2], [-2, -2]])
    wrong = -correct - 1
    metric.add("q", "map", correct, wrong, target, np.ones_like(target))
    row = metric.rows[0]
    assert row["IoU"] == 1 and row["wrong_IoU"] == 0
    assert row["logit_gap"] == 1
    assert row["peak_in_target"] is True
    metric.add(
        "unpaired", "object", correct, wrong, target, np.ones_like(target), paired=False
    )
    assert metric.rows[1]["logit_gap"] is None


def test_attention_empty_activation_and_macro_vs_pooled():
    metric = AttentionMetrics()
    valid = np.ones((2, 2), bool)
    metric.add("empty", "map", np.zeros((2, 2)), np.zeros((2, 2)), ~valid, valid)
    assert metric.result()["map"]["empty_active_fraction"] == 1
    metric.add(
        "small",
        "map",
        np.array([[1, -1], [-1, -1]]),
        np.zeros((2, 2)),
        np.array([[1, 0], [0, 0]]),
        valid,
    )
    metric.add("large", "map", -np.ones((2, 2)), np.zeros((2, 2)), valid, valid)
    result = metric.result()["map"]
    assert result["positive_macro_IoU"] == 0.5
    assert result["positive_pooled_IoU"] == 0.2
    assert result["positive_queries"] == 2 and result["empty_queries"] == 1


@pytest.mark.parametrize("damage", ["nan", "empty_valid", "duplicate"])
def test_attention_invalid(damage):
    metric = AttentionMetrics()
    logits, valid = np.zeros((2, 2)), np.ones((2, 2), bool)
    if damage == "nan":
        logits[0, 0] = np.nan
    elif damage == "empty_valid":
        valid[:] = False
    else:
        metric.add("q", "map", logits, logits, valid, valid)
    with pytest.raises(ValueError):
        metric.add("q", "map", logits, logits, valid, valid)

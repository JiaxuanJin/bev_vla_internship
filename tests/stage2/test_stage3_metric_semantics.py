"""保护语义邻域/实例注意力的定义边界，避免把正确同类实例算成语义FP。"""
import numpy as np
import pytest
from collections import Counter
from scripts.stage2.stage3_metric_review import neighborhood,binary_counts,semantic_scores

def test_other_true_same_class_instance_is_correct_semantics():
    truth=np.zeros((7,7),bool);truth[3,3]=True;truth[3,4]=True
    selected=np.zeros_like(truth);selected[3,3]=True
    roi=neighborhood(selected,radius_m=1.5,pixel_m=1.)
    counts=binary_counts(truth,truth,roi)
    assert counts['gt_pixels']==counts['predicted_pixels']==counts['intersection']==2
    assert semantic_scores(counts)['iou']==1.

def test_euclidean_physical_neighborhood_and_empty_rejection():
    mask=np.zeros((7,7),bool);mask[3,3]=True
    roi=neighborhood(mask,radius_m=1.5,pixel_m=1.)
    assert roi.sum()==9 and not roi[3,5] and roi[2,2]
    with pytest.raises(ValueError,match='NOT_EVALUABLE'):neighborhood(np.zeros_like(mask))

def test_semantic_iou_pools_counts_and_reports_absent_denominator():
    count=Counter(intersection=1,union=2,gt_pixels=1,predicted_pixels=2,roi_pixels=2)
    count.update(intersection=1,union=10,gt_pixels=10,predicted_pixels=1,roi_pixels=10)
    assert semantic_scores(count)['iou']==pytest.approx(2/12)
    assert semantic_scores(Counter())['iou'] is None

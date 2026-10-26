"""第二轮分组损失与错误方位对比的数学契约，不充当实验收益。"""

import torch

from scripts.stage2.bev_fusion import LanguageBEVFusion
from scripts.stage2.spatial_annotations import metric_grid, project_attention
from scripts.stage2.spatial_v2_model import (
    ContrastFusion, balanced_instance_loss, contrast_loss,
)


def test_sparse_positive_gradient_is_not_diluted_by_image_area():
    logits = torch.zeros(1, 1, 10, 10, requires_grad=True)
    gt = torch.zeros(10, 10, dtype=torch.long)
    gt[0, 0] = 1
    gt[0, 1] = 1
    target = torch.zeros_like(gt, dtype=torch.bool)
    target[0, 0] = True
    balanced_instance_loss(logits, gt, target, 1).backward()
    assert logits.grad[0, 0, 0, 0] < -.2
    assert logits.grad[0, 0, 0, 1] > 0
    assert torch.isfinite(logits.grad).all()


def test_contrast_gradient_increases_correct_wrong_gap():
    correct = torch.zeros(1, 1, 2, 2, requires_grad=True)
    wrong = torch.zeros_like(correct, requires_grad=True)
    target = torch.tensor([[True, False], [False, False]])
    loss, gap = contrast_loss(correct, wrong, target)
    loss.backward()
    assert gap == 0
    assert correct.grad[0, 0, 0, 0] < 0 < wrong.grad[0, 0, 0, 0]
    assert correct.grad[0, 0, 1, 1] == 0


def test_position_features_use_calibration_and_ground_plane():
    state = LanguageBEVFusion(768).state_dict()
    fusion = ContrastFusion(state)
    row = {"lidar2ego_rotation": [.70710678, 0., 0., -.70710678],
           "lidar2ego_translation": [1., 0., 1.8]}
    fusion.set_calibration(row)
    grid, ego, _ = metric_grid(row)
    projected = project_attention(fusion.position_features[:, :2], grid)[0]
    torch.testing.assert_close(projected[0], torch.tensor(ego[..., 0] / 30).float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(projected[1], torch.tensor(ego[..., 1] / 15).float(), atol=1e-5, rtol=1e-5)

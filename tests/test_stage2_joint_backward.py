"""联合反传验收的错误拒绝测试；真实模型结果另由 joint-backward 生成。"""

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2.joint_runtime import _first_tensor, _gradient_stats


def test_gradient_stats_requires_nonzero_finite_gradients():
    value = torch.nn.Parameter(torch.tensor(2.0))
    result = _gradient_stats(value.square(), [("value", value)])
    assert result["global_l2"] == 4 and result["all_finite"]
    with pytest.raises(RuntimeError, match="非零"):
        _gradient_stats(value * 0, [("value", value)])
    with pytest.raises(RuntimeError, match="有限标量"):
        _gradient_stats(value * float('inf'), [("value", value)])


def test_gt_extraction_does_not_generate_or_resize_labels():
    gt = torch.arange(6).reshape(2, 3)
    assert _first_tensor([[gt]]) is gt
    with pytest.raises(TypeError):
        _first_tensor({"labels_missing": True})

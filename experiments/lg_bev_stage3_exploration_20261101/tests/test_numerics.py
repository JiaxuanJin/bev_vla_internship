"""执行前固定的跨会话零容差比较，含损坏与极小偏差。"""

import copy

import numpy as np
import pytest
import torch

from numerics import POLICY, compare_exact, require_comparison


@pytest.mark.parametrize("damage", [None, "tiny", "nan", "dtype", "keys", "rng"])
def test_exact_boundary_comparison(damage):
    expected = {
        "adapter": torch.tensor([1.0, 0.0]),
        "optimizer": {0: {"step": torch.tensor(4.0)}},
        "rng": (np.array([42], dtype=np.uint32), 4),
        "losses": {"detection": 1.0},
    }
    actual = copy.deepcopy(expected)
    if damage == "tiny":
        actual["losses"]["detection"] += 1e-12
    elif damage == "nan":
        actual["adapter"][0] = float("nan")
    elif damage == "dtype":
        actual["adapter"] = actual["adapter"].double()
    elif damage == "keys":
        actual["optimizer"] = {}
    elif damage == "rng":
        actual["rng"][0][0] = 43
    result = compare_exact(expected, actual)
    assert POLICY["witness_atol"] == POLICY["witness_rtol"] == 0
    if damage:
        with pytest.raises(ValueError):
            require_comparison(result)
    else:
        require_comparison(result)
        assert result["max_tensor_abs_error"] == 0

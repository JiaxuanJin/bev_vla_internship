"""预检前固定的零容差比较；不增加任何优化器更新。"""

from __future__ import annotations

import math

import numpy as np
import torch

from core import require

POLICY = {
    "state_atol": 0.0,
    "state_rtol": 0.0,
    "witness_atol": 0.0,
    "witness_rtol": 0.0,
    "definition": "相同update4状态的同一下一帧Loss、全参数梯度和输入摘要完全一致",
}


def compare_exact(expected, actual):
    """逐叶比较CPU可序列化状态，保留差异路径和最大绝对差。"""
    differences, leaves = [], [0]
    maximum = [0.0]

    def visit(left, right, path):
        if isinstance(left, torch.Tensor):
            leaves[0] += 1
            ok = (
                isinstance(right, torch.Tensor)
                and left.shape == right.shape
                and left.dtype == right.dtype
            )
            if ok:
                a, b = left.detach().cpu(), right.detach().cpu()
                ok = bool(
                    torch.isfinite(a).all()
                    and torch.isfinite(b).all()
                    and torch.equal(a, b)
                )
                if a.numel():
                    delta = float((a.double() - b.double()).abs().max())
                    maximum[0] = max(
                        maximum[0], delta if math.isfinite(delta) else float("inf")
                    )
            if not ok:
                differences.append(path)
        elif isinstance(left, np.ndarray):
            leaves[0] += 1
            if not (
                isinstance(right, np.ndarray)
                and left.dtype == right.dtype
                and np.array_equal(left, right)
            ):
                differences.append(path)
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                differences.append(path + ".keys")
            else:
                for key in left:
                    visit(left[key], right[key], path + "/" + str(key))
        elif isinstance(left, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                differences.append(path + ".sequence")
            else:
                for i, value in enumerate(left):
                    visit(value, right[i], path + "/" + str(i))
        else:
            leaves[0] += 1
            if (
                type(left) is not type(right)
                or left != right
                or (isinstance(left, float) and not math.isfinite(left))
            ):
                differences.append(path)

    visit(expected, actual, "state")
    return {
        "passed": not differences,
        "different_paths": differences,
        "max_tensor_abs_error": maximum[0] if math.isfinite(maximum[0]) else None,
        "leaves_checked": leaves[0],
        "atol": 0.0,
        "rtol": 0.0,
    }


def require_comparison(result):
    """执行者不得临时放宽固定容差。"""
    require(result["passed"], "零容差数值核验失败：" + str(result["different_paths"]))

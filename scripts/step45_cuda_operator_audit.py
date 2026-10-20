#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step45：验证 Blackwell 环境下 MMCV CUDA 算子的数值正确性。

只对比相同随机输入的数学输出，不加载数据集/模型，不修改原仓库。
在原来的 .venv-bevformer 中执行。输出独立 JSON。
"""

from __future__ import annotations

import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "outputs/drivelm_check/root_cause/step45_cuda_operator_audit/results.json"
)


def errors(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """报告相同输入两套实现的数值误差。"""
    lhs = actual.detach().float().cpu()
    rhs = reference.detach().float().cpu()
    diff = (lhs - rhs).abs()
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    rel_norm = float(torch.linalg.vector_norm(lhs - rhs) /
                     torch.linalg.vector_norm(rhs).clamp_min(1e-6))
    finite = bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all())
    return {
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "relative_l2_error": rel_norm,
        "all_finite": finite,
        "pass": finite and max_abs <= 1e-3 and rel_norm <= 1e-3,
    }


def test_msdeform(device: torch.device) -> dict[str, Any]:
    """对比 CUDA 扩展与 MMCV 自带的 grid_sample 参考实现。"""
    from mmcv.ops.multi_scale_deform_attn import (
        MultiScaleDeformableAttnFunction,
        multi_scale_deformable_attn_pytorch,
    )

    torch.manual_seed(20261020)
    shapes = torch.tensor([[8, 12], [4, 6]], dtype=torch.long, device=device)
    level_start = torch.tensor([0, 96], dtype=torch.long, device=device)
    value = torch.randn(1, 120, 4, 8, device=device, dtype=torch.float32)
    sampling = torch.rand(1, 32, 4, 2, 4, 2, device=device, dtype=torch.float32)
    weights = torch.rand(1, 32, 4, 2, 4, device=device, dtype=torch.float32)
    weights = weights / weights.sum(dim=(-1, -2), keepdim=True)

    with torch.no_grad():
        gpu = MultiScaleDeformableAttnFunction.apply(
            value.contiguous(), shapes, level_start, sampling.contiguous(),
            weights.contiguous(), 64,
        )
        ref = multi_scale_deformable_attn_pytorch(
            value, shapes, sampling, weights
        )
        torch.cuda.synchronize()

    return {"status": "tested", "shape": list(gpu.shape), **errors(gpu, ref)}


def test_dcn(device: torch.device) -> dict[str, Any]:
    """使用 torchvision 独立实现检验 MMCV 的 DCNv2 算子。"""
    try:
        from torchvision.ops import deform_conv2d
    except (ImportError, RuntimeError, OSError) as exc:
        return {"status": "reference_unavailable", "details": str(exc)}

    from mmcv.ops.modulated_deform_conv import modulated_deform_conv2d

    torch.manual_seed(20261021)
    x = torch.randn(1, 8, 16, 16, device=device)
    kernel = torch.randn(8, 8, 3, 3, device=device) * 0.05
    bias = torch.randn(8, device=device) * 0.05
    # 3x3 卷积；deform_groups=1；offset 通道顺序与 MMCV/TorchVision 约定一致。
    offset = torch.randn(1, 18, 16, 16, device=device) * 0.3
    mask = torch.sigmoid(torch.randn(1, 9, 16, 16, device=device))

    with torch.no_grad():
        gpu = modulated_deform_conv2d(
            x, offset, mask, kernel, bias, 1, 1, 1, 1, 1
        )
        ref = deform_conv2d(
            x, offset, kernel, bias=bias, stride=(1, 1), padding=(1, 1),
            dilation=(1, 1), mask=mask,
        )
        torch.cuda.synchronize()

    return {"status": "tested", "shape": list(gpu.shape), **errors(gpu, ref)}


def run_test(name: str, fn: Any, device: torch.device) -> dict[str, Any]:
    """各算子独立执行；一个失败不影响另一个的诊断。"""
    try:
        result = fn(device)
    except Exception as exc:  # 诊断程序需要保存异常以供分析
        result = {
            "status": "error",
            "error_type": type(exc).__name__,
            "details": str(exc),
            "traceback_tail": traceback.format_exc()[-3500:],
        }
    print(f"[{name}]", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    """执行不涉及模型和训练数据的 GPU 核心算子基准测试。"""
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA；请使用已能运行 BEVFormer 的环境")
    import mmcv

    device = torch.device("cuda:0")
    report = {
        "purpose": "MMCV CUDA 算子 vs 独立数值参考；不等于整模型复现",
        "torch": torch.__version__,
        "mmcv": mmcv.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "multi_scale_deformable_attention": run_test("MSDeformAttn", test_msdeform, device),
        "modulated_deform_conv2d": run_test("DCNv2", test_dcn, device),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告：", OUTPUT, flush=True)
    if report["multi_scale_deformable_attention"].get("pass") is False:
        print("注意：MSDeformAttn 数值不一致，优先调查 CUDA 扩展。", flush=True)
    elif report["multi_scale_deformable_attention"].get("pass") is True:
        print("MSDeformAttn 数值一致；该测试无法排除其他算子/完整模型问题。", flush=True)


if __name__ == "__main__":
    main()

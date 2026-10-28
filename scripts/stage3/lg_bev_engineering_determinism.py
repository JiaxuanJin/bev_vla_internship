"""确定性反向兼容层：固定网格在GPU归约，动态参考反向保留CPU。"""

from __future__ import annotations

from contextlib import contextmanager
from collections import OrderedDict
import hashlib
import os
from typing import Iterator

import torch
from torch import Tensor
from torch.nn import functional as F

ORIGINAL_GRID = F.grid_sample
ORIGINAL_INTERPOLATE = F.interpolate
ORIGINAL_CONV = F.conv2d
ADJOINTS = OrderedDict()


def sparse_adjoint(
    x: Tensor,
    y: Tensor,
    height: int,
    width: int,
    device,
    *,
    border: bool = False,
    order: str = "grid",
):
    """CPU按原始scatter顺序生成邻居表，GPU固定顺序归约，不合并边界项。"""
    key = hashlib.sha256(
        x.numpy().tobytes()
        + y.numpy().tobytes()
        + str((height, width, border, device, order)).encode()
    ).hexdigest()
    if key in ADJOINTS:
        ADJOINTS.move_to_end(key)
        return ADJOINTS[key]
    if border:
        x = x.clamp(0, width - 1)
        y = y.clamp(0, height - 1)
    xf = x.floor()
    yf = y.floor()
    xx = x - xf
    yy = y - yf
    rows = []
    columns = []
    weights = []
    output = torch.arange(x.numel()).reshape(x.shape)
    for dy, wy in ((0, 1 - yy), (1, yy)):
        for dx, wx in ((0, 1 - xx), (1, xx)):
            ix = xf.long() + dx
            iy = yf.long() + dy
            if border:
                ix = ix.clamp(0, width - 1)
                iy = iy.clamp(0, height - 1)
            mass = wx * wy
            valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height) & (mass != 0)
            rows.append((iy[valid] * width + ix[valid]))
            columns.append(output[valid])
            weights.append(mass[valid])
    # 保留边界重复项及原始scatter顺序；coalesce会改变FP32求和次序。
    row, column, weight = torch.cat(rows), torch.cat(columns), torch.cat(weights)
    corner = torch.cat([torch.full_like(c, k) for k, c in enumerate(columns)])
    if order == "grid":
        vector = 16 if torch.backends.cpu.get_cpu_capability() == "AVX512" else 8
        sequence = (column // vector) * (4 * vector) + corner * vector + column % vector
    else:
        sequence = column * 4 + corner
    initial = torch.argsort(sequence, stable=True)
    index = initial[torch.argsort(row[initial], stable=True)]
    row, column, weight = row[index], column[index], weight[index]
    counts = torch.bincount(row, minlength=height * width)
    pointers = counts.cumsum(0) - counts
    count = int(counts.max()) if counts.numel() else 0
    neighbors = torch.zeros(count, height * width, dtype=torch.long)
    coefficients = torch.zeros(count, height * width, dtype=torch.float32)
    for index in range(count):
        active = counts > index
        positions = pointers[active] + index
        neighbors[index, active] = column[positions]
        coefficients[index, active] = weight[positions]
    matrix = (neighbors.to(device), coefficients.to(device))
    ADJOINTS[key] = matrix
    if len(ADJOINTS) > 12:
        ADJOINTS.popitem(last=False)
    return matrix


def apply_adjoint(matrix, gradient: Tensor, height: int, width: int) -> Tensor:
    """每个原生像素独立按登记顺序累积，不在CUDA线程间使用atomic。"""
    channels = gradient.shape[1]
    indices, weights = matrix
    flat = gradient[0].reshape(channels, -1)
    output = gradient.new_zeros((channels, height * width))
    for index in range(indices.shape[0]):
        selected = flat.index_select(1, indices[index])
        output.add_(selected * weights[index])
    return output.reshape(1, channels, height, width)


class StableConv(torch.autograd.Function):
    """原CUDNN前向保持不变，仅在GPU adjoint选择确定性的卷积反向算法。"""

    @staticmethod
    def forward(ctx, value, weight, bias, stride, padding, dilation, groups):
        ctx.save_for_backward(value, weight)
        ctx.options = (stride, padding, dilation, groups)
        ctx.has_bias = bias is not None
        return ORIGINAL_CONV(value, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, gradient):
        value, weight = ctx.saved_tensors
        stride, padding, dilation, groups = ctx.options
        with torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            benchmark=False,
            deterministic=True,
            allow_tf32=torch.backends.cudnn.allow_tf32,
        ):
            gv = (
                torch.nn.grad.conv2d_input(
                    value.shape, weight, gradient, stride, padding, dilation, groups
                )
                if ctx.needs_input_grad[0]
                else None
            )
            gw = (
                torch.nn.grad.conv2d_weight(
                    value, weight.shape, gradient, stride, padding, dilation, groups
                )
                if ctx.needs_input_grad[1]
                else None
            )
        gb = (
            gradient.sum((0, 2, 3))
            if ctx.has_bias and ctx.needs_input_grad[2]
            else None
        )
        return gv, gw, gb, None, None, None, None


class StableGrid(torch.autograd.Function):
    """前向CUDA不变；固定网格采用无atomic GPU伴随，动态网格保留参考。"""

    @staticmethod
    def forward(
        ctx,
        value: Tensor,
        grid: Tensor,
        mode: str = "bilinear",
        padding: str = "zeros",
        align: bool = False,
    ) -> Tensor:
        ctx.save_for_backward(value, grid)
        ctx.options = (mode, padding, align)
        return ORIGINAL_GRID(
            value, grid, mode=mode, padding_mode=padding, align_corners=align
        )

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor | None]:
        value, grid = ctx.saved_tensors
        mode, padding, align = ctx.options
        if (
            os.environ.get("LG_BEV_FIXED_CUDA", "1") == "1"
            and value.is_cuda
            and value.shape[0] == 1
            and not grid.requires_grad
            and mode == "bilinear"
            and padding in ("zeros", "border")
        ):
            height, width = value.shape[-2:]
            coordinates = grid.detach().cpu()[0]
            # CPU AVX2以FMA做乘法减半像素；double只模拟一次舍入，权重仍为FP32。
            x = (
                (coordinates[..., 0] + 1) * ((width - 1) / 2)
                if align
                else ((coordinates[..., 0] + 1).double() * (width / 2) - 0.5).float()
            )
            y = (
                (coordinates[..., 1] + 1) * ((height - 1) / 2)
                if align
                else ((coordinates[..., 1] + 1).double() * (height / 2) - 0.5).float()
            )
            matrix = sparse_adjoint(
                x, y, height, width, value.device, border=padding == "border"
            )
            result = (
                apply_adjoint(matrix, grad_output, height, width),
                None,
                None,
                None,
                None,
            )
            return result[: len(ctx.needs_input_grad)]
        if (
            os.environ.get("LG_BEV_CPU_DIRECT", "1") == "1"
            and not grid.requires_grad
            and mode == "bilinear"
        ):
            # 输入值不参与grad_input计算；避免D2H输入及无用的CPU前向。
            dummy = torch.empty(value.shape, dtype=value.dtype)
            gradient = torch.ops.aten.grid_sampler_2d_backward(
                grad_output.detach().cpu(),
                dummy,
                grid.detach().cpu(),
                0,
                {"zeros": 0, "border": 1, "reflection": 2}[padding],
                align,
                [True, False],
            )[0]
            return (gradient.to(value.device), None, None, None, None)[
                : len(ctx.needs_input_grad)
            ]
        with torch.enable_grad():
            cpu = value.detach().cpu().requires_grad_(True)
            coordinates = grid.detach().cpu().requires_grad_(grid.requires_grad)
            mode, padding, align = ctx.options
            out = ORIGINAL_GRID(
                cpu, coordinates, mode=mode, padding_mode=padding, align_corners=align
            )
            variables = [cpu, coordinates] if grid.requires_grad else [cpu]
            gradients = torch.autograd.grad(out, variables, grad_output.detach().cpu())
        result = (
            gradients[0].to(value.device),
            gradients[1].to(grid.device) if grid.requires_grad else None,
            None,
            None,
            None,
        )
        return result[: len(ctx.needs_input_grad)]


class StableInterpolate(torch.autograd.Function):
    """保持作者CUDA前向；默认使用同一CPU反向内核，GPU候选不采用。"""

    @staticmethod
    def forward(ctx, value: Tensor, options: dict) -> Tensor:
        ctx.save_for_backward(value)
        ctx.options = options
        return ORIGINAL_INTERPOLATE(value, **options)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        value = ctx.saved_tensors[0]
        if (
            os.environ.get("LG_BEV_UPSAMPLE_CUDA", "0") == "1"
            and value.is_cuda
            and value.shape[0] == 1
            and ctx.options["mode"] == "bilinear"
        ):
            height, width = value.shape[-2:]
            oh, ow = grad_output.shape[-2:]
            if ctx.options["align_corners"]:
                x = torch.arange(ow, dtype=torch.float32) * (
                    (width - 1) / max(1, ow - 1)
                )
                y = torch.arange(oh, dtype=torch.float32) * (
                    (height - 1) / max(1, oh - 1)
                )
            else:
                scale = ctx.options["scale_factor"]
                sx = (
                    (1 / scale if isinstance(scale, (int, float)) else 1 / scale[-1])
                    if scale is not None and not ctx.options["recompute_scale_factor"]
                    else width / ow
                )
                sy = (
                    (1 / scale if isinstance(scale, (int, float)) else 1 / scale[-2])
                    if scale is not None and not ctx.options["recompute_scale_factor"]
                    else height / oh
                )
                x = (torch.arange(ow, dtype=torch.float32) + 0.5) * sx - 0.5
                y = (torch.arange(oh, dtype=torch.float32) + 0.5) * sy - 0.5
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            matrix = sparse_adjoint(
                xx, yy, height, width, value.device, border=True, order="interpolate"
            )
            return apply_adjoint(matrix, grad_output, height, width), None
        if (
            os.environ.get("LG_BEV_CPU_DIRECT", "1") == "1"
            and ctx.options["mode"] == "bilinear"
        ):
            scale = ctx.options["scale_factor"]
            sy = (
                scale
                if isinstance(scale, (float, int))
                else scale[0] if scale else None
            )
            sx = (
                scale
                if isinstance(scale, (float, int))
                else scale[1] if scale else None
            )
            if ctx.options["recompute_scale_factor"]:
                sy = sx = None
            gradient = torch.ops.aten.upsample_bilinear2d_backward(
                grad_output.detach().cpu(),
                list(grad_output.shape[-2:]),
                list(value.shape),
                ctx.options["align_corners"],
                sy,
                sx,
            )
            return gradient.to(value.device), None
        with torch.enable_grad():
            cpu = ctx.saved_tensors[0].detach().cpu().requires_grad_(True)
            out = ORIGINAL_INTERPOLATE(cpu, **ctx.options)
            gradient = torch.autograd.grad(out, cpu, grad_output.detach().cpu())[0]
        return gradient.to(ctx.saved_tensors[0].device), None


@contextmanager
def stable_native_backward() -> Iterator[None]:
    """临时包装自研训练进程的autograd backward，不改作者forward/源码或权重。"""
    from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
    from projects.mmdet3d_plugin.bevformer.modules.multi_scale_deformable_attn_function import (
        MultiScaleDeformableAttnFunction_fp32,
    )

    function = MultiScaleDeformableAttnFunction_fp32
    original = function.__dict__["backward"]
    grid_original = F.grid_sample
    conv_original = F.conv2d
    interpolate_original = F.interpolate

    def grid(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
        if (
            torch.is_grad_enabled()
            and input.is_cuda
            and (input.requires_grad or grid.requires_grad)
        ):
            return StableGrid.apply(
                input, grid, mode, padding_mode, bool(align_corners)
            )
        return grid_original(
            input,
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

    def interpolate(
        value,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
        options = dict(
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=antialias,
        )
        if (
            torch.is_grad_enabled()
            and value.is_cuda
            and value.requires_grad
            and mode in ("bilinear", "bicubic", "trilinear")
        ):
            return StableInterpolate.apply(value, options)
        return interpolate_original(value, **options)

    def conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if (
            torch.is_grad_enabled()
            and input.is_cuda
            and (input.requires_grad or weight.requires_grad)
        ):
            return StableConv.apply(
                input, weight, bias, stride, padding, dilation, groups
            )
        return conv_original(input, weight, bias, stride, padding, dilation, groups)

    def backward(ctx, grad_output):
        value, shapes, _starts, locations, weights = ctx.saved_tensors
        if os.environ.get("LG_BEV_DYNAMIC_CUDA", "0") in ("1", "ordered"):
            from scripts.stage3.lg_bev_engineering_dynamic_cuda import (
                backward as cuda_backward,
            )

            gv, gl, gw = cuda_backward(
                value,
                shapes,
                _starts,
                locations,
                weights,
                grad_output,
                ordered=os.environ.get("LG_BEV_DYNAMIC_CUDA") == "ordered",
            )
            return gv, None, None, gl, gw, None
        with torch.enable_grad():
            cpu_value = value.detach().cpu().requires_grad_(True)
            cpu_locations = locations.detach().cpu().requires_grad_(True)
            cpu_weights = weights.detach().cpu().requires_grad_(True)
            out = multi_scale_deformable_attn_pytorch(
                cpu_value, shapes.cpu(), cpu_locations, cpu_weights
            )
            gradients = torch.autograd.grad(
                out, (cpu_value, cpu_locations, cpu_weights), grad_output.detach().cpu()
            )
        return (
            gradients[0].to(value.device),
            None,
            None,
            gradients[1].to(locations.device),
            gradients[2].to(weights.device),
            None,
        )

    function.backward = staticmethod(backward)
    F.grid_sample = grid
    F.conv2d = conv
    F.interpolate = interpolate
    try:
        yield
    finally:
        function.backward = original
        F.grid_sample = grid_original
        F.conv2d = conv_original
        F.interpolate = interpolate_original

"""未通过32帧数值门槛的MSDeform GPU反向候选；正式入口禁止使用。"""

from __future__ import annotations

import torch
from torch import Tensor


def backward(
    value: Tensor,
    shapes: Tensor,
    starts: Tensor,
    locations: Tensor,
    weights: Tensor,
    gradient: Tensor,
    *,
    ordered: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """FP32双线性导数；仅value梯度需归约，使用stable sort和唯一行写入。

    Args:
        value: [B,L,H,D] 原始FP32 value。
        shapes: 各level的高宽，仅少量形状整数读取到CPU。
        starts: 各level起点。
        locations: [B,Q,H,level,P,2]，真实原生sampling locations。
        weights: 相同采样点的attention weights。
        gradient: 原生CUDA forward的输出梯度[B,Q,H*D]。
    """
    if value.dtype != torch.float32:
        raise TypeError("动态GPU adjoint只接受FP32")
    batch, length, heads, channels = value.shape
    queries = locations.shape[1]
    points = locations.shape[4]
    go = gradient.reshape(batch, queries, heads, channels)
    gl = torch.zeros_like(locations)
    gw = torch.zeros_like(weights)
    all_indices = []
    all_contributions = []
    batch_id = torch.arange(batch, device=value.device)[:, None, None, None]
    head_id = torch.arange(heads, device=value.device)[None, None, :, None]
    flat_value = value.reshape(batch * length * heads, channels)
    sentinel = batch * length * heads
    for level, (height, width) in enumerate(shapes.detach().cpu().tolist()):
        offset = int(starts[level])
        coordinates = locations[:, :, :, level]
        # 与已验证PyTorch参考链一致：先0..1→-1..1，再align_corners=False。
        grid = 2 * coordinates - 1
        x = ((grid[..., 0] + 1) * width - 1) / 2
        y = ((grid[..., 1] + 1) * height - 1) / 2
        x0 = x.floor()
        y0 = y.floor()
        a = x - x0
        b = y - y0
        valid_point = (x > -1) & (x < width) & (y > -1) & (y < height)
        values = []
        indices = []
        coefficients = []
        for dy, wy in ((0, 1 - b), (1, b)):
            for dx, wx in ((0, 1 - a), (1, a)):
                ix = x0.long() + dx
                iy = y0.long() + dy
                valid = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
                index = (
                    batch_id * length
                    + offset
                    + iy.clamp(0, height - 1) * width
                    + ix.clamp(0, width - 1)
                ) * heads + head_id
                selected = flat_value.index_select(0, index.reshape(-1)).reshape(
                    batch, queries, heads, points, channels
                )
                values.append(selected * valid[..., None])
                indices.append(
                    torch.where(valid, index, torch.full_like(index, sentinel))
                )
                coefficients.append(wx * wy * valid)
        v00, v10, v01, v11 = values
        sample = sum(v * c[..., None] for v, c in zip(values, coefficients))
        weight_terms = sample * go[:, :, :, None, :]
        if ordered:
            if channels != 32:
                raise ValueError("CPU顺序兼容仅登记原模型D32")

            def sequential(values, indices):
                total = torch.zeros_like(values[..., 0])
                for index in indices:
                    total = total + values[..., index]
                return total

            main = sequential(weight_terms, range(16)) + sequential(
                weight_terms, range(16, 32)
            )
            tail = sequential(weight_terms, range(0, 32, 4))
            for mod in (1, 2, 3):
                tail = tail + sequential(weight_terms, range(mod, 32, 4))
            vector = 16 if torch.backends.cpu.get_cpu_capability() == "AVX512" else 8
            position = torch.arange(queries * points, device=value.device).reshape(
                1, queries, 1, points
            )
            main_limit = (queries * points // (4 * vector)) * (4 * vector)
            gw[:, :, :, level] = torch.where(position < main_limit, main, tail)
        else:
            gw[:, :, :, level] = weight_terms.sum(-1)
        grad_sample = go[:, :, :, None, :] * weights[:, :, :, level, :, None]
        dx = (v10 - v00) * (1 - b)[..., None] + (v11 - v01) * b[..., None]
        dy = (v01 - v00) * (1 - a)[..., None] + (v11 - v10) * a[..., None]
        if ordered:
            gx = torch.zeros_like(x)
            gy = torch.zeros_like(y)
            for channel in range(channels):
                gx = gx + dx[..., channel] * grad_sample[..., channel]
                gy = gy + dy[..., channel] * grad_sample[..., channel]
            gl[:, :, :, level, :, 0] = (gx * (width / 2)) * 2
            gl[:, :, :, level, :, 1] = (gy * (height / 2)) * 2
        else:
            gl[:, :, :, level, :, 0] = (dx * grad_sample).sum(-1) * width * valid_point
            gl[:, :, :, level, :, 1] = (dy * grad_sample).sum(-1) * height * valid_point
        index = torch.stack(indices, -1)
        contribution = torch.stack(
            [grad_sample * c[..., None] for c in coefficients], -2
        )
        if ordered:
            vector = 16 if torch.backends.cpu.get_cpu_capability() == "AVX512" else 8
            n = queries * points
            pad = (-n) % vector
            index = index.permute(0, 2, 1, 3, 4).reshape(batch * heads, n, 4)
            contribution = contribution.permute(0, 2, 1, 3, 4, 5).reshape(
                batch * heads, n, 4, channels
            )
            if pad:
                index = torch.cat(
                    (
                        index,
                        torch.full(
                            (batch * heads, pad, 4),
                            sentinel,
                            device=value.device,
                            dtype=torch.long,
                        ),
                    ),
                    1,
                )
                contribution = torch.cat(
                    (contribution, value.new_zeros(batch * heads, pad, 4, channels)), 1
                )
            index = index.reshape(batch * heads, -1, vector, 4).transpose(2, 3)
            contribution = contribution.reshape(
                batch * heads, -1, vector, 4, channels
            ).transpose(2, 3)
        index = index.reshape(-1)
        contribution = contribution.reshape(-1, channels)
        all_indices.append(index)
        all_contributions.append(contribution)
    indices = torch.cat(all_indices)
    contribution = torch.cat(all_contributions)
    order = torch.argsort(indices, stable=True)
    sorted_indices = indices.index_select(0, order)
    unique, counts = torch.unique_consecutive(sorted_indices, return_counts=True)
    sorted_values = contribution.index_select(0, order)
    if ordered:
        offsets = counts.cumsum(0) - counts
        reduced = value.new_zeros((len(unique), channels))
        for k in range(int(counts.max())):
            positions = offsets + torch.minimum(torch.full_like(counts, k), counts - 1)
            reduced = (
                reduced
                + sorted_values.index_select(0, positions) * (counts > k)[:, None]
            )
    else:
        reduced = torch.segment_reduce(sorted_values, "sum", lengths=counts)
    gv = value.new_zeros((sentinel + 1, channels))
    gv.index_copy_(0, unique, reduced)
    return gv[:-1].reshape_as(value), gl, gw

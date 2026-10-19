"""检查 BEVFormer-Seg Encoder、转置与地图采样器的坐标轴约定。

本脚本只使用人工构造的坐标通道，不运行网络或修改权重。
输出是坐标一致性的单元测试，不能代替真实分割 mIoU。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from audit_bev_slicer_coordinates import (
    CONFIG,
    HEAD,
    ROOT,
    load_slicer_implementation,
    read_constant_assignments,
)

OUT = ROOT / "outputs" / "drivelm_check" / "root_cause"
ENCODER = (
    ROOT
    / "BEVFormer-seg"
    / "projects"
    / "mmdet3d_plugin"
    / "bevformer"
    / "modules"
    / "encoder.py"
)


def error_metrics(
    actual: torch.Tensor,
    expected_x: torch.Tensor,
    expected_y: torch.Tensor,
) -> dict[str, float]:
    """测量两个物理坐标通道与目标网格坐标的平均偏差。"""
    return {
        "x_mae_m": float((actual[0, 0] - expected_x).abs().mean()),
        "y_mae_m": float((actual[0, 1] - expected_y).abs().mean()),
    }


def main() -> None:
    """构造已知坐标通道，检查原始与转置后采样的轴一致性。"""
    cfg = read_constant_assignments(
        CONFIG, ("det_grid_conf", "map_grid_conf", "bev_h_", "bev_w_")
    )
    det_grid = cfg["det_grid_conf"]
    map_grid = cfg["map_grid_conf"]
    height = int(cfg["bev_h_"])
    width = int(cfg["bev_w_"])

    head_text = HEAD.read_text(encoding="utf-8")
    encoder_text = ENCODER.read_text(encoding="utf-8")
    flags = {
        "seg_head_rot90_clockwise": (
            "torch.rot90(seg_bev, k=-1, dims=[2, 3])" in head_text
        ),
        "seg_head_flip_last_spatial_axis": (
            "torch.flip(seg_bev, dims=[3])" in head_text
        ),
        "encoder_x_varies_with_column": (
            "xs = torch.linspace" in encoder_text
            and "view(1, 1, W)" in encoder_text
        ),
        "encoder_y_varies_with_row": (
            "ys = torch.linspace" in encoder_text
            and "view(1, H, 1)" in encoder_text
        ),
    }
    if not all(flags.values()):
        raise RuntimeError(
            "本地源码与预期版本存在差异，已停止。请发送源码检查结果："
            f"{flags}"
        )

    slicer_class = load_slicer_implementation(HEAD)
    slicer = slicer_class(det_grid, map_grid)

    # 给原始 BEV 的列写入 X 坐标、行写入 Y 坐标。
    # 这里用采样器的归一化半径，仅测试轴约定，不模拟真实图像特征。
    x_start = det_grid["xbound"][0] + det_grid["xbound"][2] / 2
    y_start = det_grid["ybound"][0] + det_grid["ybound"][2] / 2
    x_values = torch.linspace(x_start, -x_start, width)
    y_values = torch.linspace(y_start, -y_start, height)
    yy, xx = torch.meshgrid(y_values, x_values, indexing="ij")
    coordinate_bev = torch.stack((xx, yy), dim=0).unsqueeze(0)

    # 与用户分割头中的 rot90 + flip 保持完全相同的操作。
    rotated = torch.rot90(coordinate_bev, k=-1, dims=[2, 3])
    transposed = torch.flip(rotated, dims=[3])
    transpose_equal = torch.equal(
        transposed, coordinate_bev.transpose(2, 3)
    )
    if not transpose_equal:
        raise AssertionError("rot90 + flip 并未等价于转置，检查操作版本")

    with torch.no_grad():
        without_swap = slicer(coordinate_bev)
        with_swap = slicer(transposed)

    # map_grid 的目标：输出行对应 Y，输出列对应 X。
    expected_x, expected_y = torch.meshgrid(
        slicer.map_x, slicer.map_y, indexing="xy"
    )
    if without_swap.shape != with_swap.shape or with_swap.shape[1] != 2:
        raise AssertionError("坐标采样输出形状不符合预期")
    if tuple(with_swap.shape[-2:]) != tuple(expected_x.shape):
        raise AssertionError("目标地图网格尺寸不一致")

    raw_metrics = error_metrics(without_swap, expected_x, expected_y)
    swap_metrics = error_metrics(with_swap, expected_x, expected_y)
    result = {
        "type": "synthetic_coordinate_consistency_only",
        "source_flags": flags,
        "bev_shape": [height, width],
        "map_output_shape": list(with_swap.shape),
        "rot90_plus_flip_equals_transpose": transpose_equal,
        "no_axis_swap": raw_metrics,
        "with_source_axis_swap": swap_metrics,
        "interpretation": (
            "在假设 Encoder 的 X 随列增加、Y 随行增加的人工坐标场中，"
            "比较转置前后与 Slicer 地图网格是否一致。该结果不能直接证明"
            "预训练模型的坐标错误，也不能作为正式 mIoU。"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json_file = OUT / "step6_axis_chain_synthetic.json"
    json_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    images = [
        (expected_x.numpy(), "Target map X (m)"),
        (without_swap[0, 0].numpy(), "Without axis swap: sampled X"),
        (with_swap[0, 0].numpy(), "With axis swap: sampled X"),
    ]
    for ax, (data, title) in zip(axes, images):
        im = ax.imshow(data, origin="lower", cmap="coolwarm")
        ax.set_title(title)
        ax.set_xlabel("Map column")
        ax.set_ylabel("Map row")
        fig.colorbar(im, ax=ax, shrink=0.65)
    fig.tight_layout()
    figure_file = OUT / "step6_axis_chain_synthetic.png"
    fig.savefig(figure_file, dpi=150)
    plt.close(fig)

    print("===== BEV 坐标轴完整链路：人工坐标实验 =====")
    print("本地源码检查:", flags)
    print("rot90 + flip 等价于转置:", transpose_equal)
    print("不交换轴，X/Y 坐标平均绝对误差(米):", raw_metrics)
    print("按当前分割头交换轴后，X/Y 平均绝对误差(米):", swap_metrics)
    print("[结果]", json_file)
    print("[图片]", figure_file)
    print("[提醒] 只证明人工坐标下的轴约定是否一致，不是模型 mIoU。")


if __name__ == "__main__":
    main()

"""独立 Attention 监督空间：只映射 logits，不改变原生双 Head 的任何采样。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from scripts.stage2.spatial_annotations import metric_grid
from scripts.stage3.lg_bev_data import load_supervision
from scripts.stage3.lg_bev_native import native_losses
from scripts.stage3.lg_bev_prototype import attention_supervision_loss
from scripts.stage3.lg_bev_train import training_frame


@dataclass(frozen=True)
class AttentionSupervisionSpace:
    """标定只来自传感器；实例 GT 仅在 loss 接口读取。

    Args:
        task: map 为 Ego GT 栅格，对象为原生 LiDAR 栅格。
        sampling_grid: Ego GT 像素中心在原生 LiDAR BEV 的归一化位置。
    """

    task: str
    sampling_grid: Tensor | None

    @classmethod
    def from_calibration(
        cls, task: str, calibration: dict
    ) -> AttentionSupervisionSpace:
        """白名单提取 R/t，禁止实例或预测影响监督映射。"""
        if task == "object":
            return cls(task, None)
        if task != "map":
            raise ValueError("未知监督空间")
        sensor = {
            key: calibration[key]
            for key in ("lidar2ego_rotation", "lidar2ego_translation")
        }
        return cls(task, metric_grid(sensor)[0])

    def project(self, values: Tensor) -> Tensor:
        """Attention/native ramp → GT 空间，保持 autograd；不调用 Head cropper。"""
        if values.ndim != 4 or values.shape[-2:] != (150, 150):
            raise ValueError("输入必须是原生 BEV 空间张量")
        if self.sampling_grid is None:
            return values
        return F.grid_sample(
            values,
            self.sampling_grid.to(values),
            align_corners=False,
            mode="bilinear",
            padding_mode="zeros",
        )

    def supervision(
        self, logits: Tensor, labels: dict[str, Tensor], weight: float
    ) -> dict:
        """只在损失计算时读取 target/distractor/valid，不进入模型 forward。"""
        projected = self.project(logits)
        return attention_supervision_loss(
            projected,
            labels["target"],
            labels["distractor"],
            distractor_weight=weight,
            valid_mask=labels["valid"],
        )


def bilinear_adjoint_check(
    space: AttentionSupervisionSpace, pixel: tuple[int, int]
) -> dict:
    """独立计算四邻域插值权重，与真实 autograd 对照，不由 GT 拟合变换。"""
    if space.sampling_grid is None:
        raise ValueError("map 空间才需要双线性采样")
    values = torch.zeros(1, 1, 150, 150, dtype=torch.float64, requires_grad=True)
    row, col = pixel
    space.project(values)[0, 0, row, col].backward()
    gx, gy = space.sampling_grid[0, row, col].tolist()
    x, y = (gx + 1) * 75 - 0.5, (gy + 1) * 75 - 0.5
    ix, iy = int(np.floor(x)), int(np.floor(y))
    expected = torch.zeros_like(values)
    for r in (iy, iy + 1):
        for c in (ix, ix + 1):
            if 0 <= r < 150 and 0 <= c < 150:
                expected[0, 0, r, c] = (1 - abs(x - c)) * (1 - abs(y - r))
    error = float((values.grad - expected).abs().max())
    return dict(
        gt_pixel=[row, col],
        max_weight_error=error,
        gradient_support_cells=int(torch.count_nonzero(values.grad)),
        gradient_mass=float(values.grad.sum()),
        pass_check=error < 1e-10,
    )


def objective(
    model: nn.Module,
    prediction: dict,
    detail: dict,
    row: dict,
    label_root,
    annotation: dict,
    teacher: Tensor,
    protocol: dict,
) -> tuple[Tensor, dict, dict]:
    """Attention 使用独立物理监督；原生 Head 使用其原始检测/语义 GT 协议。"""
    gt, xy = training_frame(row)
    labels = {
        k: v.cuda() for k, v in load_supervision(row, label_root, training=True).items()
    }
    parts = native_losses(model, prediction, annotation, gt)
    stats = parts.pop("filter_stats")
    space = AttentionSupervisionSpace.from_calibration(row["task"], row)
    mask = space.supervision(
        detail["attention_logits"], labels, protocol["loss"]["distractor"]
    )
    if row["task"] == "map":
        fb, lr = row["fb"], row["lr"]
    else:
        fb, lr = row["prompt"]["front_back"], row["prompt"]["left_right"]
    region = ((xy[..., 0] > 0) if fb == "front" else (xy[..., 0] < 0)) & (
        (xy[..., 1] > 0) if lr == "left" else (xy[..., 1] < 0)
    )
    kl = F.kl_div(
        prediction["seg_preds"].log_softmax(1), teacher, reduction="none"
    ).sum(1)[0]
    parts.update(
        mask=mask["total"],
        mask_bce=mask["bce"],
        mask_dice=mask["dice"],
        distractor=mask["distractor"],
        irrelevant_protect=kl[~region].mean(),
    )
    total = sum(
        parts[key] * protocol["loss"][key]
        for key in ("detection", "segmentation", "mask", "irrelevant_protect")
    )
    if not bool(torch.isfinite(total)):
        raise ValueError("监督损失非有限")
    return total, parts, stats

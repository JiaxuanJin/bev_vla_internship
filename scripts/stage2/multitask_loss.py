"""第二阶段多任务损失聚合接口。

只实现可复用的张量接口，不宣称已将检测头纳入真实训练。
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn


class MultiTaskLoss(nn.Module):
    """组合检测、地图分割及注意力损失。

    Args:
        det_weight: 检测任务系数。
        seg_weight: 地图分割任务系数。
        attention_weight: 空间注意力正则系数。
    """

    def __init__(
        self,
        det_weight: float = 1.0,
        seg_weight: float = 1.0,
        attention_weight: float = 0.1,
    ) -> None:
        super().__init__()
        values = (det_weight, seg_weight, attention_weight)
        if not all(0 <= value < float("inf") for value in values):
            raise ValueError("损失权重应为有限非负数")
        self.weights = values

    def forward(
        self,
        detection_losses: Mapping[str, Tensor],
        segmentation_loss: Tensor,
        attention_loss: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """合并已有检测头给出的多个损失；支持联合反向传播。"""
        if not detection_losses:
            raise ValueError("检测损失不能为空；未启用检测头不能伪造检测损失")
        parts = list(detection_losses.values()) + [segmentation_loss, attention_loss]
        if any(item.ndim != 0 for item in parts):
            raise ValueError("每项损失必须是标量张量")
        if not all(bool(torch.isfinite(item.detach())) for item in parts):
            raise ValueError("检测到 NaN/Inf 损失")
        det = torch.stack(list(detection_losses.values())).sum()
        weighted = {
            "detection": self.weights[0] * det,
            "segmentation": self.weights[1] * segmentation_loss,
            "attention": self.weights[2] * attention_loss,
        }
        return sum(weighted.values()), weighted


def self_test() -> None:
    """使用合成损失检查真实梯度流向每个任务分支。"""
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    z = torch.tensor(4.0, requires_grad=True)
    combined, parts = MultiTaskLoss()(
        {"loss_cls": x.square(), "loss_bbox": x.abs()},
        y.square(),
        z.square(),
    )
    combined.backward()
    assert parts.keys() == {"detection", "segmentation", "attention"}
    assert all(t.grad is not None and bool(torch.isfinite(t.grad)) for t in (x, y, z))
    print("[Step68] 多任务 Loss 接口与梯度自检 PASS（合成数据）")


if __name__ == "__main__":
    self_test()

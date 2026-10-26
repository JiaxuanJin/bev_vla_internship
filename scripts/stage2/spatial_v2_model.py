"""第二轮：方位 token 读出、可学习米制位置特征和同类方位对比损失。"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .bev_coordinates import BEVGrid
from .bev_fusion import LanguageBEVFusion
from .language_encoder import DistilBERTLanguageEncoder
from .spatial_annotations import project_attention


class DirectionReadout(nn.Module):
    """保留冻结 DistilBERT，额外读出两个方位轴的上下文 token 向量。

    不产生规则方位掩码，不读取实例、GT 或目标 ROI。
    """

    def __init__(self, base: DistilBERTLanguageEncoder):
        super().__init__()
        self.base = base
        self.hidden_size = 3 * base.hidden_size
        self.base.requires_grad_(False).eval()
        self.axis_ids = [
            [base.tokenizer.convert_tokens_to_ids(w) for w in words]
            for words in (("front", "rear", "ahead", "behind"), ("left", "right"))
        ]

    def forward(self, texts, device):
        self.base.model.eval()
        tokens = self.base.tokenize(texts, device)
        with torch.no_grad():
            hidden = self.base.model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].bool()
            pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            axes = []
            for ids in self.axis_ids:
                selected = torch.zeros_like(mask)
                for token_id in ids:
                    selected |= tokens["input_ids"] == token_id
                selected &= mask
                readout = (hidden * selected[..., None]).sum(1) / selected.sum(1, keepdim=True).clamp_min(1)
                axes.append(torch.where(selected.any(1, keepdim=True), readout, pooled))
        return torch.cat([pooled, *axes], dim=1)


class ContrastFusion(nn.Module):
    """视觉类别相似度与学习的位置相似度联合生成原生 BEV Attention。"""

    def __init__(self, initial_state):
        super().__init__()
        self.base = LanguageBEVFusion(768)
        self.base.load_state_dict(initial_state, strict=True)
        with torch.no_grad():
            self.base.alpha.fill_(.01)
        self.position_projection = nn.Sequential(
            nn.Conv2d(5, 32, 1), nn.Tanh(), nn.Conv2d(32, 32, 1),
        )
        self.direction_query = nn.Sequential(
            nn.LayerNorm(1536), nn.Linear(1536, 64), nn.GELU(), nn.Linear(64, 32),
        )
        self.geometry_scale = nn.Parameter(torch.tensor(2.))
        self.register_buffer("position_features", torch.zeros(1, 5, 150, 150), persistent=False)
        self.last_bev = None

    @property
    def alpha(self):
        return self.base.alpha

    def set_calibration(self, row):
        """仅使用传感器标定构造地面平面坐标，不使用 GT 或指令解析掩码。"""
        from pyquaternion import Quaternion
        native = BEVGrid().lidar_centers().to(self.alpha)
        rotation = torch.tensor(Quaternion(row["lidar2ego_rotation"]).rotation_matrix,
                                device=native.device, dtype=native.dtype)
        shift = native.new_tensor(row["lidar2ego_translation"])
        native[..., 2] = (-shift[2] - native[..., 0] * rotation[2, 0]
                          - native[..., 1] * rotation[2, 1]) / rotation[2, 2]
        ego = native @ rotation.T + shift
        x, y = ego[..., 0] / 30., ego[..., 1] / 15.
        features = torch.stack((x, y, x * y, (x.square() + y.square()).sqrt(), torch.ones_like(x)))
        self.position_features.copy_(features[None])

    def forward_with_details(self, bev_features, pooled_text, enabled=True):
        if not enabled:
            return {"enhanced_bev": bev_features, "attention": bev_features.new_zeros((1, 1, 150, 150)),
                    "semantic_bev": torch.zeros_like(bev_features)}
        self.last_bev = bev_features.detach()
        details = self.base.forward_with_details(bev_features, pooled_text[:, :768])
        visual_logits = torch.logit(details["attention"].clamp(1e-5, 1 - 1e-5))
        keys = F.normalize(self.position_projection(self.position_features), dim=1)
        query = F.normalize(self.direction_query(pooled_text[:, 768:]), dim=1)
        geometry = (keys * query[:, :, None, None]).sum(1, keepdim=True) * self.geometry_scale
        logits = visual_logits + geometry
        attention = logits.sigmoid()
        details.update(attention=attention, geometry_logits=geometry,
                       enhanced_bev=bev_features + self.alpha * bev_features * attention)
        return details


def balanced_instance_loss(logits: Tensor, gt: Tensor, target: Tensor, cls: int):
    """正例、同类干扰物和其他像素分组归一化，避免面积稀疏导致的全负解。"""
    value = logits.reshape(gt.shape)
    positive = F.softplus(-value[target]).mean()
    distractor = (gt == cls) & ~target
    negative = (gt != cls) & ~target
    background = F.softplus(value[negative]).mean()
    same_class = F.softplus(value[distractor]).mean() if distractor.any() else background
    probability = value.sigmoid()
    dice = 1 - (2 * probability[target].sum() + 1) / (probability.sum() + target.sum() + 1)
    return .5 * positive + .25 * same_class + .25 * background + .25 * dice


def contrast_loss(correct_logits, wrong_logits, target, margin=1.):
    """同帧同类错误方位必须在该目标上获得更低 logit。"""
    gap = (correct_logits.reshape(target.shape)[target]
           - wrong_logits.reshape(target.shape)[target]).mean()
    return F.softplus(margin - gap), gap


def spatial_pair_loss(fusion, bev, first_embedding, wrong_embedding, grid, gt, target, cls):
    """共享真实 BEV 比较两条语言指令，额外分支不重复运行视觉主干。"""
    first = fusion.forward_with_details(bev, first_embedding)
    wrong = fusion.forward_with_details(bev, wrong_embedding)
    correct_logits = project_attention(torch.logit(first["attention"].clamp(1e-5, 1 - 1e-5)), grid)
    wrong_logits = project_attention(torch.logit(wrong["attention"].clamp(1e-5, 1 - 1e-5)), grid)
    supervised = balanced_instance_loss(correct_logits, gt, target, cls)
    contrast, gap = contrast_loss(correct_logits, wrong_logits, target)
    return supervised + .5 * contrast, {"balanced": supervised, "contrast": contrast, "gap": gap}

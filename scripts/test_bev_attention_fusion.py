"""
功能：验证 BEV 注意力掩码与特征张量的融合
输入：outputs/bev_attention_front_right.npy
输出：特征形状、融合权重与 GPU 计算结果

注意：当前使用模拟 BEV 特征，并非模型实际输出。
"""

import numpy as np
import torch

# 1. 自动选择 GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 2. 读取规则法生成的注意力矩阵
mask_np = np.load(
    "outputs/bev_attention_front_right.npy"
)

assert mask_np.shape == (150, 150)

mask = torch.from_numpy(mask_np).float().to(device)

# 3. 创建模拟 BEV 特征
# B=1: 批次大小
# C=256: 特征通道数
# H/W=150: BEV 网格大小
B, C, H, W = 1, 256, 150, 150

bev_features = torch.ones(
    B, C, H, W,
    device=device
)

# 4. 添加批次维度与通道维度
# (150,150) -> (1,1,150,150)
mask = mask.unsqueeze(0).unsqueeze(0)

# 5. 进行简单的注意力加权融合
# 高注意力区域增强，低注意力区域略微抑制
fusion_weight = 0.8 + 0.7 * mask
enhanced_bev = bev_features * fusion_weight

# 6. 检查融合前后结果
print("========== BEV 特征融合测试 ==========")
print("运行设备:", device)
print("原始特征形状:", tuple(bev_features.shape))
print("注意力形状:", tuple(mask.shape))
print("增强特征形状:", tuple(enhanced_bev.shape))

print("原始特征均值:", bev_features.mean().item())
print("增强特征均值:", enhanced_bev.mean().item())

print("最小融合系数:", fusion_weight.min().item())
print("最大融合系数:", fusion_weight.max().item())

assert enhanced_bev.shape == bev_features.shape
assert torch.isfinite(enhanced_bev).all()

print("特征融合测试通过！")

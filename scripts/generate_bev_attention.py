"""
功能：基于 DriveLM 解析结果生成 BEV 空间注意力掩码
输入：outputs/drivelm_attention_targets.json
输出：
  1. outputs/bev_attention_front_right.png
  2. outputs/bev_attention_front_right.npy

说明：
当前使用人工定义的方向映射与高斯权重。
这是规则法注意力原型，不是神经网络学习得到的注意力。
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. 设置 BEV 网格尺寸，与 BEVFormer-Small 配置一致
BEV_H = 150
BEV_W = 150

# 原型中假设显示区域为自车周围 ±50 米
# 绘图坐标：右侧为正 X，前方为正 Y
# 对应常见自车坐标：绘图 X=-ego_y，绘图 Y=ego_x
RANGE = 50

# 2. 预定义语言方向对应的关注中心（单位：米）
# 这里只提供粗略方位，不代表真实目标坐标
DIRECTION_CENTERS = {
    "front": (0, 30),
    "front right": (25, 25),
    "front left": (-25, 25),
    "right": (30, 0),
    "left": (-30, 0),
    "back": (0, -30),
    "back right": (25, -25),
    "back left": (-25, -25),
}

# 3. 读取 DriveLM 语义解析结果
with open("outputs/drivelm_attention_targets.json",
          "r", encoding="utf-8") as f:
    records = json.load(f)

# 查找第一条右前方语义样本
record = next(
    (item for item in records
     if item["direction"] == "front right"),
    None
)

if record is None:
    raise ValueError("没有找到 front right 方向的问答")

direction = record["direction"]
center_x, center_y = DIRECTION_CENTERS[direction]

# 4. 建立 150 × 150 的 BEV 网格
coords_x = np.linspace(-RANGE, RANGE, BEV_W)
coords_y = np.linspace(-RANGE, RANGE, BEV_H)
X, Y = np.meshgrid(coords_x, coords_y)

# 5. 使用二维高斯函数生成平滑的空间关注权重
sigma = 14.0

attention = np.exp(
    -(
        (X - center_x) ** 2 +
        (Y - center_y) ** 2
    ) / (2 * sigma ** 2)
)

attention = attention.astype(np.float32)

# 6. 保存数值矩阵，供后续特征融合实验使用
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

np.save(
    output_dir / "bev_attention_front_right.npy",
    attention
)

# 7. 绘制注意力热力图
fig, ax = plt.subplots(figsize=(8, 7))

img = ax.imshow(
    attention,
    origin="lower",
    extent=[-RANGE, RANGE, -RANGE, RANGE],
    cmap="inferno",
    vmin=0,
    vmax=1
)

# 红色三角形表示自车位置与朝向
ax.scatter(
    0, 0,
    color="cyan",
    marker="^",
    s=130,
    label="Ego Vehicle"
)

ax.set_title("Rule-based BEV Attention - Front Right")
ax.set_xlabel("Right (+) / Left (-), metres")
ax.set_ylabel("Forward (+) / Backward (-), metres")
ax.legend(loc="lower left")
ax.grid(alpha=0.2)

fig.colorbar(img, ax=ax, label="Attention Weight")

plt.tight_layout()
plt.savefig(
    output_dir / "bev_attention_front_right.png",
    dpi=150
)
plt.close()

print("========== BEV Attention Mask ==========")
print("Direction:", direction)
print("Objects:", record["objects"])
print("Mask shape:", attention.shape)
print("Min weight:", float(attention.min()))
print("Max weight:", float(attention.max()))
print("Saved PNG and NPY to outputs/")

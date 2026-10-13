# BEVFormer-Small 原理、核心代码结构与 BEV 特征生成过程

## 1. 核心思想（一句话）

BEVFormer 把多路相机看到的不同角度图像，**通过带有几何投影的注意力机制融合成车顶俯视的 BEV 特征**，再利用 BEV 特征做 3D 目标检测或 BEV 地图分割。它不是简单把六张照片拼起来。

## 2. 工作流程

```text
六路相机图像 (N_cam=6)
      ↓  图片预处理、归一化、缩放、标定
Backbone (ResNet 等) + Neck (FPN 多尺度特征)
      ↓  多视角多尺度图像特征
BEV Queries (地面网格上的可学习查询向量)
      ├── Temporal Self-Attention   历史 BEV 与当前 BEV 交互
      └── Spatial Cross-Attention   借助 3D 参考点从六路图像采样信息
      ↓
BEV Encoder → BEV Embedding / BEV 特征图
      ├── 3D 检测头：类别、位置、长宽高、朝向、速度 → 3D Boxes
      └── BEV 分割头：每个 BEV 像素的类别 logits → 地图类别图
```

**注意**：检测与分割头来自当前本地的不同仓库配置，不能理解成对同一张输出图直接计算两种指标。

## 3. 四个关键机制

### 3.1 BEV Query（网格查询）

在车身周围划分规则的俯视网格，每个网格位置对应一个待学习的 Query。模型在训练过程中学到“这个位置应该从哪些相机特征中提取什么信息”。**Query 是抽象特征向量，不是一条真实激光雷达射线**。

### 3.2 Spatial Cross-Attention（空间交叉注意力）

先为 BEV 网格上的空间位置构造一组 3D 参考点，再通过相机外参与内参把它们投影到能看到该位置的图像平面，从多尺度特征中进行注意力采样并融合。这样前、后、左、右相机提供的信息能汇总到统一俯视坐标。

### 3.3 Temporal Self-Attention（时间自注意力）

连续驾驶视频中，可以把上一次的 BEV 特征按车辆位姿变化与当前帧对齐、进行时序交互。它能增强时间连续性，也更容易受到**错误帧顺序或错误车辆位姿**的影响。

本阶段 DriveLM 799 帧是稀疏问答关键帧，因此实际测试关闭了历史 BEV 复用（`video_test_mode=False`）；**并不等于修改或删除网络内部的 Transformer 模块**。当前分数因此只代表**独立帧协议**。

### 3.4 检测与分割任务头

- 检测：BEV 特征进入 3D box head，解码出物体 `bbox / class / score`。本地单帧结果保留 300 个候选框；阈值 ≥0.30 主要用于展示，正式 mAP 评估不使用这个阈值直接截断。
- 分割：独立 BEVFormer-seg 仓库分割头 `SegEncode(inC=256, outC=4)` 输出四通道 logits；因为 `CrossEntropyLoss(use_sigmoid=False)`，按类别通道取 `argmax` 得到地图类别图。

## 4. 本地关键源码：看什么、对应什么功能

| 文件路径（从项目根目录） | 看什么 | 关键字段/函数 |
|---|---|---|
| `BEVFormer/projects/configs/bevformer/bevformer_small.py` | 检测基线模型配置 | `bev_h=150`, `bev_w=150`, data pipeline, test dataset |
| `BEVFormer/projects/mmdet3d_plugin/bevformer/modules/transformer.py` | BEV Transformer 的核心前向 | `get_bev_features()`、BEV Query 与 Encoder 输出 |
| `BEVFormer/projects/mmdet3d_plugin/bevformer/modules/encoder.py` | BEVEncoder 和各层 | `BEVFormerEncoder`、`BEVFormerLayer`、投影/融合路径 |
| `BEVFormer/projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py` | 空间 cross-attention | 多路相机可见性、DeformableAttention |
| `BEVFormer/projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py` | 历史 BEV 信息融合 | 当前查询与前序 BEV |
| `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py` | 地图分割配置 | `seg_encoder`, `loss_seg`, `map_grid_conf`, `video_test_mode` |
| `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py` | BEV 特征切片与分割头 | `BevFeatureSlicer`, `seg_preds` |
| `BEVFormer-seg/projects/mmdet3d_plugin/datasets/utils/rasterize.py` | 真值地图矢量转栅格 | `preprocess_map`, `line_geom_to_mask` |

路径与类名来自前面实际 `grep`/代码核查；不同版本的类定义可能调整，**最终以本地文件为准**。

## 5. 实际 BEV 特征张量与可视化

在检测基线使用 `get_bev_features()` 截取的单帧输出记录为：

- 保存的特征 `bev_feature.npy`：**`(150, 150, 256)`**，即 150×150 网格、每格 256 个通道。
- `bev_feature_channels.png`：展示通道 0 / 64 / 128 / 192 的分布。不同通道对应不同学习到的响应；**不是直接对应“车/人/道路”这样的语义类别**。
- `bev_feature_heatmap.png`：对每个位置的通道向量计算 L2 范数得到响应强度，是**特征强弱图，不是检测准确率或分割预测概率**。

![BEV 特征热力图](../images/2026-10-12/bev_feature_heatmap.png)

![四个 BEV 特征通道](../images/2026-10-12/bev_feature_channels.png)

## 6. 检测网格与分割网格为什么不同？

- 检测 BEV 中间特征：150×150×256（可学习特征空间）。
- 分割 GT 与预测输出：`(200, 400)` 的二维类别栅格，对应四个互斥类别（见下表）。
- 中间特征形状与最终输出标签大小**不必相同**。分割头中存在 BEV feature slicing、重采样/映射。

| 类别索引 | 语义名称（代码） | 含义 |
|---|---|---|
| 0 | `background` | 非目标区域 |
| 1 | `divider` | 车道/道路分隔要素 |
| 2 | `ped_crossing` | 人行横道 |
| 3 | `boundary` | 道路边界 |

当前 `segmentation_classes()` 对 `(1,4,200,400)` 执行 `np.argmax(x[0], axis=0)`，最终得到 `(200,400)`，标签 0–3；全量 IoU 使用累计混淆矩阵计算。**因此没有证据表明当前低 mIoU 是 sigmoid/argmax 解码或平均口径错误造成的。**

## 7. 当前现象与开放问题

目前可确认分割模型已经生成道路结构，但 Mini 和 799 帧两个实验都出现**预测与真值位置、形状差异**。针对性排查已检查了 CAN Bus、模型几何、BEV slicer 的网格采样和输出类别；低 mIoU 的根因**仍未证实**。可优先做的是对齐**训练时约定的空间坐标和预测后处理**，并对照作者的时序完整评估协议。

不要为了让分数好看而对评估集做人工选择的平移/翻转；诊断性移动只能作为定位问题的证据，不可作为正式指标。

## 8. 官方参考资料

- [BEVFormer 原仓库与论文入口](https://github.com/fundamentalvision/BEVFormer)
- [BEVFormer 论文](https://arxiv.org/abs/2203.17270)
- [官方运行与评估文档](https://github.com/fundamentalvision/BEVFormer/tree/master/docs)

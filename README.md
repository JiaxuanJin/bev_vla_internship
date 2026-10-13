# BEV / DriveLM / VLA 线上实习项目

本仓库记录自动驾驶感知与视觉语言方向的阶段性学习、实验、评估和问题排查工作。

当前重点：nuScenes、BEVFormer、BEV Segmentation、DriveLM。
VLA 相关内容将在后续阶段持续补充。

## 1. 主要工作

- nuScenes Mini 数据读取与可视化
- BEVFormer 环境搭建及推理验证
- BEV 特征提取和可视化
- BEV Segmentation 预测与指标评估
- Temporal ON/OFF 消融对照
- DriveLM 与 BEVFormer 数据对齐检查
- 实验异常分析与阶段成果验收

## 2. 阶段性实验结果

已完成 nuScenes Mini 子集上的 BEV 分割测试。

| 实验配置 | 前景 mIoU |
|---|---:|
| Temporal ON | 6.38% |
| Temporal OFF | 5.99% |

两组实验均处理了 81 帧。结果仅用于当前实验环境下的诊断和消融分析，不代表完整数据集上的正式基准性能。

详细过程见 docs/ 中的中文 Logbook。

## 3. 仓库结构

- scripts/：精选实验与评估脚本
- scripts/ablation/：消融对照实验
- docs/：每日实验记录及排查报告
- docs/images/：实验截图和可视化结果
- results/：小型 JSON、CSV 评估文件

## 4. 运行说明

本仓库主要用于展示实验流程和阶段成果，并非独立部署包。

运行相关脚本需要另行准备：
- nuScenes 数据集
- 对应的 BEVFormer / BEVFormer-Seg 工程
- 匹配的 Python、PyTorch、MMCV 等依赖
- 对应配置文件及模型权重

数据集、模型权重和第三方完整源码不包含在本仓库中。部分脚本需要根据本地环境调整路径及依赖配置。

## 5. 项目来源与说明

本项目基于 BEVFormer、BEVFormer-Seg、MMDetection3D 和 nuScenes 等研究及开源生态开展实验。

仓库保存的是实验过程中筛选的脚本、运行记录与分析成果，不主张第三方模型架构或原始实现的所有权。相关第三方依赖应遵循各自许可证。

## 6. 后续计划

- 完善实验复现说明
- 继续 BEV 特征与模型误差分析
- 深入 DriveLM 数据对齐和多模态研究
- 逐步探索 VLA 相关实验

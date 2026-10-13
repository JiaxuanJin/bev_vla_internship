# 05 第一阶段复现与 GitHub 交付说明

## 1. 交付状态

整理日期：2026-10-13

第一阶段 BEVFormer / DriveLM 实验代码、中文实验文档、关键可视化图片及部分 JSON 评估结果已经整理并提交至 GitHub 私有仓库。

仓库：JiaxuanJin/bev_vla_internship

## 2. 已完成工作

- nuScenes Mini 数据读取与可视化
- BEVFormer-Small 环境兼容性调试及预训练权重加载
- BEV 特征提取、注意力模块测试与可视化
- BEV 分割实验及 Temporal ON/OFF 对照
- DriveLM 799 帧与 nuScenes 官方 Validation 划分对齐
- DriveLM 799 帧 3D 检测和 BEV 分割子集评估
- nuScenes 完整 Validation 6019 帧检测评估
- DriveLM VQA 初步问答样本整理
- 阶段一 Logbook、周报和关键实验结果归档

## 3. 复现环境

开发环境使用 Windows、WSL2 Ubuntu 22.04、NVIDIA RTX 5080 Laptop GPU，以及 PyTorch、MMCV、MMDetection3D 和 BEVFormer。

项目源码、依赖、模型权重和数据集需按对应版本配置。仓库保留实验脚本和必要的兼容性修改补丁。

## 4. 数据与结果边界

DriveLM 799 帧实验属于独立帧子集评估，与完整 nuScenes 官方 Validation 评估协议不同，不进行直接性能横向比较。

DriveLM VQA 已进行样本准备，但尚未完成语言模型推理与问答性能评估。

大型数据集、模型权重及部分原始推理输出不包含在本 GitHub 仓库中。

## 5. 交付核查

主要文档位于 docs/ 和 docs/stage1/。

可视化图片位于 docs/images/。

主要实验脚本位于 scripts/。

已归档的小型指标文件位于 results/。

兼容性修改补丁位于 patches/。

原始运行日志和大型中间结果保留在本地实验环境中。

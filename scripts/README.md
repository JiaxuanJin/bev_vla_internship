# 实验脚本目录说明（Scripts Guide）

> 项目：BEV / DriveLM / VLA 线上实习  
> 整理日期：2026-10-13  
> 范围：第一阶段环境验证、数据对齐、BEVFormer 检测与分割、DriveLM 问答样本准备、可视化与问题排查

## 1. 目录的主要作用

`scripts/` 存放围绕 BEVFormer、nuScenes 和 DriveLM 编写的**实验辅助脚本**。脚本按照「数据检查与预处理 → 环境与模型验证 → 小规模冒烟测试 → 正式/子集推理 → 指标计算 → 可视化与排查 → 报告归档」的流程组织。主要用于复现实验操作、定位兼容性或数据问题、保存结果及核对实验协议；**并非完整的、可直接部署的自动驾驶系统**。

脚本与第三方 BEVFormer / BEVFormer-Seg 源码、模型权重和本地 nuScenes 数据配合使用。GitHub 仓库不附带大型数据集、模型权重和全部原始推理结果，因此不能将本目录直接视为一键可运行的软件包。部分脚本保留实验机上的默认绝对路径，迁移环境时需要核对源码中的路径、依赖与配置参数。

## 2. 建议优先阅读的关键脚本

| 阅读顺序 | 脚本 | 主要作用 |
|---|---|---|
| 1 | [`prepare_drivelm_v11.py`](./prepare_drivelm_v11.py) | 检查和整理 DriveLM v1.1 数据结构及其与 nuScenes 的对应关系 |
| 2 | [`check_drivelm_val_official_split.py`](./check_drivelm_val_official_split.py) | 核对 DriveLM 验证帧是否属于 nuScenes 官方验证划分，避免训练/验证集混用 |
| 3 | [`build_drivelm_799_bevformer_pkl.py`](./build_drivelm_799_bevformer_pkl.py) | 为 DriveLM 799 帧构造 BEVFormer 推理索引，区分真实场景标识和独立帧推理策略 |
| 4 | [`run_drivelm_799_det_all.py`](./run_drivelm_799_det_all.py) | 执行 799 帧相机版 BEVFormer 3D 检测，保存每帧原始预测用于后续评估 |
| 5 | [`evaluate_drivelm_799_det_subset.py`](./evaluate_drivelm_799_det_subset.py) | 基于 799 帧子集的预测与真值统计 3D 检测 mAP、NDS 等指标 |
| 6 | [`run_drivelm_799_seg_all.py`](./run_drivelm_799_seg_all.py) | 执行 799 帧独立帧 BEV 地图分割推理和 IoU 统计 |
| 7 | [`fix_and_run_bevformer_val_6019.py`](./fix_and_run_bevformer_val_6019.py) | 针对完整 nuScenes Validation 的兼容性问题组织正式检测评估流程 |
| 8 | [`drivelm_vqa_baseline.py`](./drivelm_vqa_baseline.py) | 构建跨场景的六相机问答样本；另提供需要视觉语言模型权重的推理入口 |

## 3. 全部脚本按用途分类

### 3.1 数据读取、检查与索引准备

| 脚本 | 用途概述 |
|---|---|
| [`inspect_drivelm.py`](./inspect_drivelm.py) | 初步查看 DriveLM 标注 JSON 的组织方式、字段及问答内容，用于理解数据格式 |
| [`prepare_drivelm_v11.py`](./prepare_drivelm_v11.py) | 整理 DriveLM v1.1 标注数据，检查图像与样本对应关系，为后续实验做数据准备 |
| [`match_drivelm_nuscenes.py`](./match_drivelm_nuscenes.py) | 匹配 DriveLM 与 nuScenes 的场景/关键帧标识，核对两个数据集的关联关系 |
| [`check_drivelm_image_links.py`](./check_drivelm_image_links.py) | 检查 DriveLM 标注引用的相机图片能否映射到本地 nuScenes 图片文件 |
| [`check_drivelm_val_official_split.py`](./check_drivelm_val_official_split.py) | 对 DriveLM 验证帧进行官方 train/val 划分核查；属于数据一致性验证，不是模型评分 |
| [`build_drivelm_bevformer_pairs.py`](./build_drivelm_bevformer_pairs.py) | 建立 DriveLM 与 BEVFormer 样本之间的配对关系，供联合展示和数据追踪使用 |
| [`check_drivelm_bevformer_alignment.py`](./check_drivelm_bevformer_alignment.py) | 检查配对样本的 token、场景与图像关系是否一致 |
| [`build_drivelm_799_bevformer_pkl.py`](./build_drivelm_799_bevformer_pkl.py) | 生成 799 帧推理索引 PKL；保留官方场景信息，同时支持不跨稀疏帧复用历史 BEV 的索引形式 |
| [`inspect_drivelm_799_inference_ready.py`](./inspect_drivelm_799_inference_ready.py) | 在 GPU 推理前检查 799 帧所需的图像、元数据、配置及输入文件准备状态 |
| [`test_drivelm_799_bevformer_loader.py`](./test_drivelm_799_bevformer_loader.py) | 对 799 帧 BEVFormer 数据加载流程开展冒烟测试，重点检查模型输入能否构建 |
| [`run_drivelm_official_preprocess.py`](./run_drivelm_official_preprocess.py) | 调用/组织 DriveLM 相关的 nuScenes 官方元数据处理与预处理步骤 |
| [`prepare_bev_seg_data.py`](./prepare_bev_seg_data.py) | 为 BEV 地图分割任务整理数据配置、地图标签及加载所需的数据文件 |

### 3.2 环境、模型和权重验证

| 脚本 | 用途概述 |
|---|---|
| [`bevformer_compat/test_bevformer_build.py`](./bevformer_compat/test_bevformer_build.py) | 检查当前环境能否构建 BEVFormer 模型，定位依赖与插件兼容问题 |
| [`bevformer_compat/test_bevformer_weights.py`](./bevformer_compat/test_bevformer_weights.py) | 检查模型与预训练检测权重的加载及参数兼容情况 |
| [`run_bevformer_test_compat.py`](./run_bevformer_test_compat.py) | 测试 BEVFormer 推理入口与当前 PyTorch/MMCV 组合的兼容性 |
| [`inspect_bev_seg_checkpoint.py`](./inspect_bev_seg_checkpoint.py) | 安全读取 BEV 分割权重，检查结构、参数键和分割相关权重，不修改原文件 |
| [`verify_bev_seg_build.py`](./verify_bev_seg_build.py) | 在 CPU 上构建分割模型并对照权重键和维度；不代表已经完成 GPU 分割推理 |
| [`test_bev_seg_dataset.py`](./test_bev_seg_dataset.py) | 测试分割数据集读取、测试管线以及地图语义真值 `semantic_indices` 的生成 |
| [`check_stage1_submission.py`](./check_stage1_submission.py) | 第一阶段交付文件、结果路径等方面的检查工具；适用于提交前核验 |

### 3.3 BEVFormer 3D 检测与时序验证

| 脚本 | 用途概述 |
|---|---|
| [`run_drivelm_799_det_one_sample.py`](./run_drivelm_799_det_one_sample.py) | 先对 DriveLM 一帧执行真实 GPU 3D 检测，核对输出结构与推理入口 |
| [`run_drivelm_799_det_all.py`](./run_drivelm_799_det_all.py) | 在单帧测试通过后推理 799 帧，保存候选框、类别、置信度与逐帧耗时；支持断点续跑 |
| [`evaluate_drivelm_799_det_subset.py`](./evaluate_drivelm_799_det_subset.py) | 将 799 帧检测输出与对应真值匹配，计算子集检测性能；与完整官方验证集成绩分开报告 |
| [`fix_and_run_bevformer_val_6019.py`](./fix_and_run_bevformer_val_6019.py) | 排查并绕开旧版测试流程与新环境之间的兼容问题，运行完整 6,019 帧 Validation 评估 |
| [`test_bevformer_temporal_pair.py`](./test_bevformer_temporal_pair.py) | 对相邻帧进行时序状态/历史 BEV 相关验证，检查时序机制的使用条件 |
| [`test_bevformer_temporal_batch.py`](./test_bevformer_temporal_batch.py) | 扩大时序测试的样本范围，观察批量/连续推理中的状态传递和运行表现 |

### 3.4 BEV 地图分割、指标和空间对齐排查

| 脚本 | 用途概述 |
|---|---|
| [`run_bev_seg_one_sample.py`](./run_bev_seg_one_sample.py) | nuScenes Mini 单帧地图分割；保存预测、真值、对比图和单帧 IoU，用于功能验证 |
| [`run_bev_seg_minival.py`](./run_bev_seg_minival.py) | Mini 验证集约 81 帧逐帧分割，按累计交并比计算前景 mIoU，并导出 CSV/JSON |
| [`run_drivelm_799_seg_all.py`](./run_drivelm_799_seg_all.py) | 使用 DriveLM 799 帧的独立帧设置进行 BEV 分割与逐帧/总体指标整理 |
| [`diagnose_bev_seg_miou.py`](./diagnose_bev_seg_miou.py) | 对偏低的地图分割 IoU 开展诊断，辅助区分标签、输出、类别和几何因素 |
| [`audit_bev_seg_geometry.py`](./audit_bev_seg_geometry.py) | 非破坏性核对场景顺序、车辆位姿、`can_bus` 和输入元数据，定位地图分割的空间错位风险 |
| [`audit_bev_slicer_coordinates.py`](./audit_bev_slicer_coordinates.py) | 用人工构建的坐标通道测试 `BevFeatureSlicer` 的采样/旋转/翻转关系；不等于验证真实预测方向 |

### 3.5 BEV 特征、注意力与可视化

| 脚本 | 用途概述 |
|---|---|
| [`extract_bev_features.py`](./extract_bev_features.py) | 提取模型中的 BEV 中间特征，用于后续热力图、通道分析或注意力研究 |
| [`extract_attention_targets.py`](./extract_attention_targets.py) | 从现有实验数据中整理注意力目标/输入信号，辅助后续融合模块实验 |
| [`generate_bev_attention.py`](./generate_bev_attention.py) | 生成/处理 BEV 空间注意力表示，用于验证注意力增强思路 |
| [`test_bev_attention_fusion.py`](./test_bev_attention_fusion.py) | 在模拟 BEV 特征上测试注意力融合张量运算和数值行为；不代表已训练真实融合模型 |
| [`visualize_nuscenes.py`](./visualize_nuscenes.py) | 可视化 nuScenes 场景及多相机图片，直观检查数据读取结果 |
| [`visualize_nuscenes_boxes.py`](./visualize_nuscenes_boxes.py) | 把 nuScenes 3D 标注框展示在相机视图中，辅助理解目标标注 |
| [`visualize_lidar_bev.py`](./visualize_lidar_bev.py) | 从 LiDAR 数据生成俯视角可视化，辅助理解 BEV 坐标和道路场景 |
| [`visualize_drivelm.py`](./visualize_drivelm.py) | 展示 DriveLM 的图像与标注/问答关联，用于初期数据探索 |
| [`visualize_bevformer_predictions.py`](./visualize_bevformer_predictions.py) | 将 BEVFormer 3D 检测预测可视化到图像和俯视视图，供定性检查 |

### 3.6 实验记录和报告整理

| 脚本 | 用途概述 |
|---|---|
| [`make_logbook_day2.py`](./make_logbook_day2.py) | 生成阶段实验日志初稿，汇总当日已保存的实验材料 |
| [`make_logbook_figures.py`](./make_logbook_figures.py) | 整理或生成 Logbook 所需的实验图像，便于在 Markdown 中引用 |
| [`make_stage1_reports.py`](./make_stage1_reports.py) | 根据阶段一已有的实验文件整理技术说明与报告内容；需要核对引用的具体结果和协议 |

## 4. 主要实验结果与脚本的对应关系

| 已完成工作 | 主要脚本 | 结果说明 |
|---|---|---|
| nuScenes Mini BEV 分割 | `run_bev_seg_one_sample.py`、`run_bev_seg_minival.py` | Mini 验证集 81 帧前景 mIoU：Temporal ON 约 6.38%；OFF 对照约 5.99%，具体数值以各次运行记录为准 |
| DriveLM 799 帧 3D 检测 | `run_drivelm_799_det_all.py`、`evaluate_drivelm_799_det_subset.py` | 独立帧子集检测 mAP 约 **13.39%**、NDS 约 **25.79%** |
| DriveLM 799 帧 BEV 分割 | `run_drivelm_799_seg_all.py` | 独立帧子集前景 mIoU 约 **5.12%**（含背景约 22.22%） |
| nuScenes 完整验证集 3D 检测 | `fix_and_run_bevformer_val_6019.py` | 已完成 6,019 帧评估；完整指标以正式评估日志为准 |
| DriveLM VQA 样本准备 | `drivelm_vqa_baseline.py prepare` | 已准备 20 条跨场景问答样本及对应图像核查结果；**尚未完成 VLM 推理与正式问答评估** |

> **指标口径说明**：Mini、DriveLM 799 帧独立帧子集与完整 nuScenes Validation 的数据规模和推理协议不同，不能把 mAP、NDS、mIoU 直接混合比较。尤其不能把数据准备、输出图片或候选框数量当作模型正式评分。

## 5. 复现注意事项

1. **运行目录**：多数脚本以 `~/projects/bev_vla` 为实验工程根目录，而本 GitHub 仓库 `bev-vla-internship` 是轻量成果归档目录；直接在归档仓库内运行可能缺少第三方工程、模型、数据或输出目录。
2. **依赖环境**：部分脚本需要 `.venv-bevformer`（PyTorch / CUDA / MMCV / MMDetection3D）；DriveLM VQA 推理另需视觉语言模型及相关依赖。不能仅凭脚本文件判断环境已具备。
3. **两种任务**：3D Detection、BEV Map Segmentation、DriveLM VQA 属于不同实验链路，使用不同输出和评价指标。
4. **区分验证与正式实验**：文件名中含 `test`、`inspect`、`audit`、`verify` 的脚本主要服务于测试或排查；测试通过不等于最终任务全部完成。
5. **文件和权限**：原始数据、模型权重与大规模中间结果存于本地实验环境，未全部提交至 GitHub。复现时应确认相关数据许可和环境配置。

## 6. 相关文档

- [阶段一交付导航](../docs/stage1/00_阶段一交付导航.md)
- [DriveLM 数据集结构与标注说明](../docs/stage1/01_DriveLM数据集结构与标注说明.md)
- [BEVFormer 原理与代码结构](../docs/stage1/02_BEVFormer原理与代码结构.md)
- [基线性能测试报告](../docs/stage1/03_基线性能测试报告.md)
- [第一阶段周报（2026-10-13）](../docs/stage1/06_第一阶段周报_2026-10-13.md)
- [2026-10-13 实验日志](../docs/logbook_2026-10-13.md)

---

*说明：本文件用于说明现有实验脚本的定位、相互关系与结果适用范围，不是所有脚本的逐行 API 手册。运行参数与实际行为应以对应源码、实验日志和输出记录为准。*

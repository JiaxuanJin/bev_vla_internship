# BEVFormer-Small BEV 分割低 mIoU 专项排查日志（CN）

**日期：** 2026-10-12  
**项目：** BEVFormer / DriveLM 实习项目 · 第一阶段（环境搭建与基线复现）  
**日志类型：** 专项问题排查（独立于 `docs/logbook_2026-10-12.md`）  
**状态：** 已完成评估链路对照、时序消融与基础几何核验；**低 mIoU 根因尚未确定**。

## 1. 问题背景与排查目标

在 RTX GPU 环境中运行 BEVFormer-Small 预训练分割模型，对 **nuScenes v1.0-mini / mini_val 的 81 帧**进行道路 BEV 语义分割。可视化发现：模型能够预测出部分道路形状，但与真实地图 Ground Truth（GT）的空间覆盖范围和位置差异较大，尤其是道路分隔线及道路边界。基线的**前景 mIoU 约为 6.38%**，需要确认低分来自评估计算、模型权重、历史 BEV、相机标定、BEV 特征映射，还是数据集差异。

> **重要口径：** 作者公开报告约 **42.62% mIoU**，但不是我们在 nuScenes Mini 81 帧上得到的数值。当前测试既不是完整 nuScenes 官方验证集评估，也不能将两个数字当作同条件性能对比。不要把 6.38% 描述成“作者模型原本只有 6.38%”。

## 2. 环境、模型及数据

| 项目 | 本次实验设置 |
| --- | --- |
| 项目根目录 | `~/projects/bev_vla` |
| Python 环境 | `.venv-bevformer`，PyTorch 2.8 + CUDA |
| 分割代码仓库 | `BEVFormer-seg/` |
| 分割权重 | `/mnt/g/nus/bevformer_small_seg.pth` |
| 模型配置 | `BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py` |
| 数据 | nuScenes v1.0-mini，mini_val，**81 帧 / 2 个连续场景（40 + 41）** |
| 地图类别 | Background、Divider、Pedestrian Crossing、Boundary |
| 分割地图范围 | X ∈ [-30, 30] m，Y ∈ [-15, 15] m；0.15 m/像素；输出 200 × 400 |
| 时序开关 | 基线 `video_test_mode=True` |

权重元数据显示 `epoch=4`、`iter=112520`、`state_dict` 含 **883 个参数字段**；证明权重不是随机初始化，但仅凭这些字段不能判断它是否对应作者公开成绩的完全相同权重版本。

## 3. 排查过程与关键证据

### 3.1 基线结果与混淆矩阵分析

自写脚本 `scripts/run_bev_seg_minival.py` 已完成 **81/81** 帧推理，使用全数据集累计混淆矩阵计算各类别 Intersection/Union，而不是直接平均每帧 IoU。

| 类别 | Temporal ON IoU | 主要表现 |
| --- | ---: | --- |
| Background | **77.52%** | 背景较容易预测 |
| Divider | **2.23%** | 约 **80.71%** 的真实分隔线像素预测为背景 |
| Pedestrian Crossing | **8.73%** | 约 **57.49%** 的真实横道像素预测为背景 |
| Boundary | **8.17%** | 约 **80.77%** 的真实边界像素预测为背景 |
| **前景 mIoU（排除背景）** | **6.38%** | 三个道路前景类的平均 IoU |

**判断：** 不是简单的两个前景类别编号互换；大量道路 GT 像素与预测道路未重合。该现象可能由漏检、位置偏差、形状预测不足或共同的数据处理问题造成，不能单凭混淆矩阵认定根因。

### 3.2 原作者评估流程与自写流程对照

使用相同的 Mini 81 帧、同一份权重和模型配置，对照执行作者项目 `projects/mmdet3d_plugin/apis/test.py` 中的分割评估流程（`onehot_encoding`、`semantic_map_iou_val`）：

| 指标 | 自写评估 | 作者评估流程 |
| --- | ---: | ---: |
| Divider IoU | 2.23% | 2.23% |
| Pedestrian Crossing IoU | 8.73% | 8.73% |
| Boundary IoU | 8.17% | 8.17% |
| **前景 mIoU** | **6.38%** | **6.38%** |

对照实验 **81/81 帧已成功计算出指标**。运行过程中发生两项独立兼容性/保存问题：

1. 旧版 MMCV `MMDataParallel` 与 PyTorch 2.8 的 CUDA 设备类型接口不兼容，报 `AttributeError: 'int' object has no attribute 'type'`；对该次测试进程增加适配后成功完成推理。
2. 评估指标输出后，复制后的测试入口因缺少 `mmcv` 导入，保存 `official_results.pkl` 时出现 `NameError: name 'mmcv' is not defined`。**该错误发生在全部指标打印之后，影响结果文件保存，不影响已完成的 IoU 计算**；已修复测试入口代码，但不能宣称 `.pkl` 已成功保存。

**判断：** 两套评估计算结果相同，基本排除自写 `argmax`/IoU 统计公式导致低分的嫌疑；但两套测试仍共享模型及数据链路，因此不能排除共同存在的问题。

### 3.3 模型权重、运行配置及时序状态

从预训练 checkpoint 的 `meta.config` 与当前配置中比较 `model`、`map_grid_conf`、`img_norm_cfg`、`test_pipeline`，**四项全部一致**。当前运行配置 `video_test_mode=True`，模型 `forward_test()` 包含以下逻辑：

- 检测到 `scene_token` 改变时，将 `prev_bev` 清空；
- 开启时序时，把上一帧 BEV 传入本帧，并在推理后更新；
- 81 帧按两个场景依次排列，分别有 40 帧与 41 帧，第 40 帧为唯一一次场景切换。

这说明时序传递和跨场景重置**在代码层面具有正确机制**，但静态检查本身不能证明每帧运行时的所有状态完全正确。

### 3.4 历史 BEV 消融实验（Temporal ON / OFF）

控制相同数据、权重与评估方法，仅将 `video_test_mode` 从 `True` 调为 `False`，独立保存新结果；两组均完成 **81/81 帧**。

| 指标 | Temporal ON | Temporal OFF |
| --- | ---: | ---: |
| Divider IoU | 2.23% | 1.99% |
| Pedestrian Crossing IoU | 8.73% | 7.20% |
| Boundary IoU | 8.17% | 8.79% |
| **前景 mIoU** | **6.38%** | **5.99%** |

整体开启时序较关闭时序提升约 **0.38 个百分点**。代表帧的**单帧前景 mIoU**如下（仅用于诊断，不等同于累计 mIoU）：

| 帧索引 | Temporal ON | Temporal OFF | 说明 |
| --- | ---: | ---: | --- |
| 0 | 8.65% | 8.65% | 场景 A 首帧，没有历史 BEV |
| 40 | 2.61% | 2.61% | 场景 B 首帧，历史 BEV 重置 |
| 80 | 4.03% | 3.06% | 场景 B 后续帧，时序对预测有帮助 |

另外，按场景统计的**平均单帧 mIoU**：场景 A（40 帧）约 **5.81%**，场景 B（41 帧）约 **4.60%**；两个场景均偏低，非个别场景单独造成。

**判断：** 历史 BEV 会改变预测并带来少量平均增益，但**不是整体低精度的主要解释**；两个模式下均存在明显的道路形状与位置不一致。

### 3.5 相机标定、负深度误判与独立验证

对 81 帧检查：每帧均有六路相机；相机内参和外参矩阵维度、有限性及旋转行列式检查通过；每帧 `can_bus` 长度为 18；**结构性异常数为 0**。

初期测试曾直接把 LiDAR 坐标 `[15,0,0]` 视为“车辆前方 15 米”，所得相机深度为负（约 -0.49 至 -0.93 m），一度怀疑相机外参存储或矩阵转置存在问题。随后：

1. 用 PKL 中保存的原始姿态独立重建旋转，比较误差为 0；
2. 再与 **nuScenes 官方 `calibrated_sensor`、`ego_pose`** 记录对比，相机和 LiDAR 旋转误差均为 0；
3. 改用 **Ego 车辆坐标系**定义前方 `[15,0,0]` 与后方 `[-15,0,0]`，逐级变换到 LiDAR、Camera 后重新测试，获得合理的正深度：

| 帧索引 | CAM_FRONT 深度 | CAM_BACK 深度 |
| --- | ---: | ---: |
| 0 | 13.573 m | 14.955 m |
| 40 | 13.445 m | 14.961 m |
| 80 | 13.431 m | 14.964 m |

例如第 0 帧前相机投影像素约 `(843.12, 616.99)`，位置合理。

**纠正结论：** 负深度来自早期测试点坐标系假设不正确，**不是已证实的 PKL 标定 Bug**。不得据此前误判去转置矩阵或修改原始标定。当前通过的是基础几何合理性检查，而非对全部空间投影的完整验证。

### 3.6 BEV 特征方向、裁剪器及分割解码网络

检查 `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_seg_head.py`：

- 将 BEV embedding 重塑为特征图；
- 原作者实现中依次采用 `torch.rot90(..., k=1)`、`torch.flip(..., dims=[3])` 转换特征方向；
- `BevFeatureSlicer` 根据 BEV / 地图范围构建采样网格，并利用 `F.grid_sample(..., mode='bilinear', align_corners=True)` 进行特征采样。

检查 `BEVFormer-seg/projects/mmdet3d_plugin/bevformer/modules/seg_subnet.py`：`SegEncode` 使用 ResNet18 特征提取、多尺度上采样与卷积融合，最终输出 **4 类 × 200 × 400**；该网络内部未发现附加 `rot90`/`flip` 等空间方向操作。

**判断：** 静态检查没有发现显而易见的裁剪、转置和输出尺寸错误；不过这**不等于数学上证明预测与 GT 的空间轴完全对齐**，不得擅自删除训练时已有的旋转/翻转操作。

## 4. 排查总结与尚未解决的事项

**已获得的可靠证据：**

- 完成 Mini 81 帧基线；自写与原作者评估同得约 **6.38%**；
- 完成 Temporal ON/OFF 消融：**6.38% vs 5.99%**；
- 检查权重元数据及四项关键配置，一致；
- 核实场景排序、时序重置代码、六路相机数据结构和基础投影；
- 校正了由于误用 LiDAR 坐标系方向造成的负深度误判；
- 核查分割头特征旋转、裁剪与解码操作，尚未发现明确实现错误。

**未确认的根因：** Mini 场景与作者公开验证条件差异、预训练模型在本数据子集上的预测/泛化能力、深层 BEV 空间映射与目标标签协议之间可能尚存的差异。**不得写成“分割问题已经解决”，也不得为了提高指标对 GT 或输出图做无根据的平移、翻转或刷分操作。**

**下一步计划：** 第一阶段优先整理 GitHub 可审阅交付物（关键脚本、环境和复现说明、真实指标及可视化）；后续若要进一步定位，应在相同验证协议下进行模型/权重对照，或使用预先定义的、独立验证的空间映射诊断，而非不断重复静态 grep 检查。完整官方数据评估受数据准备条件限制，应明确区分 Mini 与完整验证集。

## 5. 关键证据文件与复现位置

| 内容 | 项目内路径 |
| --- | --- |
| Temporal ON 指标 | `outputs/bev_seg/seg_minival_metrics.json` |
| Temporal ON 逐帧 CSV | `outputs/bev_seg/seg_minival_per_frame.csv` |
| Temporal OFF 指标 | `outputs/bev_seg_no_temporal/seg_minival_metrics.json` |
| Temporal OFF 逐帧 CSV | `outputs/bev_seg_no_temporal/seg_minival_per_frame.csv` |
| 原作者流程对照日志 | `outputs/seg_official_minival/official_eval_compat.log` |
| 自写 Mini 评估脚本 | `scripts/run_bev_seg_minival.py` |
| 无时序消融脚本 | `outputs/seg_ablation/run_bev_seg_minival_no_temporal.py` |
| BEV Slicer 几何检查 | `scripts/audit_bev_slicer_coordinates.py` |
| 其他分割诊断 | `scripts/diagnose_bev_seg_miou.py`、`scripts/audit_bev_seg_geometry.py` |

**可视化证据：**

- [Temporal ON · Mini 第 0 帧](images/2026-10-12/seg_comparison_minival_00.png)
- [Temporal ON · Mini 第 40 帧](images/2026-10-12/seg_comparison_minival_40.png)
- [Temporal ON · Mini 第 80 帧](images/2026-10-12/seg_comparison_minival_80.png)
- [Temporal OFF · Mini 第 80 帧](images/2026-10-12/seg_comparison_minival_80_temporal_off.png)


---

**最终阶段结论：** 本次专项排查成功验证了评估流程的一致性，量化了时序信息对 Mini 数据的影响，并排除了基础标定数据损坏及数个明显统计/配置错误的嫌疑；低前景 mIoU 的**根本原因仍未找到**。所有指标和局限均按实际实验条件记录，保留后续可复现和进一步诊断的依据。

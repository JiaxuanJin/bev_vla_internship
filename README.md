# BEVFormer × DriveLM｜自动驾驶 BEV 感知与语言引导注意力

本仓库记录 BEVFormer、nuScenes / DriveLM 及语言引导 BEV 注意力模块的线上实习实验，包括基线复现、模型集成、真实数据评估、根因排查与可复现证据。项目分阶段开展，目前**第一阶段基线复现**和**第二阶段核心模块的工程实现与功能验收**已经完成；完整 VLA 模型训练及稳定的语义定位/感知性能收益**尚未完成验证**。

> **截至 2026-10-22：** 第二阶段依据原始任务书完成 **8 项详细任务、4 项阶段产出**的工程验收；**117 项 CPU pytest 通过**，真实 BEVFormer 双 Head 前向与梯度测试通过，并完成旧协议下 **20 步真实优化更新**及固定 6 帧 Mini Val A/B 对照。该结论仅代表功能实现和运行验证，**不代表语言定位准确或 mIoU 有提升**。

## 项目进展与交付入口

| 阶段 | 工作内容 | 当前进度 | 主要入口 |
|---|---|---|---|
| 第一阶段 | 环境搭建、DriveLM 数据核对、BEVFormer 检测/分割基线复现与评估 | 已完成 | [第一阶段交付导航](docs/stage1/00_阶段一交付导航.md) |
| 第二阶段 | DistilBERT、语义空间映射、BEV Attention/Fusion、多任务 Loss、真实双 Head 集成 | 工程实现与功能验收已完成 | [任务对照](docs/stage2/13_阶段二任务对照.md) · [源码审查](docs/stage2/21_第二阶段最终缺口审查_2026-10-22.md) |
| 后续研究 | BEV/GT 物理配准、可靠语义 grounding、稳定精度收益、DriveLM VQA 与完整 VLA 链路 | 尚未完成 | [技术限制与建议](docs/stage2/21_第二阶段最终缺口审查_2026-10-22.md) |

**第二阶段推荐阅读顺序：** [原始任务书](docs/stage2/02_第二阶段原始任务书.md) → [8+4 任务对照](docs/stage2/13_阶段二任务对照.md) → [架构设计](docs/stage2/11_模块架构设计.md) → [真实结果与图表](results/stage2_delivery_2026-10-23/README.md) → [单元测试报告](docs/stage2/12_单元测试报告.md) → [复现指引](docs/stage2/17_第二阶段复现指引.md)。

## 第二阶段：语言引导 BEV 注意力增强

### 实现与数据流

- **语言编码：** 本地预训练 DistilBERT + Tokenizer、attention mask、masked mean pooling，生成 `[B,768]` 文本向量；受约束英文规则解析负责输入检查与训练目标类别一致性校验。
- **二维语义映射：** `SemanticProjection` 将语言通道映射为 `[B,256]`，再与归一化 BEV 视觉特征交互，形成 `[B,256,150,150]` 的二维语义特征。
- **空间注意力与融合：** 由二维语义特征生成 `[B,1,150,150]` Attention Mask，通过 `BEV + α × BEV × Attention` 残差融合；关闭增强时支持与作者模型一致的精确旁路。
- **真实 BEVFormer 接入：** 在共享 BEV 特征处通过实验级运行时 Hook 注入，增强特征同时进入 **3D Detection** 和 **BEV Segmentation** 两个 Head，不修改作者第三方源码。
- **多任务损失：** Detection Loss + Segmentation Loss + Attention BCE/Dice Supervision；新增可配置的 Attention Total Variation（TV）Regularization，并进行了独立梯度与兼容性测试。

![第二阶段审查版系统架构](docs/images/stage2/stage2_audited_architecture_20261022.svg)

**主要源码：** [`scripts/stage2/`](scripts/stage2/)；统一运行入口：[`scripts/stage2/run_stage2.py`](scripts/stage2/run_stage2.py)。模块分工、张量协议和集成边界见 [架构设计文档](docs/stage2/11_模块架构设计.md)。

### 工程验收与真实实验

| 验证项目 | 已取得的证据 | 结论范围 |
|---|---|---|
| CPU 单元测试 | **117 passed**；包含文本处理、二维映射、Attention、Loss、Hook、低内存和完整性检查 | 工程功能通过 |
| 真实 GPU 前向与反向 | 联合 BEVFormer 双 Head、共享增强 BEV、真实三任务梯度；关闭增强时输出差为 0 | 集成/梯度链通过 |
| 真实微型训练 | nuScenes Mini Train 固定 20 帧，**20 次 `optimizer.step()`**，参数与 AdamW 状态更新可核验 | 证明可优化，不代表收敛 |
| 固定 Mini Val A/B | 6 帧，Original / Untrained / Trained / Shuffled 对照及预测数组只读复核 | 诊断性评估，不是官方完整评估 |
| 新增 TV 正则 | `TV=0.001` 的 GPU smoke 和梯度通过，**0 次新训练更新** | 证明实现与梯度，不证明 TV 训练收益 |

> **版本边界：** 20 步实际训练使用的旧损失协议 **TV=0**；之后新增的 TV 正则只完成了独立 smoke/梯度测试。两部分证据严格分开，不能把旧训练结果称作非零 TV 正则训练结果。

**六帧诊断性分割结果（前景 mIoU）：**

| Original | Untrained | Trained（20 步） | Shuffled Text |
|---:|---:|---:|---:|
| 31.5993% | 31.5993% | 31.5952% | 31.5965% |

训练后前景 mIoU 相对 Original **下降约 0.0041 个百分点**；检测诊断 TP/FP/FN 在 Original、Trained、Shuffled 分支均为 **207 / 599 / 35**。正常文本未稳定优于置乱文本。检测统计使用固定阈值及 XY 中心匹配，**不是官方 nuScenes AP/NDS**；BEV 与地图 GT 的完整物理空间配准仍为 `NOT_VERIFIED`。

![20 步真实训练损失（TV=0）](results/stage2_delivery_2026-10-23/figures/training_loss_20steps.png)

![固定六帧 A/B 汇总](results/stage2_delivery_2026-10-23/figures/ab_metrics_6frames.png)

更多 [原始 GT 与分割对照、Attention 正常/置乱文本热力图及溯源数据](results/stage2_delivery_2026-10-23/README.md)。图表由已有 JSONL/NPZ 离线生成；单帧与历史不同时序、不同 checkpoint、不同评估协议的数据不做直接比较。

### 第二阶段正式交付

| 交付物 | GitHub 路径 |
|---|---|
| 完整增强模块与实验级集成源码 | [`scripts/stage2/`](scripts/stage2/) |
| 模块架构图、设计与说明 | [架构设计](docs/stage2/11_模块架构设计.md) · [SVG 图](docs/images/stage2/stage2_audited_architecture_20261022.svg) |
| 真实训练、A/B 结果与图表 | [图表与证据](results/stage2_delivery_2026-10-23/README.md) |
| 单元测试与严格审查 | [单测报告](docs/stage2/12_单元测试报告.md) · [缺口审查](docs/stage2/21_第二阶段最终缺口审查_2026-10-22.md) |
| 完整复现说明、交付检查与每日记录 | [复现指引](docs/stage2/17_第二阶段复现指引.md) · [Git 交付检查](docs/stage2/22_第二阶段Git交付检查_2026-10-23.md) · [10 月 22 日 Logbook](docs/logbook_2026-10-22.md) |

## 第一阶段：环境搭建与基线复现（2026-10-10—2026-10-13）

- 搭建 WSL2、PyTorch、MMCV、MMDetection3D、BEVFormer 实验环境，并完成 GPU / CUDA 自定义算子验证。
- 完成 nuScenes Mini 数据读取、六相机图像、3D 标注与 LiDAR BEV 可视化，以及 BEVFormer 检测/分割推理。
- 核对 DriveLM v1.1 与 nuScenes 官方 Val 的对应关系：**149 个场景、799 个关键帧，匹配 799/799，场景 token 不一致为 0**。
- 对 DriveLM 799 帧稀疏子集开展独立帧 3D 检测与地图分割评估，并排查 BEV 空间方向、GT 参考系及评价协议问题。
- 准备 DriveLM 语言问答基线的 20 条样本（perception、prediction、planning、behavior 各 5 条），完成图像路径检查；**尚未完成 VQA 模型推理和回答质量评分**。

| 实验 | 指标 | 已记录结果 |
|---|---|---:|
| DriveLM 799 帧独立帧 3D 检测 | mAP / NDS | 13.39% / 25.79% |
| DriveLM 799 帧独立帧地图分割 | 前景 / 含背景 mIoU | 5.12% / 22.22% |
| nuScenes Mini 分割（81 帧） | Temporal ON / OFF 前景 mIoU | 6.38% / 5.99% |

**评估边界：** 799 帧为稀疏、独立帧子集评估，不是完整 nuScenes Val 官方成绩；81 帧 Temporal ON/OFF 属于另一实验协议，不可与 799 帧、第二阶段 6 帧 A/B 直接比较。完整口径与复现细节参见 [第一阶段基线性能报告](docs/stage1/03_基线性能测试报告.md) 和 [阶段交付导航](docs/stage1/00_阶段一交付导航.md)。

## 项目目录

```text
README.md                项目总览与交付入口
scripts/stage2/          第二阶段增强模块、真实集成、训练、审计及复现脚本
stage2/                  早期语言编码/投影兼容模块
tests/                   Python 单元测试
requirements_stage2.txt 第二阶段附加依赖
scripts/                 其他数据处理、推理、评估及图表脚本
docs/stage1/             第一阶段交付
docs/stage2/             第二阶段架构、任务对照、测试、复现及历史审查
docs/images/             真实实验图、架构图与可视化
docs/logbook_*.md        每日中文实验记录
results/                 版本化的小型 JSON/CSV/日志、测试证据和图表
patches/                 第三方工程的本地兼容补丁
```

## 快速复现与外部依赖

在已配置好的 WSL 项目根目录执行 **离线 CPU 单元测试**：

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  .venv-bevformer/bin/python -m pytest tests/ -q -p no:cacheprovider
```

在**原始实验资产已按复现指引准备完整**的机器上，可进一步执行旧实验只读核验（不会重新训练）：

```bash
.venv-bevformer/bin/python scripts/stage2/replay_stage2.py --verify-only
```

以上命令需要相应兼容环境；第二条依赖冻结旧源码与旧实验的本地资产，**仅克隆 Git 仓库不足以运行**。真实 GPU 入口、配置、checkpoint 哈希、数据根目录和环境版本见 [第二阶段复现指引](docs/stage2/17_第二阶段复现指引.md)。

本仓库**不包含**完整 nuScenes / DriveLM 数据集、第三方 BEVFormer 作者工程、大型作者 checkpoint、DistilBERT 预训练文件、Python 虚拟环境、全部融合 checkpoint、六帧 NPZ 或 171 MiB 的独立交付包。这些资产需另行准备；[独立资产清单](results/stage2_delivery_2026-10-23/non_git_assets.json) 说明其位置与边界。`results/` 中的轻量证据可以在线阅读，但不是所有实验原始文件。

## 第二阶段验收边界与后续研究方向

**阶段二交付范围：** 按指导老师原始任务书，已实现 DistilBERT 语言编码、语义到二维 BEV 的映射、空间注意力生成与融合、多任务损失、BEVFormer 实验级集成及单元测试，并交付模块源码、架构说明、集成源码和测试报告。以上属于**工程功能与可运行性验收**，最终是否通过仍以指导老师审阅为准。

- **本阶段已验证：** 117 项 CPU 单元测试、真实 GPU 双 Head 集成与梯度检查；完成 TV=0 协议的 20 步真实参数更新及固定 6 帧 A/B 诊断。新增 TV 正则已验证功能与梯度，尚未验证非零 TV 权重下的训练收益。
- **后续研究验证（非本阶段强制精度指标）：** 自然语言是否能可靠定位 BEV 目标区域、正确执行方向/否定约束、BEV 与地图 GT 的完整物理坐标一致性，以及能否稳定提升感知指标。其中坐标一致性仍是后续有效训练需要优先排查的技术风险。
- **后续开发（不在第二阶段原始交付范围）：** 完整 VLA 模型训练、DriveLM VQA 模型推理与回答质量评价、长期时序实验和大规模多随机种子正式比较。

历史阶段二实验及图像仍可从 [数据图表总览](docs/stage2/数据图表总览_2026-10-21.md)、[完整实验图索引](docs/stage2/全部实验图表索引_2026-10-21.md) 和 [2026-10-21 记录](docs/logbook_2026-10-21.md) 查询；这些历史结果保留原评估设置，不替代当前版本的验收结论。

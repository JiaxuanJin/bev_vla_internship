# 阶段一复现与 GitHub 提交前检查清单

## 1. 验收范围

本阶段交付：运行代码、文档、核心指标文件的路径与可视化示例。**不上传原始 DriveLM/nuScenes 全量数据、模型大权重、虚拟环境及原始 799 份 NPZ**（可在导师要求或合法授权的前提下另行共享）。

需要客观区分：

- **已经完成**：DriveLM 官方问答转换、Mini 与 799 帧基线推理、799 帧自定义子集指标与可视化、原理/数据/报告说明。
- **并未完成且不属于当前声称结果**：重新训练网络、完整 Trainval 感知 PKL 生成、完整 nuScenes 官方时序 Val 基准对齐、DriveLM 语言问答模型精度测试、BEV 分割空间错位根因修复。

## 2. 本机一次性验收（不重跑 GPU）

在项目根目录：

```bash
cd ~/projects/bev_vla
python scripts/check_stage1_submission.py
```

检查将读取真实指标 JSON、QA 转换汇总、主要 Python 文件及 Markdown 图片链接，不修改原始数据。检查状态 JSON 写入 `outputs/stage1_checks/stage1_audit.json`，方便截图保存证据。

如果有缺图、缺脚本或缺结果文件，请依提示处理后再提交；**文档生成完成不代表用户本机已经核验通过**。

## 3. 可复现流程（只列实际用过的入口）

| 阶段 | 命令 | 预期产出 |
|---|---|---|
| 官方 QA 转换 | `python scripts/run_drivelm_official_preprocess.py --mode full` | `full/test_llama.json`, `full/summary.json` |
| 799 帧数据加载检查 | `python scripts/test_drivelm_799_bevformer_loader.py --indices 0,399,798` | `outputs/drivelm_check/drivelm_799_dataloader_smoke.json` |
| 799 帧 3D 检测推理 | `python scripts/run_drivelm_799_det_all.py --limit 799` | `det799/run_summary.json` + NPZ |
| 799 帧检测评估 | `python scripts/evaluate_drivelm_799_det_subset.py` | `det799_subset_eval/metrics_summary.json` |
| 799 帧 BEV 地图分割 | `python scripts/run_drivelm_799_seg_all.py --limit 799` | `seg799_independent_eval/metrics_summary.json` |

本机环境曾有旧版 MMCV / MMDetection3D 与 PyTorch 2.8 的兼容补丁/警告，请不要把第三方原代码直接复制成无来源的“个人实现”。环境版本可另保存 `pip freeze` 供导师复现，但安装依赖需考虑 CUDA 与显卡架构匹配。

## 4. 明天建立 GitHub 仓库时建议的结构（今天不执行上传）

```text
bev_vla/
├── README.md                         # 明天整理项目首页，指向 docs/stage1/00
├── docs/
│   ├── stage1/                       # 今天完成的 6 份 Markdown
│   ├── logbook_2026-10-10.md
│   ├── logbook_2026-10-11.md
│   ├── logbook_2026-10-12.md
│   └── images/                       # 挑选过的必要可视化 PNG
├── scripts/                           # 自己写的实验/分析脚本及检查脚本
├── third_party/                        # 第三方官方文件，需保留版权来源和许可证
└── .gitignore                          # 明天核对后再修改，勿粗暴覆盖已有配置
```

> 原 `BEVFormer/`、`BEVFormer-seg/` 可能是外部 Git 仓库，也可能带大权重或数据目录；明天上传前要明确采用**fork/submodule/README 链接**哪种方式。不可未经检查直接 `git add .`。

## 5. 建议忽略/排除的大文件与敏感项目

- 环境和缓存：`.venv/`, `.venv-bevformer/`, `__pycache__/`, `.pytest_cache/`, `build_logs/`。
- 数据与模型：`data/`, `*.pth`, `*.pt`, `*.ckpt`, `*.zip`, `*.tgz`, `*.npy`, `*.npz`, `*.pkl`。
- 中间结果：`outputs/`, `downloads/`, `installers/`（里面可能有与项目无关的大型安装包）。
- 账号/绝对路径/私人信息：令牌、密码、学校身份认证信息、硬编码的本机用户目录不要公开。

有时 `*.json` 是小型配置，也可能是上百 MB 的完整数据；不能全加或全忽略，按用途逐一审核。今天不会改变 `.gitignore`，明天建立仓库前再操作。

## 6. 师傅应该能快速审阅的四个入口

1. `docs/stage1/00_阶段一交付导航.md`：8 项任务与 4 项产出对应关系。
2. `docs/stage1/03_基线性能测试报告.md`：mAP/NDS/mIoU、评估协议与明确局限。
3. `docs/stage1/01_DriveLM数据集结构与标注说明.md`：官方 QA 预处理证据与数据路径。
4. `docs/stage1/04_可视化成果索引.md`：BEV 特征和感知结果一键查看。

## 7. 最终验收原则

**绝不把 799 帧独立帧子集指标标注成完整官方时序成绩；不把分割 5.12% 的问题宣称已修复；不把尚未训练的网络写成已经训练。** 当前第一阶段的完成标准是流程可运行、数据一致性可证明、实验报告清楚、资源边界透明。

# Stage 2 - Language-guided BEV Attention

## 作用

这一目录提供语言条件 BEV 注意力、残差融合、共享 BEV 双 Head 集成与多任务损失实验。Step70 记录真实双 Head 前向，Step72 记录真实联合梯度（无参数更新），Step73 记录受约束语言与地图锚点功能。

完整状态见 [实证汇总](../../docs/stage2/16_阶段二实证汇总_2026-10-21.md) 和 [复现指引](../../docs/stage2/17_第二阶段复现指引.md)。

## 离线回归入口

```bash
cd ~/projects/bev_vla
.venv-bevformer/bin/python scripts/stage2/run_regression.py
```

使用已有兼容依赖环境；其他机器可替换解释器路径。该入口依次执行 Day 1 的 5 项 pytest、Step56 synthetic 和 Step68～73 的 self-test，禁用 GPU 和在线模型访问。报告写入 `outputs/stage2_regression/regression.json`，退出码非零表示存在失败。

Day 1 的独立模块位于根目录 `stage2/`，由 `run_day1.py` 和 `tests/test_stage2_day1.py` 使用；后续集成模块位于本目录。两组模块的入口和接口不同，复现时按相应脚本导入。

## 文件说明

- `language_encoder.py`：DistilBERT 文本编码与 attention-mask 池化
- `semantic_projection.py`：将文本隐藏向量映射到 256 维 BEV 通道
- `language_bev_attention.py`：生成 `(B,1,H,W)` 空间注意力
- `bev_fusion.py`：小系数残差融合，支持原始 BEV 精确回退
- `run_step56_smoke.py`：形状、有限值、前向、梯度测试
- `multitask_training.py`：检测、地图分割、注意力损失契约
- `run_step70_joint_integration.py`：共享 BEV 的双 Head 前向和旁路
- `run_step72_real_joint_gradient.py`：真实单帧三任务梯度，不更新参数
- `semantic_instruction.py`、`run_step73_fill_gaps.py`：受约束指令和独立地图锚点诊断
- `run_regression.py`：统一离线 CPU 回归

## 执行

在项目根目录 `~/projects/bev_vla` 运行：

```bash
python3 -m scripts.stage2.run_step56_smoke --text-mode synthetic
```

若需要真实预训练语言编码器，先在适合的虚拟环境中安装 `transformers` 并准备 DistilBERT 模型：

```bash
python3 -m scripts.stage2.run_step56_smoke --text-mode distilbert
```

会自动寻找 `outputs/bev_features/bev_feature.npy`。注意该文件此前可能来自 BEVFormer 检测模型，仅作为真实特征接口验证，不等于分割模型端到端集成；如不想读取可加 `--feature-source synthetic`。

## 指标边界

- 合成模式没有加载任何预训练语言模型；不能据此声称具备自然语言理解能力。
- 本阶段冒烟测试损失不是分割/检测监督信号。
- 未经训练的语言融合模块不能作为分割精度提升结论。
- 后续接入分割头需要保证 BEV 特征方向、时序状态、GT 和评估协议一致。

## 历史工具与报告

`scripts/step56_stage2_setup.py` 等旧安装器用于当时生成脚本，可能覆盖现有版本；现有仓库复现直接使用本目录代码。Step68～71 的 `--audit` 或 `--live` 可能重写对应历史 Markdown/JSON，日常自检使用 `--self-test` 或统一回归入口。

`scripts/step74_stage2_git_publisher.py --prepare` 会重新生成部分文档，`--publish` 会接着提交并推送；本轮审查未执行这两种模式。提交范围见 [审查记录](../../docs/stage2/18_提交前审查与回归_2026-10-21.md)，commit 和 push 前由用户确认。

# 第二阶段辅助脚本说明

实验入口、离线回归和指标边界见 [阶段二脚本 README](stage2/README.md)；完整环境与真实数据命令见 [复现指引](../docs/stage2/17_第二阶段复现指引.md)。仓库已包含脚本，无需从 Windows 下载 ZIP 再覆盖安装。

## 当前回归

在项目根目录运行：

```bash
.venv-bevformer/bin/python scripts/stage2/run_regression.py
```

本机该环境已有 pytest、PyTorch、Transformers 和 NumPy。输出位于被 Git 忽略的 `outputs/stage2_regression/`，本轮小型结果见 [JSON 归档](../results/stage2_2026-10-21/regression_2026-10-21.json)。

## 辅助工具范围

| 工具 | 作用与副作用 |
|---|---|
| `step55_protocol_audit.py` | 读取配置与数据协议，写审计报告 |
| `step56`～`step65` 的 `*_stage2_setup.py` | 历史脚本生成器；部分会覆盖当前脚本，日常复现直接使用 `scripts/stage2/` |
| `step59_memory_fix.py` | 历史源码补丁，不能当成训练或验证入口 |
| `step63_inspect_npz.py` | 检查本地特征缓存，写结构报告 |
| `step64_recover_report.py` | 从已有实验权重恢复结果，并可能修补训练脚本；不是 CPU 回归 |
| `step70_fix_inference_gt.py` | 过滤误传入推理接口的地图 GT 参数；现有 Step70 已修复 |
| `step72_fix_numpy.py`、`step72_run_all.sh` | 旧 NumPy 兼容补丁与真实梯度一键入口；现有 Step72 已含兼容层 |
| `step74_stage2_git_publisher.py --prepare` | 从历史 JSON 重新生成文档、SVG 和归档，可能覆盖手工整理内容 |
| `step74_stage2_git_publisher.py --publish` | 先 prepare，再自动暂存、commit 和 push；其白名单不涵盖本轮所有新增模块与测试 |

本轮未执行旧安装器、修补器或发布工具。commit 和 push 前需要用户确认最终清单；确认后按 [审查记录](../docs/stage2/18_提交前审查与回归_2026-10-21.md) 的候选范围操作，不使用 Step74 自动发布。

## 文件排除

数据软链接、数据集、`models/`、checkpoint、`outputs/`、虚拟环境、缓存、第三方源码及 `.bak` 留在本地，通过 `.gitignore` 排除。第一阶段工具与旧交付包 `.txt` 不混入第二阶段提交。

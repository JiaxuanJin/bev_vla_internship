# 正式测试目录

正式 pytest 单元测试、集成契约和回归测试统一放在已有 `tests/` 中。
目前全部属于第二阶段；尚无适合迁移的第一阶段 pytest 文件，因此不创建空的
`tests/stage1/` 或另一个 `test/` 目录。

```text
tests/
├── README.md
└── stage2/
    ├── test_stage2_day1.py
    ├── test_stage2_pipeline.py
    ├── test_stage2_joint_backward.py
    ├── test_stage2_micro_training.py
    ├── test_stage2_final_audit.py
    └── test_stage2_spatial_supervision.py
```

| 文件（位于 `stage2/`） | 用途 |
|---|---|
| `test_stage2_day1.py` | 早期根目录 `stage2/` 语言编码器与投影兼容性：形状、padding、冻结主干、梯度与非法输入 |
| `test_stage2_pipeline.py` | 后续 `scripts/stage2/` 指令解析、离线小型 DistilBERT、BEV 坐标、融合、共享双 Head、Hook 恢复与损失契约 |
| `test_stage2_joint_backward.py` | 联合反传的有限非零梯度检查及原始 GT 提取/错误拒绝 |
| `test_stage2_micro_training.py` | 混淆矩阵、诊断匹配、参数更新、流式 JSON、低内存保护与训练恢复协议 |
| `test_stage2_final_audit.py` | TV 正则、旧损失逐位兼容、非法值、原生网格监督、指令类别绑定、Hook 和冻结源码重放 |
| `test_stage2_spatial_supervision.py` | 空间实例标签、同类干扰物梯度、相关/无关区域指标、外部检测匹配及 opt-in 损失兼容 |

在项目根目录使用已有环境执行，无需下载权重或数据：

```bash
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv-bevformer/bin/python -m pytest -q -p no:cacheprovider
# 已有 68 项专项回归（62 项 pipeline + 6 项 spatial）
.venv-bevformer/bin/python -m pytest -q -p no:cacheprovider \
  tests/stage2/test_stage2_spatial_supervision.py \
  tests/stage2/test_stage2_pipeline.py
# 早期 stage2/ 兼容模块单独验证
.venv-bevformer/bin/python -m pytest -q -p no:cacheprovider \
  tests/stage2/test_stage2_day1.py
# 完整 pytest + Step56/68～73 七组历史自检，输出路径每次另取
.venv-bevformer/bin/python scripts/stage2/run_regression.py \
  --output outputs/new_test_layout_regression/regression.json
```

`pytest.ini` 默认仅收集 `tests/`，`pythonpath = .` 相对配置文件定位仓库根目录；
`--import-mode=importlib` 避免 `tests/stage2/` 干扰根目录兼容包 `stage2/`。
也支持显式指定 `tests/`、`tests/stage2/` 或单个新文件路径。
迁移不保留旧路径的副本或转发测试，避免重复收集。

根目录 `test_mmcv_cuda.py`、`scripts/test_*.py`、
`scripts/bevformer_compat/test_*.py` 和 Step26/37/45 等是需要人工选择环境、
数据或 GPU 的实验入口；其中部分在 import 时就运行模型或读取实验产物，继续留在原位。
Step56/68～73 的 `--self-test` 与实验 CLI 共用模块，保留其原入口。
这些手动入口的完整分类、迁移清单及整理前后结果见
[当天规范化报告](../docs/stage2/24_测试目录规范化_2026-10-26.md)。

历史 `results/` 中记录的旧命令、路径、源码 SHA 和结果按原文保留；复现时使用本页新命令。
CPU 合成契约测试通过不表示真实空间监督训练或指令相关精度收益已经验证。

# Step69：检测 + 分割 + 注意力多任务 Loss 模块与单元测试

> 历史快照：本文件保留 Step68/69 当时的验收状态。后续 Step70 已通过真实双 Head 前向，Step72 已通过真实单帧三任务联合梯度（无参数更新），Step73 完成受约束语言/地图锚点诊断。当前结论见 [阶段二实证汇总](stage2/16_阶段二实证汇总_2026-10-21.md) 和 [本轮测试报告](stage2/12_单元测试报告.md)。

## 模块功能

- 兼容 MMDetection 检测头 `loss_*`、`d0.loss_*` 和 decoder loss 列表。
- 计算地图分割四类 CrossEntropy 与按类别指令构造的注意力 BCE / Dice。
- 对检测、分割、注意力三条分支分别进行反向传播检查。
- 注意力坐标尺寸不一致时拒绝执行，避免直接插值掩盖坐标问题。
- 未提供检测头真实 loss 时拒绝伪造联合损失。

## 单元测试结果（CPU 合成数据，非实机检测头）

| 项目 | 结果 |
|---|---|
| mmdet_loss_dict_and_metric_filter | PASS |
| three_branch_differentiability | PASS |
| coordinate_size_mismatch_rejected | PASS |
| missing_or_nonfinite_detection_loss_rejected | PASS |
| empty_target_does_not_invent_supervision | PASS |
| no_fake_joint_training | PASS |

## 阶段二系统集成边界

- 分割链路真实运行证据：Step58、Step59、Step68。
- 现阶段检测与分割分别使用作者不同实验配置和权重。
- 单元测试通过**仅说明统一损失接口功能正确**，不代表检测头已经同帧接入。
- 真实共享 BEV 的检测头 + 分割头 + 注意力联合反传仍标记 NOT_VERIFIED。
- 物理坐标与复杂自然语言语义验证未完成，不影响本接口的单元测试结论。

## 代码与结果位置

- `scripts/stage2/multitask_training.py`
- `scripts/stage2/run_step69_loss_contract.py`
- `outputs/stage2_step69/step69_loss_contract.json`
- `docs/stage2_step69_multitask_loss_CN.md`

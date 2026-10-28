# LG-BEV V3 执行时间协议（2026-10-28）

V3代码已实施，真实GPU控制验收全部通过，独立准入为 **READY，等待单独授权**。状态以
`results/lg_bev_v3_20261028/admission.json` 为准；文件不存在或门槛未全过时仍为BLOCKED。
READY不授权训练，本轮不执行正式start/resume。原24小时合同、V2时间协议、旧准入与V3草案均保留原字节。

## 不变的训练内容

Train/Development/Cold Final分别为19123/2692/3288帧；A2+B4 frame epochs，
A9562、B19124、合计28686次优化器更新。Adapter为93866参数，作者主干、两个Head、
DistilBERT及BN状态冻结。FP32、Loss、AdamW、Scheduler、指令/样本顺序和划分不变。
正式GT已完成并通过正式来源验收，本轮复用报告与来源封存，不重建全量标签。

## V3资源语义

执行文件：`results/lg_bev_v3_20261028/time_protocol_v3.json`；独立正式run-id：
`lg_bev_formal_v3_20261028_v1`。父V1合同和V2协议由SHA绑定，不将旧V2 run改名。

- 累计硬上限为JSON `null`；48/72小时及之后每24小时只记录预警，不拒绝resume。
- 每会话27000秒。计时从本地启动请求开始，覆盖独立进程启动、模型/数据初始化、训练、日志、
  checkpoint/receipt持久化及GPU进程退出；A/B使用同一个时钟，恢复累计不清零。
- 提前600秒进入保存窗口。依据是既有跨场景B32上界106.78秒及停止/保存183.338秒，
  采用至少600秒、同时不小于实测最慢微步与保存退出上界之和2倍的规则。
  最终准入还检查本次真实验收的实测上界；不足则拒绝READY，不靠扩大时长放行。
- 到窗口后不再取新训练帧，在合法边界保存部分累积梯度；不强杀、不自动开始下一会话或关机。

`SessionClock`和`Trainer`保留历史24/48小时分支，新正式CLI默认V3。
历史V2需显式 `--protocol v2`，不能用其旧报告宣称V3已验收。

## 安全保存与恢复

checkpoint包括Adapter、梯度、AdamW、Scheduler、epoch/cursor/update、样本顺序和完整RNG，
主文件与时间receipt均原子写入并校验SHA。最新提交损坏时拒绝，不静默回退或重置状态。

V3另外有独立监督进程：等GPU训练进程真实退出，核对checkpoint后封存不可变
`sessions/<session_id>.exit.json`及SHA侧文件。退出receipt中的累计时间覆盖模型加载和保存控制开销，
下次resume使用它和checkpoint receipt的较大合法累计值。status需要GPU进程和监督进程均退出，
且退出receipt/执行协议/checkpoint全部匹配，才显示 `safe_to_shutdown=true`。
保存失败、超时或不干净退出不会被记为安全暂停。

软件按边界和预留窗口主动停止；OS/CUDA/I/O若不可响应，仍可能超过时限，此时必须记为安全验收失败，
不能以强杀、隐藏超时或伪造receipt承诺硬实时。实际7.5小时正式训练本轮未执行。

## 验收与正式启动边界

既定三个train场景32帧，独立验收run，与正式进度隔离。验证自动暂停、人工stop、跨进程恢复、
每步Loss/梯度、Adapter、AdamW、Scheduler、RNG、样本顺序、冻结Head/BN及真实退出时间。
测试专用累计49小时起点明确标为模拟，仅用于证明旧48小时限制不再误阻止恢复，不计正式资源用量。
缩短的真实会话也包含初始化和保存，不在模型加载后重置时钟。

开发集及Cold Final保持封存；A128 grounding门槛仍在未来正式A终点执行，不能提前伪报通过。
全部硬门槛通过才生成V3独立READY报告，之后仍需用户新的明确授权。

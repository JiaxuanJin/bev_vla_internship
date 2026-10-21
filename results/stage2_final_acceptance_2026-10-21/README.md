# 第二阶段最终严格验收结果（2026-10-21）

对应 [原始任务书](../../docs/stage2/02_第二阶段原始任务书.md)；完整逐项判定见 [最终验收报告](../../docs/stage2/20_第二阶段最终严格验收_2026-10-21.md)。本目录是本轮原始输出的小型副本，旧 Step72/工程补齐目录不改写。

| 文件 | 内容 |
|---|---|
| [cpu.json](cpu.json) | 统一入口 CPU 执行命令、环境、源码哈希及 PASS |
| [cpu_regression.json](cpu_regression.json) | 全部 69 项 pytest 与 7 组旧自检：命令、日志、退出码 |
| [offline_bev.json](offline_bev.json) | 真实本地 DistilBERT 和 BEV 缓存：二维特征→掩码→融合 |
| [joint_forward.json](joint_forward.json) | 新入口真实 GPU 前向、三路 bypass 对照、checkpoint 前后哈希 |
| [joint_backward.json](joint_backward.json) | 新入口 Mini Train 单次前向的三项真实 Loss、各参数梯度及联合 backward |
| [gpu_commands.json](gpu_commands.json) | GPU 两条实际命令、起止 UTC 时间、退出码及日志位置 |
| [joint_forward_output.txt](joint_forward_output.txt) | 本轮真实前向原始终端输出 |
| [joint_backward_output.txt](joint_backward_output.txt) | 本轮真实反传原始终端输出 |
| [dataset_audit.json](dataset_audit.json) | v1.0-mini、323 帧训练 infos、官方 mini_train 场景归属与注释文件哈希 |
| [acceptance.json](acceptance.json) | 原始任务、四项产出的判定与源码/文档/文件检查 |
| [proposed_commit_files.json](proposed_commit_files.json) | 等待确认的精确拟提交文件清单 |

两个 GPU 测试均在可访问 RTX 5080 Laptop GPU 的作者环境执行，不创建优化器，不写 checkpoint；前次沙箱 CUDA 失败仍在 `results/stage2_completion_2026-10-21/` 保留。语义定位、mIoU/检测精度收益、完整物理配准均属额外研究事项，未验证，不作为本任务书的附加强制门槛。

最终证据来自 `outputs/stage2_final_acceptance_seeded/`（取样前统一 Python、NumPy、Torch 种子）；补种前通过的初测原始 JSON/日志保留在 `outputs/stage2_final_acceptance/`，不混作最终源码证据。

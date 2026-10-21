# 第二阶段工程补齐：独立结果归档

本目录保存统一入口本轮实际执行的结果；原 Step55～73 结果和前次提交审查不变。

| 文件 | 实测内容 |
|---|---|
| [cpu_regression.json](cpu_regression.json) | 65 项 pytest 和 7 组既有自检通过，含源码 SHA256、解释器版本和退出码 |
| [offline_bev.json](offline_bev.json) | 真实本地预训练 DistilBERT + 已保存真实 BEV；二维语义/Attention/旁路功能通过 |
| [joint_sandbox_failure.json](joint_sandbox_failure.json) | 沙箱内 CUDA 不可用的真实失败与 traceback，保留 FAIL |
| [joint_live.json](joint_live.json) | 沙箱外作者环境真实双 Head 前向，旁路差为零；未更新参数 |
| [verification.json](verification.json) | 运行依据、原始结果映射、权重/坐标来源与代码/文档核对 |

随机融合模块的输出变化不构成定位或精度成功。当前报告与命令见 [工程缺口补齐记录](../../docs/stage2/19_第二阶段工程缺口补齐_2026-10-21.md)。

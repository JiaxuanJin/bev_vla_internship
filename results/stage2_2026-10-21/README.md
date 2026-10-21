# 第二阶段精选实验结果

Step55～73 的 17 份 JSON 为本机原始 `outputs/` 的小型副本。历史快照保留当时状态；Step71 中的未验收项应结合后续 Step72/73 阅读，不覆盖历史 JSON 来修改结论。

本轮新增回归和提交审查 JSON；不包含 checkpoint、数据集或特征缓存。

| 文件 | 内容 |
|---|---|
| [step55_step55_protocol_snapshot.json](step55_step55_protocol_snapshot.json) | 协议与环境快照 |
| [step58_step58_author_live_smoke.json](step58_step58_author_live_smoke.json) | 真实 DistilBERT 与分割链路冒烟 |
| [step59_step59_training_summary.json](step59_step59_training_summary.json) | 冻结主干的小样本训练 |
| [step60_step60_ab_summary.json](step60_step60_ab_summary.json) | 扩展同条件 A/B |
| [step61_step61_supervised_summary.json](step61_step61_supervised_summary.json) | 类别注意力监督训练 |
| [step62_step62_diagnostics.json](step62_step62_diagnostics.json) | 三类注意力、AP 与阈值诊断 |
| [step64_step64_recovered_summary.json](step64_step64_recovered_summary.json) | 平衡训练结果恢复 |
| [step65_step65_ab_summary.json](step65_step65_ab_summary.json) | 轻量 Adapter A/B |
| [step66_step66_ab_summary.json](step66_step66_ab_summary.json) | 背景保护 A/B |
| [step67_step67_ab_summary.json](step67_step67_ab_summary.json) | 训练集校准与锁定验证，最终回退 Original |
| [step68_step68_acceptance.json](step68_step68_acceptance.json) | 历史验收与文本干预对照 |
| [step69_step69_loss_contract.json](step69_step69_loss_contract.json) | 历史损失契约与合成测试 |
| [step70_audit_step70_audit.json](step70_audit_step70_audit.json) | 共享 BEV 集成静态审计 |
| [step70_live_step70_joint_live.json](step70_live_step70_joint_live.json) | 真实双 Head 单帧前向 |
| [step71_step71_acceptance.json](step71_step71_acceptance.json) | 截至 Step70 的历史验收快照 |
| [step72_step72_real_joint_gradient.json](step72_step72_real_joint_gradient.json) | 真实单帧三任务梯度，无参数更新 |
| [step73_step73_gap_check.json](step73_step73_gap_check.json) | 受约束语言与独立地图锚点诊断 |
| [regression_2026-10-21.json](regression_2026-10-21.json) | 本轮离线 CPU 回归：8 组全部通过 |
| [review_2026-10-21.json](review_2026-10-21.json) | 本轮候选提交清单及静态核对结果 |

当前状态见 [实证汇总](../../docs/stage2/16_阶段二实证汇总_2026-10-21.md)，图片来源见 [图片索引](../../docs/images/stage2/README.md)，提交范围见 [审查记录](../../docs/stage2/18_提交前审查与回归_2026-10-21.md)。

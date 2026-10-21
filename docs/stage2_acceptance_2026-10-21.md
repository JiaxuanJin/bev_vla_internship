# 第二阶段核心模块验收记录（Step68）

> 历史快照：本文件保留 Step68/69 当时的验收状态。后续 Step70 已通过真实双 Head 前向，Step72 已通过真实单帧三任务联合梯度（无参数更新），Step73 完成受约束语言/地图锚点诊断。当前结论见 [阶段二实证汇总](stage2/16_阶段二实证汇总_2026-10-21.md) 和 [本轮测试报告](stage2/12_单元测试报告.md)。

## 验收范围

- 项目：BEVFormer + DistilBERT + BEV 注意力与分割 Adapter。
- 复用 Step58～67 原始日志及冻结权重；不修改 GT、checkpoint 或训练结果。
- 时序 OFF 的单帧对照与 81 帧时序评估不得直接混用。

## 验收清单

| 检查项 | 状态 | 证据 / 缺口 |
|---|---|---|
| 真实 DistilBERT + BEVFormer 分割链路 | PASS | Step58 冻结模型同帧真实前向、关闭模块回退 |
| 真实 GT 训练与留出验证 | PASS | Step59 有真实 Train/Val、分割 Loss 与权重 |
| 语言分割精度可靠提升 | NOT_VERIFIED | Step67 校准最终选 original；既有 Adapter 未稳定改善前景与四类 IoU |
| 文本语义有效性 | TEXT_DEPENDENCE_TESTED | 真实作者 BEV + Step67 Raw Adapter，正常/置乱/常量同帧干预；详见 AP 和 mIoU |
| BEV / GT 物理坐标一致性 | NOT_VERIFIED | 相同尺寸与 Decoder replay 不等于独立物理坐标验证；仍需地图锚点与实际投影对照 |
| 检测 + 分割 + 注意力真实联合训练 | NOT_INTEGRATED | 已实现损失聚合接口并有合成梯度测试，当前真实训练是地图分割链路 |

## 真实文本干预（训练后 Raw Adapter）

- 真实验证帧：6；原始评估不修改 GT 或权重。
- 正常 vs 置乱文本 logits MAE：0.029277892。
- 置乱文字导致 argmax 像素变化比例：0.516458%。

| 前景 mIoU | Original | Normal | Shuffled | Constant |
|---|---:|---:|---:|---:|
| 真实同帧结果 | 3.7786% | 5.1743% | 5.1816% | 5.1709% |

| Attention AP | GT 占比 | 正常文本 AP | 置乱 AP |
|---|---:|---:|---:|
| Divider | 6.4092% | 8.5015% | 9.1307% |
| Ped Crossing | 1.5998% | 2.1410% | 4.2342% |
| Boundary | 7.6510% | 7.0041% | 6.4520% |

## 仍需独立完成的验证

- **BEV / GT 物理坐标：** 必须在真实地图锚点/ego2global/sensor2lidar 上完成双路径投影一致性；Decoder replay 不能代替。
- **检测多任务训练：** 当前提供可训练损失聚合接口，但未证明 BEVFormer 检测头、分割头和注意力可在同一批次联合反传。
- **开放式语言：** 三类固定文本类别模板的测试不能证明复杂自然驾驶指令可正确落到目标区域。
- **稳定精度提升：** Step67 锁定 original 是既定的安全回退，不得汇报成 Adapter 取得总体指标提升。

## 主要输入及输出

- Step58 报告：`/home/jjx2003520/projects/bev_vla/outputs/stage2_step58/step58_author_live_smoke.json`
- Step59 报告：`/home/jjx2003520/projects/bev_vla/outputs/stage2_step59/step59_training_summary.json`
- Step67 报告：`/home/jjx2003520/projects/bev_vla/outputs/stage2_step67/step67_ab_summary.json`
- 本次 JSON：`/home/jjx2003520/projects/bev_vla/outputs/stage2_step68/step68_acceptance.json`

## 结论

完成了可追溯的分割路径验收和测试证据整理；未满足的项目保留真实状态。

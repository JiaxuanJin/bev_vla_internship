# 第二阶段：模块与验收测试记录

## 本轮独立单元测试

| 测试 | 期待行为 | 结果 |
|---|---|---|
| 多任务损失接口 | 检测/分割/注意力梯度均可回传 | 合成数据 PASS |
| AP | 完全正确排序得到 1 | 合成数据 PASS |
| 四类 IoU | 完全预测匹配 GT 得到 1 | 合成数据 PASS |

## 真实模型/数据验证

| 项目 | 状态 |
|---|---|
| 真实 DistilBERT + BEVFormer 分割链路 | PASS |
| 真实 GT 训练与留出验证 | PASS |
| 语言分割精度可靠提升 | NOT_VERIFIED |
| 文本语义有效性 | TEXT_DEPENDENCE_TESTED |
| BEV / GT 物理坐标一致性 | NOT_VERIFIED |
| 检测 + 分割 + 注意力真实联合训练 | NOT_INTEGRATED |

## 遗留项目（未通过或未核验）

- Real 3-task detection + segmentation + attention joint backward：未验收。
- nuScenes 地图 GT 与 LiDAR/ego/world 的独立物理锚点一致性：未验收。
- 自由文本正确指代多个对象的语义定位：未验收。
- 在完整 81 帧时序或官方 val 上稳定改善四类 mIoU：未验收。

## 固定条件

- Stage2 Step68 同帧文字干预：真实 Mini Val，时序 OFF，冻结作者模型和 Decoder。
- 只保存报告与测试数据，不编辑作者代码、权重或数据。

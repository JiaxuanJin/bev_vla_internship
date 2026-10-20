# Step 38：评估协议与 Val 元数据审计

## 一、数据集身份
- PKL 样本数：6019，场景数：150
- Sample Token 重复数：0
- 时间戳是否非降序：False
- Step 34 与当前 PKL 哈希一致：True

## 二、作者指标公式重算
作者定义的前景 mIoU 为 divider、ped_crossing、boundary 三类 IoU 的平均，
每类 IoU 按全局 TP/(TP+FP+FN) 计算，背景不参与平均。
- original: mIoU=4.641619%，重算一致=True，最大类别差=0
- physical_ccw_candidate: mIoU=19.600046%，重算一致=True，最大类别差=0

## 三、与官方 nuScenes 元数据抽样对照
- 已核查 12 帧：通过 12 帧，未通过 0 帧。

## 四、结论边界
- 已知 Step 25 与作者 IoU 核心定义一致，本文再次基于 Step 34 混淆矩阵校验计算结果。
- 本检查不执行模型前向，不核实训练时的 forward 源码，也不证明预测/GT 几何完全一致。
- 如果官方元数据抽查通过，应优先调查 GT 栅格方向、BEV 特征取样及训练源码对应关系。
- 原始 Step 34 JSON 的 physical_ccw_candidate 为历史命名；实际高分候选是 Transpose。

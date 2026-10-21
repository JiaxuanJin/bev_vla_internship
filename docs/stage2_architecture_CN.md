# 第二阶段：语言引导 BEV 注意力模块设计说明

> 历史快照：本文件保留 Step68/69 当时的验收状态。后续 Step70 已通过真实双 Head 前向，Step72 已通过真实单帧三任务联合梯度（无参数更新），Step73 完成受约束语言/地图锚点诊断。当前结论见 [阶段二实证汇总](stage2/16_阶段二实证汇总_2026-10-21.md) 和 [本轮测试报告](stage2/12_单元测试报告.md)。

## 系统结构

```text
驾驶指令 → 本地冻结 DistilBERT → (3, 768) 类别文本向量
                                      │
六相机图像 → 冻结 BEVFormer → BEV (1, 256, 150, 150)
                                      │
              ┌───────────────────────┤
              ↓                       ↓
        原始分割 Decoder     Semantic projection / attention
              ↓                       ↓
     logits (1,4,200,400)    Attention (1,3,200,400)
              └───────────┬───────────┘
                          ↓
                   Logit Adapter（可训练）
                          ↓
                   最终四类语义地图
```

## 模块接口及限制

- 语言编码：`language_encoder.py`，预训练 DistilBERT 768 维文本特征。现有训练使用冻结、预先缓存的模板文本特征。
- 投影：`semantic_projection.py`，将 768 维输入映射到 BEV 通道空间。
- BEV 空间注意力：`language_bev_attention.py`，在 BEV 特征网格计算语言条件注意力。
- BEV 残差融合：`bev_fusion.py`，允许关闭模块后严格回退原始特征。
- 原始分割头集成：Step58 在 `transformer.forward` 输出后插入融合模块；Step65/67 使用冻结 BEV 特征上的四类 logit Adapter。
- 类别空间：Background / Divider / Ped Crossing / Boundary；GT 为 `(1,200,400)`。
- 评价：累积混淆矩阵与四类 argmax，独立报告前景三类 mIoU 和四类 mIoU。
- 检测联合损失：`multitask_loss.py` 仅提供聚合接口；当前并未验收检测 + 分割的真实联合推理与反传。

## 训练方案

- 真实 nuScenes v1.0-mini Mini Train 地图 GT；冻结 BEVFormer、分割 Decoder 和 DistilBERT，训练语言空间注意力与校正适配器。
- 有监督分割 Loss：CE / Focal CE / Soft-IoU / Tversky 实验变体；有监督 Attention BCE/Dice。
- Step65～67 建立 Bias-only、Text-shuffled、Original 等对照。
- Step67 Mini Train 校准无符合全部条件的模型，最后回退 Original。不得将 Raw Adapter 的局部收益视作已通过总体验收。

## 与阶段任务的边界

1. 真实 BEVFormer 分割链路及语言模块实现已完成；开放式自由驾驶指令定位尚未验证。
2. BEV / GT 物理坐标仍需独立锚点核验；重放 Decoder 一致仅证明内部张量排列正确。
3. 检测、分割、注意力三任务真正共同训练尚未完成；当前交付的是可扩展损失接口及其单元测试。

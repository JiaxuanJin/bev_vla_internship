# 第二阶段实验图片索引

Markdown 使用相对路径引用本目录图片。四张 PNG 来自已有 Step62 实验，未重新生成或修饰；两张 SVG 是可编辑的架构/数值摘要图。

| 图片 | 来源 | 阅读边界 |
|---|---|---|
| [stage2_architecture.svg](stage2_architecture.svg) | 根据代码接口绘制的架构示意 | 共享 BEV、语言注意力和残差融合；不表示物理坐标已闭环 |
| [stage2_evidence.svg](stage2_evidence.svg) | Step71 的 Step70 数值摘要 + Step72 梯度 JSON | 前向变化和梯度连通性，不代表精度提高 |
| [step62_divider_overlay.png](step62_divider_overlay.png) | `outputs/stage2_step62/` 同名原图 | Divider 注意力与地图 GT 的诊断对照 |
| [step62_ped_crossing_overlay.png](step62_ped_crossing_overlay.png) | 同上 | Ped Crossing 注意力与地图 GT 的诊断对照 |
| [step62_boundary_overlay.png](step62_boundary_overlay.png) | 同上 | Boundary 注意力与地图 GT 的诊断对照 |
| [step62_threshold_scan.png](step62_threshold_scan.png) | 同上 | 注意力阈值敏感性，验证集上的扫描只作诊断 |

Step62 使用 Step61 留出的 Mini Val 6 帧，时序 OFF；地图 GT 只作评估/可视化对照。相关样本、阈值、AP、IoU 和限制见 [Step62 JSON](../../../results/stage2_2026-10-21/step62_step62_diagnostics.json)。不能仅凭叠加图宣称语言正确定位或 BEV/GT 物理配准完成。

2026-10-21 复核：四张 PNG 与本地原图逐字节一致，以下 SHA256 便于后续比对。

```text
f3b8fc56bcd0ae67849795e27ac7a6288d457562976c18f1692f2b6de64bf9e9  step62_divider_overlay.png
941c6df7971af5e8fa82a64765a754391aa3bb9dd655b9b3ca7d63890a12a832  step62_ped_crossing_overlay.png
ccb89abfa1b986060ce1ee0e7b199d8ed01b8f3b1e7047e71075f673862c43a8  step62_boundary_overlay.png
882b6f60185eeb3684d4d94da0237824b1566e0602317646a34f6b05a8f1801d  step62_threshold_scan.png
```

在 VS Code 用 `Ctrl+Shift+V` 预览 [实证汇总](../../stage2/16_阶段二实证汇总_2026-10-21.md)，可同时查看四张真实实验图及两张 SVG。

## 当前统一入口架构

[stage2_final_architecture.svg](stage2_final_architecture.svg) 对应最终源码：文本与视觉生成二维 S，再进入 Attention 和残差融合；训练 GT 只进入损失分支。旧两张 SVG 和四张 Step62 PNG 保留为历史证据，最终运行数值见 [严格验收报告](../../stage2/20_第二阶段最终严格验收_2026-10-21.md)。

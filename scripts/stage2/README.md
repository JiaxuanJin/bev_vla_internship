# 第二阶段统一模块与交付入口

当前任务依据：[指导老师原始任务书](../../docs/stage2/02_第二阶段原始任务书.md)。当前源码审查：[21_最终缺口审查](../../docs/stage2/21_第二阶段最终缺口审查_2026-10-22.md)；最终Git交付：[22_交付检查](../../docs/stage2/22_第二阶段Git交付检查_2026-10-23.md)；[117项测试与真实GPU证据](../../results/stage2_final_audit_2026-10-22/README.md)；[真实20步与6帧A/B图表](../../results/stage2_delivery_2026-10-23/README.md)。

在仓库根目录运行。完整统一入口为 `scripts/stage2/run_stage2.py`，支持cpu-tests、offline-bev、joint、joint-backward和独立进程micro-train阶段。15GiB WSL优先使用低内存smoke或冻结旧源码只读重放，不直接预加载作者全量元数据。

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  .venv-bevformer/bin/python -m pytest tests/ -q -p no:cacheprovider
.venv-bevformer/bin/python scripts/stage2/replay_stage2.py --verify-only
```

GPU smoke（不创建optimizer）、固定旧20步版本验证重放及外部依赖的完整命令见 [17_复现指引](../../docs/stage2/17_第二阶段复现指引.md)。模型和数据必须本地准备，所有结果使用新输出目录。

| 模块 | 功能 |
|---|---|
| language_encoder / semantic_instruction | 真实预训练DistilBERT、规范空白、masked mean；规则解析作校验/元数据，监督类别由当前文本派生 |
| semantic_projection / language_bev_attention / bev_fusion | 真实二维S→Attention A→F+alpha F A；保留旧融合权重键 |
| pipeline / bev_coordinates / joint_runtime | 共享BEV临时Hook、序列/地图契约、真实Detection与Segmentation分支 |
| multitask_training / micro_training | 真实检测+分割Task Loss、BCE+Dice Attention Supervision、可配置TV Regularization；协议配置恢复 |
| micro_process / low_memory_dataset / runtime_memory | 独立协议/训练/验证进程、原始元数据流式读取、RSS/WSL/CUDA日志与保护 |
| integration_smoke / replay_stage2 / audit_stage2_evidence | 当前真实只读smoke、冻结旧源码验证重放、原始数组/权重审计及真实语言响应探测 |

原20次更新为TV=0；TV=0.001只有新smoke的实现/梯度证据，没有新训练。六帧Original/Trained/Shuffled foreground mIoU为0.3159926671/0.3159515546/0.3159652746，检测均207/599/35；语义定位、稳定精度收益及物理配准均未验证。

根目录 `stage2/` 为Day1兼容模块；本目录是后续完整集成。Step56～73历史入口和 `README_使用说明.md` 保留原阶段上下文；部分--live/--audit会写历史产物，交付核对不要调用旧安装器或publisher。`micro_report.py` 仅适配旧平铺目录，当前图表使用 `scripts/prepare_stage2_delivery.py` 从已有JSON/NPZ离线绘制。不要运行 `scripts/step74_stage2_git_publisher.py` 自动提交；本轮未暂存、commit或push。

# 固定256更新终点后的三组Development评价方案

> **历史方案说明：** 本文保留 2026-11-01 登记的终点评价设计与旧入口。独立 256 更新训练后续已完成，旧版评价加载失败；当前 320 帧 Development 为 NOT_COMPLETED，v2 仍待单独授权。实际继续条件以[修复验收报告](../evaluation_fix_v2_20261102/修复验收报告.md)为准，本文不是当前执行授权。

本文件只规定下一步执行方案，不构成评价授权。本轮训练授权不包含Development或Cold Final GT。

## 开始条件

1. 新独立run `lg_bev_explore_main_20261101`完成256更新、1024微步、512唯一帧各覆盖两次。
2. 最终Checkpoint、Optimizer、Scheduler、RNG、游标、源码/配置身份及真实退出链通过核查；只使用固定终点，不按Loss择优。
3. 正式196文件、33源码与共享catalog保全通过，原A128仍为GROUNDING_NOT_READY。
4. 另行取得`evaluate`和固定320帧Development GT的明确授权，批准文件绑定当前配置与源码SHA；Cold Final保持禁止。

## 固定对照与协议

- Baseline：原生BEV绕过Adapter，无训练参数、Attention为N/A。
- Stage A Only：只读A9562 state_dict，全部冻结、up保持零。
- Stage B Enhanced：只读本次固定256更新终点，全部冻结。
- 另加载已核对来源的零更新候选，仅做A0/A9562 Attention对照；不称其为正式A0回执。

三组使用同一320帧、640条预登记query、相同原生时序history、标定、GT、valid、native top300、SDK过滤、匹配和阈值。
correct/wrong/zero输入统一按正确指令GT评分，禁止为了改善结果切换评分目标或阈值。

先完成GT覆盖审核并封存`gt_coverage.json`，之后才生成任何模型预测。缺正类如实保留NOT_EVALUABLE，不按模型表现换帧、删类、扩样本或选Checkpoint。

主指标：Relevant/Irrelevant Detection AP及Relevant/Irrelevant Segmentation IoU。两个区域独立累计，不能由Overall减Relevant推导。
检测采用正确指令Ego象限及补集，x=0/y=0归front/left，car/pedestrian、0.5/1/2/4m严格距离阈值；分割采用目标mask扩张2m的ROI及valid补集，完整语义混淆矩阵、lane/crosswalk按类汇总。

同时输出Map/Object Attention正目标macro/pooled IoU、正确/错误目标区logit与概率差、空目标激活/FP、背景/同类干扰FP；补报query级全10类AP、全前景IoU、实例conditional AP及每场景/日志结果。
驾驶日志配对bootstrap明确是日志宏平均差，不冒称pooled指标差的CI。

## 将来的执行命令（本轮不执行）

仅在收到新的明确授权并生成`authorization_evaluate.approved.json`后：

```bash
.venv-bevformer/bin/python -B \
  experiments/lg_bev_stage3_exploration_20261101/cli.py evaluate \
  --config experiments/lg_bev_stage3_exploration_20261101/configs/main.json \
  --authorization experiments/lg_bev_stage3_exploration_20261101/configs/authorization_evaluate.approved.json
```

入口自动核对独立train终点、真实退出证据和身份，不使用原正式`evaluate_terminal`，不改变原run的任何gate/status/control。
输出进入独立`lg_bev_explore_main_20261101_evaluation`。每会话27000秒、提前600秒停止；评价在帧边界保存累计器，继续时绑定训练终点及事前GT覆盖摘要。

保持新增G盘资产≤8GB、共享根与新增合计≤30GB、G盘空闲≥250GB；不删除旧资产。运行前后核对正式资产及catalog。

## 结果解释

四项区域指标是自定义独立子集结果，不是官方全val。仅Adapter训练/评价日志隔离；作者主干预训练可能见过这些nuScenes train帧，不能宣称整模型未见。
本次256更新训练完成本身不证明精度提高，也不改变A128失败结论。评价结果无论正负均完整保留，之后不依据Development结果反复调参。

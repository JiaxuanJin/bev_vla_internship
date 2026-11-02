# LG-BEV 独立探索 Stage3：交付导航（截至 2026-11-02）

本目录承接[独立探索审计方案](../lg_bev_stage3_feasibility_audit_20261101/审计与最小实验方案.md)，计划于 **2026-11-06** 结项。原正式 Stage A 与独立探索 Stage B 的状态分别保留。

| 项目 | 当前真实状态 |
|---|---|
| 原正式 Stage A A128 | **NOT_PASS**；原 run 保持 `FAILED / GROUNDING_NOT_READY / can_resume=false`，禁止恢复 |
| 独立 StageB256 | **COMPLETED，256/256 更新、1024 微步**；512 帧、2 epochs、seed=20261101，Checkpoint 和恢复链已核验 |
| 8 帧三组对照 | **PREFLIGHT_PASS**；Baseline 为原生模型旁路 Adapter，A0 只用于初始化 Attention 对照 |
| Development 320 帧 | **NOT_COMPLETED**；v1 在 GT 覆盖阶段失败，预测 0/320；v2 修复已通过 155 项回归，等待新哈希授权 |
| 本次交付整理 | 只更新导航、环境记录、非 Git 资产说明和 Git 白名单；无新训练、推理或计分 |

## 阅读入口

- [最终结项报告](../../docs/final_report/BEV_DriveLM_项目总结报告.md)：技术演进、架构、真实成绩和失败边界。
- [Stage A 核心验收、CSV 和可视化](reports/stage_a_acceptance_20261102/核心验收报告.md)：A128 正式失败与固定 8 帧局部选择性分列。
- [Stage B 固定 8 帧感知对照](reports/stage_a_acceptance_20261102/既有8帧感知诊断.csv)：三组相关／无关 AP@2m_xy 与分割 IoU。仅工程诊断，不属于 320 帧 Development，不是官方 nuScenes mAP。
- [评价加载根因与 v2 修复](reports/evaluation_fix_v2_20261102/修复验收报告.md)：SDK 速度依赖、临时文件清理、155 项回归和待授权范围。
- [独立环境与非 Git 资产复现](../../docs/stage3/05_Stage3环境与非Git资产复现_2026-11-02.md)、[Stage3 requirements](../../requirements_stage3_20261102.txt)：当前 `.venv-bevformer`，不混用历史作者环境。
- [本次 GitHub 白名单](../../docs/stage3/delivery_20261102/GitHub提交白名单_20261102.json)：仅供人工审查，不自动暂存／提交／推送。
- [历史训练命令](运行命令.md)、[已完成的 GPU 迁移预检](reports/gpu_preflight_20261101/GPU预检报告.md)、[现有今日 Logbook](../../docs/logbook_2026-11-02.md)。

以下保留原实验实现和协议说明；当前进度以上表和终稿为准。StageB256 Attention 低于同 8 帧 A9562，检测无增益、分割改善很小，均保留为负向证据，不调整门槛或重训。

## 文件与入口

| 文件 | 职责 |
|---|---|
| `cli.py` | CPU核查、分动作GPU授权、启动、停止、独立继续与退出状态核验 |
| `core.py` | SHA绑定、只读Adapter迁移、数据清单检查、独立优化器、受限写入 |
| `backend.py` | 原生运行时接入、私有SceneData命名空间、冻结和媒体来源验证 |
| `worker.py` | 封存数学微步接入、预算、梯度预检、更新边界保存及固定终点评价 |
| `lifecycle.py` | Worker保存→真实退出回执→Supervisor真实退出观察的完整链条 |
| `metrics.py` / `evaluation.py` | 三组共用的区域指标、Attention、指令控制和分组统计 |
| `synthetic.py` / `tests/` | CPU合成三组验收、边界与异常测试 |
| `configs/main.json` / `short.json` | 512帧/256更新与256帧/64更新，固定seed=20261101 |
| `运行命令.md` / `实施报告.md` | 后续命令、预期结果、验收证据与限制 |

从仓库根目录执行CPU核查：

```bash
CUDA_VISIBLE_DEVICES='' .venv-bevformer/bin/python -B \
  experiments/lg_bev_stage3_exploration_20261101/cli.py cpu-check
```

## 参数、来源与隔离

Baseline绕过Adapter；Stage A Only从A9562加载后冻结；Enhanced只从A9562复制
`state_dict`，另建AdamW与Cosine调度器。Enhanced更新Adapter全部93866参数，包含
6400个up参数。作者BEVFormer、检测/分割Head、BN及DistilBERT保持eval和冻结，
学生Head保留对输入BEV的梯度。Loss、clip=5、FP32、microbatch=1和累积4沿用封存数学实现。
没有调用原正式`start/resume/run/evaluate_terminal`；只复用Trainer构造与`microstep`。

A9562源Checkpoint文件SHA：
`b19dacd76ee5861a05d6f915e4d096250d36d7280bc3701dbf5e99e1430b038a`。
其Adapter张量SHA：
`5db7c4105bb9b138773ec249b7f47641ccddc16bb64d3a3b11ebb297ce7dfe81`。
文件、receipt、execution和逐张量身份均核对；历史AdamW、部分梯度、RNG及9562步预算
均不迁入新训练。只有本候选自己保存的完整更新边界允许独立继续。

A0来自独立`daily_acceptance_v3_20261028_v1`的零更新参数候选，CPU与封存结构和
初始化seed=20261027逐张量重建一致。它不是原正式run的A0回执。A0只加入Attention
前后对照。新实验全局RNG为20261101；封存`choose_query`中的20261027是固定指令
hash排序标签，沿用同一指令采样规则，不是加载历史RNG，也不宣称改变seed会改变该指令表。

所有代码、测试和本轮报告位于本目录，不命中正式source manifest的匹配路径。
后续运行输出唯一根为：

```text
/mnt/g/BEV_VLA_Datasets/lg_bev_stage3_exploration_20261101
```

CPU准备轮次没有创建该G盘输出目录；后续获授权的GPU预检已在该目录生成独立资产。
原Checkpoint、run、共享catalog均仅读取。
新运行缓存只在新根下，原生BEV在RAM逐帧保留，不落盘保存512+320份大型BEV。
可只读校验并复制既有train场景索引到私有缓存；原生媒体摘要与训练GT来源一致后才能使用。
私有模块不改变原`SceneData/CACHE_ROOT/RollingCache`的全局绑定。
Python文件及SQLite写入审计钩子拒绝根目录外写入，第三方缓存指向新根。
此钩子不构成对任意C扩展写入的操作系统级沙箱，GPU预检前后仍须做完整保全比对。

## 固定数据和评估

仅使用已预登记清单：train为16场景前32帧共512帧；短方案取前8场景共256帧；
development固定10场景共320帧；诊断清单128帧仅保留元数据，本轮不做真实诊断。
验证每场景从真实起点连续取帧，train/development/diagnostic的帧和驾驶日志互斥。
train和diagnostic排除A128涉及的全部场景。Cold Final没有入口。

Development GT只允许在独立B固定终点完成、同源码/配置身份且另行授权后读取。
评价先保存固定320帧、640条query的GT覆盖记录，再生成任何模型预测。
缺正类记`NOT_EVALUABLE`，不会静默扩样本或删类提高均值。

三组共用同一原生时序BEV、正确GT、有效域、指令表、native top300解码、SDK过滤、
匹配和指标。correct/wrong/zero三个输入控制均按正确指令GT计分。每帧固定一条
object-all与一条map指令，类别交替、hash选方向，保留空目标。

| 指标 | 固定规则 |
|---|---|
| Relevant Detection AP | car/pedestrian，正确指令Ego象限内的GT和预测中心 |
| Irrelevant Detection AP | 相同类别和有效检测范围，独立匹配象限补集 |
| 检测匹配 | 距离严格小于0.5/1/2/4m、score全局排序、一个GT最多匹配一次；SDK 101点AP |
| Relevant Segmentation IoU | 地图目标mask扩张2m，与valid相交，累计完整语义混淆矩阵 |
| Irrelevant Segmentation IoU | valid减上述ROI，独立累计混淆矩阵；lane/crosswalk两类宏平均 |
| Map/Object Attention IoU | 同标定投影logits，sigmoid≥0.5；正目标macro与pooled分开 |
| 文本与空目标 | 正确减错误logit/概率差、空目标激活、背景/同类干扰FP、峰值/目标质量占比 |

边界x=0/y=0归front/left。跨边界框按中心分区，近距离跨区也不能互相匹配。
无正GT的整类AP/IoU为N/A；空GT预测FP继续进入全局排序。
无关指标完全独立累计，没有使用Overall减Relevant。
Baseline的Attention为N/A。补报全10类query级AP、全前景IoU及实例conditional AP，
这些均是自定义子集指标，不是官方完整val。保存每场景、每驾驶日志结果与驾驶日志
宏平均配对bootstrap差；该CI不冒充pooled指标差的CI。

## 会话与预算

单会话上限27000秒，26400秒起请求停止；训练最多补齐当前4微步更新边界后保存，
评价在帧边界保存累计器。计时包括初始化、训练、保存、Worker及Supervisor退出。
预检第一会话固定4次更新暂停，第二会话以`--continue-run`继续到8次更新；
已完成的主训练从A9562开始，预检8次更新不计入主预算。

外层启动器持全局实验锁，Supervisor真实`wait`观察Worker退出码0后才生成退出回执，
启动器再真实等待Supervisor退出后封存closure。独立`status`还核对Checkpoint摘要、
配置/源码身份、最新会话、PID消失与时限。缺证据、非零退出、PID存活、损坏或超时
均不宣称安全，不回退旧Checkpoint，也不补造过去的回执。

`safe_to_shutdown=true`仅是这个新实验的保存/进程收尾结论；`PAUSED`可以继续本探索
实验，`COMPLETED`表示固定探索预算完成。两者都不表示A128通过或原正式run解锁。
意外断电/进程强杀后的残缺会话不提供自动恢复，需要另行审计。

每次自有资产提交检查新增≤8GB、共享根+新增≤30GB、G盘剩余≥250GB。
第三方临时文件和控制台日志不通过逐字节配额管理，后续预检须复核实际占用；不得自动
清理共享缓存。预估沿用审计：预检0.5–1小时、主训练1.2–3小时、独立评价1–2.5小时，
全闭环预留6–10小时。最新32帧预检两会话共736.854秒，32微步合计29.034秒；
这是预检阶段历史记录；主训练现已完成，320帧评价仍未完成。耗时口径须包含初始化、探测、I/O及保存。

## 授权与证据边界

`configs/authorization_*.template.json`均为`approved=false`，不能启动GPU。
本次32帧GPU迁移/暂停继续预检及新G盘根写入已获授权并完成；批准范围仅为两会话共8更新。
主训练已另行授权并完成256更新；Development v1失败现场保留，修复版v2仍需单独授权。Cold Final始终禁止。
命令和预期见[运行命令](运行命令.md)，CPU准备记录见[实施报告](实施报告.md)，
本次实际结果见[GPU预检报告](reports/gpu_preflight_20261101/GPU预检报告.md)。

最新预检已验证真实任务梯度、三个场景双Head逐位恒等、G盘持久化、显存/耗时及
update4恢复边界的零容差数值一致。512帧固定预算训练已完成，长期稳定性、320帧完整评价和泛化收益尚不能由8帧预检证明。作者主干预训练可能见过候选Development帧，
只能声称Adapter训练日志隔离；A128失败迁移和单seed小子集的限制必须保留。

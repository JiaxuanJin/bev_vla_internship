# LG-BEV 独立 Stage B：32帧 / 8更新 GPU预检报告

> **历史范围说明：** 本报告统计截至 2026-11-01 的 32 帧／8 更新迁移预检，正文“下一阶段”对应当时安排。后续独立 StageB256 已完成；当前 320 帧 Development 为 NOT_COMPLETED、v2 待授权，正式 A128 仍为 NOT_PASS。最新状态见[独立实验导航](../../README.md)。

日期：2026-11-01。结论：**PREFLIGHT_PASS，最小工程闭环通过**。这不是原正式A128通过，不代表检测/分割精度提升。

本轮严格使用独立入口，共两个GPU模型会话、32个唯一训练帧、32个训练微步、8次AdamW更新。没有执行256更新或64更新训练；没有读取Development/Cold Final GT，没有终点评价、Git commit/push。

## 实际会话与退出

| 会话 | 终态 | 更新 / cursor | Worker / Supervisor退出码 | 总墙钟 | 峰值CUDA allocated / reserved |
|---|---|---|---|---|---|
| 1 | PAUSED | 4 / 16 | 0 / 0 | 378.786s | 1.981 / 2.892 GB |
| 2 | PREFLIGHT_PASS | 8 / 32 | 0 / 0 | 358.069s | 2.006 / 2.529 GB |

计时包含启动后的初始化、训练、探测、Checkpoint持久化、Worker退出和Supervisor退出观察。合计 736.854s（约12分17秒）。两会话均完整封存worker_end、exit、closure，独立status在宿主PID空间核对后返回safe_to_shutdown=true。

第一会话ID：`1793544071841122956_a6b12db4`；第二会话ID：`1793544528965319603_f1fb8615`。

退出时间来自真实观察记录（北京时间）：

- 会话1：Worker 2026-11-01T22:47:09.863001+08:00；Supervisor 2026-11-01T22:47:19.150629+08:00。
- 会话2：Worker 2026-11-01T22:54:27.045012+08:00；Supervisor 2026-11-01T22:54:36.488966+08:00。

第一Checkpoint：`/mnt/g/BEV_VLA_Datasets/lg_bev_stage3_exploration_20261101/lg_bev_explore_main_20261101_preflight/checkpoints/1793544071841122956_a6b12db4_end.pt`；SHA256 `c65f7e86a0845afc60be344a1a4c25a1141457408dc271ae3d6be00d672bbc76`。

第二Checkpoint：`/mnt/g/BEV_VLA_Datasets/lg_bev_stage3_exploration_20261101/lg_bev_explore_main_20261101_preflight/checkpoints/1793544528965319603_f1fb8615_end.pt`；SHA256 `03ad03dfae59284a5bebc52ad6efa51d47ceeddcfbe0dfd959e3de281fe00dc9`。

每个终点包含Adapter、18组AdamW状态、scheduler、Python/NumPy/CPU/CUDA RNG、epoch/cursor/updates及来源/冻结摘要。第一会话cursor=16、scheduler.last_epoch=4；第二会话cursor=32、last_epoch=8，所有AdamW step均为对应更新数。第二launch明确绑定第一end Checkpoint路径及SHA，没有使用原正式resume或原AdamW。

## 八次真实更新与梯度

以下Loss和分支梯度为每次更新前**首个微步**的额外只读探测；参数变化为该组第4微步实际optimizer.step前后差。两类记录不混作同一帧Loss。梯度探测和数值见证均不调用optimizer.step。

| update | cursor终点 | Det Loss | Seg Loss | Att Loss | Det梯度L2 | Seg梯度L2 | Att梯度L2 | 实际参数变化L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 2.482447 | 0.657512 | 1.547684 | 0.00258761 | 0.000690524 | 21.896 | 0.074559 |
| 2 | 8 | 3.026724 | 0.740816 | 2.056251 | 0.00296482 | 0.0034493 | 29.1723 | 0.053823 |
| 3 | 12 | 2.652760 | 0.652825 | 1.842893 | 0.00370501 | 0.00439423 | 19.884 | 0.047615 |
| 4 | 16 | 3.415841 | 0.361722 | 1.270790 | 0.025584 | 0.00228377 | 2.7924 | 0.043594 |
| 5 | 20 | 3.023597 | 0.449561 | 2.007687 | 0.00189039 | 0.00359145 | 8.67721 | 0.043830 |
| 6 | 24 | 2.953585 | 0.286059 | 1.124266 | 0.00249709 | 0.00308142 | 5.55551 | 0.044018 |
| 7 | 28 | 2.882755 | 0.493541 | 1.344001 | 0.00388928 | 0.00118094 | 10.718 | 0.040470 |
| 8 | 32 | 2.946525 | 0.551013 | 1.352290 | 0.00484411 | 0.00375255 | 12.0762 | 0.039633 |

所有分支Loss与梯度有限，分支总梯度非零；每次更新18个参数张量均发生非零变化。全Adapter共93866参数，原作者主干、双Head、BN和DistilBERT不进入优化器。A9562源up的6400个参数全零；更新后6400个up参数均非零。

零up边界已实际验证：首次Det/Seg梯度仅到up，language/visual/key/mask_refine任务梯度均为0；Attention梯度可到language/visual/key/mask_refine，而Attention对up梯度为0。首次Det→up L2=0.00258761、Seg→up L2=0.000690524；update1后Det/Seg→language分别为0.000491758/0.000242408，后续探测仍有限非零。

原始逐微步Loss、累计梯度范数、逐张量变化、内存与耗时见[microsteps.json](microsteps.json)；八次汇总见[updates.csv](updates.csv)；全分支/前缀梯度见[result.json](result.json)。不同帧Loss不能解释为训练收敛或泛化提升。

## 原生双Head、冻结及数值连续性

在首次梯度探测/训练更新之前，检查零up的A9562与Baseline分类、回归、分割三个原生张量逐位相同，并与同帧原生history输出一致。共三个场景起点通过，比较容差为0：

- `b93594cfacc944bca351b979e0c37397`。
- `1759e74726f7465585fd860ec81315c5`。
- `c4048cf79e5c4b88a5845dfc98b3b1e9`。

所有完整更新边界和终点都运行frozen_check：冻结模块保持eval、requires_grad=false、无参数梯度，全部参数及BN buffers摘要不变。两个会话冻结摘要相同，具体值见result.json。

开始GPU前发现旧CPU候选尚未实现真实跨会话数值见证，因此在隔离入口补充观测；不改封存源码、Loss、学习率或优化器更新逻辑。补充后111项CPU测试通过、Black88通过；旧105测试及源码快照保留在source_before。新增容差策略在[preregistration.json](preregistration.json)中先封存，SHA256 `28b9c1e0153d4184814a9722d395c9fb75589be933645b5d659e7a520548ffb4`。

第一会话update4后，对同一下一帧只读计算Loss和全参数梯度，前后恢复RNG，不累积.grad、不更新优化器；见证写入新end Checkpoint。第二会话恢复后、update5前重建相同时序前缀并比较：

| 比较项 | 结果 | 容差 / 最大张量绝对差 |
|---|---|---|
| 参数/AdamW/scheduler/RNG，745个叶项 | 完全一致 | atol=0，rtol=0 / 0 |
| 输入摘要/Loss/18组参数梯度，35个叶项 | 完全一致 | atol=0，rtol=0 / 0 |

没有临时放宽标准。该证据验证update4跨进程边界数值连续性，不冒称额外跑过一条未中断的8更新参考训练轨迹。训练日志cursor严格1–32、token顺序与预登记32帧完全相同、每个token唯一、更新严格1–8，未发现重复/丢失。

原生GPU帧前向总计37次：第一会话17、第二会话20。其中包含跨进程history重建和边界只读见证的5次重复前向；**唯一训练帧仍为32，optimizer更新仍为8**，没有另加训练预算。

## 资源、缓存与文件保全

记录的32微步耗时合计29.034s，均值0.907s，中位数1.025s，p95=1.113s，最大1.130s。该计时不含循环外的初始化、额外梯度探测、部分场景准备及容量/持久化开销，不能按此直接承诺512帧训练总时长。

进程CPU峰值RSS 4,187,049,984 bytes（4.187GB，3.899GiB）。GPU峰值allocated 2.006GB、reserved 2.892GB。NVML采样包含桌面等全卡占用，不等于PyTorch峰值；退出后采样为2662MiB，原始采样CSV一并保留。

新增G盘资产 17,974,831 bytes（17.975MB）；共享缓存20.276226GB，合计20.294201GB；G盘空闲360.168186GB。满足新增≤8GB、合计≤30GB、空闲≥250GB。没有删除旧资产。

运行前、第一会话后和第二会话后都核对保全。最终结果：196文件原字节相同，33份正式源码集合与SHA相同，受保护目录结构相同，共享catalog摘要相同。A9562源文件SHA仍为 `b19dacd76ee5861a05d6f915e4d096250d36d7280bc3701dbf5e99e1430b038a`。详见[preservation_after.json](preservation_after.json)及两份protected快照。

原正式run继续保持FAILED / GROUNDING_NOT_READY / can_resume=false；原safe_to_shutdown=false也未改写。没有补历史退出回执。新实验safe_to_shutdown=true是本次真实退出的独立结论，不能作为原A128通过或原正式Stage B解锁。

## 下一阶段的工程判断

最小工程条件已满足：A9562参数迁移、双Head恒等、93866参数更新、三分支梯度、冻结、真实暂停/继续、零容差边界比较、退出持久化、容量及正式资产保全均通过。

仍未验证：512帧全流程吞吐与长尾、256更新长期稳定性、独立评价GT覆盖及检测/分割收益。当前仅是32帧工程证据。两会话墙钟明显长于微步合计，后续预算须计入初始化、探测、I/O、容量检查和保存；保留27000秒上限及提前600秒保存策略，不能用纯微步时间推算总时间。

本轮已停止。下一阶段256更新或64更新探索训练、Development GT评价需要分别取得新的明确授权；不得使用本次仅preflight授权启动这些动作，不得绕过原A128。

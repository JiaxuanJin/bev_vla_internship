# LG-BEV 每日离线运行与安全交付

2026-10-28 Git 交付检查：**正式训练尚未启动，正式更新 0，BLOCKED**；
GT 仅完成 5787/19123 帧，640 块校验通过；V3 为 DRAFT_NOT_APPLIED。
本轮不执行 GT 构建、GPU 验收或正式 start/resume，commit/push 等待用户确认。
本页“今日/明天”指 2026-10-27/28 的历史安排，不构成启动授权。
公开仓库的代码入口、文件职责和资产前提见 [源码索引](README.md)。

2026-10-27 **禁止正式 start/resume，正式更新数为0**。即使工程达到READY，也不会自动启动。首次7.5小时会话推迟到2026-10-28，仍须用户当天新的明确授权；日期到达不能自动解锁。训练入口的 `launch_hold_20261027.json` 与独立授权校验保证这一点。今日记录只追加到 `docs/logbook_2026-10-27.md`。

用户附件已把累计活跃时间从24小时改为48小时，每次最多7.5小时，提前约120秒请求保存；旧合同不覆盖。新增时间协议为 `results/lg_bev_daily_20261027/time_protocol_v2.json`。A2/B4、19123个训练帧、28686次更新、所有损失/种子/FP32/冻结边界及数据隔离不变。停机等待不计入累计活跃时间；模型加载、场景准备、检查点和A终点128帧探针计入，不允许每次恢复清零。

## 命令与明日步骤

以下正式命令今天均不执行。明天先让Codex读取今日准入和状态，再发出新的明确授权。只有在用户授权记录绑定当前hold/协议和实际新请求，且 `admission.json` 为READY时，才允许首次start。授权文件今天不存在；不得通过伪造文件、改时间或删除门槛启动。

```bash
cd ~/projects/bev_vla
# 明天经新授权、完整准入核对之后才执行，首次只启动一个7.5小时会话：
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_daily start
```

状态监控不启动训练，可随时执行：

```bash
cd ~/projects/bev_vla
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_daily status
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_daily status --watch
nvidia-smi
```

启动后，实际run是 `G:\BEV_VLA_Datasets\lg_bev_rolling_cache\runs\lg_bev_formal_48h_20261027_v1`。进程在独立tmux内，退出Codex、关闭监控或终端不会依赖API维持计算；退出WSL、重启Windows、断电仍会中断，不能把tmux称为断电保护。实时摘要包含阶段/epoch/帧/更新/累积梯度、loss/lr、会话和累计时间、显存/RSS、GPU采样、最新checkpoint路径/SHA。GPU采样每30秒，进度约每2秒。可另开终端查看：

```bash
tail -F /mnt/g/BEV_VLA_Datasets/lg_bev_rolling_cache/runs/lg_bev_formal_48h_20261027_v1/control.jsonl
tmux -S /tmp/lg_bev_daily_20261027.sock attach -t lg-bev-formal
```

`status --watch` 用Ctrl-C只退出监控；tmux用Ctrl-B后按D分离。人工暂停：

```bash
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_daily stop --timeout 120
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_daily status
```

只有 `STOPPED/COMPLETED`、`process_alive=false`、`safe_to_shutdown=true`，且checkpoint和时间receipt重新校验成功，才可手动关机。`STOPPING`、`FAILED`、`FAILED_UNCLEAN`均不能当作已安全保存。自动暂停不自动关机，不自动衔接下一天。下一次会话需用户明确执行恢复：

```bash
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_daily resume
```

恢复使用最后一条已fsync提交事件绑定的checkpoint，含Adapter、AdamW/scheduler、CPU/CUDA/NumPy/Python RNG、epoch/cursor、部分累积梯度和累计时间。最新提交损坏时拒绝，不静默倒退。失败需先诊断；COMPLETED不重复训练。突发SIGKILL/断电仅能回到最后持久checkpoint，未保存工作可能重放，不承诺逐次更新exactly-once。

每段安全停止后，可从真实事件/检查点追加当天唯一中文logbook，不会重复登记同一提交：

```bash
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_session_logbook --record
```

## 今日工程验证与复现

每日控制与工程状态机回归：

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv-bevformer/bin/python -m pytest -q
.venv-bevformer/bin/python -m pyflakes scripts/stage3/lg_bev_daily*.py scripts/stage3/lg_bev_session_*.py scripts/stage3/lg_bev_data_parallel.py
```

真实三场景验收只用新的train场景scene-0004/0041/0045，固定12/12/8帧。已有报告拒绝覆盖；重现需全新G盘run目录。先prepare生成原始FP32和不中断参考，然后控制测试在独立 `daily_acceptance_*` run执行3秒自动暂停/人工停止/恢复，**合计仅32microsteps、8次测试更新，不能晋升正式进度**：

```bash
LG_PREFLIGHT=/mnt/g/BEV_VLA_Datasets/lg_bev_rolling_cache/runs/daily_preflight_repro_01
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_daily_acceptance prepare --run-dir "$LG_PREFLIGHT"
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_session_control_test --config "$LG_PREFLIGHT/acceptance_config.json"
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_daily_acceptance compare --run-dir "$LG_PREFLIGHT"
```

验收器强制run前缀必须为 `daily_acceptance_`；复现目录名须包含 `daily_preflight`，因此本例替换后合法。首次短测明确排除资产加载以便实测3秒暂停，正式会话始终包含资产加载，两者不能混算时间。

全量GT从真实annotation/map/calibration重建；模板明确标GT派生。大表一次流式索引，原NuScenes SDK构建同一场景视图，旧视图逐字段一致；标签算法未改。两个CPU分片仅改变构建任务分配，合并后仍是原封存帧顺序。原始来源首次实际SHA，后续只读数据用路径/size/mtime及manifest校验复用；与全量GT捕获的来源SHA逐场景对照，改变或损坏即停止。

全部大型GT、FP32参考、缓存/数据库/checkpoint/临时文件在G盘固定根目录，30GB包含所有这些产物，空闲至少250GB；不向C/D/E新增大型数据，不扩容、不删旧结果腾空间。原生模型/BN/文本冻结，真实空GT保留全部负例，采样器的 `[0,4]` 回退只在训练loss内修正为 `[0,9]`。

最终准入、真实耗时/误差/GT数量、Git人工白名单以 `results/lg_bev_daily_20261027/` 与今日logbook为准。工程READY不等于模型有效，也不保证AP/mIoU提升；开发和Cold Final仍封存。

## 今日暂停及最新资源修订

今日安全收尾已用本tmux前台的正常Ctrl-C暂停GT，原协调器PID128978、worker PID129243/129244均退出，没有残留LG-BEV进程。已校验5787个独立帧/640个GT块，剩余13336帧；已提交范围失败0，SQLite integrity_check=ok。没有生成虚假的19123帧完整报告。所有有效提交、未提交残留和日志保留，今天可以就LG-BEV部分安全关闭Windows。

明天恢复**GT准备**，以下命令不启动训练，并复用SQLite中所有已校验记录及固定分片spec；只有未提交工作可能重建：

```bash
cd ~/projects/bev_vla
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
tmux -S /tmp/lg_bev_daily_20261027.sock new-session -d -s lg-bev-gt-batch -c "$PWD" 'set -C; exec .venv-bevformer/bin/python -u -m scripts.stage3.lg_bev_gt_completion > /mnt/g/BEV_VLA_Datasets/lg_bev_rolling_cache/gt_resume_20261028_01.log 2>&1'
```

日志文件名需选未使用的新名字；若同名tmux仍在运行，先检查状态，不能重复启动。需要正常暂停GT时，附加本会话按Ctrl-C（今日已实际使用），等待父子进程退出，再复核已提交块和SQLite；不得强杀、直接关机或删除数据库/旧标签。今日父子进程的KeyboardInterrupt是正常中断记录，不是GT验证失败。暂停前已提交块均已逐项核验，恢复路径的pending识别另有CPU测试；未声称全量断电恢复已经实测。

```bash
tmux -S /tmp/lg_bev_daily_20261027.sock attach -t lg-bev-gt-batch
tail -F /mnt/g/BEV_VLA_Datasets/lg_bev_rolling_cache/gt_resume_20261028_01.log
```

最新用户要求取消累计48小时硬上限，改为累计监控和预警，单会话仍最多7.5小时。已提交独立草案 `docs/lg_bev_resource_protocol_v3_draft_2026-10-27.md`，状态DRAFT_NOT_APPLIED，原V1/V2保留。当前运行代码仍是V2且正式入口被hold锁定，不可改配置绕过。

**正式训练仍BLOCKED**：全量GT/来源封存未完成；完整真实start/自动暂停/人工停止/跨进程恢复对照未验收完成；V3代码切换及严格会话保存时间准入未完成。一次有界测试start保存了1帧部分梯度、0次更新，出现启动PID竞态，已修复但未继续开新CUDA验收；不能把230项CPU回归当作真实跨会话全部通过。遥测现在终端约2秒刷新、磁盘摘要至多60秒刷新，关键checkpoint/会话结束仍fsync；旧文中约2秒的状态描述指终端输出，磁盘摘要可能延后。

跨场景历史保守估计73.8648/60.8789小时全部保留，当前V5保守估计53.2925小时。V3将这些视为资源预警，不是自动READY的理由；当前V2源码仍会拒绝超过48小时。今天没有正式start/resume或正式优化器更新，没有开发/Cold Final评价。

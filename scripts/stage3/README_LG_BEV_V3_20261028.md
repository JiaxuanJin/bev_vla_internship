# LG-BEV V3 正式训练管理与最终准入

V3执行协议与run-id独立于旧V2，累计硬上限为null；每会话最多7.5小时，提前600秒请求保存。
**最终准入READY，等待用户单独授权；正式6 Epoch尚未启动，正式更新0。**
本轮完成260项CPU回归和32帧真实三会话完整恢复验收，Loss/梯度/参数差均0，AdamW/Scheduler/RNG一致。

入口：[执行协议](../../docs/lg_bev_resource_protocol_v3_2026-10-28.md) ·
[今日唯一Logbook](../../docs/logbook_2026-10-28.md) ·
[V3准入](../../results/lg_bev_v3_20261028/admission.json)。
准入报告只在真实门槛全过后生成；缺失时保持BLOCKED。

## 四条命令

在仓库根目录、已有兼容环境的普通WSL终端执行。前两条启动/恢复命令须等新的正式训练授权，
本轮没有执行；status/stop操作不会自动启动训练。

```bash
cd ~/projects/bev_vla
# 首次启动一个会话（READY且另行明确授权后）
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_daily start --protocol v3
# 实时监控，Ctrl-C只退出监控
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_daily status --protocol v3 --watch
# 正常提前停止，等待保存与实际退出；超时仅表示仍在STOPPING
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_daily stop --protocol v3 --timeout 600
# 下次人工恢复同一个run，不清零epoch/optimizer/scheduler/RNG/累计时间
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_daily resume --protocol v3
```

不带 `--protocol` 默认也为V3；旧V2只有显式指定才进入历史分支。
正式目录为G盘 `lg_bev_rolling_cache/runs/lg_bev_formal_v3_20261028_v1`。
tmux socket沿用 `/tmp/lg_bev_daily_20261027.sock`，正式session为 `lg-bev-formal-v3`。

## 离线与安全关机

模型、Tokenizer和数据从本机加载，Python网络连接被实际拒绝；可手动断开Wi-Fi。
tmux及其监督进程独立于Codex/VS Code连接。关闭监控终端、退出聊天不停止训练，
关闭WSL、Windows重启或断电会中断，不能把tmux当成断电保护。

达到保存窗口后自动停止，**不会自动关机或自动resume**。关机前再次运行status（可去掉--watch），
要求STOPPED/COMPLETED、`process_alive=false`、`safe_to_shutdown=true`，checkpoint、时间receipt、
执行协议和实际退出receipt均核验成功。STOPPING/FAILED不能据此关机。
正常关机后下次手工resume从最后有效提交继续；突发断电最多回到最后持久提交，未保存工作可能重放。

## 验收和资源

正式GT为19123帧，正式数据入口已生成PASS来源封存及32帧独立对照。
当前源码的V3独立32帧验收复用旧FP32参考，只补必要的真实前向/反向、停止和恢复；
验收更新不能计入正式28686次更新。正式开发/Cold Final不打开。

G盘根目录30GB、最低空闲250GB不变，所有新大型缓存/checkpoint/参考张量只在G盘。
历史53.29小时等保守估计继续披露，V3把它们作为资源/跨天安排预警，不据此降低训练预算。
READY之后仍须用户新的明确授权；`launch_hold_v3.json`保留，授权文件本轮不创建。

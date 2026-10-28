# LG-BEV GitHub 拟交付检查（2026-10-28）

交付状态：**待用户确认，尚未 commit/push**。本轮不启动 GT 构建、GPU 实验或正式训练。
下列清单是逐文件暂存白名单；未列出的历史文件保留在本机。

## 工程真实状态

| 项目 | 结论 |
|---|---|
| 正式训练 | **尚未启动、正式更新 0、BLOCKED**；正式 run 不存在 |
| GT | **5787/19123 帧**已提交，剩余 13336 帧；**640 块**通过完整性校验；SQLite integrity_check=ok |
| 资源协议 | 代码仍使用 V2；V3 为 **DRAFT_NOT_APPLIED**，不能将草案当作已生效 |
| 未完成准入 | 全量 GT/来源封存、完整真实暂停恢复对照、V3 实施及严格会话保存时间验收 |
| 评价 | 新正式 development/cold_final 未评价；无正式精度提升或达标结论 |
| 历史工程测试 | smoke、16 步 pilot、32 帧工程更新各自隔离，不能记入正式 28686 更新 |

以上 GT 和过程状态来自昨日最终收尾记录，本轮未重新扫描 G 盘数据库或 GT 块，
未重复既有 GPU 实验。当前未发现运行中的 LG-BEV Python 模块进程。

## Git 与选取依据

- 当前分支 `main`，HEAD 为 `19e8a8a347421f8988f4fff26cd4ad28abdfad5e`。
- origin：<https://github.com/JiaxuanJin/bev_vla_internship.git>。
- 只读实时 `git ls-remote --heads origin` 仅列出 `main`，与本地 HEAD 相同；本地 `origin/main` 也相同，无领先/落后提交。未执行 fetch、commit 或 push。
- 开始时暂存区为空，存在历史未提交文件。包括已修改的 `docs/stage2_oom_fix_20261022.md`、根目录使用说明、旧任务脚本和历史结果；它们不在交付清单中。
- 源码/测试/既有文档依据六份人工白名单逐项合并：本机 `results/` 下的 `lg_bev_20261027`、`lg_bev_supervision_space_20261027`、`lg_bev_formal_training_audit_20261027`、`lg_bev_gpu_release_20261027`、`lg_bev_engineering_handoff_20261027`、`lg_bev_daily_20261027` 中的 `git_whitelist.json`。
- 最后一轮白名单是增量，单独提交会缺失基础模块；本次包含依赖闭包内的 46 个 LG-BEV 源码及 7 个测试，未更改其字节。未过数值门槛的动态 CUDA 文件是已有条件依赖，明确标为禁用研究候选，不能作为通过验收的实现。
- 本轮另补根 README、源码逐文件索引、测试索引及本检查清单。7 个 JSON 只保留固定合同/V2/V3、启动 hold 和必要收尾元数据；其中最终 closing/status 由用户指定的最终记录补入，未打包整个结果目录。

## 文档与复现检查

根 README 和四份历史 LG-BEV README、训练方案均标明最新状态与历史边界，
源码索引逐一说明 46 个文件的作用，测试索引逐一说明 7 个测试文件。
历史 24 小时估计、旧实现状态和失败实验原样保留在正文，最新说明指向每日收尾。

个人主目录路径改为 `~/projects/...` 示例；tmux GT 示例使用 `-c "$PWD"`，
并保留 `set -C` 防止覆盖已有日志。昨日唯一中文 Logbook 仅替换这一条命令中的个人路径，
不新增或改写实验结论；未新建另一份每日 Logbook。原始文档已在本机保留副本。

随交付的 JSON 不改写字节。`closing_completion.json` 中的 Logbook SHA
`5ac3f2ebc3fe7cadc243615018deb278f910d424cd334b3937a6f05eade40e1a`
指向昨日原始快照，不能拿它校验公开路径调整后的 Logbook。
原合同的“未实现”字段同样是历史设计状态，不能覆盖当前 README 中的实际实现进度。

公开文档区分代码复核与真实资产复现：CPU 回归使用已有兼容环境，无需 GPU 或原始数据；
真实实验仍依赖本地数据、权重、作者工程、旧结果和未交付的大清单，不能仅凭克隆重跑。
`admission.json` 含个人绝对路径，保留在本机，未上传或伪造可运行的准入证据。

## 验证结果

- 53 个 Python 文件语法检查通过，Black 88 无需改动，pyflakes 0 项问题；仓库内 Python 导入依赖完整。
- CPU 完整回归：**230 passed、19 warnings in 3.43s**；其中 LG-BEV 收集 90 项。禁用 CUDA、禁用模型在线加载，未运行 GT 或训练入口。
- 受限沙箱首先得到 229 passed、1 failed：系统禁止创建 socket。允许本机 socket 后同一完整回归通过，测试与门槛未修改；离线防护仍拒绝 connect。日志仅保留本机。
- UTF-8 严格解码、中文替换字符、JSON、Markdown 围栏/本地链接、Bash 语法、敏感凭据及个人主目录扫描通过；外部引用链接仅做文档保留，本轮未重审上游网页。
- 所有源码和测试均与本轮开始时一致；昨日最终 source manifest 中的 15 个源码/测试 SHA 也一致。协议、hold 和收尾 JSON 保持原始字节。
- 暂存区逐文件核对完整白名单、差异、文件类型与大小；仅含 Git 版本文件的临时快照也通过完整 CPU 回归（230 passed、19 warnings），无需本机未交付结果。无关的 275 个未跟踪文件大小/修改时间及既有第二阶段未提交差异与开始时一致。

## 明确排除与剩余边界

不上传 G 盘原始数据、GT 张量、BEV/teacher 缓存、权重/checkpoint、SQLite/其他数据库、
大型日志、profiler trace、传感器图像、虚拟环境、逐帧大清单、旧失败目录或整包实验产物。
`closing_gt_progress.json` 是约 110 KiB 的提交数量/块 SHA 元数据，不含 GT 张量。
公开保留的 `/mnt/g/...` 是必需的本地资产布局，不包含个人主目录或登录凭据。

未执行 `git add .`、clean、reset 或 stash；无关未提交文件保持原状。
本轮只获准准备和暂存交付；确认后才能执行 commit 和 push，当前不自动开展后续训练工作。

## 拟提交消息与 GitHub 路径

```text
feat(stage3): deliver LG-BEV engineering with formal training blocked

Add LG-BEV source, CPU tests, file indexes and fixed resource protocols.
Record GT at 5787/19123 frames with 640 verified blocks.
Keep formal optimizer updates at zero and resource V3 as DRAFT_NOT_APPLIED.
Exclude datasets, weights, caches, databases and profiler artifacts.
```

GitHub 项目：<https://github.com/JiaxuanJin/bev_vla_internship>，拟推送分支 `main`。
提交/推送必须等待用户确认本清单；届时仍需复核分支、远端和暂存内容没有变化。

## 最终拟提交文件清单

共 **71 个文件：46 个源码、7 个测试、11 份 Markdown、7 个 JSON**。
以下为完整的显式白名单，大小为工作文件字节数；本清单自身大小以 Git 暂存检查为准。

### 源码

| 文件 | 字节 |
|---|---:|
| `scripts/stage3/lg_bev_common.py` | 2519 |
| `scripts/stage3/lg_bev_coordinate_cache_replay.py` | 3656 |
| `scripts/stage3/lg_bev_coordinate_core.py` | 6640 |
| `scripts/stage3/lg_bev_coordinate_independent_gt.py` | 7694 |
| `scripts/stage3/lg_bev_coordinate_raw_probe.py` | 4195 |
| `scripts/stage3/lg_bev_coordinate_rca.py` | 16536 |
| `scripts/stage3/lg_bev_daily.py` | 25781 |
| `scripts/stage3/lg_bev_daily_acceptance.py` | 18489 |
| `scripts/stage3/lg_bev_daily_metadata.py` | 10196 |
| `scripts/stage3/lg_bev_daily_state.py` | 7229 |
| `scripts/stage3/lg_bev_data.py` | 23339 |
| `scripts/stage3/lg_bev_data_parallel.py` | 6869 |
| `scripts/stage3/lg_bev_engineering_audit.py` | 18177 |
| `scripts/stage3/lg_bev_engineering_cache.py` | 19803 |
| `scripts/stage3/lg_bev_engineering_data.py` | 30747 |
| `scripts/stage3/lg_bev_engineering_determinism.py` | 16023 |
| `scripts/stage3/lg_bev_engineering_dynamic_cuda.py` | 7472 |
| `scripts/stage3/lg_bev_engineering_evaluate.py` | 19573 |
| `scripts/stage3/lg_bev_engineering_fast_verify.py` | 8905 |
| `scripts/stage3/lg_bev_engineering_full.py` | 25111 |
| `scripts/stage3/lg_bev_engineering_loss.py` | 8303 |
| `scripts/stage3/lg_bev_engineering_metrics.py` | 9164 |
| `scripts/stage3/lg_bev_engineering_native.py` | 14892 |
| `scripts/stage3/lg_bev_engineering_recovery.py` | 3309 |
| `scripts/stage3/lg_bev_engineering_run.py` | 5279 |
| `scripts/stage3/lg_bev_engineering_train.py` | 24069 |
| `scripts/stage3/lg_bev_evaluate.py` | 13234 |
| `scripts/stage3/lg_bev_formal_checkpoint.py` | 9398 |
| `scripts/stage3/lg_bev_formal_coverage.py` | 5776 |
| `scripts/stage3/lg_bev_formal_inventory.py` | 12732 |
| `scripts/stage3/lg_bev_formal_pilot_analysis.py` | 10832 |
| `scripts/stage3/lg_bev_formal_protocol.py` | 25751 |
| `scripts/stage3/lg_bev_gpu_benchmark.py` | 10041 |
| `scripts/stage3/lg_bev_gpu_retained.py` | 10871 |
| `scripts/stage3/lg_bev_gt_completion.py` | 10981 |
| `scripts/stage3/lg_bev_native.py` | 4440 |
| `scripts/stage3/lg_bev_prototype.py` | 12030 |
| `scripts/stage3/lg_bev_session_admission.py` | 9872 |
| `scripts/stage3/lg_bev_session_control_test.py` | 5023 |
| `scripts/stage3/lg_bev_session_logbook.py` | 2728 |
| `scripts/stage3/lg_bev_spatial_review.py` | 5213 |
| `scripts/stage3/lg_bev_supervision_metrics.py` | 18490 |
| `scripts/stage3/lg_bev_supervision_run.py` | 17722 |
| `scripts/stage3/lg_bev_supervision_space.py` | 5176 |
| `scripts/stage3/lg_bev_train.py` | 19652 |
| `scripts/stage3/lg_bev_verify.py` | 3483 |

### 测试

| 文件 | 字节 |
|---|---:|
| `tests/test_lg_bev_20261027.py` | 9714 |
| `tests/test_lg_bev_coordinate_rca.py` | 4716 |
| `tests/test_lg_bev_daily_20261027.py` | 9553 |
| `tests/test_lg_bev_engineering_20261027.py` | 20694 |
| `tests/test_lg_bev_formal_audit_20261027.py` | 6253 |
| `tests/test_lg_bev_gt_batch_recovery.py` | 1417 |
| `tests/test_lg_bev_supervision_space.py` | 3732 |

### README、协议说明与唯一 Logbook

| 文件 | 字节 |
|---|---:|
| `README.md` | 15963 |
| `docs/lg_bev_formal_training_protocol_2026-10-27.md` | 24451 |
| `docs/lg_bev_github_delivery_2026-10-28.md` | 暂存检查 |
| `docs/lg_bev_resource_protocol_v3_draft_2026-10-27.md` | 6415 |
| `docs/logbook_2026-10-27.md` | 67797 |
| `scripts/stage3/README.md` | 11340 |
| `scripts/stage3/README_LG_BEV_20261027.md` | 9132 |
| `scripts/stage3/README_LG_BEV_DAILY_20261027.md` | 9274 |
| `scripts/stage3/README_LG_BEV_ENGINEERING_20261027.md` | 16193 |
| `scripts/stage3/README_LG_BEV_SPACE_20261027.md` | 11496 |
| `tests/README.md` | 4901 |

### 固定协议与收尾元数据

| 文件 | 字节 |
|---|---:|
| `results/lg_bev_daily_20261027/closing_completion.json` | 622 |
| `results/lg_bev_daily_20261027/closing_gt_progress.json` | 112678 |
| `results/lg_bev_daily_20261027/launch_hold_20261027.json` | 434 |
| `results/lg_bev_daily_20261027/production_status_final.json` | 141 |
| `results/lg_bev_daily_20261027/resource_protocol_v3_draft.json` | 785 |
| `results/lg_bev_daily_20261027/time_protocol_v2.json` | 840 |
| `results/lg_bev_formal_training_audit_20261027/formal_execution_contract.json` | 5153 |

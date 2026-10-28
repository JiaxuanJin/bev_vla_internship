# 第三阶段 LG-BEV 源码与复现索引

2026-10-28 最新状态：**V3 工程准入 READY，正式训练未启动、正式更新 0，等待单独授权**。
GT **19123/19123 帧**、476场景、35日志，正式来源验收PASS、选中块970个。
新正式CLI默认V3，累计硬上限null、48/72小时只预警；单会话27000秒，提前600秒保存。
真实自动暂停→人工stop→跨进程resume完整状态对照通过。四条命令与关机规则见
[V3每日操作](README_LG_BEV_V3_20261028.md)。旧24/48小时协议和失败结果仍保留。

历史两步 smoke、唯一 16 步 pilot 与有界工程更新各有独立记录，不计入正式 28686 次更新。
历史 pilot 没有证明相关精度提升；新正式 development/cold_final 仍未评价。
260 项 CPU 工程回归及32帧真实控制验收不能替代正式训练或项目精度验收。

## 阅读入口与协议优先关系

| 入口 | 内容 |
|---|---|
| [V3每日操作](README_LG_BEV_V3_20261028.md) | 当前READY状态、独立run-id、四条命令及安全关机 |
| [历史V2操作](README_LG_BEV_DAILY_20261027.md) | 2026-10-27暂停状态和当时未完成项，保留历史语义 |
| [工程实现](README_LG_BEV_ENGINEERING_20261027.md) | FP32 时序、rolling cache、梯度/恢复/计分；24 小时章节是历史记录 |
| [正式固定方案](../../docs/lg_bev_formal_training_protocol_2026-10-27.md) | 固定 split、A2+B4、loss、冻结边界和计分定义，保留原始设计历史 |
| [V3执行协议](../../docs/lg_bev_resource_protocol_v3_2026-10-28.md) | 已实施并验收；原草案保留，不授予训练权限 |
| [监督空间与 pilot](README_LG_BEV_SPACE_20261027.md) | 实际 Head 根因更正、独立监督空间及失败的历史性能结果 |
| [早期原型](README_LG_BEV_20261027.md) | 首次 smoke 和旧空间门槛；15m 诊断已更正 |
| [昨日唯一中文 Logbook](../../docs/logbook_2026-10-27.md#后续每日离线运行准备全量gt部分构建与今日安全收尾) | 同日各轮历史，末节为最终收尾状态 |
| [今日唯一中文 Logbook](../../docs/logbook_2026-10-28.md) | GT修复/完成、V3实施及当前真实准入证据 |
| [Git 拟交付清单](../../docs/lg_bev_github_delivery_2026-10-28.md) | 逐文件白名单、核查结果、提交消息和排除范围 |

原 24 小时合同保留原字节；V2 仅补充资源时间语义，launch hold 仍有效。
V3执行协议独立新建，原V3草案保留为历史。日期到达、测试通过或READY均不自动授权训练。
旧文档的“当前”“今日”“未实现”和旧吞吐值按其记录阶段理解，不能覆盖末节状态。

## 47 个源码文件的职责

以下每行对应一个实际文件；路径相对于本目录。`python -m scripts.stage3.<模块名>`
须在仓库根目录执行。历史/GPU/构建入口须依各自协议使用，不据源码存在宣称实验完成。

| 基础与历史链路 | 职责 |
|---|---|
| [lg_bev_prototype.py](lg_bev_prototype.py) | LG-BEV Adapter、受限残差和显式 mask 损失 |
| [lg_bev_common.py](lg_bev_common.py) | 仓库路径、只增写 JSON、SHA 和隔离检查 |
| [lg_bev_data.py](lg_bev_data.py) | 早期场景划分、真实实例/地图标签、标定投影 |
| [lg_bev_native.py](lg_bev_native.py) | 缓存 BEV 的冻结双 Head 接入、损失与状态摘要 |
| [lg_bev_train.py](lg_bev_train.py) | 早期 smoke/pilot、梯度审计及配对诊断 |
| [lg_bev_evaluate.py](lg_bev_evaluate.py) | 早期内部开发计分与 BLOCKED 的 NOT_RUN 报告 |
| [lg_bev_verify.py](lg_bev_verify.py) | 历史 checkpoint 的零更新前后向复核 |
| [lg_bev_spatial_review.py](lg_bev_spatial_review.py) | 首次 CPU cropper 诊断，结论须结合后续根因更正 |

| 坐标与监督空间 | 职责 |
|---|---|
| [lg_bev_coordinate_core.py](lg_bev_coordinate_core.py) | 实际 Head、坐标变换和已拒绝桥接的历史诊断 |
| [lg_bev_coordinate_rca.py](lg_bev_coordinate_rca.py) | 真实运行时 hook、ramp、decoder 与原始图像复核 |
| [lg_bev_coordinate_raw_probe.py](lg_bev_coordinate_raw_probe.py) | 原始图像/LiDAR 解码、checkpoint 读取 |
| [lg_bev_coordinate_cache_replay.py](lg_bev_coordinate_cache_replay.py) | 按原增强和种子重放旧缓存来源 |
| [lg_bev_coordinate_independent_gt.py](lg_bev_coordinate_independent_gt.py) | 独立原始地图/标定 GT 核对 |
| [lg_bev_supervision_space.py](lg_bev_supervision_space.py) | Attention 独立物理监督空间及 BCE/Dice |
| [lg_bev_supervision_run.py](lg_bev_supervision_run.py) | 独立监督验收与历史固定 16 步 pilot |
| [lg_bev_supervision_metrics.py](lg_bev_supervision_metrics.py) | 该历史 pilot 的四分支计分 |

| 固定方案审计 | 职责 |
|---|---|
| [lg_bev_formal_inventory.py](lg_bev_formal_inventory.py) | 原始资产覆盖盘点、历史暴露隔离 |
| [lg_bev_formal_checkpoint.py](lg_bev_formal_checkpoint.py) | joint 权重身份、内嵌配置与 Head 状态审计 |
| [lg_bev_formal_coverage.py](lg_bev_formal_coverage.py) | 地图/对象标签覆盖和文件可读取性 |
| [lg_bev_formal_pilot_analysis.py](lg_bev_formal_pilot_analysis.py) | 历史 pilot 的训练侧只读 RCA |
| [lg_bev_formal_protocol.py](lg_bev_formal_protocol.py) | 日志隔离划分、固定预算、合同封存及只读核验 |

| 工程与数值验收 | 职责 |
|---|---|
| [lg_bev_engineering_cache.py](lg_bev_engineering_cache.py) | G 盘归属、容量/保留空间、锁、分片事务与恢复 |
| [lg_bev_engineering_data.py](lg_bev_engineering_data.py) | 正式 split 的原始 GT、GT 派生指令及无损标签块 |
| [lg_bev_engineering_native.py](lg_bev_engineering_native.py) | 六相机原生时序、严格 FP32、来源清单与前缀重建 |
| [lg_bev_engineering_loss.py](lg_bev_engineering_loss.py) | 原生任务/ROI/保护/排序损失及空 3D GT 兼容 |
| [lg_bev_engineering_train.py](lg_bev_engineering_train.py) | A/B 累积更新、优化器/RNG/部分梯度持久化及共享时钟 |
| [lg_bev_engineering_full.py](lg_bev_engineering_full.py) | 全量装配、准入、A128 端点探针及开发终点锁 |
| [lg_bev_engineering_evaluate.py](lg_bev_engineering_evaluate.py) | 四分支 conditional AP、ROI/global IoU 和分场景统计 |
| [lg_bev_engineering_metrics.py](lg_bev_engineering_metrics.py) | 流式 SQLite AP 与场景 bootstrap |
| [lg_bev_engineering_determinism.py](lg_bev_engineering_determinism.py) | 固定网格伴随和保持原 forward 的确定性反向 |
| [lg_bev_engineering_dynamic_cuda.py](lg_bev_engineering_dynamic_cuda.py) | 未过数值门槛的动态 CUDA 候选，正式入口禁用；依赖保留 |
| [lg_bev_engineering_run.py](lg_bev_engineering_run.py) | 32 帧工程准备/验收分阶段入口 |
| [lg_bev_engineering_audit.py](lg_bev_engineering_audit.py) | 原生预测不变量、缓存/流式与恢复的有界验收 |
| [lg_bev_engineering_recovery.py](lg_bev_engineering_recovery.py) | 分片落盘中断的独立子进程实验 |
| [lg_bev_engineering_fast_verify.py](lg_bev_engineering_fast_verify.py) | 快速图像流水线与原生输出/训练等价复核 |
| [lg_bev_gpu_benchmark.py](lg_bev_gpu_benchmark.py) | 候选实现的数值、GPU/CPU/传输诊断 |
| [lg_bev_gpu_retained.py](lg_bev_gpu_retained.py) | 已保留路径相对参考的有界吞吐和数值验收 |

| 每日控制与 GT 准备 | 职责 |
|---|---|
| [lg_bev_v3_acceptance.py](lg_bev_v3_acceptance.py) | 当前三场景32帧参考、真实三会话控制、完整状态逐项比较 |
| [lg_bev_daily.py](lg_bev_daily.py) | start/status/stop/resume、tmux worker、离线与授权检查 |
| [lg_bev_daily_state.py](lg_bev_daily_state.py) | V2 时钟、协议、checkpoint/receipt 校验和事件 |
| [lg_bev_daily_metadata.py](lg_bev_daily_metadata.py) | 原始大表流式索引、场景视图与来源指纹缓存 |
| [lg_bev_daily_acceptance.py](lg_bev_daily_acceptance.py) | 三场景原生参考及不中断控制参考；不是正式训练 |
| [lg_bev_data_parallel.py](lg_bev_data_parallel.py) | CPU 分片标签构建和提交合并 |
| [lg_bev_gt_completion.py](lg_bev_gt_completion.py) | 正式 train 的按场景 GT 块继续构建与 pending 恢复 |
| [lg_bev_session_admission.py](lg_bev_session_admission.py) | 全量标签/来源和控制资源准入封存，尚未完整通过 |
| [lg_bev_session_control_test.py](lg_bev_session_control_test.py) | 真实暂停/恢复控制驱动，完整对照尚未验收 |
| [lg_bev_session_logbook.py](lg_bev_session_logbook.py) | 从未来真实会话事件追加唯一中文日志 |

9 个测试文件及用途见 [测试 README](../../tests/README.md)。未迁移或改写历史 V1～V5。

## CPU 复核命令

在仓库根目录，使用已有兼容环境执行；只运行 CPU 工程测试：

```bash
cd ~/projects/bev_vla
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv-bevformer/bin/python -m pytest -q -p no:cacheprovider
.venv-bevformer/bin/python -m pyflakes scripts/stage3/lg_bev*.py tests/test_lg_bev*.py
```

预期全库 260 passed、20 warnings；其中 120 项来自 LG-BEV 测试文件。
本地 socket 创建权限是离线拒绝测试的必要条件；不需要联网下载或 GPU。
Black 88 检查请逐文件调用，避免该环境已知的多进程池挂起：

```bash
for lg_file in scripts/stage3/lg_bev*.py tests/test_lg_bev*.py; do
  .venv-bevformer/bin/python -m black --check --line-length 88 "$lg_file" || break
done
```

## 路径、资产和公开复现限制

`~/projects/bev_vla` 是工作目录示例，需换为实际克隆路径。
源码用 `lg_bev_common.ROOT` 定位仓库，用 `Path.home()` 定位作者工程。
真实运行目前保留本机 WSL 的 G 盘资产路径，尚未提供跨机器自动配置：

| 本地约定 | 用途 |
|---|---|
| `~/projects/bevformer_author_clean_5080/BEVFormer-seg` | 只读第三方作者模型代码 |
| `/mnt/g/nus/bevformer_small_joint_author.pth` | 冻结 joint 权重，SHA 见训练方案 |
| `models/distilbert-base-uncased/` | 只读本地文本编码器/Tokenizer |
| `/mnt/g/BEV_VLA_Datasets/` | nuScenes、地图、标定等合法本地原始资产 |
| `/mnt/g/BEV_VLA_Datasets/lg_bev_rolling_cache/` | GT/缓存/数据库/checkpoint 的受控根目录；总量 30GB，空闲至少 250GB |

G 盘对应 Windows 的 `G:\BEV_VLA_Datasets\`；这些是资产布局说明，不是仓库内交付文件。
Python 3.10.12、PyTorch 2.8.0+cu128 及已有 MMCV/MMDetection3D 兼容环境为本机记录，
不保证任意新环境直接可用；`.venv-bevformer/` 与第三方工程不随 Git 分发。

公开仓库交付源码、测试、README、固定合同/V2/V3执行协议及少量摘要；旧V3草案保留。
历史结果目录、原始审计、逐帧 split/来源大清单、标签、FP32 参考、模型权重、数据库和
profiler trace 保留在本机，不能仅凭克隆执行历史真实数据复现或正式准入。
其中 `admission.json` 含个人绝对路径，留在本机；缺失时管理器继续显示 BLOCKED。
早期文档中未交付的证据路径均为本机历史引用，不表示 GitHub 上有完整实验资产。

随交付保留的合同、V2/V3 JSON 和收尾 JSON 均不改写；其中源码/Logbook SHA 指向
2026-10-27 原始快照。公开文档添加状态说明并移除个人主目录后，文档 SHA 会改变，
不能用历史文档 SHA 校验公开改版，也不能把旧“未实现”合同字段当作今日实现清单。
2026-10-28的GT校验修复和V3代码/测试修改以今日Logbook及新的Git白名单为准，
不能把旧源码封存SHA当作今日已验收版本的SHA。

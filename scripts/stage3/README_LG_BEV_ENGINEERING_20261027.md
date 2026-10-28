# LG-BEV 正式工程与 GPU 验收（2026-10-27）

最新收尾状态：**正式训练未启动，正式更新 0，BLOCKED**；GT 仅完成 5787/19123 帧，
640 块校验通过；V3 为 DRAFT_NOT_APPLIED，运行代码仍为 V2。
本页的 24 小时方案和单场景估计为历史工程记录，最新跨场景证据、每日入口和未完成准入
以 [每日 README](README_LG_BEV_DAILY_20261027.md) 与 [源码索引](README.md) 为准。
2026-10-28 Git 检查不执行本页 GPU、GT 或长训命令。

本轮实现数据构建、FP32 原生时序缓存、两阶段训练状态机、正式评价器和真实 32 帧验收。完整 28,686 步训练没有启动，开发集和 Cold Final 没有打开。性能证据为 `results/lg_bev_gpu_retained_v3_20261027/`；收尾的计时/恢复修复及最新源码封存位于 `results/lg_bev_engineering_handoff_20261027/`。旧 `lg_bev_gpu_release_20261027` 和历史失败报告保留，不互相覆盖。今日完整记录见 `docs/logbook_2026-10-27.md`。

## 固定边界与缓存补充协议

- 继承封存 `formal_execution_contract.json`，保持 19,123/2,692/3,288 帧日志隔离划分、93,866 参数 Adapter、全部损失权重及 A2+B4 frame epochs。没有更换 checkpoint 或模型结构。
- WSL 注册信息实际为 `Ubuntu-22.04-E`，VHD 位于 `E:\WSL\Ubuntu-22.04`。工作区属于 E 盘；新大型标签、参考 BEV、优化器 checkpoint、SQLite 计分记录、profiler trace 和临时数据统一放 G 盘。
- 唯一根目录 `/mnt/g/BEV_VLA_Datasets/lg_bev_rolling_cache`，Windows 为 `G:\BEV_VLA_Datasets\lg_bev_rolling_cache`。总根目录预算 30,000,000,000 字节，G 空闲保留 250,000,000,000 字节，使用 SI GB。用户显式调小后持久保存，不自动扩大。
- 原生 BEV 为真实 FP32 `[1,256,150,150]`，每帧 23.04MB；8 帧约 184.32MB 一分片。没有先量化再回转。GT 布尔 mask 允许无损 packbits，与 BEV 精度无关。
- 正式顺序固定为 scene token 哈希顺序、场景内时间顺序。每 epoch 全部训练帧出现一次。只有固定、完整、合计不超过 10GB 的热场景落盘，其余场景按需重算并仅保留一个 8 帧 RAM 块，来源/时间/张量 SHA 仍入 G 盘清单。这避免 NTFS 每 epoch 写入数百 GB，不减少任何帧或更新。
- 场景 cache miss 只允许接续正确原生 history，或从该场景真实首帧完整重建前缀。所有分支共享未增强的原生 history；Adapter 输出不写回 history。缓存重建前后恢复训练 RNG。
- 文件锁、SQLite 归属清单、规范化路径和 symlink 拒绝、分片 pin、校验后再生资格约束删除。只淘汰已登记可重算分片；GT、checkpoint、未知文件、原始传感器和旧实验不会自动删除。
- 写入先预留容量，文件 `.pending` 写完并 fsync 后原子重命名；完整中断文件可恢复，坏文件识别为 miss 并保留，再用原生前缀重建。标签异常失败关闭，不静默换帧。

本地 CAN bus 原始 pose 数据未提供。沿用作者转换器的缺失 CAN fallback：先置零 18 维，再由作者 `get_data_info` 填写真实 ego position/quaternion/yaw；32 帧与原 full-info 元数据相同。没有将零加速度冒充实测传感器数据。

## 文件职责

| 文件 | 职责 |
|---|---|
| `lg_bev_engineering_cache.py` | G 盘归属、容量、分片、校验、淘汰、事务恢复 |
| `lg_bev_engineering_data.py` | 原始 3D/map GT、独立新场景、实例/空目标/反事实指令、无损标签块 |
| `lg_bev_engineering_native.py` | 完整六相机原生时序前向、严格 FP32 提取、按前缀重算 |
| `lg_bev_engineering_loss.py` | 差集排序、原生 Hungarian 目标正例权重、原生分割/ROI/KL |
| `lg_bev_engineering_train.py` | A/B 梯度累积、监控、优化器/RNG/部分梯度恢复、信号停止 |
| `lg_bev_engineering_determinism.py` | 原生 forward 不变的确定性反向及固定 GPU 线性伴随 |
| `lg_bev_engineering_dynamic_cuda.py` | 未获采用的动态 CUDA 反向候选，仅保留研究证据 |
| `lg_bev_engineering_full.py` | 全量装配、128 帧训练门槛、24 小时预算、开发终点锁 |
| `lg_bev_engineering_evaluate.py` | 四分支、SDK 口径距离 AP、ROI/global IoU、FP/空目标/分场景 |
| `lg_bev_engineering_metrics.py` | G 盘流式 SQLite AP 与场景 bootstrap，有覆盖索引 |
| `lg_bev_engineering_{run,audit,recovery,fast_verify}.py` | 32 帧选择、真实验收、实际子进程崩溃、原生预测不变量 |
| `lg_bev_gpu_{benchmark,retained}.py` | 实验候选与保留路径的 32 帧数值、GPU/CPU/传输剖析 |
| `tests/test_lg_bev_engineering_20261027.py` | 路径/容量/恢复/计分/梯度/封存准入边界 |

Attention 的标定投影、BCE/Dice、原生双 Head forward、CE 和大部分反向在 CUDA。检测 Hungarian 匹配仍按作者实现使用 CPU。动态 deformable-attention 的 CPU FP32 参考反向保留：其 CUDA 候选未通过原定 Loss/梯度/参数更新容差，正式入口拒绝 `LG_BEV_DYNAMIC_CUDA!=0`。上采样的 CUDA 候选也未过门槛，默认 `LG_BEV_UPSAMPLE_CUDA=0`，保留同一个 CPU ATen backward，去掉无需输入值的复制和 CPU forward。

固定网格 GPU 映射按输入像素预存邻居/系数，再以固定顺序 gather/multiply/add；没有 atomic 或 cuSPARSE 归约。CPU 只准备几何权重，网络 BEV/激活保持 FP32。原生所有 forward、cropper 和 decoder 未替换。每次算子替换必须同时通过参考 Loss、梯度和 AdamW 更新检查，不能仅凭前向或两条新实现路径相等宣布等价。

## 已完成的真实工程验收

固定 metadata-hash 选择的新训练场景 `scene-0050` 前 32 个连续帧，未挑难例或好预测。重新从原始 GT 构建 706 条指令、276 条空目标（39.093%），607 次目标实例出现（跨指令重复，不是 607 个独立物理实例），32 成功/0 失败，标签阶段 19.087 秒。指令显式标记 `GT_DERIVED_TEMPLATE_NOT_DRIVELM`。

32 帧原始六相机前向、严格 FP32 BEV 与完整双 Head 重放逐位一致；另一次完整重建也逐位一致。地图 GT 与作者 raster 逐像素核对；原图像流水线与去除未使用 GT 重建的快速输入流水线，其图像、BEV、cls/bbox/seg 均逐位一致。

强制缩小实例容量至约 1.152GB 后实际淘汰 10 次、提交 12 次，重建后 BEV 最大误差 0。真实子进程在 fsync 后、rename 前以退出码 73 中断；恢复 1 个真实 FP32 分片，逐位一致。坏文件被识别并保留，重新原生重建逐位一致。

全缓存/rolling/RAM 流式 32 帧均实际运行 B 阶段反向及 8 次累积更新。另在第 6 个 microstep 保存 2 个部分累积梯度，恢复后完成剩余 26 帧；参数最大差 1.863e-9。A 阶段也运行真实 attention 更新，并验收 enhanced BEV 严格恒等。正式更新计数仍为 0，所有小测各自隔离初始化。

四分支评价器已在这 32 个训练帧做功能验收；这不是开发性能报告。缺失正例类别标 `NOT_EVALUABLE`，空目标预测保留 FP，匹配边界为严格 `< distance`，没有 GT 预筛预测。全局 10 类指标在小样本缺类时不伪造 mAP。正式开发/最终指标均为 `NOT_RUN`。

## 复现与运行边界

CPU 回归与静态检查：

```bash
cd ~/projects/bev_vla
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv-bevformer/bin/python -m pytest -q
.venv-bevformer/bin/python -m pyflakes scripts/stage3/lg_bev_engineering_*.py scripts/stage3/lg_bev_gpu_*.py tests/test_lg_bev_engineering_20261027.py
```

GPU 32 帧完整工程复现，必须选择全新 out/run-id；不会打开开发/最终集或启动长训：

```bash
cd ~/projects/bev_vla
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LG_BEV_DYNAMIC_CUDA=0
LG_ENGINEERING=results/lg_bev_engineering_repro_01
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_engineering_run --phase prepare --out "$LG_ENGINEERING" --run-id engineering_repro_01
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_engineering_run --phase audit --out "$LG_ENGINEERING"
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_engineering_fast_verify --out "$LG_ENGINEERING"
```

CPU/GPU 固定映射与 pinned 传输吞吐复现，使用已经封存的同一 32 帧原始参考（不冒充一次新建 golden cache）：

```bash
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_gpu_retained --out results/lg_bev_gpu_retained_repro_01
```

下面是已实现的正式入口，**本轮没有执行**。当前若资源或全量标签未通过，会在初始化 GPU 前拒绝；不得删除准入检查绕过阻断。只有后续 READY 且用户单独授权完整训练后才使用：

```bash
cd ~/projects/bev_vla
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export LG_BEV_DYNAMIC_CUDA=0
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_engineering_train \
  --engineering-out results/lg_bev_engineering_handoff_20261027/final \
  --data-report /mnt/g/BEV_VLA_Datasets/lg_bev_rolling_cache/runs/formal_gt_20261027_01/data_report.json \
  --data-run-id formal_gt_20261027_01 --prepare-data-if-needed \
  --run-id formal_train_20261027_01 --cache-gb 30 --authorize-full-training
```

安全停止在当前 microstep 结束后原子保存，包括部分梯度；等待 `control.jsonl` 的 checkpoint 事件和进程退出：

```bash
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_engineering_full \
  --stop-run /mnt/g/BEV_VLA_Datasets/lg_bev_rolling_cache/runs/formal_train_20261027_01
```

恢复使用同一启动参数，加 `--resume` 指向 `control.jsonl` 最后一条已完成且 SHA 核对正确的 checkpoint。不能用旧 pilot 初始化、换协议或重设 RNG。SIGKILL/断电最多恢复到最后一次持久 checkpoint，未持久的计算可能重放；没有声称全量任务具备逐更新 exactly-once 的断电事务。缓存恢复不依赖保留某个可淘汰分片。

正式每 500 updates、每 epoch 末及安全停止保存；终点只选固定 B 最终 checkpoint。NaN/Inf、冻结状态变化、坏 GT、坐标/数值/预算异常立即停止。A128 训练探针不通过便停止，不进入 B；B 固定终点之前开发入口拒绝运行，Cold Final 没有提前开启入口。

Git 人工白名单见最新证据目录 `git_whitelist.json`。不默认提交权重、BEV/标签/原始数据、数据库或大型 trace；所有历史材料保留，未运行任何 Git 写操作。

## 最新吞吐和准入

相同 32 帧、相同 B8 更新，直接跨旧 CPU 参考比较，最终 GPU grid+CPU 直接伴随的 **Loss、Attention、梯度和参数更新最大差全为 0**。原始模型完整输出也保持逐位一致。失败候选不能用“新路径缓存与新路径重建相等”替代跨旧参考验收。

| 路径 | 32帧耗时 | GPU均值 | 峰值allocated |
|---|---:|---:|---:|
| CPU参考，缓存B | 27.141秒 | 7.04% | 1.800GiB |
| CPU参考，六相机流式B | 39.338秒 | 21.15% | 1.824GiB |
| 优化后，缓存B | 16.088秒 | 15.23% | 1.876GiB |
| 优化后，六相机流式B | 30.333秒 | 28.61% | 1.876GiB |
| 优化后，pinned流式B | 29.996秒 | 29.84% | 1.897GiB |
| 优化后，六相机流式A | 18.676秒 | 36.09% | 1.745GiB |

CPU参考至新流式耗时下降22.89%，约1.297倍速度。CPU均值约121.68%（约1.22逻辑核），全部案例RSS峰值约4.204GiB。H2D真实BEV传输2.012ms→0.483ms，pin本身0.988ms；整体收益仅约1.1%，默认不开启 pinned 预取，避免用一次计时噪声做决定。

按A2/B4共114,738帧指令出现、每次均重算原生特征估计26.341小时，加预登记25%场景/I/O/热降频余量为32.926小时。这是wall clock外推，不是GPU忙碌小时，也不是统计置信区间；不含标签生成和终点评价。**超过24小时，因此完整训练BLOCKED**，不减少数据/epochs/更新数放行。

两帧独立CPU/CUDA profiler显示剩余主要瓶颈为动态CPU参考反向及D2H/H2D，不是显存不足。trace在G盘，不纳入32帧吞吐。张量保持FP32，原先matmul/CUDNN TF32开关不变，未启用AMP/FP16。

最终源码封存复核 `lg_bev_gpu_release_20261027` 再运行32帧完整六相机与B前后向，退出0；缓存B15.504秒、流式B29.208秒，终点参数最大差0，32帧original预测更新后逐位不变。该独立进程用于最终源码/路径复核，同进程速度倍率仍使用上表。完整回归208项通过。

目前只允许有界工程复核和后续准入修复。全量19123帧标签尚未构建，跨场景全尺度内存/断电恢复没有实测。保留完整预算启动器与安全恢复接口；当前启动器会明确拒绝24小时资源门槛失败，不把代码存在当作READY。

## 中断后收尾与24小时解决选项

恢复工作先核对旧14个核心源码SHA、结果终止标记和208项测试记录，未从头盘点数据或重跑7组吞吐。进一步修复两个控制缺口：

- 累计活跃wall计时包含 `run` 内日志、检查点、冻结校验和退出开销；checkpoint新增独立原子 `.receipt.pt`，记录实际序列化/fsync/摘要后的累计时间，恢复取主文件与receipt较大值。已有checkpoint不覆盖，停机等待不计入活跃训练时间，续训不能将累计预算清零。超过预算不登记正式完成终点。
- A终点的warmup gate绑定协议、Adapter状态SHA和更新数；同一端点重启复用已完成报告并恢复探针耗时，不覆盖文件或重跑128帧。端点不匹配拒绝；探针异常不产生成功报告；探针内逐帧检查时间限制。

新增4项CPU控制状态机测试，回归 **212 passed**。这些更改没有触及前向、loss或优化器数学。因为完整源码SHA变化，另做一次必要的同32训练帧封存复核；不是重做checkpoint审计、全盘盘点或全部性能实验。最终记录见handoff目录。

该复核退出0，32帧原生图像/BEV/cls/bbox/seg及更新后original预测逐位不变，缓存/流式终点参数最大差0。新run计时cached B32=17.724秒、流式B32=30.682秒，包含收尾冻结校验，与旧microstep计时不混算速度倍率；没有据此宣称进一步提速。

保留25%余量时，点估计需从26.341小时降到 **19.2小时**，还需减少约 **27.11%** 耗时，约 **1.372倍**持续吞吐。如果A不变，B32需从30.333秒降到约 **19.578秒**。当前pinned约1.1%的收益不能解决此缺口。

继续遵守现合同的候选是：有界预取完全相同的原生图像预处理、为动态CPU参考反向提供pinned暂存和合并event等待、复用同一原生前向的FP32 segmentation teacher logits。三者均保持原CPU核/GT/时序顺序，但仍需新代码与数值验收，**本轮未采用或承诺收益**。Teacher必须与GT分开标记，不能借此跳过原始Baseline完整路径验收。

只扩大热缓存命中率不足以解决：10GB约容纳434帧；即使整个30GB都只放BEV（忽略标签/恢复空间，也非可执行方案），也仅1302/19123帧。train全FP32 BEV约440.59GB；不能将“假设全缓存命中”的12.379小时估计用于当前滚动方案。

若后续可靠优化仍达不到24小时，另一选项是用户明确批准新的累计时间协议，覆盖当前约33小时估计并按全尺度预检重估；**现在没有批准或实施**。将6 Epoch拆成多次恢复不降低总耗时，也不能把上限解释成每个进程24小时。当前不减少帧、Epoch、更新数或精度，也不扩容或购买硬件。

后续状态（2026-10-27安全收尾）：上述24小时结论保留为历史。用户先登记48小时/每日7.5小时V2，随后要求累计时间仅作监控预警，已另交V3草案，未静默替换旧协议。当前正式训练仍BLOCKED、正式更新0；GT已正常暂停，5787帧提交/640块核验通过，无残留LG进程。最新操作与未完成项见 `README_LG_BEV_DAILY_20261027.md`、`docs/lg_bev_resource_protocol_v3_draft_2026-10-27.md` 及今日唯一logbook。不能直接使用本页历史长训命令绕过新授权/准入。

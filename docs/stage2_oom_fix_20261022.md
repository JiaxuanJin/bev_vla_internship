# 2026-10-22 WSL 初始化 OOM 修复与单步实测

完成代码优化、静态检查和 **1 次真实 optimizer.step()**，未继续第 2～20 步，未运行验证预测。结果为 `PASS_ONE_STEP_PRESSURE_TEST`。本轮只验证首步及其释放后内存预算，不能据一次更新声称长期无泄漏或准确率提高。

原失败实验仍为 0 次更新。旧 `outputs/stage2_20261022/` 没有协议和训练日志，旧 runtime.log 哈希已复核未变。Git 保持 main，HEAD 为 `9d27b00d28a02b00079891710f0282da9e548661`，没有 commit/push。未扩大 15 GiB RAM / 4 GiB swap，未修改第三方源码、checkpoint 或原始 GT。

## OOM 证据与边界

内核于北京时间 12:17:41 报告 `global_oom`，终止 Python PID 4260，`anon-rss:14888448kB`，约 14.20 GiB。证据保存在 [old_run_evidence.json](../outputs/stage2_oom_fix_20261022_static/old_run_evidence.json)。这是宿主 RAM 耗尽的证据。

作者 `CustomNuScenesDataset` 无论 Mini infos 都硬编码 `NuScenes(version='v1.0-trainval')`，全量表包含约 1.3 GiB sample_data、616 MiB ego_pose、557 MiB sample_annotation；json.load 与反向索引会扩大 Python 常驻内存。原入口先构建训练数据集、DataLoader、iterator 和首帧，再加载 checkpoint；原训练函数又构建验证集，两个独立的全量元数据对象可同时存在。前者抬高 checkpoint 初始化基线，后者会继续放大内存。

**旧日志最后出现 checkpoint 消息，不能证明 OOM 正发生在 load_checkpoint 内部**：后续 model.cuda、DistilBERT 和验证集构建原本没有阶段标记。因此无法追溯旧实验究竟在哪个未记录操作达到最高峰。本次修复针对已确认的元数据膨胀和对象生命周期叠加，并用新日志区分每个阶段。

## 修改文件

- [joint_runtime.py](../scripts/stage2/joint_runtime.py)：micro 路径不创建 DataLoader、iterator 或首帧 batch。协议工作进程直接返回固定协议；训练/验证各自先初始化模型、加载并释放 CPU checkpoint、迁移 CUDA、初始化语言模型，随后仅构建本阶段需要的数据集。joint/joint-backward 保留原有数据加载与前向路径。
- [micro_training.py](../scripts/stage2/micro_training.py)：协议、训练和验证工作函数独立；训练前固定 20 个训练和 6 个验证样本及 token、类别、种子、输入哈希。删除训练前的多帧 probe，训练不持有验证集。仅复制可变 img_metas，图像和 GT 不 deepcopy。保留同一次真实双 Head 前向与三项 loss 计算图，释放 hook 引用及每步 loss/batch，不跨步保留图。
- [low_memory_dataset.py](../scripts/stage2/low_memory_dataset.py)：有界流式读取原始 trainval JSON，仅保留当前 infos 对应的单帧关键帧、完整样本标注及依赖位姿/标定/实例。通过子类覆写表读取方法，仅在 micro 工作进程构建数据集时临时替换作者模块内的 NuScenes 构造器，finally 恢复。地图几何、原始记录字段、GT raster 和数据处理 pipeline 不改。该视图只支持此处单帧训练/诊断，不用于历史遍历或官方全量评估。
- [runtime_memory.py](../scripts/stage2/runtime_memory.py)：flush 的阶段日志，记录 PID、RSS/HWM、WSL MemAvailable/SwapFree、CUDA allocated/reserved/peak/free；清理死对象并释放 glibc 空闲 arena。
- [micro_process.py](../scripts/stage2/micro_process.py)：顺序启动全新协议/训练/验证进程；0.2 秒采样宿主内存，低于 2048 MiB MemAvailable 或超过 11264 MiB worker RSS 时停止并保留日志。默认只做 1 步；每步保存独立权重和 AdamW 状态，恢复必须匹配固定协议与源码/输入哈希及首步通过门槛。
- [run_stage2.py](../scripts/stage2/run_stage2.py)：显式分阶段入口、默认一步、恢复参数和内存阈值，使用新目录和排他文件创建保护旧结果。
- [test_stage2_micro_training.py](../tests/test_stage2_micro_training.py)：增加跨块/Unicode JSON 读取、损坏元数据拒绝、共享图像与隔离元数据、低内存保护、恢复协议哈希和首轮 20 步拒绝测试。

## 验证与实测

py_compile、git diff --check 通过；全部 **88 项 CPU 回归通过**。在独立进程中，以未经修改的官方 NuScenes Mini 加载器作参考，检查全部 **404 个 Mini sample**：保留的 sample、sample_data、sample_annotation、ego_pose、calibrated_sensor、instance 和 Mini 对应 scene/log 均逐字段相等。该进程峰值 RSS 约 774 MiB；详见 [metadata_equality.json](../outputs/stage2_oom_fix_20261022_static/metadata_equality.json)。未加载全量 trainval 对象作对照，以免再次制造 OOM。

单步压力测试运行于 RTX 5080 Laptop GPU，沿用种子 20261021。采样峰值包含初始化、训练和最终完整性核验；0.2 秒采样并结合内核 HWM，不能保证捕捉每个瞬时峰值。

| 指标 | 实测 |
| --- | ---: |
| 协议进程峰值 RSS | 1070.86 MiB |
| 训练进程全程峰值 RSS | 3082.91 MiB（3.01 GiB） |
| 训练期间 WSL 最低 MemAvailable | 10534.79 MiB（10.29 GiB） |
| checkpoint 加载区间 HWM | 2035.66 MiB |
| checkpoint 释放后 RSS | 1296.22 MiB |
| 训练集构建完成 RSS | 1377.64 MiB |
| CUDA peak allocated | 1701.50 MiB |
| CUDA peak reserved | 2140.00 MiB |
| 首步释放后 CUDA allocated | 523.07 MiB |
| 相比 training.ready 的常驻 allocated 增量 | 18.85 MiB |
| 完成 optimizer.step 次数 | 1 |

新记录显示 checkpoint 加载后 CPU 内存释放有效；本轮训练进程最高 RSS 出现在首步之后的核验区间，已不再承受全量元数据常驻压力。watchdog 未触发，SwapFree 约 4012.54 MiB，未出现新的 OOM。

首步 total=4.066397190，detection=3.572092533，segmentation=0.406766653，attention=0.875378728。分别 autograd.grad 检查三项损失，梯度均有限且全局范数非零：检测 0.174369991、分割 0.024449211、Attention 4.544850420。alpha 从 0 更新到 0.000300000；alpha=0 初始化时检测/分割先经 alpha 获得梯度，其余融合参数同时接受 Attention 监督。

8 个参数张量共 **329,729** 个元素真实改变。独立重新加载新增 checkpoint，确认所有 AdamW step=1、参数有限、相对初始权重确有变化、协议/源码哈希匹配。作者模型及 DistilBERT 状态前后哈希一致；受保护文件全部未变。

## 产物与恢复

完整记录：[run.json](../outputs/stage2_oom_fix_20261022_step1/run.json)；固定协议：[protocol.json](../outputs/stage2_oom_fix_20261022_step1/protocol/protocol.json)；单步日志：[training.jsonl](../outputs/stage2_oom_fix_20261022_step1/train/training.jsonl)；[阶段内存](../outputs/stage2_oom_fix_20261022_step1/train/memory.jsonl)；[watchdog 全程采样](../outputs/stage2_oom_fix_20261022_step1/train/watchdog.jsonl)；[进程峰值汇总](../outputs/stage2_oom_fix_20261022_step1/train/process_summary.json)；[首步门槛](../outputs/stage2_oom_fix_20261022_step1/train/step1_gate.json)；[完整性](../outputs/stage2_oom_fix_20261022_step1/train/integrity.json)；[独立权重核验](../outputs/stage2_oom_fix_20261022_step1/verification.json)。

新增权重：[fusion_step001.pt](../outputs/stage2_oom_fix_20261022_step1/train/fusion_step001.pt)，SHA256 `4e8bca6ab68d8e601d0cc8f0987fd752c9c55927a0816f995e63e819737d0b06`。原固定协议总共 20 步，本轮完成其中第 1 步，剩余 **19 步**。以下命令仅记录，**本轮未执行**：

```bash
PY=/home/jjx2003520/projects/bevformer_author_clean_5080/.venv-author/bin/python
"$PY" -u scripts/stage2/run_stage2.py --mode micro-train \
  --micro-phase continue --micro-steps 19 --seed 20261021 \
  --protocol outputs/stage2_oom_fix_20261022_step1/protocol/protocol.json \
  --resume outputs/stage2_oom_fix_20261022_step1/train/fusion_step001.pt \
  --output outputs/stage2_oom_fix_20261022_continue/run.json
```

可将 --micro-steps 改成更小的新增步数来逐阶段恢复。每阶段须使用全新输出目录。继续过程使用同一预先固定的索引/文本、每帧种子及 AdamW 状态，不重新选择验证样本。仅在已保存第 20 步后，再独立执行 `--micro-phase validate --protocol 原协议 --resume 新目录/continue/fusion_step020.pt --output 又一新目录/run.json`，才构建验证集并评估。尚未实测后续 19 步或验证阶段，旧 micro_report.py 仅适配旧平铺 20 步结果布局，不用于本次一步报告。outputs 按仓库既有规则被忽略，交付产物需另行保留。

# LG-BEV 独立探索分支（2026-10-27）

当前交付状态见 [源码总索引](README.md)：正式训练尚未启动、正式更新 0、BLOCKED；
GT 仅完成 5787/19123 帧，640 块校验通过；V3 为 DRAFT_NOT_APPLIED。
本页为早期原型记录，下述 GPU 命令仅供历史复现，2026-10-28 Git 检查不执行。

> **后续更正与最新状态**：约15m诊断使用了未激活的 `BEVFormerHead_seg` 变换；实际模型为 `BEVFormerHead`。后续保留原生 cropper/decoder，独立 Attention 监督空间与原始预测不变量验收通过，已执行唯一16步 pilot；相关 AP 无提升，相关分割 IoU 变化约 -0.00502 个百分点。详见 [监督空间与坐标根因说明](README_LG_BEV_SPACE_20261027.md) 及当日日志文末。下面正文保留首次交付时的历史状态；`results/lg_bev_20261027/` 未改写，不再将其15m数值作为实际模型的坐标错误证据。

已完成真实 GT 监督、冻结原生双 Head 接入和两次 CUDA 更新验收。**P4 为 BLOCKED**：原作者分割 feature cropper 与逐帧 LiDAR↔Ego 标定的物理采样存在冲突，未执行 pilot 或开发集性能对照。完整过程和限制见 `docs/logbook_2026-10-27.md`。

## 环境与路径

- 工作目录示例：`~/projects/bev_vla`，公开复现时替换为实际克隆目录。
- Python：`.venv-bevformer/bin/python`，3.10.12；PyTorch 2.8.0+cu128。
- GPU：NVIDIA GeForce RTX 5080 Laptop GPU。受限沙箱内 GPU 不可访问；真实 CUDA 命令在具备本机 GPU 访问权限的终端运行。
- 原作者代码：`~/projects/bevformer_author_clean_5080/BEVFormer-seg`；不修改该目录。
- 冻结作者 checkpoint：`/mnt/g/nus/bevformer_small_joint_author.pth`。
- 原始 GT：`/mnt/g/BEV_VLA_Datasets/bev_seg_nuscenes`。
- 冻结文本编码器：`models/distilbert-base-uncased`，只读、本地加载，不下载模型。
- 复用缓存：`outputs/stage2_spatial_v2_20261026/cache/bev_*.npz`；地图 GT 来源为 `outputs/stage2_spatial_v3_20261026/annotations/` 及 V4 审计标签。逐文件 SHA 见新 `protocol.json`。

## 文件职责与张量协议

| 文件 | 内容 |
|---|---|
| `lg_bev_prototype.py` | 24 通道语言 query / 空间 key、受限残差、BCE/Dice/干扰监督；保留原模型输入输出，新增可选 valid mask |
| `lg_bev_common.py` | 独立路径、只增写 JSON、split 与来源哈希校验 |
| `lg_bev_data.py` | 原训练场景重划分，真实实例 mask、annotation token 匹配、原始地图重建和三联图 |
| `lg_bev_native.py` | 缓存原生 BEV 进入同一冻结原生双 Head、作者真实损失、逐参数梯度和状态哈希 |
| `lg_bev_train.py` | 两步链路验收、唯一 16 步 pilot 门槛、V5 autograd 只读审计、四分支与同步时延 |
| `lg_bev_evaluate.py` | 无 GT 输入的配对推理、独立内部开发计分、BLOCKED 时输出 NOT_RUN |
| `lg_bev_verify.py` | 只读 checkpoint 再前向/反向，优化器更新数为零 |
| `lg_bev_spatial_review.py` | 从原作者源码提取纯几何 cropper，CPU 重现空间失败及 ramp 图 |

模型 `forward` 只接受 BEV `[B,256,150,150]`、冻结文本 `[B,3072]`、仅由标定生成的位置 `[B,5,150,150]`。输出 `enhanced_bev`、`attention_logits`、`attention_mask`、`residual`。同一 `enhanced_bev` 进入原作者检测 decoder 和分割 decoder；无新增主干，无 GT 输入模型。

对象监督为 `[B,1,150,150]` 的原生 LiDAR footprint。注意本机 MMDetection3D 0.17.1 使用顺时针 yaw，已对照 849 个真实 `corners` 栅格检查。地图监督为原始 `[B,1,200,400]` Ego 栅格；将 native attention 用标定、像素中心和 `align_corners=False` 投影后计算损失，不 resize GT。两类标签均提供 target、distractor、valid 和目标存在标志。空目标只计负例 BCE；非空/空样本分组均值，避免空指令数量主导。

新增原型修改的完整差异在 `results/lg_bev_20261027/prototype_changes.diff`。修改前原文件的逐字节备份为 `source_before/lg_bev_prototype.py`，SHA 与初始快照完全相同。

## 数据边界与预注册

`split_id=lg_bev_20261027_scene_internal_v1`，seed=20261027。按场景从允许的原训练集固定抽出 `scene-0757` 为内部开发。训练 4 场景/32 帧/685 指令，内部开发 1 场景/8 帧/189 指令；帧、场景、指令、地图记录和对象实例无交叉。二者均曾用于历史适配器训练，只能称为训练分布内开发；作者预训练未见泛化未验证。

封存 `scene-0061/0655/0553/0001/0002`；新模型不在这些场景做前向、调参或阈值选择。V5 梯度审计使用允许的单训练帧，更新数为零。

唯一预注册 pilot 为 16 步、batch=1、lr=3e-4、AdamW weight_decay=0.01、clip=5，地图/对象交替，对象非空/空交替，终点 checkpoint 不选择最优。权重先验固定为 detection=1、原生 segmentation=1、mask=0.2、mask 内 distractor=0.5、无关区 KL=0.1。原生 segmentation 继续包含作者配置中的 loss_weight=3 和类别权重 `[0.3,2,2,2]`。

本次空间门槛不通过，因此 **pilot 更新数=0，开发/最终性能评估样本数=0**。两步 smoke 仅验收计算图和优化器，不用于性能结论。评估器及 pilot 的完整运行尚未经过真实数据验收，不能据代码存在宣称性能实验完成。

## 当前证据

- `protocol.json`：隔离协议、硬件、数据/配置/源码/checkpoint SHA。
- `supervision_audit.json`、`supervision_statistics.json`、`data_isolation_audit.json`：真实监督及交叉审计。
- `smoke_test.json`、`grad_audit.json`、`smoke/training.jsonl`：两次真实更新、逐任务逐参数梯度、冻结不变量。
- `verification_backward.json`：额外只读真实反向，命令退出码 0，新增优化器更新为 0。
- `v5_autograd_audit.json`：V5 主任务梯度未断；零初始化首步上游为零，封存 step096 上游非零。
- `paired_smoke.json`：同一训练帧的 original/zero_text/wrong_text/correct_text；每分支真实解码 300 框，不计 AP/IoU。
- `spatial_gate.json`、`spatial_review.json`、`figures/native_vs_calibrated_ramp.png`：原生 cropper 与物理标定平均约 15.0629m、最大约 29.9338m 的采样差异；CPU/CUDA 一致。
- `metrics.json/csv`：四分支 AP/IoU 统一 `NOT_RUN`，不生成占位性能数值。
- `latency.json`：单训练帧 cached-BEV 双 Head 和 adapter 同步耗时；不含文本编码、缓存 I/O 或六相机主干；RTX 3090 门槛未测。
- `history_integrity.json`：5,619 项原有文件内容核验；仅原型有明确记录的修改，原始内容已精确备份。

原始作者 checkpoint SHA256：`1345919901785b8df7ee16838d62f9f7a00444c9d8061f0dbee712d1b1b67dcf`。
本次 smoke checkpoint SHA256：`e9443caeca02940d2483e8efea7cdd0e2b18cc639229158f5e7ae8686fca59bf`。
最终登记 protocol SHA256：`6c9446742496d42857fabf36658307326083cc14c9f1c3c7c612ed5bf081556a`。预检元数据补全前版本保存在 `protocol_preflight_metadata.json`；split、模型、权重和预算均未改变，补全发生在真实 Head 前向/更新之前。

## 复现

CPU 单测与原型：

```bash
cd ~/projects/bev_vla
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-bevformer/bin/python -m pytest -q
.venv-bevformer/bin/python scripts/stage3/lg_bev_prototype.py --self-test
```

在具备 GPU 权限的终端使用新目录复现；已有目录/文件会拒绝覆盖。准备数据需要读取本地元数据，约数十秒，不需要大型下载：

```bash
cd ~/projects/bev_vla
set -e
LG_RUN=results/lg_bev_20261027/repro_01
mkdir "$LG_RUN"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/lg_bev_matplotlib
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_data --out "$LG_RUN"
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_train --out "$LG_RUN" --phase smoke
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_verify --out "$LG_RUN"
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_spatial_review --out "$LG_RUN"
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_evaluate --out "$LG_RUN" --blocked-report
```

当前不要启动 `--phase pilot` 或正式性能计分：入口会根据 `spatial_gate.json` 在初始化 GPU 模型和新建 pilot 目录前拒绝执行。代码不允许自动扩训、选择阈值、自动 commit/push 或覆盖旧结果。

## 下一步的最小修复范围

只在允许训练场景核查作者传感器坐标、map raster 轴向及 learned decoder 是否补偿了旧 cropper。当前证据是 feature 采样物理差异，不能直接等同最终语义错误。需要在保持旧 baseline 及封存集不变的前提下定义并验证独立坐标桥接协议；不得用开发 GT 拟合平移、旋转或阈值。该问题解决前，pilot 和任何“精度提升”结论维持 BLOCKED。

Git 人工检查白名单位于 `results/lg_bev_20261027/git_whitelist.json`。权重、标签 NPZ、BEV 缓存、原始传感器和含图像的本地可视化不默认纳入 Git；本次未暂存、提交或推送。

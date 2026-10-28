# LG-BEV 正式训练方案与本地资产审计（2026-10-27）

公开交付阅读说明：本文保留初次方案及当时的资源/实现状态。
截至昨日最终收尾，正式训练尚未启动、正式更新 0、BLOCKED；GT 仅完成 5787/19123 帧，
640 块通过校验；V3 仍为 DRAFT_NOT_APPLIED。后续已实现的工程入口和 V2 运行边界见
[源码索引](../scripts/stage3/README.md) 与 [每日 README](../scripts/stage3/README_LG_BEV_DAILY_20261027.md)。
正文中“全量 FP32 缓存放工作区”等早期估计已被后续 G 盘 30GB rolling cache 补充协议替代；
不将旧设计重写为已完成实验。2026-10-28 仅做 Git 检查，不执行本页资产审计或训练。

本轮实际完成 checkpoint 反序列化、G 盘全量元数据/文件覆盖盘点、原始图像抽样解码、历史训练指令 CPU 只读重放、独立驾驶日志划分及预算登记。**正式训练未启动，新增优化器更新为 0，正式性能指标均为 NOT_RUN。** 本文是一份待实现准入项的固定训练设计，不把方案或脚本存在视为训练完成。

证据目录：`results/lg_bev_formal_training_audit_20261027/`。核心登记是 `formal_training_protocol.json` 和随后补齐严格 FP32 基线、损失及计分语义的 `formal_execution_contract.json`；后者对同名事项优先。初次 FP16 资源估计保留为草估，没有覆盖；正式主实验选择无损 FP32 缓存。没有据新开发/最终集结果修改设计。

## 1. joint checkpoint 的真实身份

| 项目 | 本地 joint | 官方检测 checkpoint |
|---|---|---|
| 路径 | `/mnt/g/nus/bevformer_small_joint_author.pth` | `BEVFormer/ckpts/bevformer_small_epoch_24.pth` |
| 保存状态 | epoch 19，iter 44,555，2023-01-19 | epoch 24，iter 84,408，2022-06-21 |
| 实验名 | `bevformer_small_seg_det.py` | `bevformer_small.py` |
| state_dict 键 | 1,111 | 1,001 |
| 原生分割 decoder | 110 个键，5,205,782 个状态元素，含 buffer | 无 |
| 发布源 | Bin-ze/BEVFormer_segmentation_detection | fundamentalvision/BEVFormer |

joint 完整 SHA256：`1345919901785b8df7ee16838d62f9f7a00444c9d8061f0dbee712d1b1b67dcf`，文件 774,591,208 字节。官方检测文件 SHA256：`4ebc5810201ca1e609c29452f07c08bc4d9c9ebfda5d7f8ee1034dd446274bc5`。

两个权重的共同 1,001 个键形状完全一致，但仅部分张量逐位一致（backbone 531 个、其他 1 个），不能把 joint 当作“官方检测权重外接一个未训练分割层”。joint 的全部状态元素合计 64,774,849，包含 BN 等 buffer，不等于可训练参数数目。

joint 内嵌原训练配置：R101-DCN、256 通道、150×150 BEV、900 queries、3 层 encoder/6 层 decoder、检测 10 类、分割背景加 3 类；queue_length=3，video_test_mode=True，samples_per_gpu=2；AdamW lr=2e-4、weight_decay=.01、backbone lr_mult=.1，计划 24 epochs，FCOS3D R101 初始化。分割 CE 的 loss_weight=3、class_weight=[.3,2,2,2]。原环境记录 Python3.8.15、Torch1.9.1+cu111 和 7 张 RTX3090；这不能单独证明实际 world_size 或样本清单。当前本地配置的 lr=1e-4、batch=1 不是原训练配置。

上游 [joint 发布说明](https://github.com/Bin-ze/BEVFormer_segmentation_detection/blob/master/README.md) 的 epoch19/150×150 和 Google Drive 文件 ID 与本地 `outputs/stage2_step70/step70b_weight_ready.json` 下载记录一致，下载记录的 SHA 与实物一致。上游未提供可核验发布 SHA，不能声称完成发布者密码学认证。[官方 BEVFormer 说明](https://github.com/fundamentalvision/BEVFormer/blob/master/README.md) 的 Small 检测结果与第三方 joint 发布结果分别登记，均不是本轮重新测得的指标。

**预训练范围的边界：**内嵌配置仅引用标准 nuScenes train/val PKL 文件名，没有实际 frame/scene manifest。不能从 44,555/19 次迭代反推出精确样本数量，也不能用目前本机 mini PKL 猜测原作者只训练了 mini。官方 train 场景均按“可能被 joint 预训练见过”登记。已验证“作者预训练未见”的独立帧数为 **0**。新方案验证的是语言适配的场景/日志隔离，不能声称 foundation-unseen 泛化。

证据：`checkpoint_audit.json`、`checkpoint_embedded_config.txt`、`head_adaptation_feasibility.json`、`upstream_sources.json`。

## 2. G 盘真实数据量及可用层级

| 本地资产 | 真实去重帧数 | 场景数 | 本轮实际核查 |
|---|---:|---:|---|
| nuScenes 官方 train | 28,130 | 700 | 逐帧关联六相机、LiDAR、标定/位姿和地图；没有缺文件 |
| nuScenes 官方 val | 6,019 | 150 | 仅元数据、文件覆盖和历史暴露登记；不做模型计分 |
| DriveLM v1.1 train | 4,072 | 696 | 解析 377,956 QA、377,955 条有答案；全部 sample token 属于上述 nuScenes |
| DriveLM val q-only | 799 | 149 | 15,480 问题，答案为 0；不是额外独立验证帧 |
| BEV Segmentation 数据 | 与 nuScenes 同源 | 同源 | `samples`、trainval 元数据均解析到同一实体路径，地图 4 地点 JSON 可读 |

全量 34,149 帧的 6 个 camera 目录和 LiDAR 目录各有 34,149 个对应文件。不是把 tgz/zip 名称或图片数量除以 6 猜测帧数。所需位姿与标定引用覆盖完整，四元数/平移数值有效。抽取 3 个新候选训练场景各 6 张 JPEG，**18 张完整解码并登记 SHA**；没有声称所有传感器文件都完整解码。本轮不运行六相机网络；此前真实六相机/缓存重放证据仍在旧 RCA 目录。

原始 `sample_annotation.json` 与既有官方元数据 tgz 中对应成员的 SHA 一致。官方 train 有 6 帧、val 有 3 帧没有 annotation 记录，应保留为真实空 GT，而非丢弃负样本。因此初始仅统计非空 annotation 的 27,802 帧，在 `label_coverage_audit.json` 中明确更正为 **27,808 帧/692 场景**；原盘点文件保留，不覆写。

历史隔离排除所有 mini 场景及 scene-0001/0002 共 12 场景；其中 scene-0796 在 SDK **官方 val**，mini 的 train/val 分法不能代替官方划分。再排除这些历史场景所在的整段驾驶日志，得到 **25,103 帧/625 场景/45 个日志**。新划分不使用官方 val；历史 holdout 及旧 scene-0757 开发集均不重用。

DriveLM 17,238 个 key object 只有类别、状态、描述和 2D 框等字段，没有显式 annotation/instance token。不能把 377,956 QA 当成独立 3D 指令监督，更不能把 DriveLM/BEV Seg 帧数与 nuScenes 相加。[DriveLM 数据说明](https://github.com/OpenDriveLab/DriveLM/blob/main/docs/data_prep_nus.md) 同样以 nuScenes 为基础。本次主实验先用真实 3D/map-record GT 生成可验证的空间指令；DriveLM 仅在新训练 split 内作语言分析，未经独立 3D 关联验收的 QA 不进入损失。

三层可用性必须区分：

1. **原始资产齐全的新增适配候选：27,808 帧。**这是文件和元数据层面，不等于已生成指令监督。
2. **整日志历史隔离后的正式候选：25,103 帧。**正式训练部分为 19,123 帧。
3. **新正式 split 已验收的指令标签/时序缓存：尚未生成。**已有 40 帧的旧 LG 标签可以复用算法，不能复用为新场景数据；正式训练准入仍为 false。

已有 `outputs/bevformer_full_infos/` 下 train/val 大 PKL 可作索引核对，但作者目录当前软链接的 `nuscenes_infos_temporal_train/val.pkl` 实际为 mini 348/81 帧。现有 `load_native()`/16 步 runner 绑定 mini dataset，**不得直接给它换一份长 schedule 后冒充 full-data 训练器**。

证据：`data_inventory_summary.json`、`frame_inventory_verified.json`、`label_coverage_audit.json`、`formal_split_manifest.json`。大的逐帧清单默认留本地，不默认加入 Git。

## 3. 16 步 Pilot：确认的事实与待证假设

只读取固定终点 checkpoint，在其原来的 **16 条训练指令** CPU 重放；不在旧开发集调阈值，不做任何优化器更新。本轮没有新性能评估。

| 观测 | 实际值 | 可以得出的判断 |
|---|---:|---|
| 训练覆盖 | 13 独立帧，16 独立指令，12 正/4 空 | 仅为旧 685 指令的 .02336 个指令 epoch，不能判断正式收敛 |
| 全图 residual/BEV L2 均值 | .0002992 | 全局变化小 |
| 真实目标区 residual/BEV L2 均值 | .0002750 | 不是仅由目标稀疏稀释全局比值，目标区变化也小 |
| attention 门控后允许容量使用比例 | 2.64% | 尚未打满 5% 相对幅度限制 |
| raw-up RMS / tanh 饱和比例 | .02509 / 0 | 没有证据支持直接增大残差上限 |
| correct/wrong 冻结文本 cosine | .96345 | 指令表述很相似 |
| correct/wrong 学得 query cosine | .99907 | 当前 query 区分度低，但不是证明完全无语言梯度 |
| 目标上 correct−wrong logit 均值 | .02227，部分为负 | 没有稳定反事实区分 |
| 加权梯度 L2 均值：det/seg/mask | .03434 / .01128 / .61699 | mask 梯度已明显更大；不能简单归因于 mask 权重太小 |

旧内部开发终点已报告 correct map/object Attention IoU=.005057/.015547；相关 AP@2m_xy 无改善，相关分割 pooled IoU 微降。它们仅作既有失败事实引用，不用来选择本次新开发阈值。目标概率多在 .2–.46，部分目标/同类干扰 logit 倒置；全图梯度范数不是每项优化方向或因果贡献。

结构上，语言调制空间 gate，而残差内容上投影来自视觉/位置；原生全局 det/seg 损失大体与指令无关。当前训练没有 attention 预热或显式 correct/wrong 排序约束。零 up 使首步 native task 仅到 up，显式 mask 能训练语言；这不是断路。正式方案针对覆盖、grounding 和任务相关性逐项建立约束，**不把 16 步失败当作必须解冻 Head 的证据**。

证据：`pilot_train_probe.json`、`pilot_rca.json`；原终点及旧指标完全保留。

## 4. 固定 split 与模型不变量

seed=20261027。以完整 `log_token` 为组，在地点内按 token SHA 排序分配约 80/10/10；不根据 GT 类别分布、原模型难例或精度重划分。

| split | 独立帧 | 场景 | 驾驶日志 | 用途 |
|---|---:|---:|---:|---|
| train | 19,123 | 476 | 35 | 标签生成、训练及固定训练探针 |
| development | 2,692 | 67 | 5 | 仅固定终点一次评价 |
| cold_final | 3,288 | 82 | 5 | 现仅登记元数据；开发门槛通过并封存代码/模型后才打开一次 |

frame、scene、drive log 均不交叉；不声称跨日志一定没有道路/地图区域重叠。六个真实空 GT 均在 train 保留。任何标签关联失败先报告；不能静默删除帧、改 split 或改变预算。

冻结 joint 主干、BEV encoder、native detection/segmentation Head、cropper/decoder、BN running state 和本地 DistilBERT readout。仅训练现有 24 通道 `LanguageGroundedBEVAdapter`，**93,866 参数**，relative_bound=.05；全新 seed 初始化，不继续旧 pilot checkpoint。

Attention/BEV 为 LiDAR150×150；对象监督来自真实 annotation footprint；map attention logits 仅在损失中经 R/t、像素中心、align_corners=False 投影至 Ego200×400。原生分割保持已经学习的完整采样路径，不要求 pre-decoder 每像素等于最终 GT 坐标，不替换 cropper。GT 不进入模型 forward 的文本、BEV 或 position。

采用作者确定性 test preprocessing，无 PhotoMetricDistortion；按场景时间顺序构建原始 temporal history，逐场景 reset，所有分支共享 **未增强原生 BEV history**，enhanced BEV 不写入 history。先在新训练场景连续 ≥3 帧验证作者完整六相机输出与同一 **FP32** BEV 的完整双 Head 重放 cls/bbox/seg 逐位相同，再建缓存。若做不到，停止并先定位接口，不降低准入标准。旧单帧增强缓存不充当正式时序缓存。

## 5. 两阶段正式训练预算及损失

microbatch=1 帧，gradient accumulation=4；每个 frame epoch 每个训练帧出现一次，每帧固定选择一条真实 GT 指令及其反事实配对。末尾不足 4 的累积按实际 microbatch 数归一化。每 epoch **4,781 次更新**，不把语言提示数量算成新增帧。

### A：2 个完整 frame epochs 的语言定位预热

- 9,562 次 optimizer updates，38,246 次帧指令出现；不跑 native Head 反向。
- 将 adapter up 权重/偏置冻结为零，原生 BEV 严格恒等；其余新增模块训练。AdamW lr=3e-4、weight_decay=.01、clip=5。
- `L_A = balanced_BCE + positive_Dice + .5*distractor_negative + .1*paired_ranking`。valid/空目标处理沿用已审计逻辑：空目标只计负 BCE，无正 Dice；非空/空样本分组均值。显式 distractor 项是额外同类困难负例权重。
- 对配对文本 c/w 分别独立生成真实目标集 S_c/S_w；仅在差集 S_c\S_w 和 S_w\S_c 上对称使用 `relu(.5 - correct_logit + paired_logit)`。共享目标像素不参与排序，空差集跳过；不能把所有 wrong_text 前景都标负。
- 固定 token-hash 选取 128 个 train 帧作终点探针，地图/对象投影空间分别统计，正目标 macro Attention IoU@.5 ≥.15、目标 correct−wrong 平均 logit ≥.1、空指令激活比例 ≤.05。门槛为训练可用性假设，不是从旧开发集调出的最优值；**不通过就停止，不扩预算、不解冻 Head**。

### B：4 个完整 frame epochs 的冻结双 Head 联合训练

- 19,124 次 optimizer updates，76,492 次帧指令出现；与 A 合计 **28,686 次更新、114,738 次帧指令出现**。
- 从预热终点开始，解冻的只是新增 adapter up。新建 AdamW，lr=3e-4 cosine 到3e-5，weight_decay=.01，clip=5；不基于 dev 改 schedule。
- `L_B = L_det + L_seg_native + .2*L_attention + .1*L_pair + .5*L_map_roi + .1*KL_seg_outside`。原生分割 loss 内部仍含作者 loss_weight=3 及原类别权重；不能重复除掉这层权重。
- 对象任务：保留作者 Hungarian 匹配、类别/范围过滤和所有 GT；仅在训练损失聚合中将真实目标 annotation 对应的 matched positive cls/bbox 权重乘3，其他正例和负例权重1。需要新增可验证 loss wrapper，**不修改作者 Head forward 或源码**。地图指令不伪造汽车/行人目标。
- 地图任务：保留原生全局分割 CE，再对选中真实 map-record 的固定2m邻域加 CE，同一完整语义 GT；同类别的其他实例仍是正语义，不被改为背景。对象指令保留全局分割，不附加虚假的地图目标。
- KL 使用同帧原始冻结模型的分割分布，只作无关区保持项，不把预测当作真实实例监督。teacher 分布与 GT 分开保存，模型 forward 不读 GT。全局真实检测损失及终点全类指标保护整体检测。
- 帧级 schedule 固定交替任务，指令层在可生成时保留25%空目标；实际空帧不得删除。反事实优先同类同方位 nearest/farthest；不具备可区分目标集时，用独立 GT 验证的左右交换，否则记录跳过原因。配对表在训练前固定。

每步记录 scalar loss、梯度/更新范数、残差全图/目标区能量与空目标行为；每200步及各阶段端点记录逐任务、逐模块梯度。首个 joint 步的零 up 上游 task 梯度为零单独说明，其后必须验证 task 能到语言层。发现 NaN/Inf、冻结状态变化、GT 泄漏、坐标门槛失败或真实反向路径异常，立即停止并保存证据。

保存 A 终点、B 每个 frame epoch 和最终终点，包含 optimizer/scheduler/RNG/累计更新数/协议 SHA。**主结果只使用固定最终终点**；不挑最好 checkpoint。仅允许同协议同 RNG 的崩溃恢复，历史更新仍计入预算。训练时间上限24小时，达到上限保存为 INCOMPLETE，不把提前中断当作完成训练。

## 6. 评价口径、对照及停止条件

四个分支 `original / correct_text / wrong_text / zero_text` 使用相同帧、标定、GT、原始 temporal history、FP32 cache 和解码配置。original 完全绕过 adapter；另外三者是同一已训练 checkpoint 的推理消融，不冒称独立等预算训练对照。wrong/zero 始终按 **正确指令的 GT** 评价，不能换成错误指令 GT。

1. **相关检测主指标**：汽车/行人两个有真实指令支持的类别，中心距离 .5/1/2/4m 的 instruction-conditional mAP。每条 `(frame_token, instruction_id)` 是虚拟 query sample，按冻结表的 all/nearest/farthest 真 annotation ID 计 GT；只按请求类别过滤预测，不按 GT 距离预筛选预测，同类干扰预测计 FP。空目标指令保留预测 FP。沿用 SDK recall grid、min_recall=.1/min_precision=.1、class range/zero-point GT 规则；每类每距离 AP 等权平均。缺少正例类别标 NOT_EVALUABLE，不以0或删类掩盖。
2. **相关分割主指标**：lane marking/crosswalk 两类，目标 record raster 按像素中心欧氏距离扩张2m得到 ROI；使用完整 GT 在 ROI 内累计每类 intersection/union，最后两类 macro IoU。目标外真实同类仍是真同类，空地图指令另报负例指标。旧 pooled 单类 IoU 只作次指标。
3. **整体性能**：每帧按 token hash 固定一条 canonical 指令，检测全10类/4距离及分割全3前景类 macro IoU，不做相关 ROI 过滤；四分支一并报告。另报无关区变化、同类干扰/背景 FP、空目标、Attention IoU、逐场景结果和原始计数。
4. 固定 threshold=.5，不扫开发阈值。1000次固定 seed 场景 bootstrap，只在场景层重采样，不将同帧多条指令冒充独立样本。

[nuScenes 官方检测口径](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md) 与指令条件指标的 GT 集不同。本方案的相关指标必须命名为 `instruction_conditional_2class_mAP_4distance`，**不能叫官方10类全val mAP**。本项目整体指标也是新隔离子集上的 SDK 口径，当前限制下不重跑官方全部 val。不能用旧 AP@2m_xy 或相似名字宣称达到官方项目指标；最终项目验收需统一数据范围和口径。

目标按保守的**绝对百分点**登记：相关 conditional mAP +2.0pp、相关 macro IoU +1.5pp，整体 mAP/mIoU 下降均≤2.0pp；同时报告相对百分比。correct 必须在两项相关主指标上优于 wrong 和 zero。开发只在固定 B 终点跑一次；未达标就封存、不扩训、不据此自动再选结构/损失/lr，cold_final 不打开。通过后再封存模型、代码、阈值，cold_final 仅一次评价。结果可以失败，不承诺提升。

额外延迟≤20ms的 RTX3090 要求仍待真实对应硬件及六相机端到端测量；现有5080 cached-BEV数据不能代替。

## 7. 资源估计与实际限制

现机为 RTX5080 Laptop16GB、主机内存约15.3GiB。所有数字区分已有测量和估算：

| 项目 | 数量/预估 | 依据与限制 |
|---|---:|---|
| 严格 FP32 原生 BEV /帧 | 23.04MB | 256×150×150×4，无量化 |
| train+dev FP32 cache | 502.6176GB | 21,815帧；本轮未生成 |
| 加 cold_final 后 BEV 总量 | 578.37312GB | 最终门槛前不生成 cold cache |
| train+dev FP16 seg teacher | 13.9616GB | 仅保护损失，不作为原始预测一致性的证据 |
| 推荐可用空间 | 至少650GB；含final时750GB | 另留真实标签、框、日志及临时文件空间；禁止静默删旧结果腾空间 |
| 原生时序 cache 构建 | 粗估约6.2 GPU小时 | 旧单帧真实六相机前向约1.028秒；时序/磁盘吞吐未测，不是承诺 |
| 联合训练 B | 粗估6.4–17 GPU小时 | 76,492 microsteps，假设 .3–.8秒/步；新配对/损失实际须预检 |
| 预热 A | 未实测 | 仅原始cache读取在200MB/s下理论下限约1.22小时，尚不含计算 |
| 旧16步 joint 峰值allocated | 约1688.6MiB | 不是正式时序六相机预检/新损失的显存保证 |

盘点时 G 剩余约369GB，工作区文件系统约970GB；正式缓存放工作区，避免占满 G 及 NTFS 小文件开销。全量map/实例标签生成时间未知，先做固定32个新训练帧吞吐/失败率预检，再决定是否具备后续运行条件；不能伪报全量标签完成。

num_workers从0开始，上限2，流式读大JSON，不在15GiB内存同时加载多个完整 NuScenes 实例/大PKL和全部图像。真实连续3帧原生预检的 allocated 上限12GiB；超出先定位，不自动降低分辨率/替换模型。cache主协议FP32；adapter/native loss保留FP32，不能盲目对旧custom CUDA ops切AMP。标签、文本、teacher可另做无损/独立精度审计后缓存。

## 8. Head LoRA / 有限微调：仅论证

当前不实施，也不把它作为 adapter 失败后的自动备用分支。[LoRA 原论文](https://arxiv.org/abs/2106.09685) 给出冻结基座加低秩增量的方法，但不保证本项目视觉 Head 有收益。

真实权重结构中，6层 detection decoder cross-attention 的 value_proj/output_proj 共12个256×256矩阵，若单独建立后续假设，r=4 增加24,576参数，r=8增加49,152参数。原始矩阵冻结；未来必须保留可切回原始参数的 baseline，且不能影响 original 分支。分割最后1×1卷积只有4×128=512权重（加4偏置）；r=4 LoRA反而需528权重，因此最后层的有限微调可能比这处 LoRA 更简洁，但会有类先验/全局性能漂移风险。

应先证明：训练 attention 真正区分目标/干扰和正确/错误语言；局部 residual 不再处于近零学习状态；冻结 Head 的真实 task 梯度敏感性、目标损失下降及整体保护均可靠。如果 grounding 好而原生目标任务仍无响应，才有理由提出独立 Head 适配研究。需要新协议和合法未反复使用的评价数据；不能从本次 dev 再开参数搜索。所列参数量不含新的优化器/激活开销，不是假装已经测得的显存需求。

## 9. 已执行验证与可复现命令

本轮新增5个 `lg_bev_formal_*.py` 审计工具、13项单测及本文；完整回归 **182 passed**，Black88/pyflakes通过。真实G盘盘点、两个checkpoint完整读取/反序列化、18张图像解码和16条历史训练指令CPU重放成功。新增GPU训练/正式评价次数为0。作者代码/原始权重、V1–V5、旧LG结果保留，今日记录仅追加到 `docs/logbook_2026-10-27.md`。

最小只读复核（不生成新结果、不训练）：

```bash
cd ~/projects/bev_vla
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
  .venv-bevformer/bin/python -m pytest -q
.venv-bevformer/bin/python -m pyflakes scripts/stage3/lg_bev_formal_*.py tests/test_lg_bev_formal_audit_20261027.py
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_formal_protocol --verify-only
```

在新目录重做本轮完整 CPU 审计；工具拒绝覆盖已有结果，不下载模型，不运行优化器：

```bash
cd ~/projects/bev_vla
set -e
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
LG_FORMAL_AUDIT=results/lg_bev_formal_training_audit_repro_01
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_formal_inventory --out "$LG_FORMAL_AUDIT"
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_formal_coverage --out "$LG_FORMAL_AUDIT"
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_formal_checkpoint --out "$LG_FORMAL_AUDIT"
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_formal_pilot_analysis --out "$LG_FORMAL_AUDIT"
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_formal_protocol --out "$LG_FORMAL_AUDIT"
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_formal_protocol --out "$LG_FORMAL_AUDIT" --seal-contract
.venv-bevformer/bin/python -m scripts.stage3.lg_bev_formal_protocol --out "$LG_FORMAL_AUDIT" --verify-only
```

**当前没有可诚实给出的正式长训命令。** full-split指令/实例标签清单、连续时序原生cache builder、冻结全量联合trainer、4距离conditional mAP scorer 尚未实现/验收，均已在 `preflight_status.json` 显式阻断。后续实施应先完成这些接口、32帧数据预检与连续3帧不变量/真实反向，冻结实际命令及代码SHA，再单独安排上述预算的运行；不能调用旧 `--phase pilot` 把16步实验重复成正式训练。

Git仅供人工检查白名单见证据目录 `git_whitelist.json`；逐帧大清单、数据、缓存、checkpoint默认不入Git。没有运行git add/commit/push/stash/clean/reset。

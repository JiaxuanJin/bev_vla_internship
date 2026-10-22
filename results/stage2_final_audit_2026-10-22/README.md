# 2026-10-22 第二阶段源码审查交付证据

唯一强制验收依据：[指导老师原始任务书](../../docs/stage2/02_第二阶段原始任务书.md)。先给出严格源码结论再补齐缺口；最终报告：[21_第二阶段最终缺口审查](../../docs/stage2/21_第二阶段最终缺口审查_2026-10-22.md)。8项任务/4项产出的PASS只限实现、功能和可运行性，不代表语义定位或精度提升。

本轮新训练更新 **0次**。原20步实验（TV=0）与新增TV=0.001 smoke/冻结旧源码只读验证重放严格分开。40份旧源码编辑前已逐一匹配原协议哈希，保存于 `legacy_source/scripts/stage2/`；当前代码不可冒充旧20步版本。

| 文件 | 含义 |
|---|---|
| baseline_pytest.json / baseline_pytest.log | 修复前88项CPU基线，exit=0 |
| checks.json、pytest.log、static_compile.log、diff_check.log、loss_smoke.log | 修复后117项CPU、48文件编译、白空格检查、6项损失自检，均exit=0 |
| raw_artifact_audit.json | 独立复算20行训练JSONL、20份AdamW/checkpoint、6帧NPZ及官方split/哈希/混淆矩阵/XY检测 |
| language_probe.json | 真实预训练参数逐张量匹配；目标/方向/否定/词序token、embedding和Attention响应，仅功能探测 |
| gpu_smoke.json / gpu_smoke.log / gpu_smoke_process.json / gpu_smoke_command.json | 当前源码、旧fusion权重、新独立协议TV=0.001；真实原始/旁路/增强双Head和四来源梯度，0次更新 |
| gpu_smoke_protocol.json | 新smoke协议和当时完整源码哈希，独立于原20步协议 |
| frozen_hashes.log / legacy_source_manifest.json | 原40份源码与旧第20步融合权重哈希匹配 |
| legacy_replay_summary.json / legacy_replay.json / legacy_replay.log | 真正导入冻结源码，只读重放原6帧；所有数组及四分支指标精确一致；0次更新 |
| before_source.json / before_git_status.txt | 修改前文件哈希和未提交状态 |
| protected_experiments.json / preservation.json | 全部原实验文件前后哈希，原任务书不变，main/HEAD不变 |
| source_locations.json | 当前类、函数和关键接口的精确源码行号 |
| changes.json / after_git_status.txt | 与本次开始时相比的全部文件变更，区分已有修改和本轮新增 |
| delivery_manifest.json | 交付文件路径、SHA256、外部依赖与排除项 |
| delivery_bundle.json | 实际交付包路径、SHA256和大小，生成于打包后；不自包含进同一包以避免循环哈希 |

实际结果：Original/Trained/Shuffled foreground mIoU=0.3159926671/0.3159515546/0.3159652746；检测均207/599/35。没有稳定精度提升、真实语义定位或完整BEV/GT物理配准证据，不能与Step54/65等异checkpoint或协议直接比。

交付包为 `outputs/stage2_delivery_20261022/stage2_delivery_20261022.tar.gz`。包含本项目源码、测试、中文文档/图、当前审查小型证据、冻结旧源码、原微型实验协议/日志/全部融合checkpoint/六帧NPZ和真实BEV缓存；解压为相同相对路径。本包不包含Python虚拟环境、第三方BEVFormer源码、作者大checkpoint、DistilBERT预训练文件、nuScenes原始图像/GT/地图元数据，这些是明确外部依赖。原固定协议还绑定原机绝对输入路径；换机器的历史完整性复核需准备对应资产，不能静默改协议。所有实际文件以delivery_manifest为准。

当前源码未commit/push，所有outputs按既有规则被Git忽略；向师傅交付时请同时保留交付包或其中的原始产物，不能只发Git源码后声称权重/数组已随代码交付。复现命令见 [17_复现指引](../../docs/stage2/17_第二阶段复现指引.md)。

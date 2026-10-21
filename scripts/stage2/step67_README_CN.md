# Step67：续训 + 训练集校准 + 锁定验证（一键三阶段）

## 本次设计借鉴

- BLV（CVPR 2023）：长尾语义分割的类别/Logit 优化启发，https://github.com/grantword8/BLV
- Tversky loss：在分割任务中平衡 FP 与 FN，https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/losses/tversky.py
- LAVT（CVPR 2022）：文本区域对齐需要真实图文目标配对，https://github.com/yz93/LAVT-RIS

这些是**设计启发，不是原论文代码直接复现**。

## 输入

需要已成功完成 Step65、Step66，且存在：

- `scripts/stage2/run_step65_literature_adapter.py`
- `scripts/stage2/step66_background_safe_train.py`
- `outputs/stage2_step65/step65_ab_summary.json`、`adapter_step65.pt`、`bias_step65.pt`
- `outputs/stage2_step66/step66_ab_summary.json`、`adapter_step66.pt`
- 之前已经可运行的 Step59/60 脚本和作者模型环境。

## 安装、检查与一次运行

仓库已包含脚本，在 VS Code WSL 终端从项目根目录运行：

```bash
cd ~/projects/bev_vla
source ~/projects/bevformer_author_clean_5080/.venv-author/bin/activate
python3 scripts/stage2/run_step67_three_in_one.py --self-test
python3 -u scripts/stage2/run_step67_three_in_one.py --steps 64 --calibration-frames 8 --fresh-val-frames 12
```

脚本自动执行：① 32 帧 Mini Train 续训，保留中途权重；② 使用不同的 8 帧 Mini Train 校准，用固定策略选残差大小和背景门控；③ **先保存校准锁定记录，后加载 Mini Val** 进行 6+12+12+12 帧 A/B 验证。

校准条件：前景 mIoU 不低于原始、四类 mIoU 高于原始、背景->前景误检率相对原始增加不超过 5 个百分点。若所有候选均不符合，自动选 `original`；不为了“看见提升”而调 Mini Val 参数。

## 主要产物

`outputs/stage2_step67/`：

- `step67_method.json`：候选空间及先验选择规则
- `step67_train_calib_split.json`：Train 和校准的互斥索引/Token
- `step67_interim_*.pt`、`step67_train_history.json`：中途权重及训练日志
- `adapter_step67.pt`：最终续训权重
- `step67_selection_lock.json`：**在 Mini Val 评估前**已锁定的方案
- `step67_val_manifest.json`：所有验证 Token 分组
- `step67_ab_summary.json`、`step67_review.md`：四类 IoU / 前景 IoU / 背景误检及语言消融对照

## 重要限制

- 不改作者 checkpoint，BEVFormer 和分割 Decoder 保持冻结，时序 OFF。
- 即使正常文本胜过乱序文本，也只构成初步证据，不能宣称开放词表语言定位能力。
- 这些数据仍全部来自 nuScenes v1.0-mini，不代表作者论文官方 Val 成绩。
- 若输出目录已有文件，程序会拒绝覆盖；请保留记录，不能无差别重新选验证集反复试。

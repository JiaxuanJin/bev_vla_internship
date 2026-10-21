# Step73：补齐第二阶段两项基础功能

## 新增内容

- `semantic_instruction.py`：受约束的英文道路地图指令解析器，支持 Divider / Ped Crossing / Boundary 同义词、多目标、有限的忽略指令、前后左右组合；不支持的否定、矛盾方位和不认识的对象应拒绝。
- `run_step73_fill_gaps.py`：以真实预训练 DistilBERT 动态编码指令，以已保存真实 BEV 特征验证 2D attention 与旁路回退；使用 nuScenes 官方地图全局矢量、真实 ego 位姿和 LiDAR→Ego 标定独立做地图锚点投影诊断。

## 一键运行

```bash
cd ~/projects/bev_vla
unzip -o /mnt/c/Users/*/Downloads/step73_language_spatial_completion.zip -d scripts/stage2/
source ~/projects/bevformer_author_clean_5080/.venv-author/bin/activate
python3 scripts/stage2/run_step73_fill_gaps.py --self-test
python3 -u scripts/stage2/run_step73_fill_gaps.py --live
```

本机默认路径：

- DistilBERT：`models/distilbert-base-uncased`
- 真实 BEV 缓存：`outputs/bev_features/bev_feature.npy`
- 地图 devkit：`BEVFormer-seg/data/nuscenes`
- 前期真实 GT 样本：`outputs/stage2_step62/step62_divider_sample.npz`

文件路径不同可以使用 `--model /path`、`--bev /path`、`--map-root /path`、`--gt /path` 指定，不会自动下载权重。

## 结果

- `outputs/stage2_step73/step73_gap_check.json`：自然语言指令与 BEV 功能测试、真实地图锚点 F1/方向候选。
- `outputs/stage2_step73/step73_front_right_metric_bev_mask.npy`：基于真实 LiDAR→Ego 标定与作者 `point_cloud_range` 的方向模板，只有地图检查成功时生成。
- `docs/stage2/15_语言与空间能力补充验收.md`：中文报告，PASS / NOT_VERIFIED 明确区分。

## 重要边界

这一包**不是**完全的开放式自然语言理解模型：语言解析是受约束规则，真实 DistilBERT 只做文本编码，语言融合头在本轮测试中随机初始化；Step68 曾发现正确/置乱文本差异不足以证明准确语义定位。

地图锚点使用官方数据真实几何与 pose 独立检查 GT 的候选旋转/镜像，**不等于 BEV 模型实际坐标协议已物理闭环**。作者 BEV 150×150，检测 `point_cloud_range` [-51.2,51.2] 米，地图 GT 200×400 对应 60×30 米。禁止用简单 `resize` 当成空间协议修复。

不修改作者源码、checkpoints、GT、不调用 `optimizer.step()`，不覆盖 Step56～72 结果。

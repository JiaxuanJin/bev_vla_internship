# Stage 2 - Language-guided BEV Attention

## 作用

这一目录提供独立的语言条件 BEV 注意力与残差融合 MVP。默认只进行接口与梯度测试，不修改 BEVFormer 主干或既有预训练 checkpoint。

## 文件说明

- `language_encoder.py`：DistilBERT 文本编码与 attention-mask 池化
- `semantic_projection.py`：将文本隐藏向量映射到 256 维 BEV 通道
- `language_bev_attention.py`：生成 `(B,1,H,W)` 空间注意力
- `bev_fusion.py`：小系数残差融合，支持原始 BEV 精确回退
- `run_step56_smoke.py`：形状、有限值、前向、梯度测试

## 执行

在项目根目录 `~/projects/bev_vla` 运行：

```bash
python3 -m scripts.stage2.run_step56_smoke --text-mode synthetic
```

若需要真实预训练语言编码器，先在适合的虚拟环境中安装 `transformers` 并准备 DistilBERT 模型：

```bash
python3 -m scripts.stage2.run_step56_smoke --text-mode distilbert
```

会自动寻找 `outputs/bev_features/bev_feature.npy`。注意该文件此前可能来自 BEVFormer 检测模型，仅作为真实特征接口验证，不等于分割模型端到端集成；如不想读取可加 `--feature-source synthetic`。

## 指标边界

- 合成模式没有加载任何预训练语言模型；不能据此声称具备自然语言理解能力。
- 本阶段冒烟测试损失不是分割/检测监督信号。
- 未经训练的语言融合模块不能作为分割精度提升结论。
- 后续接入分割头需要保证 BEV 特征方向、时序状态、GT 和评估协议一致。

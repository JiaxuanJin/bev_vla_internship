#!/usr/bin/env python3
"""整理阶段二可追溯证据、中文文档和图片，并安全提交 Git。

不修改训练权重、不执行模型推理、不运行第三方安装器。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


REPO_NAME = "bev_vla_internship"
ALLOWED_REMOTE = "JiaxuanJin/bev_vla_internship"
TODAY = "2026-10-21"
OUT_DIR = Path("results/stage2_2026-10-21")
IMG_DIR = Path("docs/images/stage2")
STAGE_DIR = Path("docs/stage2")

# 只保留小型结果；大模型权重、数据集、NPY 和缓存永不进入本次提交。
EVIDENCE_FILES = {
    "step55": "outputs/protocol_audit_2026-10-21/step55_protocol_snapshot.json",
    "step58": "outputs/stage2_step58/step58_author_live_smoke.json",
    "step59": "outputs/stage2_step59/step59_training_summary.json",
    "step60": "outputs/stage2_step60/step60_ab_summary.json",
    "step61": "outputs/stage2_step61/step61_supervised_summary.json",
    "step62": "outputs/stage2_step62/step62_diagnostics.json",
    "step64": "outputs/stage2_step64/step64_recovered_summary.json",
    "step65": "outputs/stage2_step65/step65_ab_summary.json",
    "step66": "outputs/stage2_step66/step66_ab_summary.json",
    "step67": "outputs/stage2_step67/step67_ab_summary.json",
    "step68": "outputs/stage2_step68/step68_acceptance.json",
    "step69": "outputs/stage2_step69/step69_loss_contract.json",
    "step70_audit": "outputs/stage2_step70/step70_audit.json",
    "step70_live": "outputs/stage2_step70/step70_joint_live.json",
    "step71": "outputs/stage2_step71/step71_acceptance.json",
    "step72": "outputs/stage2_step72/step72_real_joint_gradient.json",
    "step73": "outputs/stage2_step73/step73_gap_check.json",
}
REAL_FIGURES = [
    "step62_divider_overlay.png",
    "step62_ped_crossing_overlay.png",
    "step62_boundary_overlay.png",
    "step62_threshold_scan.png",
]
REQUIRED = ("step71", "step72", "step73")
BANNED_EXTENSIONS = (
    ".pth", ".pt", ".ckpt", ".safetensors", ".onnx", ".pkl",
    ".npy", ".npz", ".zip", ".tgz", ".tar", ".pem", ".key", ".log",
)


def run(command: list[str], root: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    """执行非 shell 命令，避免注入与命令字符串扩展。"""
    return subprocess.run(command, cwd=root, check=check, text=True, capture_output=True)


def read_json(root: Path, name: str, required: bool = False) -> dict[str, Any]:
    """读取本地原始实验记录，不推测缺失结果。"""
    path = root / EVIDENCE_FILES[name]
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"缺少必须的实验 JSON：{path}")
        return {}
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"实验 JSON 不是对象：{path}")
    return obj


def metric(value: Any, places: int = 5) -> str:
    """数值转换；缺少真实字段时显示未记录。"""
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "未记录"
    return f"{float(value):.{places}f}"


def make_diagram() -> str:
    """输出可审核的纯 SVG 架构示意图，不依赖字体图片文件。"""
    blocks = [
        (30, 62, 215, 80, "Six Cameras", "nuScenes input", "#eff6ff", "#1d4ed8"),
        (30, 188, 215, 80, "Language Query", "English command", "#f0fdf4", "#15803d"),
        (290, 62, 232, 80, "BEVFormer", "Shared BEV 150 x 150", "#eff6ff", "#1d4ed8"),
        (290, 188, 232, 80, "DistilBERT", "768-d text embedding", "#f0fdf4", "#15803d"),
        (570, 125, 230, 96, "Spatial Attention", "Residual BEV fusion", "#fff7ed", "#c2410c"),
        (844, 40, 270, 78, "3D Detection", "cls + bbox outputs", "#eef2ff", "#4338ca"),
        (844, 146, 270, 78, "Map Segmentation", "4 x 200 x 400", "#fdf2f8", "#be185d"),
        (844, 254, 270, 78, "Attention Supervision", "GT-targeted objective", "#ecfeff", "#0e7490"),
    ]
    items = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1160" height="395" viewBox="0 0 1160 395">',
        '<rect width="1160" height="395" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}</style>',
        '<text x="30" y="31" font-size="22" font-weight="700" fill="#111827">Stage 2 | Language-Guided BEV Perception</text>',
        '<text x="30" y="368" font-size="14" fill="#64748b">Validated: real dual-head forward (Step70), real multi-loss gradients (Step72). Not a mIoU improvement claim.</text>',
    ]
    for x, y, w, h, title, sub, fill, text_color in blocks:
        items.extend([
            f'<rect x="{x}" y="{y}" rx="14" width="{w}" height="{h}" fill="{fill}" stroke="{text_color}" stroke-opacity="0.28"/>',
            f'<text x="{x+17}" y="{y+33}" font-size="17" font-weight="700" fill="{text_color}">{escape(title)}</text>',
            f'<text x="{x+17}" y="{y+56}" font-size="13" fill="#475569">{escape(sub)}</text>',
        ])
    paths = [
        "M245,102 H290", "M245,228 H290", "M522,102 H548 V153 H570",
        "M522,228 H548 V195 H570", "M800,172 H827 V79 H844",
        "M800,172 H844", "M800,172 H827 V293 H844",
    ]
    items.append('<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#475569"/></marker></defs>')
    for p in paths:
        items.append(f'<path d="{p}" fill="none" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"/>')
    items.append('</svg>')
    return "\n".join(items) + "\n"


def make_evidence_diagram(joint: dict[str, Any], grad: dict[str, Any]) -> str:
    """根据真实验收 JSON 生成证据概览 SVG。"""
    outputs = joint.get("joint_live", {}).get("values", {})
    grad_data = grad.get("gradients", {})
    names = [
        ("seg_preds", "Segmentation"),
        ("all_cls_scores", "Detection cls"),
        ("all_bbox_preds", "Detection bbox"),
    ]
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="370" viewBox="0 0 1050 370">',
        '<rect width="1050" height="370" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif}</style>',
        '<text x="25" y="32" font-size="23" font-weight="700" fill="#111827">Stage 2 | Verified Forward &amp; Gradient Checks</text>',
        '<text x="25" y="58" font-size="14" fill="#64748b">Source: Step71 acceptance and Step72 real gradient JSON</text>',
    ]
    for i, (key, label) in enumerate(names):
        y = 83 + i * 78
        record = outputs.get(key, {})
        enhanced = metric(record.get("enhanced_mae"), 6)
        bypass = metric(record.get("bypass_max"), 4)
        lines.extend([
            f'<rect x="25" y="{y}" rx="11" width="480" height="66" fill="#eff6ff"/>',
            f'<text x="43" y="{y+27}" font-size="17" font-weight="700" fill="#1e40af">{escape(label)}</text>',
            f'<text x="43" y="{y+49}" font-size="13" fill="#334155">Bypass max: {escape(bypass)}  |  Enabled MAE: {escape(enhanced)}</text>',
        ])
    order = [("detection", "Detection"), ("segmentation", "Segmentation"), ("attention", "Attention"), ("total", "Joint total")]
    for i, (key, label) in enumerate(order):
        y = 83 + i * 61
        record = grad_data.get(key, {})
        loss = metric(record.get("loss_value"), 4)
        norm = metric(record.get("global_l2"), 4)
        lines.extend([
            f'<rect x="525" y="{y}" rx="10" width="500" height="53" fill="#f0fdf4"/>',
            f'<text x="543" y="{y+22}" font-size="16" font-weight="700" fill="#166534">{escape(label)}</text>',
            f'<text x="543" y="{y+41}" font-size="12" fill="#334155">Loss: {escape(loss)}  |  Gradient L2: {escape(norm)}</text>',
        ])
    lines.append('<text x="25" y="350" font-size="13" fill="#64748b">Single-frame autograd; no optimizer.step; no evidence of converged model or language grounding.</text>')
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def report(root: Path, joint: dict[str, Any], grad: dict[str, Any], spatial: dict[str, Any], copied: list[str]) -> str:
    """从证据中生成第二阶段正式摘要。"""
    forward = joint.get("joint_live", {}).get("values", {})
    grads = grad.get("gradients", {})
    lang = spatial.get("language", {})
    geom = spatial.get("map_geometry", {})
    scores = geom.get("direct_projection", {})
    record_rows = []
    for key, title in [("seg_preds", "分割"), ("all_cls_scores", "检测分类"), ("all_bbox_preds", "检测回归")]:
        item = forward.get(key, {})
        record_rows.append(
            f"| {title} | `{item.get('shape', '未记录')}` | {metric(item.get('bypass_max'))} | {metric(item.get('enhanced_mae'))} |"
        )
    grad_rows = []
    for key, title in [("detection", "检测"), ("segmentation", "地图分割"), ("attention", "注意力"), ("total", "联合总损失")]:
        item = grads.get(key, {})
        grad_rows.append(f"| {title} | {metric(item.get('loss_value'))} | {metric(item.get('global_l2'))} | {item.get('all_finite', '未记录')} |")
    figure_lines = []
    for img, caption in [
        ("step62_divider_overlay.png", "Divider 注意力与地图 GT 可视化"),
        ("step62_ped_crossing_overlay.png", "Ped Crossing 注意力与地图 GT 可视化"),
        ("step62_boundary_overlay.png", "Boundary 注意力与地图 GT 可视化"),
    ]:
        if img in copied:
            figure_lines.append(f"**{caption}**\n\n![{caption}](../images/stage2/{img})\n")
    if not figure_lines:
        figure_lines.append("Step62 可视化文件未在本地输出目录发现，暂不插入未经核实的图片。")
    lines = [
        "# 第二阶段核心模块进展及证据汇总（2026-10-21）",
        "",
        "> 实验范围：语言编码、BEV 注意力生成与融合、联合检测/分割前向、多任务损失及梯度可达性。",
        "",
        "## 1. 系统数据流",
        "",
        "![语言引导 BEV 系统架构](../images/stage2/stage2_architecture.svg)",
        "",
        "作者联合配置采用共享 BEV 主干和检测/分割双 Head；语言模块在共享 BEV 特征上生成空间注意力并执行残差融合。",
        "",
        "## 2. 真实模型集成与梯度验收",
        "",
        "![前向与梯度验收实测摘要](../images/stage2/stage2_evidence.svg)",
        "",
        "### 2.1 Step70：真实联合 Head 前向",
        "",
        "| 输出 | 维度 | 关闭融合最大绝对差 | 开启融合平均绝对差 |",
        "|---|---|---:|---:|",
        *record_rows,
        "",
        "采用作者联合 checkpoint，关闭增强时输出与原始预测一致；开启增强后两类感知 Head 均出现响应。该对照验证可插拔集成，不表示精度提升。",
        "",
        "### 2.2 Step72：真实监督的多任务梯度",
        "",
        "| 分支 | Loss | 梯度 L2 | 梯度均有限 |",
        "|---|---:|---:|---|",
        *grad_rows,
        "",
        f"测试模式：`{grad.get('mode', '未记录')}`。使用真实 3D 和地图 GT，在冻结作者主干、单帧无历史模式下检测梯度连通性；没有执行 `optimizer.step()`。",
        "",
        "## 3. 语言与空间功能检查",
        "",
        f"- Step73 指令解析与真实 DistilBERT：`{lang.get('status', '未记录')}`；文本向量形状 `{lang.get('text_shape', '未记录')}`，注意力形状 `{lang.get('attention_shape', '未记录')}`。",
        f"- 不同指令的编码 MAE：{metric(lang.get('different_text_embedding_mae'))}；Attention MAE：{metric(lang.get('different_text_attention_mae'))}。",
        f"- 地图锚点独立投影：`{geom.get('status', '未记录')}`；direct F1（2 px 容差）= {metric(scores.get('f1_with_2px_tolerance'))}，Precision = {metric(scores.get('precision'))}，Recall = {metric(scores.get('recall'))}。",
        "- 注意力头的独立功能检查不构成目标定位准确率证明；网络 BEV 与地图 GT 物理坐标的端到端对应关系仍需进一步验证。",
        "",
        "## 4. 类别注意力示例与问题分析",
        "",
        "以下图片复制自 Step62 原始可视化结果，用于记录模型对地图要素的注意力分布，与最终语言语义有效性结论区分。",
        "",
        *figure_lines,
        "## 5. 有限的验证结论",
        "",
        "- Step65～67 在单帧 Mini Val 的部分前景类别上有改善，但 Background IoU 明显退化，未形成稳定的四类 mIoU 提升。",
        "- Step67 基于 Mini Train 校准条件锁定 Original，不应将 raw adapter 的局部指标当作正式改善。",
        "- Step68 正常文本、置乱文本与常量文本预测较接近，语言正确定位尚未证实。",
        "- Step73 具有受约束指令解析能力，不等同于任意开放自然语言理解。",
        "- 与 nuScenes 地图 GT 的物理空间完整对应未闭环。",
        "- 检测 + 分割 + 注意力联合梯度通过，不代表训练已收敛或提升检测/分割准确率。",
        "",
        "## 6. 复现及资料位置",
        "",
        "- 核心代码：`scripts/stage2/`",
        "- 架构说明：`docs/stage2/11_模块架构设计.md`",
        "- 单元测试：`docs/stage2/12_单元测试报告.md`",
        "- 阶段任务对照：`docs/stage2/13_阶段二任务对照.md`",
        "- 真实联合梯度：`docs/stage2/14_真实联合梯度验证.md`",
        "- 语言与空间功能：`docs/stage2/15_语言与空间能力补充验收.md`",
        "- 小型 JSON 结果归档：`results/stage2_2026-10-21/`",
        "",
        "统一运行命令参见 [第二阶段复现指引](17_第二阶段复现指引.md)。",
        "",
    ]
    return "\n".join(lines)


def reproduction_guide() -> str:
    return """# 第二阶段复现指引（2026-10-21）

## 1. 环境与目录

- 本地项目根目录：`~/projects/bev_vla/`。
- 作者隔离环境：`~/projects/bevformer_author_clean_5080/.venv-author/`。
- 作者联合工程：`~/projects/bevformer_author_clean_5080/BEVFormer-seg/`。
- 数据集：nuScenes v1.0-mini；训练和验证结果不混用。
- 联合权重：`/mnt/g/nus/bevformer_small_joint_author.pth`（**不纳入 Git**）。
- 指令编码模型：`models/distilbert-base-uncased/`（**不纳入 Git**）。

请根据本机 GPU/CUDA、MMCV 自定义算子和 Python 环境适配情况准备依赖；本仓库没有打包第三方源码、模型和数据。

## 2. 验证命令

```bash
cd ~/projects/bev_vla
source ~/projects/bevformer_author_clean_5080/.venv-author/bin/activate

# 多任务 Loss 单元测试
python3 scripts/stage2/run_step69_loss_contract.py --self-test

# 真实共享 BEV 检测 + 分割前向（必须准备联合 checkpoint）
python3 -u scripts/stage2/run_step70_joint_integration.py \\
  --audit --live \\
  --joint-checkpoint /mnt/g/nus/bevformer_small_joint_author.pth \\
  --index 0

# 阶段回归与阶段任务核对
python3 -u scripts/stage2/run_step71_stage2_review.py --audit --run-tests

# 真实检测 + 分割 + 注意力梯度检查，不更新权重
python3 -u scripts/stage2/run_step72_real_joint_gradient.py --live --index 0

# 受约束自然语言和地图锚点诊断
python3 -u scripts/stage2/run_step73_fill_gaps.py --live
```

> 运行 Step70、72 前请先检查作者历史依赖的 NumPy 别名和 Dataset/DataLoader 兼容补丁是否已按项目现有环境配置；不要修改第三方公开 checkpoint。

## 3. 输入/输出

- 实验小型归档：`results/stage2_2026-10-21/`。
- Step70 原始记录：`outputs/stage2_step70/step70_joint_live.json`。
- Step72 原始记录：`outputs/stage2_step72/step72_real_joint_gradient.json`。
- Step73 原始记录：`outputs/stage2_step73/step73_gap_check.json`。

本地 `outputs/` 不直接入库，只有少量 JSON 证据和实测图片进入 Git。

## 4. 配置边界

- 联合模型配置为 `bevformer_small_seg_det.py`，共享 BEV 特征大小 150×150。
- 地图分割输出 4×200×400；这两种网格不能直接按像素重合来判断物理空间一致性。
- Step70 验证单帧共享 BEV 前向；Step72 只检验真实联合梯度，未执行参数更新。
- 真实开放语言 grounding、全时序 81 帧增强评估和完整坐标物理校验仍需进一步工作。

## 5. 权重完整性

本机联合 checkpoint 的哈希由以下命令复核：

```bash
sha256sum /mnt/g/nus/bevformer_small_joint_author.pth
```

对应 10 月 21 日实验日志记录的 SHA256：
`1345919901785b8df7ee16838d62f9f7a00444c9d8061f0dbee712d1b1b67dcf`。
"""


def logbook_summary() -> str:
    return """# 2026-10-21 实验记录｜语言引导 BEV 注意力增强与联合训练链路

## 1. 当日目标

围绕第二阶段语言引导 BEV 注意力增强模块，完成语言编码、注意力映射、共享 BEV 双任务前向与联合梯度链路核验，同时归档失败试验与受限结论。

## 2. 工作过程

- **Step55：** 只读核对作者分割配置、nuScenes 版本、地图文件和评估口径，未确认作者历史训练协议与本地完全一致。
- **Step56～58：** 搭建语言编码与 BEV Attention/Fusion，完成 DistilBERT 实际文本编码和作者 BEVFormer 分割路径的运行时插入测试；关闭模块回退一致。
- **Step59～64：** 使用 Mini Train/Val 的真实 GT 验证小样本分割监督和注意力训练，生成 Divider、Ped Crossing、Boundary 类别注意力与 GT 对照。观察到前景类别极度不平衡、注意力定位不稳定。
- **Step65～67：** 对 Adapter、背景保护和校准方案做同条件 A/B；部分前景 mIoU 有改善，但 Background IoU 或四类 mIoU 下降，Step67 最终未满足校准约束，锁定 Original 回退。
- **Step68～69：** 验证文本干预对输出的影响，编写检测/分割/注意力联合 Loss 接口并完成六项单元测试；正常文本效果未稳定优于置乱文本。
- **Step70：** 下载并校验联合 checkpoint，修复数据初始化参数和测试推理 GT 传参兼容问题。真实共享 BEV 双 Head 前向通过，检测与分割输出旁路最大差均为零。
- **Step71～72：** 形成架构与验收文档，使用 Mini Train 真实 3D/地图 GT 完成 Detection、Segmentation、Attention 三分支及联合梯度检查。只调用 `backward()`，未执行 `optimizer.step()`。
- **Step73：** 增加受约束英文指令解析，使用真实 DistilBERT 编码和二维 BEV 注意力接口；进行 LiDAR/Ego 标定与地图锚点的独立坐标诊断。

## 3. 关键实测证据

![第二阶段共享 BEV 双任务架构](images/stage2/stage2_architecture.svg)

![真实双 Head 前向和三任务梯度检查](images/stage2/stage2_evidence.svg)

阶段二核心模块运行情况及详细数据参见 [阶段二技术进展与证据汇总](stage2/16_阶段二实证汇总_2026-10-21.md)。

## 4. 可视化分析

原始分割注意力与 GT 对照：

![Divider 注意力与 GT](images/stage2/step62_divider_overlay.png)

![Boundary 注意力与 GT](images/stage2/step62_boundary_overlay.png)

> 图片仅用于展示注意力分布，不将可视化响应等同于准确语言定位能力。

## 5. 当前结论与后续安排

已取得真实 DistilBERT 前向、共享 BEV 双任务 Head 前向、三任务联合 Loss 梯度，以及受约束语言解析的功能证据。主要待改进点为：模型学习到的语义定位能力、地图 GT 与 BEV 网络的完整物理坐标协议、以及稳定的精度收益。

下一步围绕统一系统运行入口、环境复现与阶段二单元测试进行工程收束；保留有效对照和负结果，不把已验证的模块可运行性扩大解释为精度改进。
"""


def safe_write(path: Path, content: str, section_heading: str | None = None) -> None:
    """单份 Logbook 已存在时只补充缺少章节，避免覆盖现有记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")
        return
    if section_heading and section_heading not in path.read_text(encoding="utf-8"):
        path.write_text(path.read_text(encoding="utf-8") + "\n\n" + content.replace(content.splitlines()[0] + "\n", "", 1), encoding="utf-8")
        print(f"[Step74] 已追加至原有日志：{path}")
        return
    print(f"[Step74] 保护已存在文件，未覆盖：{path}")


def validate_images(root: Path, copied: list[str]) -> None:
    """文档中的图片必须真实存在；不存在时移除对应引用。"""
    target = root / "docs/logbook_2026-10-21.md"
    if not target.exists():
        return
    data = target.read_text(encoding="utf-8")
    for name in REAL_FIGURES:
        if name not in copied:
            data = "\n".join(line for line in data.split("\n") if f"images/stage2/{name}" not in line)
    target.write_text(data, encoding="utf-8")
    for path in [target, root / STAGE_DIR / "16_阶段二实证汇总_2026-10-21.md"]:
        content = path.read_text(encoding="utf-8")
        for rel in re.findall(r"!\[[^]]*]\(([^)]+)\)", content):
            if not (path.parent / rel).is_file():
                raise FileNotFoundError(f"Markdown 图片链接失效：{path} -> {rel}")


def prepare(root: Path) -> list[Path]:
    """整合现有证据文件、实际可视化和工程文档。"""
    for step in REQUIRED:
        read_json(root, step, required=True)
    joint = read_json(root, "step71", required=True)
    grad = read_json(root, "step72", required=True)
    spatial = read_json(root, "step73", required=True)

    if joint.get("joint_live", {}).get("status") != "PASS":
        raise RuntimeError("Step71 未证明真实双 Head 前向通过，中止汇总")
    if grad.get("status") != "PASS_REAL_JOINT_GRADIENT_NO_OPTIMIZER_STEP":
        raise RuntimeError("Step72 真实梯度验收状态不匹配，中止汇总")
    if spatial.get("language", {}).get("status") != "PASS_FUNCTION_ONLY":
        raise RuntimeError("Step73 语言功能状态不匹配，中止汇总")

    (root / IMG_DIR).mkdir(parents=True, exist_ok=True)
    (root / OUT_DIR).mkdir(parents=True, exist_ok=True)
    (root / STAGE_DIR).mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for figure in REAL_FIGURES:
        source = root / "outputs/stage2_step62" / figure
        if source.is_file() and source.stat().st_size > 0:
            shutil.copy2(source, root / IMG_DIR / figure)
            copied.append(figure)
            print(f"[Step74] 图片：{figure}")
        else:
            print(f"[Step74] 注意：找不到原始图，跳过：{source}")

    (root / IMG_DIR / "stage2_architecture.svg").write_text(make_diagram(), encoding="utf-8")
    (root / IMG_DIR / "stage2_evidence.svg").write_text(make_evidence_diagram(joint, grad), encoding="utf-8")

    evidence_copied: list[str] = []
    for step, source in EVIDENCE_FILES.items():
        path = root / source
        if not path.is_file():
            continue
        if path.stat().st_size > 2_000_000:
            print(f"[Step74] 记录过大，跳过：{path}")
            continue
        target = root / OUT_DIR / f"{step}_{path.name}"
        shutil.copy2(path, target)
        evidence_copied.append(target.name)

    (root / OUT_DIR / "README.md").write_text(
        "# 第二阶段精选实验结果\n\n此目录为本机原始 `outputs/` 小型 JSON 的副本；不包含 checkpoint、原始数据和特征缓存。\n\n" +
        "\n".join(f"- `{x}`" for x in evidence_copied) + "\n",
        encoding="utf-8",
    )
    target_summary = root / STAGE_DIR / "16_阶段二实证汇总_2026-10-21.md"
    safe_write(target_summary, report(root, joint, grad, spatial, copied))
    safe_write(root / STAGE_DIR / "17_第二阶段复现指引.md", reproduction_guide())
    safe_write(root / "docs/logbook_2026-10-21.md", logbook_summary(), "## 2. 工作过程")

    readme = root / "README.md"
    original = readme.read_text(encoding="utf-8") if readme.is_file() else "# BEV / DriveLM / VLA 线上实习项目\n"
    old_intro = "当前为**第一阶段：环境搭建、数据对齐与基线复现**；VLA 模型训练尚未完成。"
    revised_intro = "已完成第一阶段基线复现并推进第二阶段语言引导 BEV 注意力增强；共享 BEV 双 Head 的单帧前向及真实梯度验收通过。完整 VLA 模型训练尚未完成。"
    original = original.replace(old_intro, revised_intro)
    if "## 第二阶段｜语言引导 BEV 注意力增强（2026-10-21）" not in original:
        original += """

## 第二阶段｜语言引导 BEV 注意力增强（2026-10-21）

第二阶段已实现 DistilBERT 文本编码、语义到 BEV 的二维注意力映射、共享 BEV 融合接口，以及检测 + 分割 + 注意力多任务损失。作者联合 BEVFormer 双 Head 的真实单帧前向与三任务梯度检查通过；不宣称语言语义定位准确、完整时序评估或稳定精度提升。

![阶段二系统架构](docs/images/stage2/stage2_architecture.svg)

- [阶段二实测证据与图片](docs/stage2/16_阶段二实证汇总_2026-10-21.md)
- [第二阶段复现命令](docs/stage2/17_第二阶段复现指引.md)
- [模块架构设计](docs/stage2/11_模块架构设计.md)
- [单元测试报告](docs/stage2/12_单元测试报告.md)
- [真实联合梯度验证](docs/stage2/14_真实联合梯度验证.md)
- [2026-10-21 Logbook](docs/logbook_2026-10-21.md)

> 模型权重、nuScenes 数据和完整推理产物未放入仓库；选定的 JSON 证据位于 `results/stage2_2026-10-21/`。
"""
        readme.write_text(original, encoding="utf-8")
        print("[Step74] 已更新 README 第二阶段导航")

    validate_images(root, copied)
    print(f"[Step74] 文档整理完成：{len(evidence_copied)} 份原始 JSON，{len(copied)} 张实验图，2 张 SVG")
    return [root / OUT_DIR, root / IMG_DIR, root / STAGE_DIR,
            root / "docs/logbook_2026-10-21.md", root / "README.md"]


def check_git(root: Path) -> tuple[str, str]:
    """检查 Git 远端与工作分支，禁止危险覆盖。"""
    top = run(["git", "rev-parse", "--show-toplevel"], root).stdout.strip()
    if Path(top).resolve() != root.resolve():
        raise RuntimeError(f"当前目录并非 Git 根目录：{top}")
    branch = run(["git", "branch", "--show-current"], root).stdout.strip()
    if not branch:
        raise RuntimeError("当前处于 detached HEAD；请先切换工作分支")
    url = run(["git", "remote", "get-url", "origin"], root).stdout.strip()
    if ALLOWED_REMOTE not in url:
        raise RuntimeError(f"origin 与预期仓库不符：{url}")
    current = run(["git", "diff", "--cached", "--name-only"], root).stdout.strip()
    if current:
        raise RuntimeError("检测到已暂存的其他修改；为了避免混入提交，停止自动发布。\n" + current)
    return branch, url


def publish(root: Path) -> None:
    """严格按白名单暂存、检测和提交，再安全推送当前分支。"""
    branch, remote = check_git(root)
    targets = ["README.md", "docs/logbook_2026-10-21.md"]
    targets.extend(str(p.relative_to(root)) for p in (root / STAGE_DIR).glob("*.md"))
    targets.extend(str(p.relative_to(root)) for p in (root / IMG_DIR).iterdir() if p.is_file())
    targets.extend(str(p.relative_to(root)) for p in (root / OUT_DIR).iterdir() if p.is_file())
    targets.extend(str(p.relative_to(root)) for p in (root / "scripts/stage2").rglob("*.py"))
    targets.extend(str(p.relative_to(root)) for p in (root / "scripts/stage2").rglob("*.md"))
    targets.extend(str(p.relative_to(root)) for p in (root / "docs").glob("stage2*CN.md"))
    targets.extend(str(p.relative_to(root)) for p in (root / "docs").glob("stage2_acceptance_*.md"))
    targets = sorted(set(targets))
    print(f"[Step74] 分支：{branch} | origin：{remote}")
    print(f"[Step74] 按白名单检查待提交文件：{len(targets)} 项")

    for rel in targets:
        path = root / rel
        if not path.is_file():
            continue
        if path.suffix.lower() in BANNED_EXTENSIONS or path.stat().st_size > 4_000_000:
            raise RuntimeError(f"拒绝暂存大型或禁止格式：{rel}")
        if re.search(r"(^|/)(\.env|id_rsa|credentials|secret)(/|\.|$)", rel, re.I):
            raise RuntimeError(f"拒绝暂存疑似敏感文件：{rel}")
    if targets:
        run(["git", "add", "--", *targets], root)

    staged = run(["git", "diff", "--cached", "--name-only", "-z"], root).stdout.split("\0")
    staged = [item for item in staged if item]
    if not staged:
        print("[Step74] 没有新修改需要提交；检查远端状态。")
        return
    for rel in staged:
        path = root / rel
        if not path.exists() or path.stat().st_size > 4_000_000 or path.suffix.lower() in BANNED_EXTENSIONS:
            raise RuntimeError(f"暂存检查失败：{rel}")
        if path.suffix.lower() in (".py", ".md", ".json", ".txt"):
            text_data = path.read_text(encoding="utf-8", errors="replace")
            if re.search(r"ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{30,}", text_data):
                raise RuntimeError(f"疑似敏感 Token，停止提交：{rel}")
    summary = run(["git", "diff", "--cached", "--stat"], root).stdout
    print("\n[Step74] 提交内容概览：\n" + summary)
    message = "feat: archive stage2 language-guided BEV integration and validation"
    try:
        run(["git", "commit", "-m", message], root)
    except subprocess.CalledProcessError as exc:
        print(exc.stdout, exc.stderr)
        raise RuntimeError("git commit 失败。文件已暂存，请检查 Git 身份与配置") from exc
    print(f"[Step74] 已创建本地提交，开始推送 {branch} ...")
    try:
        pushed = run(["git", "push", "-u", "origin", "HEAD"], root)
        print(pushed.stdout + pushed.stderr)
    except subprocess.CalledProcessError as exc:
        print(exc.stdout + exc.stderr)
        print("[Step74] GitHub 推送失败：本地提交仍存在。不要 force push。")
        print("[Step74] 请复制上面的错误与 git status -sb 发来继续处理。")
        raise SystemExit(2) from exc
    print(f"[Step74] PUSH SUCCESS，分支：{branch}")
    print(f"[Step74] 日志：https://github.com/{ALLOWED_REMOTE}/blob/{branch}/docs/logbook_2026-10-21.md")
    print(f"[Step74] 归档：https://github.com/{ALLOWED_REMOTE}/tree/{branch}/docs/stage2")
    print(f"[Step74] 图片：https://github.com/{ALLOWED_REMOTE}/tree/{branch}/docs/images/stage2")
    status = run(["git", "status", "--short"], root).stdout
    print(f"[Step74] 剩余未提交文件（可能是旧实验或其他项目）：\n{status or '(无)'}")


def main() -> None:
    """入口。"""
    parser = argparse.ArgumentParser(description="Step74 阶段二证据整理与 Git 安全提交")
    parser.add_argument("--prepare", action="store_true", help="只整理文档、图片和精选 JSON")
    parser.add_argument("--publish", action="store_true", help="整理后按白名单提交并推送当前分支")
    args = parser.parse_args()
    if not (args.prepare or args.publish):
        parser.error("需要 --prepare 或 --publish")
    root = Path(__file__).resolve().parents[1]
    if not (root / "scripts/stage2").is_dir():
        raise RuntimeError("未找到 scripts/stage2；请将本文件放到项目 scripts/ 下")
    prepare(root)
    if args.publish:
        publish(root)


if __name__ == "__main__":
    main()

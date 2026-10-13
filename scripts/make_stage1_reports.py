#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 BEVFormer-Small 实习第一阶段中文报告。

使用方法（在项目根目录执行）：
    python scripts/make_stage1_reports.py

说明：
- 只读取现有实验结果和目录，不修改原始数据、模型权重。
- 所有报告正文、表头、图注和终端提示均使用中文；保留必要英文技术术语。
- 严格区分 nuScenes Mini 实测结果与尚未完成的 DriveLM 验证集评估。
- 仅依赖 Python 标准库。
"""

import json
import math
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BEV_ROOT = PROJECT_ROOT / "BEVFormer"
DOCS_DIR = PROJECT_ROOT / "docs"
FEATURE_DIR = PROJECT_ROOT / "outputs" / "bev_features"
REPORT_DATE = "2026-10-12"

CLASS_NAMES = {
    "car": "小汽车", "truck": "卡车", "bus": "公交车/客车", "trailer": "拖车",
    "construction_vehicle": "工程车辆", "pedestrian": "行人", "motorcycle": "摩托车",
    "bicycle": "自行车", "traffic_cone": "交通锥", "barrier": "路障",
}
ERROR_NAMES = {
    "trans_err": "平移误差 mATE",
    "scale_err": "尺寸误差 mASE",
    "orient_err": "朝向误差 mAOE",
    "vel_err": "速度误差 mAVE",
    "attr_err": "属性误差 mAAE",
}


def fmt(value, digits=4):
    """格式化指标；没有实测数据时显示‘未提供’，绝不补零。"""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return f"{value:.{digits}f}"
    return "未提供"


def pretty(value):
    """将 Python 对象格式化为容易读的文本。"""
    if value is None:
        return "未核实"
    if isinstance(value, (list, tuple)):
        return " × ".join(str(v) for v in value)
    return str(value)


def relative(path):
    """输出相对项目根目录的路径，方便复制复现。"""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def file_size(path):
    if not path.is_file():
        return "未找到"
    size = path.stat().st_size
    return f"{size / 1024 / 1024:.2f} MB"


def md_image(path, description):
    """报告和图片都位于 docs 内，使用可在 VS Code 预览的相对路径。"""
    if path.is_file():
        return f"![{description}]({path.relative_to(DOCS_DIR).as_posix()})"
    return f"> 待补充图片：{description}（目前未找到 `{relative(path)}`）"


def dir_inventory(path, max_entries=20):
    """只列浅层文件名，避免遍历大数据集耗时太长。"""
    if not path.exists():
        return f"- ⚠️ 未找到目录：`{relative(path)}`"
    if not path.is_dir():
        return f"- `{path.name}`（文件）"
    entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold()))
    rows = [f"- `{entry.name}`（{'文件夹' if entry.is_dir() else '文件'}）" for entry in entries[:max_entries]]
    if len(entries) > max_entries:
        rows.append(f"- ……另有 {len(entries) - max_entries} 项，未在文档中展开")
    return "\n".join(rows) or "- 目录当前为空"


def read_json(path, required=False):
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"未找到必要文件：{path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_latest_metrics():
    """从真实评估目录中查找最近修改的 metrics_summary.json。"""
    base = BEV_ROOT / "test" / "bevformer_small"
    candidates = list(base.rglob("metrics_summary.json")) if base.exists() else []
    if not candidates:
        raise FileNotFoundError(
            "未找到 BEVFormer/test/bevformer_small/**/metrics_summary.json，"
            "请确认昨天的评估文件仍在。"
        )
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    return path, read_json(path, required=True)


def prediction_count(path):
    """样本数直接从结果 JSON 计数，不采用人为填写的数量。"""
    if not path.is_file():
        return "未核实"
    try:
        data = read_json(path)
        results = data.get("results")
        return len(results) if isinstance(results, dict) else "未核实"
    except (OSError, ValueError, TypeError) as exc:
        print(f"[提示] 无法读取预测样本数量：{exc}")
        return "未核实"


def main():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path, metrics = get_latest_metrics()
    results_path = metrics_path.parent / "results_nusc.json"
    sample_count = prediction_count(results_path)

    checkpoint = BEV_ROOT / "ckpts" / "bevformer_small_epoch_24.pth"
    config = BEV_ROOT / "projects" / "configs" / "bevformer" / "bevformer_small.py"
    capture_path = FEATURE_DIR / "capture_info.json"
    capture = read_json(capture_path)

    class_aps = metrics.get("mean_dist_aps") or {}
    if class_aps:
        class_table = "\n".join(
            f"| {CLASS_NAMES.get(cls, '其他')}（`{cls}`） | {fmt(val)} |"
            for cls, val in sorted(class_aps.items())
        )
    else:
        class_table = "| 当前 JSON 未提供分类 AP | 未提供 |"

    tp_errors = metrics.get("tp_errors") or {}
    error_table = "\n".join(
        f"| {chinese} | {fmt(tp_errors.get(code))} |"
        for code, chinese in ERROR_NAMES.items()
    )

    img_dir = DOCS_DIR / "images" / REPORT_DATE
    detection_img = DOCS_DIR / "images" / "2026-10-11" / "08_bevformer_predictions.png"

    baseline_report = f"""# 第一阶段：BEVFormer-Small 基线性能实验报告

**实验日期：** {REPORT_DATE}  
**当前实际验证集：** nuScenes `v1.0-mini / mini_val`（不是 DriveLM 正式验证集）  
**模型：** BEVFormer-Small（3D 目标检测版本）  
**权重：** `{relative(checkpoint)}`（{file_size(checkpoint)}）  
**配置：** `{relative(config)}`  
**评估指标来源：** `{relative(metrics_path)}`  
**预测数据来源：** `{relative(results_path)}`  
**实际评估样本量：** {sample_count}

## 一、实验目的与运行环境

本次实验使用已有预训练权重，在本地 nuScenes Mini 验证集上复现 BEVFormer-Small 的三维目标检测推理和评估过程，同时利用 PyTorch Forward Hook 提取模型内部真实的 BEV Encoder 输出。实验没有重新训练模型，也没有宣称改进基线性能。

- 已验证的硬件：NVIDIA GeForce RTX 5080 Laptop GPU（16 GB）。
- 已验证的软件：PyTorch 2.8.0+cu128、CUDA 可用；原 BEVFormer 环境通过兼容脚本适配运行。
- 关键兼容入口：`scripts/run_bevformer_test_compat.py`。
- **重要实验限制：** 当前预处理对缺失的 CAN Bus 信息采用全零回退，这与完整官方数据条件不完全一致；结果应视为本地 Mini 复现实验，不直接与官方榜单进行严格对比。

### 复现实验的主要命令

在 `~/projects/bev_vla/BEVFormer` 目录执行：

```bash
PYTHONPATH="$PWD:$PWD/tools" torchrun --standalone --nproc_per_node=1 ../scripts/extract_bev_features.py projects/configs/bevformer/bevformer_small.py ckpts/bevformer_small_epoch_24.pth --launcher pytorch --eval bbox --cfg-options data.workers_per_gpu=0
```

该命令进行真实推理与检测评估，特征提取脚本通过 Hook 额外保存第一次 Encoder 前向传播产生的 BEV 张量。

## 二、三维目标检测定量结果

| 评价指标 | 实测结果 |
|---|---:|
| 三维目标检测平均精度 mAP | **{fmt(metrics.get('mean_ap'))}** |
| nuScenes 综合检测评分 NDS | **{fmt(metrics.get('nd_score'))}** |

### 2.1 各目标类别平均精度（AP）

| 目标类别 | AP |
|---|---:|
{class_table}

### 2.2 真阳性误差指标

| 误差项 | 实测结果 |
|---|---:|
{error_table}

**结果说明：** 以上指标由本地 `metrics_summary.json` 自动读取，仅代表当前 nuScenes Mini `mini_val` 数据上的三维目标检测性能，**不是 DriveLM 验证集成绩**。

## 三、真实 BEV 特征提取与可视化

| 特征属性 | 读取结果 |
|---|---|
| 捕获位置 | BEVFormer Encoder 的真实前向输出 |
| 模型原始输出形状 | `{pretty(capture.get('original_shape'))}` |
| 保存张量形状（高 × 宽 × 通道） | `{pretty(capture.get('saved_shape'))}` |
| 数据类型 | `{pretty(capture.get('dtype'))}` |
| 最小值 / 最大值 | {fmt(capture.get('min'))} / {fmt(capture.get('max'))} |
| 均值 / 标准差 | {fmt(capture.get('mean'))} / {fmt(capture.get('std'))} |
| 原始特征数组 | `outputs/bev_features/bev_feature.npy` |

### 3.1 BEV 特征强度热力图

{md_image(img_dir / 'bev_feature_heatmap.png', '图 1：真实 BEV 特征强度热力图（256 维特征向量的 L2 范数）')}

**图示解释：** 每个网格位置的数值是该处 256 维特征向量的 L2 范数。颜色较亮表示特征向量幅值较大，**不等于车辆概率、注意力权重或分割标签**。

### 3.2 不同通道的特征响应

{md_image(img_dir / 'bev_feature_channels.png', '图 2：BEV Encoder 第 0、64、128、192 通道的空间响应')}

**图示解释：** 不同通道具有不同的空间响应分布，但仅依据通道图无法断定其对应车辆、道路或行人语义。网格坐标方向也需要结合场景和几何标定进一步确认。

### 3.3 三维目标检测与 BEV 视角示例

{md_image(detection_img, '图 3：BEVFormer-Small 三维检测预测及 BEV 俯视示例')}

## 四、目前尚未完成的官方任务

1. **DriveLM 正式验证：** 当前只复现了 nuScenes Mini，仍需完成 DriveLM 样本映射/对齐并在规定验证集上评估，不能把 Mini 指标直接写成 DriveLM 基线成绩。
2. **BEV 分割 mIoU：** 当前配置执行的是三维检测，没有直接输出 BEV 分割结果。必须补充具备对应语义分割标注和评估程序的模型/任务，或与导师确认要求后才能报告 mIoU。
3. **特征与具体场景对应：** 当前 BEV 特征来自第一次 Encoder 前向调用；尚未逐一核实其场景标识与相机、标注图的准确配对。
4. **性能对比：** 本报告是基线复现记录，没有做新模块训练，也没有改进前后的性能对照。

## 五、实验产出与复现文件

| 文件 | 用途 |
|---|---|
| `scripts/extract_bev_features.py` | 真实模型中间特征捕获脚本 |
| `outputs/bev_features/bev_feature.npy` | 保存的完整 BEV 特征张量 |
| `outputs/bev_features/capture_info.json` | 张量形状与统计信息 |
| `docs/images/{REPORT_DATE}/bev_feature_heatmap.png` | BEV 强度图 |
| `docs/images/{REPORT_DATE}/bev_feature_channels.png` | 通道响应图 |
| `{relative(metrics_path)}` | 检测指标原始 JSON |
| `{relative(results_path)}` | 三维检测预测原始 JSON |

**下一步：** 继续核对 DriveLM 数据与 nuScenes 样本的关系、处理 BEV 分割 mIoU 的可行性，完成第一阶段交付自查。
"""

    # 数据结构说明：目录来自本机实存文件；字段释义是格式知识，不冒充实际打印过的内容。
    ns_root = BEV_ROOT / "data" / "nuscenes"
    mini_meta = ns_root / "v1.0-mini"
    info_files = sorted(ns_root.glob("nuscenes_infos_temporal_*.pkl")) if ns_root.exists() else []
    info_table = "\n".join(f"| `{relative(p)}` | {file_size(p)} |" for p in info_files)
    if not info_table:
        info_table = "| 暂未定位到 PKL 文件 | — |"

    drive_candidates = [ns_root / "driveLM", ns_root / "DriveLM", ns_root / "drivelm", PROJECT_ROOT / "data" / "drivelm"]
    drive_root = next((p for p in drive_candidates if p.exists()), drive_candidates[0])

    dataset_report = f"""# 第一阶段：数据集结构与标注格式说明

**整理日期：** {REPORT_DATE}  
**已用于 BEVFormer 基线复现的数据：** nuScenes v1.0-mini  
**实习任务书要求的数据：** DriveLM-v1  
**说明：** 本文严格区分“本地已核实的文件”和“数据格式的一般性说明”。

## 一、本地数据文件结构

### 1.1 nuScenes 数据目录

路径：`{relative(ns_root)}`

{dir_inventory(ns_root)}

### 1.2 nuScenes Mini 标注元数据

路径：`{relative(mini_meta)}`

{dir_inventory(mini_meta, 12)}

### 1.3 已定位的预处理索引文件

| 文件名 | 大小 |
|---|---:|
{info_table}

`nuscenes_infos_temporal_train.pkl` 和 `nuscenes_infos_temporal_val.pkl` 是 BEVFormer 数据加载使用的预处理索引文件；它们不等同于完整原始传感器数据。

### 1.4 DriveLM 本地目录检查

检查到的路径（或优先查找位置）：`{relative(drive_root)}`

{dir_inventory(drive_root, 12)}

**注意：** 即使本地存在 DriveLM 文件夹或语言标注 JSON，也不能据此证明已下载完整 DriveLM-v1、完成官方预处理或与 `mini_val` 样本一一对应。

## 二、nuScenes 数据结构与主要标注字段

| 字段或表名 | 中文解释 | 在本项目中的用途 |
|---|---|---|
| `sample` / `token` | 场景帧及其唯一标识符 | 关联图像、标注和预测结果 |
| `sample_data` | 传感器文件路径、时间戳及采集信息 | 定位六路相机图像与 LiDAR 数据 |
| `calibrated_sensor` | 传感器标定参数（相机内外参等） | 从图像坐标理解空间几何关系 |
| `ego_pose` | 自车在全局坐标系中的位姿 | 位姿变换、时序空间对齐 |
| `sample_annotation` | 物体三维标注信息 | 三维目标检测真值数据 |
| `category` | 物体类别信息 | 分类评估及统计 |
| `nuscenes_infos_temporal_*.pkl` | 为 BEVFormer 整理的数据加载索引 | 构建训练集/验证集加载流程 |

上述内容为 nuScenes 的常见数据结构说明；**具体 PKL 内键名必须在本地读取后再逐项确认**，不能因为表里列出字段就说已逐字段验证。

## 三、DriveLM 标注与 BEV 感知的关系

DriveLM 侧重驾驶场景理解和基于场景的自然语言问答（Question-Answer，QA）。与传统 3D 检测标注相比，DriveLM 增加了问题、回答和语言层面的场景信息。语言标注要用于未来的“语言引导 BEV 注意力增强”，必须可靠地匹配到对应相机帧、目标区域或场景标识。

### 当前已验证与尚未验证的内容

- **已完成：** nuScenes Mini 的六路相机和三维检测数据探索，以及 BEVFormer-Small 的 Mini 基线评估。
- **已开展：** DriveLM 示例的结构、QA、场景和标注可视化探索。
- **尚未证实：** 当前所用 DriveLM 示例和 nuScenes Mini 的样本标识一一对应。
- **后续需要：** 获取与感知数据匹配的 DriveLM 样本，明确训练/验证划分，检查官方预处理及评估入口。

所以本文不将 nuScenes Mini 的 mAP/NDS 称为 DriveLM 验证成绩。

## 四、当前已经跑通的数据处理流程

1. 准备 nuScenes Mini 的传感器图像、标注与辅助元数据。
2. 通过预处理索引 PKL 构建 BEVFormer 数据加载器。
3. 加载 BEVFormer-Small 预训练权重，对验证样本进行三维目标检测推理。
4. 保存预测 JSON（`results_nusc.json`）和评估 JSON（`metrics_summary.json`）。
5. 在真实推理期间捕获 Encoder 的 BEV 特征，保存为 `.npy`，再制作强度图和通道图。

## 五、与实习任务书要求的差距

| 官方要求 | 当前情况 | 后续动作 |
|---|---|---|
| DriveLM-v1 完整数据下载与预处理 | 未确认完成 | 核实官方资源、数据规模、下载和预处理流程 |
| DriveLM 验证集基线评估 | 未完成正式验证 | 建立样本匹配与可靠评估方案 |
| 三维检测 mAP | 已得到 nuScenes Mini 实测数据 | 保留为 Mini 结果，不冒充 DriveLM |
| BEV 分割 mIoU | 当前检测配置未产生该指标 | 核对分割标注、模型头及评估脚本 |
| BEV 特征与检测结果可视化 | 已有真实 BEV 特征及检测图 | 在阶段报告中引用并解释局限性 |

**结论：** 当前基线感知链路已经跑通，后续重点是正式数据匹配、分割任务指标和交付材料核对。
"""

    baseline_path = DOCS_DIR / f"stage1_baseline_report_{REPORT_DATE}.md"
    dataset_path = DOCS_DIR / f"stage1_dataset_structure_{REPORT_DATE}.md"
    baseline_path.write_text(baseline_report, encoding="utf-8")
    dataset_path.write_text(dataset_report, encoding="utf-8")

    print("[完成] 中文基线性能报告：", relative(baseline_path))
    print("[完成] 中文数据集说明：", relative(dataset_path))
    print("[来源] 本次选用的指标文件：", relative(metrics_path))
    print("[实测] mAP =", fmt(metrics.get("mean_ap")), "，NDS =", fmt(metrics.get("nd_score")))
    print("[实测] 检测样本数：", sample_count, "，BEV 张量形状：", pretty(capture.get("saved_shape")))
    print("[提醒] 结果属于 nuScenes Mini，不是 DriveLM 正式验证成绩。")


if __name__ == "__main__":
    main()

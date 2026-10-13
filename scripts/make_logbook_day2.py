#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026-10-11（Day 2）中文实习日志一键整理工具。

运行：python scripts/make_logbook_day2.py
输入：项目 outputs/ 中的真实结果，及 BEVFormer/test/ 中的官方评估数据。
输出：docs/images/2026-10-11/ 和 docs/logbook_2026-10-11.md。

"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DOCS = ROOT / "docs"
IMAGE_DIR = DOCS / "images" / "2026-10-11"
LOGBOOK = DOCS / "logbook_2026-10-11.md"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# 原始实验图片保留不变，复制一份到 Markdown 的相对路径下
EXPERIMENT_IMAGES = [
    ("nuscenes_6cameras.png", "02_nuscenes_cameras.png", "nuScenes Mini 六路环视相机图像"),
    ("nuscenes_cam_front_boxes.png", "03_nuscenes_gt_boxes.png", "前视图上的 nuScenes 真实 3D 标注框（Ground Truth）"),
    ("nuscenes_lidar_bev.png", "04_lidar_bev.png", "LiDAR 点云与鸟瞰（BEV）可视化"),
    ("drivelm_multimodal_example.png", "05_drivelm_multimodal.png", "DriveLM 六路图像、2D 标注框和视觉问答"),
    ("bev_attention_front_right.png", "06_attention_mask.png", "规则法生成的右前方 BEV 空间注意力热力图"),
    ("bevformer_predictions_visualized.png", "08_bevformer_predictions.png", "BEVFormer-Small 模型预测：相机 3D 框与 BEV 鸟瞰图"),
]

# 首选 Windows 自带的中文字体：WSL 直接可以访问 Windows C 盘
# 兼容 Ubuntu 的 Noto/WenQuanYi 中文字体，避免截图里的中文显示成 □□□。
CJK_FONT_PATHS = [
    "/mnt/c/Windows/Fonts/msyh.ttc",
    "/mnt/c/Windows/Fonts/msyhbd.ttc",
    "/mnt/c/Windows/Fonts/simhei.ttf",
    "/mnt/c/Windows/Fonts/simsun.ttc",
    "/mnt/c/Windows/Fonts/Deng.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def find_cjk_font():
    """确保生成报告图片所使用的字体具备中文字符。"""
    for name in CJK_FONT_PATHS:
        if not Path(name).is_file():
            continue
        try:
            f = ImageFont.truetype(name, 22)
            # 判断不同汉字是否被渲染成了相同的缺字方框
            if bytes(f.getmask("中")) != bytes(f.getmask("文")):
                return name
        except Exception:
            continue
    raise RuntimeError(
        "找不到支持中文的字体，已停止生成终端风格图片，避免出现方框。\n"
        "请先在 WSL 执行：sudo apt update && sudo apt install -y fonts-noto-cjk\n"
        "然后重新执行本脚本。"
    )


CJK_FONT = find_cjk_font()


def screenshot_style_image(title, lines, filename):
    """把真实输出排版成中文可读的终端风格记录图（不是伪造的终端截图）。"""
    body_font = ImageFont.truetype(CJK_FONT, 23)
    header_font = ImageFont.truetype(CJK_FONT, 27)
    width, left_margin, right_margin = 1510, 30, 35
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_width = width - left_margin - right_margin

    # 按实际字体像素宽度换行，兼容中文比英文宽的情况
    rows = []
    for source_line in lines:
        line = str(source_line).expandtabs(4)
        if not line:
            rows.append("")
            continue
        current = ""
        for char in line:
            if current and dummy.textlength(current + char, font=body_font) > max_width:
                rows.append(current)
                current = char
            else:
                current += char
        if current:
            rows.append(current)
    rows = rows[:50]

    line_height = 37
    height = max(210, 110 + line_height * len(rows))
    image = Image.new("RGB", (width, height), "#101827")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 68), fill="#25344a")
    draw.text((left_margin, 16), title, font=header_font, fill="#ffffff")
    for index, line in enumerate(rows):
        success = any(k in line for k in ("通过", "成功", "True", "81/81", "mAP", "NDS", "Exit code: 0"))
        draw.text((left_margin, 87 + index * line_height), line,
                  font=body_font, fill="#86efac" if success else "#e5e7eb")
    path = IMAGE_DIR / filename
    image.save(path)
    print("[记录图]", path.name)


def environment_lines():
    """现场检测 PyTorch/GPU，并展示真实环境信息。"""
    lines = ["以下为本次运行脚本时的真实设备检查结果", "Python：" + sys.version.split()[0]]
    try:
        import torch
        lines.append("PyTorch：" + str(torch.__version__))
        lines.append("CUDA 可用：" + str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            lines.append("GPU：" + torch.cuda.get_device_name(0))
    except Exception as exc:
        lines.append("无法读取 PyTorch 信息：" + str(exc))
    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if process.returncode == 0 and process.stdout.strip():
            lines.append("nvidia-smi：" + process.stdout.strip().splitlines()[0])
    except Exception as exc:
        lines.append("nvidia-smi 暂不可用：" + str(exc))
    return lines


def find_metrics():
    """从官方评估文件读取数值，不在脚本中写死评估成绩。"""
    folder = ROOT / "BEVFormer" / "test" / "bevformer_small"
    files = list(folder.glob("*/pts_bbox/metrics_summary.json"))
    if not files:
        return None, None, None, None
    path = max(files, key=lambda f: f.stat().st_mtime)
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    predictions_path = path.parent / "results_nusc.json"
    frames = None
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as f:
            frames = len(json.load(f).get("results", {}))
    return path, metrics.get("mean_ap"), metrics.get("nd_score"), frames


def fusion_record():
    """重新执行轻量的模拟特征融合测试，只为记录真实输出；不跑 81 帧模型推理。"""
    script = ROOT / "scripts" / "test_bev_attention_fusion.py"
    if not script.exists():
        print("[跳过] 未找到模拟特征融合测试脚本")
        return
    try:
        result = subprocess.run(
            [sys.executable, str(script)], cwd=ROOT,
            capture_output=True, text=True, timeout=120,
        )
        lines = ["测试类型：模拟 BEV 特征的 GPU 张量融合（非实际模型特征）"]
        lines.extend(result.stdout.splitlines())
        if result.returncode != 0:
            lines.append("测试未通过，以下为错误摘要：")
            lines.extend(result.stderr.splitlines()[-10:])
        lines.append("Exit code: " + str(result.returncode))
        screenshot_style_image("07 — BEV 注意力融合测试记录", lines, "07_attention_fusion.png")
    except Exception as exc:
        print("[跳过] 无法重新获取融合测试结果：", exc)


def inference_record():
    """提取已保存成功运行日志的关键行，保留真实来源。"""
    candidates = [OUTPUTS / "bevformer_mini_compat.log", OUTPUTS / "bevformer_mini_inference.log"]
    log = next((p for p in candidates if p.is_file()), None)
    if log is None:
        print("[跳过] 找不到之前保存的模型推理日志")
        return
    raw = log.read_text(encoding="utf-8", errors="replace")
    raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw).replace("\r", "\n")
    keywords = (
        "[PATCH", "load checkpoint from local path", "81/81", "Results writes",
        "Found detections for", "Saving metrics to:", "mAP:", "NDS:",
    )
    selected = []
    for line in raw.splitlines():
        if any(k in line for k in keywords):
            line = line.strip()
            if len(line) > 210:
                line = line[-210:] if "81/81" in line else line[:210]
            if line not in selected:
                selected.append(line)
    if selected:
        screenshot_style_image(
            "10 — BEVFormer 推理日志摘要（来自真实日志）",
            ["来源：" + str(log.relative_to(ROOT)), *selected[:20]],
            "10_inference_log.png",
        )
    else:
        print("[跳过] 推理日志未包含可提取的关键行")


def figures_markdown():
    ordered = [
        ("01_gpu_environment.png", "CUDA / GPU 开发环境检查"),
        *[(dest, title) for _, dest, title in EXPERIMENT_IMAGES[:5]],
        ("07_attention_fusion.png", "模拟 BEV 特征的 GPU 注意力融合测试"),
        ("08_bevformer_predictions.png", "真实 BEVFormer-Small 3D 检测预测与 BEV 视角"),
        ("09_baseline_metrics.png", "nuScenes Mini 检测基线指标"),
        ("10_inference_log.png", "BEVFormer-Small 推理日志记录"),
    ]
    sections = []
    for name, desc in ordered:
        if (IMAGE_DIR / name).is_file():
            sections.append(f"### {desc}\n\n![{desc}](images/2026-10-11/{name})")
    return "\n\n".join(sections)


def value_text(number):
    return f"{number:.4f}" if isinstance(number, (int, float)) else "未获取"


def main():
    print("========== Day 2 中文实习日志与配图生成 ==========")
    print("[中文字体]", CJK_FONT)
    screenshot_style_image("01 — GPU 环境与 CUDA 验证", environment_lines(), "01_gpu_environment.png")
    for source_name, output_name, _ in EXPERIMENT_IMAGES:
        source = OUTPUTS / source_name
        if source.is_file():
            shutil.copy2(source, IMAGE_DIR / output_name)
            print("[实验图]", output_name)
        else:
            print("[缺少原图，跳过]", source_name)
    fusion_record()
    metrics_file, map_value, nds_value, n_samples = find_metrics()
    if metrics_file is not None:
        metric_lines = [
            "模型：BEVFormer-Small（已加载预训练权重）",
            "数据集：nuScenes v1.0-mini / mini_val",
            "验证样本数：" + (str(n_samples) if n_samples is not None else "未获取"),
            f"mAP：{map_value:.4f}（{map_value * 100:.2f}%）" if isinstance(map_value, (int, float)) else "mAP：未获取",
            f"NDS：{nds_value:.4f}（{nds_value * 100:.2f}%）" if isinstance(nds_value, (int, float)) else "NDS：未获取",
            "注意：CAN Bus 原始数据缺失，本次预处理使用回退向量。",
            "结果用途：nuScenes Mini 开发验证，不是 DriveLM 正式基准成绩。",
            "数据来源：" + str(metrics_file.relative_to(ROOT)),
        ]
        screenshot_style_image("09 — 3D 检测基线评估结果", metric_lines, "09_baseline_metrics.png")
    else:
        print("[跳过] 未找到官方 metrics_summary.json")
    inference_record()

    map_desc = value_text(map_value)
    nds_desc = value_text(nds_value)
    sample_desc = str(n_samples) if n_samples is not None else "未获取"
    gallery = figures_markdown()

    markdown = f"""# 实习日志 · Day 2（2026 年 10 月 11 日）

**项目名称：** 自然语言指令引导的 BEV 感知注意力增强模块研发  
**实验平台：** Windows + WSL2（Ubuntu 22.04）、NVIDIA RTX 5080 Laptop GPU  
**工作主题：** DriveLM 数据探索、语言引导 BEV 注意力原型、BEVFormer-Small 基线推理与评估

---

## 一、今日工作目标

1. 验证 Python、PyTorch、CUDA 与 GPU 开发环境正常运行。
2. 读取和可视化 nuScenes Mini 多摄像头、3D 标注与 LiDAR 点云。
3. 了解 DriveLM-v1 的问答标注结构，完成示例图像与语言信息关联。
4. 建立语言方向到 BEV 空间掩码的概念验证流程。
5. 在 nuScenes Mini 上运行真实 BEVFormer-Small 3D 检测，并记录推理结果、性能指标及遇到的兼容问题。

## 二、具体工作内容

### 2.1 环境恢复与 GPU 验证

重新打开 WSL 中的 `bev_vla` 项目，激活 `.venv-bevformer` 虚拟环境。检查 PyTorch 和 CUDA 的设备访问情况，确认 NVIDIA RTX 5080 能被当前 Python 环境识别。此处的检查用于验证 GPU 能否被调用，并不单独证明 BEVFormer 自定义 CUDA 算子全部兼容。

### 2.2 nuScenes Mini 数据读取与可视化

检查 nuScenes Mini 的 `samples`、`sweeps`、`maps`、`v1.0-mini` 等目录，使用官方 `nuscenes-devkit` 读取元数据。生成以下可视化结果：

- 六路环视摄像头图像拼接图；
- 前视相机上的 3D **真实标注框（Ground Truth）**；
- LiDAR 点云鸟瞰图（BEV 视角）。

三项结果均用于理解多传感器数据结构，不能误认为是 BEVFormer 模型的预测输出。

### 2.3 DriveLM-v1 示例数据与多模态标注分析

获取并读取 DriveLM 官方示例 JSON 和配套的相机图片，确认其中包含 `scene_description`、`key_frames`、`QA`、`image_paths` 和目标二维标注信息。完成六路摄像头、二维框、示例问题与答案的联合可视化。

将 DriveLM 示例中的场景和关键帧 ID 与 nuScenes Mini 比较，本次两份小规模样本未出现直接匹配，说明后续仍需要明确场景对应关系，不能直接把示例当作完整评估集。

### 2.4 语言语义提取与 BEV 注意力掩码原型

编写规则语义提取脚本，处理 **337 条** DriveLM 问答记录，提取关注方向、目标类别与运动状态。修复早期规则法中的方向重复识别（如同时输出 `front`、`front right`、`right`）、把 `ego car` 误当目标以及复数识别的问题。

以 `front right` 为示例，使用人工设定的方向中心与二维高斯函数，生成 **150 × 150** 的空间权重矩阵，并保存为 PNG 和 NPY 文件。随后使用模拟 BEV 张量 `(1, 256, 150, 150)` 在 GPU 上进行广播与逐元素加权融合，验证维度及数值正常。

**实验边界：** 该掩码是人工构造的空间先验，并非 DistilBERT 学习得到的注意力，也尚未融合进真实 BEVFormer 特征。

### 2.5 BEVFormer-Small 数据预处理与模型推理

检查 BEVFormer-Small 配置文件和预训练权重。通过官方数据转换脚本生成 nuScenes Mini 的时序数据索引：

- `nuscenes_infos_temporal_train.pkl`：323 条索引；
- `nuscenes_infos_temporal_val.pkl`：81 条索引。

由于当前缺少官方 CAN Bus 扩展数据，开发测试阶段使用转换器的缺失数据回退方式生成车辆运动向量，并在实验记录中明确标注该限制。

调用单 GPU `torchrun` 及必要的运行时兼容补丁，最终完成 **{sample_desc}/81** 条验证样本的真实 BEVFormer-Small 推理，成功生成 nuScenes 格式的预测 JSON 和检测评估文件。

### 2.6 真实 3D 检测结果可视化

从 `results_nusc.json` 中读取模型输出，在真实前视摄像头画面中投影 3D 预测框，并以自车为中心绘制 BEV 鸟瞰框。调试时先后解决了相机坐标轴自动缩放与透视投影未归一化的问题，最终得到相机视角与 BEV 视角一致展示的结果。

**注意：** 这张图绘制的是模型 Prediction；与前面 nuScenes 数据自带的 Ground Truth 图需要区分。可视化显示的框数还受置信度阈值影响，不能直接当作准确检测数量。

## 三、实验结果与指标

| 实验项目 | 本次结果 | 说明 |
|---|---:|---|
| nuScenes Mini 训练索引 | 323 | 元数据索引样本数 |
| nuScenes Mini 验证索引 | 81 | 元数据索引样本数 |
| 完成推理的验证样本 | {sample_desc} | 以已保存的预测 JSON 为准 |
| 3D 检测 mAP | **{map_desc}** | 开发测试结果 |
| NuScenes Detection Score（NDS） | **{nds_desc}** | 开发测试结果 |
| 语言问答规则解析 | 337 条 | 不等于全部解析准确 |
| BEV 注意力矩阵 | 150 × 150 | 人工生成的空间先验 |
| 模拟 BEV 特征融合 | 通过 | 仅验证张量融合流程 |

**结果说明：** 3D 检测指标来自本次 nuScenes Mini 的评估文件。由于原始 CAN Bus 数据缺失且使用了兼容补丁，以上数字仅用于开发阶段验证，**不是 DriveLM-v1 正式验证集成绩，也不等同于标准 nuScenes 基线复现结果**。本次尚未评估 BEV 分割 mIoU。

## 四、调试过程、问题与解决方案

| 遇到的问题 | 处理方法与结论 |
|---|---|
| 运行官方预处理程序时找不到 `tools` 模块 | 临时设置 `PYTHONPATH`，使项目根目录与工具目录可以导入 |
| CAN Bus 扩展数据缺失 | 使用官方转换器的缺失数据回退机制，仅供开发测试；正式评估仍需真实数据 |
| 普通方式运行 `tools/test.py` 触发断言 | 改为单 GPU `torchrun` 分布式测试入口 |
| DataLoader 报 `cannot pickle 'dict_keys' object` | 将 `data.workers_per_gpu` 临时设为 0 |
| 旧版 MMCV 与 PyTorch 分布式转发接口不兼容 | 在独立脚本中增加单 GPU 运行时兼容处理，不直接修改安装包 |
| MMCV Scatter 传递整数 GPU 编号时报错 | 将设备编号转换为 `torch.device` 类型 |
| 终端出现主机名解析警告 | 作为分布式初始化警告单独观察，未将其误判为本次失败根因 |
| 相机预测框位置错乱或显示为白底 | 开启投影归一化 `normalize=True`，并固定相机像素坐标轴范围 |

## 五、实验图片与运行记录

下列图片来自已生成的真实实验结果；其中标注为“记录图”的图片由真实日志或程序输出重新排版生成，**并非桌面原始截图**。

{gallery}

## 六、当前局限性

1. 目前仅接入了 DriveLM 官方小规模示例，尚未完成完整 DriveLM-v1 数据集对齐和标准评估划分。
2. nuScenes Mini 的 CAN Bus 运动信息不完整，现有评估不能用作论文或项目验收中的正式基线成绩。
3. 语言规则解析尚未进行系统准确率验证，复杂指令和多目标描述仍可能识别错误。
4. BEV 注意力掩码由规则生成，融合测试使用模拟特征，**尚未提取并修改真实 BEVFormer 中间特征**。
5. 本次只完成 3D 检测指标测试，BEV 分割 mIoU 尚未开展。

## 七、下一步工作计划

1. 将真实 BEVFormer 预测与 Ground Truth 进行可视化对比，分析误检及漏检。
2. 提取 BEVFormer 中间 BEV 特征，确认张量排列和空间坐标范围。
3. 将语言引导注意力与真实特征进行小规模融合验证，再探索可训练的文本编码和注意力模块。
4. 补齐必要的数据与标注，对增强前后的检测性能和推理时间进行公平比较。

## 八、今日总结

今天完成了 nuScenes Mini 多传感器数据探索、DriveLM 图文示例关联、337 条问答的规则解析、BEV 空间注意力与模拟特征融合原型，并成功跑通 BEVFormer-Small 在 nuScenes Mini 验证集上的真实 3D 检测推理与可视化。已经建立了可继续开发语言引导注意力模块的工程基础；后续重点是从模拟特征走向真实 BEV 特征，并建立可信的正式评估流程。
"""

    # 中文版需要替换旧的英文正文，因此先做有时间戳的安全备份。
    if LOGBOOK.exists():
        previous = LOGBOOK.read_text(encoding="utf-8")
        if previous != markdown:
            backup_dir = DOCS / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = backup_dir / f"logbook_2026-10-11_before_chinese_{stamp}.md"
            shutil.copy2(LOGBOOK, backup)
            print("[原日志已备份]", backup.relative_to(ROOT))

    LOGBOOK.write_text(markdown, encoding="utf-8")
    print("\n========== 完成 ==========")
    print("中文 Logbook：", LOGBOOK)
    print("实验图片目录：", IMAGE_DIR)
    print("请在 VS Code 中打开 Markdown，按 Ctrl + Shift + V 预览。")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DriveLM v1.1 语言标注与六路相机图像对应性检查（只读数据）。

默认读取：
    data/drivelm/QA_dataset_nus/v1_1_train_nus.json
    data/drivelm/QA_dataset_nus/v1_1_val_nus_q_only.json
    data/drivelm/nuscenes/samples/   （训练图片）
    data/drivelm/val_data/          （验证图片）
输出中文报告和逐图索引，不修改原始数据。
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath

CAMERAS = (
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)


def build_image_index(image_root):
    """按（相机名，图片名）索引，避免跨相机误匹配。"""
    index = defaultdict(list)
    counts = Counter()
    if not image_root.is_dir():
        return index, counts
    for camera in CAMERAS:
        folder = image_root / camera
        if not folder.is_dir():
            continue
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                index[(camera, file.name)].append(file)
                counts[camera] += 1
    return index, counts


def normalized_path(raw):
    """将标注中的斜杠统一为正斜杠，避免 Windows 和 Linux 差异。"""
    return str(raw).replace("\\", "/").strip()


def check_split(label, qa_path, image_root, manifest_path):
    outcome = {
        "split": label, "qa_path": qa_path, "image_root": image_root,
        "scenes": 0, "frames": 0, "references": 0,
        "valid_direct": 0, "valid_filename": 0,
        "missing": 0, "ambiguous": 0, "invalid_paths": 0,
        "frames_with_6": 0, "frames_full": 0, "counter": Counter(),
        "examples_missing": [], "examples_ambiguous": [],
        "problems": [], "sample_tokens": set(),
    }
    if not qa_path.is_file():
        outcome["problems"].append("未找到 JSON：" + str(qa_path))
        return outcome
    if not image_root.is_dir():
        outcome["problems"].append("未找到图像目录：" + str(image_root))
        return outcome

    index, outcome["counter"] = build_image_index(image_root)
    print(f"[{label}] 读取 JSON：{qa_path}", flush=True)
    with qa_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        outcome["problems"].append("JSON 顶层不是 scene_token 字典，请检查格式")
        return outcome

    outcome["scenes"] = len(data)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "数据划分", "scene_token", "frame_token", "相机", "JSON原始路径",
            "实际图片路径", "状态",
        ])
        for scene_token, scene in data.items():
            frames = scene.get("key_frames", {}) if isinstance(scene, dict) else {}
            if not isinstance(frames, dict):
                outcome["problems"].append(f"场景 {scene_token} 的 key_frames 不是字典")
                continue
            for frame_token, frame in frames.items():
                outcome["frames"] += 1
                outcome["sample_tokens"].add(str(frame_token))
                paths = frame.get("image_paths", {}) if isinstance(frame, dict) else {}
                if not isinstance(paths, dict):
                    outcome["invalid_paths"] += 1
                    if len(outcome["examples_missing"]) < 5:
                        outcome["examples_missing"].append(f"{frame_token}: image_paths 不是字典")
                    continue

                all_ok = True
                if all(cam in paths for cam in CAMERAS):
                    outcome["frames_with_6"] += 1
                else:
                    all_ok = False
                for camera in CAMERAS:
                    outcome["references"] += 1
                    raw = paths.get(camera)
                    raw_str = normalized_path(raw) if isinstance(raw, str) else ""
                    if not raw_str:
                        status, actual = "JSON缺少路径", ""
                        outcome["invalid_paths"] += 1
                        all_ok = False
                    else:
                        # 先按文件名 + 相机在当前划分的图像目录中查找；
                        # JSON 常引用 nuscenes/samples，但验证集 ZIP 实际为 val_data。
                        name = PurePosixPath(raw_str).name
                        matches = index.get((camera, name), [])
                        if not matches:
                            status, actual = "图片缺失", ""
                            outcome["missing"] += 1
                            all_ok = False
                        elif len(matches) > 1:
                            status, actual = "同名图片冲突", ""
                            outcome["ambiguous"] += 1
                            all_ok = False
                        else:
                            actual_path = matches[0]
                            actual = str(actual_path)
                            # 仅做路径归一化比较，避免把不存在的 JSON 路径视为实际文件。
                            try:
                                relative = actual_path.relative_to(image_root).as_posix()
                            except ValueError:
                                relative = ""
                            if raw_str == relative or raw_str.endswith("/" + relative):
                                outcome["valid_direct"] += 1
                                status = "已匹配（原始路径后缀）"
                            else:
                                outcome["valid_filename"] += 1
                                status = "已匹配（按相机与文件名重定位）"
                    writer.writerow([label, scene_token, frame_token, camera, raw_str, actual, status])
                    if status != "已匹配（原始路径后缀）" and status != "已匹配（按相机与文件名重定位）":
                        examples = outcome["examples_ambiguous"] if status == "同名图片冲突" else outcome["examples_missing"]
                        if len(examples) < 5:
                            examples.append(f"{frame_token} / {camera}: {raw_str or '无路径'}（{status}）")
                if all_ok:
                    outcome["frames_full"] += 1
    return outcome


def pct(n, d):
    return f"{n / d * 100:.2f}%" if d else "不可计算"


def write_report(results, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DriveLM v1.1 JSON 与相机图像对应性检查",
        "",
        f"检查时间：{datetime.now():%Y-%m-%d %H:%M}",
        "",
        "说明：该报告只验证 DriveLM JSON 的 `image_paths` 能否匹配已经解压的六路相机图片；"
        "不等于已完成 BEVFormer 3D 检测或 BEV 分割评估。",
        "",
        "| 数据集 | 场景 | 关键帧 | 应检查图片 | 成功匹配 | 匹配率 | 六路齐全的帧 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        good = r["valid_direct"] + r["valid_filename"]
        lines.append(
            f"| {r['split']} | {r['scenes']} | {r['frames']} | {r['references']} | "
            f"{good} | {pct(good, r['references'])} | {r['frames_full']} |"
        )
    lines += ["", "## 各数据集详情", ""]
    for r in results:
        lines += [
            f"### {r['split']}", "",
            f"- JSON 路径：`{r['qa_path']}`",
            f"- 图片目录：`{r['image_root']}`",
            f"- 直接路径后缀匹配：{r['valid_direct']}",
            f"- 按相机与文件名匹配（路径结构不一致）：{r['valid_filename']}",
            f"- 图片缺失：{r['missing']}；同名冲突：{r['ambiguous']}；JSON 缺失路径：{r['invalid_paths']}",
            f"- 六路相机字段齐全的帧：{r['frames_with_6']}；六路图片全部找到的帧：{r['frames_full']}",
            "", "图片目录现有文件统计：", "",
        ]
        lines += [f"- `{cam}`：{r['counter'][cam]} 张" for cam in CAMERAS]
        if r["examples_missing"]:
            lines += ["", "缺失或异常示例：", ""]
            lines += [f"- `{s}`" for s in r["examples_missing"]]
        if r["examples_ambiguous"]:
            lines += ["", "同名冲突示例：", ""]
            lines += [f"- `{s}`" for s in r["examples_ambiguous"]]
        if r["problems"]:
            lines += ["", "结构或文件问题：", ""]
            lines += [f"- {s}" for s in r["problems"]]
        lines.append("")
    a, b = results
    intersection = a["sample_tokens"] & b["sample_tokens"]
    lines += [
        "## 训练/验证划分检查", "",
        f"- 两套 JSON 的 frame_token 交集：{len(intersection)} 个。",
        "- 这只是 DriveLM 内部数据划分的检查，不是与 nuScenes Mini 的交集统计。",
        "", "## 下一步", "",
        "1. 如存在缺失图片或不完整关键帧，先检查 JSON 中的文件名与 ZIP 原始目录。",
        "2. 与 BEVFormer 的 nuScenes sample-token 和相机内外参建立对应关系。",
        "3. 仅在拥有所需 3D 检测/分割真值的适当验证集上计算相应指标。",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="检查 DriveLM JSON 与六路相机图片是否匹配")
    parser.add_argument("--root", type=Path, default=None, help="项目根目录，默认脚本上一级目录")
    args = parser.parse_args()
    root = args.root.resolve() if args.root else Path(__file__).resolve().parents[1]
    data_root = root / "data" / "drivelm"
    out = root / "outputs" / "drivelm_check"
    items = (
        ("训练集", data_root / "QA_dataset_nus" / "v1_1_train_nus.json",
         data_root / "nuscenes" / "samples", out / "训练集_图片对应表.csv"),
        ("验证集", data_root / "QA_dataset_nus" / "v1_1_val_nus_q_only.json",
         data_root / "val_data", out / "验证集_图片对应表.csv"),
    )
    results = [check_split(*item) for item in items]
    report = root / "docs" / "drivelm_v11_图片路径匹配检查.md"
    write_report(results, report)
    print("\n========== DriveLM 图片对应性检查 ==========")
    any_error = False
    for r in results:
        good = r["valid_direct"] + r["valid_filename"]
        print(f"[{r['split']}] 关键帧={r['frames']}，匹配图片={good}/{r['references']} ({pct(good, r['references'])})，六路齐全={r['frames_full']}/{r['frames']}")
        if r["missing"] or r["ambiguous"] or r["invalid_paths"] or r["problems"] or not r["frames"]:
            any_error = True
            print(f"  注意：缺失={r['missing']}，冲突={r['ambiguous']}，无效路径={r['invalid_paths']}，结构问题={r['problems']}")
    print("中文报告：", report)
    print("逐图索引：", out)
    print("============================================")
    raise SystemExit(1 if any_error else 0)


if __name__ == "__main__":
    main()

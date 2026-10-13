#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 DriveLM 验证集 × BEVFormer nuScenes Mini 验证集的联合样本索引。

仅读取本地已经准备的数据，不重跑模型、不修改源文件、不生成 Markdown。
输出：JSON、CSV，以及最多两张六路相机预览图（用于当天 Logbook）。
注意：这是样本/预测关联实验，不是 DriveLM 官方检测 mAP 或分割 mIoU。
"""

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageDraw, ImageOps

CAMERAS = (
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
)


def load_drivelm_frames(json_path):
    """DriveLM 官方数据为 scene_token -> key_frames -> frame_token。"""
    with json_path.open("r", encoding="utf-8") as f:
        scenes = json.load(f)
    if not isinstance(scenes, dict):
        raise ValueError("DriveLM JSON 顶层不是场景字典")
    result = {}
    for scene_token, scene in scenes.items():
        key_frames = scene.get("key_frames", {}) if isinstance(scene, dict) else {}
        if not isinstance(key_frames, dict):
            continue
        for token, frame in key_frames.items():
            if isinstance(frame, dict):
                result[str(token)] = {"scene_token": str(scene_token), "frame": frame}
    return result


def load_bev_infos(pkl_path):
    """仅读取用户本地自行预处理的可信 PKL（pickle 不适用于未知来源）。"""
    with pkl_path.open("rb") as f:
        raw = pickle.load(f)
    infos = raw["infos"] if isinstance(raw, dict) else raw
    if not isinstance(infos, (list, tuple)):
        raise ValueError("BEVFormer PKL 中未找到 infos 列表")
    return {str(info["token"]): info for info in infos if isinstance(info, dict) and info.get("token")}


def load_predictions(test_root):
    """选择本机最近修改的 nuScenes 检测结果，不发起新一轮耗时推理。"""
    candidates = list(test_root.rglob("results_nusc.json")) if test_root.is_dir() else []
    if not candidates:
        return {}, None
    selected = max(candidates, key=lambda p: p.stat().st_mtime)
    with selected.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    results = raw.get("results", {}) if isinstance(raw, dict) else {}
    if not isinstance(results, dict):
        raise ValueError("results_nusc.json 的 results 字段不是字典")
    return results, selected


def finite_array(data, expected_shape):
    try:
        array = np.asarray(data, dtype=float)
        if array.shape == expected_shape and np.isfinite(array).all():
            return array.tolist()
    except (TypeError, ValueError):
        pass
    return None


def qa_stats(frame):
    """仅从 DriveLM 验证集提取可用的题目样例，不编造不存在的答案。"""
    qa = frame.get("QA", {})
    result, examples = {}, []
    if not isinstance(qa, dict):
        return result, examples
    for category, items in qa.items():
        if isinstance(items, list):
            result[str(category)] = len(items)
            for item in items:
                if len(examples) >= 3:
                    break
                if isinstance(item, dict):
                    q = item.get("Q", item.get("question", ""))
                    if isinstance(q, str) and q:
                        examples.append({"类型": category, "问题": q[:600]})
        elif isinstance(items, dict):
            result[str(category)] = len(items)
            for item in items.values():
                if len(examples) >= 3:
                    break
                if isinstance(item, dict):
                    q = item.get("Q", item.get("question", ""))
                    if isinstance(q, str) and q:
                        examples.append({"类型": category, "问题": q[:600]})
    return result, examples


def prepare_images(root, image_paths, info):
    """严格用相机名 + 文件名定位图片，避免误读其他划分。"""
    cameras_info = info.get("cams", {})
    if not isinstance(cameras_info, dict) or not isinstance(image_paths, dict):
        raise ValueError("相机元信息或 image_paths 缺失")
    aligned, locations = {}, {}
    for cam_name in CAMERAS:
        dl_path = image_paths.get(cam_name)
        bev_cam = cameras_info.get(cam_name)
        if not isinstance(dl_path, str) or not isinstance(bev_cam, dict):
            raise ValueError(f"{cam_name}: 相机路径或标定信息缺失")
        basename = PurePosixPath(dl_path.replace("\\", "/")).name
        bev_basename = PurePosixPath(str(bev_cam.get("data_path", "")).replace("\\", "/")).name
        if not basename or basename != bev_basename:
            raise ValueError(f"{cam_name}: DriveLM / BEVFormer 图像文件名不一致")
        image_file = root / "data" / "drivelm" / "val_data" / cam_name / basename
        if not image_file.is_file():
            raise FileNotFoundError(f"缺失 DriveLM 图片：{image_file}")
        intrinsic = finite_array(bev_cam.get("cam_intrinsic"), (3, 3))
        rotation = finite_array(bev_cam.get("sensor2lidar_rotation"), (3, 3))
        translation = finite_array(bev_cam.get("sensor2lidar_translation"), (3,))
        if intrinsic is None or rotation is None or translation is None:
            raise ValueError(f"{cam_name}: 相机内外参无效")
        aligned[cam_name] = {
            "图片相对路径": image_file.relative_to(root).as_posix(),
            "相机内参": intrinsic,
            "sensor2lidar_rotation": rotation,
            "sensor2lidar_translation": translation,
        }
        locations[cam_name] = image_file
    return aligned, locations


def make_preview(paths, token, out_path, detections_at_03):
    """将六路相机图片合成 3×2 网格；无编码字符依赖。"""
    cell_w, cell_h, caption_h, margin = 480, 270, 28, 12
    width = 3 * (cell_w + margin) + margin
    height = 2 * (cell_h + caption_h + margin) + margin + 38
    board = Image.new("RGB", (width, height), "#f4f6f9")
    draw = ImageDraw.Draw(board)
    draw.text((margin, 12), f"DriveLM + BEVFormer Mini-val | sample: {token[:18]}... | predictions >=0.3: {detections_at_03}", fill="#202830")
    for i, name in enumerate(CAMERAS):
        x = margin + (i % 3) * (cell_w + margin)
        y = margin + 38 + (i // 3) * (cell_h + caption_h + margin)
        with Image.open(paths[name]) as im:
            tile = ImageOps.contain(im.convert("RGB"), (cell_w, cell_h), method=Image.Resampling.BILINEAR)
        board.paste(tile, (x + (cell_w - tile.width) // 2, y + (cell_h - tile.height) // 2))
        draw.text((x + 4, y + cell_h + 4), name, fill="#202830")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(out_path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description="DriveLM + BEVFormer Mini-val 联合样本构建")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="bev_vla 项目根目录")
    parser.add_argument("--previews", type=int, default=2, help="最多生成几张六路相机预览图（0 禁用）")
    args = parser.parse_args()
    root = args.root.resolve()
    drive_json = root / "data/drivelm/QA_dataset_nus/v1_1_val_nus_q_only.json"
    mini_pkl = root / "BEVFormer/data/nuscenes/nuscenes_infos_temporal_val.pkl"
    for source in (drive_json, mini_pkl):
        if not source.is_file():
            raise FileNotFoundError(source)

    dl_frames = load_drivelm_frames(drive_json)
    mini_infos = load_bev_infos(mini_pkl)
    predictions, pred_file = load_predictions(root / "BEVFormer/test/bevformer_small")
    tokens = sorted(set(dl_frames) & set(mini_infos))
    print(f"[读取] DriveLM 验证帧 {len(dl_frames)}，BEVFormer Mini 验证帧 {len(mini_infos)}")
    print(f"[交集] 共 {len(tokens)} 个共同 sample token（预期此前检查为 13）")
    print(f"[检测结果] {pred_file if pred_file else '未找到，将仅整理图像和标定'}")

    pairs, csv_rows, previews, failures = [], [], [], []
    for token in tokens:
        try:
            dl = dl_frames[token]
            frame = dl["frame"]
            aligned, paths = prepare_images(root, frame.get("image_paths", {}), mini_infos[token])
            questions_per_category, examples = qa_stats(frame)
            preds = predictions.get(token, [])
            if not isinstance(preds, list):
                preds = []
            high = [p for p in preds if isinstance(p, dict) and isinstance(p.get("detection_score"), (int, float))
                    and p["detection_score"] >= 0.3]
            classes = dict(Counter(str(p.get("detection_name", "unknown")) for p in high))
            record = {
                "sample_token": token,
                "DriveLM_scene_token": dl["scene_token"],
                "数据来源": "DriveLM v1.1 val q_only × nuScenes Mini val",
                "相机": aligned,
                "QA分类数量": questions_per_category,
                "问题样例": examples,
                "BEVFormer预测总框数": len(preds),
                "BEVFormer置信度>=0.3的框数": len(high),
                "置信度>=0.3的类别数": classes,
                "有对应BEVFormer预测结果": token in predictions,
            }
            pairs.append(record)
            csv_rows.append({
                "sample_token": token,
                "DriveLM_scene_token": dl["scene_token"],
                "匹配相机数": len(aligned),
                "QA总条目数": sum(questions_per_category.values()),
                "预测是否存在": token in predictions,
                "3D检测框总数": len(preds),
                "置信度>=0.3的检测框数": len(high),
            })
            if len(previews) < max(0, args.previews):
                previews.append((token, paths, len(high)))
        except (FileNotFoundError, ValueError, TypeError, KeyError) as error:
            failures.append({"sample_token": token, "问题": str(error)})
            print(f"[异常] {token}: {error}")

    out_dir = root / "outputs" / "drivelm_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "说明": "仅进行联合数据索引及现有检测预测关联，不能视为 DriveLM 正式验证集 mAP/mIoU",
        "交集样本数": len(tokens),
        "成功构建联合样本数": len(pairs),
        "失败数": len(failures),
        "失败详情": failures,
        "BEVFormer检测结果来源": str(pred_file.relative_to(root)) if pred_file else None,
        "匹配记录": pairs,
    }
    json_out = out_dir / "drivelm_minival_joint_pairs.json"
    csv_out = out_dir / "drivelm_minival_joint_pairs.csv"
    json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]) if csv_rows else ["sample_token", "问题"])
        w.writeheader()
        w.writerows(csv_rows)

    img_dir = root / "docs/images/2026-10-12"
    for i, (token, paths, count) in enumerate(previews, 1):
        path = img_dir / f"drivelm_bevformer_joint_{i:02d}.png"
        make_preview(paths, token, path, count)
        print(f"[图片] {path.relative_to(root)}")

    print("\n========== 联合样本构建结果 ==========")
    print(f"[成功] {len(pairs)} / {len(tokens)} 共同帧，六路图片、相机标定均已核验")
    print(f"[预测] 与现有 BEVFormer 检测结果关联：{sum(p['有对应BEVFormer预测结果'] for p in pairs)} / {len(pairs)}")
    print(f"[保存] {json_out.relative_to(root)}")
    print(f"[保存] {csv_out.relative_to(root)}")
    print("[提醒] 此处没有重新运行模型，也没有计算新的 mAP 或 mIoU，不会创建 Markdown。")
    if failures or not pairs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

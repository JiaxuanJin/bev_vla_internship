#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DriveLM 799 帧：一键 GPU 冒烟测试 + 完整 BEVFormer-Small 独立帧 3D 检测。

使用方法（项目根目录、已激活 .venv-bevformer）：
    python scripts/run_drivelm_799_det_all.py 2>&1 | tee outputs/drivelm_check/det799_run.log

执行流程：先验证样本 0/399/798，再自动推理全部 799 帧。只加载一次模型；
每帧原始框、分数和类别保存为 NPZ，便于后续使用真实 3D GT 计算子集 mAP。
中途退出后原命令重试，会根据已保存的帧文件继续运行。绝不写 Markdown。

注意：由于 DriveLM 验证帧稀疏，此处明确关闭时序历史 BEV，因此结果是
“独立帧检测基线”，不是完整时序模型复现，也不是官方完整 nuScenes Val mAP。
"""

import argparse
import copy
import csv
import gc
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer"
CONFIG = REPO / "projects/configs/bevformer/bevformer_small.py"
PKL = ROOT / "data/drivelm/nuscenes_infos_temporal_drivelm799_independent.pkl"
CKPT = REPO / "ckpts/bevformer_small_epoch_24.pth"
DEFAULT_OUT = ROOT / "outputs/drivelm_check/det799"
CAMERAS = {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}
FALLBACK_CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer",
                    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
FILE_VERSION = 1


def file_sha256(path):
    """用实际文件内容识别配置、权重和数据索引，防止断点续跑混入其他版本。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    """避免突然断电/中断时写出半截 JSON。"""
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def unpack(value):
    """拆开 BEVFormer 的单个测试增强与 MMCV DataContainer。"""
    from mmcv.parallel import DataContainer
    for _ in range(8):
        if isinstance(value, DataContainer):
            value = value.data
        elif isinstance(value, (tuple, list)) and len(value) == 1:
            value = value[0]
        else:
            break
    return value


def to_numpy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def frame_file(folder, index):
    return folder / f"frame_{index:04d}.npz"


def summarise_frame(index, token, boxes, scores, labels, duration, threshold, names):
    """分数阈值只用于展示统计，磁盘 NPZ 始终保存全部原始候选框。"""
    mask = scores >= threshold
    result = {
        "index": int(index), "sample_token": str(token),
        "boxes_total": int(len(scores)), "boxes_above_threshold": int(mask.sum()),
        "seconds": round(float(duration), 4),
        "max_score": round(float(scores.max()), 6) if len(scores) else None,
    }
    result.update({f"count_{name}": int(np.sum(labels[mask] == i))
                   for i, name in enumerate(names)})
    return result


def verify_arrays(boxes, scores, labels, index, count_classes):
    if boxes.ndim != 2 or not (len(boxes) == len(scores) == len(labels)):
        raise ValueError(f"第 {index} 帧框、类别和分数数量不一致: "
                         f"{boxes.shape}, {scores.shape}, {labels.shape}")
    if not (np.isfinite(boxes).all() and np.isfinite(scores).all()):
        raise ValueError(f"第 {index} 帧预测出现 NaN 或 Inf")
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= count_classes):
        raise ValueError(f"第 {index} 帧类别编号超出范围")


def save_prediction(out_file, index, token, boxes, scores, labels, duration):
    """先写 .tmp.npz，完整写入后再重命名。"""
    temp = out_file.with_name(out_file.stem + ".tmp.npz")
    try:
        np.savez_compressed(temp, version=np.array(FILE_VERSION, dtype=np.int32),
                            sample_index=np.array(index, dtype=np.int32),
                            sample_token=np.array(token),
                            boxes_3d=boxes.astype(np.float32),
                            scores_3d=scores.astype(np.float32),
                            labels_3d=labels.astype(np.int64),
                            inference_seconds=np.array(duration, dtype=np.float64))
        os.replace(temp, out_file)
    finally:
        if temp.exists():
            temp.unlink()


def load_prediction(out_file, index, token, threshold, names):
    """续跑时校验已完成的文件，损坏或 Token 不匹配则停止，不默默跳过。"""
    with np.load(out_file, allow_pickle=False) as z:
        required = {"version", "sample_index", "sample_token", "boxes_3d", "scores_3d",
                    "labels_3d", "inference_seconds"}
        if not required.issubset(z.files):
            raise ValueError(f"不完整的缓存: {out_file}")
        if int(z["version"]) != FILE_VERSION or int(z["sample_index"]) != index or str(z["sample_token"]) != token:
            raise ValueError(f"缓存样本标识或格式不一致: {out_file}")
        boxes, scores, labels = z["boxes_3d"], z["scores_3d"], z["labels_3d"]
        duration = float(z["inference_seconds"])
        verify_arrays(boxes, scores, labels, index, len(names))
        return summarise_frame(index, token, boxes, scores, labels, duration, threshold, names)


def save_progress(out_dir, identity, rows, expected, elapsed, complete, error=None):
    """定期保存 JSON + CSV，mAP 暂不能计算，绝不将检出框数量冒充性能指标。"""
    ordered = [rows[i] for i in sorted(rows)]
    frame_seconds = sum(float(r["seconds"]) for r in ordered)
    data = {
        "protocol": "DriveLM 799 keyframes within nuScenes val; independent-frame camera-only BEVFormer-Small",
        "note": "真实 GPU 3D 检测预测；未计算子集 mAP，未使用历史 BEV，不是完整 nuScenes 官方 Val 成绩",
        "identity": identity, "requested_frames": expected,
        "finished_frames": len(ordered), "complete": bool(complete), "error": error,
        "score_threshold_for_display_only": identity["score_threshold"],
        "total_raw_boxes": sum(int(r["boxes_total"]) for r in ordered),
        "total_above_threshold": sum(int(r["boxes_above_threshold"]) for r in ordered),
        "mean_inference_seconds": frame_seconds / len(ordered) if ordered else None,
        "elapsed_wall_seconds": round(float(elapsed), 2),
        "per_frame_npz": "frames/frame_0000.npz ... frames/frame_0798.npz",
    }
    atomic_json(out_dir / "run_summary.json", data)
    csv_target = out_dir / "per_frame.csv"
    temp = csv_target.with_name(csv_target.name + ".tmp")
    if ordered:
        with temp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ordered[0].keys()))
            writer.writeheader()
            writer.writerows(ordered)
    else:
        temp.write_text("index,sample_token\n", encoding="utf-8-sig")
    os.replace(temp, csv_target)
    return data


def main():
    parser = argparse.ArgumentParser(description="DriveLM 799 帧一键测试并完成真实 GPU 3D 检测")
    parser.add_argument("--limit", type=int, default=799, help="本次处理多少帧，默认 799")
    parser.add_argument("--min-score", type=float, default=0.30, help="仅用于检测数量统计")
    parser.add_argument("--save-every", type=int, default=20, help="每完成多少帧更新一次汇总；每帧 NPZ 均立即保存")
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    import torch
    args.checkpoint = args.checkpoint.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，检查当前 .venv-bevformer 环境")
    if not (1 <= args.limit <= 799) or args.save_every <= 0:
        raise ValueError("--limit 需在 1~799，--save-every 必须为正整数")
    if not (0.0 <= args.min_score <= 1.0):
        raise ValueError("--min-score 必须位于 0~1")
    for path in (CONFIG, PKL, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"找不到必需文件：{path}")
    out_dir = args.out.resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    print("[阶段 1/5] 检查输入版本、CUDA 和断点续跑状态", flush=True)
    identity = {
        "format_version": FILE_VERSION,
        "model_config_sha256": file_sha256(CONFIG),
        "checkpoint_sha256": file_sha256(args.checkpoint.resolve()),
        "pkl_sha256": file_sha256(PKL),
        "score_threshold": args.min_score,
        "requested_frames": args.limit,
    }
    identity_file = out_dir / "run_identity.json"
    if identity_file.exists():
        old_identity = json.loads(identity_file.read_text(encoding="utf-8"))
        if old_identity != identity:
            raise ValueError("现有输出目录使用了不同的权重/配置/PKL/阈值/帧数。"
                             "请更换 --out 路径，避免混合不同实验结果。")
        print("[续跑] 发现一致的历史任务，将校验并复用已完成的 NPZ", flush=True)
    else:
        if any(frames_dir.iterdir()):
            raise RuntimeError("输出目录已有预测文件但缺少 run_identity.json。请用新的 --out 目录。")
        atomic_json(identity_file, identity)
    print(f"[GPU] {torch.cuda.get_device_name(0)}", flush=True)

    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    print("[阶段 2/5] 加载原版 BEVFormer 检测配置、数据集、预训练权重（仅一次）", flush=True)
    plugin = importlib.import_module("projects.mmdet3d_plugin")
    if not Path(plugin.__file__).resolve().is_relative_to(REPO.resolve()):
        raise RuntimeError(f"导入了错误的插件仓库: {plugin.__file__}")
    cfg = Config.fromfile(str(CONFIG))
    val_cfg = copy.deepcopy(cfg.data.test)
    val_cfg["ann_file"] = str(PKL)
    val_cfg["test_mode"] = True
    for key in ("samples_per_gpu", "workers_per_gpu"):
        val_cfg.pop(key, None)
    modality = val_cfg.get("modality", {})
    if not modality.get("use_camera", False) or modality.get("use_lidar", True):
        raise RuntimeError("检测配置必须是 6 路相机输入，并关闭 LiDAR 点云输入")
    dataset = build_dataset(val_cfg)
    if len(dataset) != 799:
        raise RuntimeError(f"期望 DriveLM 验证集 799 帧，实际读取 {len(dataset)} 帧")
    tokens = [str(row["token"]) for row in dataset.data_infos[:args.limit]]
    if len(set(tokens)) != args.limit:
        raise RuntimeError("PKL 中发现重复 sample token")
    for i, row in enumerate(dataset.data_infos[:args.limit]):
        if set(row.get("cams", {})) != CAMERAS:
            raise RuntimeError(f"第 {i} 帧缺少六路相机标定")
        if not str(row.get("scene_token", "")).startswith("drive_independent_"):
            raise RuntimeError(f"第 {i} 帧不是独立帧 PKL；禁止错误复用历史 BEV")

    names = list(getattr(cfg, "class_names", []))
    if len(names) != 10:
        names = FALLBACK_CLASSES
    rows = {}
    for i, token in enumerate(tokens):
        path = frame_file(frames_dir, i)
        if path.exists():
            rows[i] = load_prediction(path, i, token, args.min_score, names)
    print(f"[数据] 总帧数=799，本次目标={args.limit}，已有有效缓存={len(rows)}", flush=True)
    if len(rows) == args.limit:
        info = save_progress(out_dir, identity, rows, args.limit, time.monotonic()-started, True)
        print(f"[已完成] {info['finished_frames']}/{args.limit}，无需再次推理", flush=True)
        return

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False  # DriveLM 子集稀疏；禁止历史 BEV 直接传递
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    package = torch.load(str(args.checkpoint.resolve()), map_location="cpu", weights_only=True)
    weights = package.get("state_dict", package)
    if not isinstance(weights, dict):
        raise TypeError("权重文件 state_dict 结构不正确")
    if weights and all(str(k).startswith("module.") for k in weights):
        weights = {k.removeprefix("module."): v for k, v in weights.items()}
    result = model.load_state_dict(weights, strict=True)
    print(f"[权重] {len(weights)} 项全部严格匹配，missing={len(result.missing_keys)}，"
          f"unexpected={len(result.unexpected_keys)}", flush=True)
    del weights, package
    gc.collect()
    model = model.cuda().eval()

    def predict(index):
        """与此前已经实测通过的单帧脚本使用同一个官方 test_pipeline 和 forward 接口。"""
        if index in rows:
            print(f"  [缓存] {index+1}/{args.limit} 已有合格预测，跳过", flush=True)
            return
        info = dataset.data_infos[index]
        token = str(info["token"])
        sample = dataset[index]
        if not isinstance(sample, dict) or "img" not in sample or "img_metas" not in sample:
            raise RuntimeError(f"第 {index} 帧 BEVFormer 数据加载失败")
        img = unpack(sample["img"])
        meta = unpack(sample["img_metas"])
        if not torch.is_tensor(img) or img.ndim != 4 or tuple(img.shape[:2]) != (6, 3):
            raise RuntimeError(f"第 {index} 帧预处理图像形状异常：{getattr(img, 'shape', None)}")
        if not isinstance(meta, dict) or meta.get("sample_idx") != token:
            raise RuntimeError(f"第 {index} 帧元数据与样本 token 不一致")
        if len(meta.get("lidar2img", [])) != 6:
            raise RuntimeError(f"第 {index} 帧缺少 lidar2img 标定")
        img = img.unsqueeze(0).cuda()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model(return_loss=False, rescale=True, img=[img], img_metas=[[meta]])
        torch.cuda.synchronize()
        duration = time.perf_counter() - t0
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 1:
            raise RuntimeError(f"第 {index} 帧模型返回异常：{type(outputs).__name__}")
        prediction = outputs[0].get("pts_bbox") if isinstance(outputs[0], dict) else None
        if not isinstance(prediction, dict):
            raise RuntimeError(f"第 {index} 帧没有 pts_bbox 检测结果")
        for key in ("boxes_3d", "scores_3d", "labels_3d"):
            if key not in prediction:
                raise RuntimeError(f"第 {index} 帧缺少预测字段 {key}")
        boxes = to_numpy(prediction["boxes_3d"].tensor)
        scores = to_numpy(prediction["scores_3d"]).reshape(-1)
        labels = to_numpy(prediction["labels_3d"]).reshape(-1)
        verify_arrays(boxes, scores, labels, index, len(names))
        target = frame_file(frames_dir, index)
        save_prediction(target, index, token, boxes, scores, labels, duration)
        rows[index] = summarise_frame(index, token, boxes, scores, labels, duration, args.min_score, names)
        del img, outputs, prediction, boxes, scores, labels, sample
        print(f"  [完成] {index+1:>3}/{args.limit} | boxes={rows[index]['boxes_total']} | "
              f">={args.min_score:.2f}: {rows[index]['boxes_above_threshold']} | "
              f"GPU={duration:.2f}s", flush=True)

    active_index = -1
    try:
        smoke_indices = sorted({0, args.limit // 2, args.limit - 1})
        print(f"[阶段 3/5] 冒烟测试 GPU 推理：{smoke_indices}", flush=True)
        for i in smoke_indices:
            active_index = i
            predict(i)
        save_progress(out_dir, identity, rows, args.limit, time.monotonic()-started, False)
        print("[冒烟测试通过] 自动开始全部帧的检测，不需要人工确认。", flush=True)

        print(f"[阶段 4/5] 推理 DriveLM 全部 {args.limit} 帧；每 {args.save_every} 帧汇总", flush=True)
        for i in range(args.limit):
            active_index = i
            predict(i)
            if (i + 1) % args.save_every == 0 or (i + 1) == args.limit:
                status = save_progress(out_dir, identity, rows, args.limit,
                                       time.monotonic()-started, False)
                print(f"[进度] 已完成 {status['finished_frames']}/{args.limit}，"
                      f"原始检测框累计 {status['total_raw_boxes']}，中间结果已保存", flush=True)
        print("[阶段 5/5] 汇总全部真实预测文件", flush=True)
        if len(rows) != args.limit:
            raise RuntimeError("有帧尚未完成，拒绝输出 complete=True")
        status = save_progress(out_dir, identity, rows, args.limit,
                               time.monotonic()-started, True)
    except BaseException as exc:
        save_progress(out_dir, identity, rows, args.limit, time.monotonic()-started,
                      False, f"frame={active_index}; {type(exc).__name__}: {exc}")
        print(f"[中断] {len(rows)}/{args.limit} 帧完成。重新运行同一命令将继续未完成的帧。", flush=True)
        raise

    print("\n" + "=" * 60, flush=True)
    print(f"[成功] DriveLM 799 帧真实独立帧 GPU 检测完成：{status['finished_frames']}/{args.limit}", flush=True)
    print(f"[统计] 原始框累计：{status['total_raw_boxes']}；>= {args.min_score}: "
          f"{status['total_above_threshold']}", flush=True)
    print(f"[性能] 单帧 GPU 前向均时约 {status['mean_inference_seconds']:.3f} 秒；"
          f"本次总运行 {status['elapsed_wall_seconds']:.1f} 秒", flush=True)
    print(f"[结果] {out_dir / 'run_summary.json'}", flush=True)
    print(f"[逐帧] {out_dir / 'per_frame.csv'}", flush=True)
    print(f"[原始预测] {frames_dir}/frame_0000.npz ~ frame_0798.npz", flush=True)
    print("[注意] 尚未计算 DriveLM 799 子集 mAP；后续必须结合对应真实 3D GT 单独评估。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[失败] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(limit=15)
        sys.exit(1)

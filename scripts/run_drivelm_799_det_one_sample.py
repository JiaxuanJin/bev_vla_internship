#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DriveLM 799 帧中的单帧 BEVFormer-Small GPU 3D 检测冒烟测试。

在 ~/projects/bev_vla 根目录：
    python scripts/run_drivelm_799_det_one_sample.py --index 0

沿用已通过检查的 independent PKL + 原 BEVFormer test_pipeline。
只做相机输入的独立帧推理，不下载数据、不修改原仓库、不计算 mAP。
输出 JSON + NPZ 到 outputs/drivelm_check/。
"""

import argparse
import copy
import gc
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
OUTPUT = ROOT / "outputs/drivelm_check"
CAMERAS = {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
           "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}


def unpack(value):
    """去除单个测试增强和 MMCV DataContainer 封装，与前序加载测试保持一致。"""
    from mmcv.parallel import DataContainer
    for _ in range(8):
        if isinstance(value, DataContainer):
            value = value.data
        elif isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        else:
            break
    return value


def as_numpy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def main():
    parser = argparse.ArgumentParser(description="DriveLM × BEVFormer 单帧 GPU 3D 检测")
    parser.add_argument("--index", type=int, default=0, help="799 帧索引：0 ~ 798")
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--min-score", type=float, default=0.30, help="仅影响统计展示，不更改模型输出")
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查当前 .venv-bevformer 环境")
    if not (0 <= args.min_score <= 1):
        raise ValueError("--min-score 必须在 0 与 1 之间")
    for path in (CONFIG, PKL, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"必需文件不存在：{path}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # 只加载原始 BEVFormer 检测仓库的插件，不导入 BEVFormer-seg，避免重名模块冲突。
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    print("[1/5] 导入原 BEVFormer 检测插件与配置...", flush=True)
    plugin = importlib.import_module("projects.mmdet3d_plugin")
    if not Path(plugin.__file__).resolve().is_relative_to(REPO.resolve()):
        raise RuntimeError(f"错误地导入了其他仓库插件：{plugin.__file__}")
    cfg = Config.fromfile(str(CONFIG))
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg["ann_file"] = str(PKL)
    dataset_cfg["test_mode"] = True
    dataset_cfg.pop("samples_per_gpu", None)
    dataset_cfg.pop("workers_per_gpu", None)
    modality = dataset_cfg.get("modality", {})
    if not modality.get("use_camera", False) or modality.get("use_lidar", True):
        raise RuntimeError("当前测试只适合 use_camera=True 且 use_lidar=False 的配置")
    dataset = build_dataset(dataset_cfg)
    if len(dataset) != 799:
        raise RuntimeError(f"预期读取 799 帧，实际读取 {len(dataset)} 帧")
    if not (0 <= args.index < len(dataset)):
        raise IndexError(f"--index={args.index} 超出 0~{len(dataset)-1}")

    print(f"[2/5] 读取第 {args.index} 帧六路真实相机图像和标定...", flush=True)
    info = dataset.data_infos[args.index]
    token = str(info["token"])
    if set(info.get("cams", {})) != CAMERAS:
        raise ValueError("PKL 相机通道数量/名称不正确")
    if not str(info.get("scene_token", "")).startswith("drive_independent_"):
        raise ValueError("必须使用独立帧 PKL，避免错误的时序 BEV 复用")
    item = dataset[args.index]
    if not isinstance(item, dict) or "img" not in item or "img_metas" not in item:
        raise RuntimeError(f"数据加载结果异常：{type(item).__name__}")
    img, meta = unpack(item["img"]), unpack(item["img_metas"])
    if not torch.is_tensor(img) or tuple(img.shape[:2]) != (6, 3) or img.ndim != 4:
        raise RuntimeError(f"期望图片输入 (6,3,H,W)，实际 {getattr(img, 'shape', None)}")
    if not isinstance(meta, dict) or meta.get("sample_idx") != token:
        raise RuntimeError("数据元信息与样本 Token 不一致")
    if len(meta.get("lidar2img", [])) != 6:
        raise RuntimeError("图像元信息缺少完整 lidar2img 标定")
    print(f"[数据] token={token}; images={tuple(img.shape)}; 6 路标定存在", flush=True)

    print("[3/5] 构建检测模型并安全加载原版预训练权重...", flush=True)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    # DriveLM 799 帧不连续：明确关闭历史 BEV，作为独立帧检测基线。
    cfg.model.video_test_mode = False
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    # 不自动使用 weights_only=False；若旧权重包含不支持的对象，先核实来源再处理。
    package = torch.load(str(args.checkpoint.resolve()), map_location="cpu", weights_only=True)
    if not isinstance(package, dict):
        raise TypeError("权重文件顶层必须是字典")
    weights = package.get("state_dict", package)
    if not isinstance(weights, dict):
        raise TypeError("权重 state_dict 不是字典")
    if weights and all(str(k).startswith("module.") for k in weights):
        weights = {k.removeprefix("module."): v for k, v in weights.items()}
    loaded = model.load_state_dict(weights, strict=True)
    print(f"[权重] 匹配 {len(weights)} 项；missing={len(loaded.missing_keys)}; "
          f"unexpected={len(loaded.unexpected_keys)}", flush=True)
    del package, weights
    gc.collect()
    model = model.cuda().eval()
    # 来自 dataset 的 img 是 (6,3,H,W)，为模型增加 Batch 维度。
    img = img.unsqueeze(0).cuda()
    print(f"[GPU] {torch.cuda.get_device_name(0)}; img={tuple(img.shape)}", flush=True)

    print("[4/5] 执行独立帧 3D 检测 GPU 推理...", flush=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(return_loss=False, rescale=True, img=[img], img_metas=[[meta]])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise TypeError(f"模型返回结果结构异常：{type(outputs).__name__}")
    if not isinstance(outputs[0], dict) or "pts_bbox" not in outputs[0]:
        raise KeyError(f"未找到 3D 检测 pts_bbox：{outputs[0]}")
    result = outputs[0]["pts_bbox"]
    for key in ("boxes_3d", "scores_3d", "labels_3d"):
        if key not in result:
            raise KeyError(f"3D 检测结果缺少 {key}：{list(result)}")
    boxes = as_numpy(result["boxes_3d"].tensor)
    scores = as_numpy(result["scores_3d"]).reshape(-1)
    labels = as_numpy(result["labels_3d"]).reshape(-1)
    if boxes.ndim != 2 or not (len(boxes) == len(scores) == len(labels)):
        raise ValueError(f"检测张量数量不一致：boxes={boxes.shape}, "
                         f"scores={scores.shape}, labels={labels.shape}")
    if not (np.isfinite(boxes).all() and np.isfinite(scores).all()):
        raise ValueError("检测结果出现 NaN 或 Inf")
    if len(labels) and (labels.min() < 0 or labels.max() >= 10):
        raise ValueError("预测类别编号不在 nuScenes 10 类范围内")
    names = list(getattr(cfg, "class_names", []))
    if len(names) != 10:
        names = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
                 "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
    keep = np.flatnonzero(scores >= args.min_score)
    top_ids = keep[np.argsort(scores[keep])[::-1]][:15]
    top = [
        {"class": names[int(labels[i])], "score": round(float(scores[i]), 6),
         "box_lidar": [round(float(v), 5) for v in boxes[i].tolist()]}
        for i in top_ids
    ]

    print("[5/5] 保存原始检测数组和 JSON 摘要...", flush=True)
    npz_path = OUTPUT / f"drivelm_799_det_sample{args.index}.npz"
    json_path = OUTPUT / f"drivelm_799_det_sample{args.index}.json"
    np.savez_compressed(npz_path, boxes_3d=boxes, scores_3d=scores, labels_3d=labels)
    payload = {
        "dataset": "DriveLM v1.1 val ∩ nuScenes val, independent frame",
        "sample_index": args.index,
        "sample_token": token,
        "official_scene_token": str(info.get("official_scene_token", "")),
        "evaluation_scope": "真实独立帧 GPU 3D 检测，非官方 nuScenes 完整 val 指标",
        "uses_temporal_history": False,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_loaded_strictly": True,
        "image_shape": list(img.shape),
        "inference_seconds": round(elapsed, 3),
        "boxes_total": int(len(scores)),
        "score_threshold_display_only": float(args.min_score),
        "boxes_above_threshold": int(len(keep)),
        "counts_by_class_above_threshold": {
            names[c]: int(np.sum(labels[keep] == c)) for c in range(len(names))
        },
        "highest_confidence_examples": top,
        "raw_arrays_path": str(npz_path),
        "note": "只完成单帧真实预测，未计算 mAP/mIoU；全 799 帧评估需 GT 与子集评估协议。",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n========== DriveLM GPU 3D 检测成功 ==========")
    print(f"样本序号: {args.index}, token: {token}")
    print(f"模型预测: {len(scores)} 个 3D 框；分数 >= {args.min_score}: {len(keep)} 个")
    print(f"独立帧 GPU 用时: {elapsed:.2f} 秒")
    for row in top[:5]:
        print(f"  {row['class']:>20} | score={row['score']:.3f}")
    print(f"[保存] {json_path}")
    print(f"[保存] {npz_path}")
    print("[说明] 这是 DriveLM 真实 GPU 推理，不是 mAP，也未使用连续历史 BEV。", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[失败] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(limit=12)
        sys.exit(1)

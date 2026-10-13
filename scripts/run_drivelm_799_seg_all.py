#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DriveLM 799 帧 BEVFormer-Small BEV 分割：先抽样测试，再全量真实评估。

在 ~/projects/bev_vla 根目录运行：
    python -u scripts/run_drivelm_799_seg_all.py --limit 799

设计原则：
1. 使用 **真实场景 PKL**：分割仓库要查询 nuScenes 的 scene_token 和地图。
2. 禁用历史 BEV：DriveLM 799 帧稀疏分布，不能假定前一条就是连续视频帧。
3. 首先在 GPU 上测试第 0/399/798 帧，再自动完成 799 帧；模型仅构建/加载一次。
4. 统计整套数据的像素交并比，而不是平均单帧 mIoU。
5. 每 20 帧存一次进度；异常中断可重启，继续处理未完成帧。
6. 不训练、不下载、不覆盖 nuScenes Mini 实验，不生成 Markdown。

注意：这些指标称为「DriveLM 799 帧子集 / 独立帧 BEV 分割 mIoU」，
并非完整 nuScenes Val 排行榜，也不能代表 DriveLM 问答任务的准确率。
"""

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
import csv
from pathlib import Path

import numpy as np
import torch

# 复用之前已经在用户机器上成功跑通的输入拆包、预测类别及统计函数。
from run_bev_seg_one_sample import (
    ROOT, SEG_REPO, CONFIG, DEFAULT_CKPT,
    unwrap_single_aug, to_numpy, segmentation_classes, make_figure,
)
from run_bev_seg_minival import (
    CLASS_NAMES, NUM_CLASSES, confusion_of_frame, stats_from_confusion,
)

OUT = ROOT / "outputs/drivelm_check/seg799_independent_eval"
PREVIEW_DIR = ROOT / "docs/images/2026-10-12"
# 注意！分割数据集查询官方场景时不能使用虚构的独立帧 scene_token。
OFFICIAL_PKL = ROOT / "data/drivelm/nuscenes_infos_temporal_drivelm799_officialscene.pkl"
EXPECTED = 799
VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json_atomic(path: Path, data):
    """临时文件写好后再替换，降低中断时进度文件损坏的风险。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def run_identity(checkpoint: Path) -> dict:
    stat = checkpoint.stat()
    return {
        "script_version": VERSION,
        "dataset_pkl": str(OFFICIAL_PKL.resolve()),
        "dataset_sha256": sha256(OFFICIAL_PKL),
        "config_sha256": sha256(CONFIG),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "model_video_test_mode": False,
        "class_names": list(CLASS_NAMES),
        "data_protocol": "DriveLM 799 nuScenes official val frames; sparse independent-frame inference",
    }


def read_progress(path: Path, identity: dict):
    """验证进度身份和顺序；不接受其它实验的缓存冒充本次结果。"""
    if not path.exists():
        return [], np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("identity") != identity:
        raise RuntimeError("已有进度文件来自不同权重/PKL/配置！请不要混用；先检查已有文件后更换输出目录。")
    rows = data.get("per_frame", [])
    if not isinstance(rows, list) or any(row.get("index") != i for i, row in enumerate(rows)):
        raise RuntimeError("已有进度文件样本顺序不连续，不允许自动恢复")
    confusion = np.asarray(data.get("confusion_matrix"), dtype=np.int64)
    if confusion.shape != (NUM_CLASSES, NUM_CLASSES) or np.any(confusion < 0):
        raise RuntimeError("已有进度文件混淆矩阵不合法")
    if int(confusion.sum()) != len(rows) * 200 * 400:
        raise RuntimeError("已有进度像素数与帧数不匹配；请检查进度文件")
    return rows, confusion


def write_progress(identity, confusion, per_frame, dataset_size, target_count, *, error=None):
    per_class, fg, with_bg = stats_from_confusion(confusion)
    complete = len(per_frame) == target_count and target_count == dataset_size
    report = {
        "identity": identity,
        "dataset": "DriveLM v1.1 799 frames / nuScenes Val subset",
        "evaluation_type": "BEVFormer-Small independent-frame BEV map segmentation",
        "important_note": "799帧子集独立帧评估，非完整nuScenes Val、非DriveLM语言问答指标；mIoU不应与作者完整时序/不同数据协议直接比较。",
        "dataset_samples": dataset_size,
        "requested_samples": target_count,
        "evaluated_samples": len(per_frame),
        "complete": complete,
        "error": error,
        "foreground_mIoU": fg,
        "mIoU_including_background": with_bg,
        "classes": per_class,
        "confusion_matrix_gt_rows_pred_cols": confusion.tolist(),
        "aggregation": "global intersection/union over frames (no mean of per-frame IoU)",
        "temporal_policy": "cfg.model.video_test_mode=False and official scene tokens retained for map lookup",
    }
    # 一个文件同时含全部 per-frame 和 confusion，恢复时不会错配。
    save_json_atomic(OUT / "progress.json", {
        "identity": identity,
        "confusion_matrix": confusion.tolist(),
        "per_frame": per_frame,
    })
    save_json_atomic(OUT / "metrics_summary.json", report)

    csv_file = OUT / "per_frame.csv"
    temp_csv = csv_file.with_suffix(".csv.tmp")
    keys = ["index", "sample_token", "scene_token", "foreground_mIoU_single_frame",
            "IoU_background", "IoU_divider", "IoU_ped_crossing", "IoU_boundary", "seconds"]
    with temp_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(per_frame)
    os.replace(temp_csv, csv_file)
    np.save(OUT / "confusion.npy", confusion)
    return report


def unpack_and_check(dataset, index):
    """使用原仓库真实 test_pipeline；核验六路相机、地图真值、官方 scene_token。"""
    item = dataset[index]
    if not isinstance(item, dict) or not {"img", "img_metas", "semantic_indices"}.issubset(item):
        raise RuntimeError(f"第{index}帧缺少必要字段：{list(item) if isinstance(item, dict) else type(item)}")
    img = unwrap_single_aug(item["img"])
    meta = unwrap_single_aug(item["img_metas"])
    gt = np.squeeze(to_numpy(unwrap_single_aug(item["semantic_indices"]))).astype(np.int64)
    if not isinstance(meta, dict) or not torch.is_tensor(img):
        raise TypeError(f"第{index}帧数据类型异常：meta={type(meta)}, img={type(img)}")
    if img.ndim == 4:
        img = img.unsqueeze(0)
    if img.ndim != 5 or img.shape[:2] != (1, 6):
        raise ValueError(f"第{index}帧图像维度异常：{tuple(img.shape)}")
    if gt.shape != (200, 400):
        raise ValueError(f"第{index}帧分割真值必须为(200,400)，实际{gt.shape}")
    if np.any(gt < 0) or np.any(gt >= NUM_CLASSES):
        raise ValueError(f"第{index}帧标签值超出0..3：{np.unique(gt)}，不自动修正类别")
    expected_scene = dataset.data_infos[index].get("official_scene_token")
    if str(meta.get("scene_token")) != str(expected_scene):
        raise ValueError(f"第{index}帧真实场景Token不一致！")
    if str(expected_scene).startswith("drive_independent_"):
        raise ValueError("分割必须用官方场景Token查地图，不能使用独立帧虚拟场景PKL")
    return img, meta, gt


def gpu_prediction(model, item, index):
    img, meta, gt = item
    gpu_img = img.to("cuda:0", non_blocking=False)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, img=[gpu_img], img_metas=[[meta]])
    if not isinstance(result, (list, tuple)) or not result or not isinstance(result[0], dict):
        raise RuntimeError(f"第{index}帧模型输出格式异常")
    logits = result[0].get("seg_preds")
    if logits is None:
        raise RuntimeError(f"第{index}帧模型未输出seg_preds")
    pred, shape = segmentation_classes(logits)
    if pred.shape != gt.shape:
        raise ValueError(f"第{index}帧预测和GT尺寸不一致: {pred.shape} != {gt.shape}")
    del gpu_img, result, logits
    return pred, shape


def main():
    parser = argparse.ArgumentParser(description="DriveLM 799帧独立帧 BEVFormer-Small 分割推理+真实mIoU")
    parser.add_argument("--limit", type=int, default=799, help="处理多少帧，默认799")
    parser.add_argument("--save-every", type=int, default=20, help="每多少帧保存中间结果，默认20")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--smoke-indices", default="0,399,798", help="正式评估前的GPU抽检帧索引；空字符串为跳过")
    parser.add_argument("--preview-indices", default="0,399,798", help="保存真值和预测对比图的帧索引")
    args = parser.parse_args()
    if not (1 <= args.limit <= EXPECTED) or args.save_every < 1:
        raise ValueError("--limit必须在1..799；--save-every必须>=1")
    for file in (CONFIG, OFFICIAL_PKL, args.checkpoint):
        if not file.is_file():
            raise FileNotFoundError(f"缺少必要文件：{file}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请确认进入 .venv-bevformer 环境")
    preview = {int(x.strip()) for x in args.preview_indices.split(",") if x.strip()}
    smoke = {int(x.strip()) for x in args.smoke_indices.split(",") if x.strip()}
    OUT.mkdir(parents=True, exist_ok=True)
    identity = run_identity(args.checkpoint)
    per_frame, confusion = read_progress(OUT / "progress.json", identity)
    if len(per_frame) > args.limit:
        raise ValueError(f"已有{len(per_frame)}帧进度，当前limit={args.limit}较小；请提高limit")

    # 原分割仓库需要 data/nuscenes 相对路径，请切到它的项目根目录。
    os.chdir(SEG_REPO)
    sys.path.insert(0, str(SEG_REPO))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    importlib.import_module("projects.mmdet3d_plugin")

    print("[1/5] 使用真实官方场景PKL，构建DriveLM分割数据集...", flush=True)
    cfg = Config.fromfile(str(CONFIG))
    ds_cfg = copy.deepcopy(cfg.data.test)
    ds_cfg.ann_file = str(OFFICIAL_PKL.resolve())
    ds_cfg.test_mode = True
    ds_cfg.pop("samples_per_gpu", None)
    ds_cfg.pop("workers_per_gpu", None)
    dataset = build_dataset(ds_cfg)
    if len(dataset) != EXPECTED:
        raise RuntimeError(f"分割验证集应有799帧，实际{len(dataset)}帧")
    limit = min(len(dataset), args.limit)
    # 不把稀疏DriveLM帧错误串成连续时序：需要关闭history BEV，而不是伪造scene token。
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    print(f"[数据] {len(dataset)}帧，恢复进度={len(per_frame)}帧，video_test_mode=False", flush=True)

    print("[2/5] 构建GPU模型，严格匹配分割权重（仅加载一次）...", flush=True)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    ckpt = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    match = model.load_state_dict(state, strict=True)
    if match.missing_keys or match.unexpected_keys:
        raise RuntimeError("分割权重匹配异常")
    del ckpt, state
    gc.collect()
    model = model.cuda().eval()
    if model.video_test_mode is not False:
        raise RuntimeError("video_test_mode未成功禁用，停止以防时序串帧")
    print("[权重] 严格匹配成功；分割独立帧模式已经确认", flush=True)

    print("[3/5] GPU抽样测试（不过则不会开始799帧全量评估）...", flush=True)
    for index in sorted(smoke):
        if not 0 <= index < len(dataset):
            raise ValueError(f"抽检样本{index}超出0..798")
        item = unpack_and_check(dataset, index)
        pred, logits_shape = gpu_prediction(model, item, index)
        cm = confusion_of_frame(item[2], pred)
        _, fg, _ = stats_from_confusion(cm)
        print(f"  [抽检通过] index={index} | logits={logits_shape} | IoU={fg:.6f}", flush=True)
        del item, pred

    if preview:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[4/5] 开始完整评估，断点起始={len(per_frame)}/{limit} ...", flush=True)
    start_time = time.monotonic()
    active = None
    try:
        for index in range(len(per_frame), limit):
            active = index
            began = time.monotonic()
            item = unpack_and_check(dataset, index)
            pred, logits_shape = gpu_prediction(model, item, index)
            cm = confusion_of_frame(item[2], pred)
            confusion += cm
            cls, fg, _ = stats_from_confusion(cm)
            per_frame.append({
                "index": index,
                "sample_token": str(dataset.data_infos[index]["token"]),
                "scene_token": str(dataset.data_infos[index]["official_scene_token"]),
                "foreground_mIoU_single_frame": fg,
                **{f"IoU_{x['class_name']}": x["iou"] for x in cls},
                "seconds": round(time.monotonic() - began, 3),
            })
            if index in preview:
                path = PREVIEW_DIR / f"drivelm799_seg_comparison_{index:03d}.png"
                make_figure(item[2], pred, path)
                print(f"  [图片] {path}", flush=True)
            if (index + 1) % args.save_every == 0 or index + 1 == limit:
                report = write_progress(identity, confusion, per_frame, len(dataset), limit)
                print(f"[累计] {index+1}/{limit} | 前景mIoU={report['foreground_mIoU']:.6f} | 已保存", flush=True)
            else:
                print(f"[完成] {index+1}/{limit} | 单帧前景IoU={fg:.6f} | 耗时={per_frame[-1]['seconds']:.2f}s", flush=True)
            del item, pred
    except BaseException as exc:
        write_progress(identity, confusion, per_frame, len(dataset), limit,
                       error=f"index={active}: {type(exc).__name__}: {exc}")
        print(f"[异常] index={active}；已保存完成的{len(per_frame)}帧进度", flush=True)
        raise

    print("[5/5] 统计真实地图标签与模型预测的像素交并比...", flush=True)
    final = write_progress(identity, confusion, per_frame, len(dataset), limit)
    print("=" * 66)
    print(f"[成功] DriveLM BEV分割评估完成：{len(per_frame)}/{len(dataset)}", flush=True)
    for row in final["classes"]:
        print(f"  {row['class_name']}: IoU={row['iou']} GT像素={row['gt_pixels']} 预测像素={row['pred_pixels']}")
    print(f"[实测] 799帧前景mIoU={final['foreground_mIoU']:.6f}")
    print(f"[实测] 799帧含背景mIoU={final['mIoU_including_background']:.6f}")
    print(f"[耗时] GPU评估主体 {time.monotonic() - start_time:.1f}秒")
    print("[结果]", OUT / "metrics_summary.json")
    print("[提醒] 仅DriveLM 799帧子集独立帧分割评估，不是完整nuScenes官方Val成绩，也不是语言问答准确率。")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[失败] {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        sys.exit(1)

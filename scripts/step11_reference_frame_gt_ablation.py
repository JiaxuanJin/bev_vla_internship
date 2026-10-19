#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DriveLM799 参考坐标 GT 消融：将原点与朝向影响分开比较。

前提：已经完成 Step 9，存在 9 帧真实预测和 GT 的 npz 缓存。
只重建 nuScenes 地图 GT，不重新运行 GPU、不修改模型或官方指标。

用法：
    python -u scripts/step11_reference_frame_gt_ablation.py

输出：
    outputs/drivelm_check/root_cause/step11_reference_frame/
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEG_REPO = ROOT / "BEVFormer-seg"
CONFIG = SEG_REPO / "projects/configs/bevformer/bevformer_small_seg.py"
DATA_PKL = (
    ROOT / "data/drivelm/nuscenes_infos_temporal_drivelm799_officialscene.pkl"
)
STEP9_DIR = ROOT / "outputs/drivelm_check/root_cause/step9_drivelm799"
STEP9_JSON = STEP9_DIR / "results.json"
OUT = ROOT / "outputs/drivelm_check/root_cause/step11_reference_frame"
CLASSES = ("background", "divider", "ped_crossing", "boundary")
VARIANTS = (
    "ego_reference",
    "lidar_center_ego_heading",
    "ego_center_lidar_heading",
    "lidar_center_lidar_heading",
)


def file_sha256(path: Path) -> str:
    """流式计算文件 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def confusion_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """计算四类像素混淆矩阵，行 GT、列 Pred。"""
    if gt.shape != pred.shape or gt.shape != (200, 400):
        raise ValueError(f"GT/预测尺寸异常: gt={gt.shape}, pred={pred.shape}")
    if not np.isin(gt, (0, 1, 2, 3)).all():
        raise ValueError(f"GT 中存在未知类别: {np.unique(gt).tolist()}")
    if not np.isin(pred, (0, 1, 2, 3)).all():
        raise ValueError(f"预测中存在未知类别: {np.unique(pred).tolist()}")
    packed = gt.astype(np.int64) * 4 + pred.astype(np.int64)
    return np.bincount(packed.ravel(), minlength=16).reshape(4, 4)


def metrics(cm: np.ndarray) -> dict[str, Any]:
    """根据累计混淆矩阵计算前景 mIoU 和分类指标。"""
    gt_count = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    correct = np.diag(cm)
    union = gt_count + pred_count - correct
    class_iou = [
        float(correct[i] / union[i]) if union[i] else None
        for i in range(4)
    ]
    fg = [score for score in class_iou[1:] if score is not None]
    true_fg = int(gt_count[1:].sum())
    predicted_fg = int(pred_count[1:].sum())
    fg_intersection = int(cm[1:, 1:].sum())
    binary_union = true_fg + predicted_fg - fg_intersection
    return {
        "foreground_miou": float(np.mean(fg)) if fg else None,
        "foreground_binary_iou": (
            fg_intersection / binary_union if binary_union else None
        ),
        "class_ious": dict(zip(CLASSES, class_iou)),
        "gt_foreground_pixels": true_fg,
        "pred_foreground_pixels": predicted_fg,
        "confusion_matrix": cm.tolist(),
    }


def rasterize_vectors(rasterizer: Any, vectors: list[dict]) -> np.ndarray:
    """使用分割仓库原生 RasterizeMapVectors 生成标签。"""
    # RasterizeMapVectors 会原地裁剪 vectors 中的 pts，因此必须深拷贝。
    result = rasterizer({"vectors": copy.deepcopy(vectors)})
    mask = result["semantic_indices"]
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    return np.asarray(mask, dtype=np.int64)


def make_preview(index: int, masks: dict[str, np.ndarray], pred: np.ndarray) -> None:
    """保存原始预测与四种参考框架 GT 的小型对比图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("Prediction",) + VARIANTS
    arrays = (pred,) + tuple(masks[name] for name in VARIANTS)
    fig, axes = plt.subplots(1, 5, figsize=(19, 4), constrained_layout=True)
    for axis, name, data in zip(axes, names, arrays):
        axis.imshow(data, origin="lower", vmin=0, vmax=3, cmap="viridis")
        axis.set_title(name, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(OUT / f"gt_reference_{index:03d}.png", dpi=140)
    plt.close(fig)


def main() -> None:
    """逐样本重建四种地图 GT，与相同的预测比较。"""
    parser = argparse.ArgumentParser(description="DriveLM799 坐标参考 GT 消融")
    parser.add_argument(
        "--preview", default="0,200,700", help="保存预览图的样本索引"
    )
    args = parser.parse_args()
    preview_indices = {
        int(part.strip()) for part in args.preview.split(",") if part.strip()
    }

    for path in (CONFIG, DATA_PKL, STEP9_JSON):
        if not path.is_file():
            raise FileNotFoundError(f"缺少文件: {path}")

    step9 = json.loads(STEP9_JSON.read_text(encoding="utf-8"))
    if step9.get("sample_count") != len(step9.get("rows", [])):
        raise ValueError("Step 9 样本计数与记录数不匹配")
    if step9.get("config_sha256") != file_sha256(CONFIG):
        raise RuntimeError("配置 SHA256 与 Step 9 不符，已停止")
    if step9.get("pkl_sha256") != file_sha256(DATA_PKL):
        raise RuntimeError("PKL SHA256 与 Step 9 不符，已停止")
    print("[通过] Step 9 配置及 PKL 哈希校验", flush=True)

    # 第三方工程使用相对路径，配置与插件必须从该根目录加载。
    os.chdir(SEG_REPO)
    sys.path.insert(0, str(SEG_REPO))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from pyquaternion import Quaternion

    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.pipelines.rasterize import (
        RasterizeMapVectors,
    )

    cfg = Config.fromfile(str(CONFIG))
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.ann_file = str(DATA_PKL.resolve())
    dataset_cfg.test_mode = True
    dataset_cfg.pop("samples_per_gpu", None)
    dataset_cfg.pop("workers_per_gpu", None)
    dataset = build_dataset(dataset_cfg)
    if len(dataset) != 799:
        raise RuntimeError(f"实际数据集只有 {len(dataset)} 帧，预期 799")
    rasterizer = RasterizeMapVectors(map_grid_conf=cfg.map_grid_conf)
    print("[完成] 地图矢量生成器及原始 RasterizeMapVectors 已加载", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    accumulated = {name: np.zeros((4, 4), dtype=np.int64) for name in VARIANTS}
    details = []
    tabular_rows = []

    for record in step9["rows"]:
        index = int(record["index"])
        raw_path = STEP9_DIR / f"raw_sample_{index:03d}.npz"
        if not raw_path.is_file():
            raise FileNotFoundError(f"缺少 Step 9 真实预测缓存: {raw_path}")
        with np.load(raw_path, allow_pickle=False) as saved:
            original_gt = saved["gt"].astype(np.int64)
            pred = saved["pred"].astype(np.int64)

        if str(dataset.data_infos[index]["token"]) != str(record["sample_token"]):
            raise RuntimeError(f"第 {index} 帧 token 与 Step 9 不一致")

        # 官方 dataset.get_data_info 通过 scene → log 查询 location，
        # 不是在原始 PKL 直接查询 location。
        item = dataset.get_data_info(index)
        location = str(item["location"])
        ego_translation = np.asarray(item["ego2global_translation"], dtype=float)
        ego_rotation = Quaternion(item["ego2global_rotation"])
        source_gt = rasterize_vectors(rasterizer, item["vectors"])
        if not np.array_equal(source_gt, original_gt):
            mismatch = int(np.count_nonzero(source_gt != original_gt))
            raise RuntimeError(
                f"第 {index} 帧重建原始 GT 与 Step 9 不一致："
                f"{mismatch} 个像素不同。停止实验，先核对 GT 流程。"
            )

        nusc = dataset.nusc
        sample = nusc.get("sample", str(record["sample_token"]))
        lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        calib = nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])
        lidar_ego_translation = np.asarray(calib["translation"], dtype=float)
        lidar_ego_rotation = Quaternion(calib["rotation"])

        # T_global_lidar = T_global_ego @ T_ego_lidar
        lidar_global_translation = ego_translation + ego_rotation.rotate(
            lidar_ego_translation
        )
        lidar_global_rotation = ego_rotation * lidar_ego_rotation
        pose_specs = {
            "lidar_center_ego_heading": (
                lidar_global_translation,
                ego_rotation.elements,
            ),
            "ego_center_lidar_heading": (
                ego_translation,
                lidar_global_rotation.elements,
            ),
            "lidar_center_lidar_heading": (
                lidar_global_translation,
                lidar_global_rotation.elements,
            ),
        }
        masks = {"ego_reference": source_gt}
        for variant, (translation, rotation) in pose_specs.items():
            vectors = dataset.vector_map.gen_vectorized_samples(
                location, translation, rotation
            )
            masks[variant] = rasterize_vectors(rasterizer, vectors)

        frame_scores = {}
        for variant in VARIANTS:
            cm = confusion_matrix(masks[variant], pred)
            accumulated[variant] += cm
            score = metrics(cm)
            score["gt_pixels_changed_from_original"] = int(
                np.count_nonzero(masks[variant] != source_gt)
            )
            frame_scores[variant] = score
            tabular_rows.append(
                {
                    "index": index,
                    "scene_token": record["scene_token"],
                    "location": location,
                    "variant": variant,
                    "foreground_miou": score["foreground_miou"],
                    "foreground_binary_iou": score["foreground_binary_iou"],
                    "gt_pixels_changed": score["gt_pixels_changed_from_original"],
                }
            )

        # 保存每帧所有候选 GT，供师傅复核；不覆盖原始 GT/预测文件。
        np.savez_compressed(
            OUT / f"gt_reference_{index:03d}.npz", **masks
        )
        if index in preview_indices:
            make_preview(index, masks, pred)
        details.append(
            {
                "index": index,
                "sample_token": record["sample_token"],
                "scene_token": record["scene_token"],
                "location": location,
                "original_gt_reproduction_exact": True,
                "scores": frame_scores,
            }
        )
        compare = ", ".join(
            f"{variant}={frame_scores[variant]['foreground_miou']:.3%}"
            for variant in VARIANTS
        )
        print(f"[{index:03d}] {location}: {compare}", flush=True)

    aggregate = {variant: metrics(accumulated[variant]) for variant in VARIANTS}
    payload = {
        "experiment": "DriveLM799 GT reference-frame ablation",
        "sample_count": len(details),
        "distinct_scenes": len({x["scene_token"] for x in details}),
        "checkpoint_sha256_from_step9": step9["checkpoint_sha256"],
        "config_sha256": step9["config_sha256"],
        "pkl_sha256": step9["pkl_sha256"],
        "original_gt_reproduced_for_all_samples": True,
        "warning": (
            "这是对固定预测重建假设地图 GT 的参考系消融，"
            "只对选定 9 个样本有效；不能把任一候选 GT 视为官方真值，"
            "也不能将候选指标替换 799 帧原始正式成绩。"
        ),
        "aggregate": aggregate,
        "samples": details,
    }
    (OUT / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUT / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tabular_rows[0]))
        writer.writeheader()
        writer.writerows(tabular_rows)

    print("\n===== 9 帧累计前景 mIoU =====", flush=True)
    for variant in VARIANTS:
        score = aggregate[variant]["foreground_miou"]
        print(f"  {variant}: {score:.4%}", flush=True)
    print(f"[保存] {OUT / 'results.json'}", flush=True)
    print(f"[保存] {OUT / 'per_sample.csv'}", flush=True)
    print("[注意] 不改变原始 799 帧 mIoU，也不直接宣称根因已证实", flush=True)


if __name__ == "__main__":
    main()

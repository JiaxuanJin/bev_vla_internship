#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 31：真实模型算子数值对照 + BEV/GT 物理坐标链路诊断。

从项目根目录运行：
    python -u scripts/step31_direct_causal_verification.py

只读原始数据与权重；对比 native CUDA deformable-attention 与 PyTorch
参考实现，并使用源码中的 BevFeatureSlicer 传递人工物理坐标场。
结论区分数值核对、坐标一致性和真正的因果性能验证；不修改正式指标。
"""

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer-seg"
CONFIG_PATH = REPO / "projects/configs/bevformer/bevformer_small_seg.py"
PKL_PATH = (
    ROOT / "outputs/drivelm_check/root_cause/step21_full_val/"
    "nuscenes_infos_temporal_val_full6019_paths_fixed.pkl"
)
CHECKPOINT_PATH = Path("/mnt/g/nus/bevformer_small_seg.pth")
OUTPUT_DIR = ROOT / "outputs/drivelm_check/root_cause/step31_direct_verification"


def save_json(path: Path, record: dict) -> None:
    """以 UTF-8 保存诊断证据。"""
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    """流式计算输入文件哈希，方便复现。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_spatial_chain(config) -> dict:
    """沿原始分割头的真实空间操作传递已知物理坐标 X、Y。"""
    from projects.mmdet3d_plugin.bevformer.dense_heads.bevformer_seg_head import (
        BevFeatureSlicer,
    )

    det = config.model.pts_bbox_head.det_grid_conf
    mapping = config.model.pts_bbox_head.map_grid_conf
    h = int(config.model.pts_bbox_head.bev_h)
    w = int(config.model.pts_bbox_head.bev_w)
    map_h = int(round((mapping["ybound"][1] - mapping["ybound"][0]) / mapping["ybound"][2]))
    map_w = int(round((mapping["xbound"][1] - mapping["xbound"][0]) / mapping["xbound"][2]))

    # Encoder 3D reference_points 采用 X 对应 query 的列、Y 对应行。
    # 计算参照点中心，而不是通过预测 logits 猜测物理位置。
    x_min, x_max = config.point_cloud_range[0], config.point_cloud_range[3]
    y_min, y_max = config.point_cloud_range[1], config.point_cloud_range[4]
    ref_x = torch.linspace(x_min + (x_max - x_min) / (2 * w),
                           x_max - (x_max - x_min) / (2 * w), w)
    ref_y = torch.linspace(y_min + (y_max - y_min) / (2 * h),
                           y_max - (y_max - y_min) / (2 * h), h)
    yy, xx = torch.meshgrid(ref_y, ref_x, indexing="ij")
    encoder_xy = torch.stack((xx, yy), dim=0).unsqueeze(0)

    # 源码中的 rot90(k=-1) + flip(dim=3) 等价于转置。
    source_transformed = torch.flip(
        torch.rot90(encoder_xy, k=-1, dims=[2, 3]), dims=[3]
    )
    assert torch.equal(source_transformed, encoder_xy.transpose(2, 3))

    slicer = BevFeatureSlicer(det, mapping).cpu().eval()
    with torch.inference_mode():
        stock_xy = slicer(source_transformed).squeeze(0).numpy()
        not_transposed_xy = slicer(encoder_xy).squeeze(0).numpy()

    # RasterizeMapVectors -> cv2.polylines：col->X, row->Y。
    map_x_min, _, res_x = mapping["xbound"]
    map_y_min, _, res_y = mapping["ybound"]
    target_x = map_x_min + (np.arange(map_w) + 0.5) * res_x
    target_y = map_y_min + (np.arange(map_h) + 0.5) * res_y
    expected_x = np.broadcast_to(target_x[None, :], (map_h, map_w))
    expected_y = np.broadcast_to(target_y[:, None], (map_h, map_w))

    def metrics(xy):
        return {
            "x_mae_m": float(np.abs(xy[0] - expected_x).mean()),
            "y_mae_m": float(np.abs(xy[1] - expected_y).mean()),
            "xy_mae_m": float((np.abs(xy[0] - expected_x) + np.abs(xy[1] - expected_y)).mean() / 2),
            "center_xy_m": [float(xy[0, map_h // 2, map_w // 2]),
                            float(xy[1, map_h // 2, map_w // 2])],
        }

    result = {
        "test_type": "synthetic_coordinate_ramp_through_real_bev_feature_slicer",
        "reason": "编码器物理 X/Y 查询通过原始 rot90+flip 和真实 Slicer 传递；未使用模型训练后的特征",
        "shape": [map_h, map_w],
        "map_grid_conf": dict(mapping),
        "source_rot90_plus_flip_equals_transpose": True,
        "source_original_transform": metrics(stock_xy),
        "control_no_transform": metrics(not_transposed_xy),
        "important_limitation": (
            "此测试可检测几何约定是否一致，但不能从坐标斜坡单独推断"
            "已训练权重的模型表现，也不能把去转置视为模型修复；"
            "之前真实模型消融已观察到去转置降低 IoU。"
        ),
    }
    print("\n===== A：BEV -> Slicer -> GT 物理坐标 =====", flush=True)
    print("原始空间操作的平均 XY 误差(m):", round(result["source_original_transform"]["xy_mae_m"], 5), flush=True)
    print("不转置对照的平均 XY 误差(m):", round(result["control_no_transform"]["xy_mae_m"], 5), flush=True)
    print("注意：这是已知物理坐标场的几何检查，不是模型 mIoU。", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
        charts = [
            expected_x, stock_xy[0], not_transposed_xy[0],
            expected_y, stock_xy[1], not_transposed_xy[1],
        ]
        captions = ["Expected X (m)", "Source pipeline X (m)", "No-transpose X (m)",
                    "Expected Y (m)", "Source pipeline Y (m)", "No-transpose Y (m)"]
        for axis, field, heading in zip(axes.flat, charts, captions):
            im = axis.imshow(field, origin="upper", cmap="coolwarm")
            axis.set_title(heading)
            fig.colorbar(im, ax=axis, shrink=0.8)
        fig.savefig(OUTPUT_DIR / "spatial_coordinate_audit.png", dpi=130)
        plt.close(fig)
    except Exception as exc:  # 可视化失败不影响严格计算
        result["plot_error"] = str(exc)
    return result


def audit_real_cuda_attention(config, max_checks: int, sample_indices: list[int]) -> dict:
    """真实模型前向中比较 CUDA attention 算子和 PyTorch 参考实现。"""
    import mmcv
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch

    sys.path.insert(0, str(ROOT / "scripts"))
    from run_bev_seg_one_sample import segmentation_classes, to_numpy, unwrap_single_aug

    module_names = [
        "projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention",
        "projects.mmdet3d_plugin.bevformer.modules.temporal_self_attention",
        "projects.mmdet3d_plugin.bevformer.modules.decoder",
    ]
    modules = [importlib.import_module(name) for name in module_names]
    originals = {}
    for module in modules:
        for attr in ("MultiScaleDeformableAttnFunction_fp32", "MultiScaleDeformableAttnFunction_fp16"):
            if hasattr(module, attr):
                originals[(module, attr)] = getattr(module, attr)

    metrics = []
    invocations = [0]
    captured_failures = []
    reference_ab_failures = []

    class ReferenceOnlyFunction:
        """将模型里的融合 CUDA 实现临时替换为 mmcv 的纯 PyTorch 参考实现。"""
        @staticmethod
        def apply(value, spatial_shapes, start_index, locations, weights, im2col_step):
            return multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, locations, weights
            )

    def wrap_function(original_class, precision):
        """只在真实前向的前 N 次调用额外计算参考结果；返回仍用原始 CUDA。"""
        class ComparedFunction:
            @staticmethod
            def apply(value, spatial_shapes, start_index, locations, weights, im2col_step):
                invocations[0] += 1
                native = original_class.apply(
                    value, spatial_shapes, start_index, locations, weights, im2col_step
                )
                if len(metrics) + len(captured_failures) >= max_checks:
                    return native
                try:
                    with torch.no_grad():
                        reference = multi_scale_deformable_attn_pytorch(
                            value, spatial_shapes, locations, weights
                        )
                        error = (native.float() - reference.float()).abs()
                        rms_scale = float(reference.float().square().mean().sqrt().item())
                        metrics.append({
                            "precision": precision,
                            "shape": list(native.shape),
                            "mae": float(error.mean().item()),
                            "max_abs": float(error.max().item()),
                            "relative_mae_to_rms": float(error.mean().item() / max(rms_scale, 1e-8)),
                            "rms_reference": rms_scale,
                            "fraction_abs_gt_1e_minus_3": float((error > 1e-3).float().mean().item()),
                            "all_finite": bool(torch.isfinite(native).all().item() and torch.isfinite(reference).all().item()),
                        })
                except Exception as exc:
                    captured_failures.append(str(exc))
                    if "out of memory" in str(exc).lower():
                        torch.cuda.empty_cache()
                return native
        return ComparedFunction

    dataset_cfg = copy.deepcopy(config.data.test)
    dataset_cfg.ann_file = str(PKL_PATH)
    dataset_cfg.test_mode = True
    dataset_cfg.pop("samples_per_gpu", None)
    dataset_cfg.pop("workers_per_gpu", None)
    dataset = build_dataset(dataset_cfg)

    # 无时序状态的独立帧，确保复跑时注意力输入不受场景状态影响。
    config.model.pretrained = None
    config.model.train_cfg = None
    config.model.video_test_mode = False
    model = build_model(config.model, test_cfg=config.get("test_cfg"))
    ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=True)
    load_report = model.load_state_dict(ckpt["state_dict"], strict=True)
    del ckpt
    print("\n===== B：真实模型 CUDA 与 PyTorch attention =====", flush=True)
    print("Strict checkpoint: missing=", len(load_report.missing_keys),
          "unexpected=", len(load_report.unexpected_keys), flush=True)
    model = model.to("cuda:0").eval()

    compared_wrappers = {
        (module, attr): wrap_function(
            original, "fp16" if attr.endswith("fp16") else "fp32"
        )
        for (module, attr), original in originals.items()
    }

    try:
        for (module, attr), wrapped in compared_wrappers.items():
            setattr(module, attr, wrapped)
        sample_outputs = []
        for index in sample_indices:
            item = dataset[index]
            image = unwrap_single_aug(item["img"])
            meta = unwrap_single_aug(item["img_metas"])
            gt = np.squeeze(to_numpy(unwrap_single_aug(item["semantic_indices"]))).astype(np.uint8)
            if image.ndim == 4:
                image = image.unsqueeze(0)
            if image.ndim != 5 or image.shape[1] != 6:
                raise ValueError(f"第 {index} 帧六路相机形状异常：{tuple(image.shape)}")
            start = time.monotonic()
            with torch.inference_mode():
                output = model(
                    return_loss=False, rescale=True,
                    img=[image.cuda().clone()], img_metas=[[copy.deepcopy(meta)]],
                )
            prediction, _ = segmentation_classes(output[0]["seg_preds"])
            if prediction.shape != gt.shape:
                raise RuntimeError("预测与 GT 尺寸不同")
            ious = []
            for class_id in (1, 2, 3):
                tp = np.logical_and(gt == class_id, prediction == class_id).sum()
                union = np.logical_or(gt == class_id, prediction == class_id).sum()
                ious.append(float(tp / union) if union else 0.0)
            # 与上面相同的真实输入和权重，只把注意力融合 CUDA 算子
            # 替换为 mmcv 提供的 Python/torch 参考计算，实现因果配对实验。
            reference_result = {"completed": False, "error": None}
            for (module, attr) in originals:
                setattr(module, attr, ReferenceOnlyFunction)
            try:
                with torch.inference_mode():
                    ref_output = model(
                        return_loss=False, rescale=True,
                        img=[image.cuda().clone()],
                        img_metas=[[copy.deepcopy(meta)]],
                    )
                ref_prediction, _ = segmentation_classes(ref_output[0]["seg_preds"])
                ref_iou = []
                for class_id in (1, 2, 3):
                    tp_ref = np.logical_and(
                        gt == class_id, ref_prediction == class_id
                    ).sum()
                    union_ref = np.logical_or(
                        gt == class_id, ref_prediction == class_id
                    ).sum()
                    ref_iou.append(float(tp_ref / union_ref) if union_ref else 0.0)
                reference_result.update({
                    "completed": True,
                    "reference_miou": float(np.mean(ref_iou)),
                    "reference_class_ious": ref_iou,
                    "changed_pixels": int(np.count_nonzero(prediction != ref_prediction)),
                    "pixel_change_fraction": float(np.mean(prediction != ref_prediction)),
                    "reference_minus_native_miou_percentage_points": float(
                        (np.mean(ref_iou) - np.mean(ious)) * 100
                    ),
                })
                print(
                    f"  PyTorch reference: {np.mean(ref_iou):.3%}, "
                    f"像素变化比例 {np.mean(prediction != ref_prediction):.4%}",
                    flush=True,
                )
            except Exception as exc:
                reference_result["error"] = str(exc)
                reference_ab_failures.append(f"index={index}: {exc}")
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                print("  参考算子全模型 A/B 未成功:", exc, flush=True)
            finally:
                for (module, attr), wrapped in compared_wrappers.items():
                    setattr(module, attr, wrapped)

            sample_outputs.append({
                "index": index,
                "sample_token": dataset.data_infos[index]["token"],
                "scene_token": dataset.data_infos[index]["scene_token"],
                "foreground_miou_single_frame": float(np.mean(ious)),
                "foreground_iou_per_class": ious,
                "latency_s": round(time.monotonic() - start, 2),
                "reference_attention_ab": reference_result,
            })
            print(f"样本 {index}: mIoU={np.mean(ious):.3%}, "
                  f"累计关注算子={len(metrics)}, 总调用={invocations[0]}", flush=True)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
                for ax, data, caption in zip(axes, (gt, prediction), ("GT", "Prediction")):
                    ax.imshow(data, vmin=0, vmax=3, cmap="tab10", interpolation="nearest")
                    ax.set_title(caption)
                figure.tight_layout()
                figure.savefig(OUTPUT_DIR / f"sample_{index:04d}.png", dpi=130)
                plt.close(figure)
            except Exception as exc:
                print("可视化保存失败，不影响指标:", exc, flush=True)
    finally:
        # 恢复函数别名，保持当前 Python 进程的模块定义不变。
        for (module, attr), original in originals.items():
            setattr(module, attr, original)

    return {
        "test_type": "real_model_attention_cuda_vs_reference",
        "note": "使用真实 6 路图像和已发布权重；在同一次前向对照算子参考值，模型实际预测仍来自原 CUDA 算子",
        "environment": {"torch": torch.__version__, "mmcv": mmcv.__version__, "gpu": torch.cuda.get_device_name(0)},
        "sample_indices": sample_indices,
        "samples": sample_outputs,
        "native_operator_invocations": invocations[0],
        "comparisons": metrics,
        "reference_failures": captured_failures,
        "full_model_reference_ab_failures": reference_ab_failures,
        "max_relative_mae_to_rms": max((entry["relative_mae_to_rms"] for entry in metrics), default=None),
        "limitation": "若 CUDA 与 PyTorch 一致，只能说明本次实际采样的注意力调用数值正常，不能排除所有环境或坐标因素",
    }


def main() -> None:
    """运行一轮两类聚焦对照，并写出 JSON / PNG / 日志。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 3000])
    parser.add_argument("--max-ops", type=int, default=4)
    parser.add_argument("--spatial-only", action="store_true", help="仅执行 CPU 物理坐标链，跳过 GPU 前向")
    args = parser.parse_args()
    if args.max_ops < 1 or any(idx < 0 or idx >= 6019 for idx in args.indices):
        parser.error("--max-ops 需 >=1，index 必须属于 [0, 6018]")
    for path in (CONFIG_PATH, PKL_PATH, CHECKPOINT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"关键文件不存在：{path}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    importlib.import_module("projects.mmdet3d_plugin")
    from mmcv import Config
    config = Config.fromfile(str(CONFIG_PATH))

    record = {
        "experiment": "Step31 direct real-model numerical and spatial coordinate verification",
        "input_identity": {"checkpoint_sha256": sha256(CHECKPOINT_PATH),
                           "val_pkl_sha256": sha256(PKL_PATH)},
        "official_6019_fg_miou_baseline": 0.046416,
        "spatial": audit_spatial_chain(config),
    }
    save_json(OUTPUT_DIR / "results.json", record)
    if not args.spatial_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用")
        record["attention"] = audit_real_cuda_attention(config, args.max_ops, args.indices)
        save_json(OUTPUT_DIR / "results.json", record)
        print("\n===== 真实注意力算子数值比较 =====", flush=True)
        print("比较次数:", len(record["attention"]["comparisons"]), flush=True)
        print("最大相对 MAE:", record["attention"]["max_relative_mae_to_rms"], flush=True)
        print("参考实现错误:", record["attention"]["reference_failures"], flush=True)
        for sample in record["attention"]["samples"]:
            print(f"参考替换 A/B 样本 {sample['index']}: "
                  f"{sample['reference_attention_ab']}", flush=True)
    print("\n===== Step 31 完成 =====", flush=True)
    print("结果:", OUTPUT_DIR / "results.json", flush=True)
    print("重要：物理坐标斜坡或算子差异并不自动等于模型低 IoU 根因。", flush=True)


if __name__ == "__main__":
    main()

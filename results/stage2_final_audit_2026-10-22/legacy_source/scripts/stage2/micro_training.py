"""真实统一系统的固定协议微型优化；只写新增实验产物，不改作者或 GT。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import copy
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .joint_runtime import _first_tensor
from .runtime_memory import memory_event, memory_stage, memory_snapshot, release_memory
from .multitask_training import LossConfig, Stage2MultiTaskLoss
from .run_step70_joint_integration import sha256

TEXTS = ("Find lane markings", "Locate pedestrian crossings", "Find road boundaries")
MODES = ("original", "untrained", "trained", "shuffled")
CLASSES = ("Background", "Divider", "Ped Crossing", "Boundary")


def write_json(path: Path, value: Any) -> None:
    """排他写入，避免覆盖任何历史产物。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_hash(value: Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def module_hash(module: nn.Module) -> str:
    """包括参数和缓冲区，检查冻结网络在训练前后逐位一致。"""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def confusion(gt: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    if gt.shape != prediction.shape or gt.size == 0:
        raise ValueError("预测和 GT 必须形状相同且非空，禁止 resize")
    if any(np.any((x < 0) | (x > 3)) for x in (gt, prediction)):
        raise ValueError("类别必须在 0～3")
    return np.bincount(
        4 * gt.astype(np.int64).ravel() + prediction.astype(np.int64).ravel(),
        minlength=16,
    ).reshape(4, 4)


def segmentation_metrics(matrix: np.ndarray) -> dict[str, Any]:
    """按汇总混淆矩阵计算；无并集类别返回 null，不伪造满分。"""
    tp = np.diag(matrix).astype(float)
    union = matrix.sum(0) + matrix.sum(1) - tp
    ious = [float(tp[c] / union[c]) if union[c] else None for c in range(4)]
    foreground = [x for x in ious[1:] if x is not None]
    present = [x for x in ious if x is not None]
    background = int(matrix[0].sum())
    return {
        "class_iou": dict(zip(CLASSES, ious)),
        "foreground_miou": float(np.mean(foreground)) if foreground else None,
        "four_class_miou": float(np.mean(present)) if present else None,
        "background_false_positive_rate": (
            float(matrix[0, 1:].sum() / background) if background else None
        ),
        "confusion_matrix_gt_rows_pred_columns": matrix.tolist(),
        "status": "DIAGNOSTIC_SAME_AUTHOR_RASTER_PROTOCOL",
        "physical_alignment": "NOT_VERIFIED",
    }


def detection_counts(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
) -> dict[str, Any]:
    """固定阈值、同类、按置信度贪心的一对一 XY 中心匹配；非官方 AP。"""
    matched: set[int] = set()
    tp, distances = 0, []
    valid_gt = (
        (gt_labels >= 0)
        & (np.abs(gt_boxes[:, 0]) <= 51.2)
        & (np.abs(gt_boxes[:, 1]) <= 51.2)
    )
    gt_boxes, gt_labels = gt_boxes[valid_gt], gt_labels[valid_gt]
    selected = np.flatnonzero(
        (scores >= 0.1) & (np.abs(boxes[:, 0]) <= 51.2) & (np.abs(boxes[:, 1]) <= 51.2)
    )
    for index in selected[np.argsort(-scores[selected], kind="stable")]:
        candidates = [
            i
            for i in range(len(gt_boxes))
            if i not in matched and gt_labels[i] == labels[index]
        ]
        if not candidates:
            continue
        dist = np.linalg.norm(gt_boxes[candidates, :2] - boxes[index, :2], axis=1)
        best = int(np.argmin(dist))
        if dist[best] <= 2.0:
            matched.add(candidates[best])
            tp += 1
            distances.append(float(dist[best]))
    return {
        "tp": tp,
        "fp": int(len(selected)) - tp,
        "fn": int(len(gt_boxes)) - tp,
        "matched_xy_error_sum_m": sum(distances),
    }


def parameter_stats(
    module: nn.Module, before: dict[str, Tensor]
) -> dict[str, dict[str, Any]]:
    result = {}
    for name, value in module.named_parameters():
        delta = value.detach().cpu() - before[name].cpu()
        grad = value.grad
        finite = bool(torch.isfinite(value).all()) and (
            grad is None or bool(torch.isfinite(grad).all())
        )
        if not finite:
            raise RuntimeError(f"参数或梯度出现 NaN/Inf：{name}")
        result[name] = {
            "numel": value.numel(),
            "value_l2": float(value.detach().norm()),
            "delta_l2": float(delta.norm()),
            "delta_max_abs": float(delta.abs().max()),
            "changed_elements": int(torch.count_nonzero(delta)),
            "gradient_l2": float(grad.detach().norm()) if grad is not None else None,
            "finite": finite,
        }
    return result


def memory_stats() -> dict[str, float]:
    torch.cuda.synchronize()
    return memory_snapshot()


def get_batch(dataset: Any, index: int, seed: int) -> dict[str, Any]:
    from mmcv.parallel import collate

    seed_all(seed)
    # 直接 prepare，禁止 __getitem__ 在空标签时静默替换另一个样本。
    sample = (
        dataset.prepare_test_data(index)
        if dataset.test_mode
        else dataset.prepare_train_data(index)
    )
    if sample is None:
        raise RuntimeError(f"固定样本 {index} 无有效 GT；禁止替换验证/训练样本")
    return collate([sample], samples_per_gpu=1)


def real_training_forward(
    model: Any,
    wrapped: Any,
    pipeline: Any,
    data: dict[str, Any],
    text: str,
    target_class: int,
    audit_gradients: bool = False,
) -> tuple[Tensor, dict[str, Any], dict[str, Any]]:
    """三项损失严格来自一次作者双 Head 前向，保留新增模块计算图。"""
    if not all(k in data for k in ("semantic_indices", "gt_bboxes_3d", "gt_labels_3d")):
        raise ValueError("缺少真实 3D 或地图 GT")
    raw_gt = _first_tensor(data["semantic_indices"])
    gt_hash = tensor_hash(raw_gt)
    if raw_gt.numel() != 200 * 400:
        raise ValueError("地图 GT 必须为 200×400；禁止修改尺寸")
    gt = raw_gt.reshape(1, 200, 400).long().cuda()
    captured: dict[str, Tensor] = {}
    calls = 0

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        nonlocal calls
        calls += 1
        captured.update(output)

    head = model.pts_bbox_head
    handle = head.register_forward_hook(capture)
    try:
        with pipeline.inject_shared_bev(
            head.transformer,
            [text],
            capture_details=True,
        ) as state:
            author_losses = wrapped(return_loss=True, **forward_payload(data))
            if state["hook_calls"] != 1 or calls != 1:
                raise RuntimeError("必须为单次共享 BEV 和双 Head 前向")
            detection = {
                k: v
                for k, v in author_losses.items()
                if (k.startswith("loss") or ".loss" in k) and "seg" not in k.lower()
            }
            if not any("cls" in k for k in detection) or not any(
                "bbox" in k for k in detection
            ):
                raise RuntimeError("缺少作者真实检测 cls/bbox loss")
            attention = state["details"]["attention"]
            attention_logits = head.feat_cropper(
                torch.flip(
                    torch.logit(attention.clamp(1e-5, 1 - 1e-5)).transpose(2, 3),
                    dims=[2],
                )
            )
            total, parts = Stage2MultiTaskLoss(LossConfig())(
                detection,
                captured["seg_preds"],
                gt,
                attention_logits,
                torch.tensor([target_class], device="cuda", dtype=torch.long),
            )
            if tensor_hash(raw_gt) != gt_hash:
                raise RuntimeError("GT 张量被修改")
            details = {
                "shapes": state["trace"],
                "hook_calls": state["hook_calls"],
                "head_calls": calls,
                "gt_sha256": gt_hash,
                "gt_class_counts": {str(c): int((gt == c).sum()) for c in range(4)},
                "attention_target_present": bool((gt == target_class).any()),
                "author_loss_keys": list(author_losses),
                "author_weighted_seg_loss": float(author_losses["loss_seg"].detach()),
            }
            if audit_gradients:
                from torch.nn import functional as F
                from .joint_runtime import _gradient_stats
                from .multitask_training import (
                    aggregate_detection_losses,
                    supervised_attention_loss,
                )

                parameters = list(pipeline.fusion.named_parameters())
                terms = {
                    "detection": aggregate_detection_losses(detection),
                    "segmentation": F.cross_entropy(captured["seg_preds"], gt),
                    "attention": supervised_attention_loss(
                        attention_logits,
                        gt,
                        torch.tensor([target_class], device="cuda", dtype=torch.long),
                        LossConfig(),
                    ),
                }
                details["task_gradients"] = {
                    name: _gradient_stats(term, parameters)
                    for name, term in terms.items()
                    if name != "attention" or details["attention_target_present"]
                }
                if not details["attention_target_present"]:
                    details["task_gradients"]["attention"] = {
                        "status": "TARGET_ABSENT_ZERO_LOSS"
                    }
            return total, {k: float(v) for k, v in parts.items()}, details
    finally:
        handle.remove()
        captured.clear()


def _sample_record(dataset: Any, index: int, text: str) -> dict[str, Any]:
    info = dataset.data_infos[index]
    sample = dataset.nusc.get("sample", info["token"])
    scene = dataset.nusc.get("scene", sample["scene_token"])
    return {
        "index": index,
        "token": info["token"],
        "scene_token": info["scene_token"],
        "scene_name": scene["name"],
        "text": text,
        "cameras": {
            k: str(Path(v["data_path"]).resolve()) for k, v in info["cams"].items()
        },
    }


def forward_payload(data, inference=False):
    # Author inference mutates can_bus metadata. Images and GT need no copy;
    # MMDataParallel scatters them without in-place changes. Preserve GT hash checks.
    result = {
        k: v
        for k, v in data.items()
        if not inference
        or k not in ("semantic_indices", "gt_bboxes_3d", "gt_labels_3d")
    }
    if "img_metas" in result:
        result["img_metas"] = copy.deepcopy(result["img_metas"])
    return result


def protected_files(args, cfg):
    root = Path(cfg.data.train.data_root).resolve()
    files = {
        args.joint_checkpoint.resolve(),
        args.model / "model.safetensors",
        Path(cfg.data.train.ann_file).resolve(),
        Path(cfg.data.test.ann_file).resolve(),
        *root.glob("v1.0-mini/*.json"),
        *root.glob("v1.0-trainval/*.json"),
        *root.glob("maps/**/*.json"),
        *args.author_root.resolve().glob("projects/**/*.py"),
    }
    return {str(p): sha256(p) for p in sorted(files) if p.is_file()}


def evaluate(
    model: Any,
    wrapped: Any,
    pipeline: Any,
    dataset: Any,
    protocol: dict[str, Any],
    initial: dict[str, Tensor],
    output: Path,
) -> dict[str, Any]:
    """更新全部结束并保存最终权重后才读取验证预测；不据结果调参。"""
    trained = {
        k: v.detach().cpu().clone() for k, v in pipeline.fusion.state_dict().items()
    }
    matrices = {mode: np.zeros((4, 4), dtype=np.int64) for mode in MODES}
    counts = {
        mode: {"tp": 0, "fp": 0, "fn": 0, "matched_xy_error_sum_m": 0.0}
        for mode in MODES
    }
    records = []
    for frame in protocol["validation"]:
        index, text = frame["index"], frame["text"]
        data = get_batch(dataset, index, protocol["seed"] + 1000 + index)
        raw_gt = _first_tensor(data["semantic_indices"])
        gt_hash = tensor_hash(raw_gt)
        gt = raw_gt.reshape(200, 400).cpu().numpy().astype(np.uint8)
        annotation = dataset.get_ann_info(index)
        gt_boxes = annotation["gt_bboxes_3d"].tensor.cpu().numpy()
        gt_labels = np.asarray(annotation["gt_labels_3d"])
        payload = {
            "gt": gt,
            "token": np.array(frame["token"]),
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
        }
        predictions: dict[str, dict[str, np.ndarray]] = {}
        for mode in MODES:
            pipeline.fusion.load_state_dict(initial if mode == "untrained" else trained)
            model.prev_frame_info = {
                "prev_bev": None,
                "scene_token": None,
                "prev_pos": 0,
                "prev_angle": 0,
            }
            inference = forward_payload(data, inference=True)
            captured: dict[str, np.ndarray] = {}

            def capture(_module: Any, _inputs: Any, outputs: Any) -> None:
                for key in ("seg_preds", "all_cls_scores", "all_bbox_preds"):
                    value = outputs[key]
                    if not torch.isfinite(value).all():
                        raise RuntimeError(f"验证预测非有限：{key}")
                    captured[key] = value.detach().float().cpu().numpy().copy()

            handle = model.pts_bbox_head.register_forward_hook(capture)
            used_text = " ".join(reversed(text.split())) if mode == "shuffled" else text
            context = (
                nullcontext(None)
                if mode == "original"
                else pipeline.inject_shared_bev(
                    model.pts_bbox_head.transformer,
                    [used_text],
                    strict=mode != "shuffled",
                    capture_details=True,
                )
            )
            try:
                with torch.inference_mode(), context as state:
                    result = wrapped(return_loss=False, rescale=True, **inference)[0]
                    if state is not None:
                        if state["hook_calls"] != 1:
                            raise RuntimeError("验证单帧 Hook 次数异常")
                        att = state["details"]["attention"]
                        projected = model.pts_bbox_head.feat_cropper(
                            torch.flip(att.transpose(2, 3), dims=[2])
                        )
                        payload[mode + "_attention"] = att[0, 0].cpu().numpy()
                        payload[mode + "_attention_map_grid"] = (
                            projected[0, 0].cpu().numpy()
                        )
            finally:
                handle.remove()
            seg = captured["seg_preds"][0]
            pred = seg.argmax(0).astype(np.uint8)
            matrices[mode] += confusion(gt, pred)
            det = result["pts_bbox"]
            boxes = det["boxes_3d"].tensor.cpu().numpy()
            scores = det["scores_3d"].cpu().numpy()
            labels = det["labels_3d"].cpu().numpy()
            local_counts = detection_counts(boxes, scores, labels, gt_boxes, gt_labels)
            for key, value in local_counts.items():
                counts[mode][key] += value
            payload.update(
                {
                    mode: pred,
                    mode + "_logits": seg,
                    mode + "_boxes": boxes,
                    mode + "_scores": scores,
                    mode + "_labels": labels,
                }
            )
            predictions[mode] = captured
            print(f"[A/B] val={index} mode={mode} completed", flush=True)
        zero_deltas = {
            k: float(
                np.abs(predictions["original"][k] - predictions["untrained"][k]).max()
            )
            for k in predictions["original"]
        }
        if any(v != 0 for v in zero_deltas.values()):
            raise RuntimeError("零残差统一系统与作者原模型不一致")
        if tensor_hash(raw_gt) != gt_hash:
            raise RuntimeError("验证 GT 被修改")
        path = output / f"val_{index:03d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **payload)
        records.append(
            frame
            | {
                "artifact": path.name,
                "sha256": sha256(path),
                "gt_sha256": gt_hash,
                "zero_init_max_abs": zero_deltas,
                "changed_pixels_trained_vs_original": int(
                    np.count_nonzero(payload["trained"] != payload["original"])
                ),
                "changed_pixels_shuffled_vs_trained": int(
                    np.count_nonzero(payload["trained"] != payload["shuffled"])
                ),
            }
        )
    pipeline.fusion.load_state_dict(trained)
    metrics = {}
    for mode in MODES:
        c = counts[mode]
        metrics[mode] = {
            "segmentation": segmentation_metrics(matrices[mode]),
            "detection": c
            | {
                "precision_at_2m": c["tp"] / max(c["tp"] + c["fp"], 1),
                "recall_at_2m": c["tp"] / max(c["tp"] + c["fn"], 1),
                "mean_matched_xy_error_m": (
                    c["matched_xy_error_sum_m"] / c["tp"] if c["tp"] else None
                ),
                "status": "DIAGNOSTIC_FIXED_SCORE_0.1_CLASSWISE_XY_2M",
                "official_nuscenes_map_nds": "NOT_VERIFIED",
            },
        }
    return {"metrics": metrics, "frames": records}


def prepare_protocol(cfg, args):
    """CPU-only process: fix both subsets before any model or predictions exist."""
    import mmcv
    from nuscenes.utils.splits import create_splits_scenes

    output = args.output.resolve().parent
    splits, records, annotations, versions = create_splits_scenes(), {}, {}, {}
    root = Path(cfg.data.train.data_root).resolve()
    scenes = {
        r["token"]: r
        for r in json.loads((root / "v1.0-trainval/scene.json").read_text())
    }
    for key, dataset_cfg, size, count, split in (
        ("train", cfg.data.train, 323, 20, "mini_train"),
        ("validation", cfg.data.test, 81, 6, "mini_val"),
    ):
        with memory_stage(
            "protocol.annotations." + key, minimum=args.min_host_available_mib
        ):
            ann = mmcv.load(dataset_cfg.ann_file)
            infos = sorted(ann["infos"], key=lambda r: r["timestamp"])
            if len(infos) != size:
                raise ValueError(f"Expected {size} {key} frames")
            for info in infos:
                if scenes[info["scene_token"]]["name"] not in splits[split]:
                    raise ValueError(f"Not official {split}")
            records[key] = [
                {
                    "index": index,
                    "token": infos[index]["token"],
                    "scene_token": infos[index]["scene_token"],
                    "scene_name": scenes[infos[index]["scene_token"]]["name"],
                    "text": TEXTS[n % 3],
                    "target_class": n % 3 + 1,
                    "cameras": {
                        k: str(Path(v["data_path"]).resolve())
                        for k, v in infos[index]["cams"].items()
                    },
                }
                for n, index in enumerate(
                    np.linspace(0, size - 1, count, dtype=int).tolist()
                )
            ]
            annotations[key] = str(Path(dataset_cfg.ann_file).resolve())
            versions[key] = ann["metadata"]["version"]
            del infos, ann
        release_memory()
    if {r["token"] for r in records["train"]} & {
        r["token"] for r in records["validation"]
    }:
        raise ValueError("Train/validation overlap")
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "steps": 20,
        "batch_size": 1,
        "optimizer": "AdamW",
        "learning_rate": 3e-4,
        "weight_decay": 0.01,
        "gradient_clip_max_norm": 5.0,
        "alpha_init": 0.0,
        "loss": asdict(LossConfig()),
        **records,
        "train_ann": annotations["train"],
        "val_ann": annotations["validation"],
        "dataset_metadata_version": versions,
        "selection": "linspace indices fixed before predictions; no replacement or result-based selection",
        "temporal_mode": "single frame; queue_length=1; no history",
        "validation_schedule": "separate process after step20; subset fixed before step1",
        "checkpoint": str(args.joint_checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.joint_checkpoint),
        "distilbert_sha256": sha256(args.model / "model.safetensors"),
        "config_sha256": sha256(
            args.joint_config
            or args.author_root
            / "projects/configs/bevformer/bevformer_small_seg_det.py"
        ),
        "source_sha256": {
            str(p): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "physical_alignment": "NOT_VERIFIED; same-author-raster diagnostic only",
        "metadata_loader": "stream original trainval tables; retain single-frame dependencies, no GT/source edits",
        "detection_protocol": "native decoded boxes; score>=0.1; classwise greedy XY<=2m; diagnostic, not AP/NDS",
        "frozen": ["entire author model", "DistilBERT"],
    }
    write_json(output / "protected_before.json", protected_files(args, cfg))
    write_json(output / "protocol.json", protocol)
    return {
        "status": "PASS_FIXED_PROTOCOL",
        "optimizer_steps": 0,
        "protocol": str(output / "protocol.json"),
    }


def read_protocol(args):
    if args.protocol is None:
        raise ValueError("A fixed --protocol is required")
    protocol = json.loads(args.protocol.read_text())
    if protocol["seed"] != args.seed or protocol["steps"] != 20:
        raise ValueError("Protocol seed/steps mismatch")
    checks = {
        str(args.joint_checkpoint.resolve()): protocol["checkpoint_sha256"],
        str(args.model / "model.safetensors"): protocol["distilbert_sha256"],
        **protocol["source_sha256"],
        **json.loads((args.protocol.parent / "protected_before.json").read_text()),
    }
    for path, expected in checks.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Protocol input changed: {path}")
    config = (
        args.joint_config
        or args.author_root / "projects/configs/bevformer/bevformer_small_seg_det.py"
    )
    if sha256(config) != protocol["config_sha256"]:
        raise ValueError("Protocol config changed")
    return protocol


def build_micro_dataset(cfg, args, protocol):
    import mmcv
    from mmdet3d.datasets import build_dataset
    from .low_memory_dataset import bounded_metadata

    training = args.micro_worker == "train"
    dataset_cfg = copy.deepcopy(cfg.data.train if training else cfg.data.test)
    dataset_cfg.test_mode = not training
    if training:
        dataset_cfg.queue_length = 1
    dataset_cfg.pop("samples_per_gpu", None)
    dataset_cfg.pop("workers_per_gpu", None)
    expected_ann = protocol["train_ann" if training else "val_ann"]
    if str(Path(dataset_cfg.ann_file).resolve()) != expected_ann:
        raise ValueError("Dataset annotation path differs from protocol")
    with memory_stage(
        "build_dataset.train" if training else "build_dataset.validation",
        minimum=args.min_host_available_mib,
    ):
        ann = mmcv.load(expected_ann)
        tokens = {info["token"] for info in ann["infos"]}
        del ann
        with bounded_metadata(tokens):
            dataset = build_dataset(dataset_cfg)
    for frame in protocol["train" if training else "validation"]:
        if dataset.data_infos[frame["index"]]["token"] != frame["token"]:
            raise ValueError("Fixed protocol token/index mismatch")
    release_memory()
    return dataset


def load_resume(path, protocol_path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Resume protocol hash mismatch")
    if not 1 <= saved["steps"] <= 20:
        raise ValueError("Invalid resume step")
    return saved


def check_integrity(model, pipeline, before, args):
    protected = json.loads((args.protocol.parent / "protected_before.json").read_text())
    result = {
        "protected_files_unchanged": all(
            sha256(Path(p)) == h for p, h in protected.items()
        ),
        "frozen_states_unchanged": before
        == (module_hash(model), module_hash(pipeline.encoder)),
    }
    write_json(args.output.parent / "integrity.json", result)
    if not all(result.values()):
        raise RuntimeError("Frozen state or protected input changed")
    return result


def save_training_checkpoint(path, pipeline, optimizer, step, args, initial):
    with path.open("xb") as stream:
        torch.save(
            {
                "state_dict": pipeline.fusion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "steps": step,
                "initial": initial,
                "protocol_sha256": sha256(args.protocol),
            },
            stream,
        )
        stream.flush()
        import os

        os.fsync(stream.fileno())


def run_micro_training(model, wrapped, pipeline, train, protocol, args):
    """Train only; no validation dataset, iterator, probes or retained loss graphs."""
    output = args.output.parent
    seed_all(args.seed)
    before = (module_hash(model), module_hash(pipeline.encoder))
    model.requires_grad_(False).eval()
    pipeline.encoder.requires_grad_(False).eval()
    pipeline.fusion.requires_grad_(True).train()
    optimizer = torch.optim.AdamW(
        pipeline.fusion.parameters(), lr=3e-4, weight_decay=0.01
    )
    start = 0
    if args.resume:
        saved = load_resume(args.resume, args.protocol)
        start, initial = saved["steps"], saved["initial"]
        pipeline.fusion.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        del saved
    else:
        with torch.no_grad():
            pipeline.fusion.alpha.zero_()
        initial = {
            k: v.detach().cpu().clone() for k, v in pipeline.fusion.state_dict().items()
        }
        with (output / "fusion_initial.pt").open("xb") as stream:
            torch.save(
                {"state_dict": initial, "protocol_sha256": sha256(args.protocol)},
                stream,
            )
    count = args.micro_steps
    if not args.resume and count != 1:
        raise ValueError("First run must execute exactly one optimizer.step")
    if start + count > protocol["steps"]:
        raise ValueError("Requested updates exceed fixed 20-step protocol")
    if args.resume:
        gate = json.loads((args.resume.parent / "step1_gate.json").read_text())
        if not gate["passed"] or gate["protocol_sha256"] != sha256(args.protocol):
            raise ValueError("Missing successful step1 memory/update gate")
    had_override = "obtain_history_bev" in model.__dict__
    original_local = model.__dict__.get("obtain_history_bev")
    model.obtain_history_bev = lambda *_a, **_k: None
    torch.cuda.reset_peak_memory_stats()
    baseline = memory_event(
        "training.ready", min_available_mib=args.min_host_available_mib
    )
    rows = []
    try:
        with (output / "training.jsonl").open("x", encoding="utf-8") as log:
            for n in range(start, start + count):
                frame, started = protocol["train"][n], time.monotonic()
                memory_event(
                    f"step{n+1}.batch.before",
                    min_available_mib=args.min_host_available_mib,
                    min_cuda_free_mib=args.min_cuda_free_mib,
                )
                with memory_stage(f"step{n+1}.get_batch"):
                    data = get_batch(train, frame["index"], args.seed + frame["index"])
                optimizer.zero_grad(set_to_none=True)
                with memory_stage(f"step{n+1}.forward"):
                    total, losses, details = real_training_forward(
                        model,
                        wrapped,
                        pipeline,
                        data,
                        frame["text"],
                        frame["target_class"],
                        audit_gradients=n == 0,
                    )
                with memory_stage(f"step{n+1}.backward"):
                    total.backward()
                pre_step = {
                    k: p.detach().cpu().clone()
                    for k, p in pipeline.fusion.named_parameters()
                }
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        pipeline.fusion.parameters(), 5.0, error_if_nonfinite=True
                    )
                )
                if (
                    grad_norm <= 0
                    or any(p.grad is not None for p in model.parameters())
                    or any(p.grad is not None for p in pipeline.encoder.parameters())
                ):
                    raise RuntimeError("Invalid fusion/frozen gradients")
                with memory_stage(
                    f"step{n+1}.optimizer_step", minimum=args.min_host_available_mib
                ):
                    optimizer.step()
                stats = parameter_stats(pipeline.fusion, pre_step)
                if not any(r["changed_elements"] for r in stats.values()):
                    raise RuntimeError("No parameter changed after optimizer.step")
                # Drop the loss and batch BEFORE sampling settled memory or starting another step.
                del total, data, pre_step
                optimizer.zero_grad(set_to_none=True)
                release_memory()
                torch.cuda.synchronize()
                memory = memory_event(
                    f"step{n+1}.settled", min_available_mib=args.min_host_available_mib
                )
                row = frame | {
                    "step": n + 1,
                    "optimizer_step_executed": True,
                    "losses": losses,
                    "details": details,
                    "parameters": stats,
                    "memory": memory,
                    "gradient_global_l2_before_clip": grad_norm,
                    "alpha": float(pipeline.fusion.alpha.detach()),
                    "seconds": time.monotonic() - started,
                }
                log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                log.flush()
                import os

                os.fsync(log.fileno())
                rows.append(row)
                checkpoint = output / f"fusion_step{n+1:03d}.pt"
                save_training_checkpoint(
                    checkpoint, pipeline, optimizer, n + 1, args, initial
                )
                print(
                    f"[train] step={n+1}/20 total={losses['total_loss']:.6f}",
                    flush=True,
                )
        integrity = check_integrity(model, pipeline, before, args)
        # One update cannot prove absence of long-run leaks. Gate only the measured first-step budget.
        growth = memory["allocated_mib"] - baseline["allocated_mib"]
        gate = {
            "passed": growth < 1024
            and memory["MemAvailable_mib"] >= args.min_host_available_mib,
            "claim": "one real update and settled memory budget; long-run stability unverified",
            "settled_cuda_growth_mib": growth,
            "memory": memory,
            "protocol_sha256": sha256(args.protocol),
            "checkpoint_sha256": sha256(checkpoint),
            "optimizer_steps_total": start + count,
        }
        if start == 0:
            write_json(output / "step1_gate.json", gate)
        else:
            write_json(
                output / "step1_gate.json",
                json.loads((args.resume.parent / "step1_gate.json").read_text()),
            )
        if not gate["passed"]:
            raise MemoryError("First-step settled memory growth exceeds budget")
    finally:
        if had_override:
            model.obtain_history_bev = original_local
        else:
            delattr(model, "obtain_history_bev")
    return {
        "status": "PASS_REAL_OPTIMIZER_STEPS",
        "optimizer_steps": len(rows),
        "optimizer_steps_total": start + len(rows),
        "new_weights": str(checkpoint),
        "new_weights_sha256": sha256(checkpoint),
        "memory": memory,
        "integrity": integrity,
        "validation_executed": False,
        "accuracy_improvement": "NOT_VERIFIED",
    }


def run_micro_validation(model, wrapped, pipeline, val, protocol, args):
    if args.resume is None:
        raise ValueError("Validation requires --resume final step20 checkpoint")
    saved = load_resume(args.resume, args.protocol)
    if saved["steps"] != 20:
        raise ValueError("Validation only after all 20 fixed updates")
    before = (module_hash(model), module_hash(pipeline.encoder))
    pipeline.fusion.load_state_dict(saved["state_dict"])
    initial = saved["initial"]
    del saved
    with memory_stage("validation.evaluate", minimum=args.min_host_available_mib):
        ab = evaluate(
            model, wrapped, pipeline, val, protocol, initial, args.output.parent
        )
    write_json(args.output.parent / "ab_results.json", ab)
    check_integrity(model, pipeline, before, args)
    return {"status": "PASS_FIXED_AB", "optimizer_steps": 0, "ab": ab}

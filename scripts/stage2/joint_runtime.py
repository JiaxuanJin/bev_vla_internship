"""统一系统的真实 BEVFormer 双 Head 验证后端；不改第三方源码和 GT。"""

import copy
from dataclasses import asdict
import hashlib
import importlib
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

from .runtime_memory import memory_stage, memory_event, release_memory
from .bev_coordinates import BEVGrid
from .pipeline import Stage2Pipeline
from .multitask_training import (
    LossConfig,
    Stage2MultiTaskLoss,
    aggregate_detection_losses,
    supervised_attention_loss,
)
from .run_step70_joint_integration import compare_tensors, sha256, tensor_copy


def run_joint(args):
    protocol_only = getattr(args, "micro_worker", None) == "protocol"
    if not protocol_only and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA_UNAVAILABLE：真实联合 BEVFormer 依赖 CUDA/MMCV 自定义算子"
        )
    # 训练流水线含 NumPy / Python 随机增强；必须在构建样本前设置种子。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    author = args.author_root.resolve()
    checkpoint = args.joint_checkpoint.resolve()
    config = (
        args.joint_config.resolve()
        if args.joint_config
        else author / "projects/configs/bevformer/bevformer_small_seg_det.py"
    )
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError("缺少真实联合 checkpoint 或配置")
    # 兼容旧作者流水线，仅影响本进程，不编辑 NumPy 安装或数据。
    for name, replacement in (
        ("bool", np.bool_),
        ("int", int),
        ("float", float),
        ("complex", complex),
    ):
        if name not in np.__dict__:
            setattr(np, name, replacement)
    before_hash = sha256(checkpoint)
    result_record = None
    micro = args.mode == "micro-train"
    backward = args.mode == "joint-backward" or (
        micro and args.micro_worker in ("train", "smoke")
    )
    cwd, previous_path = Path.cwd(), list(sys.path)
    try:
        sys.path.insert(0, str(author))
        os.chdir(author)
        from mmcv import Config
        from mmcv.parallel import MMDataParallel
        from mmcv.runner import load_checkpoint
        from mmdet3d.datasets import build_dataset
        from mmdet3d.models import build_model

        importlib.import_module("projects.mmdet3d_plugin")
        from projects.mmdet3d_plugin.datasets.builder import build_dataloader

        cfg = Config.fromfile(str(config))
        head = cfg.model.pts_bbox_head
        if dict(head.task) != {"seg": True, "det": True}:
            raise ValueError("配置必须启用真实检测和地图分割双 Head")
        grid = BEVGrid(
            int(head.bev_h),
            int(head.bev_w),
            int(head.transformer.embed_dims),
            tuple(cfg.point_cloud_range),
        )
        if (grid.height, grid.width, grid.channels) != (150, 150, 256):
            raise ValueError("本入口只验证 Small 150×150×256 配置")
        cfg.model.pretrained = None
        if not backward:
            cfg.model.train_cfg = None
        cfg.model.video_test_mode = False
        if micro:
            from .micro_training import (
                prepare_protocol,
                read_protocol,
                build_micro_dataset,
            )

            if protocol_only:
                return prepare_protocol(cfg, args)
            protocol = read_protocol(args)
        else:
            dataset_cfg = cfg.data.train if backward else cfg.data.test
            if backward:
                dataset_cfg.queue_length = 1
            else:
                dataset_cfg.test_mode = True
            dataset_cfg.pop("samples_per_gpu", None)
            dataset_cfg.pop("workers_per_gpu", None)
            with memory_stage("build_dataset.joint"):
                dataset = build_dataset(dataset_cfg)
            if not 0 <= args.index < len(dataset):
                raise IndexError("样本 index 超出真实数据集范围")
            with memory_stage("dataloader.first_batch"):
                loader = build_dataloader(
                    dataset,
                    samples_per_gpu=1,
                    workers_per_gpu=0,
                    dist=False,
                    shuffle=False,
                    nonshuffler_sampler=cfg.data.nonshuffler_sampler,
                )
                iterator = iter(loader)
                for _ in range(args.index + 1):
                    data = next(iterator)
        minimum = getattr(args, "min_host_available_mib", 0) if micro else 0
        with memory_stage("build_model", minimum=minimum):
            model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        with memory_stage("load_checkpoint", minimum=minimum):
            saved = load_checkpoint(model, str(checkpoint), map_location="cpu")
            model.CLASSES = saved.get("meta", {}).get(
                "CLASSES", cfg.data.train.classes if micro else dataset.CLASSES
            )
            del saved
        release_memory()
        memory_event("load_checkpoint.released")
        model.requires_grad_(False).eval()
        with memory_stage("model.cuda", minimum=minimum):
            wrapped = MMDataParallel(model.cuda(), device_ids=[0])
        release_memory()
        torch.manual_seed(args.seed)
        with memory_stage("DistilBERT.init", minimum=minimum):
            pipeline = Stage2Pipeline.from_local_model(args.model, grid)
        with memory_stage("DistilBERT.cuda", minimum=minimum):
            pipeline = pipeline.cuda().eval()
        release_memory()
        if micro:
            from .micro_training import run_micro_training, run_micro_validation

            # No DataLoader, iterator or image batch exists during initialization.
            dataset = build_micro_dataset(cfg, args, protocol)
            if args.micro_worker == "smoke":
                from .integration_smoke import run_micro_smoke

                worker = run_micro_smoke
            else:
                worker = run_micro_training if backward else run_micro_validation
            result_record = worker(model, wrapped, pipeline, dataset, protocol, args)
            result_record.update(checkpoint_sha256=before_hash)
            return result_record
        if backward:
            result_record = _real_joint_backward(model, wrapped, pipeline, data, args)
            result_record.update(
                sample_token=str(dataset.data_infos[args.index]["token"]),
                dataset="nuScenes Mini Train, real training pipeline, single frame",
                checkpoint_sha256=before_hash,
            )
            return result_record
        transformer = model.pts_bbox_head.transformer
        keys = ("seg_preds", "all_cls_scores", "all_bbox_preds")
        captured, traces = {}, {}
        head_calls = 0

        def head_hook(_module, _inputs, output):
            nonlocal head_calls
            head_calls += 1
            if not isinstance(output, dict):
                raise RuntimeError("联合 Head 必须输出字典")
            captured.update({key: tensor_copy(output.get(key), torch) for key in keys})
            if any(captured[key] is None for key in keys):
                raise RuntimeError("真实联合 Head 缺少检测/分割张量")

        handle = model.pts_bbox_head.register_forward_hook(head_hook)
        predictions = {}
        try:
            for mode in ("original", "bypass", "enhanced"):
                captured.clear()
                head_calls = 0
                model.prev_frame_info = {
                    "prev_bev": None,
                    "scene_token": None,
                    "prev_pos": 0,
                    "prev_angle": 0,
                }
                inference = copy.deepcopy(data)
                # GT 只留在原 batch；绝不作为推理的输入或目标位置模板。
                for name in ("semantic_indices", "gt_bboxes_3d", "gt_labels_3d"):
                    inference.pop(name, None)
                with torch.inference_mode():
                    if mode == "original":
                        result = wrapped(return_loss=False, rescale=True, **inference)
                    else:
                        with pipeline.inject_shared_bev(
                            transformer,
                            [args.text[0]],
                            enabled=mode == "enhanced",
                        ) as state:
                            result = wrapped(
                                return_loss=False, rescale=True, **inference
                            )
                        if state["hook_calls"] != 1:
                            raise RuntimeError("单帧必须恰好调用一次共享 BEV Hook")
                        traces[mode] = state
                if (
                    head_calls != 1
                    or len(result) != 1
                    or any(result[0].get(k) is None for k in ("pts_bbox", "seg_preds"))
                ):
                    raise RuntimeError("未取得单帧真实双 Head 预测")
                predictions[mode] = dict(captured)
        finally:
            handle.remove()
        metrics = {
            key: {
                "bypass_vs_original": compare_tensors(
                    predictions["original"][key], predictions["bypass"][key], torch
                ),
                "enhanced_vs_original": compare_tensors(
                    predictions["original"][key], predictions["enhanced"][key], torch
                ),
            }
            for key in keys
        }
        if any(
            v["bypass_vs_original"]["max_abs_difference"] != 0 for v in metrics.values()
        ):
            raise AssertionError("真实双 Head 的 bypass 必须与原始预测精确一致")
        result_record = {
            "status": "PASS_REAL_DUAL_HEAD_FUNCTION_ONLY",
            "sample_token": str(dataset.data_infos[args.index]["token"]),
            "text": args.text[0],
            "tensor_results": metrics,
            "hook_traces": traces,
            "checkpoint_sha256": before_hash,
            "coordinate_contract": "TENSOR_ORDER_ONLY",
            "module_weights_trained": False,
            "optimizer_step_executed": False,
            "semantic_localization": "NOT_VERIFIED",
            "accuracy_improvement": "NOT_VERIFIED",
            "checkpoint_provenance": "REQUIRES_MANUAL_VALIDATION",
        }
        return result_record
    finally:
        os.chdir(cwd)
        sys.path[:] = previous_path
        after_hash = sha256(checkpoint)
        if after_hash != before_hash:
            raise RuntimeError("checkpoint 哈希发生变化")
        if result_record is not None:
            result_record.update(
                checkpoint_sha256_after=after_hash, checkpoint_modified=False
            )


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                continue
    elif hasattr(value, "data") and not isinstance(value, (str, bytes, dict)):
        return _first_tensor(value.data)
    raise TypeError("无法从真实地图 GT 的 DataContainer 取得张量")


def _gradient_stats(loss, named_parameters):
    """检查真实 loss 图到融合参数的连通性；零梯度/NaN 不可报告 PASS。"""
    if loss.numel() != 1 or not torch.isfinite(loss).all():
        raise RuntimeError("loss 必须是有限标量")
    gradients = torch.autograd.grad(
        loss,
        [p for _, p in named_parameters],
        retain_graph=True,
        allow_unused=True,
    )
    values = {}
    for (name, _), grad in zip(named_parameters, gradients):
        if grad is not None:
            if not torch.isfinite(grad).all():
                raise RuntimeError(f"融合参数梯度非有限：{name}")
            values[name] = float(grad.detach().float().norm())
    norm = sum(value * value for value in values.values()) ** 0.5
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("该项真实 loss 没有产生非零且有限的语言融合梯度")
    return {
        "loss_value": float(loss.detach()),
        "global_l2": norm,
        "all_finite": True,
        "grad_parameter_tensors": len(values),
        "parameter_l2": values,
    }


def _real_joint_backward(model, wrapped, pipeline, data, args):
    """新入口同一次真实训练前向；没有调用 Step72，也没有 optimizer。"""
    from torch.nn import functional as F

    if not all(k in data for k in ("semantic_indices", "gt_bboxes_3d", "gt_labels_3d")):
        raise ValueError("缺少真实地图或 3D GT，禁止代用合成标签")
    raw_gt = _first_tensor(data["semantic_indices"])
    if raw_gt.numel() != 200 * 400:
        raise ValueError("真实 GT 必须是单帧 200×400，禁止 resize")
    gt = raw_gt.reshape(1, 200, 400).long().cuda()
    if int(gt.min()) < 0 or int(gt.max()) > 3:
        raise ValueError("真实地图标签必须为 0～3")
    gt_hash = hashlib.sha256(raw_gt.cpu().numpy().tobytes()).hexdigest()
    from .semantic_instruction import parse_instruction

    instruction = parse_instruction(args.text[0])
    if len(instruction.targets) != 1:
        raise ValueError("本次单帧反传仅支持一个明确正向地图类别")
    target_class = {"divider": 1, "ped_crossing": 2, "boundary": 3}[
        instruction.targets[0]
    ]
    if not bool((gt == target_class).any()):
        raise ValueError("真实帧中没有指令目标，不能伪造注意力正样本")
    target = torch.tensor([target_class], device=gt.device, dtype=torch.long)
    head = model.pts_bbox_head
    if not hasattr(head, "feat_cropper"):
        raise ValueError("联合 Head 缺少作者地图采样器")
    params = list(pipeline.fusion.named_parameters())
    before = {
        k: value.detach().clone() for k, value in pipeline.fusion.state_dict().items()
    }
    captured = {}
    head_calls = 0

    def capture(_module, _inputs, output):
        nonlocal head_calls
        head_calls += 1
        for key in ("seg_preds", "all_cls_scores", "all_bbox_preds"):
            if not isinstance(output, dict) or not torch.is_tensor(output.get(key)):
                raise RuntimeError(f"真实联合 Head 缺少 {key}")
        captured.update(output)  # 保留图，不能使用推理时的 detach 副本。

    had_history_override = "obtain_history_bev" in model.__dict__
    original_local = model.__dict__.get("obtain_history_bev")
    model.obtain_history_bev = lambda *_args, **_kwargs: None
    handle = head.register_forward_hook(capture)
    try:
        with torch.enable_grad(), pipeline.inject_shared_bev(
            head.transformer,
            [args.text[0]],
            capture_details=True,
        ) as state:
            author_losses = wrapped(return_loss=True, **copy.deepcopy(data))
            if state["hook_calls"] != 1 or head_calls != 1:
                raise RuntimeError("三任务必须来自同一次真实共享 BEV / 双 Head 前向")
            if not any("seg" in key.lower() for key in author_losses):
                raise RuntimeError("作者训练前向未返回分割损失")
            detection = {
                key: value
                for key, value in author_losses.items()
                if (key.startswith("loss") or ".loss" in key)
                and "seg" not in key.lower()
            }
            if not any("cls" in key for key in detection) or not any(
                "bbox" in key for key in detection
            ):
                raise RuntimeError("真实 3D 检测损失缺少 cls 或 bbox")
            attention = state["details"]["attention"]
            attention_logits = head.feat_cropper(
                torch.flip(
                    torch.logit(attention.clamp(1e-5, 1 - 1e-5)).transpose(2, 3),
                    dims=[2],
                )
            )
            if tuple(attention_logits.shape) != (1, 1, 200, 400):
                raise RuntimeError("作者采样器输出维度错误，禁止插值 GT")
            segmentation = captured["seg_preds"]
            config = LossConfig()
            total, _ = Stage2MultiTaskLoss(config)(
                detection,
                segmentation,
                gt,
                attention_logits,
                target,
            )
            terms = {
                "detection": aggregate_detection_losses(detection),
                "segmentation": F.cross_entropy(segmentation, gt),
                "attention": supervised_attention_loss(
                    attention_logits, gt, target, config
                ),
                "total": total,
            }
            stats = {
                name: _gradient_stats(loss, params) for name, loss in terms.items()
            }
            total.backward()
            grads = [p.grad for _, p in params if p.grad is not None]
            norm = sum(float(g.float().norm()) ** 2 for g in grads) ** 0.5
            if (
                not grads
                or not all(torch.isfinite(g).all() for g in grads)
                or not np.isfinite(norm)
                or norm <= 1e-12
            ):
                raise RuntimeError("联合 backward 后融合梯度为零或非有限")
            if any(p.grad is not None for p in model.parameters()) or any(
                p.grad is not None for p in pipeline.encoder.parameters()
            ):
                raise RuntimeError("冻结的作者模型或 DistilBERT 意外收到梯度")
            if any(
                not torch.equal(value, before[name])
                for name, value in pipeline.fusion.state_dict().items()
            ):
                raise RuntimeError("未授权的融合参数更新")
            if hashlib.sha256(raw_gt.cpu().numpy().tobytes()).hexdigest() != gt_hash:
                raise RuntimeError("输入 GT 张量发生变化")
            return {
                "status": "PASS_REAL_UNIFIED_JOINT_BACKWARD_NO_OPTIMIZER_STEP",
                "forward_calls": 1,
                "shared_bev_hook_calls": state["hook_calls"],
                "head_calls": head_calls,
                "shapes": state["trace"],
                "text": args.text[0],
                "target_class": target_class,
                "gt_class_counts": {str(c): int((gt == c).sum()) for c in range(4)},
                "gt_tensor_sha256": gt_hash,
                "gt_tensor_unchanged": True,
                "author_loss_keys": list(author_losses),
                "loss_config": asdict(config),
                "loss_sources": {
                    "detection": "author forward_train 3D cls+bbox loss dict",
                    "segmentation": "same-forward real segmentation logits + real map GT, CrossEntropy",
                    "attention": "same-forward Attention via author orientation/cropper + real map class GT; no language-region GT",
                },
                "gradients": stats,
                "backward_global_l2": norm,
                "frozen_author_and_distilbert_gradients_absent": True,
                "optimizer_step_executed": False,
                "fusion_parameters_updated": False,
                "semantic_localization": "NOT_VERIFIED",
                "accuracy_improvement": "NOT_VERIFIED",
                "bev_gt_physical_alignment": "NOT_VERIFIED",
                "temporal_mode": "single frame; history disabled",
            }
    finally:
        handle.remove()
        if had_history_override:
            model.obtain_history_bev = original_local
        else:
            delattr(model, "obtain_history_bev")

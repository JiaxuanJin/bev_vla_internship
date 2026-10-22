"""真实共享 BEV / 双 Head / 新损失的单帧 smoke；绝不创建 optimizer。"""

from contextlib import nullcontext
from dataclasses import asdict

import torch

from .micro_training import (
    check_integrity,
    forward_payload,
    get_batch,
    module_hash,
    real_training_forward,
    sha256,
)
from .multitask_training import LossConfig
from .runtime_memory import memory_event, memory_stage, release_memory


def load_fusion_weights(pipeline, path):
    """只加载独立 fusion state_dict，严格保持作者模型和 encoder 的命名空间。"""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    pipeline.fusion.load_state_dict(saved["state_dict"], strict=True)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "recorded_steps": saved.get("steps"),
    }


def run_micro_smoke(model, wrapped, pipeline, dataset, protocol, args):
    frame = protocol["train"][0]
    model.requires_grad_(False).eval()
    pipeline.requires_grad_(False).eval()
    pipeline.fusion.requires_grad_(True)
    loaded = (
        load_fusion_weights(pipeline, args.fusion_checkpoint)
        if args.fusion_checkpoint
        else None
    )
    frozen_before = (module_hash(model), module_hash(pipeline.encoder))
    fusion_before = module_hash(pipeline.fusion)
    data = get_batch(dataset, frame["index"], args.seed + frame["index"])
    keys = ("seg_preds", "all_cls_scores", "all_bbox_preds")
    captured, predictions = {}, {}
    head = model.pts_bbox_head

    def capture(_module, _inputs, output):
        for key in keys:
            captured[key] = output[key].detach().cpu().clone()

    had_local = "obtain_history_bev" in model.__dict__
    original = model.__dict__.get("obtain_history_bev")
    model.obtain_history_bev = lambda *_a, **_k: None
    handle = head.register_forward_hook(capture)
    try:
        for mode in ("original", "bypass"):
            captured.clear()
            context = (
                nullcontext()
                if mode == "original"
                else pipeline.inject_shared_bev(head.transformer, enabled=False)
            )
            with torch.no_grad(), context, memory_stage("smoke." + mode):
                wrapped(return_loss=True, **forward_payload(data))
            predictions[mode] = dict(captured)
        bypass = {
            k: float(
                (predictions["original"][k] - predictions["bypass"][k]).abs().max()
            )
            for k in keys
        }
        if any(bypass.values()):
            raise RuntimeError("真实三类双 Head 输出 bypass 非逐位一致")
        config = LossConfig(**protocol["loss"])
        with memory_stage(
            "smoke.enhanced_backward", minimum=args.min_host_available_mib
        ):
            total, losses, details = real_training_forward(
                model,
                wrapped,
                pipeline,
                data,
                frame["text"],
                frame["target_class"],
                audit_gradients=True,
                loss_config=config,
            )
            total.backward()
        enhanced_delta = {
            k: float((predictions["original"][k] - captured[k]).abs().max())
            for k in keys
        }
        if not all(enhanced_delta[k] > 0 for k in keys):
            raise RuntimeError("增强分支未影响真实检测/分割输出")
        if any(p.grad is not None for p in model.parameters()) or any(
            p.grad is not None for p in pipeline.encoder.parameters()
        ):
            raise RuntimeError("冻结模型出现梯度")
        if module_hash(pipeline.fusion) != fusion_before:
            raise RuntimeError("smoke 意外更新参数")
        del total, data
        pipeline.zero_grad(set_to_none=True)
        release_memory()
        integrity = check_integrity(model, pipeline, frozen_before, args)
        return {
            "status": "PASS_REAL_INTEGRATION_SMOKE_NO_UPDATE",
            "optimizer_steps": 0,
            "optimizer_step_executed": False,
            "fusion_parameters_updated": False,
            "loaded_fusion": loaded,
            "loss_config": asdict(config),
            "bypass_max_abs": bypass,
            "enhanced_max_abs": enhanced_delta,
            "losses": losses,
            "details": details,
            "integrity": integrity,
            "memory": memory_event("smoke.settled"),
            "semantic_localization": "NOT_VERIFIED",
            "physical_alignment": "NOT_VERIFIED",
        }
    finally:
        handle.remove()
        if had_local:
            model.obtain_history_bev = original
        else:
            delattr(model, "obtain_history_bev")

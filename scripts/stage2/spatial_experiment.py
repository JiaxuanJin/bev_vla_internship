"""真实空间监督实验：固定协议、分进程单步/续训和独立 GPU A/B。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2.micro_training import (
    build_micro_dataset, forward_payload, get_batch, parameter_stats,
    real_training_forward, seed_all, tensor_hash,
)
from scripts.stage2.multitask_training import LossConfig
from scripts.stage2.pipeline import Stage2Pipeline
from scripts.stage2.run_step70_joint_integration import sha256
from scripts.stage2.runtime_memory import memory_event, release_memory
from scripts.stage2.spatial_annotations import load_target, project_attention

AUTHOR = Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg"
HISTORICAL = ROOT / "outputs/stage2_oom_fix_20261022_continue/continue/fusion_step020.pt"
INITIAL = ROOT / "outputs/stage2_oom_fix_20261022_step1/train/fusion_initial.pt"
LEGACY_PROTOCOL = ROOT / "outputs/stage2_oom_fix_20261022_step1/protocol/protocol.json"


def write_json(path, value):
    """不覆盖已有实验记录。"""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def prepare(args):
    """只用训练标签确定更新顺序，不访问验证预测。"""
    manifest = json.loads((args.annotations / "manifest.json").read_text())
    train = [r for r in manifest["annotations"] if r["split"] == "train"]
    val = [r for r in manifest["annotations"] if r["split"] == "validation"]
    order = np.random.RandomState(20261026).permutation(len(train)).tolist()
    schedule = [train[n] for n in order[:20]]
    assert len(schedule) == 20 and len(val) > 0
    assert not {r["scene_token"] for r in schedule} & {r["scene_token"] for r in val}
    legacy = json.loads(LEGACY_PROTOCOL.read_text())
    protocol = {
        "seed": 20261026, "steps": 20, "train": schedule, "validation": val,
        "annotations": str(args.annotations),
        "manifest_sha256": sha256(args.annotations / "manifest.json"),
        "mask_sha256": {r["mask_file"]: sha256(args.annotations / r["mask_file"])
                        for r in schedule + val},
        "initial": str(INITIAL), "initial_sha256": sha256(INITIAL),
        "historical": str(HISTORICAL), "historical_sha256": sha256(HISTORICAL),
        "checkpoint": legacy["checkpoint"], "checkpoint_sha256": legacy["checkpoint_sha256"],
        "distilbert_sha256": legacy["distilbert_sha256"],
        "loss": asdict(LossConfig()), "learning_rate": 3e-4,
        "weight_decay": .01, "gradient_clip": 5.,
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                          sorted((ROOT / "scripts/stage2").glob("*.py"))},
        "selection": "seeded permutation of 30 train annotations; first 20; all 10 valid held-out instructions evaluated; no validation tuning",
        "attention_projection": "calibrated Ego GT centers to native LiDAR BEV; align_corners=False",
        "frozen": ["author BEVFormer weights", "DistilBERT"],
        "historical_comparison_limit": "historical 20 class updates used original fixed frames and class-only text; new 20 spatial updates use accepted subset and direction text; controlled inference comparison, not training-data ablation",
    }
    write_json(args.output_dir / "protocol.json", protocol)
    print("PROTOCOL_FROZEN", len(schedule), "updates;", len(val), "held-out instructions", flush=True)


def verify(protocol):
    """恢复前严格检查协议绑定的源码和标签。"""
    directory = Path(protocol["annotations"])
    assert sha256(directory / "manifest.json") == protocol["manifest_sha256"]
    for name, digest in protocol["mask_sha256"].items():
        assert sha256(directory / name) == digest
    for name, digest in protocol["source_sha256"].items():
        assert sha256(ROOT / name) == digest, name


def initialize(phase):
    """复用已验证的低内存数据集构建和真实双 Head 模型。"""
    if not torch.cuda.is_available():
        raise RuntimeError("真实空间实验必须使用 CUDA")
    for name, value in (("bool", np.bool_), ("int", int), ("float", float), ("complex", complex)):
        if name not in np.__dict__:
            setattr(np, name, value)
    seed_all(20261026)
    sys.path.insert(0, str(AUTHOR))
    os.chdir(AUTHOR)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model

    importlib.import_module("projects.mmdet3d_plugin")
    config = AUTHOR / "projects/configs/bevformer/bevformer_small_seg_det.py"
    cfg = Config.fromfile(str(config))
    cfg.model.pretrained = None
    cfg.model.video_test_mode = False
    if phase == "evaluate":
        cfg.model.train_cfg = None
    memory_event("spatial.build.before", min_available_mib=2048)
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    saved = load_checkpoint(model, "/mnt/g/nus/bevformer_small_joint_author.pth", map_location="cpu")
    model.CLASSES = saved.get("meta", {}).get("CLASSES", cfg.data.train.classes)
    del saved
    model.requires_grad_(False).eval()
    wrapped = MMDataParallel(model.cuda(), device_ids=[0])
    pipeline = Stage2Pipeline.from_local_model(ROOT / "models/distilbert-base-uncased").cuda().eval()
    legacy = json.loads(LEGACY_PROTOCOL.read_text())
    dataset_args = SimpleNamespace(micro_worker="train" if phase == "train" else "validate",
                                   min_host_available_mib=2048)
    dataset = build_micro_dataset(cfg, dataset_args, legacy)
    model.obtain_history_bev = lambda *_a, **_k: None
    release_memory()
    memory_event("spatial.initialized", min_available_mib=2048)
    return model, wrapped, pipeline, dataset


def train(args, protocol):
    """每步保存 checkpoint、三任务损失、梯度、参数更新和峰值显存。"""
    model, wrapped, pipeline, dataset = initialize("train")
    pipeline.fusion.requires_grad_(True).train()
    optimizer = torch.optim.AdamW(pipeline.fusion.parameters(), lr=3e-4, weight_decay=.01)
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        assert saved["protocol_sha256"] == sha256(args.protocol)
        start = saved["steps"]
        pipeline.fusion.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
    else:
        if args.steps != 1:
            raise ValueError("首次进程严格限制为一次更新")
        saved = torch.load(INITIAL, map_location="cpu", weights_only=False)
        pipeline.fusion.load_state_dict(saved["state_dict"])
        start = 0
    del saved
    assert start + args.steps <= protocol["steps"]
    directory = Path(protocol["annotations"])
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    rows = []
    with (args.output_dir / "training.jsonl").open("x") as log:
        for index in range(start, start + args.steps):
            row = protocol["train"][index]
            memory_event(f"spatial.step{index+1}.before", min_available_mib=2048,
                         min_cuda_free_mib=768)
            gt, target, grid = load_target(row, directory, "cuda")
            data = get_batch(dataset, row["index"], protocol["seed"] + row["index"])
            # 训练数据流水线的原始 GT 必须逐像素匹配预先生成的实例标签。
            from scripts.stage2.joint_runtime import _first_tensor
            actual = _first_tensor(data["semantic_indices"]).reshape(200, 400).long().cuda()
            if not torch.equal(actual, gt):
                raise ValueError("实际训练 GT 与冻结标签不同")
            optimizer.zero_grad(set_to_none=True)
            total, losses, details = real_training_forward(
                model, wrapped, pipeline, data, row["text"], row["target_class"],
                audit_gradients=index == 0, loss_config=LossConfig(**protocol["loss"]),
                spatial_targets=[target], attention_sampling_grid=grid)
            total.backward()
            before = {k: p.detach().cpu().clone() for k, p in pipeline.fusion.named_parameters()}
            norm = float(torch.nn.utils.clip_grad_norm_(pipeline.fusion.parameters(), 5., error_if_nonfinite=True))
            assert norm > 0
            assert all(p.grad is None for p in model.parameters())
            assert all(p.grad is None for p in pipeline.encoder.parameters())
            optimizer.step()
            stats = parameter_stats(pipeline.fusion, before)
            assert any(r["changed_elements"] for r in stats.values())
            torch.cuda.synchronize()
            record = {"step": index + 1, "instruction": row, "losses": losses,
                      "details": details, "gradient_l2_before_clip": norm,
                      "optimizer_step_executed": True, "parameter_changes": stats,
                      "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                      "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
            checkpoint = args.output_dir / f"fusion_step{index+1:03d}.pt"
            with checkpoint.open("xb") as stream:
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in pipeline.fusion.state_dict().items()},
                            "optimizer": optimizer.state_dict(), "steps": index + 1,
                            "protocol_sha256": sha256(args.protocol)}, stream)
            record["checkpoint_sha256"] = sha256(checkpoint)
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()
            os.fsync(log.fileno())
            rows.append(record)
            del total, data, before, actual, gt, target, grid
            optimizer.zero_grad(set_to_none=True)
            release_memory()
            memory_event(f"spatial.step{index+1}.settled", min_available_mib=2048)
            print(f"SPATIAL_OPTIMIZER_STEP {index+1} loss={losses['total_loss']:.6f} att={losses['attention_loss']:.6f}", flush=True)
    write_json(args.output_dir / "result.json", {
        "status": "REAL_GPU_SPATIAL_UPDATES", "optimizer_steps_this_process": len(rows),
        "total_steps": start + len(rows), "seconds": time.monotonic() - started,
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "command": [sys.executable, *sys.argv]})


def evaluate(args, protocol):
    """在同一实例/ROI 上重新推理历史与空间模型，不把 GT 输入推理。"""
    model, wrapped, pipeline, dataset = initialize("evaluate")
    historical = torch.load(HISTORICAL, map_location="cpu", weights_only=False)["state_dict"]
    saved = torch.load(args.resume, map_location="cpu", weights_only=False)
    assert saved["protocol_sha256"] == sha256(args.protocol) and saved["steps"] == 20
    spatial = saved["state_dict"]
    del saved
    rows = []
    for number, row in enumerate(protocol["validation"]):
        data = get_batch(dataset, row["index"], protocol["seed"] + 1000 + row["index"])
        gt, target, grid = load_target(row, Path(protocol["annotations"]), "cuda")
        from scripts.stage2.joint_runtime import _first_tensor
        assert torch.equal(_first_tensor(data["semantic_indices"]).reshape(200, 400).long().cuda(), gt)
        payload = {"gt": gt.cpu().numpy(), "target": target.instance_mask.cpu().numpy(),
                   "region": target.related_region.cpu().numpy(), "token": row["sample_token"]}
        with np.load(Path(protocol["annotations"]) / row["mask_file"]) as label:
            payload["other_instances"] = label["other_instances"]
            payload["selected_full_instance"] = label["selected_full_instance"]
        branches = {
            "original": (None, row["text"]),
            "historical": (historical, row["text"]),
            "spatial": (spatial, row["text"]),
            "incorrect": (spatial, row["text"].replace("left", "RIGHT").replace("right", "left").replace("RIGHT", "right")),
            "shuffled": (spatial, " ".join(reversed(row["text"].split()))),
        }
        for branch, (state_dict, text) in branches.items():
            captured = {}

            def capture(_module, _inputs, outputs):
                captured["segmentation"] = outputs["seg_preds"].detach()

            handle = model.pts_bbox_head.register_forward_hook(capture)
            model.prev_frame_info = {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}
            try:
                with torch.inference_mode():
                    if state_dict is None:
                        result = wrapped(return_loss=False, rescale=True, **forward_payload(data, inference=True))[0]
                    else:
                        pipeline.fusion.load_state_dict(state_dict)
                        with pipeline.inject_shared_bev(model.pts_bbox_head.transformer,
                                                       [text], strict=branch != "shuffled",
                                                       capture_details=True) as state:
                            result = wrapped(return_loss=False, rescale=True, **forward_payload(data, inference=True))[0]
                            att = state["details"]["attention"]
                            payload[branch + "_attention_native"] = att[0, 0].cpu().numpy()
                            payload[branch + "_attention"] = project_attention(att, grid)[0, 0].cpu().numpy()
                    payload[branch] = captured["segmentation"][0].argmax(0).cpu().numpy().astype(np.uint8)
                    det = result["pts_bbox"]
                    payload[branch + "_boxes"] = det["boxes_3d"].tensor.cpu().numpy()
                    payload[branch + "_scores"] = det["scores_3d"].cpu().numpy()
                    payload[branch + "_labels"] = det["labels_3d"].cpu().numpy()
            finally:
                handle.remove()
            print("AB_GPU", number, branch, row["instruction_id"], flush=True)
        annotation = dataset.get_ann_info(row["index"])
        payload["gt_boxes"] = annotation["gt_bboxes_3d"].tensor.numpy()
        payload["gt_labels"] = np.asarray(annotation["gt_labels_3d"])
        filename = f"instruction_{number:03d}.npz"
        np.savez_compressed(args.output_dir / filename, **payload)
        rows.append({**row, "prediction_file": filename, "branch_text": {k: v[1] for k, v in branches.items()}})
        del data, gt, target, grid, payload, captured, annotation
        release_memory()
    write_json(args.output_dir / "predictions.json", {
        "annotations": rows, "spatial_checkpoint": str(args.resume),
        "spatial_sha256": sha256(args.resume), "historical_sha256": sha256(HISTORICAL),
        "author_sha256": protocol["checkpoint_sha256"],
        "protocol_sha256": sha256(args.protocol), "command": [sys.executable, *sys.argv],
        "optimizer_steps_evaluation": 0, "gt_used_for_inference": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("protocol", "train", "evaluate"), required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.phase == "protocol":
        args.annotations = args.annotations.resolve()
        prepare(args)
        return
    args.protocol = args.protocol.resolve()
    if args.resume:
        args.resume = args.resume.resolve()
    protocol = json.loads(args.protocol.read_text())
    verify(protocol)
    if args.phase == "train":
        train(args, protocol)
    else:
        evaluate(args, protocol)


if __name__ == "__main__":
    main()

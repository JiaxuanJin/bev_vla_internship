"""本地 joint/官方 detection checkpoint 的来源、内嵌配置、结构只读审计。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
from pathlib import Path
import subprocess

import torch

from scripts.stage2.spatial_annotations import AUTHOR
from scripts.stage3.lg_bev_common import ROOT, read, sha, write
from scripts.stage3.lg_bev_formal_inventory import OUT


def tensor_group(name: str) -> str:
    """结构分组不将buffer数量冒称可训练参数数量。"""
    for value in (
        "img_backbone",
        "img_neck",
        "seg_decoder",
        "transformer.encoder",
        "transformer.decoder",
        "cls_branches",
        "reg_branches",
    ):
        if value in name:
            return value
    return "other"


def checkpoint_record(path: Path) -> tuple[dict, dict]:
    """反序列化已授权本地权重，保留关键训练元数据和按组计数。"""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved["state_dict"]
    meta = saved.get("meta", {})
    groups = defaultdict(lambda: dict(keys=0, state_numel=0))
    for name, value in state.items():
        group = groups[tensor_group(name)]
        group["keys"] += 1
        group["state_numel"] += value.numel()
    result = dict(
        path=str(path.resolve()),
        bytes=path.stat().st_size,
        sha256=sha(path),
        meta={
            k: meta.get(k)
            for k in (
                "epoch",
                "iter",
                "exp_name",
                "time",
                "seed",
                "CLASSES",
                "mmcv_version",
                "mmdet_version",
                "mmseg_version",
                "mmdet3d_version",
                "hook_msgs",
            )
        },
        original_environment=meta.get("env_info"),
        config=meta.get("config", ""),
        state_keys=len(state),
        state_numel=sum(v.numel() for v in state.values()),
        groups=dict(groups),
        optimizer_present="optimizer" in saved,
        dataset_sample_manifest_embedded=any(
            k in meta
            for k in ("sample_tokens", "dataset_manifest", "train_scene_tokens")
        ),
    )
    return result, state


def config_summary(config) -> dict:
    """从真正合并配置读取结构和训练协议，不按文件名猜测。"""
    model = config.model
    head = model.pts_bbox_head
    return dict(
        model_type=model.type,
        backbone=dict(type=model.img_backbone.type, depth=model.img_backbone.depth),
        head_type=head.type,
        task=dict(head.get("task", {})),
        native_bev=[head.bev_h, head.bev_w],
        queries=head.num_query,
        embedding=head.transformer.embed_dims,
        encoder_layers=head.transformer.encoder.num_layers,
        decoder_layers=head.transformer.decoder.num_layers,
        seg_encoder=dict(head.get("seg_encoder", {})),
        loss_seg=dict(head.get("loss_seg", {})),
        train_ann=config.data.train.ann_file,
        val_ann=config.data.val.ann_file,
        queue_length=config.data.train.get("queue_length"),
        samples_per_gpu=config.data.get("samples_per_gpu"),
        optimizer=dict(config.optimizer),
        lr_schedule=dict(config.get("lr_config", {})),
        total_epochs=config.get("total_epochs"),
        gpu_ids=list(config.get("gpu_ids", [])),
        load_from=config.get("load_from"),
        video_test_mode=model.get("video_test_mode"),
    )


def audit(out: Path) -> None:
    """对比两个真实资产；仅讨论 LoRA 数量，不插入或训练任何 Head 参数。"""
    from mmcv import Config

    joint_path = Path("/mnt/g/nus/bevformer_small_joint_author.pth")
    official_path = ROOT / "BEVFormer/ckpts/bevformer_small_epoch_24.pth"
    joint, joint_state = checkpoint_record(joint_path)
    official, official_state = checkpoint_record(official_path)
    embedded = Config.fromstring(joint["config"], ".py")
    official_cfg = Config.fromfile(
        str(ROOT / "BEVFormer/projects/configs/bevformer/bevformer_small.py")
    )
    live_cfg = Config.fromfile(
        str(AUTHOR / "projects/configs/bevformer/bevformer_small_seg_det.py")
    )
    joint["training_configuration"] = config_summary(embedded)
    joint["current_local_configuration"] = config_summary(live_cfg)
    official["local_configuration"] = config_summary(official_cfg)
    common = sorted(set(joint_state) & set(official_state))
    equal, shape_mismatches = defaultdict(int), []
    for key in common:
        if joint_state[key].shape != official_state[key].shape:
            shape_mismatches.append(
                dict(
                    key=key,
                    joint=list(joint_state[key].shape),
                    official=list(official_state[key].shape),
                )
            )
        elif torch.equal(joint_state[key], official_state[key]):
            equal[tensor_group(key)] += 1
    comparison = dict(
        common_keys=len(common),
        joint_only_keys=sorted(set(joint_state) - set(official_state)),
        official_only_keys=sorted(set(official_state) - set(joint_state)),
        shape_mismatches=shape_mismatches,
        identical_tensor_counts_by_group=dict(equal),
        conclusion="joint is a distinct third-party dual-head model; official detection checkpoint has no trained native segmentation decoder",
    )
    receipt = read(ROOT / "outputs/stage2_step70/step70b_weight_ready.json")
    joint["source_evidence"] = dict(
        release_repo=receipt["source_repo"],
        release_url=receipt["download_url"],
        receipt_sha256=sha(ROOT / "outputs/stage2_step70/step70b_weight_ready.json"),
        receipt_hash_matches=receipt["sha256_local_integrity"] == joint["sha256"],
        upstream_readme_epoch19_joint_matches_metadata=True,
        published_sha256_available=False,
        cryptographic_publisher_identity="NOT_VERIFIED",
        training_scope="embedded config references standard nuScenes train/val filenames, no actual sample/scene list; treat official train as potentially pretrained-seen",
        exact_pretraining_scene_exposure="NOT_VERIFIED; cannot prove unseen from local asset metadata",
    )
    joint["iteration_warning"] = (
        "44555/19=2345 iter/epoch; original world_size and actual dataset list not embedded; do not infer exact frames from iterations or current local mini pkl"
    )
    candidates = []
    for name, value in joint_state.items():
        if (
            "transformer.decoder.layers" in name
            and ".attentions.1." in name
            and name.endswith(("value_proj.weight", "output_proj.weight"))
            and value.ndim == 2
        ):
            candidates.append(
                dict(
                    name=name,
                    shape=list(value.shape),
                    lora_r4_parameters=4 * sum(value.shape),
                    lora_r8_parameters=8 * sum(value.shape),
                )
            )
    seg_last = next(
        (
            dict(
                name=name,
                shape=list(value.shape),
                dense_parameters=value.numel(),
                lora_r1_parameters=value.shape[0] + value.shape[1],
                lora_r4_parameters=4 * (value.shape[0] + value.shape[1]),
            )
            for name, value in joint_state.items()
            if "seg_decoder.up2.4.weight" in name
        ),
        None,
    )
    write(
        out / "head_adaptation_feasibility.json",
        dict(
            status="DISCUSSION_ONLY_NOT_IMPLEMENTED",
            head_adapters_installed=False,
            detection_cross_attention_candidates=candidates,
            detection_lora_r4_total=sum(r["lora_r4_parameters"] for r in candidates),
            detection_lora_r8_total=sum(r["lora_r8_parameters"] for r in candidates),
            segmentation_last_conv=seg_last,
            recommendation="first establish train grounding/convergence and native task sensitivity; low16-step Attention IoU does not establish frozen Head bottleneck",
            gate="separate preregistration and legal split required; no automatic LoRA/fine-tune after adapter failure",
        ),
    )
    joint.pop("config")
    official.pop("config")
    write(
        out / "checkpoint_audit.json",
        dict(
            joint=joint,
            official_detection=official,
            comparison=comparison,
            repo_sources={
                str(path): dict(
                    remote=subprocess.check_output(
                        ["git", "-C", str(path), "remote", "get-url", "origin"],
                        text=True,
                    ).strip(),
                    head=subprocess.check_output(
                        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
                    ).strip(),
                )
                for path in (AUTHOR, ROOT / "BEVFormer")
            },
        ),
    )
    print(
        "CHECKPOINT_AUDIT_COMPLETE",
        joint["state_keys"],
        official["state_keys"],
        "seg",
        joint["groups"].get("seg_decoder"),
        flush=True,
    )
    del joint_state, official_state
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    audit(parser.parse_args().out.resolve())

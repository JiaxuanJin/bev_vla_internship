"""生成正式训练设计与不可执行长训的预检：只写元数据协议，不训练。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import math
from pathlib import Path

from scripts.stage3.lg_bev_common import read, sha, write
from scripts.stage3.lg_bev_formal_inventory import META, OUT

SEED = 20261027


def assign_log_splits(groups: list[dict]) -> dict[str, str]:
    """按地点分层、log整组哈希划分；不读取任何GT类别或模型表现。"""
    by_location = defaultdict(list)
    log_locations = {}
    for group in groups:
        token, location = group["log_token"], group["location"]
        if token in log_locations and log_locations[token] != location:
            raise ValueError("同一drive log出现不同location")
        log_locations[token] = location
        by_location[group["location"]].append(group["log_token"])
    result = {}
    for location, logs in sorted(by_location.items()):
        ordered = sorted(
            set(logs),
            key=lambda token: hashlib.sha256(
                f"LG-BEV-formal-{SEED}:{location}:{token}".encode()
            ).hexdigest(),
        )
        if len(ordered) < 3:
            for token in ordered:
                result[token] = "train"
            continue
        count = max(1, int(len(ordered) * 0.1))
        for index, token in enumerate(ordered):
            result[token] = (
                "cold_final"
                if index < count
                else "development" if index < 2 * count else "train"
            )
    return result


def budget(frame_count: int, accumulation: int = 4) -> dict:
    """固定完整frame epoch，不把语言指令数冒称独立帧或独立样本。"""
    if frame_count <= 0 or accumulation <= 0:
        raise ValueError("训练帧数和累积步数必须为正")
    updates = math.ceil(frame_count / accumulation)
    return dict(
        micro_batch_frames=1,
        gradient_accumulation=accumulation,
        warmup_frame_epochs=2,
        joint_frame_epochs=4,
        updates_per_epoch=updates,
        warmup_optimizer_updates=2 * updates,
        joint_optimizer_updates=4 * updates,
        total_optimizer_updates=6 * updates,
        sampled_instruction_occurrences=6 * frame_count,
        native_joint_micro_forwards=4 * frame_count,
        per_epoch="one fixed GT-grounded prompt per frame; pair text uses same frame; final partial accumulation normalized by actual microbatch count",
        mask_warmup_heads_executed=False,
        long_training_hard_cap_hours=24,
        timeout_policy="save interruption checkpoint and mark INCOMPLETE; do not claim completed epochs or select a better checkpoint",
    )


def validate_split_records(records: list[dict]) -> dict:
    """场景、frame token和drive log均禁止跨split。"""
    frame_ids, scene_splits, log_splits = set(), {}, {}
    if not records:
        raise ValueError("空清单")
    for row in records:
        token, scene, log, split = (
            row["sample_token"],
            row["scene_token"],
            row["log_token"],
            row["split"],
        )
        if split not in {"train", "development", "cold_final"}:
            raise ValueError("未知split")
        if token in frame_ids:
            raise ValueError("重复帧token")
        if scene in scene_splits and scene_splits[scene] != split:
            raise ValueError("场景跨split")
        if log in log_splits and log_splits[log] != split:
            raise ValueError("drive log跨split")
        if (
            row["official_split"] != "train"
            or row["partition"] != "eligible_official_train"
        ):
            raise ValueError("封存或官方val帧进入正式候选")
        frame_ids.add(token)
        scene_splits[scene] = split
        log_splits[log] = split
    return dict(
        unique_frames=len(frame_ids),
        scenes=len(scene_splits),
        logs=len(log_splits),
        leakage=False,
    )


def build(out: Path) -> None:
    """先冻结新数据边界和完整预算；任何长训均不在本工具中实现。"""
    frames = read(out / "frame_inventory_verified.json")["frames"]
    scenes = {r["token"]: r for r in read(META / "scene.json")}
    inventory = read(out / "data_inventory_summary.json")
    historical_names = set(inventory["quarantined_scene_names"])
    historical_logs = {
        r["log_token"] for r in scenes.values() if r["name"] in historical_names
    }
    candidates = [
        r
        for r in frames
        if r["partition"] == "eligible_official_train"
        and r["joint_ready_verified"]
        and scenes[r["scene_token"]]["log_token"] not in historical_logs
    ]
    groups = {
        scenes[r["scene_token"]]["log_token"]: dict(
            log_token=scenes[r["scene_token"]]["log_token"], location=r["location"]
        )
        for r in candidates
    }
    assignment = assign_log_splits(list(groups.values()))
    records = [
        {
            **r,
            "log_token": scenes[r["scene_token"]]["log_token"],
            "split": assignment[scenes[r["scene_token"]]["log_token"]],
        }
        for r in candidates
    ]
    checks = validate_split_records(records)
    counts = {}
    for split in ("train", "development", "cold_final"):
        rows = [r for r in records if r["split"] == split]
        if not rows:
            raise ValueError("缺少独立split，禁止输出可训练协议")
        counts[split] = dict(
            frames=len(rows),
            scenes=len({r["scene_token"] for r in rows}),
            drive_logs=len({r["log_token"] for r in rows}),
            locations=dict(
                __import__("collections").Counter(r["location"] for r in rows)
            ),
            empty_gt_frames=sum(not r["annotation_record_present"] for r in rows),
        )
    write(
        out / "formal_split_manifest.json",
        dict(
            frames=records,
            metadata_only=True,
            no_GTs_or_predictions_for_cold_final_loaded=True,
            prior_history_quarantined_logs=sorted(historical_logs),
            counts=counts,
            split_checks=checks,
        ),
    )
    training_budget = budget(counts["train"]["frames"])
    checkpoint = read(out / "checkpoint_audit.json")["joint"]
    plan = dict(
        status="DESIGN_FROZEN_LONG_TRAINING_NOT_RUN",
        protocol_id="lg_bev_formal_adapter_only_20261027_v1",
        seed=SEED,
        split_counts=counts,
        split_manifest_sha256=sha(out / "formal_split_manifest.json"),
        history=dict(
            raw_candidates_frames=27808,
            raw_candidate_scenes=692,
            after_whole_historical_log_quarantine=checks,
            old_development_and_holdout_reused=False,
            official_val_reused=False,
            author_pretraining_unseen_verified=False,
            exposure="official train may be seen by foundation joint checkpoint; scene/log-disjoint adaptation generalization only",
        ),
        model=dict(
            checkpoint=checkpoint["path"],
            checkpoint_sha256=checkpoint["sha256"],
            backbone="frozen author R101-DCN",
            native_heads="frozen unchanged native detection + segmentation; no cropper replacement",
            text_encoder="frozen local DistilBERT3072 readout",
            trainable="existing LanguageGroundedBEVAdapter24 only",
            relative_bound=0.05,
            initialization="fresh seeded adapter; no pilot checkpoint continuation",
        ),
        budget=training_budget,
        temporal_protocol=dict(
            history="shared unenhanced original BEV history, author sequential video_test_mode=True",
            reset="reset per scene; chronological frames; never pass any enhanced BEV into history",
            native_baseline="same original pretrained model/config, same original history/cache for all four branches",
            gate="must compare >=3 consecutive train frames against actual original full-camera model, bitwise/head tolerance preregistered before caching; current single-frame cache runner cannot be silently reused",
        ),
        stage_warmup=dict(
            epochs=2,
            optimizer="AdamW",
            lr=0.0003,
            weight_decay=0.01,
            frozen_up_projection=True,
            native_bev_identity_required=True,
            losses=dict(
                mask_bce=1.0,
                positive_dice=1.0,
                same_class_distractor_negative=0.5,
                correct_wrong_pair_ranking=0.1,
                ranking_logit_margin=0.5,
            ),
            train_gate=dict(
                fixed_metadata_seeded_train_probe_frames=128,
                target_mask_iou_at_0_5_min=0.15,
                mean_correct_wrong_target_logit_gap_min=0.1,
                empty_prompt_active_fraction_max=0.05,
                failure="stop, record grounding NOT_READY; do not unfreeze Head or extend budget",
            ),
        ),
        stage_joint=dict(
            epochs=4,
            optimizer="fresh AdamW at warmup terminal",
            lr=0.0003,
            lr_schedule="cosine from3e-4 to3e-5 over fixed joint updates, no dev-tuned schedule",
            weight_decay=0.01,
            clip=5.0,
            unfreeze_up_only_within_new_adapter=True,
            losses=dict(
                native_detection=1.0,
                native_segmentation=1.0,
                explicit_attention_bce_dice_distractor=0.2,
                paired_text_ranking=0.1,
                map_target_neighborhood_ce=0.5,
                outside_instruction_seg_teacher_KL=0.1,
            ),
            object_native_detection="keep original Hungarian assignment; multiply matched true target GT cls/bbox weights by3; all other positive/negative weights1; exact IDs from train rawGT",
            map_semantic="original global4-class loss untouched; extra weighted CE only fixed2m true-map-record neighborhood, same GT; other sameclass instances remain positive",
            task_separation="no map-to-object proxy; map prompts keep global detector, object prompts keep global semantics",
            native_gradient_logging="per-step scalars/norm/update/residual, per-task gradients each200updates and endpoints; original model/text all-frozen checks",
        ),
        language_schedule=dict(
            frame_epoch="each train frame once per epoch; tasks alternate deterministically across epochs",
            positive_empty="target availability from raw GT only; 25% empty at instruction scheduler level when possible; never drop actual empty frames",
            counterfactual="prefer real same-frame sameclass+direction nearest/farthest pair; otherwise exact left/right swap with independently computed GT; no predictions used as GT",
            frozen_label_manifest_required=True,
            frozen_correct_wrong_pair_table_required=True,
            DriveLM="only4072 train-frame overlap for language/context;17,238 key-object records lack explicit3Dinstance token; do not use all377956QA as3D supervision",
        ),
        checkpoints=dict(
            steps="warmup terminal, every joint frame epoch, terminal; each with optimizer/scheduler/RNG/update/protocol hash",
            primary="fixed terminal only",
            development_selection=False,
            resume="only crash recovery with exact protocol/RNG/state; count all updates toward fixed budget; no rollback by dev score",
        ),
        evaluation=dict(
            branches=["original", "correct_text", "wrong_text", "zero_text"],
            controls="matched inference ablations of single trained adapter; not separately trained controls",
            development="once at fixed joint terminal; no per-epoch dev threshold/lr/structure choice",
            cold_final="metadata only now; open once after model+code+threshold+dev decision sealed, never train on final GT",
            related_detection="instruction-conditioned mAP-like-nuScenes over car/pedestrian and centre distances[0.5,1,2,4]m; SDK class range/zero-point rules; all scores; NOT official whole-val nuScenes mAP",
            related_segmentation="macro class IoU over groundable lane/crosswalk in fixed2m instance ROI; full original GT; no pooled-vs-macro substitution",
            overall="one token-hash-selected canonical instruction per frame, all10 detection classes, all3 foreground map classes; original+same3controls",
            secondary="old nonofficial AP@2m_xy, attentionIoU per native/map space, background/distractor FP, per-scene metrics, class counts, native scalar loss",
            threshold=0.5,
            uncertainty="1000 fixed-seed scene bootstrap; report raw TP/union and counts; no precision beyond actual sample support",
            official_full_val_benchmark="BLOCKED under current frozen/historical eval policy; do not claim full official mAP from new internal train-subset split",
        ),
        success=dict(
            interpretation="conservative absolute percentage points; also report relative percentages",
            related_detection_delta_pp_min=2.0,
            related_segmentation_delta_pp_min=1.5,
            overall_detection_drop_pp_max=2.0,
            overall_segmentation_drop_pp_max=2.0,
            correct_text_must_exceed_zero_and_wrong=True,
            improvement_not_guaranteed=True,
            latency="RTX3090 extra<=20ms requires actual3090/end-to-end timing, not5080 cached timing",
        ),
        stop=dict(
            nonfinite="abort immediately, preserve checkpoint/evidence",
            freeze_violation="abort immediately",
            spatial_lineage_leakage="abort before train",
            train_grounding_gate="stop after fixed warmup if gate fails",
            gradient_failure="abort if real task required path nonfinite/disconnected; zero-up first joint step documented separately",
            development_failure="seal terminal, no expansion, do not open final; any subsequent research needs separate new data/protocol",
            long_training_started=False,
        ),
        head_adaptation="DISCUSSION_ONLY; separate hypothesis and new preregistration; no Head parameters in optimizer",
        implementation_readiness=dict(
            full_instruction_gt_manifest="NOT_YET_GENERATED",
            temporal_native_cache_builder="NOT_YET_IMPLEMENTED_FOR_FORMAL_SPLIT",
            full_data_frozen_joint_trainer="NOT_YET_IMPLEMENTED",
            mAP_4distance_conditional_scorer="NOT_YET_IMPLEMENTED",
            current16step_runner="incompatible full data indices/mini dataset; not a formal training command",
        ),
        audit_lineage={
            str(out / name): sha(out / name)
            for name in (
                "checkpoint_audit.json",
                "label_coverage_audit.json",
                "pilot_rca.json",
                "pilot_train_probe.json",
            )
        },
    )
    write(out / "formal_training_protocol.json", plan)
    native_bytes = 256 * 150 * 150 * 2
    cache_before_final = counts["train"]["frames"] + counts["development"]["frames"]
    all_frames = sum(v["frames"] for v in counts.values())
    joint_frames = training_budget["native_joint_micro_forwards"]
    write(
        out / "resource_estimate.json",
        dict(
            hardware="current RTX5080Laptop16GB, host15.3GiB; no actual formal training measurements",
            measured_existing=dict(
                single_frame_raw_test_seconds=1.027826055,
                cached_joint_16steps_seconds=4.608078386,
                cached_joint_peak_allocated_mib=1688.607421875,
            ),
            native_bev_bytes_per_frame=native_bytes,
            deterministic_fp16_cache_before_final_bytes=cache_before_final
            * native_bytes,
            cache_all_splits_upper_bytes=all_frames * native_bytes,
            fp16_seg_teacher_bytes_before_final=cache_before_final * 4 * 200 * 400 * 2,
            cold_final_cache_not_generated_now=True,
            recommended_cache_root="workspace filesystem, notG: avoid eatingG free space and NTFS small-file overhead",
            camera_cache_estimate_hours=cache_before_final * 1.027826055 / 3600,
            camera_cache_estimate_limit="single-frame observed timing only; temporal cache/preprocessing/I/O must be profiled, no throughput guarantee",
            joint_microstep_estimated_seconds=[0.3, 0.8],
            joint_gpu_hours_range=[
                joint_frames * 0.3 / 3600,
                joint_frames * 0.8 / 3600,
            ],
            warmup_IO_lower_bound_hours_at200MBps=2
            * counts["train"]["frames"]
            * native_bytes
            / 200e6
            / 3600,
            map_GT_preprocessing_time="UNMEASURED for full corpus, profile32train frames before full generation",
            workers=0,
            maximum_initial_workers=2,
            mixed_precision="cacheFP16, adapter/native lossFP32; no blindAMP changes to legacy custom ops",
            temporal_full_camera_memory="UNMEASURED; bounded3-frame preflight required, abort ifallocated>12GiB",
            no_full_backbone_backward=True,
            LoRA_or_Head_finetune_memory="not included, not implemented",
        ),
    )
    write(
        out / "preflight_status.json",
        dict(
            status="PLAN_ONLY_TRAINING_NOT_STARTED",
            training_allowed=False,
            code_and_manifest_leakage_checks="PASS",
            remaining_gates=plan["implementation_readiness"],
            reason="user asked for formal scheme and audit, explicitly no automatic long training; required full-data components must be implemented/validated before later training",
        ),
    )
    print("FORMAL_PROTOCOL_READY_NOT_TRAINING", counts, training_budget, flush=True)


def verify(out: Path) -> None:
    """核对登记哈希、真实划分、历史log禁入和预算；不执行任何模型或训练。"""
    protocol = read(out / "formal_training_protocol.json")
    if sha(out / "formal_split_manifest.json") != protocol["split_manifest_sha256"]:
        raise ValueError("正式split清单哈希变化")
    for path, digest in protocol["audit_lineage"].items():
        if sha(path) != digest:
            raise ValueError(f"审计来源变化: {path}")
    manifest = read(out / "formal_split_manifest.json")
    if validate_split_records(manifest["frames"]) != manifest["split_checks"]:
        raise ValueError("清单数量校验失败")
    historical_logs = set(manifest["prior_history_quarantined_logs"])
    if any(r["log_token"] in historical_logs for r in manifest["frames"]):
        raise ValueError("历史drive log泄漏")
    train_count = sum(r["split"] == "train" for r in manifest["frames"])
    if protocol["budget"] != budget(train_count):
        raise ValueError("固定预算与实际训练帧数不匹配")
    if read(out / "preflight_status.json")["training_allowed"]:
        raise ValueError("本轮审计不得自动授权长训练")
    contract_path = out / "formal_execution_contract.json"
    if contract_path.exists():
        contract = read(contract_path)
        if contract["parent_protocol_sha256"] != sha(
            out / "formal_training_protocol.json"
        ):
            raise ValueError("正式执行补充协议的父协议哈希变化")
        if contract["training_started"] or contract["optimizer_updates_this_audit"]:
            raise ValueError("只读审计出现训练记录")
    print("FORMAL_PROTOCOL_VERIFIED_NO_TRAINING")


def seal_contract(out: Path) -> None:
    """补齐主指标与严格FP32基线约束，保留先前FP16资源草估，不启动训练。"""
    verify(out)
    plan = read(out / "formal_training_protocol.json")
    counts = plan["split_counts"]
    before_final = counts["train"]["frames"] + counts["development"]["frames"]
    total = sum(v["frames"] for v in counts.values())
    write(
        out / "formal_execution_contract.json",
        dict(
            status="FORMAL_DESIGN_SEALED_IMPLEMENTATION_GATES_PENDING",
            parent_protocol_sha256=sha(out / "formal_training_protocol.json"),
            split_manifest_sha256=plan["split_manifest_sha256"],
            training_started=False,
            optimizer_updates_this_audit=0,
            clarification="this addendum governs strict baseline/cache and exact scoring/loss semantics; first FP16 storage estimate is retained, not overwritten; no experiments follow either design in this audit",
            strict_baseline=dict(
                primary_cache_dtype="float32 lossless native BEV, no compression quantization",
                bytes_per_frame=256 * 150 * 150 * 4,
                cache_train_plus_dev_bytes=before_final * 256 * 150 * 150 * 4,
                cache_all_splits_bytes=total * 256 * 150 * 150 * 4,
                disk_reserve_before_final_gb=650,
                disk_reserve_all_splits_gb=750,
                FP16="storage alternative only, not primary formal baseline; requires separate registered quantization analysis, no silent adoption",
                parity=">=3 consecutive raw TRAIN frames: full original forward vs FP32 BEV full-Head replay cls/bbox/seg bitwise equal, including current native temporal history; fail closed otherwise",
                raw_pipeline="deterministic author test transform, no PhotoMetricDistortion; eval state; native scene chronology/history preserved",
                unchanged="original checkpoint, all native parameters/buffers, cropper, decoder, text encoder; no enhanced feature into native temporal history",
            ),
            grounding_contract=dict(
                warmup_probe="128 train frames chosen by token hash before training, positive-only IoU macro across map/object spaces; empty active fraction separate; no dev labels",
                bce="per-example valid positives and negatives balanced; positives/empty examples separately averaged; no positive Dice for empty target",
                ranking="for paired actual texts c,w with independently grounded target sets S_c,S_w: mean relu(0.5-logit_c+logit_w) on S_c minus S_w, and symmetric term on S_w minus S_c; skip empty set, never mark all wrong-text foreground as negative",
                overlap="do not rank shared target pixels; different instruction IDs are insufficient unless grounded target sets differ",
                gt_input_policy="target sets used in loss/scorer only, never adapter/native model inputs; position only calibration-derived",
                full_labels="all train frames retained including6 actual emptyGT; record per-frame valid task, failed associations and out-of-map cells, cannot silently drop/reassign frames or recompute budget",
            ),
            scoring_contract=dict(
                related_det_classes=["car", "pedestrian"],
                related_det_thresholds_m=[0.5, 1.0, 2.0, 4.0],
                query_unit="virtual sample=(frame token, fixed instruction ID), never claim repeated prompts as new independent frames",
                related_gt="all/nearest/farthest exact annotation IDs per frozen instruction table; predicted boxes filtered only by requested class, not GT proximity; sameclass distractor boxes count FP, other classes ignored for related metric",
                ap="SDK recall grid/min_recall0.1/min_precision0.1 and score sorting; equal mean over2classes x4distances; empty-target instructions retain all sameclass predictions as FP; absent positive class makes gate NOT_EVALUABLE, not zero",
                raw_metric_name="instruction_conditional_2class_mAP_4distance; not official10class full-val nuScenes mAP",
                related_seg_classes=["lane_marking", "crosswalk"],
                related_seg_roi="GT selected-record raster dilated by Euclidean pixel-centre distance<=2m, fixed0.15m raster; fullGT labels in ROI, never replace other true sameclass instances by background",
                related_seg_macro="pool intersection/union within each class over frozen positive queries then average2class IoUs; empty map queries reported separately, never inflate positive mIoU",
                overall="one hash-selected canonical prompt per frame, full10class detection4distance and3foreground semantic macroIoU; no ROI filtering",
                wrong_zero="use same correct-query GT to score all4branches; wrong text has separate GT only during paired training, never change evaluation target",
                statistical_unit="drive scene for1000 seeded bootstrap replicates, not language prompt; disclose same-location overlap may persist across logs",
                target="absolute+2.0pp related conditionalmAP,+1.5pp related macroIoU; overall mAP/mIoU decrease<=2.0pp; correct must exceed wrong/zero on both related metrics; no claim official project acceptance without aligned benchmark scope",
            ),
            remaining_gates=plan["implementation_readiness"],
            long_training_command="NONE: formal full-data trainer/scorer/cache/label pipeline must be implemented and validated in a later explicitly scheduled run",
        ),
    )
    print("FORMAL_EXECUTION_CONTRACT_SEALED_NO_TRAINING")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--verify-only", action="store_true")
    actions.add_argument("--seal-contract", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.out.resolve())
    elif args.seal_contract:
        seal_contract(args.out.resolve())
    else:
        build(args.out.resolve())

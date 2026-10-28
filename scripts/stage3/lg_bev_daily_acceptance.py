"""不同真实训练场景的离线前向、会话控制和恢复验收，绝不冒充正式更新。"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import psutil
import torch

from scripts.stage3.lg_bev_common import read, sha, write
from scripts.stage3.lg_bev_engineering_cache import (
    CACHE_ROOT,
    RollingCache,
    atomic_save,
)
from scripts.stage3.lg_bev_engineering_data import (
    SceneData,
    LabelStore,
    formal_frames,
    prepare_labels,
    initialize_author,
)
from scripts.stage3.lg_bev_engineering_native import NativeRuntime
from scripts.stage3.lg_bev_engineering_audit import (
    native_audit,
    Reference,
    maximum_delta,
)
from scripts.stage3.lg_bev_engineering_full import SceneProvider
from scripts.stage3.lg_bev_engineering_train import Trainer
from scripts.stage3.lg_bev_daily_state import (
    OUT,
    POLICY,
    source_manifest,
    event,
    latest_checkpoint,
)
from scripts.stage3.lg_bev_daily import offline_environment, deny_network
from scripts.stage3.lg_bev_native import module_sha

PREFLIGHT = CACHE_ROOT / "runs/daily_preflight_20261027_v2"


def local_encoder():
    """强制本地DistilBERT；网络拒绝已由调用者安装。"""
    from scripts.stage2.language_encoder import DistilBERTLanguageEncoder
    from scripts.stage2.spatial_v3_model import InstanceReadout

    from scripts.stage3.lg_bev_common import ROOT

    encoder = (
        InstanceReadout(
            DistilBERTLanguageEncoder(
                str(ROOT / "models/distilbert-base-uncased"), local_files_only=True
            )
        )
        .cuda()
        .eval()
    )
    encoder.requires_grad_(False)
    return encoder


def preflight_rows() -> tuple[list[dict], list[dict]]:
    """按场景名选三个不同日志的首12/12/8帧，不看GT类别、loss或性能。"""
    all_rows = formal_frames()
    groups = {}
    for row in all_rows:
        groups.setdefault(row["scene_token"], []).append(row)
    chosen, seen_logs, full = [], set(), []
    for rows in sorted(groups.values(), key=lambda g: g[0]["scene_name"]):
        rows = sorted(rows, key=lambda r: r["timestamp"])
        if rows[0]["log_token"] in seen_logs:
            continue
        seen_logs.add(rows[0]["log_token"])
        full.extend(rows)
        chosen.extend(rows[: (12 if len(seen_logs) < 3 else 8)])
        if len(seen_logs) == 3:
            break
    if len(chosen) != 32:
        raise ValueError("三个完整训练场景不足")
    return chosen, full


def prepare(reuse: Path | None = None, reuse_labels: Path | None = None) -> None:
    """生成独立小测标签/原生FP32及不中断参考；保存真实吞吐和冻结证据。"""
    offline_environment()
    attempts = deny_network()
    source = source_manifest()
    cache = RollingCache()
    PREFLIGHT.mkdir(parents=True, exist_ok=False)
    out = PREFLIGHT / "evidence"
    out.mkdir()
    rows, full = preflight_rows()
    write(
        out / "selection.json",
        dict(
            rows=rows, scope="only formal TRAIN; deterministic first3logs; no dev/cold"
        ),
    )
    labels_from = reuse_labels or reuse
    if labels_from is not None:
        report = read(labels_from / "data_report.json")
        if {r["sample_token"] for r in report["frames"]} != {
            r["sample_token"] for r in rows
        }:
            raise ValueError("复用标签不属于同一32帧")
        for item in report["frames"]:
            if sha(Path(item["label_path"])) != item["label_sha256"]:
                raise ValueError("复用标签SHA不符")
        write(
            PREFLIGHT / "data_report.json", {**report, "reused_from": str(labels_from)}
        )
    else:
        report = prepare_labels(cache, PREFLIGHT, rows)
    initialize_author()
    groups = {}
    for r in full:
        groups.setdefault(r["scene_token"], []).append(r)
    scenes, timings = {}, []
    for scene, group in groups.items():
        start = time.perf_counter()
        scenes[scene] = SceneData(group)
        timings.append(time.perf_counter() - start)
    torch.cuda.reset_peak_memory_stats()
    model_start = time.perf_counter()
    runtime = NativeRuntime(fast_images=True)
    boot_seconds = time.perf_counter() - model_start
    frozen = module_sha(runtime.model)
    if reuse is not None:
        raw = read(reuse / "evidence/native_temporal_audit.json")
        if (
            raw["frames"] != 32
            or not raw["head_replay_bitwise"]
            or not raw["true_FP32"]
        ):
            raise ValueError("复用原生证据未通过严格32帧门槛")
        prior_cross = reuse / "evidence/cross_scene.json"
        preserved = read(OUT / "preserved_before.json")
        from scripts.stage3.lg_bev_common import ROOT

        for name in (
            "lg_bev_engineering_native.py",
            "lg_bev_engineering_data.py",
            "lg_bev_engineering_determinism.py",
        ):
            path = ROOT / "scripts/stage3" / name
            digest = (
                read(prior_cross)["source_sha256"][str(path)]
                if prior_cross.exists()
                else preserved[str(path.relative_to(ROOT))]["sha256"]
            )
            if sha(path) != digest:
                raise ValueError("原生前向/数据代码变化，不能复用旧raw证据")
        reference = Reference(raw["reference"])
        for record in raw["reference"]:
            if sha(Path(record["path"])) != record["sha256"]:
                raise ValueError("复用FP32原始缓存损坏")
        raw = {**raw, "reused_from": str(reuse), "fresh_raw_forward_claimed": False}
        write(out / "native_temporal_audit.json", raw)
    else:
        reference, raw = native_audit(
            out, dict(rows=rows, run=str(PREFLIGHT)), cache, runtime, scenes
        )
    for trace in raw["traces"]:
        if trace["history_used"] != (trace["index"] > 0):
            raise RuntimeError("跨场景history reset或连续前缀不一致")
    encoder = local_encoder()
    labels = LabelStore(report)
    tokens = [r["sample_token"] for r in rows]
    trainers, durations = {}, {}
    for stage in ("A", "B"):
        trainer = Trainer(
            runtime.model,
            encoder,
            labels,
            tokens,
            reference.get,
            stage,
            9562 if stage == "A" else 19124,
            sha(POLICY),
            retain_audit_tensors=True,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        trainer.run(32, log=PREFLIGHT / ("reference_" + stage + ".jsonl"))
        torch.cuda.synchronize()
        durations[stage] = time.perf_counter() - start
        trainers[stage] = trainer
    with cache.locked():
        cache.reserve_for_artifact(5_000_000)
        atomic_save(
            PREFLIGHT / "uninterrupted_reference.pt",
            dict(
                state_dict=trainers["B"].adapter.state_dict(),
                optimizer=trainers["B"].optimizer.state_dict(),
                updates=trainers["B"].updates,
                records=trainers["B"].records,
            ),
        )
    del scenes
    provider = SceneProvider(cache, runtime, full)
    provider.hot_scenes = set()  # 小测测无命中/无全量落盘的保守再生成本。
    access = []

    def get(token):
        previous = provider.active
        previous_metrics = (
            len(provider.provider.metrics) if provider.provider is not None else 0
        )
        start = time.perf_counter()
        value = provider.get(token)
        if not torch.equal(value, reference.get(token)):
            raise RuntimeError("新原生时序重建BEV与原FP32参考非逐位一致")
        new_metrics = provider.provider.metrics[
            0 if previous != provider.active else previous_metrics :
        ]
        access.append(
            dict(
                seconds=time.perf_counter() - start,
                scene_boot=previous != provider.active,
                generation_seconds=sum(m["seconds"] for m in new_metrics),
                raw_forwards=sum(m["raw_forwards"] for m in new_metrics),
            )
        )
        return value

    streamed = Trainer(
        runtime.model,
        encoder,
        labels,
        tokens,
        get,
        "B",
        19124,
        sha(POLICY),
        retain_audit_tensors=True,
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    streamed.run(32, log=PREFLIGHT / "native_stream_B.jsonl")
    torch.cuda.synchronize()
    durations["stream_B"] = time.perf_counter() - start
    delta, relative = maximum_delta(
        trainers["B"].update_records[-1]["state"], streamed.update_records[-1]["state"]
    )
    loss_delta = max(
        abs(a["losses"]["total"] - b["losses"]["total"])
        for a, b in zip(trainers["B"].records, streamed.records)
    )
    if delta > 1e-6 or loss_delta > 1e-5:
        raise RuntimeError("不同场景流式训练偏离参考")
    if module_sha(runtime.model) != frozen or attempts:
        raise RuntimeError("冻结状态或离线约束失败")
    generated_frames = sum(a["raw_forwards"] for a in access)
    native_per_frame = sum(a["generation_seconds"] for a in access) / generated_frames
    boot_mean = (
        sum(
            max(0, a["seconds"] - a["generation_seconds"])
            for a in access
            if a["scene_boot"]
        )
        / 3
    )
    # 每帧数值不变的来源manifest复用：实测另一个epoch的场景冷启动。
    warm_provider = SceneProvider(cache, runtime, full)
    warm_provider.hot_scenes = set()
    warm_boot = []
    warm_generations = []
    for token in tokens:
        previous = warm_provider.active
        before = (
            len(warm_provider.provider.metrics)
            if warm_provider.provider is not None
            else 0
        )
        start = time.perf_counter()
        value = warm_provider.get(token)
        elapsed = time.perf_counter() - start
        metrics = warm_provider.provider.metrics[
            0 if previous != warm_provider.active else before :
        ]
        warm_generations.extend(metrics)
        if previous != warm_provider.active:
            if not all(warm_provider.provider.source_manifest_hits.values()):
                raise RuntimeError("来源manifest未在跨epoch复用")
            warm_boot.append(max(0, elapsed - sum(m["seconds"] for m in metrics)))
        if not torch.equal(value, reference.get(token)):
            raise RuntimeError("来源缓存复用改变真实BEV")
    warm_boot_mean = sum(warm_boot) / len(warm_boot)
    native_per_frame = max(
        native_per_frame,
        sum(m["seconds"] for m in warm_generations)
        / sum(m["raw_forwards"] for m in warm_generations),
    )
    if source != source_manifest() or module_sha(runtime.model) != frozen:
        raise RuntimeError("验收期间源码或冻结状态变化")
    # 128帧probe只能在A实际终点执行；这里只按每次最坏完整scene前缀估算资源。
    all_counts = {}
    for row in formal_frames():
        all_counts[row["scene_token"]] = all_counts.get(row["scene_token"], 0) + 1
    probe_upper_seconds = 128 * (
        max(all_counts.values()) * native_per_frame + warm_boot_mean + 0.1
    )
    checkpoint_save_reserve_seconds = 80 * 10
    full_label_startup_reserve_seconds = 7 * 120
    point_hours = (
        durations["A"] / 32 * 38246
        + durations["B"] / 32 * 76492
        + native_per_frame * 114738
        + boot_mean * 476
        + warm_boot_mean * 476 * 5
        + boot_seconds * 7
        + probe_upper_seconds
        + checkpoint_save_reserve_seconds
        + full_label_startup_reserve_seconds
    ) / 3600
    summary = dict(
        status="PASS",
        frames=32,
        scenes=[r[0]["scene_name"] for r in groups.values()],
        train_only=True,
        raw_FP32_heads_bitwise=raw["head_replay_bitwise"],
        parameter_max_abs=delta,
        parameter_relative_l2=relative,
        loss_max_abs=loss_delta,
        reference_seconds=durations,
        native_generation_seconds_per_frame=native_per_frame,
        generated_raw_frames=generated_frames,
        native_scene_boot_seconds=boot_mean,
        subsequent_epoch_scene_boot_seconds=warm_boot_mean,
        source_manifest_reuse_BEV_bitwise=True,
        deferred_128probe_estimate_seconds=probe_upper_seconds,
        checkpoint_save_reserve_seconds=checkpoint_save_reserve_seconds,
        full_label_startup_reserve_seconds=full_label_startup_reserve_seconds,
        scene_constructor_seconds=timings,
        process_model_boot_seconds=boot_seconds,
        active_point_hours=point_hours,
        conservative_active_hours=point_hours * 1.25,
        margin=0.25,
        resource_48h_pass=point_hours * 1.25 <= 48,
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        host_available_bytes=psutil.virtual_memory().available,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        cuda_device=torch.cuda.get_device_name(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32,
        frozen_sha256=frozen,
        offline_network_attempts=len(attempts),
        source_sha256=source,
        full_training_updates=0,
        estimate="包含全帧原生再生、476场景每epoch初始化和最多7次模型启动；不假设cache命中；不是全量计时承诺",
    )
    write(out / "cross_scene.json", summary)
    config = dict(
        scope="bounded_real_train_acceptance",
        formal_training=False,
        run_id=PREFLIGHT.name.replace("daily_preflight", "daily_acceptance"),
        stage="B",
        total_microsteps=32,
        budget_updates=19124,
        session_seconds=3.0,
        resume_session_seconds=120,
        label_report=str(PREFLIGHT / "data_report.json"),
        native_report=str(out / "native_temporal_audit.json"),
        tokens=tokens,
        clock_starts_after_assets_for_short_test=True,
        policy_sha256=sha(POLICY),
        source_sha256=source,
    )
    write(PREFLIGHT / "acceptance_config.json", config)
    print(
        "CROSS_SCENE_ACCEPTANCE_REFERENCE_READY",
        json.dumps(summary, ensure_ascii=False),
        flush=True,
    )


def acceptance_worker(work, config: dict) -> None:
    """真实GPU会话使用同一管理器，测试3秒/120秒而非真的等待7.5小时。"""
    run = work.run
    if config["source_sha256"] != source_manifest() or config["policy_sha256"] != sha(
        POLICY
    ):
        raise ValueError("验收配置源码/协议已变化")
    execution = dict(
        scope=config["scope"],
        source_sha256=source_manifest(),
        policy_sha256=sha(POLICY),
        tokens=config["tokens"],
        formal_training=False,
    )
    if not (run / "execution.json").exists():
        write(run / "execution.json", execution)
    elif read(run / "execution.json") != execution:
        raise ValueError("验收恢复执行协议变化")
    event(
        run,
        dict(
            event="process_start",
            pid=os.getpid(),
            run_id=run.name,
            acceptance_only=True,
        ),
        work.cache,
    )
    runtime = NativeRuntime(fast_images=True)
    encoder = local_encoder()
    labels = LabelStore(read(Path(config["label_report"])))
    reference = Reference(read(Path(config["native_report"]))["reference"])
    trainer = Trainer(
        runtime.model,
        encoder,
        labels,
        config["tokens"],
        reference.get,
        "B",
        config["budget_updates"],
        sha(run / "execution.json"),
    )
    trainer.session = work.clock
    trainer.total_budget_seconds = 172800
    trainer.progress_callback = work.progress
    trainer.checkpoint_callback = work.note_checkpoint
    if work.args.checkpoint:
        trainer.resume(work.args.checkpoint)
        work.clock.session_seconds = config["resume_session_seconds"]
    # 只对3秒短测排除资产初始化；正式会话从worker启动即累计，不走此分支。
    if config["clock_starts_after_assets_for_short_test"]:
        work.clock.started = time.perf_counter()
    consumed = trainer.epoch * len(trainer.tokens) + trainer.cursor
    trainer.run(
        config["total_microsteps"] - consumed,
        log=run / ("training_" + work.args.session_id + ".jsonl"),
        checkpoint_dir=run / "checkpoints",
        cache=work.cache,
    )
    consumed = trainer.epoch * len(trainer.tokens) + trainer.cursor
    finished = consumed == config["total_microsteps"]
    work.finalize(
        trainer,
        "COMPLETED" if finished else "STOPPED",
        reason="BOUNDED_ACCEPTANCE_COMPLETE" if finished else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "compare"])
    parser.add_argument("--run-dir", type=Path, default=PREFLIGHT)
    parser.add_argument("--reuse-native", type=Path)
    parser.add_argument("--reuse-labels", type=Path)
    args = parser.parse_args()
    if not args.run_dir.resolve().is_relative_to((CACHE_ROOT / "runs").resolve()):
        raise ValueError("验收产物只能在G盘受控runs目录")
    PREFLIGHT = args.run_dir
    if args.phase == "prepare":
        prepare(args.reuse_native, args.reuse_labels)
    else:
        config = read(PREFLIGHT / "acceptance_config.json")
        run = CACHE_ROOT / "runs" / config["run_id"]
        actual = latest_checkpoint(run)["state"]
        expected = torch.load(
            PREFLIGHT / "uninterrupted_reference.pt",
            map_location="cpu",
            weights_only=False,
        )
        delta, relative = maximum_delta(expected["state_dict"], actual["state_dict"])
        if delta > 1e-6 or actual["updates"] != expected["updates"]:
            raise RuntimeError("跨进程跨会话更新偏离不中断参考")
        write(
            PREFLIGHT / "evidence/manager_resume.json",
            dict(
                status="PASS",
                parameter_max_abs=delta,
                parameter_relative_l2=relative,
                stage_updates=actual["updates"],
                formal_updates=0,
                latest_checkpoint_sha256=latest_checkpoint(run, load=False)["sha256"],
            ),
        )

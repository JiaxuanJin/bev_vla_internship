"""保留通过数值验收的FP32路径，实测同32帧吞吐与6epoch预算。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import statistics
import time

import torch

from scripts.stage3.lg_bev_common import ROOT, read, sha, write
from scripts.stage3.lg_bev_engineering_audit import Reference, maximum_delta
from scripts.stage3.lg_bev_engineering_cache import RollingCache
from scripts.stage3.lg_bev_engineering_data import SceneData, LabelStore, formal_frames
from scripts.stage3.lg_bev_engineering_native import NativeRuntime, NativeCacheProvider
from scripts.stage3.lg_bev_engineering_train import Trainer
from scripts.stage3.lg_bev_gpu_benchmark import Monitor
from scripts.stage3.lg_bev_native import module_sha


def transfer_probe(value: torch.Tensor) -> dict:
    """真实23MB BEV拷贝；分开报告pin与传输开销，不改变FP32。"""
    results = {}
    for pinned in (False, True):
        start = time.perf_counter()
        host = value.pin_memory() if pinned else value
        pin_seconds = time.perf_counter() - start
        timings = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            gpu = host.to("cuda", non_blocking=pinned)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        results[str(pinned)] = dict(
            pin_seconds=pin_seconds,
            mean_seconds=statistics.mean(timings),
            bitwise=torch.equal(gpu.cpu(), value),
        )
    return results


def compare(target: Trainer, candidate: Trainer, tolerance: dict) -> dict:
    """每个累积更新逐项对照，门槛沿用GPU运行前登记值。"""
    updates = []
    for a, b in zip(target.update_records, candidate.update_records):
        ga, gr = maximum_delta(a["gradients"], b["gradients"])
        pa, pr = maximum_delta(a["state"], b["state"])
        updates.append(
            dict(
                gradient_max_abs=ga,
                gradient_relative_l2=gr,
                parameter_max_abs=pa,
                parameter_relative_l2=pr,
            )
        )
    loss = max(
        abs(a["losses"]["total"] - b["losses"]["total"])
        for a, b in zip(target.records, candidate.records)
    )
    passed = loss <= tolerance["loss_max_abs"] and all(
        r["gradient_max_abs"] <= tolerance["gradient_max_abs"]
        and r["gradient_relative_l2"] <= tolerance["gradient_relative_l2"]
        and r["parameter_max_abs"] <= tolerance["parameter_max_abs"]
        for r in updates
    )
    return dict(
        loss_max_abs=loss,
        updates=updates,
        numerical_pass=passed,
        attention_bitwise=all(
            a["attention_sha256"] == b["attention_sha256"]
            for a, b in zip(target.records, candidate.records)
        ),
    )


def benchmark(out: Path) -> None:
    """不采用失败CUDA候选；保守估计包含每epoch重新计算全部原生特征。"""
    parent = ROOT / "results/lg_bev_engineering_20261027/final_v5"
    plan = read(parent / "engineering_supplement.json")
    cache = RollingCache()
    run = cache.root / "runs" / out.name
    run.mkdir(parents=True, exist_ok=False)
    source = {
        str(p): sha(p)
        for p in (ROOT / "scripts/stage3").glob("lg_bev_engineering_*.py")
    }
    tolerance = plan["tolerances_registered_before_gpu"]
    write(
        out / "protocol.json",
        dict(
            scope="same32 train frames; B32/A32 each8 test updates",
            tolerances=tolerance,
            sources=source,
            run=str(run),
            conservative_margin=0.25,
            accepted_dynamic_backward="CPU FP32 reference",
            formal_updates=0,
        ),
    )
    reference = Reference(read(parent / "native_temporal_audit.json")["reference"])
    labels = LabelStore(read(parent / "data_report.json"))
    tokens = [r["sample_token"] for r in plan["rows"]]
    scene = plan["rows"][0]["scene_token"]
    data = SceneData([r for r in formal_frames() if r["scene_token"] == scene])
    runtime = NativeRuntime(fast_images=True)
    from scripts.stage2.language_encoder import DistilBERTLanguageEncoder
    from scripts.stage2.spatial_v3_model import InstanceReadout

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
    frozen = (module_sha(runtime.model), module_sha(encoder))
    results, baselines = {}, {}
    cases = [
        ("CPU_fixed_cached_B", False, False, False, "B"),
        ("CPU_fixed_stream_B", False, True, False, "B"),
        ("GPU_fixed_cached_B", True, False, False, "B"),
        ("GPU_fixed_stream_B", True, True, False, "B"),
        ("GPU_fixed_pinned_B", True, True, True, "B"),
        ("GPU_fixed_cached_A", True, False, False, "A"),
        ("GPU_fixed_stream_A", True, True, False, "A"),
    ]
    for name, fixed_cuda, streamed, pinned, stage in cases:
        os.environ["LG_BEV_DYNAMIC_CUDA"] = "0"
        os.environ["LG_BEV_UPSAMPLE_CUDA"] = "0"
        os.environ["LG_BEV_CPU_DIRECT"] = str(int(fixed_cuda))
        os.environ["LG_BEV_FIXED_CUDA"] = str(int(fixed_cuda))
        provider = (
            NativeCacheProvider(cache, runtime, {scene: data}, admit_disk=False)
            if streamed
            else reference
        )
        trainer = Trainer(
            runtime.model,
            encoder,
            labels,
            tokens,
            provider.get,
            stage,
            8,
            sha(out / "protocol.json"),
            retain_audit_tensors=True,
        )
        trainer.pin_transfer = pinned
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        with Monitor() as monitor:
            begin.record()
            start = time.perf_counter()
            trainer.run(32, log=run / (name + ".jsonl"))
            end.record()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        samples = monitor.samples
        gpu = [s["gpu_percent"] for s in samples if s.get("gpu_percent") is not None]
        record = dict(
            seconds=elapsed,
            seconds_per_frame=elapsed / 32,
            cuda_timeline_ms=begin.elapsed_time(end),
            updates=trainer.updates,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_utilization_mean=statistics.mean(gpu) if gpu else None,
            cpu_percent_mean=statistics.mean(s["cpu_percent"] for s in samples),
            cpu_peak_rss_bytes=max(s["rss_bytes"] for s in samples),
            samples=samples,
            fixed_cuda=fixed_cuda,
            native_stream=streamed,
            pinned=pinned,
            stage=stage,
        )
        baselines.setdefault(stage, trainer)
        record.update(compare(baselines[stage], trainer, tolerance))
        results[name] = record
        write(out / (name + ".json"), record)
        print("RETAINED_CASE", name, elapsed, record["numerical_pass"], flush=True)
        if not record["numerical_pass"]:
            write(
                out / "decision.json",
                dict(
                    status="BLOCKED_NUMERICAL",
                    case=name,
                    formal_updates=0,
                    full_training_started=False,
                ),
            )
            return
    os.environ["LG_BEV_FIXED_CUDA"] = "1"
    write(out / "transfer_probe.json", transfer_probe(reference.get(tokens[0])))
    # profiler trace与临时数据仅写G盘，剖析不计入吞吐。
    with cache.locked():
        cache.reserve_for_artifact(256_000_000)
    profiler = Trainer(
        runtime.model,
        encoder,
        labels,
        tokens,
        reference.get,
        "B",
        8,
        sha(out / "protocol.json"),
    )
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) as prof:
            profiler.run(2)
        trace = run / "profile_two_microsteps.json"
        prof.export_chrome_trace(str(trace))
        write(
            out / "profiler.json",
            dict(
                status="PASS",
                trace=str(trace),
                trace_bytes=trace.stat().st_size,
                cpu_table=prof.key_averages().table(
                    sort_by="self_cpu_time_total", row_limit=25
                ),
                gpu_table=prof.key_averages().table(
                    sort_by="self_device_time_total", row_limit=25
                ),
            ),
        )
    except (RuntimeError, AssertionError, AttributeError) as exc:
        write(out / "profiler.json", dict(status="UNAVAILABLE", error=repr(exc)))
    if (module_sha(runtime.model), module_sha(encoder)) != frozen:
        raise RuntimeError("冻结权重或BN变化")
    if any(sha(Path(p)) != h for p, h in source.items()):
        raise RuntimeError("验收中源码发生变化")
    a = results["GPU_fixed_stream_A"]["seconds_per_frame"]
    b = results["GPU_fixed_stream_B"]["seconds_per_frame"]
    hours = (a * 38246 + b * 76492) / 3600
    write(
        out / "decision.json",
        dict(
            status=(
                "GPU32_PASS_FULL_TRAINING_BLOCKED"
                if hours * 1.25 > 24
                else "GPU32_PASS_FULL_DATA_ADMISSION_PENDING"
            ),
            training_6epoch_point_hours=hours,
            conservative_training_hours=hours * 1.25,
            budget_24h_pass=hours * 1.25 <= 24,
            estimate="raw native features regenerated for every frame every epoch; +25% thermal/I/O/scene margin; excludes labels and terminal evaluation; no assumed cache-hit benefit",
            speedup_stream=results["CPU_fixed_stream_B"]["seconds"]
            / results["GPU_fixed_stream_B"]["seconds"],
            frozen_sha256=frozen,
            source_sha256=source,
            root_usage_bytes=cache.usage(),
            g_free_bytes=shutil.disk_usage(cache.root).free,
            capacity_bytes=cache.capacity,
            reserve_bytes=cache.reserve,
            formal_updates=0,
            full_training_started=False,
            remaining="full19123 labels and full-scale admission not run; failed dynamic CUDA stays disabled",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    benchmark(parser.parse_args().out.resolve())

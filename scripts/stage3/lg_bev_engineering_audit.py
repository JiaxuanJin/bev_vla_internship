"""固定32帧真实GPU验收：严格native/FP32对照、滚动淘汰与部分累积恢复。"""

from __future__ import annotations

from pathlib import Path
import resource
import shutil
import time
import threading

import torch
import psutil

from scripts.stage3.lg_bev_common import ROOT, read, sha, write
from scripts.stage3.lg_bev_engineering_cache import RollingCache, atomic_save
from scripts.stage3.lg_bev_engineering_data import LabelStore, SceneData, formal_frames
from scripts.stage3.lg_bev_engineering_native import NativeRuntime, NativeCacheProvider
from scripts.stage3.lg_bev_engineering_train import Trainer
from scripts.stage3.lg_bev_native import dual_head, module_sha


class Reference:
    """完整临时参考缓存以8帧块保存在G盘，只在RAM保留一个块。"""

    def __init__(self, records: list[dict]) -> None:
        self.lookup = {t: r for r in records for t in r["tokens"]}
        self.path = None
        self.value = {}

    def get(self, token: str) -> torch.Tensor:
        """只读校验过的完整参考cache。"""
        row = self.lookup[token]
        if row["path"] != self.path:
            if sha(Path(row["path"])) != row["sha256"]:
                raise ValueError("参考缓存损坏")
            self.value = torch.load(row["path"], map_location="cpu", weights_only=True)
            self.path = row["path"]
        return self.value[token]

    def peek(self, token: str) -> torch.Tensor | None:
        """不改变分片位置或读取次序的只读预取探针。"""
        return self.value.get(token)


def maximum_delta(a: dict, b: dict) -> tuple[float, float]:
    """按全部参数/梯度报告最大绝对差与相对L2。"""
    if set(a) != set(b):
        raise ValueError("比较张量集合不一致")
    maximum = 0.0
    numerator = 0.0
    denominator = 0.0
    for key in a:
        delta = (a[key] - b[key]).double()
        maximum = max(maximum, float(delta.abs().max()))
        numerator += float(delta.square().sum())
        denominator += float(a[key].double().square().sum())
    return maximum, (numerator / max(denominator, 1e-30)) ** 0.5


def native_audit(
    out: Path,
    supplement: dict,
    cache: RollingCache,
    runtime: NativeRuntime,
    scenes: dict,
) -> tuple[Reference, dict]:
    """先真实顺序六相机再缓存完整Head重放，逐输出逐位断言。"""
    rows = supplement["rows"]
    selected = {r["sample_token"] for r in rows}
    run = Path(supplement["run"])
    records = []
    traces = []
    block = {}
    write_stats = []
    for data in scenes.values():
        for i, row in enumerate(data.rows):
            if row["sample_token"] not in selected:
                break
            bev, prediction, trace = runtime.forward(data, i)
            with torch.no_grad():
                replay = dual_head(runtime.model, bev.cuda())
            differences = {
                k: float((prediction[k] - replay[k].cpu()).abs().max())
                for k in prediction
            }
            if any(differences.values()):
                raise RuntimeError(f"原始FP32双Head重放非逐位一致:{differences}")
            traces.append({**trace, "native_cached_head_max_abs": differences})
            block[row["sample_token"]] = bev
            if len(block) == 8 or len(traces) == len(rows):
                path = run / "reference" / f"full_{len(records):03d}.pt"
                with cache.locked():
                    cache.admit(sum(v.numel() * 4 for v in block.values()) + 2**20)
                    start = time.perf_counter()
                    atomic_save(path, block)
                    seconds = time.perf_counter() - start
                records.append(
                    dict(path=str(path), sha256=sha(path), tokens=list(block))
                )
                write_stats.append(
                    dict(
                        bytes=path.stat().st_size,
                        seconds=seconds,
                        MBps=path.stat().st_size / seconds / 1e6,
                    )
                )
                block = {}
    if len(traces) != 32 or not all(t["history_used"] for t in traces[1:3]):
        raise RuntimeError("真实连续时序覆盖失败")
    result = dict(
        status="PASS",
        frames=len(traces),
        raw_six_camera_forward=True,
        true_FP32=True,
        head_replay_bitwise=True,
        traces=traces,
        reference=records,
        write_throughput=write_stats,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        cpu_rss_bytes=psutil.Process().memory_info().rss,
    )
    if result["peak_cuda_allocated_bytes"] > 12 * 2**30:
        raise RuntimeError("原生前向超过预注册12GiB")
    write(out / "native_temporal_audit.json", result)
    return Reference(records), result


def audit(out: Path) -> None:
    """只用32个正式train帧，有限真实更新与评价功能检查；不碰开发/最终集。"""
    supplement = read(out / "engineering_supplement.json")
    run = Path(supplement["run"])
    cache = RollingCache()
    cache.recover()
    rows = supplement["rows"]
    tokens = [r["sample_token"] for r in rows]
    groups = {}
    for row in formal_frames():
        groups.setdefault(row["scene_token"], []).append(row)
    scenes = {
        scene: SceneData(groups[scene])
        for scene in dict.fromkeys(r["scene_token"] for r in rows)
    }
    runtime = NativeRuntime()
    before = module_sha(runtime.model)
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    stop_monitor = threading.Event()
    sampled_peak = {"bytes": cache.usage()}

    def monitor():
        while not stop_monitor.wait(0.2):
            sampled_peak["bytes"] = max(sampled_peak["bytes"], cache.usage())

    observer = threading.Thread(target=monitor, daemon=True)
    observer.start()
    native_path = out / "native_temporal_audit.json"
    if native_path.exists():
        native = read(native_path)
        reference = Reference(native["reference"])
    else:
        reference, native = native_audit(out, supplement, cache, runtime, scenes)
    provider = NativeCacheProvider(cache, runtime, scenes)
    # 低于30GB的工程压力上限：只留约2个8帧分片，强制真实淘汰/重建。
    owned_bytes = cache.db.execute(
        "SELECT COALESCE(SUM(bytes),0) FROM shards WHERE state='verified'"
    ).fetchone()[0]
    stress_capacity = (
        cache.usage() - owned_bytes + 2 * (8 * 23_040_000 + 2**20) + 10_000_000
    )
    cache.capacity = min(cache.capacity, stress_capacity)
    with cache.locked():
        cache.reserve_for_artifact(0)
    deltas = []
    read_stats = []
    for token in tokens:
        start = time.perf_counter()
        value = provider.get(token)
        elapsed = time.perf_counter() - start
        difference = float((value - reference.get(token)).abs().max())
        if difference != 0:
            raise RuntimeError("真实滚动重建BEV发生数值变化")
        deltas.append(difference)
        read_stats.append(elapsed)
    rebuild_start = time.perf_counter()
    rebuilt = provider.get(tokens[0])
    rebuild_seconds = time.perf_counter() - rebuild_start
    if not torch.equal(rebuilt, reference.get(tokens[0])):
        raise RuntimeError("淘汰后重建不一致")
    events = dict(
        cache.db.execute("SELECT event,COUNT(*) FROM events GROUP BY event").fetchall()
    )
    write(
        out / "rolling_cache_audit.json",
        dict(
            status="PASS",
            frames=len(tokens),
            bev_bitwise=True,
            max_abs=max(deltas),
            stress_capacity_bytes=cache.capacity,
            authorized_max_bytes=30_000_000_000,
            root_usage_bytes=cache.usage(),
            g_free_bytes=shutil.disk_usage(cache.root).free,
            events=events,
            rebuild_seconds=rebuild_seconds,
            generations=provider.metrics,
            read_or_rebuild_seconds=read_stats,
        ),
    )
    from scripts.stage3.lg_bev_engineering_recovery import audit_recovery

    write(
        out / "cache_crash_recovery.json",
        audit_recovery(cache, provider, reference, tokens[0], run),
    )
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
    labels = LabelStore(read(out / "data_report.json"))
    protocol = sha(out / "engineering_supplement.json")

    def trainer(get_bev, stage="B"):
        return Trainer(
            runtime.model,
            encoder,
            labels,
            tokens,
            get_bev,
            stage,
            8,
            protocol,
            retain_audit_tensors=True,
        )

    # 不更改损失，只验证CPU参考adjoint和原生CUDA梯度仍在原先上界内。
    cuda_probe = trainer(reference.get)
    cuda_record = cuda_probe.microstep()
    cpu_probe = trainer(reference.get)
    cpu_probe.run(1)
    a = {
        n: p.grad.detach().cpu()
        for n, p in cuda_probe.adapter.named_parameters()
        if p.grad is not None
    }
    b = {
        n: p.grad.detach().cpu()
        for n, p in cpu_probe.adapter.named_parameters()
        if p.grad is not None
    }
    adjoint_abs, adjoint_rel = maximum_delta(a, b)
    if (
        adjoint_abs > supplement["tolerances_registered_before_gpu"]["gradient_max_abs"]
        or adjoint_rel
        > supplement["tolerances_registered_before_gpu"]["gradient_relative_l2"]
    ):
        raise RuntimeError("确定性adjoint与原生CUDA梯度偏离，禁止采用")
    write(
        out / "adjoint_compatibility.json",
        dict(
            status="PASS",
            original_cuda_vs_cpu_adjoint_max_abs=adjoint_abs,
            relative_l2=adjoint_rel,
            native_forward_loss_max_abs=abs(
                cuda_record["losses"]["total"] - cpu_probe.records[0]["losses"]["total"]
            ),
            optimizer_updates=0,
            original_forward_unchanged=True,
            implementation="original CUDA forward; CPU FP32 grid/interpolate/MSDeform adjoint only",
        ),
    )
    baseline = trainer(reference.get)
    attention_delta = 0.0
    from scripts.stage3.lg_bev_data import geometry
    from scripts.stage3.lg_bev_engineering_train import choose_query, encode_texts

    for index, token in enumerate(tokens):
        label = labels.get(token)
        q = choose_query(label, 0, index)
        text = encode_texts(encoder, [q["text"]])[q["text"]]
        with torch.no_grad():
            position = geometry(label["frame"], device="cuda")
            x = baseline.adapter(reference.get(token).cuda(), text, position)
            y = baseline.adapter(provider.get(token).cuda(), text, position)
            attention_delta = max(
                attention_delta,
                float((x["attention_logits"] - y["attention_logits"]).abs().max()),
            )
    if attention_delta:
        raise RuntimeError("同参数的32帧Attention未逐位一致")
    write(
        out / "attention_input_equivalence.json",
        dict(status="PASS", frames=32, same_adapter_attention_max_abs=attention_delta),
    )
    baseline.run(32, log=run / "reference_train.jsonl")
    rolling = trainer(provider.get)
    rolling.run(32, log=run / "rolling_train.jsonl")
    tolerance = supplement["tolerances_registered_before_gpu"]
    comparisons = []
    for a, b in zip(baseline.update_records, rolling.update_records):
        grad_abs, grad_rel = maximum_delta(a["gradients"], b["gradients"])
        param_abs, param_rel = maximum_delta(a["state"], b["state"])
        if (
            grad_abs > tolerance["gradient_max_abs"]
            or grad_rel > tolerance["gradient_relative_l2"]
            or param_abs > tolerance["parameter_max_abs"]
        ):
            raise RuntimeError(
                f"完整/滚动梯度或更新不一致:{grad_abs,grad_rel,param_abs}"
            )
        comparisons.append(
            dict(
                update=a["update"],
                gradient_max_abs=grad_abs,
                gradient_relative_l2=grad_rel,
                parameter_max_abs=param_abs,
                parameter_relative_l2=param_rel,
            )
        )
    stream_provider = NativeCacheProvider(cache, runtime, scenes, admit_disk=False)
    streamed = trainer(stream_provider.get)
    streamed.run(32, log=run / "streaming_train.jsonl")
    stream_abs, stream_rel = maximum_delta(
        baseline.update_records[-1]["state"], streamed.update_records[-1]["state"]
    )
    if stream_abs > tolerance["parameter_max_abs"]:
        raise RuntimeError("RAM流式原生重算与完整参考更新不一致")
    losses = max(
        abs(a["losses"]["total"] - b["losses"]["total"])
        for a, b in zip(baseline.records, rolling.records)
    )
    attention_equal = all(
        a["attention_sha256"] == b["attention_sha256"]
        for a, b in zip(baseline.records, rolling.records)
    )
    # 参数更新如受CUDA backward浮点顺序影响，attention后续也可能有微差，不能伪称逐位。
    if losses > tolerance["loss_max_abs"]:
        raise RuntimeError("完整/滚动loss差异超预登记上界")
    interrupted = trainer(reference.get)
    interrupted.run(6, log=run / "interrupted_part1.jsonl")
    checkpoint = run / "checkpoints" / "smoke_partial_B_u1_m6.pt"
    interrupted.checkpoint(checkpoint, cache)
    resume = trainer(provider.get)
    resume.resume(checkpoint)
    resume.run(26, log=run / "interrupted_part2.jsonl")
    resume_abs, resume_rel = maximum_delta(
        baseline.update_records[-1]["state"], resume.update_records[-1]["state"]
    )
    if resume_abs > tolerance["parameter_max_abs"]:
        raise RuntimeError("部分梯度恢复后的参数不一致")
    warmup = trainer(provider.get, "A")
    warmup.run(8, log=run / "warmup_smoke.jsonl")
    if any(p.detach().count_nonzero() for p in warmup.adapter.up.parameters()):
        raise RuntimeError("预热up非零")
    write(
        out / "training_equivalence.json",
        dict(
            status="PASS",
            frames=32,
            updates_per_case=8,
            full_vs_rolling=comparisons,
            loss_max_abs=losses,
            attention_hash_all_equal_after_training=attention_equal,
            attention_note="forward given identical adapter/input must be bitwise; learned trajectories can differ within registered CUDA backward bound",
            interrupt_microstep=6,
            saved_partial_gradients=2,
            resume_parameter_max_abs=resume_abs,
            resume_parameter_relative_l2=resume_rel,
            warmup_smoke_updates=2,
            formal_updates=0,
            all_frozen_unchanged=module_sha(runtime.model) == before,
            RAM_streaming_parameter_max_abs=stream_abs,
            RAM_streaming_parameter_relative_l2=stream_rel,
            reference_task_gradients=[
                r["task_gradients"] for r in baseline.records if r["task_gradients"]
            ],
            last_residual_ratio=baseline.records[-1]["residual_ratio"],
            last_query_cosine=baseline.records[-1]["query_cosine"],
        ),
    )
    if module_sha(runtime.model) != before:
        raise RuntimeError("作者状态改变")
    from scripts.stage3.lg_bev_engineering_evaluate import functional_audit
    from scripts.stage3.lg_bev_engineering_metrics import MetricSpool

    spool = MetricSpool(run / "functional_ap_events.sqlite3", cache)
    metrics = functional_audit(
        runtime.model,
        baseline.adapter,
        encoder,
        labels,
        tokens,
        reference.get,
        spool=spool,
    )
    write(out / "evaluator_functional_audit.json", metrics)
    path = Path(native["reference"][0]["path"])
    read_start = time.perf_counter()
    read_bytes = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            read_bytes += len(block)
    read_seconds = time.perf_counter() - read_start
    stop_monitor.set()
    observer.join()
    write(
        out / "resource_measurements.json",
        dict(
            seconds=time.perf_counter() - began,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            cpu_current_rss_bytes=psutil.Process().memory_info().rss,
            cache_root_final_bytes=cache.usage(),
            cache_sampled_peak_bytes=sampled_peak["bytes"],
            cache_capacity_bytes=cache.capacity,
            g_free_bytes=shutil.disk_usage(cache.root).free,
            native_raw_calls_this_process=runtime.raw_calls,
            native_gpu_seconds=runtime.raw_seconds,
            reference32_training_seconds=baseline.wall_seconds,
            rolling32_training_seconds=rolling.wall_seconds,
            streaming32_training_seconds=streamed.wall_seconds,
            warmup8_training_seconds=warmup.wall_seconds,
            warm_file_read_MBps=read_bytes / read_seconds / 1e6,
            warm_file_read_seconds=read_seconds,
            baseline_frozen_sha256=before,
            text_frozen_sha256=module_sha(encoder),
        ),
    )
    write(
        out / "ready.json",
        dict(
            status="READY_FOR_FULL_DATA_PREPARATION",
            engineering_32frames="PASS",
            formal_training_started=False,
            full_training_allowed=False,
            reason="full19,123-frame label manifest and128-frame warmup probe must be generated and separately authorize full budget",
            cold_final_opened=False,
            development_evaluated=False,
        ),
    )
    print("ENGINEERING32_PASS_NO_FULL_TRAINING", flush=True)

"""同一32真实训练帧的FP32 GPU优化数值/吞吐验收，不启动正式长训。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

import psutil
import torch

from scripts.stage3.lg_bev_common import ROOT, read, sha, write
from scripts.stage3.lg_bev_engineering_audit import Reference, maximum_delta
from scripts.stage3.lg_bev_engineering_cache import RollingCache
from scripts.stage3.lg_bev_engineering_data import SceneData, LabelStore, formal_frames
from scripts.stage3.lg_bev_engineering_native import NativeRuntime, NativeCacheProvider
from scripts.stage3.lg_bev_engineering_train import Trainer
from scripts.stage3.lg_bev_native import module_sha


class Monitor:
    """在后台采集NVML/nvidia-smi整卡利用率与当前进程CPU/RSS，采样不读环境。"""

    def __init__(self) -> None:
        self.samples = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self) -> None:
        process = psutil.Process()
        process.cpu_percent()
        while not self.stop.is_set():
            sample = dict(
                time=time.time(),
                cpu_percent=process.cpu_percent(),
                rss_bytes=process.memory_info().rss,
            )
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                fields = result.stdout.strip().split(",")
                sample["nvidia_smi_raw"] = result.stdout.strip()
                if result.returncode == 0 and len(fields) >= 3:
                    for key, value in zip(
                        (
                            "gpu_percent",
                            "memory_bus_percent",
                            "device_used_mib",
                            "power_watts",
                        ),
                        fields,
                    ):
                        try:
                            sample[key] = float(value.strip())
                        except ValueError:
                            sample[key] = None
            except Exception as exc:
                sample["gpu_query_error"] = repr(exc)
            self.samples.append(sample)
            self.stop.wait(0.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(6)


def benchmark(out: Path, backend: str = "1") -> None:
    """CPU参考、CUDA动态反向、真实原生流式及pinned版本采用相同配置与顺序。"""
    parent = ROOT / "results/lg_bev_engineering_20261027/final_v5"
    plan = read(parent / "engineering_supplement.json")
    tokens = [r["sample_token"] for r in plan["rows"]]
    cache = RollingCache()
    run = cache.root / "runs" / out.name
    run.mkdir(parents=True, exist_ok=False)
    write(
        out / "run.json",
        dict(
            run=str(run),
            parent_supplement_sha256=sha(parent / "engineering_supplement.json"),
            formal_updates=0,
        ),
    )
    reference = Reference(read(parent / "native_temporal_audit.json")["reference"])
    labels = LabelStore(read(parent / "data_report.json"))
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
    original_model = module_sha(runtime.model)
    original_text = module_sha(encoder)
    cases = {}
    trainers = {}
    tolerance = read(out / "protocol.json")["tolerances"]
    for name, dynamic, stream, pinned in [
        ("CPU_dynamic_reference", False, False, False),
        ("CUDA_dynamic", True, False, False),
        ("CUDA_native_stream", True, True, False),
        ("CUDA_pinned_transfer", True, True, True),
    ]:
        os.environ["LG_BEV_DYNAMIC_CUDA"] = backend if dynamic else "0"
        provider = (
            NativeCacheProvider(cache, runtime, {scene: data}, admit_disk=False)
            if stream
            else reference
        )
        trainer = Trainer(
            runtime.model,
            encoder,
            labels,
            tokens,
            provider.get,
            "B",
            8,
            sha(out / "protocol.json"),
            retain_audit_tensors=True,
        )
        trainer.pin_transfer = pinned
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        with Monitor() as monitor:
            begin_event.record()
            start = time.perf_counter()
            trainer.run(32, log=run / (name + ".jsonl"))
            end_event.record()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        samples = monitor.samples
        record = dict(
            seconds=elapsed,
            seconds_per_frame=elapsed / 32,
            stream_elapsed_cuda_ms=begin_event.elapsed_time(end_event),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_utilization_mean=(
                statistics.mean(
                    s["gpu_percent"]
                    for s in samples
                    if s.get("gpu_percent") is not None
                )
                if any(s.get("gpu_percent") is not None for s in samples)
                else None
            ),
            cpu_percent_mean=statistics.mean(s["cpu_percent"] for s in samples),
            cpu_peak_rss_bytes=max(s["rss_bytes"] for s in samples),
            samples=samples,
            updates=trainer.updates,
            dynamic_deform_cuda=dynamic,
            pinned_transfer=pinned,
            native_six_camera_stream=stream,
        )
        if name != "CPU_dynamic_reference":
            baseline = trainers["CPU_dynamic_reference"]
            per_update = []
            for a, b in zip(baseline.update_records, trainer.update_records):
                ga, gr = maximum_delta(a["gradients"], b["gradients"])
                pa, pr = maximum_delta(a["state"], b["state"])
                per_update.append(
                    dict(
                        update=a["update"],
                        gradient_max_abs=ga,
                        gradient_relative_l2=gr,
                        parameter_max_abs=pa,
                        parameter_relative_l2=pr,
                    )
                )
            loss = max(
                abs(a["losses"]["total"] - b["losses"]["total"])
                for a, b in zip(baseline.records, trainer.records)
            )
            record["numerics"] = dict(loss_max_abs=loss, updates=per_update)
            record["numerical_pass"] = loss <= tolerance["loss_max_abs"] and all(
                r["gradient_max_abs"] <= tolerance["gradient_max_abs"]
                and r["gradient_relative_l2"] <= tolerance["gradient_relative_l2"]
                and r["parameter_max_abs"] <= tolerance["parameter_max_abs"]
                for r in per_update
            )
        else:
            record["numerical_pass"] = True
        cases[name] = record
        trainers[name] = trainer
        write(out / (name + ".json"), record)
        print("GPU_CASE", name, elapsed, "PASS", record["numerical_pass"], flush=True)
        if not record["numerical_pass"]:
            # 拒绝不可靠算子，保留已验证CPU动态adjoint，不用后续性能掩盖失败。
            os.environ["LG_BEV_DYNAMIC_CUDA"] = "0"
            write(
                out / "decision.json",
                dict(
                    status="BLOCKED_NUMERICAL_CUDA_NOT_ADOPTED",
                    failed_case=name,
                    retained="CPU dynamic deformable adjoint + validated GPU fixed interpolation",
                    formal_updates=0,
                ),
            )
            return
    if (
        module_sha(runtime.model) != original_model
        or module_sha(encoder) != original_text
    ):
        raise RuntimeError("冻结状态改变")
    best = min(
        ("CUDA_native_stream", "CUDA_pinned_transfer"),
        key=lambda n: cases[n]["seconds"],
    )
    write(
        out / "decision.json",
        dict(
            status="GPU32_NUMERICAL_AND_THROUGHPUT_PASS",
            best_stream_case=best,
            speedup_cached=cases["CPU_dynamic_reference"]["seconds"]
            / cases["CUDA_dynamic"]["seconds"],
            B4epochs_point_hours=cases[best]["seconds_per_frame"] * 76492 / 3600,
            formal_updates=0,
            full_training_started=False,
            baseline_frozen_sha256=original_model,
            text_frozen_sha256=original_text,
            limitation="6epoch estimate still needs A2epoch native throughput; no formal performance evaluation",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/lg_bev_gpu_optimization_20261027"
    )
    parser.add_argument("--backend", choices=["1", "ordered"], default="1")
    args = parser.parse_args()
    benchmark(args.out.resolve(), args.backend)

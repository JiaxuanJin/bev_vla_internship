"""Flushed stage telemetry and fail-closed host/CUDA memory guards."""

from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path


def host_memory(pid=None):
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemAvailable", "MemTotal", "SwapFree"):
            values[key + "_mib"] = int(value.split()[0]) / 1024
    try:
        for line in Path(f"/proc/{pid or os.getpid()}/status").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in ("VmRSS", "VmHWM"):
                values[key + "_mib"] = int(value.split()[0]) / 1024
    except FileNotFoundError:
        pass
    return values


def memory_snapshot():
    import torch

    values = host_memory() | {"pid": os.getpid()}
    # Logging must not initialize a CUDA context before checkpoint loading.
    if torch.cuda.is_initialized():
        free, total = torch.cuda.mem_get_info()
        values.update(
            cuda_free_mib=free / 2**20,
            cuda_total_mib=total / 2**20,
            allocated_mib=torch.cuda.memory_allocated() / 2**20,
            reserved_mib=torch.cuda.memory_reserved() / 2**20,
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
        )
    return values


def memory_event(label, *, min_available_mib=0, min_cuda_free_mib=0):
    row = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "stage": label,
        **memory_snapshot(),
    }
    print("[memory] " + json.dumps(row), flush=True)
    path = os.environ.get("STAGE2_MEMORY_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
    if row["MemAvailable_mib"] < min_available_mib:
        raise MemoryError(
            f"LOW_HOST_MEMORY at {label}: {row['MemAvailable_mib']:.0f} MiB available"
        )
    if row.get("cuda_free_mib", float("inf")) < min_cuda_free_mib:
        raise MemoryError(f"LOW_CUDA_MEMORY at {label}")
    return row


@contextmanager
def memory_stage(label, *, minimum=0):
    memory_event(label + ".before", min_available_mib=minimum)
    try:
        yield
    finally:
        memory_event(label + ".after")


def release_memory():
    """Release dead Python objects and glibc arenas; never detach a live graph."""
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass

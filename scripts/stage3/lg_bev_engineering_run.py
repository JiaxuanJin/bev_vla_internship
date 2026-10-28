"""正式工程真实验收入口；全部大型产物在G盘，不自动启动完整长训。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.stage3.lg_bev_common import ROOT, sha, write
from scripts.stage3.lg_bev_engineering_cache import RollingCache
from scripts.stage3.lg_bev_engineering_data import FORMAL, fixed_frames, prepare_labels

OUT = ROOT / "results/lg_bev_engineering_20261027"


def environment(cache: RollingCache) -> dict:
    """读取宿主分区登记，所有临时目录指向G盘管理范围。"""
    (cache.root / "tmp").mkdir(exist_ok=True)
    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ[key] = str(cache.root / "tmp")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-ChildItem HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss | ForEach-Object { Get-ItemProperty $_.PSPath | Select-Object DistributionName,BasePath } | ConvertTo-Json -Compress",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        host = json.loads(result.stdout)
    except Exception as exc:
        raise RuntimeError("不能确认WSL宿主分区，拒绝大型写入") from exc
    return dict(
        wsl_host=host,
        workspace_host_partition="E:",
        large_writes_only=str(cache.root),
        g_free_bytes=shutil.disk_usage(cache.root).free,
        root_usage_bytes=cache.usage(),
        capacity_bytes=cache.capacity,
        reserve_bytes=cache.reserve,
        tmpdir=os.environ["TMPDIR"],
    )


def prepare(out: Path, run_id: str) -> None:
    """冻结32帧、工程补充和容差，再实际重建原始GT。"""
    cache = RollingCache()
    env = environment(cache)
    run = cache.root / "runs" / run_id
    run.mkdir(parents=True, exist_ok=False)
    rows = fixed_frames()
    supplement = dict(
        engineering_id=run_id,
        formal_contract_sha256=sha(FORMAL / "formal_execution_contract.json"),
        split_manifest_sha256=sha(FORMAL / "formal_split_manifest.json"),
        run=str(run),
        rows=rows,
        environment=env,
        training_target_or_budget_changed=False,
        full_training_authorized=False,
        shard_frames=8,
        cache_policy="30GB total root, freeG>=250GB; ownership only; FP32; regenerate fromscene prefix; all bigdata onG",
        formal_admission="fixed hash selected complete hot scenes <=10GB durable FP32; remaining scenes one8frame RAM FP32 shard with Gcatalog provenance; no frame/order/budget/loss changes; avoid writing full corpus each epoch",
        deterministic_adjoint="native CUDA forward unchanged; fixed-grid FP32 GPU adjoint; interpolation identical direct CPU ATen backward; MSDeform CPU reference; original tolerances preserved; failed GPU candidates disabled",
        can_bus="local CAN pose archive unavailable; exactly author converter zeros18 fallback, then authorget_data_info fills real ego pose/quaternion/yaw; not measured acceleration; 32train full-info metadata confirmed same fallback",
        tolerances_registered_before_gpu=dict(
            cache_storage_max_abs=0.0,
            raw_head_replay_max_abs=0.0,
            repeated_raw_bev_max_abs=0.0,
            attention_max_abs=0.0,
            loss_max_abs=1e-5,
            gradient_max_abs=2e-5,
            gradient_relative_l2=1e-4,
            parameter_max_abs=1e-6,
            cause="CUDA grid_sample and legacy deformable-attention backward use atomic accumulation; forward/cache must remain bitwise; backward tolerances apply only repeat execution, never storage loss",
        ),
        full_run_gate="READY requires real32frame data/cache/native/gradient/resume/evaluator gates; full run needs separate user authorization",
    )
    write(out / "engineering_supplement.json", supplement)
    write(
        run / "scope.json",
        dict(
            owner="LG-BEV_ENGINEERING",
            run_id=run_id,
            created=time.time(),
            only_train=True,
        ),
    )
    report = prepare_labels(cache, run, rows)
    write(out / "data_report.json", report)
    print(
        "REAL_DATA_READY",
        report["success"],
        report["label_count"],
        report["empty_query_fraction"],
        flush=True,
    )


def main() -> None:
    """只选择明确phase，不隐式跨入正式训练。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "audit"], required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--run-id", default="engineering_20261027_01")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("run-id只能是单层目录名")
    if args.phase == "prepare":
        prepare(args.out.resolve(), args.run_id)
    else:
        from scripts.stage3.lg_bev_engineering_audit import audit

        audit(args.out.resolve())


if __name__ == "__main__":
    main()

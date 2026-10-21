#!/usr/bin/env python3
"""离线运行第二阶段 CPU 回归，保存结果，不覆盖历史实验报告。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/stage2_regression/regression.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    commands = {
        "day1_pytest": [
            "-m", "pytest", "tests/test_stage2_day1.py", "-q", "-p", "no:cacheprovider",
        ],
        "step56_synthetic": [
            "-m", "scripts.stage2.run_step56_smoke",
            "--text-mode", "synthetic", "--feature-source", "synthetic",
            "--output-dir", str(output.parent / "step56_synthetic"),
        ],
        **{
            f"step{step}_self_test": [f"scripts/stage2/{script}.py", "--self-test"]
            for step, script in (
                (68, "run_step68_acceptance"),
                (69, "run_step69_loss_contract"),
                (70, "run_step70_joint_integration"),
                (71, "run_step71_stage2_review"),
                (72, "run_step72_real_joint_gradient"),
                (73, "run_step73_fill_gaps"),
            )
        },
    }
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
    }
    versions = {}
    for name in ("torch", "transformers", "numpy", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    files = sorted({
        *ROOT.glob("scripts/stage2/*.py"),
        *ROOT.glob("stage2/*.py"),
        *ROOT.glob("stage2/*.json"),
        *ROOT.glob("tests/test_stage2*.py"),
        ROOT / "requirements_stage2.txt",
    })
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "versions": versions,
        "mode": "offline_cpu_synthetic_regression",
        "real_model_inference_rerun": False,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
        "tests": {},
    }
    for name, command in commands.items():
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, *command], cwd=ROOT, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=180, check=False,
            )
            item = {
                "command": [sys.executable, *command],
                "status": "PASS" if proc.returncode == 0 else "FAIL",
                "exit_code": proc.returncode,
                "output_tail": proc.stdout[-6000:],
            }
        except subprocess.TimeoutExpired as exc:
            tail = exc.stdout or ""
            if isinstance(tail, bytes):
                tail = tail.decode("utf-8", errors="replace")
            item = {
                "command": [sys.executable, *command],
                "status": "FAIL", "reason": "TIMEOUT_180_SECONDS",
                "output_tail": tail[-6000:],
            }
        item["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["tests"][name] = item
        print(f"{name}: {item['status']}", flush=True)
        if item["status"] != "PASS":
            print(item["output_tail"], flush=True)
    passed = all(item["status"] == "PASS" for item in result["tests"].values())
    result["status"] = "PASS" if passed else "FAIL"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {output}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

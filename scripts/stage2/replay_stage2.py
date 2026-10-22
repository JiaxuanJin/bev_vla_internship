#!/usr/bin/env python3
"""验证冻结源码并在独立低内存进程重放旧 20 步权重的验证；禁止训练。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def verify_snapshot(snapshot, protocol_path):
    protocol = json.loads(protocol_path.read_text())
    mapping = {}
    for original, expected in protocol["source_sha256"].items():
        archived = snapshot / "scripts/stage2" / Path(original).name
        if digest(archived) != expected:
            raise ValueError(f"冻结源码不匹配: {archived}")
        mapping[str(Path(original).resolve())] = archived
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=ROOT / "results/stage2_final_audit_2026-10-22/legacy_source",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "outputs/stage2_oom_fix_20261022_step1/protocol/protocol.json",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=ROOT
        / "outputs/stage2_oom_fix_20261022_continue/continue/fusion_step020.pt",
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default="125237365f3b456384009168ad8b97df6f0e46805331f79c7e1c344d4c2e9d5a",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--min-host-available-mib", type=int, default=2048)
    parser.add_argument("--max-worker-rss-mib", type=int, default=11264)
    args = parser.parse_args()
    args.snapshot, args.protocol, args.resume = [
        p.resolve() for p in (args.snapshot, args.protocol, args.resume)
    ]
    mapping = verify_snapshot(args.snapshot, args.protocol)
    if digest(args.resume) != args.expected_checkpoint_sha256:
        raise ValueError("旧融合 checkpoint 哈希不匹配")
    if args.verify_only:
        print(
            json.dumps(
                {
                    "status": "PASS_FROZEN_SOURCE_AND_WEIGHT_HASHES",
                    "source_files": len(mapping),
                    "checkpoint_sha256": digest(args.resume),
                    "optimizer_steps": 0,
                }
            ),
            flush=True,
        )
        return 0
    if args.output is None:
        parser.error("重放必须指定全新 --output 目录")
    args.output = args.output.resolve()
    if args.worker:
        # 真正执行冻结模块；只将原源码地址的哈希读取重定位到已逐文件验证的快照。
        # 数据、权重、配置和协议原文的哈希检查全部保留，绝不改写旧协议。
        sys.path.insert(0, str(args.snapshot))
        from scripts.stage2 import micro_training

        if not Path(micro_training.__file__).resolve().is_relative_to(args.snapshot):
            raise RuntimeError("未导入冻结源码")
        original_digest = micro_training.sha256
        micro_training.sha256 = lambda path: original_digest(
            mapping.get(str(Path(path).resolve()), path)
        )
        seed = json.loads(args.protocol.read_text())["seed"]
        sys.argv = [
            str(args.snapshot / "scripts/stage2/run_stage2.py"),
            "--mode",
            "micro-train",
            "--micro-worker",
            "validate",
            "--protocol",
            str(args.protocol),
            "--resume",
            str(args.resume),
            "--seed",
            str(seed),
            "--model",
            str(ROOT / "models/distilbert-base-uncased"),
            "--min-host-available-mib",
            str(args.min_host_available_mib),
            "--max-worker-rss-mib",
            str(args.max_worker_rss_mib),
            "--output",
            str(args.output),
        ]
        runpy.run_path(sys.argv[0], run_name="__main__")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=False)
    from scripts.stage2.micro_process import monitor_command

    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--snapshot",
        str(args.snapshot),
        "--protocol",
        str(args.protocol),
        "--resume",
        str(args.resume),
        "--expected-checkpoint-sha256",
        args.expected_checkpoint_sha256,
        "--min-host-available-mib",
        str(args.min_host_available_mib),
        "--max-worker-rss-mib",
        str(args.max_worker_rss_mib),
        "--output",
        str(args.output),
    ]
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MPLCONFIGDIR": "/tmp/stage2-mpl",
        "STAGE2_MEMORY_LOG": str(args.output.parent / "memory.jsonl"),
    }
    summary = monitor_command(
        command,
        args.output.parent,
        args,
        worker="frozen_validate",
        env=env,
        result_path=args.output,
    )
    print(
        json.dumps({"status": summary["result"]["status"], "optimizer_steps": 0}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    # 当作脚本运行时可找到当前工程；worker 之后会将冻结源码置于最前。
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())

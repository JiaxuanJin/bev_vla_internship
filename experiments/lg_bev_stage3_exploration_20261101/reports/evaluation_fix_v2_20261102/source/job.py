"""固定评价；逐帧保存原始预测，累计器只从真实安全退出的边界恢复。"""

import gzip
import pickle
import resource
import signal
import time
import torch
from core import read, require, selected_rows, seed_cpu, sha, tensor_sha
from lifecycle import save_checkpoint, saved_end
from eval_driver import (
    EvaluationState,
    adapters_for_evaluation,
    evaluate_frame,
    ground_truth_coverage,
)


def evaluate(args, config, protocol, store, rid, ident, trained, previous):
    from runtime_v2 import Runtime
    from scripts.stage3.lg_bev_engineering_train import frozen_check
    from checks import check_records

    stopped = [False]
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, lambda *_: stopped.__setitem__(0, True))
    session = f"{rid}/sessions/{args.session_id}"

    def stop():
        return (
            stopped[0]
            or time.monotonic() - args.started >= 26400
            or store.path(session + "/stop.request.json").exists()
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    development = args.action == "development"
    rows = selected_rows(config, "development" if development else "diagnostic")
    selected = protocol["development_tokens" if development else "diagnostic_tokens"]
    by_token = {r["sample_token"]: r for r in rows}
    rows = [by_token[t] for t in selected]
    require(len(rows) == (320 if development else 8), "Wrong frame list")
    runtime = Runtime(config, store, development_authorized=development)
    # 读取标签和媒体前，限制为该动作已经封存的清单。
    runtime.rows = {r["sample_token"]: r for r in rows}
    seed_cpu(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    adapters = adapters_for_evaluation(config, trained)
    digests = {k: tensor_sha(a.state_dict()) for k, a in adapters.items()}
    require(
        digests["enhanced"] == protocol["trained_adapter_sha256"],
        "Wrong Enhanced adapter",
    )
    frozen_check(runtime.model, runtime.encoder, trained["frozen_sha256"])
    cursor, metrics, artifacts, check_count = 0, EvaluationState(), [], 0
    if previous:
        require(
            previous["trained_adapter_sha256"] == protocol["trained_adapter_sha256"],
            "Changed evaluation weights",
        )
        cursor, metrics, artifacts, check_count = (
            previous[k] for k in ("cursor", "metrics", "artifacts", "check_count")
        )
        for a in artifacts:
            require(
                sha(store.path(a["path"])) == a["sha256"],
                "Raw prediction artifact changed",
            )
    coverage_path = store.path(rid + "/gt_coverage.json")
    if not coverage_path.exists():
        coverage = ground_truth_coverage(runtime, rows, stop)
        if coverage is not None:
            store.json(
                rid + "/gt_coverage.json",
                dict(
                    identity=ident,
                    trained_adapter_sha256=protocol["trained_adapter_sha256"],
                    **coverage,
                ),
            )
    coverage_sha = sha(coverage_path) if coverage_path.exists() else None
    if coverage_sha:
        require(read(coverage_path)["identity"] == ident, "Changed coverage identity")
    if previous and cursor:
        require(previous["coverage_sha256"] == coverage_sha, "Changed GT coverage")

    def state():
        return dict(
            identity=ident,
            cursor=cursor,
            metrics=metrics,
            artifacts=artifacts,
            check_count=check_count,
            trained_adapter_sha256=protocol["trained_adapter_sha256"],
            coverage_sha256=coverage_sha,
            preflight_passed=not development and cursor == len(rows),
        )

    while coverage_sha and cursor < len(rows) and not stop():
        start = time.monotonic()
        records = evaluate_frame(
            runtime, adapters, rows[cursor]["sample_token"], cursor
        )
        if not development:
            check_count += check_records(records)
        for record in records:
            metrics.add(record)
        raw = gzip.compress(
            pickle.dumps(
                dict(index=cursor, token=rows[cursor]["sample_token"], records=records),
                protocol=5,
            ),
            compresslevel=3,
            mtime=0,
        )
        rel = f"{rid}/predictions/{cursor:04d}.pkl.gz"
        path = store.bytes(rel, raw)
        artifacts.append(
            dict(
                index=cursor,
                token=rows[cursor]["sample_token"],
                path=rel,
                sha256=sha(path),
                bytes=len(raw),
            )
        )
        cursor += 1
        store.json(
            f"{session}/frames/{cursor:04d}.json",
            dict(
                cursor=cursor,
                total_frames=len(rows),
                token=rows[cursor - 1]["sample_token"],
                wall_seconds=time.monotonic() - start,
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                raw_prediction=artifacts[-1],
            ),
        )
        print(f"{args.action}: {cursor}/{len(rows)} frames committed", flush=True)
        if cursor % 32 == 0:
            save_checkpoint(
                store, rid, args.session_id, state(), f"periodic_{cursor:04d}"
            )
    frozen_check(runtime.model, runtime.encoder, trained["frozen_sha256"])
    require(
        digests == {k: tensor_sha(a.state_dict()) for k, a in adapters.items()},
        "Evaluation mutated adapter",
    )
    outcome = (
        ("COMPLETED" if development else "PREFLIGHT_PASS")
        if cursor == len(rows)
        else "PAUSED"
    )
    if cursor == len(rows):
        store.json(rid + "/metrics.json", metrics.result())
        store.json(
            rid + "/raw_manifest.json", dict(identity=ident, artifacts=artifacts)
        )
    relative, digest = save_checkpoint(store, rid, args.session_id, state(), "end")
    saved_end(
        store,
        rid,
        args.session_id,
        ident,
        relative,
        digest,
        outcome,
        evaluated_frames=cursor,
        preflight_passed=not development and cursor == len(rows),
        check_count=check_count,
        all_a_only_heads_bitwise_equal=True,
        frozen_sha256=trained["frozen_sha256"],
        adapter_digests=digests,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )

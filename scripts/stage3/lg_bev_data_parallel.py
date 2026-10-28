"""至多两个CPU进程构建剩余真实标签；复用原构建器和已提交标签，不重分split。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.stage3.lg_bev_common import ROOT, read, sha, write
from scripts.stage3.lg_bev_engineering_cache import CACHE_ROOT, RollingCache
from scripts.stage3.lg_bev_engineering_data import formal_frames, prepare_labels

PARENT = CACHE_ROOT / "runs/formal_gt_20261027_v1"


def worker(spec: Path) -> None:
    """工作分片只分生成任务；GT计算仍调用原prepare_labels，完整scene上下文不裁剪。"""
    config = read(spec)
    cache = RollingCache()
    run = Path(config["run"])
    if not run.resolve().is_relative_to(PARENT.resolve()):
        raise ValueError("标签分片超出正式GT归属根目录")
    if config["builder_sha256"] != sha(
        ROOT / "scripts/stage3/lg_bev_engineering_data.py"
    ):
        raise ValueError("GT构建器源码发生变化")
    if (run / "data_report.json").exists():
        return
    from scripts.stage3 import lg_bev_engineering_data as data_module

    original = data_module.SceneData

    class IndexedScene(original):
        def __init__(self, rows, **kwargs):
            path = CACHE_ROOT / "metadata" / (rows[0]["scene_token"] + ".pt")
            start = time.monotonic()
            while not (path.exists() and path.with_suffix(".sha256").exists()):
                if time.monotonic() - start > 3600:
                    raise TimeoutError("等待已校验场景索引超时，不回退错误/竞争时序")
                time.sleep(1)
            super().__init__(rows, **kwargs)

    data_module.SceneData = IndexedScene
    prepare_labels(cache, run, config["rows"])


def build() -> None:
    """合并已提交记录与两份不交叉剩余任务，完整19123帧成功才写最终report。"""
    cache = RollingCache()
    rows = formal_frames()
    if (PARENT / "data_report.json").exists():
        raise ValueError("完整标签报告已经存在，拒绝重复构建")
    scope = read(PARENT / "scope.json")
    if scope["builder_sha256"] != sha(
        ROOT / "scripts/stage3/lg_bev_engineering_data.py"
    ):
        raise ValueError("正式GT归属/构建器不符")
    existing = {
        r["sample_token"]: r
        for (encoded,) in cache.db.execute(
            "SELECT record FROM labels WHERE run=?", (str(PARENT),)
        )
        for r in [json.loads(encoded)]
    }
    remaining = [r for r in rows if r["sample_token"] not in existing]
    groups = {}
    for r in remaining:
        groups.setdefault(r["scene_token"], []).append(r)
    parts = [[], []]
    for i, group in enumerate(groups.values()):
        parts[i % 2].extend(group)
    started = time.perf_counter()
    processes, configs = [], []
    for i, selected in enumerate(parts):
        run = PARENT / "parallel" / f"part_{i}"
        run.mkdir(parents=True, exist_ok=True)
        spec = run / "spec.json"
        config = dict(
            run=str(run),
            rows=selected,
            builder_sha256=scope["builder_sha256"],
            fixed_split_unchanged=True,
            worker_count=2,
        )
        if spec.exists():
            if read(spec) != config:
                raise ValueError("已有分片任务不同，拒绝覆盖")
        else:
            write(spec, config)
        configs.append(config)
        log = run / ("build_" + str(time.time_ns()) + ".log")
        stream = log.open("x")
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "scripts.stage3.lg_bev_data_parallel",
                    "--worker",
                    str(spec),
                ],
                stdout=stream,
                stderr=subprocess.STDOUT,
                cwd=ROOT,
            )
        )
        stream.close()
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None, 0) for p in processes):
            for p in processes:
                if p.poll() is None:
                    p.send_signal(signal.SIGINT)
            raise RuntimeError("标签分片失败；保留已提交内容，禁止伪造完整报告")
        time.sleep(1)
    reports = [read(Path(c["run"]) / "data_report.json") for c in configs]
    merged = dict(existing)
    for report in reports:
        if report["status"] != "PASS" or report["failures"]:
            raise ValueError("标签分片未完整通过")
        for item in report["frames"]:
            if item["sample_token"] in merged:
                raise ValueError("标签分片出现重复frame")
            merged[item["sample_token"]] = item
    if set(merged) != {r["sample_token"] for r in rows} or len(merged) != 19123:
        raise ValueError("标签没有精确覆盖完整19123帧")
    records = [merged[r["sample_token"]] for r in rows]
    paths = {r["label_path"]: r["label_sha256"] for r in records}
    for p, digest in paths.items():
        if sha(Path(p)) != digest:
            raise ValueError("最终GT块SHA校验失败")
    result = dict(
        status="PASS",
        success=19123,
        failures=[],
        frames=records,
        label_count=sum(r["queries"] for r in records),
        empty_queries=sum(r["empty_queries"] for r in records),
        target_instance_occurrences=sum(
            r["target_instance_occurrences"] for r in records
        ),
        seconds=time.perf_counter() - started,
        language_source="GT_DERIVED_TEMPLATE_NOT_DRIVELM",
        raw_metadata_sha256=reports[0]["raw_metadata_sha256"],
        reused_committed_frames=len(existing),
        parallel_workers=2,
        builder_sha256=scope["builder_sha256"],
        orchestration_sha256=sha(Path(__file__)),
        record_order="sealed formal frame order; no training schedule change",
    )
    result["empty_query_fraction"] = result["empty_queries"] / result["label_count"]
    if reports[0]["raw_metadata_sha256"] != reports[1]["raw_metadata_sha256"]:
        raise ValueError("分片原始来源SHA不一致")
    with cache.locked():
        cache.reserve_for_artifact(len(json.dumps(result).encode()) + 2**20)
        write(PARENT / "data_report.json", result)
        cache.db.executemany(
            "INSERT OR REPLACE INTO labels VALUES(?,?,?)",
            [(str(PARENT), r["sample_token"], json.dumps(r)) for r in records],
        )
        cache.db.commit()
    print("FULL_TRAIN_19123_LABELS_PASS", result["label_count"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    worker(args.worker) if args.worker else build()

"""复用已提交GT，按完整场景批量持久化剩余标签，减少NTFS逐行fsync。"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from scripts.stage3.lg_bev_common import ROOT, read, sha, write
from scripts.stage3.lg_bev_engineering_cache import (
    CACHE_ROOT,
    RollingCache,
    atomic_save,
)
from scripts.stage3.lg_bev_engineering_data import (
    SceneData,
    formal_frames,
    masks,
    nested_equal,
)
from scripts.stage3.lg_bev_data_parallel import PARENT

RUN = CACHE_ROOT / "runs/formal_gt_batch_20261027_v1"


def save_owned_block(cache: RollingCache, path: Path, packed: dict, size: int) -> None:
    """调用方持锁；遗留pending完整则核对后恢复，损坏则改名留存再生。"""
    if not path.resolve().is_relative_to(cache.root.resolve()):
        raise ValueError("待恢复标签块越界")
    cache.reserve_for_artifact(size + 2**20)
    if path.exists():
        if not nested_equal(
            torch.load(path, map_location="cpu", weights_only=False), packed
        ):
            raise ValueError("未登记块与重建内容不同，拒绝覆盖")
        return
    pending = path.with_name(path.name + ".pending")
    if pending.exists():
        try:
            valid = nested_equal(
                torch.load(pending, map_location="cpu", weights_only=False), packed
            )
        except Exception:
            valid = False
        if valid:
            os.rename(pending, path)
            return
        os.rename(pending, path.with_name(path.name + f".interrupted_{time.time_ns()}"))
    atomic_save(path, packed)


def merged_records(cache: RollingCache) -> dict:
    """仅复用已登记且SHA正确的块；重复token须逐字段一致后去重。"""
    selected = {r["sample_token"] for r in formal_frames()}
    records, loaded, verified = {}, {}, set()
    entries = cache.db.execute("SELECT run,record FROM labels").fetchall()
    entries.sort(key=lambda v: (v[0], json.loads(v[1])["label_path"]))
    for run, encoded in entries:
        if not (
            run == str(PARENT)
            or run.startswith(str(PARENT / "parallel") + "/")
            or run.startswith(str(RUN) + "/")
        ):
            continue
        record = json.loads(encoded)
        token = record["sample_token"]
        if token not in selected:
            raise ValueError("标签包含非train帧")
        path = record["label_path"]
        if path not in loaded:
            if path not in verified and sha(Path(path)) != record["label_sha256"]:
                raise ValueError("已提交GT块损坏")
            verified.add(path)
            loaded[path] = torch.load(path, map_location="cpu", weights_only=False)
        if token in records:
            previous = records[token]["label_path"]
            if previous not in loaded:
                loaded[previous] = torch.load(
                    previous, map_location="cpu", weights_only=False
                )
            if not nested_equal(loaded[path][token], loaded[previous][token]):
                raise ValueError("重复帧GT内容冲突")
        else:
            records[token] = record
        # 旧块有多个token，保持有界内存；仅最近块用于实际重复比较。
        if len(loaded) > 4:
            keep = {path, records[token]["label_path"]}
            loaded = {p: v for p, v in loaded.items() if p in keep}
    return records


def worker(spec: Path) -> None:
    """GT计算调用原SceneData.label/masks，只有分片和提交粒度改变。"""
    config = read(spec)
    if config["builder_sha256"] != sha(
        ROOT / "scripts/stage3/lg_bev_engineering_data.py"
    ):
        raise ValueError("原GT算法变化")
    run = Path(config["run"])
    if not run.resolve().is_relative_to(RUN.resolve()):
        raise ValueError("GT分片越界")
    cache = RollingCache()
    existing = {
        json.loads(value)["sample_token"]
        for (value,) in cache.db.execute(
            "SELECT record FROM labels WHERE run=?", (str(run),)
        )
    }
    full, groups = formal_frames(), {}
    for r in full:
        groups.setdefault(r["scene_token"], []).append(r)
    for scene, selected in config["groups"].items():
        missing = set(selected) - existing
        if not missing:
            continue
        data = SceneData(groups[scene])
        packed, records = {}, []
        for row in data.rows:
            token = row["sample_token"]
            if token not in missing:
                continue
            item = data.label(token)
            qs = item["queries"]
            records.append(
                dict(
                    sample_token=token,
                    scene_token=scene,
                    queries=len(qs),
                    empty_queries=sum(not masks(item, q)["target"].any() for q in qs),
                    target_instance_occurrences=sum(len(q["selected"]) for q in qs),
                    object_instances=len(item["boxes"]),
                    map_instances=len(item["map_ids"]),
                )
            )
            item = dict(item)
            for field in ("object_masks", "map_masks"):
                value = item.pop(field).numpy()
                item[field + "_shape"] = list(value.shape)
                item[field + "_packed"] = torch.from_numpy(
                    np.packbits(value.reshape(-1))
                )
            packed[token] = item
        path = run / (scene + ".pt")
        stream = io.BytesIO()
        torch.save(packed, stream)
        size = stream.tell()
        with cache.locked():
            save_owned_block(cache, path, packed, size)
        digest = sha(path)
        for record in records:
            record.update(label_path=str(path), label_sha256=digest)
        # 一整个原子块对应一次journal fsync、一次SQLite FULL提交；崩溃后
        # 未登记但完整的块仍逐字段复核，不少保存任何实例/帧/来源。
        with cache.locked():
            cache.reserve_for_artifact(2**20)
            with (run / "labels_commit.jsonl").open("a") as journal:
                journal.write("".join(json.dumps(r) + "\n" for r in records))
                journal.flush()
                os.fsync(journal.fileno())
            cache.db.executemany(
                "INSERT OR REPLACE INTO labels VALUES(?,?,?)",
                [(str(run), r["sample_token"], json.dumps(r)) for r in records],
            )
            cache.db.commit()
        print("SCENE_GT_COMMITTED", scene, len(records), flush=True)
        del data, packed, stream


def build() -> None:
    """固定两份scene任务，不改变训练帧顺序，完整覆盖才封存最终report。"""
    cache, started = RollingCache(), time.perf_counter()
    if (PARENT / "data_report.json").exists():
        raise ValueError("最终报告已存在，拒绝覆盖")
    RUN.mkdir(exist_ok=True)
    specs = [RUN / f"part_{i}" / "spec.json" for i in range(2)]
    if not all(p.exists() for p in specs):
        records = merged_records(cache)
        rows, groups = formal_frames(), {}
        for r in rows:
            if r["sample_token"] not in records:
                groups.setdefault(r["scene_token"], []).append(r["sample_token"])
        parts = [{}, {}]
        for i, (scene, tokens) in enumerate(groups.items()):
            parts[i % 2][scene] = tokens
        for i, path in enumerate(specs):
            path.parent.mkdir(exist_ok=True)
            write(
                path,
                dict(
                    run=str(path.parent),
                    groups=parts[i],
                    builder_sha256=sha(
                        ROOT / "scripts/stage3/lg_bev_engineering_data.py"
                    ),
                    reused_frames=len(records),
                ),
            )
    processes = []
    for spec in specs:
        log = spec.parent / ("build_" + str(time.time_ns()) + ".log")
        with log.open("x") as stream:
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "scripts.stage3.lg_bev_gt_completion",
                        "--worker",
                        str(spec),
                    ],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    cwd=ROOT,
                )
            )
    while any(p.poll() is None for p in processes):
        if any(p.poll() not in (None, 0) for p in processes):
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            raise RuntimeError("分片失败，保留全部提交；不生成完整报告")
        time.sleep(1)
    records = merged_records(cache)
    rows = formal_frames()
    if len(records) != 19123 or set(records) != {r["sample_token"] for r in rows}:
        raise ValueError("未精确覆盖19123帧")
    ordered = [records[r["sample_token"]] for r in rows]
    from scripts.stage2.spatial_annotations import DATA

    result = dict(
        status="PASS",
        success=19123,
        failures=[],
        frames=ordered,
        label_count=sum(r["queries"] for r in ordered),
        empty_queries=sum(r["empty_queries"] for r in ordered),
        target_instance_occurrences=sum(
            r["target_instance_occurrences"] for r in ordered
        ),
        seconds=time.perf_counter() - started,
        language_source="GT_DERIVED_TEMPLATE_NOT_DRIVELM",
        raw_metadata_sha256={
            str(DATA / "v1.0-trainval" / n): sha(DATA / "v1.0-trainval" / n)
            for n in (
                "sample_annotation.json",
                "calibrated_sensor.json",
                "ego_pose.json",
            )
        },
        builder_sha256=sha(ROOT / "scripts/stage3/lg_bev_engineering_data.py"),
        orchestration_sha256=sha(Path(__file__)),
        storage_supplement="whole-scene GT blocks; once-per-block journal fsync/SQLite FULL; exact original label math",
        record_order="sealed formal frame order",
    )
    result["empty_query_fraction"] = result["empty_queries"] / result["label_count"]
    with cache.locked():
        cache.reserve_for_artifact(len(json.dumps(result).encode()) + 2**20)
        write(PARENT / "data_report.json", result)
        cache.db.executemany(
            "INSERT OR REPLACE INTO labels VALUES(?,?,?)",
            [(str(PARENT), r["sample_token"], json.dumps(r)) for r in ordered],
        )
        cache.db.commit()
    print("FULL_19123_TRAIN_GT_PASS", result["label_count"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    worker(args.worker) if args.worker else build()

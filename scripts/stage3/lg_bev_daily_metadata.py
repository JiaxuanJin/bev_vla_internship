"""一次流式索引替代逐场景重复扫大JSON；标签算法和SceneData均不修改。"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
import uuid

import torch

from scripts.stage2.low_memory_dataset import json_records
from scripts.stage2.spatial_annotations import DATA
from scripts.stage3.lg_bev_common import read, sha, write
from scripts.stage3.lg_bev_engineering_cache import (
    CACHE_ROOT,
    RollingCache,
    atomic_save,
)
from scripts.stage3.lg_bev_engineering_data import formal_frames, nested_equal

DIRECTORY = CACHE_ROOT / "metadata_bulk_train_v1"
DATABASE = DIRECTORY / "source.sqlite3"
TABLES = [
    "sample_data",
    "sample_annotation",
    "sample",
    "ego_pose",
    "calibrated_sensor",
    "instance",
    "category",
    "attribute",
    "visibility",
    "sensor",
    "log",
    "scene",
    "map",
]


def build() -> None:
    """只索引正式train的sample/annotation/传感器引用，公共表按原SDK保留。"""
    cache = RollingCache()
    DIRECTORY.mkdir(exist_ok=True)
    if DATABASE.exists():
        manifest = read(DIRECTORY / "manifest.json")
        if sha(DATABASE) != manifest["sha256"]:
            raise ValueError("派生元数据索引损坏")
        return
    with cache.locked():
        cache.reserve_for_artifact(2_000_000_000)
    # OFF journal仅用于全新可重建临时文件；中断文件保留，绝不续接半事务。
    pending = DIRECTORY / ("source_heap_" + uuid.uuid4().hex[:10] + ".pending.sqlite3")
    db = sqlite3.connect(pending)
    db.execute("PRAGMA temp_store=MEMORY")
    # 只作用于可重建pending索引；256MiB页缓存避免NTFS随机页反复写入。
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA cache_size=-262144")
    db.execute(
        "CREATE TABLE IF NOT EXISTS records(name TEXT,token TEXT,seq INTEGER,scene TEXT,payload TEXT)"
    )
    # pending只顺序写heap，完成后统一建索引；不改变主Rolling Cache的FULL事务。
    # 没有完整性检查/SHA/逐字段SDK对照前，任何pending都不能成为训练输入。
    legacy = DIRECTORY / "source.pending.sqlite3"
    if (
        legacy.exists()
        and db.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 0
    ):
        previous = sqlite3.connect(f"file:{legacy}?mode=ro", uri=True)
        if previous.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("待继承的旧派生索引不完整，保留现场并停止")
        cursor = previous.execute("SELECT name,token,seq,scene,payload FROM records")
        while True:
            batch = cursor.fetchmany(1000)
            if not batch:
                break
            db.executemany("INSERT INTO records VALUES(?,?,?,?,?)", batch)
        db.commit()
        previous.close()
        print("INDEX_COMMITTED_ROWS_INHERITED", flush=True)
    rows = formal_frames()
    scene_of = {r["sample_token"]: r["scene_token"] for r in rows}
    wanted = {k: set() for k in ("ego_pose", "calibrated_sensor", "instance")}
    # 从已提交部分恢复，不删除已有派生数据；输入JSON仍顺序读取校验。
    for (payload,) in db.execute(
        "SELECT payload FROM records WHERE name='sample_data'"
    ):
        item = json.loads(payload)
        wanted["ego_pose"].add(item["ego_pose_token"])
        wanted["calibrated_sensor"].add(item["calibrated_sensor_token"])
    for (payload,) in db.execute(
        "SELECT payload FROM records WHERE name='sample_annotation'"
    ):
        wanted["instance"].add(json.loads(payload)["instance_token"])
    counts = {}
    existing = {
        name: (count, seq)
        for name, count, seq in db.execute(
            "SELECT name,COUNT(*),MAX(seq) FROM records GROUP BY name"
        )
    }
    started = time.perf_counter()
    for name in TABLES:
        count, last_seq = existing.get(name, (0, -1))
        last_seq = -1 if last_seq is None else last_seq
        for seq, item in enumerate(
            json_records(DATA / "v1.0-trainval" / (name + ".json"))
        ):
            if seq <= last_seq:
                continue
            scene = scene_of.get(item.get("sample_token", item["token"]))
            if name in ("sample_data", "sample_annotation", "sample"):
                if scene is None or (
                    name == "sample_data" and not item["is_key_frame"]
                ):
                    continue
            elif name in wanted and item["token"] not in wanted[name]:
                continue
            if name == "sample_data":
                wanted["ego_pose"].add(item["ego_pose_token"])
                wanted["calibrated_sensor"].add(item["calibrated_sensor_token"])
            if name == "sample_annotation":
                wanted["instance"].add(item["instance_token"])
            db.execute(
                "INSERT INTO records VALUES(?,?,?,?,?)",
                (name, item["token"], seq, scene, json.dumps(item)),
            )
            count += 1
            if count % 10000 == 0:
                db.commit()
                with cache.locked():
                    cache.reserve_for_artifact(32_000_000)
        db.commit()
        counts[name] = count
        print("INDEX_TABLE", name, count, flush=True)
    assert counts["sample"] == 19123
    db.execute("CREATE UNIQUE INDEX record_identity ON records(name,token)")
    db.execute("CREATE INDEX by_scene ON records(name,scene,seq)")
    db.commit()
    if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise ValueError("派生索引完整性检查失败")
    db.close()
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.rename(pending, DATABASE)
    write(
        DIRECTORY / "manifest.json",
        dict(
            sha256=sha(DATABASE),
            rows=counts,
            seconds=time.perf_counter() - started,
            frame_count=19123,
            scope="仅正式train样本；公共scene/log/map表按旧SDK原样保留",
        ),
    )


def populate() -> None:
    """用同一NuScenes SDK构造相同有界表；既有场景逐字段一致，不覆盖。"""
    from nuscenes import NuScenes

    cache = RollingCache()
    if sha(DATABASE) != read(DIRECTORY / "manifest.json")["sha256"]:
        raise ValueError("派生索引SHA不符")
    db = sqlite3.connect(f"file:{DATABASE}?mode=ro", uri=True)
    rows = formal_frames()
    scene_tokens = list(dict.fromkeys(r["scene_token"] for r in rows))
    stats = {
        name: (
            str((DATA / "v1.0-trainval" / name).resolve()),
            (DATA / "v1.0-trainval" / name).stat().st_size,
            (DATA / "v1.0-trainval" / name).stat().st_mtime_ns,
        )
        for name in [
            "sample.json",
            "sample_data.json",
            "sample_annotation.json",
            "ego_pose.json",
            "calibrated_sensor.json",
        ]
    }
    checked = created = 0
    for scene in scene_tokens:
        selected = {
            name: [
                json.loads(v)
                for (v,) in db.execute(
                    "SELECT payload FROM records WHERE name=? AND scene=? ORDER BY seq",
                    (name, scene),
                )
            ]
            for name in ("sample", "sample_data", "sample_annotation")
        }
        wanted = dict(
            ego_pose={r["ego_pose_token"] for r in selected["sample_data"]},
            calibrated_sensor={
                r["calibrated_sensor_token"] for r in selected["sample_data"]
            },
            instance={r["instance_token"] for r in selected["sample_annotation"]},
        )

        class Indexed(NuScenes):
            def __load_table__(self, name):
                if name in selected:
                    return selected[name]
                if name in wanted:
                    tokens = sorted(wanted[name])
                    if not tokens:
                        return []
                    query = (
                        "SELECT payload FROM records WHERE name=? AND token IN ("
                        + ",".join("?" for _ in tokens)
                        + ") ORDER BY seq"
                    )
                    values = db.execute(query, [name] + tokens)
                else:
                    values = db.execute(
                        "SELECT payload FROM records WHERE name=? ORDER BY seq", (name,)
                    )
                return [json.loads(v) for (v,) in values]

        nusc = Indexed(version="v1.0-trainval", dataroot=str(DATA), verbose=False)
        tables = {name: getattr(nusc, name) for name in nusc.table_names}
        tables["map"] = [
            {k: v for k, v in r.items() if k != "mask"} for r in tables["map"]
        ]
        value = dict(tables=tables, source_stats=stats)
        path = CACHE_ROOT / "metadata" / (scene + ".pt")
        with cache.locked():
            if path.exists():
                prior = torch.load(path, map_location="cpu", weights_only=False)
                if sha(path) != path.with_suffix(
                    ".sha256"
                ).read_text().strip() or not nested_equal(prior, value):
                    raise ValueError("流式索引与原场景SDK表不一致：" + scene)
                checked += 1
            else:
                cache.reserve_for_artifact(len(json.dumps(tables).encode()) + 2**20)
                atomic_save(path, value)
                path.with_suffix(".sha256").write_text(sha(path))
                created += 1
        print("METADATA_SCENE", scene, checked, created, flush=True)
        del nusc, tables, value
    write(
        DIRECTORY / "equivalence.json",
        dict(
            status="PASS",
            scenes=len(scene_tokens),
            existing_exact_comparisons=checked,
            newly_indexed=created,
            same_original_sdk=True,
            labels_or_sensor_values_changed=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["build", "populate"])
    args = parser.parse_args()
    build() if args.phase == "build" else populate()

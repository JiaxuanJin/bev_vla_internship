"""只读盘点真实唯一帧、传感器覆盖和历史隔离；不生成 holdout GT 或预测。"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import pickle
import time
import zipfile

from scripts.stage2.low_memory_dataset import json_records
from scripts.stage3.lg_bev_common import ROOT, read, sha, write

OUT = ROOT / "results/lg_bev_formal_training_audit_20261027"
RAW = Path("/mnt/g/BEV_VLA_Datasets/nuscenes")
META = Path("/mnt/g/BEV_VLA_Datasets/nuscenes_trainval_meta/v1.0-trainval")
CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


def select_frame_split(scene_name: str, official: str, quarantined: set[str]) -> str:
    """盘点不把已暴露验证场景重新分配为训练；新split由后续协议固定。"""
    if scene_name in quarantined:
        return "historical_quarantine"
    return "eligible_official_train" if official == "train" else "official_val_reserved"


def qa_summary(path: Path, sample_tokens: set[str]) -> tuple[dict, set[str]]:
    """按真实 sample token 去重 QA，指令数不等同独立帧数。"""
    data = read(path)
    frames, qa_count, answers, objects, explicit_ids = set(), 0, 0, 0, 0
    for scene in data.values():
        for token, frame in scene.get("key_frames", {}).items():
            frames.add(token)
            for pairs in frame.get("QA", {}).values():
                if not isinstance(pairs, list):
                    continue
                qa_count += len(pairs)
                answers += sum(
                    isinstance(pair, dict) and bool(pair.get("A")) for pair in pairs
                )
            for value in frame.get("key_object_infos", {}).values():
                objects += 1
                if isinstance(value, dict) and any(
                    key in value
                    for key in (
                        "annotation_token",
                        "sample_annotation_token",
                        "instance_token",
                    )
                ):
                    explicit_ids += 1
    result = dict(
        path=str(path),
        resolved=str(path.resolve()),
        bytes=path.stat().st_size,
        sha256=sha(path),
        scenes=len(data),
        unique_frames=len(frames),
        qa_pairs=qa_count,
        answered_pairs=answers,
        key_objects=objects,
        objects_with_explicit_3d_token_field=explicit_ids,
        raw_nuscenes_token_matches=len(frames & sample_tokens),
        use="QA may support language analysis; 3D/instance association must be audited, not assumed",
    )
    del data
    gc.collect()
    return result, frames


def inventory(out: Path) -> None:
    """流式读取原始记录；统计真实文件存在，不解压大包，不读取holdout几何。"""
    from nuscenes.utils.splits import create_splits_scenes

    started = time.perf_counter()
    scenes = {r["token"]: r for r in read(META / "scene.json")}
    logs = {r["token"]: r for r in read(META / "log.json")}
    samples = {r["token"]: r for r in read(META / "sample.json")}
    splits = create_splits_scenes()
    official_train, official_val = set(splits["train"]), set(splits["val"])
    mini_scenes = {r["name"] for r in read(RAW / "v1.0-mini/scene.json")}
    quarantined = mini_scenes | {"scene-0001", "scene-0002"}
    # DriveLM799及官方val已有基线评估；仅做暴露登记，不重算其任何指标。
    with Path(
        "/mnt/g/BEV_VLA_Datasets/drivelm/nuscenes_infos_temporal_drivelm799_officialscene.pkl"
    ).open("rb") as stream:
        drive799 = pickle.load(stream)["infos"]
    old_eval_scenes = {
        r.get("official_scene_token", r["scene_token"]) for r in drive799
    }
    calibrations = {r["token"]: r for r in read(META / "calibrated_sensor.json")}
    sensors = {r["token"]: r for r in read(META / "sensor.json")}
    file_sets, sensor_inventory = {}, []
    for channel in (*CAMERAS, "LIDAR_TOP"):
        directory = RAW / "samples" / channel
        file_sets[channel] = {
            item.name
            for item in os.scandir(directory)
            if item.name.endswith((".jpg", ".pcd.bin"))
        }
        sensor_inventory.append(
            dict(
                channel=channel,
                directory=str(directory.resolve()),
                physical_files=len(file_sets[channel]),
            )
        )
    by_sample = {token: {} for token in samples}
    for index, record in enumerate(json_records(META / "sample_data.json")):
        if not record["is_key_frame"] or record["sample_token"] not in by_sample:
            continue
        sensor = sensors[
            calibrations[record["calibrated_sensor_token"]]["sensor_token"]
        ]
        channel = sensor["channel"]
        if channel in file_sets:
            by_sample[record["sample_token"]][channel] = dict(
                filename=record["filename"],
                sample_data_token=record["token"],
                calibrated_sensor_token=record["calibrated_sensor_token"],
                ego_pose_token=record["ego_pose_token"],
            )
        if index and index % 200000 == 0:
            print("INVENTORY_METADATA_RECORDS", index, flush=True)
    # 这里只读annotation的sample token关联存在性，不输出/使用holdout框或类别。
    has_annotations = set()
    for record in json_records(META / "sample_annotation.json"):
        has_annotations.add(record["sample_token"])
    map_root = Path("/mnt/g/BEV_VLA_Datasets/bev_seg_nuscenes/maps/expansion")
    map_files = {p.stem: p for p in map_root.glob("*.json")}
    records = []
    for token, sample in samples.items():
        scene = scenes[sample["scene_token"]]
        scene_name = scene["name"]
        official = (
            "train"
            if scene_name in official_train
            else "val" if scene_name in official_val else "unknown"
        )
        channels = by_sample[token]
        present = {
            channel: channel in channels
            and Path(channels[channel]["filename"]).name in file_sets[channel]
            for channel in (*CAMERAS, "LIDAR_TOP")
        }
        location = logs[scene["log_token"]]["location"]
        six_cameras = all(present[c] for c in CAMERAS)
        annotation = token in has_annotations
        map_present = location in map_files
        records.append(
            dict(
                sample_token=token,
                scene_token=scene["token"],
                scene_name=scene_name,
                official_split=official,
                timestamp=sample["timestamp"],
                location=location,
                previous_sample_token=sample["prev"],
                next_sample_token=sample["next"],
                partition=select_frame_split(scene_name, official, quarantined),
                six_cameras_present=six_cameras,
                lidar_present=present["LIDAR_TOP"],
                annotation_record_present=annotation,
                map_present=map_present,
                joint_ready_presence=six_cameras and annotation and map_present,
                historical_baseline_eval_scene=sample["scene_token"] in old_eval_scenes,
                media={c: channels[c] for c in channels},
            )
        )
    write(
        out / "frame_inventory.json",
        dict(
            frames=records,
            policy="metadata and filename presence only; no holdout GT geometry/predictions/model access",
        ),
    )
    qa_sources, qa_sets = [], {}
    for label, path in (
        (
            "drivelm_train",
            Path("/mnt/g/BEV_VLA_Datasets/drivelm/QA_dataset_nus/v1_1_train_nus.json"),
        ),
        (
            "drivelm_val_q_only",
            Path(
                "/mnt/g/BEV_VLA_Datasets/drivelm/QA_dataset_nus/v1_1_val_nus_q_only.json"
            ),
        ),
    ):
        value, tokens = qa_summary(path, set(samples))
        value["label"] = label
        qa_sources.append(value)
        qa_sets[label] = tokens
    groups = {}
    for key in ("train", "val"):
        rows = [r for r in records if r["official_split"] == key]
        groups[key] = dict(
            metadata_frames=len(rows),
            scenes=len({r["scene_token"] for r in rows}),
            all6_camera_frames=sum(r["six_cameras_present"] for r in rows),
            lidar_frames=sum(r["lidar_present"] for r in rows),
            joint_ready_presence_frames=sum(r["joint_ready_presence"] for r in rows),
            historical_quarantined_frames=sum(
                r["partition"] == "historical_quarantine" for r in rows
            ),
            candidate_new_adaptation_frames=sum(
                r["partition"] == "eligible_official_train"
                and r["joint_ready_presence"]
                for r in rows
            ),
        )
    candidates = [
        r
        for r in records
        if r["partition"] == "eligible_official_train" and r["joint_ready_presence"]
    ]
    eligible_scenes = {r["scene_name"] for r in candidates}
    archive_inventory = []
    for path in sorted(Path("/mnt/g/nus").glob("*")):
        if path.suffix not in (".zip", ".tgz"):
            continue
        item = dict(
            path=str(path), bytes=path.stat().st_size, fully_extracted_not_assumed=True
        )
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
            item["entries"] = len(names)
            item["jpg_entries"] = sum(n.lower().endswith(".jpg") for n in names)
        archive_inventory.append(item)
    free = __import__("shutil").disk_usage("/mnt/g")
    write(
        out / "data_inventory_summary.json",
        dict(
            status="READ_ONLY_INVENTORY_COMPLETE",
            unique_raw_frames=len(samples),
            raw_scenes=len(scenes),
            sensor_inventory=sensor_inventory,
            official_split_counts=groups,
            allowed_new_adaptation=dict(
                unique_frames=len(candidates), scenes=len(eligible_scenes)
            ),
            quarantined_scene_names=sorted(quarantined),
            old_drivelm799=dict(frames=len(drive799), scenes=len(old_eval_scenes)),
            official_val_policy="reserved due historical baseline/validation exposure; not reused for formal development or unsealed in this task",
            strict_author_pretraining_unseen_verified_frames=0,
            counts_by_missing_sensor={
                c: sum(
                    c not in by_sample[t]
                    or Path(by_sample[t][c]["filename"]).name not in file_sets[c]
                    for t in samples
                )
                for c in (*CAMERAS, "LIDAR_TOP")
            },
            drivelm=qa_sources,
            drivelm_train_eligible_joint_frames=len(
                qa_sets["drivelm_train"] & {r["sample_token"] for r in candidates}
            ),
            drivelm_is_nuscenes_subset_not_additive=True,
            aliases=dict(
                bev_seg_samples=str(
                    Path("/mnt/g/BEV_VLA_Datasets/bev_seg_nuscenes/samples").resolve()
                ),
                canonical_samples=str((RAW / "samples").resolve()),
                bev_seg_metadata=str(
                    Path(
                        "/mnt/g/BEV_VLA_Datasets/bev_seg_nuscenes/v1.0-trainval"
                    ).resolve()
                ),
            ),
            map_files=[
                dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
                for p in map_files.values()
            ],
            archives=archive_inventory,
            free_g_bytes=free.free,
            free_workspace_bytes=__import__("shutil").disk_usage(ROOT).free,
            evidence_level="all-frame sensor filename coverage, raw annotation association and map file availability; not full sensor decoding or full instruction-instance supervision readiness",
            metadata_sha256={
                str(META / name): sha(META / name)
                for name in (
                    "scene.json",
                    "sample.json",
                    "sample_data.json",
                    "sample_annotation.json",
                    "calibrated_sensor.json",
                )
            },
            seconds=time.perf_counter() - started,
        ),
    )
    print(
        "FORMAL_INVENTORY_DONE",
        groups,
        "new",
        len(candidates),
        len(eligible_scenes),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    inventory(parser.parse_args().out.resolve())

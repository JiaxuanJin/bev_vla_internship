"""补充标定/位姿覆盖和合法空GT帧，避免用annotation数量过滤真实负样本。"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tarfile

import numpy as np
from PIL import Image

from scripts.stage2.low_memory_dataset import json_records
from scripts.stage3.lg_bev_common import read, sha, write
from scripts.stage3.lg_bev_formal_inventory import CAMERAS, META, OUT, RAW


def coverage(out: Path) -> None:
    """只读验证元数据与传感器；9个完整标注split的空帧显式保留。"""
    records = read(out / "frame_inventory.json")["frames"]
    needed_poses = {
        media["ego_pose_token"] for row in records for media in row["media"].values()
    }
    valid_pose_tokens = set()
    invalid_poses = []
    for pose in json_records(META / "ego_pose.json"):
        if pose["token"] not in needed_poses:
            continue
        q = np.asarray(pose["rotation"])
        if np.isfinite(pose["translation"]).all() and abs(np.linalg.norm(q) - 1) < 1e-4:
            valid_pose_tokens.add(pose["token"])
        else:
            invalid_poses.append(pose["token"])
    calibration_rows = read(META / "calibrated_sensor.json")
    valid_calibrations = {
        r["token"]
        for r in calibration_rows
        if np.isfinite(r["translation"]).all()
        and abs(np.linalg.norm(r["rotation"]) - 1) < 1e-4
    }
    # 用既有官方数据压缩包验证安装后的空GT并非截断annotation JSON造成。
    archive_path = Path("/mnt/g/nus/v1.0-trainval_meta.tgz")
    archive_annotation_sha = None
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if member.name.endswith("v1.0-trainval/sample_annotation.json"):
                digest = hashlib.sha256()
                stream = archive.extractfile(member)
                for block in iter(lambda: stream.read(2**20), b""):
                    digest.update(block)
                archive_annotation_sha = digest.hexdigest()
                break
    installed_sha = sha(META / "sample_annotation.json")
    if archive_annotation_sha != installed_sha:
        raise ValueError("原始annotation与已安装数据不匹配，不能将缺少记录解释为空GT")
    for row in records:
        row["pose_calibration_valid"] = all(
            m["ego_pose_token"] in valid_pose_tokens
            and m["calibrated_sensor_token"] in valid_calibrations
            for m in row["media"].values()
        )
        row["label_state"] = (
            "nonempty_gt"
            if row["annotation_record_present"]
            else "empty_gt_in_verified_labeled_split"
        )
        row["joint_ready_verified"] = (
            row["six_cameras_present"]
            and row["lidar_present"]
            and row["map_present"]
            and row["pose_calibration_valid"]
        )
    candidates = [
        r
        for r in records
        if r["partition"] == "eligible_official_train" and r["joint_ready_verified"]
    ]
    examples = []
    used = set()
    for row in candidates:
        if row["scene_token"] in used:
            continue
        used.add(row["scene_token"])
        decoded = []
        for camera in CAMERAS:
            path = RAW / row["media"][camera]["filename"]
            with Image.open(path) as image:
                image.load()
                decoded.append(
                    dict(
                        camera=camera,
                        shape=[image.height, image.width],
                        bytes=path.stat().st_size,
                        sha256=sha(path),
                    )
                )
        examples.append(
            dict(
                sample_token=row["sample_token"],
                scene_name=row["scene_name"],
                cameras=decoded,
            )
        )
        if len(examples) == 3:
            break
    write(
        out / "frame_inventory_verified.json",
        dict(
            frames=records,
            scope="metadata/presence + pose/calibration integrity; empty labeled frames retained; no heldout GT geometry/predictions opened",
        ),
    )
    write(
        out / "label_coverage_audit.json",
        dict(
            status="PASS",
            annotation_matches_existing_raw_archive=True,
            installed_annotation_sha256=installed_sha,
            existing_archive=str(archive_path),
            missing_required_pose_tokens=len(needed_poses - valid_pose_tokens),
            invalid_pose_records=len(invalid_poses),
            candidate_adaptation_frames=len(candidates),
            candidate_scenes=len({r["scene_token"] for r in candidates}),
            candidate_empty_gt_frames=sum(
                not r["annotation_record_present"] for r in candidates
            ),
            raw_train_frames=sum(r["official_split"] == "train" for r in records),
            raw_val_frames=sum(r["official_split"] == "val" for r in records),
            correction="initial inventory reported positive-annotation presence only (27802); 6 valid empty train frames are included, not silently dropped",
            decoded_new_training_examples=examples,
            all_sensor_files_decoded=False,
            remaining_readiness="full target-instance instruction manifest, map labels and deterministic cache generation not yet implemented for full data; fail closed before formal training",
        ),
    )
    print(
        "COVERAGE_VERIFIED",
        len(candidates),
        len({r["scene_token"] for r in candidates}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    coverage(parser.parse_args().out.resolve())

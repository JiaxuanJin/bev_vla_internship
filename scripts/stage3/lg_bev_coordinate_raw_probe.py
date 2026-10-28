"""实际读取 G 盘六相机、LiDAR、标注和 checkpoint；不把缓存复用称为原始读取。"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import torch

from scripts.stage2.spatial_annotations import AUTHOR, DATA
from scripts.stage3.lg_bev_common import ROOT, read, sha, write
from scripts.stage3.lg_bev_coordinate_core import COORD_OUT, legal_frames


def probe(out: Path) -> None:
    """解析原始传感器内容及权重张量，记录完整文件哈希。"""
    frame = legal_frames()[0]
    legacy = read(ROOT / "outputs/stage2_oom_fix_20261022_step1/protocol/protocol.json")
    metadata_path = Path(legacy["train_ann"]).resolve()
    with metadata_path.open("rb") as stream:
        infos = sorted(pickle.load(stream)["infos"], key=lambda row: row["timestamp"])
    info = infos[frame["index"]]
    assert info["token"] == frame["sample_token"]
    records = []
    for channel, item in info["cams"].items():
        path = Path(item["data_path"])
        path = (path if path.is_absolute() else AUTHOR / path).resolve()
        with Image.open(path) as image:
            image.load()
            pixels = np.asarray(image)
        records.append(
            dict(
                kind="raw_camera",
                channel=channel,
                path=str(path),
                bytes=path.stat().st_size,
                sha256=sha(path),
                shape=list(pixels.shape),
                mean_pixel=float(pixels.mean()),
                decoded_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
            )
        )
    path = Path(info["lidar_path"])
    path = (path if path.is_absolute() else AUTHOR / path).resolve()
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
    records.append(
        dict(
            kind="raw_lidar",
            path=str(path),
            bytes=path.stat().st_size,
            sha256=sha(path),
            points=len(points),
            finite=bool(np.isfinite(points).all()),
            xyz_min=points[:, :3].min(0).tolist(),
            xyz_max=points[:, :3].max(0).tolist(),
        )
    )
    for table in ("calibrated_sensor", "ego_pose", "sample_annotation"):
        path = DATA / "v1.0-mini" / f"{table}.json"
        data = read(path)
        records.append(
            dict(
                kind="raw_metadata",
                table=table,
                path=str(path),
                bytes=path.stat().st_size,
                sha256=sha(path),
                records=len(data),
                selected_sample_records=sum(
                    r.get("sample_token") == frame["sample_token"] for r in data
                ),
            )
        )
    checkpoint_path = Path(legacy["checkpoint"])
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = saved["state_dict"]
    records.append(
        dict(
            kind="raw_checkpoint_deserialized",
            path=str(checkpoint_path),
            bytes=checkpoint_path.stat().st_size,
            sha256=sha(checkpoint_path),
            tensor_keys=len(state),
            floating_parameters_finite=all(
                bool(torch.isfinite(v).all())
                for v in state.values()
                if isinstance(v, torch.Tensor) and v.is_floating_point()
            ),
        )
    )
    write(
        out / "raw_access_probe.json",
        dict(
            status="PASS",
            sample_token=frame["sample_token"],
            scene_name=frame["scene_name"],
            annotation_pickle=dict(path=str(metadata_path), sha256=sha(metadata_path)),
            reads=records,
            scope="raw file decoding only; no neural forward; no BEV cache reads; legal manifest supplies frame identity only",
        ),
    )
    print(
        "RAW_ACCESS_PROBE_PASS",
        frame["scene_name"],
        len(points),
        len(state),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=COORD_OUT)
    probe(parser.parse_args().out.resolve())

"""独立读取原始 map 节点/线和 3D annotation，验证米制轴向与标定。"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import cv2
import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import LineString, box

from scripts.stage2.spatial_annotations import DATA
from scripts.stage3.lg_bev_common import ROOT, read, sha, write
from scripts.stage3.lg_bev_coordinate_core import COORD_OUT, legal_frames


def independent_line_mask(frame: dict, raw_map: dict) -> tuple[np.ndarray, list[dict]]:
    """不调用作者 vector/raster 工具，直接解析全局节点并转换/裁剪/栅格化。"""
    nodes = {r["token"]: r for r in raw_map["node"]}
    lines = {r["token"]: r for r in raw_map["line"]}
    rotation = Quaternion(frame["ego2global_rotation"]).rotation_matrix
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    yaw_rotation = np.array([[c, -s], [s, c]])
    shift = np.asarray(frame["ego2global_translation"][:2])
    records = {
        layer: {r["token"]: r for r in raw_map[layer]}
        for layer in ("lane_divider", "road_divider")
    }
    result = np.zeros((200, 400), dtype=np.uint8)
    audits = []
    for identity in frame["instance_ids"]:
        _, layer, token = identity.split("/")
        if layer not in records:
            continue
        record = records[layer][token]
        vertices = np.array(
            [
                [nodes[k]["x"], nodes[k]["y"]]
                for k in lines[record["line_token"]]["node_tokens"]
            ]
        )
        ego = (vertices - shift) @ yaw_rotation
        geometry = LineString(ego).intersection(box(-30, -15, 30, 15))
        pieces = (
            list(geometry.geoms)
            if geometry.geom_type == "MultiLineString"
            else [geometry]
        )
        count, max_quantization = 0, 0.0
        for line in pieces:
            if line.is_empty or line.length == 0:
                continue
            # 固定 0.05m 致密采样，仅忠实描画原始线，不由 GT 调整。
            distances = np.linspace(
                0, line.length, max(2, int(np.ceil(line.length / 0.05)) + 1)
            )
            points = np.array(
                [[line.interpolate(d).x, line.interpolate(d).y] for d in distances]
            )
            pixels = np.floor((points + np.array([30, 15])) / 0.15).astype(np.int32)
            cv2.polylines(result, [pixels], False, 1, thickness=5)
            valid = ((pixels >= 0) & (pixels < np.array([400, 200]))).all(1)
            recovered = (pixels[valid] + 0.5) * 0.15 - np.array([30, 15])
            quantization = np.linalg.norm(recovered - points[valid], axis=1)
            max_quantization = max(max_quantization, float(quantization.max(initial=0)))
            if max_quantization > np.sqrt(2) * 0.075 + 1e-8:
                raise ValueError("独立 GT 像素中心协议不一致")
            count += len(points)
        audits.append(
            dict(
                raw_record_id=identity,
                sampled_points=count,
                max_pixel_center_quantization_m=max_quantization,
            )
        )
    return result.astype(bool), audits


def run(out: Path) -> None:
    """固定四个允许训练场景首帧；原始节点和 annotation 均从 G 盘读取。"""
    legacy = read(ROOT / "outputs/stage2_oom_fix_20261022_step1/protocol/protocol.json")
    with Path(legacy["train_ann"]).open("rb") as stream:
        infos = sorted(pickle.load(stream)["infos"], key=lambda r: r["timestamp"])
    annotations = read(DATA / "v1.0-mini/sample_annotation.json")
    selected = [
        next(f for f in legal_frames() if f["scene_name"] == scene)
        for scene in sorted({f["scene_name"] for f in legal_frames()})
    ]
    records = []
    for frame in selected:
        info = infos[frame["index"]]
        assert info["token"] == frame["sample_token"]
        raw_map_path = DATA / "maps/expansion" / f"{frame['location']}.json"
        manual_mask, map_records = independent_line_mask(frame, read(raw_map_path))
        # 原缓存仅用于相同 GT 的独立对照，不参与原始节点栅格构造。
        with np.load(frame["source_file"]) as data:
            gt = data["gt"]
        raw = [r for r in annotations if r["sample_token"] == info["token"]]
        ego_rotation = Quaternion(frame["ego2global_rotation"]).rotation_matrix
        lidar_rotation = Quaternion(frame["lidar2ego_rotation"]).rotation_matrix
        global_centres = np.asarray([r["translation"] for r in raw])
        ego = (
            global_centres - np.asarray(frame["ego2global_translation"])
        ) @ ego_rotation
        lidar = (ego - np.asarray(frame["lidar2ego_translation"])) @ lidar_rotation
        source_centres = info["gt_boxes"][info["valid_flag"], :3]
        distances = np.linalg.norm(source_centres[:, None] - lidar[None], axis=-1)
        nearest = distances.argmin(1)
        error = float(distances.min(1).max())
        assert len(set(nearest)) == len(nearest) and error < 0.001
        global_reconstructed = (
            lidar @ lidar_rotation.T + np.asarray(frame["lidar2ego_translation"])
        ) @ ego_rotation.T + np.asarray(frame["ego2global_translation"])
        roundtrip = float(
            np.linalg.norm(global_reconstructed - global_centres, axis=1).max()
        )
        gt_line = gt == 1
        other_priority = gt > 1
        dilated = cv2.dilate(
            gt_line.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
        count = int(manual_mask.sum())
        records.append(
            dict(
                scene_name=frame["scene_name"],
                sample_token=frame["sample_token"],
                raw_map_path=str(raw_map_path),
                raw_map_sha256=sha(raw_map_path),
                independent_map_records=map_records,
                manual_line_pixels=count,
                direct_same_class_coverage=float(
                    (manual_mask & gt_line).sum() / max(count, 1)
                ),
                coverage_with_1pixel_quantization=float(
                    (manual_mask & dilated).sum() / max(count, 1)
                ),
                higher_priority_overlap_pixels=int(
                    (manual_mask & other_priority).sum()
                ),
                note="manual raw-polyline dense raster vs author's sampled-vector/priority raster; report differences without fitting either",
                native_gt_objects=len(source_centres),
                matched_annotation_tokens=[raw[k]["token"] for k in nearest],
                max_raw_annotation_to_pickle_error_m=error,
                raw_global_lidar_roundtrip_m=roundtrip,
            )
        )
    write(
        out / "independent_gt.json",
        dict(
            status="PASS_COORDINATE_AND_ANNOTATION_CHAIN",
            records=records,
            independent_of_author_vector_raster=True,
            optimizer_updates=0,
            calibration_chain="raw world annotation -> inverse full Ego pose -> inverse LiDAR calibration -> metadata GT; exact inverse roundtrip",
            map_chain="raw global node/line table -> analytic inverse yaw -> metric clipping -> independent OpenCV raster -> GT pixel centre",
            evaluation_split="train only; no internal_dev/sealed GT",
            source_sha256=sha(__file__),
        ),
    )
    print("INDEPENDENT_GT_PASS", len(records), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=COORD_OUT)
    run(parser.parse_args().out.resolve())

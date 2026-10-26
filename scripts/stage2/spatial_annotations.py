"""从 nuScenes 地图记录生成可追溯的空间标签及米制 BEV 采样网格。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.spatial_supervision import SpatialTarget, validate_spatial_target

AUTHOR = Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg"
DATA = Path("/mnt/g/BEV_VLA_Datasets/bev_seg_nuscenes")


def load_module(name: str, path: Path):
    """只读导入作者几何工具，避免构建整套模型。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest_array(array: np.ndarray) -> str:
    """数组内容摘要。"""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def metric_grid(info: dict) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """GT 米制像素中心经 Ego→LiDAR 标定采样原生 BEV，绝不 resize GT。"""
    from pyquaternion import Quaternion

    y, x = np.meshgrid(-15 + (np.arange(200) + .5) * .15,
                       -30 + (np.arange(400) + .5) * .15, indexing="ij")
    ego = np.stack((x, y, np.zeros_like(x)), axis=-1)
    rotation = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
    shift = np.asarray(info["lidar2ego_translation"])
    lidar = (ego - shift) @ rotation
    # BEVFormer 的 reference_points 为 (col+.5)/W, (row+.5)/H。
    # align_corners=False 因而以物理范围 102.4 m 归一化，无索引转置或翻转。
    grid = torch.tensor(lidar[..., :2] / 51.2, dtype=torch.float64)[None]
    return grid, ego, lidar


def project_attention(attention: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """把原生 BEV 概率或 logits 按标定投影到原始 GT 网格，保留梯度。"""
    if attention.shape[-2:] != (150, 150) or grid.shape != (1, 200, 400, 2):
        raise ValueError("需要原生 150×150 BEV 和已核验 200×400 米制采样网格")
    return F.grid_sample(attention, grid.to(attention), mode="bilinear",
                         padding_mode="zeros", align_corners=False)


def load_target(row: dict, directory: Path, device="cpu"):
    """按 manifest 加载实例标签；返回独立 GT 和物理采样网格。"""
    with np.load(directory / row["mask_file"]) as data:
        gt = torch.tensor(data["gt"], device=device, dtype=torch.long)
        if digest_array(data["gt"]) != row["gt_sha256"]:
            raise ValueError("GT 摘要不匹配")
        target = SpatialTarget(
            row["instruction_id"], row["target_class"], row["instance_id"],
            torch.tensor(data["target"], device=device, dtype=torch.bool),
            torch.tensor(data["region"], device=device, dtype=torch.bool),
            row["annotation_source"],
        )
        grid = torch.tensor(data["sampling_grid"], device=device)
    validate_spatial_target(target, gt)
    return gt, target, grid


def tagged_vectors(mapper, info: dict, location: str):
    """复用原作者取线/采样算法，同时保留 map record token。

    Crossing 的两个长边属于同一真实 polygon record，不能被当成两个实例。
    Boundary 是 road/lane 的 union 轮廓，本协议不伪造它的唯一 record ID。
    """
    from nuscenes.map_expansion.map_api import NuScenesMapExplorer
    from nuscenes.eval.common.utils import quaternion_yaw
    from pyquaternion import Quaternion
    from shapely import affinity

    x, y = info["ego2global_translation"][:2]
    angle = quaternion_yaw(Quaternion(info["ego2global_rotation"])) / np.pi * 180
    patch_box = (x, y, 30., 60.)
    patch = NuScenesMapExplorer.get_patch_coord(patch_box, angle)
    api = mapper.nusc_maps[location]
    tagged = []
    for layer in ("road_divider", "lane_divider", "ped_crossing"):
        records = getattr(api, layer)
        for record in records:
            if layer == "ped_crossing":
                # 修改的只是当前进程内的遍历视图，原始 map JSON 不写入。
                api.ped_crossing = [record]
                try:
                    geometries = mapper.get_ped_crossing_line(patch_box, angle, location)
                finally:
                    api.ped_crossing = records
            else:
                line = api.extract_line(record["line_token"])
                clipped = line.intersection(patch)
                if clipped.is_empty:
                    continue
                clipped = affinity.rotate(clipped, -angle, origin=(x, y))
                geometries = [affinity.affine_transform(
                    clipped, [1., 0., 0., 1., -x, -y])]
            for points, count in mapper._one_type_line_geom_to_vectors(geometries):
                tagged.append((dict(pts=points.astype(float), pts_num=count,
                                    type=1 if layer == "ped_crossing" else 0),
                               f"{location}/{layer}/{record['token']}"))
    baseline = mapper.gen_vectorized_samples(
        location, info["ego2global_translation"], info["ego2global_rotation"])
    prefix = [v for v in baseline if v["type"] in (0, 1)]
    if len(prefix) != len(tagged) or any(
        v["type"] != t["type"] or not np.array_equal(v["pts"], t["pts"])
        for v, (t, _) in zip(prefix, tagged)
    ):
        raise ValueError("保留 ID 后的矢量与作者原始顺序/坐标不一致")
    return baseline, tagged


def main() -> None:
    """先冻结固定样本的几何标签，再允许任何新模型验证。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tools = AUTHOR / "projects/mmdet3d_plugin/datasets/utils"
    vector = load_module("spatial_author_vector", tools / "vector_map.py")
    raster = load_module("spatial_author_raster", tools / "rasterize.py")
    mapper = vector.VectorizedLocalMap(str(DATA), (30., 60.), (200, 400))
    legacy = json.loads((ROOT / "outputs/stage2_oom_fix_20261022_step1/protocol/protocol.json").read_text())
    scenes = {r["token"]: r for r in json.loads((DATA / "v1.0-mini/scene.json").read_text())}
    logs = {r["token"]: r for r in json.loads((DATA / "v1.0-mini/log.json").read_text())}
    sample_data = [r for r in json.loads((DATA / "v1.0-mini/sample_data.json").read_text())
                   if r["is_key_frame"]]
    lidar_sensor_tokens = {
        r["token"] for r in json.loads((DATA / "v1.0-mini/calibrated_sensor.json").read_text())
        if r["translation"][2] > 1.8 and len(r["camera_intrinsic"]) == 0}
    lidar_records = {r["sample_token"]: r for r in sample_data
                     if r["calibrated_sensor_token"] in lidar_sensor_tokens}
    rows, rejected, frames = [], [], []
    ramp_errors, legacy_errors = [], []
    map_names = {1: "lane markings", 2: "crosswalk"}
    for split, pickle_split in (("train", "train"), ("validation", "val")):
        with (DATA / f"nuscenes_infos_temporal_{pickle_split}.pkl").open("rb") as stream:
            infos = sorted(pickle.load(stream)["infos"], key=lambda r: r["timestamp"])
        for fixed in legacy[split]:
            info = infos[fixed["index"]]
            assert info["token"] == fixed["token"]
            location = logs[scenes[info["scene_token"]]["log_token"]]["location"]
            vectors, tagged = tagged_vectors(mapper, info, location)
            semantic, *_ = raster.preprocess_map(vectors, (30., 60.), (200, 400), 3, 5, 36)
            gt = (semantic.numpy() * np.arange(1, 4)[:, None, None]).sum(0).astype(np.uint8)
            exact = None
            if split == "validation":
                old = ROOT / f"outputs/stage2_oom_fix_20261022_validation/validate/val_{fixed['index']:03d}.npz"
                with np.load(old) as cached:
                    exact = bool(np.array_equal(gt, cached["gt"]))
                if not exact:
                    raise ValueError(f"重建原生 GT 不等于历史 GT: {info['token']}")
            instance_vectors = {}
            for v, identity in tagged:
                instance_vectors.setdefault(identity, []).append(v)
            masks = {}
            for identity, vv in instance_vectors.items():
                single, *_ = raster.preprocess_map(vv, (30., 60.), (200, 400), 3, 5, 36)
                cls = vv[0]["type"] + 1
                masks[identity] = (cls, single[cls - 1].numpy() & (gt == cls))
            for cls in (1, 2):
                union = np.zeros_like(gt, dtype=bool)
                for c, mask in masks.values():
                    if c == cls:
                        union |= mask
                assert np.array_equal(union, gt == cls)
            sampling, ego, lidar = metric_grid(info)
            native = BEVGrid().lidar_centers().numpy()
            ramp = torch.tensor(native[..., :2].transpose(2, 0, 1))[None]
            projected = project_attention(ramp, sampling)[0].numpy().transpose(1, 2, 0)
            error = float(np.max(np.abs(projected - lidar[..., :2])))
            ramp_errors.append(error)
            assert error < 1e-8
            # 数值重现旧 cropper 的坐标变换，统计其距真实 Ego 网格的误差。
            old_grid = torch.tensor(ego[..., :2] / 50.86, dtype=torch.float64)[None]
            old_lidar = F.grid_sample(torch.flip(ramp.transpose(2, 3), [2]),
                                     old_grid, align_corners=True)[0].numpy().transpose(1, 2, 0)
            legacy_error = float(np.linalg.norm(old_lidar - lidar[..., :2], axis=-1).mean())
            legacy_errors.append(legacy_error)
            frames.append({**fixed, "split": split, "location": location,
                           "gt_sha256": digest_array(gt), "legacy_gt_exact": exact,
                           "physical_ramp_max_error_m": error,
                           "legacy_projection_mean_error_m": legacy_error})
            for cls in (1, 2):
                for fb, lr in (("front", "left"), ("front", "right"),
                               ("rear", "left"), ("rear", "right")):
                    region = ((ego[..., 0] > 0) if fb == "front" else (ego[..., 0] < 0))
                    region &= ((ego[..., 1] > 0) if lr == "left" else (ego[..., 1] < 0))
                    candidates = [key for key, (c, mask) in masks.items()
                                  if c == cls and (mask & region).any()]
                    key = f"{info['token']}_c{cls}_{fb}_{lr}"
                    base = dict(instruction_id=key, sample_token=info["token"],
                                scene_token=info["scene_token"], index=fixed["index"],
                                split=split, target_class=cls,
                                text=f"Find {map_names[cls]} on the {fb} {lr}",
                                candidate_instance_ids=candidates)
                    if len(candidates) != 1:
                        rejected.append({**base, "reason": "AMBIGUOUS" if candidates else "TARGET_ABSENT"})
                        continue
                    identity = candidates[0]
                    target_mask = masks[identity][1] & region
                    if target_mask.sum() < 8:
                        rejected.append({**base, "reason": "TOO_FEW_TARGET_PIXELS"})
                        continue
                    filename = key + ".npz"
                    other_instances = np.zeros_like(target_mask)
                    for other_id, (other_cls, other_mask) in masks.items():
                        if other_cls == cls and other_id != identity:
                            other_instances |= other_mask
                    other_instances &= ~masks[identity][1]
                    np.savez_compressed(
                        output / filename, gt=gt, target=target_mask,
                        selected_full_instance=masks[identity][1],
                        other_instances=other_instances,
                        region=region, sampling_grid=sampling.numpy())
                    sd = lidar_records.get(info["token"], {})
                    row = {**base, "instance_id": identity, "mask_file": filename,
                           "annotation_source": "nuscenes_map_record_exact_author_raster",
                           "coordinate_frame": "segmentation_gt_pixel_grid",
                           "gt_sha256": digest_array(gt),
                           "target_sha256": digest_array(target_mask),
                           "roi_definition": "Ego metric quadrant; x front, y left; fixed before predictions",
                           "target_pixels": int(target_mask.sum()),
                           "distractor_pixels": int(((gt == cls) & ~target_mask).sum()),
                           "other_instance_pixels": int(other_instances.sum()),
                           "ego_pose_token": sd.get("ego_pose_token"),
                           "calibrated_sensor_token": sd.get("calibrated_sensor_token"),
                           "lidar2ego_rotation": info["lidar2ego_rotation"],
                           "lidar2ego_translation": info["lidar2ego_translation"]}
                    load_target(row, output)
                    rows.append(row)
            print(split, fixed["index"], "annotated", flush=True)
    assert not ({r["scene_token"] for r in rows if r["split"] == "train"} &
                {r["scene_token"] for r in rows if r["split"] == "validation"})
    manifest = {"seed": 20261026, "selection": "original fixed 20 train / 6 val frames; all 2 classes x 4 quadrants; reject absent/ambiguous; no prediction selection",
                "annotations": rows, "rejected": rejected, "frames": frames,
                "counts": dict(Counter(r["split"] for r in rows)),
                "rejection_counts": dict(Counter(r["reason"] for r in rejected)),
                "alignment": {"native_bev": [150, 150], "gt": [200, 400],
                    "gt_pixel_resolution_m": .15, "gt_xbound": [-30, 30],
                    "gt_ybound": [-15, 15], "grid_sample_align_corners": False,
                    "metric_ramp_max_error_m": max(ramp_errors),
                    "legacy_projection_mean_error_m": float(np.mean(legacy_errors)),
                    "legacy_projection_max_frame_mean_error_m": max(legacy_errors),
                    "validation_gt_exact_frames": sum(f["legacy_gt_exact"] is True for f in frames),
                    "boundary_status": "NOT_ANNOTATED_UNION_HAS_NO_UNIQUE_RECORD_ID"}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("counts", "rejection_counts", "alignment")}, indent=2))


if __name__ == "__main__":
    main()

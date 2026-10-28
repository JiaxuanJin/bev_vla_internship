"""真实地图记录与车辆/行人 annotation 的实例监督，GT 不进入 forward。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage3.lg_bev_common import (
    DEFAULT_OUT,
    FORBIDDEN,
    ROOT,
    SEED,
    assert_splits,
    read,
    sha,
    write,
)

V4 = ROOT / "outputs/stage3_conditional_mask_v4_20261026"
DATA = Path("/mnt/g/BEV_VLA_Datasets/bev_seg_nuscenes")


def geometry(row: dict, *, device: str = "cpu") -> Tensor:
    """仅用 LiDAR→Ego 标定和 Ego z=0 平面生成五通道位置。"""
    from pyquaternion import Quaternion

    xyz = BEVGrid().lidar_centers(dtype=torch.float32, device=device)
    rotation = xyz.new_tensor(Quaternion(row["lidar2ego_rotation"]).rotation_matrix)
    shift = xyz.new_tensor(row["lidar2ego_translation"])
    xyz[..., 2] = (
        -shift[2] - xyz[..., 0] * rotation[2, 0] - xyz[..., 1] * rotation[2, 1]
    ) / rotation[2, 2]
    ego = xyz @ rotation.T + shift
    x, y = ego[..., 0] / 30, ego[..., 1] / 15
    return torch.stack((x, y, x * y, (x * x + y * y).sqrt(), torch.ones_like(x)))[None]


def box_masks(boxes: np.ndarray) -> tuple[np.ndarray, int]:
    """LiDAR 朝向盒 footprint；亚像素实例保留包含中心的栅格。

    Args:
        boxes: 原生 LiDARInstance3DBoxes，中心 z 为底面，yaw 绕 LiDAR z。

    Returns:
        每实例 150×150 掩码以及亚像素中心回填次数。
    """
    xy = BEVGrid().lidar_centers().numpy()[..., :2]
    masks, fallback = [], 0
    for box in boxes:
        delta = xy - box[:2]
        c, s = np.cos(box[6]), np.sin(box[6])
        # 本地 MMDetection3D 0.17.1 的 corners 使用顺时针 yaw；不能套用新版约定。
        u = delta[..., 0] * c - delta[..., 1] * s
        v = delta[..., 0] * s + delta[..., 1] * c
        mask = (np.abs(u) <= box[3] / 2) & (np.abs(v) <= box[4] / 2)
        if not mask.any() and (np.abs(box[:2]) < 51.2).all():
            col, row = np.floor((box[:2] + 51.2) / (102.4 / 150)).astype(int)
            mask[row, col] = True
            fallback += 1
        masks.append(mask)
    return np.asarray(masks, dtype=bool).reshape(-1, 150, 150), fallback


def exclusive_masks(
    masks: np.ndarray, selected: list[int], distractors: list[int]
) -> tuple[np.ndarray, np.ndarray, int]:
    """目标像素优先；记录真实 footprint 重叠，随后排除负样本冲突。"""
    shape = masks.shape[1:]
    target = masks[selected].any(0) if selected else np.zeros(shape, dtype=bool)
    other = masks[distractors].any(0) if distractors else np.zeros(shape, dtype=bool)
    conflict = int((target & other).sum())
    return target, other & ~target, conflict


def load_supervision(row: dict, out: Path, *, training: bool) -> dict[str, Tensor]:
    """训练接口拒绝内部开发 GT；评估 GT 由独立计分阶段读取。"""
    if training and row["split"] != "train":
        raise ValueError("训练禁止读取 internal_dev GT")
    path = out / row["label_file"]
    if sha(path) != row["label_sha256"]:
        raise ValueError("监督文件摘要不匹配")
    with np.load(path) as data:
        return {k: torch.from_numpy(data[k].copy()) for k in data.files}


def prepare(out: Path) -> None:
    """固定场景划分后，重核原始 annotation/标定并构造独立监督清单。"""
    from pyquaternion import Quaternion

    from scripts.stage2.spatial_annotations import metric_grid, project_attention
    from scripts.stage2.spatial_detection import ego_centers, in_region
    from scripts.stage3.lg_bev_native import load_native

    old = read(V4 / "labels/manifest.json")
    frames = [dict(f) for f in old["frames"] if f["split"] == "train"]
    scenes = sorted({f["scene_name"] for f in frames})
    dev = str(np.random.RandomState(SEED).permutation(scenes)[0])
    for frame in frames:
        frame["prior_split"] = "V4/V5_train"
        frame["split"] = "internal_dev" if frame["scene_name"] == dev else "train"
    assert_splits(frames)
    # 只构建原生模型和训练元数据；此阶段不执行任何场景的模型前向。
    model, _, encoder, dataset = load_native()
    raw_path = DATA / "v1.0-mini/sample_annotation.json"
    raw = read(raw_path)
    raw_by_sample = {f["sample_token"]: [] for f in frames}
    for record in raw:
        if record["sample_token"] in raw_by_sample:
            raw_by_sample[record["sample_token"]].append(record)
    sensors = {r["token"]: r for r in read(DATA / "v1.0-mini/sensor.json")}
    calibrations = {
        r["token"]: r for r in read(DATA / "v1.0-mini/calibrated_sensor.json")
    }
    lidar_data = {
        r["sample_token"]: r
        for r in read(DATA / "v1.0-mini/sample_data.json")
        if r["is_key_frame"]
        and sensors[calibrations[r["calibrated_sensor_token"]]["sensor_token"]][
            "channel"
        ]
        == "LIDAR_TOP"
    }
    lineage = {str(V4 / "labels/manifest.json"): sha(V4 / "labels/manifest.json")}
    for path in (raw_path, DATA / "v1.0-mini/calibrated_sensor.json"):
        lineage[str(path)] = sha(path)
    rows, audits = [], []
    by_token = {f["sample_token"]: f for f in frames}
    for frame in frames:
        token = frame["sample_token"]
        for key in ("source", "cache"):
            path = frame[f"{key}_file"]
            if sha(path) != frame[f"{key}_sha256"]:
                raise ValueError("旧缓存摘要不匹配")
            lineage[path] = sha(path)
        calibration = calibrations[lidar_data[token]["calibrated_sensor_token"]]
        for key in ("rotation", "translation"):
            np.testing.assert_allclose(calibration[key], frame[f"lidar2ego_{key}"])
        grid, _, lidar = metric_grid(frame)
        ramp = BEVGrid().lidar_centers()[..., :2].permute(2, 0, 1)[None]
        error = float(
            (
                project_attention(ramp, grid)[0].permute(1, 2, 0)
                - torch.from_numpy(lidar[..., :2])
            )
            .abs()
            .max()
        )
        if error > 1e-8:
            raise ValueError("标定 ramp 失败，禁止训练")
        ann = dataset.get_ann_info(frame["index"])
        boxes, labels = ann["gt_bboxes_3d"].tensor.numpy(), ann["gt_labels_3d"]
        centers = ego_centers(boxes, frame)
        records = raw_by_sample[token]
        world = centers @ Quaternion(frame["ego2global_rotation"]).rotation_matrix.T
        world += np.asarray(frame["ego2global_translation"])
        distances = np.linalg.norm(
            world[:, None] - np.asarray([a["translation"] for a in records])[None],
            axis=-1,
        )
        nearest = distances.argmin(1)
        match_error = float(distances.min(1).max()) if len(boxes) else 0.0
        if match_error > 0.001 or len(set(nearest)) != len(nearest):
            raise ValueError("真实 annotation 无法唯一匹配，禁止训练")
        ids = [records[i]["token"] for i in nearest]
        instances = [records[i]["instance_token"] for i in nearest]
        masks, fallback = box_masks(boxes)
        corner_checks = verify_box_corners(ann["gt_bboxes_3d"], masks)
        inside = (np.abs(boxes[:, :2]) < 51.2).all(1)
        for name, cls in (("cars", 0), ("pedestrians", 8)):
            for fb in ("front", "rear"):
                for lr in ("left", "right"):
                    prompt = dict(class_id=cls, front_back=fb, left_right=lr)
                    candidates = np.flatnonzero(
                        (labels == cls) & inside & in_region(centers, prompt)
                    ).tolist()
                    # 集合指令支持空目标；限定词按原始几何中心距离确定实例。
                    variants = [("all", candidates)]
                    if candidates:
                        order = sorted(
                            candidates, key=lambda k: np.linalg.norm(centers[k, :2])
                        )
                        gap = (
                            (
                                np.linalg.norm(centers[order[1], :2])
                                - np.linalg.norm(centers[order[0], :2])
                            )
                            if len(order) > 1
                            else 1e9
                        )
                        if gap >= 1.0:
                            variants.append(("nearest", [order[0]]))
                        far_gap = (
                            (
                                np.linalg.norm(centers[order[-1], :2])
                                - np.linalg.norm(centers[order[-2], :2])
                            )
                            if len(order) > 1
                            else 1e9
                        )
                        if len(order) > 1 and far_gap >= 1.0:
                            variants.append(("farthest", [order[-1]]))
                    for qualifier, selected in variants:
                        distractors = np.flatnonzero((labels == cls) & inside).tolist()
                        distractors = [i for i in distractors if i not in selected]
                        target, other, conflicts = exclusive_masks(
                            masks, selected, distractors
                        )
                        text = (
                            f"Find {name} on the {fb} {lr}"
                            if qualifier == "all"
                            else f"Find the {qualifier} {name} on the {fb} {lr}"
                        )
                        wrong = (
                            text.replace("left", "RIGHT")
                            .replace("right", "left")
                            .replace("RIGHT", "right")
                        )
                        same_direction = [i for i in distractors if i in candidates]
                        row = dict(
                            **frame,
                            task="object",
                            text=text,
                            wrong_text=wrong,
                            qualifier=qualifier,
                            prompt=prompt,
                            instruction_id=f"{token}_{name}_{fb}_{lr}_{qualifier}",
                            selected_gt_indices=selected,
                            target_annotation_tokens=[ids[i] for i in selected],
                            distractor_annotation_tokens=[ids[i] for i in distractors],
                            gt_annotation_tokens=ids,
                            gt_instance_tokens=instances,
                            same_direction_distractors=len(same_direction),
                            overlap_removed_pixels=conflicts,
                        )
                        save_label(
                            out,
                            row,
                            target,
                            other,
                            np.ones_like(target),
                            boxes=boxes,
                            labels=labels,
                        )
                        rows.append(row)
        audits.append(
            dict(
                sample_token=token,
                split=frame["split"],
                ramp_max_error_m=error,
                annotation_max_error_m=match_error,
                subpixel_fallback=fallback,
                raw_objects=len(boxes),
                native_corner_checks=corner_checks,
                outside_bev=int((~inside).sum()),
            )
        )
    # 地图监督保留真实 map record ID；只复用审计通过的原训练 GT。
    for old_row in old["annotations"]:
        token = old_row["sample_token"]
        if token not in by_token:
            continue
        row = {**old_row, **by_token[token], "task": "map"}
        row["wrong_text"] = row.get("counterfactual_text") or row["text"].replace(
            "left", "RIGHT"
        ).replace("right", "left").replace("RIGHT", "right")
        path = V4 / "labels" / old_row["label_file"]
        if sha(path) != old_row["label_sha256"]:
            raise ValueError("原始地图实例标签摘要错误")
        lineage[str(path)] = sha(path)
        with np.load(path) as data:
            target, other = data["target"], data["same_class_other"]
            save_label(out, row, target, other, np.ones_like(target))
        rows.append(row)
    raw_map_checks = rebuild_map_checks(frames, lineage)
    make_figures(out, rows, dataset)
    manifest = dict(frames=frames, annotations=rows)
    write(out / "manifest.json", manifest)
    legacy = read(ROOT / "outputs/stage2_oom_fix_20261022_step1/protocol/protocol.json")
    for path in (
        legacy["checkpoint"],
        ROOT / "models/distilbert-base-uncased/model.safetensors",
        legacy["train_ann"],
        ROOT / "models/distilbert-base-uncased/config.json",
        ROOT / "models/distilbert-base-uncased/tokenizer.json",
        ROOT / "models/distilbert-base-uncased/vocab.txt",
        Path.home()
        / "projects/bevformer_author_clean_5080/BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg_det.py",
    ):
        lineage[str(path)] = sha(path)
    counts = {}
    for split in ("train", "internal_dev"):
        rr = [r for r in rows if r["split"] == split]
        ff = [f for f in frames if f["split"] == split]
        counts[split] = dict(
            instructions=len(rr),
            frames=len(ff),
            scenes=len({f["scene_token"] for f in ff}),
            task_counts=dict(Counter(r["task"] for r in rr)),
            empty_targets=sum(not r["target_present"] for r in rr),
        )
    sources = sorted((ROOT / "scripts/stage3").glob("lg_bev*.py"))
    protocol = dict(
        split_id="lg_bev_20261027_scene_internal_v1",
        seed=SEED,
        hardware=dict(
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()),
            python=__import__("sys").version,
        ),
        frames=frames,
        counts=counts,
        internal_dev_scene=dev,
        forbidden_scenes=sorted(FORBIDDEN),
        manifest_sha256=sha(out / "manifest.json"),
        lineage_sha256=lineage,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
        exposure="全部为历史 V1–V5 已使用训练分布。新 LG-BEV 随机初始化；内部开发场景不参与本次更新。作者预训练暴露未验证；不声称未见泛化。",
        independent_new_development="NOT_AVAILABLE: use allowed historical train scenes",
        final_holdout="BLOCKED: no verified unexposed final data; old scenes sealed",
        checkpoint=dict(path=legacy["checkpoint"], sha256=sha(legacy["checkpoint"])),
        model=dict(
            hidden_channels=24, relative_bound=0.05, initialization="prototype zero up"
        ),
        model_config=dict(hidden_channels=24, relative_bound=0.05),
        budget=dict(
            pilot_updates=16,
            batch_size=1,
            lr=0.0003,
            weight_decay=0.01,
            clip=5.0,
            smoke_updates=2,
            no_expansion=True,
        ),
        loss=dict(
            detection=1.0,
            segmentation=1.0,
            mask=0.2,
            distractor=0.5,
            irrelevant_protect=0.1,
            hypothesis="固定先验权重，未由旧开发表现反推；seg 使用作者 loss_seg 含权重3和类别权重",
        ),
        schedule="seeded deterministic map/object alternation; object positive/empty alternating when available; terminal only; smoke adapter discarded before pilot",
        metrics=dict(
            threshold=0.5,
            detection="noninterpolated AP@2m_xy; full top300, class/quadrant unique greedy match; plural prompts only; NOT official nuScenes AP",
            segmentation="map target-class IoU in fixed2m buffer; irrelevant foreground mIoU outside quadrant; no object-to-map mapping",
            attention="pooled target IoU, sameclass distractor FP rate, background FP rate; empty prompts retained",
        ),
        gate="P4 only after split, raw GT lineage, geometry ramp, finite dual-head backward, freeze invariants and correct/wrong attention delta >1e-8 PASS; any unresolved spatial conflict BLOCKED; no retries/threshold/architecture selection",
        decision="one fixed16-update pilot only; useful iff correct exceeds original, zero and wrong on relevant AP and SegIoU, irrelevant drop<=0.1pp; no expansion even if passes",
    )
    write(out / "protocol.json", protocol)
    write(
        out / "supervision_audit.json",
        dict(
            status="PASS",
            counts=counts,
            frames=audits,
            raw_map_reconstruction=raw_map_checks,
            target_pixels=sum(r["target_pixels"] for r in rows),
            distractor_pixels=sum(r["distractor_pixels"] for r in rows),
            conflicts_after_exclusion=0,
            overlap_removed_pixels=sum(
                r.get("overlap_removed_pixels", 0) for r in rows
            ),
            same_direction_object_prompts=sum(
                r.get("same_direction_distractors", 0) > 0 for r in rows
            ),
            policy="map=audited original raw-map-derived training labels; object=raw mini annotations matched<1mm; GT only loss/scoring",
        ),
    )
    print("LG_PROTOCOL", counts, "internal_dev", dev, flush=True)
    del model, encoder


def verify_box_corners(boxes: object, masks: np.ndarray) -> int:
    """独立闭多边形逐像素核验，并检查旧版原生 corners 的浮点误差界。

    GT 保留 box_masks 的原计算：FP32 sin/cos 系数、FP64 像素中心和 <=。
    原生 corners 在 FP32 中旋转后再平移，近边缘的点可能跨过其舍入边。
    因此先核验原生 corners 的顺时针几何，再用 GT 仿射的逆构造 FP64
    闭多边形；不能将有舍入误差的 FP32 corners 当作精确的离散标签。
    """
    from shapely.geometry import Polygon
    from shapely.vectorized import contains, touches

    xy = BEVGrid().lidar_centers().numpy()[..., :2]
    values = boxes.tensor.numpy()
    if masks.shape != (len(boxes), 150, 150) or masks.dtype != np.bool_:
        raise ValueError("对象掩码必须为每框一个150×150布尔栅格")
    corners = boxes.corners.numpy() if len(boxes) else np.empty((0, 8, 3))
    native_c = boxes.tensor[:, 6].cos().numpy()
    native_s = boxes.tensor[:, 6].sin().numpy()
    # 原生旋转每坐标至多三个乘积、两次求和及一次平移；局部半尺寸乘2^-1精确。
    # gamma_6 是这些浮点运算的前向误差界，不是像素容差。
    unit = np.finfo(values.dtype).eps / 2
    gamma = 6 * unit / (1 - 6 * unit)
    signs = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]], dtype=np.float64)
    for k, (box, corner) in enumerate(zip(values, corners)):
        if not np.isfinite(box[:7]).all() or (box[3:6] <= 0).any():
            raise ValueError(f"非法原生3D框: {k}")
        local = signs * box[3:5] / 2
        native_rotation = np.array(
            [[native_c[k], -native_s[k]], [native_s[k], native_c[k]]],
            dtype=np.float64,
        )
        reference = local @ native_rotation + box[:2]
        bound = gamma * (np.abs(local) @ np.abs(native_rotation) + np.abs(box[:2]))
        np.testing.assert_array_less(
            np.abs(corner[[0, 3, 7, 4], :2] - reference),
            np.nextafter(bound, np.inf),
            err_msg=f"原生corners不符合旧版顺时针yaw及浮点误差界: box={k}",
        )
        # 原生成器的世界→局部矩阵A使用已舍入的系数，det(A)不严格为1。
        # 逆矩阵必须除以 c*c+s*s，不能用A.T假装精确正交。
        c, s = float(np.cos(box[6])), float(np.sin(box[6]))
        vertices = local @ np.array([[c, -s], [s, c]]) / (c * c + s * s)
        polygon = Polygon(vertices + box[:2])
        raster = contains(polygon, xy[..., 0], xy[..., 1]) | touches(
            polygon, xy[..., 0], xy[..., 1]
        )
        # 只有几何footprint为空时应用既有中心回填，并同样逐像素严格核验。
        if not raster.any() and (np.abs(box[:2]) < 51.2).all():
            col, row = np.floor((box[:2] + 51.2) / (102.4 / 150)).astype(int)
            raster[row, col] = True
        np.testing.assert_array_equal(raster, masks[k], err_msg=f"box={k}")
    return len(corners)


def rebuild_map_checks(frames: list[dict], lineage: dict) -> list[dict]:
    """每个允许训练场景首帧从原始 map JSON 重建全部实例，逐像素核对。"""
    from scripts.stage2.spatial_annotations import AUTHOR, load_module, tagged_vectors

    tools = AUTHOR / "projects/mmdet3d_plugin/datasets/utils"
    vector = load_module("lg_raw_vector", tools / "vector_map.py")
    raster = load_module("lg_raw_raster", tools / "rasterize.py")
    mapper = vector.VectorizedLocalMap(str(DATA), (30.0, 60.0), (200, 400))
    checks = []
    for scene in sorted({f["scene_name"] for f in frames if f["split"] == "train"}):
        frame = next(f for f in frames if f["scene_name"] == scene)
        vectors, tagged = tagged_vectors(mapper, frame, frame["location"])
        semantic, *_ = raster.preprocess_map(
            vectors, (30.0, 60.0), (200, 400), 3, 5, 36
        )
        gt = (semantic.numpy() * np.arange(1, 4)[:, None, None]).sum(0).astype(np.uint8)
        groups: dict[str, list] = {}
        for value, identity in tagged:
            groups.setdefault(identity, []).append(value)
        with np.load(frame["source_file"]) as source:
            np.testing.assert_array_equal(gt, source["gt"])
            for k, identity in enumerate(frame["instance_ids"]):
                values = groups[identity]
                single, *_ = raster.preprocess_map(
                    values, (30.0, 60.0), (200, 400), 3, 5, 36
                )
                cls = values[0]["type"] + 1
                mask = single[cls - 1].numpy().astype(bool) & (gt == cls)
                np.testing.assert_array_equal(mask, source["masks"][k])
        path = DATA / "maps/expansion" / f"{frame['location']}.json"
        lineage[str(path)] = sha(path)
        checks.append(
            dict(
                sample_token=frame["sample_token"],
                scene_name=scene,
                semantic_exact=True,
                instance_masks_exact=True,
                instance_record_ids=frame["instance_ids"],
            )
        )
    return checks


def make_figures(out: Path, rows: list[dict], dataset: object) -> None:
    """固定前三个训练帧出 image/真实目标/同类干扰三联图，保留 annotation token。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    selected = []
    for row in rows:
        if (
            row["split"] == "train"
            and row["task"] == "object"
            and row["target_present"]
            and row["same_direction_distractors"]
            and row["qualifier"] == "nearest"
            and row["sample_token"] not in {r["sample_token"] for r in selected}
        ):
            selected.append(row)
        if len(selected) == 3:
            break
    for k, row in enumerate(selected):
        label = load_supervision(row, out, training=True)
        info = dataset.data_infos[row["index"]]
        camera = Path(info["cams"]["CAM_FRONT"]["data_path"])
        image = Image.open(camera)
        image.thumbnail((800, 450))
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        axes[0].imshow(image)
        axes[0].set_title("CAM_FRONT (original sensor image)")
        for ax, key, title in zip(
            axes[1:],
            ("target", "distractor"),
            ("GT target footprint", "Same-class distractors"),
        ):
            ax.imshow(
                label[key][0, 0], origin="lower", extent=(-51.2, 51.2, -51.2, 51.2)
            )
            ax.set_xlabel("LiDAR x / m (column)")
            ax.set_ylabel("LiDAR y / m (row)")
            ax.set_title(title)
        fig.suptitle(
            f"{row['text']}\nsample={row['sample_token']}\nannotation={row['target_annotation_tokens']}",
            fontsize=8,
        )
        path = out / "figures" / f"gt_triplet_{k:02d}.png"
        path.parent.mkdir(exist_ok=True)
        fig.savefig(path, dpi=120)
        plt.close(fig)


def save_label(
    out: Path,
    row: dict,
    target: np.ndarray,
    other: np.ndarray,
    valid: np.ndarray,
    **extra: np.ndarray,
) -> None:
    """只增写新监督，形状为 [1,1,H,W]，明确标注目标存在。"""
    if (target & other).any():
        raise ValueError("目标/干扰标签冲突")
    path = out / "labels" / f"{row['instruction_id']}.npz"
    path.parent.mkdir(exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            target=target[None, None],
            distractor=other[None, None],
            valid=valid[None, None],
            **extra,
        )
    row.update(
        label_file=str(path.relative_to(out)),
        label_sha256=sha(path),
        target_pixels=int(target.sum()),
        distractor_pixels=int(other.sum()),
        target_present=bool(target.any()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    prepare(parser.parse_args().out.resolve())

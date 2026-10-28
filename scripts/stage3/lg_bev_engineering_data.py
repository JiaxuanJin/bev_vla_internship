"""正式split的真实GT构建：原始六相机/3D标注/地图记录及GT派生语言。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from scripts.stage2.low_memory_dataset import subset_nuscenes_class
from scripts.stage2.spatial_annotations import DATA, tagged_vectors, metric_grid
from scripts.stage3.lg_bev_common import ROOT, read, sha, write
from scripts.stage3.lg_bev_data import box_masks, exclusive_masks, verify_box_corners
from scripts.stage3.lg_bev_engineering_cache import (
    CACHE_ROOT,
    RollingCache,
    atomic_save,
)
from scripts.stage3.lg_bev_formal_inventory import CAMERAS

FORMAL = ROOT / "results/lg_bev_formal_training_audit_20261027"
CLASSES = (
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
)


def formal_frames(split: str = "train") -> list[dict]:
    """读取封存清单，不重新划分；训练入口默认拒绝其他split。"""
    contract = read(FORMAL / "formal_execution_contract.json")
    if sha(FORMAL / "formal_split_manifest.json") != contract["split_manifest_sha256"]:
        raise ValueError("正式split来源已变化")
    return [
        r
        for r in read(FORMAL / "formal_split_manifest.json")["frames"]
        if r["split"] == split
    ]


def fixed_frames(count: int = 32) -> list[dict]:
    """元数据哈希选新训练场景，场景内从真正起点连续取帧，不按表现选样本。"""
    groups: dict[str, list] = {}
    for row in formal_frames():
        groups.setdefault(row["scene_token"], []).append(row)
    scenes = sorted(
        groups,
        key=lambda t: hashlib.sha256(("eng32-20261027:" + t).encode()).hexdigest(),
    )
    chosen = []
    for scene in scenes:
        rows = sorted(groups[scene], key=lambda r: r["timestamp"])
        if rows[0]["previous_sample_token"]:
            raise ValueError("正式场景缺少起点，不能构建正确时序")
        chosen.extend(rows[: count - len(chosen)])
        if len(chosen) == count:
            return chosen
    raise ValueError("训练帧不足")


def initialize_author() -> Any:
    """仅注册原作者类；没有调用旧mini初始化器或更改作者源码。"""
    import sys
    import importlib
    from mmcv import Config
    from scripts.stage2.spatial_annotations import AUTHOR

    for name, value in [
        ("bool", np.bool_),
        ("int", int),
        ("float", float),
        ("complex", complex),
    ]:
        if name not in np.__dict__:
            setattr(np, name, value)
    if str(AUTHOR) not in sys.path:
        sys.path.insert(0, str(AUTHOR))
    importlib.import_module("projects.mmdet3d_plugin")
    return Config.fromfile(
        str(AUTHOR / "projects/configs/bevformer/bevformer_small_seg_det.py")
    )


class SceneData:
    """整个允许场景的有界原始SDK视图，所有前后annotation均在场景内。

    SDK元数据裁剪只限制内存；native temporal遍历使用完整scene infos，
    不把单帧mini缓存当作队列，也不跨场景沿用history。
    """

    def __init__(self, rows: list[dict], *, allowed_split: str = "train") -> None:
        from nuscenes import NuScenes

        if (
            allowed_split not in ("train", "development")
            or not rows
            or any(r["split"] != allowed_split for r in rows)
        ):
            raise ValueError(
                "禁止打开Cold Final/混合split；development仅由正式终点评价入口调用"
            )
        if len({r["scene_token"] for r in rows}) != 1:
            raise ValueError("SceneData一次只构建一个完整场景")
        self.rows = sorted(rows, key=lambda r: r["timestamp"])
        self.cfg = initialize_author()
        index_path = CACHE_ROOT / "metadata" / f'{rows[0]["scene_token"]}.pt'
        source_stats = {
            name: (
                str((DATA / "v1.0-trainval" / name).resolve()),
                (DATA / "v1.0-trainval" / name).stat().st_size,
                (DATA / "v1.0-trainval" / name).stat().st_mtime_ns,
            )
            for name in (
                "sample.json",
                "sample_data.json",
                "sample_annotation.json",
                "ego_pose.json",
                "calibrated_sensor.json",
            )
        }
        if index_path.exists():
            if sha(index_path) != index_path.with_suffix(".sha256").read_text().strip():
                raise ValueError("原始场景索引损坏")
            saved = torch.load(index_path, map_location="cpu", weights_only=False)
            if saved["source_stats"] != source_stats:
                raise ValueError("原始元数据已变化，拒绝复用场景索引")
            tables = saved["tables"]

            class IndexedNuScenes(NuScenes):
                def __load_table__(self, table_name):
                    return copy.deepcopy(tables[table_name])

            self.nusc = IndexedNuScenes(
                version="v1.0-trainval", dataroot=str(DATA), verbose=False
            )
        else:
            cls = subset_nuscenes_class(NuScenes, {r["sample_token"] for r in rows})
            self.nusc = cls(version="v1.0-trainval", dataroot=str(DATA), verbose=False)
            owned = RollingCache()
            tables = {name: getattr(self.nusc, name) for name in self.nusc.table_names}
            tables["map"] = [
                {k: v for k, v in record.items() if k != "mask"}
                for record in tables["map"]
            ]
            with owned.locked():
                owned.reserve_for_artifact(len(json.dumps(tables).encode()) + 2**20)
                atomic_save(index_path, dict(tables=tables, source_stats=source_stats))
                index_path.with_suffix(".sha256").write_text(sha(index_path))
        from projects.mmdet3d_plugin.datasets.nuscenes_dataset import (
            CustomNuScenesDataset,
        )
        from projects.mmdet3d_plugin.datasets.utils.vector_map import VectorizedLocalMap
        from mmdet3d.core.bbox import get_box_type
        from mmdet3d.datasets.pipelines import Compose

        # 使用同一作者类的方法和原始test Compose，避免第二份完整NuScenes驻留。
        self.dataset = CustomNuScenesDataset.__new__(CustomNuScenesDataset)
        ds = self.dataset
        ds.nusc = self.nusc
        ds.vector_map = VectorizedLocalMap(str(DATA), (30.0, 60.0), (200, 400))
        ds.data_root = str(DATA)
        ds.modality = dict(use_camera=True)
        ds.test_mode = True
        ds.with_velocity = True
        ds.use_valid_flag = bool(self.cfg.data.train.get("use_valid_flag", False))
        ds.box_type_3d, ds.box_mode_3d = get_box_type("LiDAR")
        ds.CLASSES = CLASSES
        ds.filter_empty_gt = False
        ds.pipeline = Compose(copy.deepcopy(self.cfg.test_pipeline))
        image_steps = copy.deepcopy(self.cfg.test_pipeline)
        image_steps = [
            step for step in image_steps if step["type"] != "RasterizeMapVectors"
        ]
        for step in image_steps:
            for inner in step.get("transforms", []):
                if inner["type"] == "CustomCollect3D":
                    inner["keys"] = ["img"]
        self.image_pipeline = Compose(image_steps)
        ds.data_infos = [self._info(row, index) for index, row in enumerate(self.rows)]
        self.index = {r["sample_token"]: i for i, r in enumerate(self.rows)}

    def _info(self, row: dict, index: int) -> dict:
        """按官方converter公式从原始SDK标定生成info，而非复用旧LG标签。"""
        from pyquaternion import Quaternion
        from mmdet3d.datasets import NuScenesDataset

        n = self.nusc
        sample = n.get("sample", row["sample_token"])
        lidar = n.get("sample_data", sample["data"]["LIDAR_TOP"])
        calibration = n.get("calibrated_sensor", lidar["calibrated_sensor_token"])
        pose = n.get("ego_pose", lidar["ego_pose_token"])
        lr = Quaternion(calibration["rotation"]).rotation_matrix
        er = Quaternion(pose["rotation"]).rotation_matrix
        lt = np.array(calibration["translation"])
        et = np.array(pose["translation"])
        cameras = {}
        for channel in CAMERAS:
            sd = n.get("sample_data", sample["data"][channel])
            c = n.get("calibrated_sensor", sd["calibrated_sensor_token"])
            p = n.get("ego_pose", sd["ego_pose_token"])
            cr = Quaternion(c["rotation"]).rotation_matrix
            pr = Quaternion(p["rotation"]).rotation_matrix
            # 与obtain_sensor2top的row-vector公式逐项一致，保留异步相机ego位姿。
            r = (cr.T @ pr.T) @ (np.linalg.inv(er).T @ np.linalg.inv(lr).T)
            t = (np.array(c["translation"]) @ pr.T + np.array(p["translation"])) @ (
                np.linalg.inv(er).T @ np.linalg.inv(lr).T
            )
            t -= (
                et @ (np.linalg.inv(er).T @ np.linalg.inv(lr).T)
                + lt @ np.linalg.inv(lr).T
            )
            path = DATA / sd["filename"]
            if not path.is_file():
                raise FileNotFoundError(path)
            cameras[channel] = dict(
                data_path=str(path),
                sample_data_token=sd["token"],
                sensor2lidar_rotation=r.T,
                sensor2lidar_translation=t,
                cam_intrinsic=np.array(c["camera_intrinsic"]),
            )
        _, boxes, _ = n.get_sample_data(lidar["token"])
        annotations = [n.get("sample_annotation", t) for t in sample["anns"]]
        velocity = np.array([n.box_velocity(t)[:2] for t in sample["anns"]]).reshape(
            -1, 2
        )
        for k, v in enumerate(velocity):
            velocity[k] = (np.r_[v, 0.0] @ np.linalg.inv(er).T @ np.linalg.inv(lr).T)[
                :2
            ]
        names = [NuScenesDataset.NameMapping.get(b.name, b.name) for b in boxes]
        gt = np.concatenate(
            (
                np.array([b.center for b in boxes]).reshape(-1, 3),
                np.array([b.wlh for b in boxes]).reshape(-1, 3),
                -np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(
                    -1, 1
                )
                - np.pi / 2,
            ),
            axis=1,
        )
        return dict(
            token=sample["token"],
            scene_token=row["scene_token"],
            timestamp=sample["timestamp"],
            prev=sample["prev"],
            next=sample["next"],
            frame_idx=index,
            cams=cameras,
            lidar_path=str(DATA / lidar["filename"]),
            sweeps=[],
            can_bus=np.zeros(18),
            lidar2ego_rotation=calibration["rotation"],
            lidar2ego_translation=calibration["translation"],
            ego2global_rotation=pose["rotation"],
            ego2global_translation=pose["translation"],
            gt_boxes=gt,
            gt_names=np.array(names),
            gt_velocity=velocity,
            num_lidar_pts=np.array([a["num_lidar_pts"] for a in annotations]),
            num_radar_pts=np.array([a["num_radar_pts"] for a in annotations]),
            valid_flag=np.array(
                [a["num_lidar_pts"] + a["num_radar_pts"] > 0 for a in annotations],
                dtype=bool,
            ),
            raw_annotation_tokens=sample["anns"],
        )

    def label(self, token: str) -> dict:
        """真实对象footprint、地图实例、完整语义GT及独立语言配对。"""
        from projects.mmdet3d_plugin.datasets.utils.rasterize import preprocess_map
        from scripts.stage2.spatial_detection import ego_centers

        k = self.index[token]
        row = self.rows[k]
        info = self.dataset.data_infos[k]
        vectors, tagged = tagged_vectors(self.dataset.vector_map, info, row["location"])
        semantic, *_ = preprocess_map(vectors, (30.0, 60.0), (200, 400), 3, 5, 36)
        gt = (semantic.numpy() * np.arange(1, 4)[:, None, None]).sum(0).astype(np.uint8)
        grouped = {}
        for v, identity in tagged:
            grouped.setdefault(identity, []).append(v)
        map_masks = []
        map_ids = []
        map_classes = []
        for identity, vv in grouped.items():
            raster, *_ = preprocess_map(vv, (30.0, 60.0), (200, 400), 3, 5, 36)
            c = vv[0]["type"] + 1
            mask = raster[c - 1].numpy().astype(bool) & (gt == c)
            if mask.any():
                map_masks.append(mask)
                map_ids.append(identity)
                map_classes.append(c)
        mm = np.array(map_masks, dtype=bool).reshape(-1, 200, 400)
        for c in (1, 2):
            union = (
                mm[np.array(map_classes) == c].any(0)
                if c in map_classes
                else np.zeros_like(gt, dtype=bool)
            )
            if not np.array_equal(union, gt == c):
                raise ValueError("地图实例union与完整GT不一致")
        ann = self.dataset.get_ann_info(k)
        raw_valid = (
            info["valid_flag"]
            if self.dataset.use_valid_flag
            else info["num_lidar_pts"] > 0
        )
        ids = [t for t, keep in zip(info["raw_annotation_tokens"], raw_valid) if keep]
        boxes = ann["gt_bboxes_3d"].tensor.numpy()
        labels = ann["gt_labels_3d"]
        om, fallback = box_masks(boxes)
        checked = verify_box_corners(ann["gt_bboxes_3d"], om)
        centers = ego_centers(boxes, info)
        _, ego, _ = metric_grid(info)
        calibration = {
            key: info[key]
            for key in (
                "lidar2ego_rotation",
                "lidar2ego_translation",
                "ego2global_rotation",
                "ego2global_translation",
            )
        }
        queries = []
        for task, classes in [("object", (0, 8)), ("map", (1, 2))]:
            for c in classes:
                for fb in ("front", "rear"):
                    for lr in ("left", "right"):
                        if task == "object":
                            region = (
                                (centers[:, 0] > 0)
                                if fb == "front"
                                else (centers[:, 0] < 0)
                            ) & (
                                (centers[:, 1] > 0)
                                if lr == "left"
                                else (centers[:, 1] < 0)
                            )
                            eligible = np.flatnonzero(
                                (labels == c)
                                & region
                                & (np.abs(boxes[:, :2]) < 51.2).all(1)
                            ).tolist()
                            ordered = sorted(
                                (float(np.linalg.norm(centers[i, :2])), i)
                                for i in eligible
                            )
                        else:
                            region = (
                                (ego[..., 0] > 0)
                                if fb == "front"
                                else (ego[..., 0] < 0)
                            ) & (
                                (ego[..., 1] > 0) if lr == "left" else (ego[..., 1] < 0)
                            )
                            ordered = sorted(
                                (
                                    float(
                                        np.linalg.norm(
                                            ego[mm[i] & region, :2], axis=-1
                                        ).min()
                                    ),
                                    i,
                                )
                                for i, cls in enumerate(map_classes)
                                if cls == c and (mm[i] & region).sum() >= 8
                            )
                        choices = (
                            [("all", [i for _, i in ordered])]
                            if task == "object" or not ordered
                            else []
                        )
                        if ordered and (
                            len(ordered) == 1 or ordered[1][0] - ordered[0][0] >= 1
                        ):
                            choices.append(("nearest", [ordered[0][1]]))
                        if len(ordered) > 1 and ordered[-1][0] - ordered[-2][0] >= 1:
                            choices.append(("farthest", [ordered[-1][1]]))
                        for qualifier, selected in choices:
                            name = {
                                ("object", 0): "cars",
                                ("object", 8): "pedestrians",
                                ("map", 1): "lane markings",
                                ("map", 2): "crosswalk",
                            }[(task, c)]
                            text = f'Find {"the "+qualifier+" " if qualifier!="all" else ""}{name} on the {fb} {lr}'
                            queries.append(
                                dict(
                                    task=task,
                                    target_class=c,
                                    fb=fb,
                                    lr=lr,
                                    qualifier=qualifier,
                                    selected=selected,
                                    text=text,
                                    instruction_id=f"{token}_{task}_{c}_{fb}_{lr}_{qualifier}",
                                    language_source="GT_DERIVED_TEMPLATE_NOT_DRIVELM",
                                )
                            )
        by = {
            (q["task"], q["target_class"], q["fb"], q["lr"], q["qualifier"]): q
            for q in queries
        }
        for q in queries:
            opposite = "farthest" if q["qualifier"] == "nearest" else "nearest"
            pair = (
                by.get((q["task"], q["target_class"], q["fb"], q["lr"], opposite))
                if q["qualifier"] != "all"
                else None
            )
            if pair is None:
                pair = by.get(
                    (
                        q["task"],
                        q["target_class"],
                        q["fb"],
                        "right" if q["lr"] == "left" else "left",
                        q["qualifier"],
                    )
                )
            q["pair_id"] = pair["instruction_id"] if pair else None
            q["wrong_text"] = (
                pair["text"]
                if pair
                else q["text"]
                .replace("left", "RIGHT")
                .replace("right", "left")
                .replace("RIGHT", "right")
            )
            q["pair_skip_reason"] = (
                None
                if pair
                else "counterfactual has ambiguous/no fixed target table; ranking skipped"
            )
            target_ids = ids if q["task"] == "object" else map_ids
            q["target_instance_ids"] = [target_ids[i] for i in q["selected"]]
        from mmdet3d.core.bbox import LiDARInstance3DBoxes

        velocity = np.nan_to_num(info["gt_velocity"])
        evaluation_boxes = LiDARInstance3DBoxes(
            np.concatenate((info["gt_boxes"], velocity), axis=1),
            box_dim=9,
            origin=(0.5, 0.5, 0.5),
        ).tensor
        return dict(
            frame={**row, **calibration},
            gt=torch.from_numpy(gt),
            boxes=torch.from_numpy(boxes),
            labels=torch.from_numpy(labels),
            object_masks=torch.from_numpy(om),
            map_masks=torch.from_numpy(mm),
            map_classes=torch.tensor(map_classes, dtype=torch.long),
            map_ids=map_ids,
            annotation_tokens=ids,
            queries=queries,
            corner_checks=checked,
            subpixel_fallback=fallback,
            eval_boxes=evaluation_boxes,
            eval_labels=torch.tensor(
                [CLASSES.index(n) if n in CLASSES else -1 for n in info["gt_names"]],
                dtype=torch.long,
            ),
            eval_num_points=torch.from_numpy(
                info["num_lidar_pts"] + info["num_radar_pts"]
            ),
            eval_annotation_tokens=info["raw_annotation_tokens"],
            bicycle_racks=info["gt_boxes"][
                info["gt_names"] == "static_object.bicycle_rack"
            ].tolist(),
            source_hashes={
                str(
                    DATA
                    / self.nusc.get(
                        "sample_data", self.nusc.get("sample", token)["data"][c]
                    )["filename"]
                ): sha(
                    DATA
                    / self.nusc.get(
                        "sample_data", self.nusc.get("sample", token)["data"][c]
                    )["filename"]
                )
                for c in (*CAMERAS, "LIDAR_TOP")
            },
        )


def masks(label: dict, query: dict, device: str = "cpu") -> dict[str, torch.Tensor]:
    """按实例索引即时组合mask，避免为每条指令复制大型标签文件。"""
    task = query["task"]
    source = label["object_masks" if task == "object" else "map_masks"].numpy()
    classes = label["labels" if task == "object" else "map_classes"].numpy()
    selected = query["selected"]
    others = [
        i
        for i, c in enumerate(classes)
        if c == query["target_class"] and i not in selected
    ]
    target, distractor, _ = exclusive_masks(source, selected, others)
    if task == "map":
        _, xy, _ = metric_grid(label["frame"])
        region = ((xy[..., 0] > 0) if query["fb"] == "front" else (xy[..., 0] < 0)) & (
            (xy[..., 1] > 0) if query["lr"] == "left" else (xy[..., 1] < 0)
        )
        target &= region
        distractor = (
            source[classes == query["target_class"]].any(0)
            if (classes == query["target_class"]).any()
            else np.zeros_like(target)
        ) & ~target
    return {
        k: torch.from_numpy(v.copy())[None, None].to(device)
        for k, v in dict(
            target=target, distractor=distractor, valid=np.ones_like(target)
        ).items()
    }


def prepare_labels(
    cache: RollingCache, run: Path, rows: list[dict], *, split: str = "train"
) -> dict:
    """分场景重建真实GT并以8帧块原子保存；失败不得静默删帧。"""
    started = time.perf_counter()
    all_train = formal_frames(split)
    groups = {}
    if not run.resolve().is_relative_to(cache.root.resolve()):
        raise ValueError("新增标签只能写G盘缓存根目录")
    for r in all_train:
        groups.setdefault(r["scene_token"], []).append(r)
    selected = {r["sample_token"] for r in rows}
    records = []
    failures = []
    journal = run / "labels_commit.jsonl"
    completed = {}
    verified = set()
    for (encoded,) in cache.db.execute(
        "SELECT record FROM labels WHERE run=?", (str(run),)
    ):
        item = json.loads(encoded)
        if item["sample_token"] not in selected:
            raise ValueError("标签恢复catalog含其他split/frame")
        if item["label_path"] not in verified:
            if sha(Path(item["label_path"])) != item["label_sha256"]:
                raise ValueError("恢复标签块损坏")
            verified.add(item["label_path"])
        completed[item["sample_token"]] = item
    for scene in dict.fromkeys(r["scene_token"] for r in rows):
        selected_scene = [
            r
            for r in sorted(groups[scene], key=lambda r: r["timestamp"])
            if r["sample_token"] in selected
        ]
        if all(r["sample_token"] in completed for r in selected_scene):
            records.extend(completed[r["sample_token"]] for r in selected_scene)
            continue
        data = SceneData(groups[scene], allowed_split=split)
        block = {}
        for row in data.rows:
            if row["sample_token"] not in selected:
                continue
            if row["sample_token"] in completed:
                records.append(completed[row["sample_token"]])
                continue
            try:
                label = data.label(row["sample_token"])
                block[row["sample_token"]] = label
            except Exception as exc:
                failures.append(dict(sample_token=row["sample_token"], error=repr(exc)))
                raise
            if (
                len(block) == 8
                or row["sample_token"] == selected_scene[-1]["sample_token"]
            ):
                path = run / "labels" / f"block_{len(records):06d}.pt"
                with cache.locked():
                    cache.reserve_for_artifact(
                        sum(
                            sum(
                                v.numel() * v.element_size()
                                for v in item.values()
                                if isinstance(v, torch.Tensor)
                            )
                            for item in block.values()
                        )
                        + 2**20
                    )
                    packed = {}
                    for t, item in block.items():
                        item = dict(item)
                        for field in ("object_masks", "map_masks"):
                            original = item.pop(field).numpy()
                            item[field + "_shape"] = list(original.shape)
                            item[field + "_packed"] = torch.from_numpy(
                                np.packbits(original.reshape(-1))
                            )
                        packed[t] = item
                    if path.exists():
                        prior = torch.load(path, map_location="cpu", weights_only=False)
                        if not nested_equal(prior, packed):
                            raise ValueError(
                                "未登记既有GT块与重新构建内容不同，禁止覆盖"
                            )
                    else:
                        atomic_save(path, packed)
                digest = sha(path)
                for token, item in block.items():
                    qs = item["queries"]
                    empty = sum(not masks(item, q)["target"].any() for q in qs)
                    records.append(
                        dict(
                            sample_token=token,
                            scene_token=scene,
                            label_path=str(path),
                            label_sha256=digest,
                            queries=len(qs),
                            empty_queries=empty,
                            target_instance_occurrences=sum(
                                len(q["selected"]) for q in qs
                            ),
                            object_instances=len(item["annotation_tokens"]),
                            map_instances=len(item["map_ids"]),
                        )
                    )
                    with journal.open("a") as stream:
                        stream.write(json.dumps(records[-1]) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                cache.db.executemany(
                    "INSERT OR REPLACE INTO labels VALUES(?,?,?)",
                    [
                        (str(run), r["sample_token"], json.dumps(r))
                        for r in records[-len(block) :]
                    ],
                )
                cache.db.commit()
                block = {}
            print("RAW_LABEL_FRAME", row["scene_name"], row["sample_token"], flush=True)
        del data
    report = dict(
        status="PASS" if len(records) == len(rows) else "FAIL",
        success=len(records),
        failures=failures,
        label_count=sum(r["queries"] for r in records),
        target_instance_occurrences=sum(
            r["target_instance_occurrences"] for r in records
        ),
        empty_queries=sum(r["empty_queries"] for r in records),
        seconds=time.perf_counter() - started,
        frames=records,
        language_source="GT_DERIVED_TEMPLATE_NOT_DRIVELM",
        raw_metadata_sha256={
            str(DATA / "v1.0-trainval" / n): sha(DATA / "v1.0-trainval" / n)
            for n in (
                "sample_annotation.json",
                "calibrated_sensor.json",
                "ego_pose.json",
            )
        },
    )
    report["empty_query_fraction"] = report["empty_queries"] / report["label_count"]
    with cache.locked():
        cache.reserve_for_artifact(len(json.dumps(report).encode()) + 2**20)
        write(run / "data_report.json", report)
    return report


def nested_equal(a: Any, b: Any) -> bool:
    """验证崩溃后已完成但未登记的GT块，不覆盖旧内容或只比较形状。"""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(nested_equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            isinstance(b, type(a))
            and len(a) == len(b)
            and all(nested_equal(x, y) for x, y in zip(a, b))
        )
    return bool(a == b)


class LabelStore:
    """每次只在内存保留一个已校验8帧标签块。"""

    def __init__(self, report: dict) -> None:
        self.records = {r["sample_token"]: r for r in report["frames"]}
        self.path = None
        self.block = {}

    def get(self, token: str) -> dict:
        """标签读取前校验摘要，不对原GT作任何写入。"""
        row = self.records[token]
        if self.path != row["label_path"]:
            path = Path(row["label_path"])
            if sha(path) != row["label_sha256"]:
                raise ValueError("标签块损坏")
            self.block = torch.load(path, map_location="cpu", weights_only=False)
            self.path = row["label_path"]
        item = dict(self.block[token])
        for field in ("object_masks", "map_masks"):
            if field in item:
                continue
            shape = item.pop(field + "_shape")
            value = item.pop(field + "_packed").numpy()
            item[field] = torch.from_numpy(
                np.unpackbits(value, count=int(np.prod(shape)))
                .reshape(shape)
                .astype(bool)
            )
        return item

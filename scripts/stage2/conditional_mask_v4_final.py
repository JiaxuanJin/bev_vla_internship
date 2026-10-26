"""V4 封存后一次性新适配场景评价；预训练未见泛化仍不可证明。"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.stage2.conditional_mask_v4_data import OUT, SEED, region_mask, sha, write
from scripts.stage2.conditional_mask_v4_experiment import load_runtime, shuffle_words
from scripts.stage2.conditional_mask_v4_evaluate import evaluate_maps, evaluate_objects
from scripts.stage2.spatial_annotations import (
    AUTHOR,
    DATA,
    load_module,
    metric_grid,
    tagged_vectors,
)
from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.low_memory_dataset import bounded_metadata, json_records
from scripts.stage2.micro_training import get_batch, forward_payload
from scripts.stage2.spatial_detection import (
    ego_centers,
    in_region,
    parse_object_instruction,
)
from scripts.stage2.spatial_v3_experiment import decode, head_forward


def prepare() -> dict:
    """先固定未用于本轮选择的场景和权重，再构造其真实标签。"""
    scene_data = json.loads((DATA / "v1.0-trainval/scene.json").read_text())
    scenes = {s["token"]: s for s in scene_data}
    logs = {
        r["token"]: r for r in json.loads((DATA / "v1.0-trainval/log.json").read_text())
    }
    mini_scenes = {
        r["token"] for r in json.loads((DATA / "v1.0-mini/scene.json").read_text())
    }
    full_path = (
        ROOT / "outputs/bevformer_full_infos/bevformer_full_infos_temporal_train.pkl"
    )
    with full_path.open("rb") as f:
        full = pickle.load(f)
    candidates = sorted(
        {i["scene_token"] for i in full["infos"]} - mini_scenes,
        key=lambda s: scenes[s]["name"],
    )
    scene = next(s for s in candidates if scenes[s]["name"] != "scene-0001")
    assert scenes[scene]["name"] == "scene-0002"
    available = sorted(
        [i for i in full["infos"] if i["scene_token"] == scene],
        key=lambda i: i["timestamp"],
    )
    selected = [available[n] for n in np.linspace(0, len(available) - 1, 4, dtype=int)]
    checkpoints = json.loads((OUT / "training_summary.json").read_text())["branches"]
    sensors = []
    for info in selected:
        for p in [Path(c["data_path"]) for c in info["cams"].values()] + [
            Path(info["lidar_path"])
        ]:
            assert p.is_file()
            sensors.append(dict(path=str(p), bytes=p.stat().st_size))
    write(
        OUT / "final_protocol.json",
        dict(
            scene_token=scene,
            scene_name=scenes[scene]["name"],
            sample_tokens=[i["token"] for i in selected],
            source_infos_sha256=sha(full_path),
            sensor_files=sensors,
            checkpoints=checkpoints,
            threshold=0.5,
            optimizer_updates_after_this_point=0,
            selection="metadata-only next official-train scene outside mini after already-opened scene0001; fixed4linspaceframes; all AB terminal weights sealed",
            scope="new adaptation-scene evaluation, not pretraining-unseen; author officialTrain pretraining overlap possible",
            exposure_audit="rg on all prior outputs JSON/JSONL/MD found scene0002 only in DriveLM official full test/test_eval/test_llama preprocessing. No adaptation training/prediction/selection artifact found. Metadata preprocessing exposure disclosed; absence of unrecorded use cannot be proved.",
            initial_protocol_supplement="Initial strict final BLOCKED remains true for pretraining-unseen. Subsequent asset search found a traceable adaptation-scene candidate; no weight/threshold change before opening.",
            source_sha256={
                str(p.relative_to(ROOT)): sha(p)
                for p in (
                    Path(__file__),
                    ROOT / "scripts/stage2/conditional_mask_v4_evaluate.py",
                )
            },
        ),
    )
    directory = OUT / "final_labels"
    directory.mkdir()
    tools = AUTHOR / "projects/mmdet3d_plugin/datasets/utils"
    vector = load_module("v4_final_vector", tools / "vector_map.py")
    raster = load_module("v4_final_raster", tools / "rasterize.py")
    mapper = vector.VectorizedLocalMap(str(DATA), (30.0, 60.0), (200, 400))
    frames, annotations, rejected = [], [], []
    for index, info in enumerate(selected):
        frame = dict(
            index=index,
            dataset_index=index,
            split="final",
            scene_token=scene,
            scene_name=scenes[scene]["name"],
            sample_token=info["token"],
            location=logs[scenes[scene]["log_token"]]["location"],
            **{
                k: info[k]
                for k in (
                    "lidar2ego_rotation",
                    "lidar2ego_translation",
                    "ego2global_rotation",
                    "ego2global_translation",
                )
            },
        )
        vectors, tagged = tagged_vectors(mapper, info, frame["location"])
        semantic, *_ = raster.preprocess_map(
            vectors, (30.0, 60.0), (200, 400), 3, 5, 36
        )
        gt = (semantic.numpy() * np.arange(1, 4)[:, None, None]).sum(0).astype(np.uint8)
        groups = defaultdict(list)
        for v, identity in tagged:
            groups[identity].append(v)
        identities, classes, masks = [], [], []
        for identity, vv in groups.items():
            single, *_ = raster.preprocess_map(vv, (30.0, 60.0), (200, 400), 3, 5, 36)
            cls = vv[0]["type"] + 1
            mask = single[cls - 1].numpy().astype(bool) & (gt == cls)
            if mask.any():
                identities.append(identity)
                classes.append(cls)
                masks.append(mask)
        masks = np.asarray(masks, dtype=bool)
        classes = np.asarray(classes)
        grid, ego, _ = metric_grid(info)
        source = directory / f"frame_{index:03d}.npz"
        np.savez_compressed(
            source,
            gt=gt,
            masks=masks,
            classes=classes,
            ego_xy=ego[..., :2],
            sampling_grid=grid.numpy(),
        )
        frame.update(
            instance_ids=identities, source_file=str(source), source_sha256=sha(source)
        )
        frames.append(frame)
        for cls, name in ((1, "lane markings"), (2, "crosswalk")):
            for fb, lr in (
                ("front", "left"),
                ("front", "right"),
                ("rear", "left"),
                ("rear", "right"),
            ):
                region = region_mask(ego, fb, lr)
                candidates = [
                    k
                    for k, c in enumerate(classes)
                    if c == cls and (masks[k] & region).any()
                ]
                ordered = sorted(
                    (
                        float(
                            np.linalg.norm(
                                ego[..., :2][masks[k] & region], axis=-1
                            ).min()
                        ),
                        k,
                    )
                    for k in candidates
                )
                choices = []
                if len(ordered) == 1:
                    choices = [("unique", ordered[0])]
                elif len(ordered) > 1:
                    if ordered[1][0] - ordered[0][0] >= 1:
                        choices.append(("nearest", ordered[0]))
                    if ordered[-1][0] - ordered[-2][0] >= 1:
                        choices.append(("farthest", ordered[-1]))
                eligible = [
                    (q, d, k) for q, (d, k) in choices if (masks[k] & region).sum() >= 8
                ]
                for q, (d, k) in choices:
                    if (masks[k] & region).sum() < 8:
                        rejected.append(
                            dict(
                                sample_token=info["token"],
                                qualifier=q,
                                instance_id=identities[k],
                                reason="true selected target<8px",
                            )
                        )
                for qualifier, distance, k in eligible:
                    target = masks[k] & region
                    others = [masks[j] & region & ~target for j in candidates if j != k]
                    prefix = "" if qualifier == "unique" else qualifier + " "
                    text = f"Find the {prefix}{name} on the {fb} {lr}"
                    opposite = "farthest" if qualifier == "nearest" else "nearest"
                    instruction_id = f"{info['token']}_c{cls}_{fb}_{lr}_{qualifier}"
                    filename = directory / (instruction_id + ".npz")
                    np.savez_compressed(
                        filename,
                        target=target,
                        same_class_other=masks[classes == cls].any(0) & ~target,
                        quadrant_other=(
                            np.logical_or.reduce(others)
                            if others
                            else np.zeros_like(target)
                        ),
                        region=region,
                    )
                    annotations.append(
                        {
                            **frame,
                            "instruction_id": instruction_id,
                            "text": text,
                            "target_class": cls,
                            "qualifier": qualifier,
                            "fb": fb,
                            "lr": lr,
                            "instance_id": identities[k],
                            "selected_index": k,
                            "distance_m": distance,
                            "label_file": str(filename),
                            "label_sha256": sha(filename),
                            "counterfactual_text": (
                                f"Find the {opposite} {name} on the {fb} {lr}"
                                if len(eligible) == 2
                                else None
                            ),
                            "target_pixels": int(target.sum()),
                            "visible_candidate_ids": [
                                identities[j] for _, j in ordered
                            ],
                        }
                    )
        print("V4_FINAL_REAL_LABEL", info["token"], len(identities), flush=True)
    with (directory / "dataset_infos.pkl").open("xb") as f:
        pickle.dump(dict(infos=selected, metadata={"version": "v1.0-trainval"}), f)
    raw = defaultdict(list)
    wanted = {i["token"] for i in selected}
    for row in json_records(DATA / "v1.0-trainval/sample_annotation.json"):
        if row["sample_token"] in wanted:
            raw[row["sample_token"]].append(row)
    write(directory / "object_records.json", raw)
    result = dict(
        frames=frames,
        annotations=annotations,
        counts=dict(Counter(r["split"] for r in annotations)),
        paired_counts=dict(
            Counter(r["split"] for r in annotations if r["counterfactual_text"])
        ),
        rejected=rejected,
    )
    write(directory / "manifest.json", result)
    return result


def object_prompts(frames: list[dict], dataset: Any) -> list[dict]:
    """只读真实annotation中心核验对象指令，复用相同区域集合协议。"""
    from pyquaternion import Quaternion

    raw = json.loads((OUT / "final_labels/object_records.json").read_text())
    result = []
    for frame in frames:
        ann = dataset.get_ann_info(frame["index"])
        boxes, labels = ann["gt_bboxes_3d"].tensor.numpy(), ann["gt_labels_3d"]
        ego = ego_centers(boxes, frame)
        world = ego @ Quaternion(
            frame["ego2global_rotation"]
        ).rotation_matrix.T + np.asarray(frame["ego2global_translation"])
        records = raw[frame["sample_token"]]
        dd = np.linalg.norm(
            world[:, None] - np.array([r["translation"] for r in records])[None],
            axis=-1,
        )
        ids = dd.argmin(1)
        assert dd.min(1).max() < 0.001 and len(set(ids)) == len(ids)
        for name in ("cars", "pedestrians"):
            for fb, lr in (
                ("front", "left"),
                ("front", "right"),
                ("rear", "left"),
                ("rear", "right"),
            ):
                text = f"Find {name} on the {fb} {lr}"
                prompt = parse_object_instruction(text)
                region = in_region(ego, prompt)
                result.append(
                    {
                        **frame,
                        "instruction_id": f"{frame['sample_token']}_{name}_{fb}_{lr}",
                        "text": text,
                        "prompt": prompt,
                        "gt_annotation_ids": [records[i]["token"] for i in ids],
                        "gt_instance_ids": [records[i]["instance_token"] for i in ids],
                        "relevant_gt_indices": np.flatnonzero(
                            (labels == prompt["class_id"]) & region
                        ).tolist(),
                        "irrelevant_gt_indices": np.flatnonzero(
                            (labels == prompt["class_id"]) & ~region
                        ).tolist(),
                        "geometry_max_error_m": float(dd.min(1).max()),
                    }
                )
    return result


def main() -> None:
    """模型封存后只打开一次，完成后禁止新选择与追加优化。"""
    from mmcv import Config
    from scripts.stage2.joint_runtime import _first_tensor

    argparse.ArgumentParser(description=__doc__).parse_args()
    model, wrapped, pipeline, old_dataset, old_manifest, old_features, text = (
        load_runtime()
    )
    from mmdet3d.datasets import build_dataset

    old_features.clear()
    manifest = prepare()
    cfg = Config.fromfile(
        str(AUTHOR / "projects/configs/bevformer/bevformer_small_seg_det.py")
    )
    dc = copy.deepcopy(cfg.data.test)
    dc.ann_file = str(OUT / "final_labels/dataset_infos.pkl")
    dc.test_mode = True
    dc.pop("samples_per_gpu", None)
    with bounded_metadata({f["sample_token"] for f in manifest["frames"]}):
        dataset = build_dataset(dc)
    cache = OUT / "final_cache"
    cache.mkdir()
    features, proof = {}, []
    for frame in manifest["frames"]:
        data = get_batch(dataset, frame["index"], SEED)
        with np.load(frame["source_file"]) as f:
            np.testing.assert_array_equal(
                _first_tensor(data["semantic_indices"]).reshape(200, 400).cpu().numpy(),
                f["gt"],
            )
        transformer = model.pts_bbox_head.transformer
        original = transformer.get_bev_features
        captured = {}

        def capture(*args: Any, **kwargs: Any) -> torch.Tensor:
            value = original(*args, **kwargs)
            captured["bev"] = BEVGrid().to_map(value).detach()
            return value

        transformer.get_bev_features = capture
        model.prev_frame_info = dict(
            prev_bev=None, scene_token=None, prev_pos=0, prev_angle=0
        )
        try:
            with torch.no_grad():
                direct = wrapped(
                    return_loss=False,
                    rescale=True,
                    **forward_payload(data, inference=True),
                )[0]["pts_bbox"]
        finally:
            del transformer.get_bev_features
        features[frame["index"]] = captured["bev"]
        with torch.no_grad():
            result = decode(model, head_forward(model, captured["bev"]))
        errors = {}
        for k, key in (
            ("boxes", "boxes_3d"),
            ("scores", "scores_3d"),
            ("labels", "labels_3d"),
        ):
            expected = (
                direct[key].tensor.cpu().numpy()
                if k == "boxes"
                else direct[key].cpu().numpy()
            )
            errors[k] = float(np.abs(result[k] - expected).max())
        assert max(errors.values()) < 1e-4
        filename = cache / f"bev_{frame['index']:03d}.npz"
        np.savez_compressed(filename, bev=captured["bev"].cpu().numpy())
        proof.append(
            dict(
                sample_token=frame["sample_token"],
                cache_file=str(filename),
                sha256=sha(filename),
                full_image_head_equivalence=errors,
                author_pipeline_gt_exact=True,
            )
        )
    write(
        cache / "cache.json",
        dict(
            frames=proof,
            pipeline="real six-camera test; FP32 BEV, no language injection",
        ),
    )
    sentences = {r["text"] for r in manifest["annotations"]}
    from scripts.stage2.spatial_v2_experiment import swap_direction

    sentences |= {swap_direction(r["text"]) for r in manifest["annotations"]}
    sentences |= {
        r["counterfactual_text"]
        for r in manifest["annotations"]
        if r["counterfactual_text"]
    }
    sentences |= {shuffle_words(r["text"]) for r in manifest["annotations"]}
    for sentence in sorted(sentences - text.keys()):
        with torch.no_grad():
            text[sentence] = pipeline.encoder([sentence], "cuda")
    training = json.loads((OUT / "training_summary.json").read_text())
    maps = evaluate_maps(model, pipeline, manifest, features, text, training, "final")
    objects = evaluate_objects(
        model,
        pipeline,
        dataset,
        manifest,
        features,
        text,
        "final",
        object_prompts(manifest["frames"], dataset),
    )
    write(
        OUT / "report/final_metrics.json",
        dict(
            **maps,
            detection=objects,
            scene_name="scene-0002",
            optimizer_updates_after_opening=0,
            scope="new adaptation-scene-only, author pretraining overlap possible; strict pretraining-unseen generalization BLOCKED",
        ),
    )


if __name__ == "__main__":
    main()

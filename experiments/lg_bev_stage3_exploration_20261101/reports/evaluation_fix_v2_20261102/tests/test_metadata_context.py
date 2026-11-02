"""与原场景完整索引及未修改的官方 SDK 逐标注比较速度。"""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

import metadata_context
from core import SHARED, read, selected_rows, sha


@pytest.fixture(scope="module")
def scene_tables():
    original = Path(__file__).resolve().parents[3]
    config = read(original / "configs/main.json")
    rows = selected_rows(config, "diagnostic")[:32]
    path = SHARED / "metadata" / (rows[0]["scene_token"] + ".pt")
    assert sha(path) == path.with_suffix(".sha256").read_text().strip()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    return rows, saved["tables"]


def indexed_class(tables):
    from nuscenes import NuScenes

    class IndexedNuScenes(NuScenes):
        def __load_table__(self, name):
            return copy.deepcopy(tables[name])

    return IndexedNuScenes


@pytest.mark.parametrize("prefix", [2, 32])
def test_velocity_matches_full_scene_sdk(scene_tables, monkeypatch, prefix):
    from nuscenes import NuScenes
    from scripts.stage2.spatial_annotations import DATA

    rows, tables = scene_tables
    monkeypatch.setattr(
        metadata_context,
        "json_records",
        lambda path: iter(copy.deepcopy(tables[Path(path).stem])),
    )
    tokens = {r["sample_token"] for r in rows[:prefix]}
    original_tables = copy.deepcopy(tables)
    repaired = metadata_context.velocity_subset_nuscenes_class(NuScenes, tokens)(
        version="v1.0-trainval", dataroot=str(DATA), verbose=False
    )
    reference = indexed_class(tables)(
        version="v1.0-trainval", dataroot=str(DATA), verbose=False
    )
    annotations = [
        a for a in reference.sample_annotation if a["sample_token"] in tokens
    ]
    assert annotations
    assert repaired.velocity_context["context_annotation_tokens"]
    for annotation in annotations:
        token = annotation["token"]
        np.testing.assert_allclose(
            repaired.box_velocity(token),
            reference.box_velocity(token),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
        for name in ("prev", "next", "translation", "sample_token"):
            assert repaired.get("sample_annotation", token)[name] == annotation[name]
    # 缓存 IndexedNuScenes 的构造路径也必须保留相同速度。
    cached = {
        name: copy.deepcopy(getattr(repaired, name)) for name in repaired.table_names
    }
    for row in cached["map"]:
        row.pop("mask", None)
    restored = indexed_class(cached)(
        version="v1.0-trainval", dataroot=str(DATA), verbose=False
    )
    for annotation in annotations:
        np.testing.assert_allclose(
            restored.box_velocity(annotation["token"]),
            reference.box_velocity(annotation["token"]),
            rtol=0,
            atol=0,
            equal_nan=True,
        )
    assert tables == original_tables


def test_missing_neighbor_stops_instead_of_zero_fill(scene_tables, monkeypatch):
    from nuscenes import NuScenes
    from scripts.stage2.spatial_annotations import DATA

    rows, tables = scene_tables
    tokens = {r["sample_token"] for r in rows[:2]}
    broken = copy.deepcopy(tables)
    selected = [a for a in broken["sample_annotation"] if a["sample_token"] in tokens]
    chosen = next(a for a in selected if a["next"])
    chosen["next"] = "missing-original-neighbor"
    monkeypatch.setattr(
        metadata_context,
        "json_records",
        lambda path: iter(copy.deepcopy(broken[Path(path).stem])),
    )
    with pytest.raises(ValueError, match="原始相邻标注依赖不完整"):
        metadata_context.velocity_subset_nuscenes_class(NuScenes, tokens)(
            version="v1.0-trainval", dataroot=str(DATA), verbose=False
        )


def test_context_cannot_cross_scene(scene_tables, monkeypatch):
    from nuscenes import NuScenes
    from scripts.stage2.spatial_annotations import DATA

    rows, tables = scene_tables
    tokens = {rows[0]["sample_token"]}
    broken = copy.deepcopy(tables)
    for sample in broken["sample"]:
        if sample["token"] not in tokens:
            sample["scene_token"] = "unauthorized-scene"
    monkeypatch.setattr(
        metadata_context,
        "json_records",
        lambda path: iter(copy.deepcopy(broken[Path(path).stem])),
    )
    with pytest.raises(ValueError, match="跨越授权场景"):
        metadata_context.velocity_subset_nuscenes_class(NuScenes, tokens)(
            version="v1.0-trainval", dataroot=str(DATA), verbose=False
        )

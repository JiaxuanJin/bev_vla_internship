"""全量训练标签来源封存及每日运行准入；绝不调用训练start/resume。"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import psutil
import torch

from scripts.stage3.lg_bev_common import read, sha, write
from scripts.stage3.lg_bev_daily_state import DATA_REPORT, OUT, POLICY, source_manifest
from scripts.stage3.lg_bev_engineering_cache import CACHE_ROOT, RollingCache
from scripts.stage3.lg_bev_engineering_data import formal_frames, nested_equal


def validate_data(preflight: Path) -> dict:
    """逐块SHA/内容验证19123帧，来源索引只提取路径SHA，不把GT传给模型。"""
    started = time.perf_counter()
    report = read(DATA_REPORT)
    rows = formal_frames()
    by_token = {r["sample_token"]: r for r in rows}
    records = report["frames"]
    if (
        report["status"] != "PASS"
        or report["success"] != 19123
        or report["failures"]
        or len(records) != 19123
        or {r["sample_token"] for r in records} != set(by_token)
    ):
        raise ValueError("完整19123帧尚未成功，不能封存来源或放行")
    paths: dict[str, str] = {}
    token_paths = {}
    for record in records:
        token = record["sample_token"]
        if record["scene_token"] != by_token[token]["scene_token"]:
            raise ValueError("标签场景归属不符")
        path = Path(record["label_path"])
        if not path.resolve().is_relative_to(CACHE_ROOT.resolve()):
            raise ValueError("标签路径不属于G盘独立根目录")
        if str(path) in paths and paths[str(path)] != record["label_sha256"]:
            raise ValueError("同一块摘要冲突")
        paths[str(path)] = record["label_sha256"]
        token_paths[token] = str(path)
    scenes, empty_frames, filtered_empty = {}, [], 0
    visited = set()
    for path, digest in paths.items():
        if sha(Path(path)) != digest:
            raise ValueError("GT分片SHA校验失败：" + path)
        block = torch.load(path, map_location="cpu", weights_only=False)
        for token, label in block.items():
            if token_paths.get(token) != path or token in visited:
                raise ValueError("GT分片含额外/重复token")
            visited.add(token)
            row = by_token[token]
            if label["frame"]["scene_token"] != row["scene_token"]:
                raise ValueError("GT内容场景不符")
            if label["boxes"].ndim != 2 or label["boxes"].shape[1] != 9:
                raise ValueError("原生训练框必须为[N,9]，空框也不例外")
            if len(label["boxes"]) == 0:
                filtered_empty += 1
            if not label["eval_annotation_tokens"]:
                empty_frames.append(token)
            if any(
                q["language_source"] != "GT_DERIVED_TEMPLATE_NOT_DRIVELM"
                for q in label["queries"]
            ):
                raise ValueError("语言来源未明确标记")
            sources = scenes.setdefault(row["scene_token"], {})
            if len(label["source_hashes"]) != 7:
                raise ValueError("六相机与LiDAR来源不全")
            for raw_path, raw_digest in label["source_hashes"].items():
                if raw_path in sources and sources[raw_path] != raw_digest:
                    raise ValueError("同一原始来源SHA冲突")
                sources[raw_path] = raw_digest
    if visited != set(by_token) or len(scenes) != 476 or len(empty_frames) != 6:
        raise ValueError("全量场景/帧或六个真实空GT样本未保留")
    # 同一32帧来自不同执行分片的内容必须完全一致，不重新选择样本。
    small = read(preflight / "data_report.json")
    original = {}
    for path in {r["label_path"] for r in small["frames"]}:
        original.update(torch.load(path, map_location="cpu", weights_only=False))
    compared = 0
    for path in dict.fromkeys(token_paths[t] for t in original):
        block = torch.load(path, map_location="cpu", weights_only=False)
        for token in original.keys() & block.keys():
            if not nested_equal(block[token], original[token]):
                raise ValueError("全量标签与独立32帧参考不一致")
            compared += 1
    if compared != 32:
        raise ValueError("32帧独立交叉核对不完整")
    provenance = dict(
        data_report_sha256=sha(DATA_REPORT),
        scenes=scenes,
        frames=19123,
        label_blocks=len(paths),
        raw_sources=sum(len(v) for v in scenes.values()),
        source="actual SHA captured during original GT generation; no GT tensors",
    )
    cache = RollingCache()
    provenance_path = DATA_REPORT.parent / "source_provenance.json"
    with cache.locked():
        cache.reserve_for_artifact(len(json.dumps(provenance).encode()) + 2**20)
        if provenance_path.exists():
            if read(provenance_path) != provenance:
                raise ValueError("既有来源封存不同，拒绝覆盖")
        else:
            write(provenance_path, provenance)
    summary = dict(
        status="PASS",
        success=19123,
        failures=0,
        scenes=476,
        logs=len({r["log_token"] for r in rows}),
        blocks=len(paths),
        label_count=report["label_count"],
        empty_queries=report["empty_queries"],
        empty_query_fraction=report["empty_query_fraction"],
        target_instance_occurrences=report["target_instance_occurrences"],
        true_empty_annotation_frames=empty_frames,
        filtered_empty_frames=filtered_empty,
        independent_32frames_bitwise=compared,
        data_report_sha256=sha(DATA_REPORT),
        provenance_sha256=sha(provenance_path),
        validation_seconds=time.perf_counter() - started,
        formal_updates=0,
    )
    write(OUT / "full_gt_integrity.json", summary)
    return summary


def seal(preflight: Path) -> dict:
    """只有真实证据全部通过才登记READY；任何状态都不自动开始训练。"""
    data_path = OUT / "full_gt_integrity.json"
    cross_path = preflight / "evidence/cross_scene.json"
    resume_path = preflight / "evidence/manager_resume.json"
    control_path = OUT / "manager_control_acceptance.json"
    metadata_path = CACHE_ROOT / "metadata_bulk_train_v1/equivalence.json"
    data, cross, resume, control, metadata = map(
        read, (data_path, cross_path, resume_path, control_path, metadata_path)
    )
    if cross["source_sha256"] != source_manifest():
        raise ValueError("GPU复核源码发生变化，不能沿用旧证据")
    cache = RollingCache()
    usage = cache.usage()
    free = shutil.disk_usage(CACHE_ROOT).free
    gates = dict(
        full_train_labels=(
            data["status"] == "PASS"
            and data["success"] == 19123
            and not data["failures"]
            and metadata["status"] == "PASS"
            and metadata["scenes"] == 476
        ),
        cross_scene_native_FP32=(
            cross["status"] == "PASS"
            and cross["frames"] == 32
            and len(cross["scenes"]) == 3
            and cross["raw_FP32_heads_bitwise"]
            and cross["loss_max_abs"] <= 1e-5
            and cross["parameter_max_abs"] <= 1e-6
        ),
        real_backward_resume=(
            resume["status"] == "PASS" and resume["stage_updates"] == 8
        ),
        offline_local_assets=(
            cross["offline_network_attempts"] == 0
            and control["offline_network_attempts"] == 0
        ),
        detached_manager=(control["status"] == "PASS" and control["detached_tmux"]),
        session_and_total_budget=all(
            control[k]
            for k in (
                "automatic_stop_verified",
                "manual_stop_verified",
                "partial_gradient_resume_verified",
                "safe_to_shutdown_after_exit",
                "completed_refuses_resume",
                "formal_launch_locked",
            )
        ),
        storage_and_memory=(
            usage <= 30_000_000_000
            and free >= 250_000_000_000
            and cross["cuda_peak_allocated_bytes"] < 12 * 2**30
            and cross["cpu_peak_rss_bytes"] < 12 * 2**30
            and psutil.virtual_memory().available > 2**30
        ),
        resource_48h=(cross["conservative_active_hours"] <= 48),
    )
    artifacts = [
        data_path,
        DATA_REPORT,
        DATA_REPORT.parent / "source_provenance.json",
        cross_path,
        resume_path,
        control_path,
        metadata_path,
        CACHE_ROOT / "metadata_bulk_train_v1/manifest.json",
        OUT / "pytest_final.log",
        OUT / "launch_hold_20261027.json",
        Path(__file__),
    ]
    result = dict(
        status="READY" if all(gates.values()) else "BLOCKED",
        gates={k: "PASS" if v else "FAIL" for k, v in gates.items()},
        conservative_active_hours=cross["conservative_active_hours"],
        active_point_hours=cross["active_point_hours"],
        policy_sha256=sha(POLICY),
        source_sha256=source_manifest(),
        artifacts={str(p): sha(p) for p in artifacts},
        cache_bytes=usage,
        g_free_bytes=free,
        formal_updates=0,
        formal_start_or_resume_authorized=False,
        next_action="2026-10-28仍须新的明确用户授权；READY不触发任何训练",
    )
    write(OUT / "admission.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["validate-data", "seal"])
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    result = (
        validate_data(args.preflight)
        if args.phase == "validate-data"
        else seal(args.preflight)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

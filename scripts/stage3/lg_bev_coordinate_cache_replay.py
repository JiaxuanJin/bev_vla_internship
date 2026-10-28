"""按历史缓存的训练增强/seed 重跑真实六相机，区分缓存差异与空间问题。"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.micro_training import forward_payload, get_batch
from scripts.stage3.lg_bev_common import ROOT, sha, write
from scripts.stage3.lg_bev_coordinate_core import COORD_OUT, legal_frames
from scripts.stage3.lg_bev_native import load_native, module_sha
from scripts.stage3.lg_bev_train import frame_input


def replay(out: Path) -> None:
    """只重放历史缓存的真实图像路径，不更新参数或覆盖缓存。"""
    model, wrapped, _, dataset = load_native()
    frame = legal_frames()[0]
    original_source = ROOT / "scripts/stage2/spatial_v2_experiment.py"
    seed = 20261027 + frame["index"]
    write(
        out / "cache_replay_protocol.json",
        dict(
            scene_name=frame["scene_name"],
            sample_token=frame["sample_token"],
            seed=seed,
            original_cache_source_sha256=sha(original_source),
            preprocessing="original train pipeline includes PhotoMetricDistortionMultiViewImage; test pipeline does not",
            expected="same source/seed/model frozen, relative L2<=1e-3 for float16 cache quantization; report failure without retries",
            optimizer_updates=0,
        ),
    )
    before = module_sha(model)
    captured = []
    transformer = model.pts_bbox_head.transformer
    original = transformer.get_bev_features

    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        captured.append(value.detach().clone())
        return value

    data = get_batch(dataset, frame["index"], seed)
    transformer.get_bev_features = capture
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            losses = wrapped(return_loss=True, **forward_payload(data))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        transformer.get_bev_features = original
    assert len(captured) == 1
    fresh = BEVGrid().to_map(captured[0])
    cached, _ = frame_input(frame)
    difference = float((fresh - cached).norm() / fresh.norm())
    assert module_sha(model) == before
    write(
        out / "cache_replay.json",
        dict(
            status="PASS" if difference <= 1e-3 else "CACHE_REPRODUCTION_FAILED",
            scene_name=frame["scene_name"],
            sample_token=frame["sample_token"],
            seed=seed,
            six_raw_cameras_forwarded=True,
            features_injected=False,
            cached_bev_comparison_only=True,
            fresh_cached_relative_l2=difference,
            float16_roundtrip_relative_l2=float(
                (fresh - fresh.half().float()).norm() / fresh.norm()
            ),
            rounded_fresh_matches_cache_bitwise=torch.equal(
                fresh.half().float(), cached
            ),
            seconds=elapsed,
            optimizer_updates=0,
            frozen_sha256=before,
            losses={k: float(v.detach()) for k, v in losses.items()},
            distinction="raw test-pipeline forward and historical augmented train cache have different input photometry; cached feature comparison does not equate to raw fresh inference",
        ),
    )
    print("CACHE_REPLAY", difference, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=COORD_OUT)
    replay(parser.parse_args().out.resolve())

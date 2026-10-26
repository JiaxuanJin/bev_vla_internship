"""V5 有限预算协议与全部前序证据封存；不访问已开启留出预测。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from scripts.stage2.conditional_mask_v4_data import HISTORY, ROOT, inventory, sha, write

V4 = ROOT / "outputs/stage3_conditional_mask_v4_20261026"
OUT = Path(
    os.environ.get("BEV_V5_OUTPUT", ROOT / "outputs/stage3_coupled_v5_20261026")
).resolve()
SEED = 20261031
BRANCHES = ("global_control", "targeted_no_language", "targeted_language")


def prepare() -> None:
    """先固定预算、扩训门槛、样本，再执行任何新预测。"""
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / "frozen_history.json", dict(files=inventory([*HISTORY, V4])))
    old_sources = [
        p
        for p in sorted((ROOT / "scripts/stage2").glob("*.py"))
        if not p.name.startswith("coupled_v5")
    ]
    write(
        OUT / "frozen_sources.json",
        dict(files=[dict(path=str(p), sha256=sha(p)) for p in old_sources]),
    )
    logbook = ROOT / "docs/logbook_2026-10-26.md"
    write(
        OUT / "logbook_prefix.json",
        dict(bytes=logbook.stat().st_size, sha256=sha(logbook)),
    )
    (OUT / "initial_git_status.txt").write_text(
        subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    )
    manifest = json.loads((V4 / "labels/manifest.json").read_text())
    assert manifest["counts"] == {"dev": 66, "train": 276}
    for frame in manifest["frames"]:
        assert frame["scene_name"] not in (
            "scene-0061",
            "scene-0553",
            "scene-0001",
            "scene-0002",
        )
        assert sha(frame["source_file"]) == frame["source_sha256"]
        assert sha(frame["cache_file"]) == frame["cache_sha256"]
    shutil.copy2(V4 / "labels/manifest.json", OUT / "map_manifest.json")
    protocol = dict(
        seed=SEED,
        branches=BRANCHES,
        data=dict(
            train_map_instructions=276,
            train_frames=40,
            train_scenes=5,
            dev_map_instructions=66,
            dev_frames=8,
            dev_scene="scene-0655",
            lineage=str(V4 / "labels/manifest.json"),
            lineage_sha256=sha(V4 / "labels/manifest.json"),
            source="unchanged real author BEV caches, real vector records and raw3Dannotation tokens",
            exposure="scene0655 is an already observed DEVELOPMENT scene, never called unseen. V3 trained all6 scenes; V4 Mask readout trained only5. No heldout0553/0001/0002 accessed.",
        ),
        model="native256->16 visual+5channel calibrated geometry, frozen3072Dtext FiLM,16->256 zero-init residual; relative per-channel residual <=0.05 inputRMS; same enhancedBEV enters actual native doubleHead",
        trainable="new adapter only; author backbone/doubleHead/BERT/V3/V4Mask all frozen",
        budget=dict(
            pilot_updates_per_branch=96,
            map_updates=48,
            object_updates=48,
            maximum_updates_per_branch=256,
            maximum_total_updates=768,
            batch_size=1,
            lr=0.0003,
            weight_decay=0.01,
            gradient_clip=5.0,
            no_new_architecture_after_v5=True,
        ),
        schedule="identical seeded map/object alternation per branch; object positive/empty prompts balanced when available; class balance deterministic car/pedestrian; raw GT identities fixed before training",
        loss=dict(
            global_control="global native detection +global 4class CE; zero text; no target-specific weights",
            targeted_no_language="zero text, identical target weights to language branch",
            targeted_language="full frozen text",
            map="native global detection + global4classCE +1*2m target-neighborhood CE +0.5*outside-quadrant CE +0.1*outside-quadrant KL to Original",
            objects="same-class/quadrant real car/pedestrian Hungarian positive cls/bbox weights3; all other positives/negativequeries1 +global map4classCE +0.1*outside-object-quadrant KL",
            mask="V4 readout FROZEN on enhancedBEV for diagnostic IoU only; no new Mask or Attention loss",
            gt_filter="exact author ObjectRangeFilter+ObjectNameFilter+limit_yaw for native detection loss; original AP GT protocol unchanged",
        ),
        metrics=dict(
            relevant_segmentation="pooled target-class IoU in fixed2m Euclidean target-record buffer intersected quadrant; independent unchanged full semanticGT",
            irrelevant_segmentation="foreground 3class meanIoU outside instruction quadrant; full confusion and TP/union saved",
            detection="noninterpolated AP@2m_xy all native top300 scores; sameclass and fixed Ego quadrant partitions; greedy unique GT; car/pedestrian macro, empty prompts retained; NOT nuScenes official AP",
            grounding="frozen V4 Head on each branch BEV; pooled instanceIoU at preregistered0.5, distractor/background FP; correct/wrong sameclass-direction tests",
            threshold=0.5,
        ),
        expansion_gate="Only if targeted_language exceeds BOTH matched controls by >=0.01pp on BOTH RelevantSegIoU and RelevantAP, exceeds Original on both, and IrrelevantSeg/AP decline <=0.1pp vsOriginal, continue ALL3 to step256 with same schedule/init/optimizer. Otherwise STOP after96. No lr/threshold/ROI/architecture retries. At256 same gate vscontrols; preserve terminal legal candidates, do not select by Mask.",
        final_policy=dict(
            strict_pretraining_unseen="BLOCKED: no verified author-pretraining-unseen scene",
            new_adaptation_final="only if expansion gate yields a useful model, preregister one traceable new scene after excluding0553/0001/0002 and all dev before single evaluation; otherwise NOT_RUN, no final set opened for failed candidates",
        ),
        closure="After bounded V5: stage3 performance/DriveLM-scope/checkpoint/reproduction closure and stage4 acceptance/Git whitelist. No V6/V7, no automatic commit/push.",
        source_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / "scripts/stage2").glob("coupled_v5*.py"))
        },
    )
    write(OUT / "protocol.json", protocol)
    print(
        "V5_PREREGISTERED",
        dict(pilot=96, branches=BRANCHES, maximum_updates=768),
        flush=True,
    )


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()

"""第二轮真实 GPU 对比学习：训练/开发选型、独立最终集一次评价。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.micro_training import (
    forward_payload, get_batch, parameter_stats, seed_all,
)
from scripts.stage2.multitask_training import aggregate_detection_losses
from scripts.stage2.run_step70_joint_integration import sha256
from scripts.stage2.runtime_memory import release_memory
from scripts.stage2.spatial_annotations import load_target, project_attention
from scripts.stage2.spatial_experiment import INITIAL, initialize, write_json
from scripts.stage2.spatial_v2_model import (
    ContrastFusion, DirectionReadout, spatial_pair_loss,
)

V1 = ROOT / "outputs/stage2_spatial_20261026/train_continue/fusion_step020.pt"


def swap_direction(text):
    return text.replace("left", "RIGHT").replace("right", "left").replace("RIGHT", "right")


def configure_v2(pipeline):
    initial = torch.load(INITIAL, map_location="cpu", weights_only=False)["state_dict"]
    pipeline.encoder = DirectionReadout(pipeline.encoder).cuda().eval()
    seed_all(20261027)
    pipeline.fusion = ContrastFusion(initial).cuda().eval()
    return pipeline


def cache(args, manifest):
    """只缓存训练/开发的真实冻结 BEV；最终场景不进入诊断或训练缓存。"""
    model, wrapped, pipeline, dataset = initialize("train")
    frame_records = []
    for row in manifest["frames"]:
        if row["split"] == "final":
            continue
        captured = {}
        original = model.pts_bbox_head.transformer.get_bev_features

        def capture_bev(*a, **kw):
            value = original(*a, **kw)
            captured["bev"] = BEVGrid().to_map(value).detach()
            return value

        model.pts_bbox_head.transformer.get_bev_features = capture_bev
        try:
            data = get_batch(dataset, row["index"], 20261027 + row["index"])
            from scripts.stage2.joint_runtime import _first_tensor
            actual = _first_tensor(data["semantic_indices"]).reshape(200, 400).numpy()
            with np.load(args.annotations / row["frame_file"]) as label:
                np.testing.assert_array_equal(actual, label["gt"])
            with torch.no_grad():
                wrapped(return_loss=True, **forward_payload(data))
            filename = f"bev_{row['index']:03d}.npz"
            np.savez_compressed(args.output_dir / filename,
                                bev=captured["bev"].cpu().numpy().astype(np.float16))
            frame_records.append({**row, "cache_file": filename, "sha256": sha256(args.output_dir / filename)})
        finally:
            del model.pts_bbox_head.transformer.get_bev_features
        del data, captured
        release_memory()
        print("REAL_BEV_CACHE", row["split"], row["index"], flush=True)
    readout = DirectionReadout(pipeline.encoder).cuda().eval()
    texts = ["Find lane markings on the front left", "Find lane markings on the front right"]
    vectors = readout(texts, "cuda")
    cosine = {
        "pooled_left_right_cosine": float(F.cosine_similarity(vectors[:1, :768], vectors[1:, :768])),
        "lr_token_left_right_cosine": float(F.cosine_similarity(vectors[:1, 1536:], vectors[1:, 1536:])),
    }
    train = [r for r in manifest["annotations"] if r["split"] == "train"]
    probabilities = [20 * r["target_pixels"] / (20 * r["target_pixels"] + 80000 - r["target_pixels"]) for r in train]
    diagnostics = {**cosine, "uniform_bce_optimum_with_pos_cap20": {
        "min": min(probabilities), "mean": float(np.mean(probabilities)), "max": max(probabilities),
        "below_threshold_half_count": sum(p < .5 for p in probabilities), "count": len(probabilities)},
        "v1_alpha": .0041174935176968575,
        "v1_gradient_evidence": "detection/segmentation reached alpha; attention projection parameters changed; no disconnected gradient found",
        "confirmed_structure": ["V1 did not provide explicit metric coordinate features", "V1 used only positive instruction samples, no paired wrong-direction contrast", "V1 masked mean dilutes changed direction token"],
        "coordinates": "same verified calibrated projection as V1; no GT edits or flipped target repair",
        "scope": "train/dev only, no final prediction metrics"}
    write_json(args.output_dir / "cache.json", {"frames": frame_records})
    write_json(args.output_dir / "root_diagnosis.json", diagnostics)
    print("ROOT_DIAGNOSIS", json.dumps(diagnostics), flush=True)


def make_protocol(args, manifest):
    train = [r for r in manifest["annotations"] if r["split"] == "train"]
    dev = [r for r in manifest["annotations"] if r["split"] == "dev"]
    final_frames = [r for r in manifest["frames"] if r["split"] == "final"]
    chosen = [final_frames[n]["index"] for n in np.linspace(0, len(final_frames) - 1, 6, dtype=int)]
    final = [r for r in manifest["annotations"] if r["split"] == "final" and r["index"] in chosen]
    assert train and dev and final
    assert not ({r["scene_token"] for r in train} & {r["scene_token"] for r in dev + final})
    protocol = {"seed": 20261027, "annotations": str(args.annotations),
        "annotation_sha256": sha256(args.annotations / "manifest.json"),
        "cache": str(args.cache), "train": train, "dev": dev, "final": final,
        "final_frame_indices": chosen, "warmup_steps": 200, "joint_steps": 40,
        "warmup_lr": .001, "joint_lr": .0003, "weight_decay": .01,
        "contrast_margin": 1., "joint_attention_weight": .2,
        "candidate_steps": [20, 100, 200, 220, 240],
        "selection": "among joint checkpoints220/240 maximize pooled dev Attention IoU, then target logit gap; warmup20/100/200 diagnostic only; final evaluation once after sealing",
        "init": "same original untrained fusion_initial as V1, no V1-trained weights for V2; V1 used only as frozen comparison",
        "final_isolation": "scene-0553 excluded from both round1 optimizer and all round2 train/dev; only6 predefined linspace frames for final comparison",
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__), ROOT / "scripts/stage2/spatial_v2_model.py",
            ROOT / "scripts/stage2/spatial_v2_data.py")}}
    write_json(args.output_dir / "protocol.json", protocol)
    print("FROZEN_V2", len(train), len(dev), len(final), chosen, flush=True)


def embeddings(pipeline, rows):
    texts = sorted({r["text"] for r in rows} | {swap_direction(r["text"]) for r in rows})
    result = {}
    for offset in range(0, len(texts), 16):
        batch = texts[offset:offset + 16]
        values = pipeline.encoder(batch, "cuda")
        result.update({text: value[None] for text, value in zip(batch, values)})
    return result


def dev_score(pipeline, rows, features, text_features, annotations):
    """只在从训练侧隔离的开发场景计算候选选择分数。"""
    intersection, union, gaps, win = 0, 0, [], 0
    pipeline.eval()
    with torch.inference_mode():
        for row in rows:
            gt, target, grid = load_target(row, annotations, "cuda")
            pipeline.fusion.set_calibration(row)
            bev = features[row["index"]]
            a = pipeline.fusion.forward_with_details(bev, text_features[row["text"]])["attention"]
            b = pipeline.fusion.forward_with_details(bev, text_features[swap_direction(row["text"])])["attention"]
            pa, pb = project_attention(a, grid)[0, 0], project_attention(b, grid)[0, 0]
            truth = target.instance_mask
            intersection += int(((pa >= .5) & truth).sum())
            union += int(((pa >= .5) | truth).sum())
            gap = float((torch.logit(pa.clamp(1e-5, 1 - 1e-5))[truth]
                         - torch.logit(pb.clamp(1e-5, 1 - 1e-5))[truth]).mean())
            gaps.append(gap); win += int(gap > 0)
    pipeline.train(); pipeline.encoder.eval()
    return {"attention_iou": intersection / union, "mean_logit_gap": float(np.mean(gaps)),
            "correct_text_wins": win, "count": len(rows), "tp": intersection, "union": union}


def save_checkpoint(path, pipeline, optimizer, step, protocol_path, stage):
    with path.open("xb") as stream:
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in pipeline.fusion.state_dict().items()},
                    "optimizer": optimizer.state_dict(), "step": step,
                    "stage": stage, "protocol_sha256": sha256(protocol_path)}, stream)
    return sha256(path)


def train(args, protocol):
    model, wrapped, pipeline, dataset = initialize("train")
    pipeline = configure_v2(pipeline)
    frames = json.loads((args.cache / "cache.json").read_text())["frames"]
    features = {}
    for row in frames:
        with np.load(args.cache / row["cache_file"]) as cached:
            features[row["index"]] = torch.tensor(cached["bev"], device="cuda", dtype=torch.float32)
    rows, dev = protocol["train"], protocol["dev"]
    text_features = embeddings(pipeline, rows + dev)
    optimizer = torch.optim.AdamW(pipeline.fusion.parameters(), lr=.001, weight_decay=.01)
    candidates = []
    rng = np.random.RandomState(protocol["seed"])
    schedule = rng.randint(0, len(rows), 240).tolist()
    torch.cuda.reset_peak_memory_stats()
    pipeline.train(); pipeline.encoder.eval()
    with (args.output_dir / "training.jsonl").open("x") as log:
        for number in range(1, 241):
            row = rows[schedule[number - 1]]
            gt, target, grid = load_target(row, args.annotations, "cuda")
            pipeline.fusion.set_calibration(row)
            optimizer.zero_grad(set_to_none=True)
            joint = number > 200
            if joint:
                for group in optimizer.param_groups:
                    group["lr"] = .0003
                data = get_batch(dataset, row["index"], 20261027 + row["index"])
                captured = {}

                def capture(_module, _inputs, value):
                    captured.update(value)

                handle = model.pts_bbox_head.register_forward_hook(capture)
                try:
                    with pipeline.inject_shared_bev(model.pts_bbox_head.transformer,
                                                   [row["text"]], strict=False,
                                                   capture_details=True):
                        author_losses = wrapped(return_loss=True, **forward_payload(data))
                        bev = pipeline.fusion.last_bev
                        detection = aggregate_detection_losses({k: v for k, v in author_losses.items()
                            if (k.startswith("loss") or ".loss" in k) and "seg" not in k.lower()})
                        segmentation = F.cross_entropy(captured["seg_preds"], gt[None])
                finally:
                    handle.remove()
            else:
                bev = features[row["index"]]
                detection = gt.new_tensor(0., dtype=torch.float32)
                segmentation = detection
            attention, parts = spatial_pair_loss(
                pipeline.fusion, bev, text_features[row["text"]],
                text_features[swap_direction(row["text"])], grid, gt,
                target.instance_mask, row["target_class"])
            total = detection + segmentation + (.2 if joint else 1.) * attention
            task_gradients = None
            if number == 201:
                parameters = list(pipeline.fusion.parameters())
                task_gradients = {}
                for key, term in (("detection", detection), ("segmentation", segmentation), ("attention", attention)):
                    gradients = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
                    task_gradients[key] = float(sum(g.detach().square().sum() for g in gradients if g is not None).sqrt())
            total.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(pipeline.fusion.parameters(), 5., error_if_nonfinite=True))
            before = {k: p.detach().cpu().clone() for k, p in pipeline.fusion.named_parameters()}
            optimizer.step()
            delta = parameter_stats(pipeline.fusion, before)
            assert any(item["changed_elements"] for item in delta.values())
            record = {"step": number, "stage": "real_joint_three_loss" if joint else "cached_real_BEV_attention_contrast",
                "instruction_id": row["instruction_id"], "frame_index": row["index"],
                "total_loss": float(total.detach()), "detection_loss": float(detection.detach()) if joint else None,
                "segmentation_loss": float(segmentation.detach()) if joint else None,
                "attention_loss": float(attention.detach()),
                "balanced_loss": float(parts["balanced"].detach()), "contrast_loss": float(parts["contrast"].detach()),
                "correct_minus_wrong_logit": float(parts["gap"].detach()),
                "gradient_l2": norm, "parameter_updates": delta, "optimizer_step_executed": True,
                "task_gradients": task_gradients, "alpha": float(pipeline.fusion.alpha.detach()),
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
            log.write(json.dumps(record) + "\n"); log.flush()
            if number in (20, 100, 200, 220, 240):
                checkpoint = args.output_dir / f"fusion_step{number:03d}.pt"
                digest = save_checkpoint(checkpoint, pipeline, optimizer, number, args.protocol, record["stage"])
                score = dev_score(pipeline, dev, features, text_features, args.annotations)
                candidates.append({"step": number, "checkpoint": str(checkpoint), "sha256": digest, "dev": score})
                print("DEV_ONLY_CANDIDATE", number, json.dumps(score), flush=True)
            if joint:
                del data, captured, author_losses, bev
            del total, attention, parts, before, gt, target, grid
            optimizer.zero_grad(set_to_none=True)
            if number % 10 == 0:
                release_memory()
                print("V2_OPTIMIZER_STEP", number, record["stage"], record["attention_loss"], flush=True)
    best = max(candidates, key=lambda c: (c["dev"]["attention_iou"], c["dev"]["mean_logit_gap"]))
    # 选择限于已有真实三任务更新的两个预注册候选，不查看最终集。
    selected = max((c for c in candidates if c["step"] in (220, 240)),
                   key=lambda c: (c["dev"]["attention_iou"], c["dev"]["mean_logit_gap"]))
    write_json(args.output_dir / "selection.json", {"candidates": candidates,
        "best_attention_candidate": best, "selected": selected,
        "selection_rule": "joint220/240 only, dev pooled attention IoU then logit gap; no final metrics used",
        "optimizer_steps": 240, "cached_attention_only_updates": 200, "real_joint_updates": 40,
        "final_evaluation_opened": False})


def final_dataset():
    """Mini Train中的独立场景使用作者测试流水线，不使用训练增强。"""
    model, wrapped, pipeline, _ = initialize("evaluate")
    from mmcv import Config
    from mmdet3d.datasets import build_dataset
    from scripts.stage2.low_memory_dataset import bounded_metadata
    from scripts.stage2.spatial_experiment import AUTHOR
    cfg = Config.fromfile(str(AUTHOR / "projects/configs/bevformer/bevformer_small_seg_det.py"))
    dataset_cfg = copy.deepcopy(cfg.data.test)
    dataset_cfg.ann_file = str(Path(cfg.data.train.ann_file).resolve())
    dataset_cfg.test_mode = True
    dataset_cfg.pop("samples_per_gpu", None)
    import mmcv
    ann = mmcv.load(dataset_cfg.ann_file)
    tokens = {r["token"] for r in ann["infos"]}
    del ann
    with bounded_metadata(tokens):
        dataset = build_dataset(dataset_cfg)
    release_memory()
    return model, wrapped, pipeline, dataset


def evaluate(args, protocol):
    selection = json.loads(args.selection.read_text())
    chosen = selection["selected"]
    assert sha256(Path(chosen["checkpoint"])) == chosen["sha256"]
    marker = args.selection.parent / "final_evaluation_opened.json"
    write_json(marker, {"selected_checkpoint": chosen, "final_indices": protocol["final_frame_indices"],
                        "note": "selection sealed before reading final prediction metrics"})
    model, wrapped, baseline, dataset = final_dataset()
    from scripts.stage2.pipeline import Stage2Pipeline
    pipeline = Stage2Pipeline(baseline.encoder, baseline.grid).cuda()
    pipeline = configure_v2(pipeline)
    pipeline.fusion.load_state_dict(torch.load(chosen["checkpoint"], map_location="cpu", weights_only=False)["state_dict"])
    pipeline.eval()
    baseline.fusion.load_state_dict(torch.load(V1, map_location="cpu", weights_only=False)["state_dict"])
    baseline.eval()
    predictions = []
    for number, row in enumerate(protocol["final"]):
        data = get_batch(dataset, row["index"], 20261027 + row["index"])
        gt, target, grid = load_target(row, args.annotations, "cuda")
        from scripts.stage2.joint_runtime import _first_tensor
        assert torch.equal(_first_tensor(data["semantic_indices"]).reshape(200, 400).long().cuda(), gt)
        with np.load(args.annotations / row["mask_file"]) as label:
            payload = {k: label[k] for k in ("gt", "target", "region", "other_instances", "selected_full_instance")}
        payload["token"] = row["sample_token"]
        pipeline.fusion.set_calibration(row)
        branches = (("original", None, row["text"]), ("historical", baseline, row["text"]),
                    ("spatial", pipeline, row["text"]), ("incorrect", pipeline, swap_direction(row["text"])),
                    ("shuffled", pipeline, " ".join(reversed(row["text"].split()))))
        for branch, system, text in branches:
            captured = {}
            def capture(_module, _inputs, outputs):
                captured["seg"] = outputs["seg_preds"].detach()
            handle = model.pts_bbox_head.register_forward_hook(capture)
            model.prev_frame_info = {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}
            try:
                with torch.inference_mode():
                    if system is None:
                        result = wrapped(return_loss=False, rescale=True, **forward_payload(data, inference=True))[0]
                    else:
                        with system.inject_shared_bev(model.pts_bbox_head.transformer, [text], strict=False,
                                                       capture_details=True) as state:
                            result = wrapped(return_loss=False, rescale=True, **forward_payload(data, inference=True))[0]
                            attention = state["details"]["attention"]
                            payload[branch + "_attention_native"] = attention[0, 0].cpu().numpy()
                            payload[branch + "_attention"] = project_attention(attention, grid)[0, 0].cpu().numpy()
                    payload[branch] = captured["seg"][0].argmax(0).cpu().numpy().astype(np.uint8)
                    detection = result["pts_bbox"]
                    payload[branch + "_boxes"] = detection["boxes_3d"].tensor.cpu().numpy()
                    payload[branch + "_scores"] = detection["scores_3d"].cpu().numpy()
                    payload[branch + "_labels"] = detection["labels_3d"].cpu().numpy()
            finally:
                handle.remove()
        annotation = dataset.get_ann_info(row["index"])
        payload["gt_boxes"] = annotation["gt_bboxes_3d"].tensor.numpy()
        payload["gt_labels"] = np.asarray(annotation["gt_labels_3d"])
        filename = f"instruction_{number:03d}.npz"
        np.savez_compressed(args.output_dir / filename, **payload)
        predictions.append({**row, "prediction_file": filename})
        del data, gt, target, grid, payload, captured, annotation
        release_memory()
        print("FINAL_ONCE_AB", number + 1, len(protocol["final"]), row["text"], flush=True)
    write_json(args.output_dir / "predictions.json", {"annotations": predictions,
        "historical_sha256": sha256(V1), "spatial_sha256": chosen["sha256"],
        "author_sha256": json.loads((ROOT / "outputs/stage2_spatial_20261026/preflight.json").read_text())["assets"]["/mnt/g/nus/bevformer_small_joint_author.pth"]["sha256"],
        "note": "historical branch is round1 spatial model; spatial branch is round2, same new final set",
        "selected_step": chosen["step"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cache", "protocol", "train", "final"), required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("annotations", "cache", "protocol", "selection", "output_dir"):
        value = getattr(args, name)
        if value:
            setattr(args, name, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.annotations / "manifest.json").read_text())
    if args.phase == "cache":
        cache(args, manifest)
    elif args.phase == "protocol":
        make_protocol(args, manifest)
    else:
        protocol = json.loads(args.protocol.read_text())
        assert sha256(args.annotations / "manifest.json") == protocol["annotation_sha256"]
        for name, digest in protocol["source_sha256"].items():
            assert sha256(ROOT / name) == digest, name
        if args.phase == "train":
            train(args, protocol)
        else:
            evaluate(args, protocol)


if __name__ == "__main__":
    main()

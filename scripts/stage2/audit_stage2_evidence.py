#!/usr/bin/env python3
"""只读复算 20 步/6 帧证据，或探测真实预训练语言响应；不训练。"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def reference_detection(boxes, scores, labels, gt_boxes, gt_labels):
    """独立按置信度顺序匹配同类 XY 中心；不是 IoU、AP 或 NDS。"""
    valid = (gt_labels >= 0) & (np.abs(gt_boxes[:, :2]) <= 51.2).all(1)
    truth, classes = gt_boxes[valid], gt_labels[valid]
    selected = np.flatnonzero((scores >= 0.1) & (np.abs(boxes[:, :2]) <= 51.2).all(1))
    unused = set(range(len(truth)))
    distances = []
    for index in selected[np.argsort(-scores[selected], kind="stable")]:
        candidates = sorted(i for i in unused if classes[i] == labels[index])
        if candidates:
            separation = np.linalg.norm(
                truth[candidates, :2] - boxes[index, :2], axis=1
            )
            best = int(separation.argmin())
            if separation[best] <= 2:
                unused.remove(candidates[best])
                distances.append(float(separation[best]))
    return dict(
        tp=len(distances),
        fp=len(selected) - len(distances),
        fn=len(unused),
        matched_xy_error_sum_m=sum(distances),
    )


def audit_artifacts(root=ROOT):
    from nuscenes.utils.splits import create_splits_scenes
    from scripts.stage2.semantic_instruction import attention_target_class

    first = root / "outputs/stage2_oom_fix_20261022_step1"
    continuation = root / "outputs/stage2_oom_fix_20261022_continue/continue"
    validation = root / "outputs/stage2_oom_fix_20261022_validation/validate"
    protocol_path = first / "protocol/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    rows = [
        json.loads(line)
        for path in (first / "train/training.jsonl", continuation / "training.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert [r["step"] for r in rows] == list(range(1, 21))
    assert all(r["optimizer_step_executed"] for r in rows)
    assert {r["token"] for r in protocol["train"]}.isdisjoint(
        r["token"] for r in protocol["validation"]
    )
    for row, expected in zip(rows, protocol["train"]):
        assert all(
            row[k] == expected[k] for k in ("index", "token", "text", "target_class")
        )
        assert row["target_class"] == attention_target_class(row["text"])
        assert row["details"]["hook_calls"] == row["details"]["head_calls"] == 1
        assert all(p["finite"] for p in row["parameters"].values())
        l = row["losses"]
        assert np.isclose(
            l["total_loss"],
            l["detection_loss"] + l["segmentation_loss"] + 0.1 * l["attention_loss"],
            atol=1e-6,
        )
    # 原协议与新 TV 实现区分：旧实验没有该项。
    assert protocol["loss"].get("attention_tv_weight", 0) == 0
    split_counts = {}
    # 元数据根目录由作者配置绑定；相机 images 和元数据可能位于不同挂载点。
    author_data = (
        Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg/data/nuscenes"
    )
    scenes = {
        r["token"]: r["name"]
        for r in json.loads((author_data / "v1.0-trainval/scene.json").read_text())
    }
    for key, ann_path, split, size, count in (
        ("train", protocol["train_ann"], "mini_train", 323, 20),
        ("validation", protocol["val_ann"], "mini_val", 81, 6),
    ):
        with Path(ann_path).open("rb") as stream:
            infos = sorted(pickle.load(stream)["infos"], key=lambda r: r["timestamp"])
        assert len(infos) == size
        assert all(
            scenes[r["scene_token"]] in create_splits_scenes()[split] for r in infos
        )
        assert [r["index"] for r in protocol[key]] == np.linspace(
            0, size - 1, count, dtype=int
        ).tolist()
        assert all(infos[r["index"]]["token"] == r["token"] for r in protocol[key])
        split_counts[key] = {
            "all_infos": size,
            "fixed_frames": count,
            "official_split": split,
        }
    checkpoints = []
    for step in range(1, 21):
        path = (
            first / "train" if step == 1 else continuation
        ) / f"fusion_step{step:03d}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["steps"] == step and saved["protocol_sha256"] == digest(
            protocol_path
        )
        assert {int(s["step"]) for s in saved["optimizer"]["state"].values()} == {step}
        assert all(torch.isfinite(v).all() for v in saved["state_dict"].values())
        checkpoints.append({"step": step, "sha256": digest(path)})
    assert (
        checkpoints[-1]["sha256"]
        == json.loads((continuation / "result.json").read_text())["new_weights_sha256"]
    )
    ab = json.loads((validation / "ab_results.json").read_text())
    assert (
        json.loads((validation / "result.json").read_text())["status"]
        == "PASS_FIXED_AB"
    )
    assert len(ab["frames"]) == 6
    matrices = {m: np.zeros((4, 4), dtype=np.int64) for m in ab["metrics"]}
    counts = {m: dict(tp=0, fp=0, fn=0, matched_xy_error_sum_m=0.0) for m in matrices}
    controls = []
    for frame, expected in zip(ab["frames"], protocol["validation"]):
        assert all(frame[k] == expected[k] for k in ("index", "token", "text"))
        path = validation / frame["artifact"]
        assert digest(path) == frame["sha256"]
        with np.load(path, allow_pickle=False) as data:
            gt = data["gt"]
            assert gt.shape == (200, 400) and str(data["token"]) == frame["token"]
            assert (
                hashlib.sha256(gt.astype(np.int64).tobytes()).hexdigest()
                == frame["gt_sha256"]
            )
            assert np.array_equal(data["original_logits"], data["untrained_logits"])
            assert all(v == 0 for v in frame["zero_init_max_abs"].values())
            for mode in matrices:
                assert np.array_equal(data[mode], data[mode + "_logits"].argmax(0))
                for truth in range(4):
                    for pred in range(4):
                        matrices[mode][truth, pred] += int(
                            ((gt == truth) & (data[mode] == pred)).sum()
                        )
                c = reference_detection(
                    data[mode + "_boxes"],
                    data[mode + "_scores"],
                    data[mode + "_labels"],
                    data["gt_boxes"],
                    data["gt_labels"],
                )
                for k in counts[mode]:
                    counts[mode][k] += c[k]
            controls.append(
                {
                    "index": frame["index"],
                    "attention_mae_normal_vs_shuffled": float(
                        np.abs(
                            data["trained_attention"] - data["shuffled_attention"]
                        ).mean()
                    ),
                    "logit_mae_normal_vs_shuffled": float(
                        np.abs(data["trained_logits"] - data["shuffled_logits"]).mean()
                    ),
                    "changed_pixels_normal_vs_shuffled": int(
                        (data["trained"] != data["shuffled"]).sum()
                    ),
                }
            )
    metrics = {}
    for mode, matrix in matrices.items():
        diagonal = matrix.diagonal().astype(float)
        union = matrix.sum(0) + matrix.sum(1) - diagonal
        iou = np.divide(diagonal, union, out=np.full(4, np.nan), where=union > 0)
        foreground = float(np.nanmean(iou[1:]))
        expected = ab["metrics"][mode]
        assert (
            matrix.tolist()
            == expected["segmentation"]["confusion_matrix_gt_rows_pred_columns"]
        )
        assert np.isclose(
            foreground, expected["segmentation"]["foreground_miou"], rtol=0, atol=1e-14
        )
        assert all(
            np.isclose(counts[mode][k], expected["detection"][k]) for k in counts[mode]
        )
        assert np.isclose(
            matrix[0, 1:].sum() / matrix[0].sum(),
            expected["segmentation"]["background_false_positive_rate"],
        )
        metrics[mode] = {
            "foreground_miou": foreground,
            "class_iou": iou.tolist(),
            "detection": counts[mode],
        }
    protected = json.loads((first / "protocol/protected_before.json").read_text())
    assert all(digest(Path(p)) == h for p, h in protected.items())
    assert all(
        json.loads((p / "integrity.json").read_text())[k]
        for p in (first / "train", continuation, validation)
        for k in ("protected_files_unchanged", "frozen_states_unchanged")
    )
    return {
        "status": "PASS_RAW_ARTIFACT_RECOMPUTATION",
        "optimizer_steps": 20,
        "checkpoints": checkpoints,
        "protocol_sha256": digest(protocol_path),
        "official_splits": split_counts,
        "metrics": metrics,
        "text_controls": controls,
        "protected_files": len(protected),
        "accuracy_improvement": "NOT_VERIFIED",
        "semantic_localization": "NOT_VERIFIED",
        "physical_alignment": "NOT_VERIFIED",
        "historical_loss": "det + seg + 0.1*(BCE+0.25*Dice); TV absent",
    }


def probe_language():
    from safetensors import safe_open
    from scripts.stage2.pipeline import Stage2Pipeline
    from scripts.stage2.run_step56_smoke import load_bev_feature
    from scripts.stage2.integration_smoke import load_fusion_weights

    torch.manual_seed(20261021)
    model_dir = ROOT / "models/distilbert-base-uncased"
    pipeline = Stage2Pipeline.from_local_model(model_dir).eval()
    config = json.loads((ROOT / "stage2/tokenizer_config.json").read_text())
    assert digest(model_dir / "model.safetensors") == config["expected_model_sha256"]
    state = pipeline.encoder.model.state_dict()
    with safe_open(
        str(model_dir / "model.safetensors"), framework="pt", device="cpu"
    ) as stream:
        assert all(
            torch.equal(
                v, stream.get_tensor(k if k in stream.keys() else "distilbert." + k)
            )
            for k, v in state.items()
        )
    loaded = load_fusion_weights(
        pipeline,
        ROOT / "outputs/stage2_oom_fix_20261022_continue/continue/fusion_step020.pt",
    )
    bev, source = load_bev_feature(
        ROOT / "outputs/bev_features/bev_feature.npy", "real", torch.device("cpu")
    )
    texts = [
        "Find lane markings",
        "Find road boundaries",
        "Find lane markings ahead",
        "Find lane markings behind",
        "Find lane markings on the left",
        "Find lane markings on the right",
        "Find lane markings and road boundaries",
        "Find lane markings but ignore road boundaries",
        "markings lane Find",
    ]
    outputs = []
    with torch.inference_mode():
        for n, text in enumerate(texts):
            result = pipeline(bev, [text], strict=n != len(texts) - 1)
            tokens = pipeline.encoder.tokenize([text], "cpu")["input_ids"][0].tolist()
            outputs.append(
                {
                    "text": text,
                    "tokens": tokens,
                    "instruction": result["instructions"][0],
                    "embedding": result["text_embedding"][0].numpy().copy(),
                    "attention": result["attention"][0].numpy().copy(),
                }
            )
            del result
    pairs = {}
    for label, a, b in (
        ("target_word", 0, 1),
        ("front_rear", 2, 3),
        ("left_right", 4, 5),
        ("negation", 6, 7),
        ("word_shuffle", 0, 8),
    ):
        one, two = outputs[a], outputs[b]
        pairs[label] = {
            "texts": [one["text"], two["text"]],
            "tokens_differ": one["tokens"] != two["tokens"],
            "embedding_mae": float(np.abs(one["embedding"] - two["embedding"]).mean()),
            "attention_mae": float(np.abs(one["attention"] - two["attention"]).mean()),
            "attention_max_abs": float(
                np.abs(one["attention"] - two["attention"]).max()
            ),
        }
        assert (
            pairs[label]["tokens_differ"]
            and pairs[label]["embedding_mae"] > 0
            and pairs[label]["attention_mae"] > 0
        )
    return {
        "status": "PASS_REAL_PRETRAINED_LANGUAGE_RESPONSE_ONLY",
        "model_sha256": digest(model_dir / "model.safetensors"),
        "all_encoder_tensors_match_safetensors": True,
        "loaded_fusion": loaded,
        "bev_source": source,
        "pairs": pairs,
        "attention_spatial_std": float(outputs[0]["attention"].std()),
        "parser_role": "validation and metadata only; raw normalized text is encoded",
        "semantic_localization": "NOT_VERIFIED",
        "optimizer_steps": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("artifacts", "language"), default="artifacts"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("拒绝覆盖旧审计结果")
    result = audit_artifacts() if args.mode == "artifacts" else probe_language()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(result["status"], "->", args.output, flush=True)


if __name__ == "__main__":
    main()

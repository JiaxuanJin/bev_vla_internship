"""16步pilot只读原因分析：仅重放训练指令，不优化、不扫描旧开发阈值。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.stage2.language_encoder import DistilBERTLanguageEncoder
from scripts.stage2.spatial_v3_model import InstanceReadout
from scripts.stage3.lg_bev_common import ROOT, SEED, read, sha, write
from scripts.stage3.lg_bev_data import load_supervision
from scripts.stage3.lg_bev_formal_inventory import OUT
from scripts.stage3.lg_bev_prototype import LanguageGroundedBEVAdapter
from scripts.stage3.lg_bev_supervision_run import SPACE_OUT
from scripts.stage3.lg_bev_supervision_space import AttentionSupervisionSpace
from scripts.stage3.lg_bev_train import frame_input


def native_target_mass(row: dict, target: torch.Tensor) -> torch.Tensor:
    """监督投影的adjoint给出目标在原生网格的权重，不拟合位移或阈值。"""
    if row["task"] == "object":
        return target.float()
    source = torch.zeros(1, 1, 150, 150, requires_grad=True)
    projected = AttentionSupervisionSpace.from_calibration("map", row).project(source)
    return torch.autograd.grad((projected * target).sum(), source)[0].detach()


def relative_rms(
    value: torch.Tensor, reference: torch.Tensor, mass: torch.Tensor
) -> float | None:
    """局部能量比与全图能量比分开记录，防止把稀疏目标的全局小残差误判为零。"""
    if mass.sum() == 0:
        return None
    return float(
        (
            (value.square() * mass).sum()
            / (reference.square() * mass).sum().clamp_min(1e-20)
        ).sqrt()
    )


def analyze(out: Path) -> None:
    """冻结文本编码器和LG-BEV，在历史16条训练指令上重放终点。"""
    protocol = read(SPACE_OUT / "protocol.json")
    training = read(SPACE_OUT / "pilot_training.json")
    manifest = read(SPACE_OUT / "manifest.json")
    rows_by_id = {
        r["instruction_id"]: r for r in manifest["annotations"] if r["split"] == "train"
    }
    rows = [rows_by_id[item["instruction_id"]] for item in training["steps"]]
    assert all(r["split"] == "train" for r in rows)
    saved_path = Path(training["checkpoint"]["path"])
    assert sha(saved_path) == training["checkpoint"]["sha256"]
    saved = torch.load(saved_path, map_location="cpu", weights_only=False)
    assert saved["optimizer_updates"] == 16
    torch.manual_seed(SEED)
    adapter = LanguageGroundedBEVAdapter(**protocol["model_config"]).eval()
    initial = {
        name: value.detach().clone() for name, value in adapter.named_parameters()
    }
    adapter.load_state_dict(saved["state_dict"])
    encoder = InstanceReadout(
        DistilBERTLanguageEncoder(
            str(ROOT / "models/distilbert-base-uncased"), local_files_only=True
        )
    ).eval()
    encoder.requires_grad_(False)
    texts = sorted({r[k] for r in rows for k in ("text", "wrong_text")})
    encoded = {}
    for offset in range(0, len(texts), 16):
        batch = texts[offset : offset + 16]
        with torch.no_grad():
            values = encoder(batch, "cpu")
        encoded.update({text: value[None] for text, value in zip(batch, values)})
    records = []
    for row in rows:
        bev, position = frame_input(row, device="cpu")
        labels = load_supervision(row, SPACE_OUT, training=True)
        mass = native_target_mass(row, labels["target"])
        raw_up = []
        handle = adapter.up.register_forward_hook(
            lambda _m, _a, value: raw_up.append(value.detach())
        )
        with torch.no_grad():
            correct = adapter(bev, encoded[row["text"]], position)
        handle.remove()
        with torch.no_grad():
            wrong = adapter(bev, encoded[row["wrong_text"]], position)
        space = AttentionSupervisionSpace.from_calibration(row["task"], row)
        correct_logits = space.project(correct["attention_logits"])
        wrong_logits = space.project(wrong["attention_logits"])
        target, distractor = labels["target"].bool(), labels["distractor"].bool()
        rms = bev.square().mean((2, 3), keepdim=True).sqrt()
        ceiling = adapter.relative_bound * rms * correct["attention_mask"]
        embedding_cosine = float(
            F.cosine_similarity(encoded[row["text"]], encoded[row["wrong_text"]])
        )
        with torch.no_grad():
            qc, qw = adapter.language(encoded[row["text"]]), adapter.language(
                encoded[row["wrong_text"]]
            )
        records.append(
            dict(
                instruction_id=row["instruction_id"],
                task=row["task"],
                sample_token=row["sample_token"],
                target_present=row["target_present"],
                target_pixels=int(target.sum()),
                target_fraction=float(target.float().mean()),
                relative_global_residual_l2=float(
                    correct["residual"].norm() / bev.norm()
                ),
                relative_target_residual_l2=relative_rms(
                    correct["residual"], bev, mass
                ),
                attention_gated_ceiling_relative_l2=float(
                    ceiling.expand_as(bev).norm() / bev.norm()
                ),
                fraction_of_attention_gated_capacity=float(
                    correct["residual"].norm()
                    / ceiling.expand_as(bev).norm().clamp_min(1e-20)
                ),
                raw_up_rms=float(raw_up[0].square().mean().sqrt()),
                tanh_saturation_fraction=float((raw_up[0].abs() > 2).float().mean()),
                native_attention_mean=float(correct["attention_mask"].mean()),
                target_attention_probability=(
                    float(correct_logits.sigmoid()[target].mean())
                    if target.any()
                    else None
                ),
                target_correct_wrong_logit_gap=(
                    float((correct_logits - wrong_logits)[target].mean())
                    if target.any()
                    else None
                ),
                target_minus_distractor_logit_gap=(
                    float(
                        correct_logits[target].mean()
                        - correct_logits[distractor].mean()
                    )
                    if target.any() and distractor.any()
                    else None
                ),
                correct_wrong_mean_absolute_logit_delta=float(
                    (correct_logits - wrong_logits).abs().mean()
                ),
                frozen_text_cosine=embedding_cosine,
                query_cosine=float(F.cosine_similarity(qc, qw)),
            )
        )
    changes = {
        name: dict(
            delta_l2=float((value.detach() - initial[name]).norm()),
            initial_l2=float(initial[name].norm()),
            final_l2=float(value.detach().norm()),
        )
        for name, value in adapter.named_parameters()
    }
    norm_summary = {}
    for key in (
        "relative_global_residual_l2",
        "relative_target_residual_l2",
        "fraction_of_attention_gated_capacity",
        "raw_up_rms",
        "tanh_saturation_fraction",
        "frozen_text_cosine",
        "query_cosine",
        "target_correct_wrong_logit_gap",
    ):
        values = [r[key] for r in records if r[key] is not None]
        norm_summary[key] = dict(
            mean=float(np.mean(values)), min=float(min(values)), max=float(max(values))
        )
    write(
        out / "pilot_train_probe.json",
        dict(
            status="READ_ONLY_TRAIN_REPLAY_COMPLETE",
            records=records,
            aggregate=norm_summary,
            parameter_changes_vs_exact_seeded_init=changes,
            checkpoint_sha256=sha(saved_path),
            optimizer_updates=0,
            native_heads_executed=False,
            old_development_predictions_read_for_tuning=False,
            caveat="probe reuses16 training instructions at terminal weights; describes forward signals, not AP/IoU or a causal proof",
        ),
    )
    steps = training["steps"]
    task_gradient = {
        key: dict(
            mean_unweighted_l2=float(
                np.mean([r["task_gradient_l2"][key] for r in steps])
            ),
            mean_weighted_l2=float(
                np.mean(
                    [r["task_gradient_l2"][key] * protocol["loss"][key] for r in steps]
                )
            ),
        )
        for key in ("detection", "segmentation", "mask")
    }
    write(
        out / "pilot_rca.json",
        dict(
            optimizer_updates=16,
            unique_train_frames_seen=len({r["sample_token"] for r in rows}),
            unique_train_instructions_seen=len({r["instruction_id"] for r in rows}),
            instruction_epoch_fraction=16 / len(rows_by_id),
            task_counts=dict(Counter(r["task"] for r in rows)),
            positive_steps=sum(r["target_present"] for r in rows),
            empty_steps=sum(not r["target_present"] for r in rows),
            gradient_l2_by_component=task_gradient,
            clipping_steps=sum(
                r["gradient_l2"] > protocol["budget"]["clip"] for r in steps
            ),
            confirmed=[
                "only16 updates/12positive instructions, no mask warmup or complete data epoch",
                "zero-up makes first native task gradients reach up only; explicit mask trains language",
                "language conditions residual spatial gate; up feature content is visual+position",
                "current native det/global seg objectives are largely instruction-independent; no explicit correct/wrong ranking loss",
                "all native tasks remain differentiable; checkpoint/state updates recorded",
            ],
            hypotheses_not_proved=[
                "limited updates and sparse foreground may hinder grounding",
                "nearby frozen/query text embeddings may need explicit counterfactual training",
                "global residual norm alone cannot distinguish desirable sparse enhancement from ineffective local updates",
                "frozen native Heads may or may not be a later bottleneck; lowAttention IoU is insufficient evidence",
            ],
            no_new_threshold_architecture_or_optimizer_trial=True,
            recommendation="new scene-disjoint training protocol, mask warmup with identity baseline, then fixed joint budget; no automatic Head LoRA",
        ),
    )
    print("PILOT_TRAIN_PROBE_COMPLETE", norm_summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    analyze(parser.parse_args().out.resolve())

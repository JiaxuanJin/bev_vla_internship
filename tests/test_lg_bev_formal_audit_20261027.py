"""正式方案的数据隔离、预算和只读准入验收；不读取封存GT或启动训练。"""

import json

import pytest
import torch

from scripts.stage3.lg_bev_common import sha
from scripts.stage3.lg_bev_formal_inventory import select_frame_split
from scripts.stage3.lg_bev_formal_pilot_analysis import relative_rms
from scripts.stage3.lg_bev_formal_protocol import (
    assign_log_splits,
    budget,
    seal_contract,
    validate_split_records,
    verify,
)


def row(token: str, scene: str, log: str, split: str = "train") -> dict:
    """构造不包含GT的最小清单记录。"""
    return dict(
        sample_token=token,
        scene_token=scene,
        log_token=log,
        split=split,
        official_split="train",
        partition="eligible_official_train",
    )


def test_log_assignment_independent_of_order_and_frame_count():
    groups = [dict(log_token=f"a{i}", location="A") for i in range(10)]
    groups += [dict(log_token=f"b{i}", location="B") for i in range(3)]
    assignment = assign_log_splits(groups)
    assert assignment == assign_log_splits(list(reversed(groups)) + groups[:2])
    assert {assignment[f"b{i}"] for i in range(3)} == {
        "train",
        "development",
        "cold_final",
    }
    assert len(assignment) == 13


def test_log_location_conflict_rejected():
    with pytest.raises(ValueError, match="location"):
        assign_log_splits(
            [dict(log_token="x", location="A"), dict(log_token="x", location="B")]
        )


@pytest.mark.parametrize(
    "records,reason",
    [
        ([row("1", "s", "l"), row("1", "s", "l")], "重复"),
        ([row("1", "s", "l"), row("2", "s", "k", "cold_final")], "场景"),
        ([row("1", "s", "l"), row("2", "t", "l", "development")], "log"),
        ([row("1", "s", "l", "invalid")], "未知"),
        ([], "空"),
    ],
)
def test_leakage_rejected(records, reason):
    with pytest.raises(ValueError, match=reason):
        validate_split_records(records)


def test_official_val_and_historical_quarantine_rejected():
    for field, value in [
        ("official_split", "val"),
        ("partition", "historical_quarantine"),
    ]:
        record = row("1", "s", "l")
        record[field] = value
        with pytest.raises(ValueError, match="封存"):
            validate_split_records([record])
    assert select_frame_split("old", "train", {"old"}) == "historical_quarantine"
    assert select_frame_split("new", "val", set()) == "official_val_reserved"


def test_actual_empty_gt_frame_is_not_dropped_by_split_validator():
    record = {**row("1", "s", "l"), "annotation_record_present": False}
    assert validate_split_records([record])["unique_frames"] == 1


def test_budget_counts_real_frames_and_partial_accumulation():
    plan = budget(19123)
    assert plan["updates_per_epoch"] == 4781
    assert plan["total_optimizer_updates"] == 28686
    assert plan["native_joint_micro_forwards"] == 76492
    assert budget(5)["total_optimizer_updates"] == 12
    assert plan["mask_warmup_heads_executed"] is False
    with pytest.raises(ValueError):
        budget(0)


def test_local_residual_probe_separates_sparse_local_and_global_energy():
    reference = torch.ones(1, 2, 2, 2)
    value = torch.zeros_like(reference)
    value[:, :, 0, 0] = 0.1
    mass = torch.zeros(1, 1, 2, 2)
    mass[:, :, 0, 0] = 1
    assert relative_rms(value, reference, mass) == pytest.approx(0.1)
    assert float(value.norm() / reference.norm()) == pytest.approx(0.05)
    assert relative_rms(value, reference, torch.zeros_like(mass)) is None


def test_verify_never_authorizes_long_training(tmp_path, capsys):
    records = [row("1", "s", "l")]
    manifest = dict(
        frames=records,
        split_checks=validate_split_records(records),
        prior_history_quarantined_logs=["old"],
    )
    manifest_path = tmp_path / "formal_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    protocol = dict(
        split_manifest_sha256=sha(manifest_path),
        audit_lineage={},
        budget=budget(1),
    )
    (tmp_path / "formal_training_protocol.json").write_text(json.dumps(protocol))
    status = tmp_path / "preflight_status.json"
    status.write_text(json.dumps(dict(training_allowed=False)))
    verify(tmp_path)
    assert "VERIFIED_NO_TRAINING" in capsys.readouterr().out
    status.write_text(json.dumps(dict(training_allowed=True)))
    with pytest.raises(ValueError, match="不得自动授权"):
        verify(tmp_path)
    manifest_path.write_text("{}")
    with pytest.raises(ValueError, match="哈希"):
        verify(tmp_path)


def test_execution_contract_requires_lossless_baseline_and_keeps_training_off(tmp_path):
    records = [
        row("1", "s", "l"),
        row("2", "s2", "l2", "development"),
        row("3", "s3", "l3", "cold_final"),
    ]
    manifest_path = tmp_path / "formal_split_manifest.json"
    manifest_path.write_text(
        json.dumps(
            dict(
                frames=records,
                split_checks=validate_split_records(records),
                prior_history_quarantined_logs=[],
            )
        )
    )
    (tmp_path / "formal_training_protocol.json").write_text(
        json.dumps(
            dict(
                split_manifest_sha256=sha(manifest_path),
                audit_lineage={},
                budget=budget(1),
                split_counts={
                    split: dict(frames=1)
                    for split in ("train", "development", "cold_final")
                },
                implementation_readiness={"trainer": "NOT_IMPLEMENTED"},
            )
        )
    )
    (tmp_path / "preflight_status.json").write_text(
        json.dumps(dict(training_allowed=False))
    )
    seal_contract(tmp_path)
    contract = json.loads((tmp_path / "formal_execution_contract.json").read_text())
    assert not contract["training_started"]
    assert contract["optimizer_updates_this_audit"] == 0
    assert contract["strict_baseline"]["bytes_per_frame"] == 23040000
    assert contract["strict_baseline"]["cache_train_plus_dev_bytes"] == 46080000
    assert contract["remaining_gates"]["trainer"] == "NOT_IMPLEMENTED"
    with pytest.raises(FileExistsError):
        seal_contract(tmp_path)

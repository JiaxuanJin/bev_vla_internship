"""V3与历史协议分离、累计无限/会话有界、退出receipt及恢复状态回归。"""

import json

import pytest
import torch

from scripts.stage3.lg_bev_common import sha
from scripts.stage3.lg_bev_daily_state import (
    SessionClock,
    validate_policy,
    V3_POLICY,
    V3_RUN_ID,
    RUN_ID,
    POLICY,
)
from scripts.stage3.lg_bev_engineering_train import Trainer


def test_v3_is_separate_null_policy_with_historical_hash_binding():
    policy = validate_policy(V3_POLICY)
    assert policy["cumulative_hard_limit_seconds"] is None
    assert policy["session_active_seconds"] == 27000
    assert policy["stop_save_margin_seconds"] >= 600
    assert policy["parent_v2_sha256"] == sha(POLICY)
    assert V3_RUN_ID != RUN_ID
    assert validate_policy(POLICY)["total_active_seconds"] == 172800


@pytest.mark.parametrize("hours", [49, 73, 120])
def test_over48_and72_hours_are_warning_only(monkeypatch, hours):
    now = [0.0]
    monkeypatch.setattr(
        "scripts.stage3.lg_bev_daily_state.time.perf_counter", lambda: now[0]
    )
    clock = SessionClock(hours * 3600, total_seconds=None, margin_seconds=600)
    trainer = Trainer.__new__(Trainer)
    trainer.session, trainer.stop_requested = clock, False
    trainer.total_budget_seconds = None
    assert clock.reason() is None and not trainer.should_stop()
    assert 48 in clock.resource_warnings()
    if hours >= 72:
        assert 72 in clock.resource_warnings()
    now[0] = 26400
    assert clock.reason() == "DAILY_SESSION_LIMIT" and trainer.should_stop()


def test_v3_same_clock_counts_startup_and_spans_a_b(monkeypatch):
    now = [300.0]
    monkeypatch.setattr(
        "scripts.stage3.lg_bev_daily_state.time.perf_counter", lambda: now[0]
    )
    clock = SessionClock(
        73 * 3600, total_seconds=None, margin_seconds=600, started_at=0.0
    )
    assert clock.elapsed() == 300 and clock.total() == 73 * 3600 + 300
    stages = [Trainer.__new__(Trainer), Trainer.__new__(Trainer)]
    for trainer in stages:
        trainer.session, trainer.total_budget_seconds, trainer.stop_requested = (
            clock,
            None,
            False,
        )
    now[0] = 26399
    assert not any(t.should_stop() for t in stages)
    now[0] = 26400
    assert all(t.should_stop() for t in stages)


@pytest.mark.parametrize("previous", [float("inf"), float("nan"), -1])
def test_null_limit_does_not_allow_nonfinite_or_negative_cumulative(previous):
    with pytest.raises(ValueError):
        SessionClock(previous, total_seconds=None)


@pytest.mark.parametrize("bad_limit", [172800, 10**30, float("inf")])
def test_v3_rejects_silent_big_number_instead_of_null(tmp_path, bad_limit):
    value = json.loads(V3_POLICY.read_text())
    value["cumulative_hard_limit_seconds"] = bad_limit
    path = tmp_path / "bad_policy.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="V3"):
        validate_policy(path)


def stopped_fixture(tmp_path):
    run = tmp_path
    checkpoint = run / "checkpoints/a.pt"
    checkpoint.parent.mkdir()
    execution = run / "execution.json"
    execution.write_text("{}")
    torch.save(dict(protocol_sha256=sha(execution)), checkpoint)
    receipt = checkpoint.with_name("a.pt.receipt.pt")
    torch.save(dict(checkpoint_sha256=sha(checkpoint), wall_seconds=180000.0), receipt)
    record = dict(
        event="checkpoint",
        path=str(checkpoint),
        sha256=sha(checkpoint),
        receipt_sha256=sha(receipt),
        execution_sha256=sha(execution),
    )
    (run / "control.jsonl").write_text(json.dumps(record) + "\n")
    status = dict(
        status="STOPPED",
        pid=999999999,
        protocol_version="V3",
        checkpoint_sha256=record["sha256"],
        safe_to_shutdown=False,
        can_resume=True,
    )
    (run / "status.json").write_text(json.dumps(status))
    return run, status, execution


def exit_receipt(run, status, execution):
    path = run / "exit.json"
    path.write_text(
        json.dumps(
            dict(
                worker_exit_code=0,
                checkpoint_sha256=status["checkpoint_sha256"],
                execution_sha256=sha(execution),
                session_active_seconds=200,
                session_limit_seconds=300,
                total_active_seconds=180200,
            )
        )
    )
    path.with_suffix(".sha256").write_text(sha(path))
    status["exit_receipt"] = str(path)
    (run / "status.json").write_text(json.dumps(status))
    return path


def test_v3_requires_real_exit_receipt_before_safe_shutdown(tmp_path):
    from scripts.stage3.lg_bev_daily import status

    run, value, execution = stopped_fixture(tmp_path)
    assert not status(run)["safe_to_shutdown"]
    exit_receipt(run, value, execution)
    closed = status(run)
    assert closed["safe_to_shutdown"] and closed["total_active_seconds"] == 180200


def test_live_supervisor_or_tampered_receipt_never_claims_safe(tmp_path, monkeypatch):
    from scripts.stage3 import lg_bev_daily as daily

    run, value, execution = stopped_fixture(tmp_path)
    path = exit_receipt(run, value, execution)
    value.update(supervisor_pid=42, safe_to_shutdown=True)
    (run / "status.json").write_text(json.dumps(value))
    monkeypatch.setattr(daily, "alive", lambda v, _: v.get("pid") == 42)
    assert not daily.status(run)["safe_to_shutdown"]
    monkeypatch.setattr(daily, "alive", lambda v, _: False)
    path.write_text(path.read_text() + " ")
    assert not daily.status(run)["safe_to_shutdown"]


def test_v3_ready_is_not_launch_authorization(monkeypatch):
    from scripts.stage3 import lg_bev_daily as daily
    from scripts.stage3.lg_bev_daily_state import V3_OUT

    monkeypatch.setattr(daily, "POLICY", V3_POLICY)
    monkeypatch.setattr(daily, "OUT", V3_OUT)
    with pytest.raises(ValueError, match="AUTHORIZATION"):
        daily.require_launch_authorization()


def test_checkpoint_resume_preserves_partial_optimizer_scheduler_rng_and_time(tmp_path):
    from scripts.stage3.lg_bev_prototype import LanguageGroundedBEVAdapter
    from scripts.stage3.lg_bev_v3_acceptance import exact_tree

    encoder = torch.nn.Linear(2, 2).requires_grad_(False)
    adapter = LanguageGroundedBEVAdapter(hidden_channels=24, relative_bound=0.05)
    trainer = Trainer(
        None,
        encoder,
        None,
        ["one", "two"],
        lambda _: None,
        "B",
        10,
        "protocol",
        adapter=adapter,
    )
    for p in trainer.adapter.parameters():
        p.grad = torch.ones_like(p) * 1e-5
    trainer.optimizer.step()
    trainer.scheduler.step()
    trainer.cursor, trainer.accumulated, trainer.updates = 1, 1, 1
    trainer.session = SessionClock(73 * 3600, total_seconds=None, margin_seconds=600)
    path = tmp_path / "partial.pt"
    trainer.checkpoint(path)
    saved = torch.load(path, weights_only=False)
    other = Trainer(
        None,
        encoder,
        None,
        ["one", "two"],
        lambda _: None,
        "B",
        10,
        "protocol",
        adapter=LanguageGroundedBEVAdapter(hidden_channels=24, relative_bound=0.05),
    )
    other.session = SessionClock(73 * 3600 + 5, total_seconds=None, margin_seconds=600)
    other.total_budget_seconds = None
    other.resume(path)
    assert other.cursor == 1 and other.accumulated == 1 and other.updates == 1
    assert other.elapsed_wall() >= 73 * 3600 + 5
    assert exact_tree(saved["optimizer"], other.optimizer.state_dict())
    assert exact_tree(saved["scheduler"], other.scheduler.state_dict())
    assert all(
        torch.equal(p.grad, saved["gradients"][n])
        for n, p in other.adapter.named_parameters()
    )
    assert not other.should_stop()

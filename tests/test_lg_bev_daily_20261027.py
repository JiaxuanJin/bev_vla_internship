"""离线会话与累计预算、损坏拒绝、状态派生及正式准入边界。"""

from types import SimpleNamespace
import json
import socket

import pytest
import torch

from scripts.stage3.lg_bev_daily_state import (
    SessionClock,
    validate_policy,
    POLICY,
    checkpoint_valid,
)
from scripts.stage3.lg_bev_common import sha


def test_v2_only_changes_authorized_resource_time():
    policy = validate_policy(POLICY)
    assert policy["total_active_seconds"] == 172800
    assert policy["session_active_seconds"] == 27000
    assert policy["training_content_changed"] is False


def test_daily_limit_preserves_total_on_next_session(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(
        "scripts.stage3.lg_bev_daily_state.time.perf_counter", lambda: clock[0]
    )
    day1 = SessionClock(3600)
    clock[0] += 26880
    assert day1.reason() == "DAILY_SESSION_LIMIT"
    total = day1.total()
    clock[0] += 86400  # 停机等待不计入第二次会话。
    day2 = SessionClock(total)
    assert day2.total() == total and day2.elapsed() == 0
    clock[0] += 10
    assert day2.total() == total + 10


def test_total_limit_is_not_48hours_per_session(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        "scripts.stage3.lg_bev_daily_state.time.perf_counter", lambda: clock[0]
    )
    session = SessionClock(172600)
    clock[0] = 81
    assert session.reason() == "TOTAL_ACTIVE_TIME_LIMIT"
    with pytest.raises(ValueError):
        SessionClock(172801)


def test_user_request_stops_at_shared_clock_boundary(tmp_path):
    path = tmp_path / "request"
    clock = SessionClock(0, stop_path=path)
    path.write_text("stop")
    assert clock.reason() == "USER_STOP"


def test_same_session_budget_spans_stage_a_and_b(monkeypatch):
    from scripts.stage3.lg_bev_engineering_train import Trainer

    times = [0.0]
    monkeypatch.setattr(
        "scripts.stage3.lg_bev_daily_state.time.perf_counter", lambda: times[0]
    )
    shared = SessionClock(500, session_seconds=100, margin_seconds=0)
    a = Trainer.__new__(Trainer)
    b = Trainer.__new__(Trainer)
    for trainer in (a, b):
        trainer.session = shared
        trainer.total_budget_seconds = 172800
        trainer.stop_requested = False
    times[0] = 60
    assert a.elapsed_wall() == b.elapsed_wall() == 560
    times[0] = 101
    assert a.should_stop() and b.should_stop()


def test_offline_guard_rejects_actual_connection_call(monkeypatch):
    from scripts.stage3.lg_bev_daily import deny_network

    monkeypatch.setattr(socket.socket, "connect", socket.socket.connect)
    attempts = deny_network()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        with pytest.raises(OSError, match="禁止网络"):
            stream.connect(("127.0.0.1", 9))
    assert len(attempts) == 1


def checkpoint_fixture(tmp_path):
    run = tmp_path / "run"
    directory = run / "checkpoints"
    directory.mkdir(parents=True)
    (run / "execution.json").write_text("{}")
    path = directory / "A_u000001.pt"
    torch.save(dict(protocol_sha256=sha(run / "execution.json")), path)
    digest = sha(path)
    receipt = path.with_name(path.name + ".receipt.pt")
    torch.save(dict(checkpoint_sha256=digest, wall_seconds=100), receipt)
    record = dict(path=str(path), sha256=digest, receipt_sha256=sha(receipt))
    return run, path, receipt, record


def test_checkpoint_sha_and_receipt_are_verified(tmp_path):
    run, path, _receipt, record = checkpoint_fixture(tmp_path)
    assert checkpoint_valid(run, record)["wall_seconds"] == 100
    with path.open("r+b") as stream:
        stream.write(b"BAD!")
    with pytest.raises(ValueError, match="损坏"):
        checkpoint_valid(run, record)


def test_missing_or_corrupt_time_receipt_is_not_silently_accepted(tmp_path):
    run, _path, receipt, record = checkpoint_fixture(tmp_path)
    receipt.unlink()
    with pytest.raises(ValueError, match="receipt"):
        checkpoint_valid(run, record)
    torch.save(dict(checkpoint_sha256="wrong", wall_seconds=0), receipt)
    with pytest.raises(ValueError, match="receipt"):
        checkpoint_valid(run, record)


def test_early_production_start_fails_before_launch(tmp_path, monkeypatch):
    from scripts.stage3 import lg_bev_daily

    monkeypatch.setattr(lg_bev_daily, "OUT", tmp_path)
    with pytest.raises(ValueError, match="BLOCKED"):
        lg_bev_daily.admission()


def test_paused_warmup_does_not_seal_completed_gate(tmp_path, monkeypatch):
    from scripts.stage3 import lg_bev_engineering_full as full

    trainer = SimpleNamespace(
        adapter=torch.nn.Linear(1, 1),
        protocol_sha256="x",
        updates=9562,
        wall_seconds=0,
        elapsed_wall=lambda: 1,
    )
    monkeypatch.setattr(full, "warmup_probe", lambda _: dict(status="PAUSED"))
    assert full.completed_warmup_gate(trainer, tmp_path)["status"] == "PAUSED"
    assert not (tmp_path / "warmup_gate.json").exists()


def test_status_never_claims_safe_shutdown_for_dead_running_process(tmp_path):
    from scripts.stage3.lg_bev_daily import status

    (tmp_path / "status.json").write_text(
        json.dumps(dict(status="RUNNING", pid=999999999))
    )
    result = status(tmp_path)
    assert result["status"] == "FAILED_UNCLEAN" and not result["safe_to_shutdown"]


def test_starting_uses_real_tmux_pid_before_worker_first_status(tmp_path, monkeypatch):
    from scripts.stage3 import lg_bev_daily as daily

    (tmp_path / "status.json").write_text(
        json.dumps(dict(status="STARTING", pid=None, tmux_session="test"))
    )
    monkeypatch.setattr(
        daily.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="123\n"),
    )
    monkeypatch.setattr(daily, "alive", lambda summary, _: summary["pid"] == 123)
    result = daily.status(tmp_path)
    assert result["status"] == "STARTING" and result["process_alive"]
    assert not result.get("safe_to_shutdown", False)


def test_blocked_admission_never_displays_ready(tmp_path, monkeypatch):
    from scripts.stage3 import lg_bev_daily as daily

    run = tmp_path / "not_started"
    monkeypatch.setattr(daily, "OUT", tmp_path)
    monkeypatch.setattr(daily, "RUN", run)
    (tmp_path / "admission.json").write_text('{"status":"BLOCKED"}')
    assert daily.status(run)["status"] == "BLOCKED"


def test_real_legacy_sampler_empty_3d_shape_preserves_all_negatives():
    """真实本机PseudoSampler的空3D回退修复，不删除负样本/改变匹配。"""
    from mmdet.core.bbox.samplers import PseudoSampler
    from mmdet.core.bbox.assigners import AssignResult
    from scripts.stage3.lg_bev_engineering_loss import empty_positive_shape

    sampler = PseudoSampler()
    head = SimpleNamespace(sampler=sampler)
    assignment = AssignResult(0, torch.zeros(4, dtype=torch.long), torch.zeros(4))
    boxes = torch.zeros(4, 10)
    gt = torch.empty(0, 9)
    original = sampler.sample(assignment, boxes, gt)
    assert original.pos_gt_bboxes.shape == (0, 4)
    with pytest.raises(RuntimeError, match="forced"):
        with empty_positive_shape(head):
            actual = sampler.sample(assignment, boxes, gt)
            assert actual.pos_gt_bboxes.shape == (0, 9)
            assert torch.equal(actual.neg_inds, original.neg_inds)
            assert actual.pos_inds.numel() == 0
            raise RuntimeError("forced")
    assert "sample" not in sampler.__dict__


def test_formal_launch_needs_new_authorization_not_just_date(tmp_path, monkeypatch):
    from scripts.stage3 import lg_bev_daily

    monkeypatch.setattr(lg_bev_daily, "OUT", tmp_path)
    (tmp_path / "launch_hold_20261027.json").write_text("{}")
    with pytest.raises(ValueError, match="明天仍须"):
        lg_bev_daily.require_launch_authorization()
    (tmp_path / "launch_authorization.json").write_text(
        json.dumps(
            dict(
                status="AUTHORIZED_BY_NEW_EXPLICIT_USER_REQUEST",
                hold_sha256="wrong",
                authorization_date="2099-01-01",
                policy_sha256="wrong",
            )
        )
    )
    with pytest.raises(ValueError, match="不能按日期"):
        lg_bev_daily.require_launch_authorization()


def test_source_fingerprint_cache_reuses_actual_sha_and_rejects_mutation(
    tmp_path, monkeypatch
):
    from scripts.stage3 import lg_bev_engineering_native as native
    from scripts.stage3.lg_bev_engineering_cache import RollingCache

    raw = tmp_path / "raw"
    raw.mkdir()
    for i in range(7):
        (raw / str(i)).write_bytes(bytes([i]) * 16)
    monkeypatch.setattr(native, "DATA", raw)
    cache = RollingCache(
        tmp_path / "owned", test_root=True, reserve=0, capacity=3000000
    )
    data = SimpleNamespace(
        rows=[
            dict(
                scene_token="s", media={str(i): dict(filename=str(i)) for i in range(7)}
            )
        ]
    )
    expected = {str(raw / str(i)): sha(raw / str(i)) for i in range(7)}
    actual, hit = native.verified_scene_sources(cache, data, expected)
    assert actual == expected and not hit
    original_sha = native.sha
    monkeypatch.setattr(
        native,
        "sha",
        lambda _: (_ for _ in ()).throw(AssertionError("不应重复读源文件SHA")),
    )
    assert native.verified_scene_sources(cache, data, expected) == (expected, True)
    monkeypatch.setattr(native, "sha", original_sha)
    (raw / "0").write_bytes(b"changed")
    with pytest.raises(ValueError, match="来源路径"):
        native.verified_scene_sources(cache, data, expected)

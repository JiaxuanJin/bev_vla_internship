"""真实CPU子进程退出证明、隔离路径、授权与失效关闭。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from core import HERE, OUTPUT, SHARED, Store, authorize, code_manifest, read, sha
from lifecycle import verified_checkpoint, record_closure


@pytest.mark.parametrize("relative", ["../outside", "/tmp/outside", "x/../../outside"])
def test_path_escape_denied(store, relative):
    with pytest.raises(ValueError):
        store.bytes(relative, b"x")


@pytest.mark.parametrize(
    "root", [SHARED, SHARED / "runs/test", OUTPUT.parent / "other"]
)
def test_formal_and_arbitrary_roots_denied(root):
    with pytest.raises(ValueError):
        Store(root)


def test_symlink_and_overwrite_denied(store, tmp_path):
    path = store.bytes("asset", b"original")
    with pytest.raises(ValueError):
        store.bytes("asset", b"changed")
    assert path.read_bytes() == b"original"
    (tmp_path / "link").symlink_to(tmp_path)
    with pytest.raises(ValueError):
        store.bytes("link/asset", b"changed")
    os.link(path, tmp_path / "hardlink")
    with pytest.raises(ValueError):
        store.bytes("hardlink", b"changed")


def test_write_fence_in_subprocess(store, tmp_path):
    root = tmp_path / "isolated"
    root.mkdir()
    script = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from backend import install_write_fence
root = Path(sys.argv[2])
install_write_fence(root)
(root / 'allowed').write_text('允许')
try:
    (root.parent / 'outside').write_text('禁止')
except ValueError:
    pass
else:
    raise AssertionError('根外写入未阻止')
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(HERE), str(root)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (root / "allowed").exists() and not (tmp_path / "outside").exists()


def test_development_and_unknown_frames_denied_before_scene_creation():
    from backend import Runtime

    runtime = object.__new__(Runtime)
    runtime.rows = {"dev": {"split": "development"}}
    runtime.development_authorized = False
    with pytest.raises(ValueError, match="开发GT"):
        runtime._scene("dev")
    with pytest.raises(ValueError, match="未授权帧"):
        runtime._scene("cold")


@pytest.mark.parametrize(
    "damage", ["approved", "action", "config", "code", "request", "development", "cold"]
)
def test_authorization_fail_closed(tmp_path, damage):
    path = HERE / "configs/main.json"
    approval = {
        "approved": True,
        "actions": ["evaluate"],
        "config_sha256": sha(path),
        "code_sha256": code_manifest(),
        "user_request": "仅测试的合成授权，不启动运行",
        "development_gt": True,
        "cold_final": False,
    }
    field, value = {
        "approved": ("approved", False),
        "action": ("actions", []),
        "config": ("config_sha256", "wrong"),
        "code": ("code_sha256", {}),
        "request": ("user_request", ""),
        "development": ("development_gt", False),
        "cold": ("cold_final", True),
    }[damage]
    approval[field] = value
    auth = tmp_path / "authorization.json"
    auth.write_text(json.dumps(approval))
    with pytest.raises(ValueError):
        authorize(path, auth, "evaluate")


def completed_cpu_session(store, outcome="PAUSED"):
    """夹具只生成明确标记的CPU合成实验记录，绝不接触历史run。"""
    script = """
import sys, subprocess, time
sys.path.insert(0, sys.argv[1])
from core import Store
from lifecycle import save_checkpoint, saved_end, record_exit
store = Store(sys.argv[2], test=True)
expected = {'test_only': 'CPU合成会话'}
start = time.monotonic()
child = subprocess.Popen([sys.executable, '-B', '-c', 'pass'])
code = child.wait()
relative, digest = save_checkpoint(store, 'synthetic', '1_a', {'identity': expected}, 'end')
saved_end(store, 'synthetic', '1_a', expected, relative, digest, sys.argv[3])
record_exit(store, 'synthetic', '1_a', expected, child, code, start)
"""
    start = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", script, str(HERE), str(store.root), outcome],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    _, stderr = process.communicate(timeout=60)
    assert process.returncode == 0, stderr
    expected = {"test_only": "CPU合成会话"}
    record_closure(
        store, "synthetic", "1_a", expected, process, process.returncode, start
    )
    return expected, store.path("synthetic/sessions/1_a")


@pytest.mark.parametrize("outcome", ["PAUSED", "COMPLETED", "PREFLIGHT_PASS"])
def test_real_cpu_wait_exit_closure(store, outcome):
    expected, _ = completed_cpu_session(store, outcome)
    state, end, _ = verified_checkpoint(
        store, "synthetic", expected, require_outcome=outcome
    )
    assert state["identity"] == expected and end["outcome"] == outcome


@pytest.mark.parametrize(
    "damage",
    [
        "checkpoint",
        "missing_exit",
        "missing_end",
        "identity",
        "alive",
        "nonzero",
        "overrun",
        "new_incomplete",
        "wrong_outcome",
    ],
)
def test_session_evidence_fail_closed(store, damage):
    expected, session = completed_cpu_session(store)
    if damage == "checkpoint":
        path = store.path(read(session / "worker_end.json")["checkpoint"])
        path.write_bytes("损坏的合成Checkpoint".encode())
    elif damage.startswith("missing"):
        (
            session / ("exit.json" if damage == "missing_exit" else "worker_end.json")
        ).unlink()
    elif damage == "identity":
        expected = {"test_only": "错误身份"}
    elif damage in ("alive", "nonzero"):
        receipt = read(session / "exit.json")
        receipt["worker_pid" if damage == "alive" else "worker_exit_code"] = (
            os.getpid() if damage == "alive" else 1
        )
        (session / "exit.json").write_text(json.dumps(receipt))
        closure = read(session / "closure.json")
        closure["receipt_sha256"] = sha(session / "exit.json")
        (session / "closure.json").write_text(json.dumps(closure))
    elif damage == "overrun":
        (session / "overrun.json").write_text("{}")
    elif damage == "new_incomplete":
        store.path("synthetic/sessions/2_b").mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        verified_checkpoint(
            store,
            "synthetic",
            expected,
            require_outcome="COMPLETED" if damage == "wrong_outcome" else None,
        )


def test_real_nonzero_exit_does_not_issue_receipt(store):
    from lifecycle import record_exit

    process = subprocess.Popen([sys.executable, "-B", "-c", "raise SystemExit(3)"])
    code = process.wait(timeout=30)
    with pytest.raises(ValueError, match="未正常退出"):
        record_exit(store, "synthetic", "1_a", {}, process, code, time.monotonic())
    assert not store.path("synthetic/sessions/1_a/exit.json").exists()


@pytest.mark.parametrize("failure", ["new_assets", "combined", "free_space"])
def test_resource_limits_before_any_write(monkeypatch, failure):
    import core
    from types import SimpleNamespace

    store = Store()
    monkeypatch.setattr(core.Path, "exists", lambda _: True)
    monkeypatch.setattr(
        core,
        "tree_bytes",
        lambda path: (
            8_000_000_000
            if failure == "new_assets" and path == OUTPUT
            else 30_000_000_000 if failure == "combined" and path == SHARED else 0
        ),
    )
    monkeypatch.setattr(
        core.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(
            free=250_000_000_000 if failure == "free_space" else 400_000_000_000
        ),
    )
    with pytest.raises(ValueError):
        store.reserve_for_artifact(4096)

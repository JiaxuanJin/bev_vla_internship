"""独立会话收尾与继续训练校验，不兼容旧正式resume或历史补回执。"""

from __future__ import annotations

import os
from pathlib import Path
import time

import torch

from core import code_manifest, read, require, sha


def identity(config_path, action):
    """配置和候选源码共同构成独立运行身份。"""
    return {
        "config_sha256": sha(config_path),
        "code_sha256": code_manifest(),
        "action": action,
        "schema": "LG_BEV_EXPLORATION_V1",
    }


def process_absent(pid):
    """PID复用或权限不明保守视为仍存活，不发送任何信号。"""
    require(type(pid) is int and pid > 0, "无效PID")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def session_directories(store, run_id):
    """最新会话缺证据时拒绝，禁止静默回退较旧Checkpoint。"""
    root = store.path(run_id + "/sessions")
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def verified_checkpoint(store, run_id, expected, *, require_outcome=None):
    """只接受本候选真实闭环会话及完整Checkpoint摘要。"""
    sessions = session_directories(store, run_id)
    require(sessions, "独立实验尚无可用会话")
    session = sessions[-1]
    closure = read(session / "closure.json")
    receipt = read(session / "exit.json")
    end = read(session / "worker_end.json")
    require(closure["receipt_sha256"] == sha(session / "exit.json"), "退出回执损坏")
    require(
        receipt["worker_end_sha256"] == sha(session / "worker_end.json"), "保存证据损坏"
    )
    require(
        closure["identity"] == receipt["identity"] == end["identity"] == expected,
        "独立会话身份不匹配",
    )
    require(
        closure["session_id"]
        == receipt["session_id"]
        == end["session_id"]
        == session.name,
        "会话ID错配",
    )
    require(
        closure["supervisor_exit_code"] == receipt["worker_exit_code"] == 0,
        "进程异常退出",
    )
    require(
        receipt["outcome"]
        == end["outcome"]
        in ("PAUSED", "COMPLETED", "PREFLIGHT_PASS"),
        "不能继续异常或未知终态",
    )
    require(not (session / "overrun.json").exists(), "会话超时")
    require(
        0 <= receipt["elapsed_seconds"] <= closure["elapsed_seconds"] <= 27000,
        "会话时限证据无效",
    )
    require(
        process_absent(receipt["worker_pid"])
        and process_absent(receipt["supervisor_pid"]),
        "Worker或Supervisor仍存活或无法确认",
    )
    if require_outcome is not None:
        require(end["outcome"] == require_outcome, "终态不满足入口要求")
    path = store.path(end["checkpoint"])
    require(path.is_relative_to(store.path(run_id)), "Checkpoint不属于该实验")
    require(
        sha(path) == end["checkpoint_sha256"] == receipt["checkpoint_sha256"],
        "独立Checkpoint损坏",
    )
    state = torch.load(path, map_location="cpu", weights_only=False)
    require(state["identity"] == expected, "Checkpoint内嵌身份不符")
    return state, end, path


def save_checkpoint(store, run_id, session_id, state, suffix):
    """不可覆盖保存；只保存新实验状态，不改源Checkpoint。"""
    relative = f"{run_id}/checkpoints/{session_id}_{suffix}.pt"
    path = store.tensor(relative, state)
    return relative, sha(path)


def saved_end(
    store, run_id, session_id, expected, checkpoint, digest, outcome, **details
):
    """Worker持久化保存完成事件，尚不宣称进程已退出。"""
    return store.json(
        f"{run_id}/sessions/{session_id}/worker_end.json",
        {
            "identity": expected,
            "session_id": session_id,
            "checkpoint": checkpoint,
            "checkpoint_sha256": digest,
            "outcome": outcome,
            **details,
        },
    )


def record_exit(store, run_id, session_id, expected, child, code, started):
    """Supervisor只能记录自己wait得到的真实退出码，不接受历史状态替代。"""
    require(child.poll() == code and code == 0, "Worker未正常退出")
    session = store.path(f"{run_id}/sessions/{session_id}")
    end = read(session / "worker_end.json")
    require(
        end["identity"] == expected and end["session_id"] == session_id, "保存身份错误"
    )
    require(
        sha(store.path(end["checkpoint"])) == end["checkpoint_sha256"], "保存后摘要变化"
    )
    elapsed = time.monotonic() - started
    require(elapsed <= 27000, "GPU会话退出超时")
    return store.json(
        f"{run_id}/sessions/{session_id}/exit.json",
        {
            "identity": expected,
            "session_id": session_id,
            "worker_pid": child.pid,
            "supervisor_pid": os.getpid(),
            "worker_exit_code": code,
            "elapsed_seconds": elapsed,
            "worker_exit_observed_at": time.time(),
            "checkpoint_sha256": end["checkpoint_sha256"],
            "outcome": end["outcome"],
            "worker_end_sha256": sha(session / "worker_end.json"),
        },
    )


def record_closure(store, run_id, session_id, expected, supervisor, code, started):
    """外层启动器真实观察Supervisor退出，封存独立核验结果。"""
    require(supervisor.poll() == code and code == 0, "Supervisor未正常退出")
    session = store.path(f"{run_id}/sessions/{session_id}")
    receipt = read(session / "exit.json")
    require(
        receipt["supervisor_pid"] == supervisor.pid and receipt["identity"] == expected,
        "Supervisor身份不符",
    )
    require(process_absent(receipt["worker_pid"]), "GPU进程仍存活")
    elapsed = time.monotonic() - started
    require(elapsed <= 27000, "包含Supervisor退出的会话超时")
    path = store.json(
        f"{run_id}/sessions/{session_id}/closure.json",
        {
            "identity": expected,
            "session_id": session_id,
            "supervisor_exit_code": code,
            "supervisor_exit_observed_at": time.time(),
            "elapsed_seconds": elapsed,
            "receipt_sha256": sha(session / "exit.json"),
            "timing_scope": "启动至真实Supervisor退出观察；包含模型初始化、训练、Checkpoint保存和Worker退出",
        },
    )
    if time.monotonic() - started > 27000:
        store.json(f"{run_id}/sessions/{session_id}/overrun.json", {"overrun": True})
        raise RuntimeError("最终收尾持久化超过会话时限")
    return path

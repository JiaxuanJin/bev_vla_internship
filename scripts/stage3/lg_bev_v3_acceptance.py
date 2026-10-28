"""V3独立32帧真实验收：复用FP32参考、跨进程控制及完整状态对照。"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

import numpy as np
import torch

from scripts.stage3.lg_bev_common import ROOT, read, sha, write
from scripts.stage3.lg_bev_daily_state import (
    V3_POLICY,
    V3_RUN,
    source_manifest,
    latest_checkpoint,
    validate_policy,
    event,
    atomic_json,
)
from scripts.stage3.lg_bev_engineering_cache import (
    CACHE_ROOT,
    RollingCache,
    atomic_save,
)
from scripts.stage3.lg_bev_engineering_audit import Reference, maximum_delta
from scripts.stage3.lg_bev_engineering_data import LabelStore
from scripts.stage3.lg_bev_engineering_native import NativeRuntime
from scripts.stage3.lg_bev_engineering_train import Trainer, rng_state
from scripts.stage3.lg_bev_native import module_sha, dual_head

PRIOR = CACHE_ROOT / "runs/daily_preflight_20261027_v5"
PREFLIGHT = CACHE_ROOT / "runs/daily_preflight_v3_20261028_v1"


def cpu_copy(value):
    """独立复制所有持久状态，避免后续optimizer原地修改参考。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def seed_all() -> None:
    """只固定独立小测初始化；恢复仍使用checkpoint的完整RNG。"""
    random.seed(20261027)
    np.random.seed(20261027)
    torch.manual_seed(20261027)
    torch.cuda.manual_seed_all(20261027)


def snapshot(trainer: Trainer) -> dict:
    """与真实checkpoint同名字段，另存不中断参考和部分梯度。"""
    return cpu_copy(
        dict(
            state_dict=trainer.adapter.state_dict(),
            gradients={
                n: p.grad
                for n, p in trainer.adapter.named_parameters()
                if p.grad is not None
            },
            optimizer=trainer.optimizer.state_dict(),
            scheduler=trainer.scheduler.state_dict() if trainer.scheduler else None,
            rng=rng_state(),
            cursor=trainer.cursor,
            epoch=trainer.epoch,
            updates=trainer.updates,
            accumulated=trainer.accumulated,
            tokens=trainer.tokens,
            baseline=trainer.baseline,
            budget_updates=trainer.budget_updates,
        )
    )


def capture(trainer: Trainer, record: dict, audit: dict) -> None:
    """只复制验收张量，不额外反向或改变任何训练数学。"""
    audit["records"].append(copy.deepcopy(record))
    gradients = (
        trainer.update_records[-1]["gradients"]
        if "update" in record
        else {
            n: p.grad
            for n, p in trainer.adapter.named_parameters()
            if p.grad is not None
        }
    )
    audit["gradients"].append(cpu_copy(gradients))
    if "update" in record:
        audit["updates"].append(cpu_copy(trainer.update_records[-1]))


def save(cache: RollingCache, path: Path, value) -> None:
    """所有新张量只在G盘保存，容量不够即拒绝而非删除历史产物。"""
    with cache.locked():
        cache.admit(64_000_000)
        atomic_save(path, cpu_copy(value))


def prepare(preflight: Path, prior: Path = PRIOR) -> dict:
    """复用三场景标签/真实FP32，当前源码仅做必要的缓存/流式B32复核。"""
    from scripts.stage3.lg_bev_daily import offline_environment, deny_network
    from scripts.stage3.lg_bev_daily_acceptance import preflight_rows, local_encoder
    from scripts.stage3.lg_bev_engineering_full import SceneProvider

    validate_policy(V3_POLICY)
    offline_environment()
    attempts = deny_network()
    sources = source_manifest()
    cache = RollingCache()
    preflight.mkdir(parents=True, exist_ok=False)
    evidence = preflight / "evidence"
    evidence.mkdir()
    rows, full = preflight_rows()
    tokens = [r["sample_token"] for r in rows]
    prior_cross = read(prior / "evidence/cross_scene.json")
    for name in [
        "lg_bev_engineering_native.py",
        "lg_bev_engineering_data.py",
        "lg_bev_engineering_determinism.py",
    ]:
        path = ROOT / "scripts/stage3" / name
        if sha(path) != prior_cross["source_sha256"][str(path)]:
            raise ValueError("原生/标签数学变化，不能复用旧FP32参考")
    report = read(prior / "data_report.json")
    if {r["sample_token"] for r in report["frames"]} != set(tokens):
        raise ValueError("复用标签不是既定三个train场景32帧")
    raw = read(prior / "evidence/native_temporal_audit.json")
    if not (raw["true_FP32"] and raw["head_replay_bitwise"] and raw["frames"] == 32):
        raise ValueError("旧FP32真实参考未通过")
    for entry in raw["reference"]:
        if sha(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("FP32参考损坏")
    write(preflight / "data_report.json", {**report, "reused_from": str(prior)})
    write(
        evidence / "native_temporal_audit.json",
        {
            **raw,
            "reused_from": str(prior),
            "fresh_raw_forward_claimed": False,
        },
    )
    write(evidence / "selection.json", dict(rows=rows, train_only=True))
    torch.cuda.reset_peak_memory_stats()
    runtime = NativeRuntime(fast_images=True)
    encoder = local_encoder()
    frozen = (module_sha(runtime.model), module_sha(encoder))
    labels = LabelStore(report)
    reference = Reference(raw["reference"])
    seed_all()
    trainer = Trainer(
        runtime.model,
        encoder,
        labels,
        tokens,
        reference.get,
        "B",
        19124,
        sha(V3_POLICY),
        retain_audit_tensors=True,
    )
    if sum(p.numel() for p in trainer.adapter.parameters() if p.requires_grad) != 93866:
        raise ValueError("Adapter可训练参数数目变化")
    audit = dict(records=[], gradients=[], updates=[], stops={})

    def collect(current, record):
        capture(current, record, audit)
        if current.cursor in (1, 2, 32):
            audit["stops"][current.cursor] = snapshot(current)

    trainer.progress_callback = collect
    began = time.perf_counter()
    trainer.run(32, log=preflight / "uninterrupted_B.jsonl")
    reference_seconds = time.perf_counter() - began
    audit["final"] = snapshot(trainer)
    save(cache, preflight / "uninterrupted_reference_v3.pt", audit)
    old = torch.load(
        prior / "uninterrupted_reference.pt", map_location="cpu", weights_only=False
    )
    old_delta, _ = maximum_delta(old["state_dict"], audit["final"]["state_dict"])
    old_loss = max(
        abs(a["losses"]["total"] - b["losses"]["total"])
        for a, b in zip(old["records"], audit["records"])
    )
    if old_delta > 1e-6 or old_loss > 1e-5:
        raise RuntimeError("当前训练数学偏离原V5缓存参考")

    # 当前代码重新执行原生时序；额外前缀8帧不冒充独立32帧。
    provenance = read(CACHE_ROOT / "runs/formal_gt_20261027_v1/source_provenance.json")
    provider = SceneProvider(
        cache, runtime, full, source_provenance=provenance["scenes"]
    )
    provider.hot_scenes = set()
    raw_forward = runtime.forward
    traces = []

    def checked_forward(data, index):
        bev, prediction, trace = raw_forward(data, index)
        token = data.rows[index]["sample_token"]
        if token in set(tokens):
            with torch.no_grad():
                replay = dual_head(runtime.model, bev.cuda())
            delta = {
                k: float((prediction[k] - replay[k].cpu()).abs().max())
                for k in prediction
            }
            if any(delta.values()) or not torch.equal(bev, reference.get(token)):
                raise RuntimeError("当前原始FP32/双Head与参考非逐位一致")
            traces.append({**trace, "head_max_abs": delta})
        return bev, prediction, trace

    runtime.forward = checked_forward
    seed_all()
    streamed = Trainer(
        runtime.model,
        encoder,
        labels,
        tokens,
        provider.get,
        "B",
        19124,
        sha(V3_POLICY),
        retain_audit_tensors=True,
    )
    began = time.perf_counter()
    streamed.run(32, log=preflight / "fresh_stream_B.jsonl")
    stream_seconds = time.perf_counter() - began
    runtime.forward = raw_forward
    delta, relative = maximum_delta(
        audit["final"]["state_dict"], cpu_copy(streamed.adapter.state_dict())
    )
    loss = max(
        abs(a["losses"]["total"] - b["losses"]["total"])
        for a, b in zip(audit["records"], streamed.records)
    )
    if delta > 1e-6 or loss > 1e-5 or len(traces) != 32:
        raise RuntimeError("当前跨场景流式训练偏离参考")
    if frozen != (module_sha(runtime.model), module_sha(encoder)) or attempts:
        raise RuntimeError("冻结或离线边界失败")
    if sources != source_manifest():
        raise RuntimeError("验收期间源码变化")
    result = dict(
        status="PASS",
        frames=32,
        scenes=sorted({r["scene_name"] for r in rows}),
        train_only=True,
        raw_FP32_heads_bitwise=True,
        fresh_raw_traces=traces,
        parameter_max_abs=delta,
        parameter_relative_l2=relative,
        loss_max_abs=loss,
        original_v5_parameter_max_abs=old_delta,
        original_v5_loss_max_abs=old_loss,
        reference_seconds=reference_seconds,
        current_stream_seconds=stream_seconds,
        max_microstep_seconds=max(
            r["microstep_wall_seconds"] for r in streamed.records
        ),
        frozen_sha256=frozen,
        source_sha256=sources,
        offline_network_attempts=0,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        active_point_hours=prior_cross["active_point_hours"],
        conservative_active_hours=prior_cross["conservative_active_hours"],
        resource_estimate_source_sha256=sha(prior / "evidence/cross_scene.json"),
        resource_estimate_is_warning_only=True,
        full_training_updates=0,
        policy_sha256=sha(V3_POLICY),
    )
    write(evidence / "cross_scene_v3.json", result)
    config = dict(
        scope="bounded_real_train_acceptance",
        protocol_version="V3",
        formal_training=False,
        run_id=preflight.name.replace("daily_preflight", "daily_acceptance"),
        stage="B",
        total_microsteps=32,
        budget_updates=19124,
        session_seconds=300.0,
        margin_seconds=180.0,
        simulated_previous_seconds=49 * 3600.0,
        simulation_scope="only bounded cumulative-time fixture; never formal resource usage",
        label_report=str(preflight / "data_report.json"),
        native_report=str(evidence / "native_temporal_audit.json"),
        reference=str(preflight / "uninterrupted_reference_v3.pt"),
        evidence_dir=str(evidence),
        tokens=tokens,
        policy_sha256=sha(V3_POLICY),
        source_sha256=sources,
    )
    write(preflight / "acceptance_config.json", config)
    print("V3_REFERENCE_AND_CROSS_SCENE_PASS", json.dumps(result), flush=True)
    return result


def acceptance_worker(work, config: dict) -> None:
    """真实管理器三会话合计32微步；测试时钟含资产和保存，不重置起点。"""
    from scripts.stage3.lg_bev_daily_acceptance import local_encoder

    if config["source_sha256"] != source_manifest() or config["policy_sha256"] != sha(
        V3_POLICY
    ):
        raise ValueError("V3验收源码/协议变化")
    run = work.run
    execution = dict(
        scope=config["scope"],
        policy_sha256=sha(V3_POLICY),
        source_sha256=source_manifest(),
        tokens=config["tokens"],
        formal_training=False,
        cumulative_hard_limit_seconds=None,
    )
    if not (run / "execution.json").exists():
        write(run / "execution.json", execution)
    elif read(run / "execution.json") != execution:
        raise ValueError("V3验收恢复执行协议变化")
    event(
        run,
        dict(
            event="process_start",
            pid=os.getpid(),
            run_id=run.name,
            session_id=work.args.session_id,
            acceptance_only=True,
        ),
        work.cache,
    )
    runtime = NativeRuntime(fast_images=True)
    encoder = local_encoder()
    labels = LabelStore(read(Path(config["label_report"])))
    reference = Reference(read(Path(config["native_report"]))["reference"])
    seed_all()
    trainer = Trainer(
        runtime.model,
        encoder,
        labels,
        config["tokens"],
        reference.get,
        "B",
        19124,
        sha(run / "execution.json"),
        retain_audit_tensors=True,
    )
    trainer.session = work.clock
    trainer.total_budget_seconds = None
    trainer.checkpoint_callback = work.note_checkpoint
    if work.args.checkpoint:
        trainer.resume(work.args.checkpoint)
    audit = dict(records=[], gradients=[], updates=[])

    def progress(current, record):
        capture(current, record, audit)
        work.progress(current, record)
        consumed = current.epoch * len(current.tokens) + current.cursor
        if consumed == 1 and not work.args.checkpoint:
            # 真正等待缩短的会话进入保存窗口；没有重置/加速时钟，也不多做GPU更新。
            while not work.clock.reason():
                time.sleep(0.05)
        if consumed == 2:
            atomic_json(
                run / ("manual_ready_" + work.args.session_id + ".json"),
                dict(cursor=consumed, session_id=work.args.session_id),
                work.cache,
            )
            while not work.clock.reason():
                time.sleep(0.05)

    trainer.progress_callback = progress
    consumed = trainer.epoch * len(trainer.tokens) + trainer.cursor
    trainer.run(
        32 - consumed,
        log=run / ("training_" + work.args.session_id + ".jsonl"),
        checkpoint_dir=run / "checkpoints",
        cache=work.cache,
    )
    audit["state_at_stop"] = snapshot(trainer)
    audit["bn_frozen"] = all(
        not m.training
        for m in runtime.model.modules()
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
    )
    audit["frozen_grad_none"] = all(p.grad is None for p in runtime.model.parameters())
    save(work.cache, run / ("audit_" + work.args.session_id + ".pt"), audit)
    consumed = trainer.epoch * len(trainer.tokens) + trainer.cursor
    work.finalize(
        trainer,
        "COMPLETED" if consumed == 32 else "STOPPED",
        reason="BOUNDED_ACCEPTANCE_COMPLETE" if consumed == 32 else None,
    )


def control(config_path: Path) -> dict:
    """真实独立进程自动暂停、手工stop、两次resume和退出确认。"""
    from scripts.stage3 import lg_bev_daily as daily

    daily.configure_protocol("v3")
    config = read(config_path)
    run = CACHE_ROOT / "runs" / config["run_id"]
    if run.exists():
        raise ValueError("验收run已存在，拒绝覆盖或把失败重写成PASS")
    calls = []

    def invoke(action: str, expected: int = 0):
        command = [
            sys.executable,
            "-m",
            "scripts.stage3.lg_bev_daily",
            action,
            "--protocol",
            "v3",
            "--acceptance-config",
            str(config_path),
        ]
        if action == "stop":
            command += ["--timeout", "300"]
        result = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, timeout=420
        )
        calls.append(
            dict(
                action=action,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
        if result.returncode != expected:
            raise RuntimeError(str(calls[-1]))

    def wait(target: str) -> dict:
        end = time.monotonic() + 600
        while time.monotonic() < end:
            value = daily.status(run)
            if value["status"].startswith("FAILED"):
                raise RuntimeError(str(value))
            if value["status"] == target and not value.get("process_alive"):
                if not value.get("safe_to_shutdown"):
                    raise RuntimeError("进程退出但receipt不完整")
                return value
            time.sleep(0.5)
        raise TimeoutError(str(daily.status(run)))

    invoke("start")
    first = wait("STOPPED")
    if (
        first["stop_reason"] != "DAILY_SESSION_LIMIT"
        or first["partial_accumulation"] != 1
    ):
        raise ValueError("真实自动暂停未覆盖一帧部分梯度")
    invoke("resume")
    deadline = time.monotonic() + 180
    while not list(run.glob("manual_ready_*.json")):
        if time.monotonic() >= deadline or daily.status(run)["status"].startswith(
            "FAILED"
        ):
            raise RuntimeError("人工stop准备失败")
        time.sleep(0.3)
    invoke("stop")
    second = wait("STOPPED")
    if second["stop_reason"] != "USER_STOP" or second["partial_accumulation"] != 2:
        raise ValueError("人工暂停未覆盖两帧累积")
    invoke("resume")
    third = wait("COMPLETED")
    invoke("resume", expected=2)
    if V3_RUN.exists():
        raise RuntimeError("有界验收创建了正式run")
    sessions = [v["session_id"] for v in (first, second, third)]
    totals = [v["total_active_seconds"] for v in (first, second, third)]
    if not (totals[0] > 172800 and totals[0] < totals[1] < totals[2]):
        raise ValueError("超过48小时恢复或跨进程累计有误")
    try:
        daily.require_launch_authorization()
    except ValueError:
        locked = True
    else:
        locked = False
    result = dict(
        status="PASS",
        sessions=sessions,
        automatic_stop_verified=True,
        manual_stop_verified=True,
        partial_gradient_resume_verified=True,
        safe_to_shutdown_after_exit=True,
        completed_refuses_resume=True,
        over48h_real_resume_verified=True,
        detached_tmux=all(
            v["detached_tmux"] and v.get("supervisor_pid")
            for v in (first, second, third)
        ),
        formal_launch_locked=locked,
        offline_network_attempts=0,
        session_summaries=[first, second, third],
        cumulative_seconds=totals,
        calls=calls,
        formal_updates=0,
    )
    write(Path(config["evidence_dir"]) / "manager_control_v3.json", result)
    print("V3_CONTROL_PASS", json.dumps(result), flush=True)
    return result


def exact_tree(a, b) -> bool:
    """optimizer/scheduler/RNG等状态要求逐项相同，不靠摘要猜测。"""
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(exact_tree(a[k], b[k]) for k in a)
        )
    if isinstance(a, (tuple, list)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(exact_tree(x, y) for x, y in zip(a, b))
        )
    return a == b


def compare(config_path: Path) -> dict:
    """样本/Loss/每步梯度/参数/AdamW/scheduler/RNG及退出时钟全部核验。"""
    config = read(config_path)
    run = CACHE_ROOT / "runs" / config["run_id"]
    expected = torch.load(config["reference"], map_location="cpu", weights_only=False)
    managed = read(Path(config["evidence_dir"]) / "manager_control_v3.json")
    audits = [
        torch.load(run / ("audit_" + s + ".pt"), map_location="cpu", weights_only=False)
        for s in managed["sessions"]
    ]
    records = [r for a in audits for r in a["records"]]
    gradients = [g for a in audits for g in a["gradients"]]
    if len(records) != 32 or [r["token"] for r in records] != config["tokens"]:
        raise ValueError("恢复丢帧/重放或顺序改变")
    loss = max(
        abs(a["losses"][k] - b["losses"][k])
        for a, b in zip(expected["records"], records)
        for k in a["losses"]
    )
    grad_deltas = [
        maximum_delta(a, b) for a, b in zip(expected["gradients"], gradients)
    ]
    if loss > 1e-5 or any(a > 2e-5 or r > 1e-4 for a, r in grad_deltas):
        raise ValueError("恢复的逐微步Loss/梯度偏离不中断参考")
    final = latest_checkpoint(run)["state"]
    delta, relative = maximum_delta(
        expected["final"]["state_dict"], final["state_dict"]
    )
    if delta > 1e-6 or final["updates"] != 8:
        raise ValueError("恢复终点参数或更新数不符")
    checks = {}
    for i, audit in enumerate(audits):
        wanted = expected["stops"][(1, 2, 32)[i]]
        got = audit["state_at_stop"]
        for key in [
            "optimizer",
            "scheduler",
            "rng",
            "tokens",
            "cursor",
            "epoch",
            "accumulated",
            "updates",
            "baseline",
            "budget_updates",
        ]:
            if not exact_tree(wanted[key], got[key]):
                raise ValueError(f"会话{i+1}的{key}与不中断参考不同")
        for key in ["state_dict", "gradients"]:
            dd, rr = maximum_delta(wanted[key], got[key])
            if (
                dd > (1e-6 if key == "state_dict" else 2e-5)
                or key == "gradients"
                and rr > 1e-4
            ):
                raise ValueError("部分累积或Adapter状态改变")
        if not audit["bn_frozen"] or not audit["frozen_grad_none"]:
            raise ValueError("原生Head/BN冻结边界失败")
        value = managed["session_summaries"][i]
        receipt = read(Path(value["exit_receipt"]))
        if receipt["session_active_seconds"] > receipt["session_limit_seconds"]:
            raise ValueError("真实保存/退出超过会话上限")
        checks[str(i + 1)] = dict(
            optimizer_exact=True,
            scheduler_exact=True,
            rng_exact=True,
            partial_gradients=got["accumulated"],
            session_seconds=receipt["session_active_seconds"],
        )
    for key in ["optimizer", "scheduler", "rng"]:
        if not exact_tree(expected["final"][key], final[key]):
            raise ValueError("最终checkpoint完整状态偏离参考: " + key)
    ends = [
        json.loads(line)
        for line in (run / "control.jsonl").read_text().splitlines()
        if json.loads(line).get("event") == "session_end"
    ]
    if len(ends) != 3 or any(e["offline_network_attempts"] for e in ends):
        raise ValueError("离线/会话事件验收失败")
    stop_durations = [v["session_active_seconds"] for v in managed["session_summaries"]]
    max_micro = max(r["microstep_wall_seconds"] for r in records)
    # 停止会话总时长本身含初始化和等待，作为保存/退出更保守的实测上界。
    required_margin = max(600, 2 * (max_micro + max(stop_durations)))
    if validate_policy(V3_POLICY)["stop_save_margin_seconds"] < required_margin:
        raise ValueError("V3保存预留小于实测保守界，需新版本协议及重验")
    result = dict(
        status="PASS",
        frames=32,
        stage_updates=8,
        parameter_max_abs=delta,
        parameter_relative_l2=relative,
        loss_max_abs=loss,
        gradient_max_abs=max(a for a, _ in grad_deltas),
        gradient_max_relative_l2=max(r for _, r in grad_deltas),
        sessions=checks,
        sample_order_exact=True,
        optimizer_exact=True,
        scheduler_exact=True,
        rng_exact=True,
        partial_gradient_exact=True,
        bn_and_head_frozen=True,
        required_margin_seconds=required_margin,
        max_microstep_seconds=max_micro,
        policy_sha256=sha(V3_POLICY),
        source_sha256=source_manifest(),
        formal_updates=0,
    )
    write(Path(config["evidence_dir"]) / "manager_resume_v3.json", result)
    print("V3_COMPLETE_STATE_COMPARISON_PASS", json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "control", "compare"])
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT)
    parser.add_argument("--prior", type=Path, default=PRIOR)
    args = parser.parse_args()
    if not args.preflight.resolve().is_relative_to((CACHE_ROOT / "runs").resolve()):
        raise ValueError("所有大型验收产物必须在G盘")
    if args.phase == "prepare":
        prepare(args.preflight, args.prior)
    elif args.phase == "control":
        control(args.preflight / "acceptance_config.json")
    else:
        compare(args.preflight / "acceptance_config.json")

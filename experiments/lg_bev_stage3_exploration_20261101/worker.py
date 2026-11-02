"""独立GPU工作函数；只有CLI完成授权检查后才导入和调用。"""

from __future__ import annotations

import math
import random
import resource
import signal
import time

import numpy as np
import torch

from core import (
    freeze,
    fresh_optimizer,
    load_adapter,
    read,
    require,
    seed_cpu,
    selected_rows,
    sha,
    tensor_sha,
)
from lifecycle import save_checkpoint, saved_end
from numerics import POLICY, compare_exact, require_comparison


def cpu_copy(value):
    """复制可训练状态，冻结原作者权重不写入新Checkpoint。"""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return value


def rng_state():
    """仅在已授权GPU工作进程中读取CUDA RNG。"""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(value):
    """只恢复本探索实验自身的状态，不读取Stage A历史RNG。"""
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["cpu"])
    torch.cuda.set_rng_state_all(value["cuda"])


def preflight_rows(rows):
    """固定32帧跨三个场景，不计入主训练预算。"""
    selected = rows[:12] + rows[32:42] + rows[64:74]
    require(
        len(selected) == 32 and len({r["scene_token"] for r in selected}) == 3,
        "预检清单不是32帧三个场景",
    )
    return selected


def validate_training_state(state, tokens, budget, target, frozen):
    """纯CPU核查新实验自己的完整更新边界，不接受旧正式格式。"""
    require(state["frozen_sha256"] == frozen, "冻结主干或文本权重改变")
    require(state["tokens"] == tokens and state["budget"] == budget, "样本或预算变化")
    require(state["accumulated"] == 0, "仅从完整优化器边界继续")
    updates, cursor, epoch = (state[k] for k in ("updates", "cursor", "epoch"))
    require(all(type(v) is int for v in (updates, cursor, epoch)), "训练游标类型错误")
    require(
        0 <= updates < target and 0 <= cursor <= len(tokens) and epoch >= 0,
        "继续游标越界或已经完成",
    )
    require(epoch * len(tokens) + cursor == 4 * updates, "帧数和更新数错配")
    require(state["scheduler"]["last_epoch"] == updates, "scheduler步数错配")
    optimizer = state["optimizer"]["state"]
    require(
        (len(optimizer) == len(state["adapter"])) if updates else not optimizer,
        "优化器参数集合不完整",
    )
    for value in state["adapter"].values():
        require(
            value.dtype == torch.float32 and torch.isfinite(value).all(),
            "Adapter状态无效",
        )
    for value in optimizer.values():
        require(int(value["step"]) == updates, "优化器步数错配")
        require(
            all(torch.isfinite(value[k]).all() for k in ("exp_avg", "exp_avg_sq")),
            "优化器状态含非有限值",
        )


def probe_gradients(runtime, adapter, token, epoch, index):
    """检查det/seg到up与language的真实梯度，不以总Loss梯度冒充。"""
    from scripts.stage3.lg_bev_data import geometry
    from scripts.stage3.lg_bev_engineering_train import choose_query, encode_texts
    from scripts.stage3.lg_bev_engineering_loss import objective

    label = runtime.validate_label(token)
    query = choose_query(label, epoch, index)
    bev = runtime.get(token).cuda()
    texts = encode_texts(runtime.encoder, [query["text"], query["wrong_text"]])
    position = geometry(label["frame"], device="cuda")
    correct = adapter(bev, texts[query["text"]], position)
    wrong = adapter(bev, texts[query["wrong_text"]], position)
    correct["source_bev"] = bev
    _, parts = objective(runtime.model, correct, wrong, label, query, "B")
    named = list(adapter.named_parameters())
    result = {}
    for key in ("detection", "segmentation", "attention"):
        gradients = torch.autograd.grad(
            parts[key], [p for _, p in named], retain_graph=True, allow_unused=True
        )
        require(
            all(g is None or torch.isfinite(g).all() for g in gradients),
            "任务梯度非有限",
        )
        result[key] = {
            prefix: math.sqrt(
                sum(
                    float(g.detach().square().sum())
                    for (name, _), g in zip(named, gradients)
                    if g is not None and name.startswith(prefix)
                )
            )
            for prefix in ("up.", "language.", "visual.", "key.", "mask_refine.")
        }
        result[key].update(
            loss=float(parts[key].detach()),
            total_l2=math.sqrt(
                sum(
                    float(g.detach().square().sum()) for g in gradients if g is not None
                )
            ),
        )
    return result


def boundary_witness(runtime, adapter, trainer):
    """只读下一帧Loss/全参数梯度见证，前后还原RNG且不写入.grad。"""
    from scripts.stage3.lg_bev_data import geometry
    from scripts.stage3.lg_bev_engineering_train import choose_query, encode_texts
    from scripts.stage3.lg_bev_engineering_loss import objective

    before = rng_state()
    try:
        token = trainer.tokens[trainer.cursor]
        label = runtime.validate_label(token)
        query = choose_query(label, trainer.epoch, trainer.cursor)
        bev = runtime.get(token).cuda()
        texts = encode_texts(runtime.encoder, [query["text"], query["wrong_text"]])
        position = geometry(label["frame"], device="cuda")
        correct = adapter(bev, texts[query["text"]], position)
        wrong = adapter(bev, texts[query["wrong_text"]], position)
        correct["source_bev"] = bev
        total, parts = objective(runtime.model, correct, wrong, label, query, "B")
        named = list(adapter.named_parameters())
        gradients = torch.autograd.grad(
            total, [p for _, p in named], allow_unused=False
        )
        require(all(torch.isfinite(g).all() for g in gradients), "边界见证梯度非有限")
        require(all(p.grad is None for _, p in named), "见证不得写入参数.grad")
        return {
            "token": token,
            "cursor": trainer.cursor,
            "updates": trainer.updates,
            "instruction_id": query["instruction_id"],
            "policy": POLICY,
            "inputs_sha256": tensor_sha(
                {
                    "bev": bev,
                    "position": position,
                    "correct": texts[query["text"]],
                    "wrong": texts[query["wrong_text"]],
                }
            ),
            "losses": {k: float(v.detach()) for k, v in parts.items()},
            "gradients": {
                name: g.detach().cpu().clone() for (name, _), g in zip(named, gradients)
            },
        }
    finally:
        restore_rng(before)


def train(
    config,
    store,
    run_id,
    session_id,
    expected,
    started,
    *,
    previous=None,
    preflight=False,
):
    """复用封存数学微步，自行管理预算、持久化、暂停及新实验继续。"""
    from backend import Runtime, TrainLabels
    from scripts.stage3.lg_bev_engineering_train import Trainer, frozen_check
    from scripts.stage3.lg_bev_engineering_determinism import stable_native_backward
    from scripts.stage3.lg_bev_native import dual_head
    from scripts.stage3.lg_bev_data import geometry
    from scripts.stage3.lg_bev_engineering_train import encode_texts, choose_query
    from evaluation import head_equal

    runtime = Runtime(config, store)
    rows = selected_rows(config, "train")
    if preflight:
        rows = preflight_rows(rows)
    tokens = [r["sample_token"] for r in rows]
    seed_cpu(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    adapter = load_adapter(config).cuda()
    trainer = Trainer(
        runtime.model,
        runtime.encoder,
        TrainLabels(runtime),
        tokens,
        runtime.get,
        "B",
        config["updates"],
        expected["config_sha256"],
        adapter=adapter,
    )
    trainer.optimizer, trainer.scheduler = fresh_optimizer(adapter, config["updates"])
    target = 8 if preflight else config["updates"]
    evidence = {
        "initial_gradient": None,
        "after_update_gradient": None,
        "identity_frames": [],
        "raw_replay_frames": [],
        "formal_training": False,
        "gradient_checks": [],
        "numeric_policy": POLICY,
    }
    if previous is not None:
        validate_training_state(
            previous, tokens, config["updates"], target, trainer.baseline
        )
        adapter.load_state_dict(previous["adapter"], strict=True)
        trainer.optimizer.load_state_dict(previous["optimizer"])
        trainer.scheduler.load_state_dict(previous["scheduler"])
        trainer.cursor, trainer.epoch, trainer.updates = (
            previous[k] for k in ("cursor", "epoch", "updates")
        )
        evidence = previous["preflight_evidence"]
        restore_rng(previous["rng"])
        restored = cpu_copy(
            {
                "adapter": adapter.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "scheduler": trainer.scheduler.state_dict(),
                "rng": rng_state(),
            }
        )
        state_check = compare_exact({k: previous[k] for k in restored}, restored)
        store.json(
            f"{run_id}/sessions/{session_id}/restore_state_comparison.json", state_check
        )
        require_comparison(state_check)
    stop_requested = [False]
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, lambda *_: stop_requested.__setitem__(0, True))

    def stop():
        return (
            stop_requested[0]
            or time.monotonic() - started >= 26400
            or store.path(f"{run_id}/sessions/{session_id}/stop.request.json").exists()
        )

    def snapshot():
        frozen_check(runtime.model, runtime.encoder, trainer.baseline)
        require(trainer.accumulated == 0, "禁止丢失部分梯度")
        return cpu_copy(
            {
                "identity": expected,
                "adapter": adapter.state_dict(),
                "optimizer": trainer.optimizer.state_dict(),
                "scheduler": trainer.scheduler.state_dict(),
                "rng": rng_state(),
                "cursor": trainer.cursor,
                "epoch": trainer.epoch,
                "updates": trainer.updates,
                "accumulated": 0,
                "tokens": tokens,
                "budget": config["updates"],
                "preflight_evidence": evidence,
                "source_a9562": config["assets"]["a9562"],
                "frozen_sha256": trainer.baseline,
            }
        )

    with stable_native_backward():
        if preflight and previous is not None:
            require(previous.get("boundary_witness") is not None, "缺少跨会话数值见证")
            actual = boundary_witness(runtime, adapter, trainer)
            comparison = compare_exact(previous["boundary_witness"], actual)
            store.json(
                f"{run_id}/sessions/{session_id}/boundary_comparison.json", comparison
            )
            require_comparison(comparison)
        while trainer.updates < target:
            if (
                preflight
                and previous is None
                and trainer.updates == config["preflight_pause_updates"]
            ):
                break
            if stop() and trainer.accumulated == 0:
                break
            if trainer.cursor == len(tokens):
                require(trainer.accumulated == 0, "epoch尾残留梯度")
                trainer.cursor = 0
                trainer.epoch += 1
            token = tokens[trainer.cursor]
            if preflight and trainer.cursor in (0, 12, 22):
                label = runtime.validate_label(token)
                bev = runtime.get(token).cuda()
                query = choose_query(label, 0, trainer.cursor)
                text = encode_texts(runtime.encoder, [query["text"]])[query["text"]]
                frozen_a = freeze(load_adapter(config)).cuda()
                with torch.no_grad():
                    native = dual_head(runtime.model, bev)
                    after = dual_head(
                        runtime.model,
                        frozen_a(bev, text, geometry(label["frame"], device="cuda"))[
                            "enhanced_bev"
                        ],
                    )
                    require(head_equal(native, after), "零up双Head不一致")
                    native_cpu = {
                        k: native[k].detach().cpu() for k in runtime.current_prediction
                    }
                    require(
                        head_equal(native_cpu, runtime.current_prediction),
                        "原生history重放不一致",
                    )
                evidence["identity_frames"].append(token)
                evidence["raw_replay_frames"].append(token)
                del frozen_a
            if preflight and trainer.accumulated == 0:
                gradients = probe_gradients(
                    runtime, adapter, token, trainer.epoch, trainer.cursor
                )
                audit = {
                    "before_update": trainer.updates,
                    "token": token,
                    "branches": gradients,
                }
                store.json(
                    f"{run_id}/sessions/{session_id}/gradients/{trainer.updates:02d}.json",
                    audit,
                )
                evidence["gradient_checks"].append(audit)
                for task in ("detection", "segmentation", "attention"):
                    require(
                        math.isfinite(gradients[task]["loss"])
                        and gradients[task]["total_l2"] > 0,
                        "任务Loss或总梯度异常",
                    )
                for task in ("detection", "segmentation"):
                    require(gradients[task]["up."] > 0, "任务未到达up")
                    require(
                        (
                            (gradients[task]["language."] == 0)
                            if trainer.updates == 0
                            else gradients[task]["language."] > 0
                        ),
                        "任务到language梯度异常",
                    )
                if trainer.updates in (0, 1):
                    field = (
                        "initial_gradient"
                        if trainer.updates == 0
                        else "after_update_gradient"
                    )
                    evidence[field] = gradients
            before_parameters = (
                cpu_copy(adapter.state_dict())
                if preflight and trainer.accumulated == 3
                else None
            )
            microstep_start = time.monotonic()
            record = trainer.microstep()
            record["wall_seconds"] = time.monotonic() - microstep_start
            record["cpu_peak_rss_bytes"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )
            record["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
            record["cuda_reserved_bytes"] = torch.cuda.memory_reserved()
            if before_parameters is not None:
                record["parameter_delta_by_tensor"] = {
                    name: float(
                        (p.detach().cpu() - before_parameters[name]).double().norm()
                    )
                    for name, p in adapter.named_parameters()
                }
                require(
                    record["parameter_delta_by_tensor"]["up.weight"] > 0,
                    "up.weight未实际更新",
                )
            step = trainer.epoch * len(tokens) + trainer.cursor
            store.json(f"{run_id}/sessions/{session_id}/steps/{step:06d}.json", record)
            if trainer.accumulated == 0 and (preflight or trainer.updates % 64 == 0):
                save_checkpoint(
                    store,
                    run_id,
                    session_id,
                    snapshot(),
                    f"periodic_{trainer.updates:06d}",
                )
        witness = None
        if preflight and trainer.updates == 4:
            witness = boundary_witness(runtime, adapter, trainer)
    outcome = "COMPLETED" if trainer.updates == target else "PAUSED"
    if preflight and outcome == "COMPLETED":
        for task in ("detection", "segmentation"):
            require(evidence["initial_gradient"][task]["up."] > 0, "首步任务未到达up")
            require(
                evidence["initial_gradient"][task]["language."] == 0,
                "零up任务梯度上游异常",
            )
            require(
                evidence["after_update_gradient"][task]["language."] > 0,
                "up更新后任务未到达language",
            )
        require(len(set(evidence["identity_frames"])) == 3, "预检缺少跨场景恒等验证")
        outcome = "PREFLIGHT_PASS"
    state = snapshot()
    state["boundary_witness"] = witness
    relative, digest = save_checkpoint(store, run_id, session_id, state, "end")
    saved_end(
        store,
        run_id,
        session_id,
        expected,
        relative,
        digest,
        outcome,
        updates=trainer.updates,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        native_raw_frames=runtime.native.raw_calls,
        elapsed_before_checkpoint_exit_seconds=time.monotonic() - started,
        preflight_evidence=evidence,
    )


def evaluate(
    config, store, run_id, session_id, expected, started, trained, *, previous=None
):
    """开发GT仅由已授权的独立评价进程访问；暂停保存累计器而非重算或挑结果。"""
    from backend import Runtime
    from evaluation import (
        EvaluationState,
        adapters_for_evaluation,
        evaluate_frame,
        ground_truth_coverage,
    )

    runtime = Runtime(config, store, development_authorized=True)
    adapters = adapters_for_evaluation(config, trained)
    rows = selected_rows(config, "development")
    seed_cpu(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    cursor, metrics = 0, EvaluationState()
    if previous:
        require(
            previous["trained_adapter_sha256"] == trained["adapter_digest"],
            "评价模型改变",
        )
        cursor, metrics = previous["cursor"], previous["metrics"]
    stop_requested = [False]
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, lambda *_: stop_requested.__setitem__(0, True))

    def stop():
        return (
            stop_requested[0]
            or time.monotonic() - started >= 26400
            or store.path(f"{run_id}/sessions/{session_id}/stop.request.json").exists()
        )

    coverage_path = store.path(f"{run_id}/gt_coverage.json")
    if coverage_path.exists():
        coverage = read(coverage_path)
        require(
            coverage["identity"] == expected
            and coverage["trained_adapter_sha256"] == trained["adapter_digest"],
            "GT覆盖协议或模型身份变化",
        )
    else:
        require(cursor == 0, "评价已计分却缺少事前GT覆盖记录")
        coverage = ground_truth_coverage(runtime, rows, stop)
        if coverage is not None:
            store.json(
                f"{run_id}/gt_coverage.json",
                {
                    "identity": expected,
                    "trained_adapter_sha256": trained["adapter_digest"],
                    **coverage,
                },
            )
    coverage_sha = sha(coverage_path) if coverage_path.exists() else None
    if previous and previous["cursor"]:
        require(previous["coverage_sha256"] == coverage_sha, "事前GT覆盖记录变化")
    while coverage_sha is not None and cursor < len(rows):
        if stop():
            break
        for record in evaluate_frame(
            runtime, adapters, rows[cursor]["sample_token"], cursor
        ):
            metrics.add(record)
        cursor += 1
    state = {
        "identity": expected,
        "cursor": cursor,
        "metrics": metrics,
        "trained_adapter_sha256": trained["adapter_digest"],
        "coverage_sha256": coverage_sha,
    }
    relative, digest = save_checkpoint(store, run_id, session_id, state, "end")
    outcome = "COMPLETED" if cursor == len(rows) else "PAUSED"
    if outcome == "COMPLETED":
        store.json(f"{run_id}/metrics.json", metrics.result())
    saved_end(
        store,
        run_id,
        session_id,
        expected,
        relative,
        digest,
        outcome,
        evaluated_frames=cursor,
    )

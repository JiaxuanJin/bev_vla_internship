"""正式全量装配：有界场景加载、全量标签准入、A/B状态机和终点评价锁。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import uuid

import numpy as np
import torch

from scripts.stage3.lg_bev_common import ROOT, read, sha, write
from scripts.stage3.lg_bev_engineering_cache import CACHE_ROOT, RollingCache, contained
from scripts.stage3.lg_bev_engineering_data import (
    FORMAL,
    LabelStore,
    SceneData,
    formal_frames,
    prepare_labels,
    masks,
)
from scripts.stage3.lg_bev_engineering_native import NativeRuntime, NativeCacheProvider
from scripts.stage3.lg_bev_engineering_train import Trainer, choose_query, encode_texts
from scripts.stage3.lg_bev_data import geometry
from scripts.stage3.lg_bev_supervision_space import AttentionSupervisionSpace
from scripts.stage3.lg_bev_native import module_sha


class SceneProvider:
    """正式运行只在RAM保留当前场景；跨epoch可从G盘的原始元数据索引重建。"""

    def __init__(
        self,
        cache: RollingCache,
        runtime: NativeRuntime,
        rows: list[dict],
        *,
        split: str = "train",
        labels: LabelStore | None = None,
        source_provenance: dict | None = None,
    ) -> None:
        self.cache, self.runtime, self.split = cache, runtime, split
        self.groups = {}
        self.scenes = {}
        self.provider = None
        for row in rows:
            self.groups.setdefault(row["scene_token"], []).append(row)
            self.scenes[row["sample_token"]] = row["scene_token"]
        self.active = None
        self.labels = labels
        self.source_provenance = source_provenance
        # NTFS全量反复写入没有复用收益：固定完整热场景落盘，其他场景只持有一个RAM块。
        # 只影响缓存准入，不影响任何帧、loss、预算、原生history或训练顺序。
        available = min(10_000_000_000, max(0, cache.capacity - 5_000_000_000))
        self.hot_scenes = set()
        used = 0
        for scene in sorted(
            self.groups,
            key=lambda s: hashlib.sha256(("hotset-20261027:" + s).encode()).hexdigest(),
        ):
            size = len(self.groups[scene]) * 23_040_000 + 2**20
            if used + size <= available:
                self.hot_scenes.add(scene)
                used += size

    def get(self, token: str) -> torch.Tensor:
        """命中/重建统一接口；原生history重建不影响训练RNG。"""
        scene = self.scenes[token]
        if self.active != scene:
            self.provider = None
            data = SceneData(self.groups[scene], allowed_split=self.split)
            expected = None
            if self.source_provenance is not None:
                expected = {scene: self.source_provenance[scene]}
            elif self.labels is not None:
                sources, block, previous = {}, {}, None
                for row in self.groups[scene]:
                    item = self.labels.records[row["sample_token"]]
                    path = Path(item["label_path"])
                    if path != previous:
                        if sha(path) != item["label_sha256"]:
                            raise ValueError("GT来源块损坏")
                        block = torch.load(path, map_location="cpu", weights_only=False)
                        previous = path
                    # 只白名单读取原始来源SHA；不解包/读取任何实例或语义mask。
                    sources.update(block[row["sample_token"]]["source_hashes"])
                expected = {scene: sources}
            self.provider = NativeCacheProvider(
                self.cache,
                self.runtime,
                {scene: data},
                admit_disk=scene in self.hot_scenes,
                expected_sources=expected,
            )
            self.active = scene
        return self.provider.get(token)

    def peek(self, token: str) -> torch.Tensor | None:
        """不切换scene、不生成新history的RAM探针。"""
        return self.provider.peek(token) if self.provider is not None else None


def ordered_tokens(rows: list[dict]) -> list[str]:
    """固定整场景哈希顺序，场景内时间顺序；每epoch完整覆盖全部帧。"""
    return [
        r["sample_token"]
        for r in sorted(
            rows,
            key=lambda r: (
                hashlib.sha256(
                    ("formal_order_20261027:" + r["scene_token"]).encode()
                ).hexdigest(),
                r["timestamp"],
            ),
        )
    ]


def build_full_data(run_id: str) -> Path:
    """显式调用后构建完整train真实标签，不打开开发/最终集，不启动优化器。"""
    cache = RollingCache()
    run = contained(cache.root, "runs/" + run_id)
    scope = dict(
        owner="LG-BEV_FORMAL_TRAIN_GT",
        split="train",
        formal_contract_sha256=sha(FORMAL / "formal_execution_contract.json"),
        builder_sha256=sha(ROOT / "scripts/stage3/lg_bev_engineering_data.py"),
    )
    if run.exists():
        if read(run / "scope.json") != scope:
            raise ValueError("已存在数据run的归属或builder版本不同")
        if (run / "data_report.json").exists():
            return run / "data_report.json"
    else:
        run.mkdir(parents=True, exist_ok=False)
        write(run / "scope.json", scope)
    prepare_labels(cache, run, formal_frames())
    return run / "data_report.json"


def warmup_probe(trainer: Trainer) -> dict:
    """固定128训练帧正目标宏IoU、反事实logit差及空目标激活门槛。"""
    ious = {"map": [], "object": []}
    gaps = []
    empty = []
    tokens = sorted(
        trainer.tokens,
        key=lambda t: hashlib.sha256(
            ("warmup-probe-20261027:" + t).encode()
        ).hexdigest(),
    )[:128]
    probe_start = time.perf_counter()
    budget = getattr(trainer, "total_budget_seconds", 86400)
    for i, token in enumerate(tokens):
        if getattr(trainer, "session", None) is not None and trainer.should_stop():
            return dict(status="PAUSED", inspected_frames=i)
        if (
            getattr(trainer, "session", None) is None
            and budget is not None
            and trainer.elapsed_wall() + time.perf_counter() - probe_start >= budget
        ):
            return dict(status="INCOMPLETE_TIME_LIMIT", inspected_frames=i)
        label = trainer.labels.get(token)
        q = choose_query(label, 0, i)
        text = encode_texts(trainer.encoder, [q["text"], q["wrong_text"]])
        bev = trainer.get_bev(token).cuda()
        position = geometry(label["frame"], device="cuda")
        with torch.no_grad():
            c = trainer.adapter(bev, text[q["text"]], position)
            w = trainer.adapter(bev, text[q["wrong_text"]], position)
            space = AttentionSupervisionSpace.from_calibration(
                q["task"], label["frame"]
            )
            logits = space.project(c["attention_logits"])
            wrong = space.project(w["attention_logits"])
            target = masks(label, q, "cuda")["target"].bool()
            active = logits.sigmoid() >= 0.5
            if target.any():
                ious[q["task"]].append(
                    float((active & target).sum() / (active | target).sum())
                )
                gaps.append(float((logits - wrong)[target].mean()))
            else:
                empty.append(float(active.float().mean()))
    scores = {k: float(np.mean(v)) if v else None for k, v in ious.items()}
    gap = float(np.mean(gaps)) if gaps else None
    negative = float(np.mean(empty)) if empty else None
    passed = (
        all(v is not None and v >= 0.15 for v in scores.values())
        and gap is not None
        and gap >= 0.1
        and negative is not None
        and negative <= 0.05
    )
    return dict(
        status="PASS" if passed else "GROUNDING_NOT_READY",
        tokens=tokens,
        positive_macro_IoU=scores,
        target_correct_wrong_logit_gap=gap,
        empty_active_fraction=negative,
    )


def completed_warmup_gate(trainer: Trainer, run: Path) -> dict:
    """绑定精确A端点复用已完成探针；不覆盖/重选门槛，恢复计入探针耗时。"""
    identity = dict(
        protocol_sha256=trainer.protocol_sha256,
        adapter_sha256=module_sha(trainer.adapter),
        updates=trainer.updates,
    )
    path = run / "warmup_gate.json"
    if path.exists():
        gate = read(path)
        if gate.get("identity") != identity:
            raise ValueError("warmup gate不属于同一A端点，拒绝复用")
        trainer.wall_seconds = max(trainer.wall_seconds, gate["elapsed_wall_seconds"])
        if getattr(trainer, "session", None) is not None:
            trainer.session.previous = max(
                trainer.session.previous, gate["elapsed_wall_seconds"]
            )
        return gate
    start = time.perf_counter()
    try:
        gate = warmup_probe(trainer)
    finally:
        trainer.wall_seconds += time.perf_counter() - start
    gate.update(identity=identity, elapsed_wall_seconds=trainer.elapsed_wall())
    if gate["status"] == "PAUSED":
        return gate
    write(path, gate)
    return gate


def full_training(args: argparse.Namespace) -> None:
    """READY和明确full授权同时满足才构建GPU；不执行任何自动性能调参。"""
    if not args.authorize_full_training:
        raise ValueError("未显式授权完整长训")
    if (
        os.environ.get("LG_BEV_DYNAMIC_CUDA", "0") != "0"
        or os.environ.get("LG_BEV_UPSAMPLE_CUDA", "0") != "0"
    ):
        raise ValueError("实验CUDA动态反向未通过数值验收，不得用于正式训练")
    if getattr(args, "cache_gb", None) is not None:
        configured = RollingCache()
        configured.configure_capacity(int(args.cache_gb * 1e9), user_requested=True)
    engineering = read(args.engineering_out / "ready.json")
    if engineering["engineering_32frames"] != "PASS":
        raise ValueError("32帧工程验收未通过")
    resource_gate = read(args.engineering_out / "resource_admission.json")
    daily = getattr(args, "daily_policy", None)
    v3 = False
    if daily:
        from scripts.stage3.lg_bev_daily_state import validate_policy, source_manifest
        from scripts.stage3 import lg_bev_daily as manager
        from scripts.stage3.lg_bev_daily_state import V3_VERSION, V3_RUN_ID

        policy = validate_policy(daily)
        v3 = policy.get("version") == V3_VERSION
        manager.configure_protocol("v3" if v3 else "v2")
        manager.require_launch_authorization()
        if v3 and args.run_id != V3_RUN_ID:
            raise ValueError("V3必须使用独立正式run-id")
        admission = read(args.daily_admission)
        if admission["status"] != "READY" or not all(
            v == "PASS" for v in admission["gates"].values()
        ):
            raise ValueError("48小时正式准入尚未全部通过")
        if not v3 and admission["conservative_active_hours"] > 48:
            raise ValueError("全尺度预检超过48小时，禁止扩预算")
        for path, digest in admission["artifacts"].items():
            if sha(Path(path)) != digest:
                raise ValueError("准入证据已变化")
        if admission["source_sha256"] != source_manifest():
            raise ValueError("源码偏离新准入验收")
    elif not resource_gate["budget_24h_pass"]:
        raise ValueError("当前吞吐未满足24小时保守预算，禁止启动完整长训")
    fast_gate = read(args.engineering_out / "fast_pipeline_equivalence.json")
    if fast_gate["status"] != "PASS":
        raise ValueError("原生等价的image-only数据通路尚未通过32帧验收")
    expected_sources = {
        str(p) for p in (ROOT / "scripts/stage3").glob("lg_bev_engineering_*.py")
    }
    if not daily and set(fast_gate.get("source_sha256", {})) != expected_sources:
        raise ValueError("验收报告缺少完整工程源码SHA封存")
    for path, digest in ({} if daily else fast_gate.get("source_sha256", {})).items():
        if sha(Path(path)) != digest:
            raise ValueError("工程源码已偏离通过验收的版本")
    if args.prepare_data_if_needed and not args.data_report.exists():
        path = build_full_data(args.data_run_id)
        if path != args.data_report:
            raise ValueError("准备后的data-report路径不匹配")
    report = read(args.data_report)
    rows = formal_frames()
    tokens = ordered_tokens(rows)
    if (
        report["status"] != "PASS"
        or {r["sample_token"] for r in report["frames"]} != set(tokens)
        or report["success"] != 19123
    ):
        raise ValueError("完整19123帧真实标签尚未就绪；禁止用32帧报告启动正式预算")
    for item in report["frames"]:
        path = Path(item["label_path"])
        if not path.is_relative_to(CACHE_ROOT):
            raise ValueError("全量标签不得写C/D/E盘")
    # 逐块校验一次，训练中每次换块仍重新校验，禁止静默跳过坏数据。
    for path, digest in {
        r["label_path"]: r["label_sha256"] for r in report["frames"]
    }.items():
        if sha(Path(path)) != digest:
            raise ValueError("全量标签块损坏")
    cache = RollingCache()
    cache.recover()
    run = contained(cache.root, "runs/" + args.run_id)
    protocol = dict(
        formal_contract_sha256=sha(FORMAL / "formal_execution_contract.json"),
        full_data_report_sha256=sha(args.data_report),
        order_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        source_sha256={
            str(p): sha(p)
            for p in sorted((ROOT / "scripts/stage3").glob("lg_bev_engineering_*.py"))
        },
        budget=dict(A=9562, B=19124, total=28686),
        rolling_cache_max=30_000_000_000,
        reserve_g=250_000_000_000,
        full_training_authorized_by_explicit_cli_flag=True,
    )
    if daily:
        protocol.update(
            time_protocol_sha256=sha(daily),
            session_active_seconds=27000,
            daily_source_sha256=source_manifest(),
            source_provenance_sha256=sha(args.source_provenance),
        )
        if v3:
            protocol.update(
                version="LG_BEV_FORMAL_EXECUTION_V3_20261028",
                run_id=args.run_id,
                cumulative_hard_limit_seconds=None,
                stop_save_margin_seconds=policy["stop_save_margin_seconds"],
            )
        else:
            protocol["total_active_seconds"] = 172800
    if args.resume:
        if read(run / "execution.json") != protocol:
            raise ValueError("续训源码/标签/合同与首次启动不一致")
    else:
        if daily and run.exists():
            marker = read(run / "prelaunch.json")
            if marker["owner"] != args.run_id or marker["policy_sha256"] != sha(daily):
                raise ValueError("既有run不是本管理器明确预创建的目录")
        else:
            run.mkdir(parents=True, exist_ok=False)
        write(run / "execution.json", protocol)
    execution_sha = sha(run / "execution.json")
    from scripts.stage2.language_encoder import DistilBERTLanguageEncoder
    from scripts.stage2.spatial_v3_model import InstanceReadout

    runtime = NativeRuntime(fast_images=True)
    labels = LabelStore(report)
    provenance = read(args.source_provenance) if daily else None
    if daily and provenance["data_report_sha256"] != sha(args.data_report):
        raise ValueError("来源SHA索引不属于同一完整GT报告")
    provider = SceneProvider(
        cache,
        runtime,
        rows,
        labels=labels,
        source_provenance=provenance["scenes"] if provenance else None,
    )
    encoder = (
        InstanceReadout(
            DistilBERTLanguageEncoder(
                str(ROOT / "models/distilbert-base-uncased"), local_files_only=True
            )
        )
        .cuda()
        .eval()
    )
    encoder.requires_grad_(False)
    state = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume
        else None
    )
    with (run / "control.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                dict(
                    event="process_start",
                    pid=os.getpid(),
                    run_id=args.run_id,
                    resume=str(args.resume),
                )
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    stage = state["stage"] if state else "A"
    warm = Trainer(
        runtime.model, encoder, labels, tokens, provider.get, "A", 9562, execution_sha
    )

    def attach(trainer):
        if daily:
            trainer.session = args.session_clock
            trainer.total_budget_seconds = None if v3 else 172800
            trainer.progress_callback = args.progress_callback
            trainer.checkpoint_callback = getattr(args, "checkpoint_callback", None)
        return trainer

    attach(warm)
    if stage == "A":
        if args.resume:
            warm.resume(args.resume)
        warm.run(
            log=run
            / f"A_from_{warm.updates:06d}_{warm.cursor:06d}_{uuid.uuid4().hex[:8]}.jsonl",
            checkpoint_dir=run / "checkpoints",
            cache=cache,
        )
        if warm.updates != 9562 or warm.should_stop():
            if daily:
                args.finalize(warm, "STOPPED")
                return
            write(
                run / f"A_incomplete_{uuid.uuid4().hex[:8]}.json",
                dict(
                    status="INCOMPLETE",
                    updates=warm.updates,
                    wall_seconds=warm.elapsed_wall(),
                ),
            )
            return
        if (
            daily
            and args.session_clock.session_seconds - args.session_clock.elapsed() < 3600
        ):
            args.finalize(warm, "STOPPED", reason="WARMUP_GATE_DEFERRED")
            return
        gate = completed_warmup_gate(warm, run)
        if gate["status"] != "PASS":
            if daily:
                args.finalize(
                    warm,
                    "STOPPED" if gate["status"] == "PAUSED" else "FAILED",
                    reason=gate["status"],
                )
            return
    else:
        if read(run / "warmup_gate.json")["status"] != "PASS":
            raise ValueError("B续训缺少A门槛")
    joint = Trainer(
        runtime.model,
        encoder,
        labels,
        tokens,
        provider.get,
        "B",
        19124,
        execution_sha,
        adapter=warm.adapter if stage == "A" else None,
    )
    attach(joint)
    joint.wall_seconds = warm.wall_seconds
    if stage == "B":
        joint.resume(args.resume)
    joint.run(
        log=run
        / f"B_from_{joint.updates:06d}_{joint.cursor:06d}_{uuid.uuid4().hex[:8]}.jsonl",
        checkpoint_dir=run / "checkpoints",
        cache=cache,
    )
    if joint.updates != 19124 or joint.should_stop():
        if daily:
            args.finalize(joint, "STOPPED")
            return
        write(
            run / f"B_incomplete_{uuid.uuid4().hex[:8]}.json",
            dict(
                status="INCOMPLETE",
                updates=joint.updates,
                wall_seconds=joint.elapsed_wall(),
            ),
        )
        return
    terminal = run / "adapter_terminal.pt"
    digest = joint.checkpoint(terminal, cache)
    write(
        run / "terminal.json",
        dict(
            status="FIXED_TERMINAL",
            checkpoint=str(terminal),
            checkpoint_sha256=digest,
            A_updates=9562,
            B_updates=19124,
            total_updates=28686,
            execution_sha256=execution_sha,
            development_evaluated=False,
            cold_final_opened=False,
        ),
    )
    if daily:
        args.finalize(joint, "COMPLETED", reason="FIXED_28686_UPDATES_COMPLETE")


def evaluate_terminal(args: argparse.Namespace) -> None:
    """正式开发只允许固定终点一次；Cold Final无CLI入口。"""
    terminal = read(args.training_run / "terminal.json")
    if (
        terminal["total_updates"] != 28686
        or terminal["status"] != "FIXED_TERMINAL"
        or sha(args.checkpoint) != terminal["checkpoint_sha256"]
    ):
        raise ValueError("不是完整固定预算终点，禁止打开开发集")
    marker = args.training_run / "development_opened.json"
    write(
        marker,
        dict(
            checkpoint_sha256=terminal["checkpoint_sha256"],
            policy="ONE_FIXED_TERMINAL_EVALUATION_NO_SELECTION",
        ),
    )
    cache = RollingCache()
    run = args.training_run / "development"
    run.mkdir(exist_ok=False)
    rows = formal_frames("development")
    report = prepare_labels(cache, run, rows, split="development")
    from scripts.stage2.language_encoder import DistilBERTLanguageEncoder
    from scripts.stage2.spatial_v3_model import InstanceReadout
    from scripts.stage3.lg_bev_prototype import LanguageGroundedBEVAdapter
    from scripts.stage3.lg_bev_engineering_evaluate import functional_audit
    from scripts.stage3.lg_bev_engineering_metrics import MetricSpool

    runtime = NativeRuntime()
    provider = SceneProvider(cache, runtime, rows, split="development")
    encoder = (
        InstanceReadout(
            DistilBERTLanguageEncoder(
                str(ROOT / "models/distilbert-base-uncased"), local_files_only=True
            )
        )
        .cuda()
        .eval()
    )
    encoder.requires_grad_(False)
    adapter = LanguageGroundedBEVAdapter(hidden_channels=24, relative_bound=0.05).cuda()
    adapter.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
    )
    spool = MetricSpool(run / "ap_events.sqlite3", cache)
    result = functional_audit(
        runtime.model,
        adapter,
        encoder,
        LabelStore(report),
        ordered_tokens(rows),
        provider.get,
        all_queries=True,
        spool=spool,
        bootstrap=True,
    )
    result.update(
        status="FIXED_DEVELOPMENT_TERMINAL_EVALUATION",
        formal_development_or_final_evaluations=1,
    )
    write(run / "metrics.json", result)
    branches = result["branches"]
    correct = branches["correct_text"]
    original = branches["original"]

    def metric(branch, key):
        return (
            branches[branch][key]["mAP"]
            if key.endswith("mAP")
            else branches[branch][key]
        )

    related = correct["related_conditional_mAP"]["mAP"]
    base = original["related_conditional_mAP"]["mAP"]
    seg = correct["related_macro_IoU"]
    base_seg = original["related_macro_IoU"]
    overall = correct["overall_10class_mAP"]["mAP"]
    base_overall = original["overall_10class_mAP"]["mAP"]
    miou = correct["overall_3class_mIoU"]["mIoU"]
    base_miou = original["overall_3class_mIoU"]["mIoU"]
    evaluable = all(
        v is not None
        for v in (related, base, seg, base_seg, overall, base_overall, miou, base_miou)
    )
    passed = (
        evaluable
        and related - base >= 0.02
        and seg - base_seg >= 0.015
        and overall - base_overall >= -0.02
        and miou - base_miou >= -0.02
    )
    if passed:
        passed = all(
            metric("correct_text", "related_conditional_mAP")
            > metric(b, "related_conditional_mAP")
            and metric("correct_text", "related_macro_IoU")
            > metric(b, "related_macro_IoU")
            for b in ("wrong_text", "zero_text")
        )
    write(
        run / "decision.json",
        dict(
            status="PASS" if passed else "FAIL_OR_NOT_EVALUABLE",
            expand=False,
            cold_final_opened=False,
            automatic_final_authorization=False,
            related_detection_delta_pp=(
                (related - base) * 100
                if related is not None and base is not None
                else None
            ),
            related_seg_delta_pp=(
                (seg - base_seg) * 100
                if seg is not None and base_seg is not None
                else None
            ),
        ),
    )


def main() -> None:
    """完整标签准备或安全停止既有本项目进程；不隐式开启长训。"""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-train", action="store_true")
    group.add_argument("--stop-run", type=Path)
    parser.add_argument("--run-id", default="formal_gt_20261027_01")
    args = parser.parse_args()
    if args.prepare_train:
        print(build_full_data(args.run_id))
    else:
        path = args.stop_run.resolve()
        if not path.is_relative_to(CACHE_ROOT):
            raise ValueError("停止目标必须为本项目G盘run")
        events = [
            json.loads(line)
            for line in (path / "control.jsonl").read_text().splitlines()
        ]
        event = next(e for e in reversed(events) if e["event"] == "process_start")
        pid = event["pid"]
        command = Path(f"/proc/{pid}/cmdline").read_bytes()
        if (
            not (
                b"lg_bev_engineering_train" in command
                or b"scripts.stage3.lg_bev_daily" in command
            )
            or event["run_id"].encode() not in command
        ):
            raise ValueError("PID已失效或不属于该run")
        os.kill(pid, signal.SIGUSR1)
        print("STOP_REQUESTED_AT_SAFE_MICROBATCH_BOUNDARY", pid)


if __name__ == "__main__":
    main()

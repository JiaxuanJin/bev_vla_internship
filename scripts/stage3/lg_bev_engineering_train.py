"""可中断的正式A/B训练状态机；全量启动需独立授权和实际READY门槛。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from scripts.stage3.lg_bev_common import ROOT, sha
from scripts.stage3.lg_bev_data import geometry
from scripts.stage3.lg_bev_engineering_cache import RollingCache, atomic_save
from scripts.stage3.lg_bev_engineering_data import LabelStore
from scripts.stage3.lg_bev_engineering_loss import objective
from scripts.stage3.lg_bev_native import module_sha
from scripts.stage3.lg_bev_prototype import LanguageGroundedBEVAdapter


def choose_query(label: dict, epoch: int, index: int) -> dict:
    """每帧每epoch一条，地图/对象与25%空指令固定轮换；不丢弃真实空帧。"""
    task = "map" if (epoch + index) % 2 == 0 else "object"
    pool = [q for q in label["queries"] if q["task"] == task]
    want_empty = (epoch + index) % 4 == 3
    stratified = [q for q in pool if bool(not q["selected"]) == want_empty]
    if not pool:
        raise ValueError("帧缺少完整任务指令表，禁止静默换帧")
    pool = stratified or pool
    return sorted(
        pool,
        key=lambda q: hashlib.sha256(
            f'20261027:{epoch}:{q["instruction_id"]}'.encode()
        ).hexdigest(),
    )[0]


def encode_texts(encoder: nn.Module, texts: list[str]) -> dict[str, Tensor]:
    """冻结本地文本编码；缓存只在RAM且按小批更新，无模型下载。"""
    unique = sorted(set(texts))
    result = {}
    for i in range(0, len(unique)):
        batch = unique[i : i + 1]
        with torch.no_grad():
            values = encoder(batch, "cuda")
        result.update(
            {text: value[None].detach() for text, value in zip(batch, values)}
        )
    return result


def rng_state() -> dict:
    """保存Python/NumPy/CPU/CUDA全部随机状态。"""
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def restore_rng(state: dict) -> None:
    """恢复状态不再次重设初始seed。"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def frozen_check(
    model: nn.Module | None, encoder: nn.Module, baseline: tuple[str | None, str]
) -> None:
    """参数与BN buffer全状态摘要、requires_grad及梯度不变量。"""
    for module, expected in zip((model, encoder), baseline):
        if module is None:
            continue
        if module.training or any(
            p.requires_grad or p.grad is not None for p in module.parameters()
        ):
            raise RuntimeError("冻结模块进入训练态或出现梯度")
        if module_sha(module) != expected:
            raise RuntimeError("冻结状态发生变化")


class Trainer:
    """状态包含未完成梯度累积；恢复不依赖任何可淘汰cache分片。

    Args:
        get_bev: 返回严格FP32原生BEV的cache或完整参考提供器。
        stage: A冻结up，B对同一enhanced BEV进行原生联合监督。
        budget_updates: 当前阶段固定总数，用于scheduler及越界拒绝。
    """

    def __init__(
        self,
        model: nn.Module | None,
        encoder: nn.Module,
        labels: LabelStore,
        tokens: list[str],
        get_bev: Callable[[str], Tensor],
        stage: str,
        budget_updates: int,
        protocol_sha256: str,
        *,
        adapter: nn.Module | None = None,
        retain_audit_tensors: bool = False,
    ) -> None:
        if stage not in ("A", "B") or not tokens or budget_updates <= 0:
            raise ValueError("无效训练预算")
        self.model, self.encoder, self.labels = model, encoder, labels
        self.tokens, self.get_bev, self.stage = tokens, get_bev, stage
        self.budget_updates, self.protocol_sha256 = budget_updates, protocol_sha256
        if adapter is None:
            torch.manual_seed(20261027)
            torch.cuda.manual_seed_all(20261027)
            adapter = LanguageGroundedBEVAdapter(
                hidden_channels=24, relative_bound=0.05
            ).cuda()
        self.adapter = adapter.train()
        self.adapter.up.requires_grad_(stage == "B")
        self.optimizer = torch.optim.AdamW(
            [p for p in self.adapter.parameters() if p.requires_grad],
            lr=3e-4,
            weight_decay=0.01,
        )
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=budget_updates, eta_min=3e-5
            )
            if stage == "B"
            else None
        )
        self.cursor = 0
        self.updates = 0
        self.accumulated = 0
        self.epoch = 0
        self.records = []
        self.update_records = []
        self.text_cache = {}
        self.retain_audit_tensors = retain_audit_tensors
        self.pin_transfer = False
        self.prefetched_token = None
        self.prefetched_bev = None
        self.prefetch_host = None
        self.transfer_stream = None
        self.position_cache = {}
        self.stop_requested = False
        self.wall_seconds = 0.0
        self._run_clock = None
        self.session = None
        self.total_budget_seconds = 86400.0
        self.progress_callback = None
        self.checkpoint_callback = None
        self.baseline = (
            module_sha(model) if model is not None else None,
            module_sha(encoder),
        )

    def elapsed_wall(self) -> float:
        """累计活跃wall时间；续训沿用持久值，不包含用户停机等待。"""
        if getattr(self, "session", None) is not None:
            return self.session.total()
        return self.wall_seconds + (
            time.perf_counter() - self._run_clock
            if self._run_clock is not None
            else 0.0
        )

    def should_stop(self) -> bool:
        """同一边界检查手工、会话及累计预算，兼容封存的24小时旧入口。"""
        session = getattr(self, "session", None)
        return (
            self.stop_requested
            or self.elapsed_wall() >= getattr(self, "total_budget_seconds", 86400.0)
            or bool(session is not None and session.reason())
        )

    def checkpoint(self, path: Path, cache: RollingCache | None = None) -> str:
        """原子保存优化器/scheduler、精确cursor、部分梯度和RNG，禁止覆盖。"""
        value = dict(
            stage=self.stage,
            state_dict={
                n: v.detach().cpu() for n, v in self.adapter.state_dict().items()
            },
            gradients={
                n: p.grad.detach().cpu()
                for n, p in self.adapter.named_parameters()
                if p.grad is not None
            },
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict() if self.scheduler else None,
            cursor=self.cursor,
            updates=self.updates,
            accumulated=self.accumulated,
            epoch=self.epoch,
            budget_updates=self.budget_updates,
            protocol_sha256=self.protocol_sha256,
            tokens=self.tokens,
            rng=rng_state(),
            wall_seconds=self.elapsed_wall(),
            baseline=self.baseline,
            cache_dependency="NONE; every native shard regenerates fromscene start",
        )
        if cache is not None:
            with cache.locked():
                cache.reserve_for_artifact(8_000_000)
                atomic_save(path, value)
        else:
            atomic_save(path, value)
        digest = sha(path)
        # 独立原子receipt补记CPU复制/序列化/fsync/校验耗时；旧checkpoint不覆盖。
        receipt = dict(checkpoint_sha256=digest, wall_seconds=self.elapsed_wall())
        receipt_path = path.with_name(path.name + ".receipt.pt")
        if cache is not None:
            with cache.locked():
                cache.reserve_for_artifact(8192)
                atomic_save(receipt_path, receipt)
        else:
            atomic_save(receipt_path, receipt)
        return digest

    def resume(self, path: Path) -> None:
        """严格核对协议、顺序、冻结模块和预算，恢复部分累积而不是重跑前缀更新。"""
        state = torch.load(path, map_location="cpu", weights_only=False)
        for key, value in [
            ("stage", self.stage),
            ("tokens", self.tokens),
            ("budget_updates", self.budget_updates),
            ("protocol_sha256", self.protocol_sha256),
            ("baseline", self.baseline),
        ]:
            if state[key] != value:
                raise ValueError(f"恢复协议不匹配:{key}")
        self.adapter.load_state_dict(state["state_dict"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler:
            self.scheduler.load_state_dict(state["scheduler"])
        for n, p in self.adapter.named_parameters():
            p.grad = (
                state["gradients"].get(n, None).to(p.device)
                if n in state["gradients"]
                else None
            )
        for name in ("cursor", "updates", "accumulated", "epoch", "wall_seconds"):
            setattr(self, name, state[name])
        receipt_path = path.with_name(path.name + ".receipt.pt")
        if receipt_path.exists():
            receipt = torch.load(receipt_path, map_location="cpu", weights_only=False)
            if receipt["checkpoint_sha256"] != sha(path):
                raise ValueError("checkpoint时间receipt摘要不符")
            self.wall_seconds = max(self.wall_seconds, receipt["wall_seconds"])
        restore_rng(state["rng"])
        if getattr(self, "session", None) is not None:
            self.session.previous = max(self.session.previous, self.wall_seconds)
        if self.updates > self.budget_updates:
            raise ValueError("恢复步数超过预算")

    def microstep(self) -> dict:
        """一个真实帧的前后向；每4帧/epoch尾组更新一次，实际分母一致。"""
        if self.updates >= self.budget_updates:
            raise ValueError("固定阶段更新预算已经耗尽")
        start = time.perf_counter()
        token = self.tokens[self.cursor]
        label = self.labels.get(token)
        if label["frame"]["split"] != "train":
            raise ValueError("训练读取非train标签")
        q = choose_query(label, self.epoch, self.cursor)
        missing = [t for t in (q["text"], q["wrong_text"]) if t not in self.text_cache]
        if missing:
            if len(self.text_cache) > 512:
                self.text_cache.clear()
            self.text_cache.update(encode_texts(self.encoder, missing))
        _held_host = None
        if self.pin_transfer and self.prefetched_token == token:
            torch.cuda.current_stream().wait_stream(self.transfer_stream)
            bev = self.prefetched_bev
            bev.record_stream(torch.cuda.current_stream())
            _held_host = self.prefetch_host
            self.prefetched_token = None
            self.prefetched_bev = None
            self.prefetch_host = None
        else:
            bev = self.get_bev(token)
            if bev.dtype != torch.float32:
                raise TypeError("训练BEV必须保持FP32")
            if self.pin_transfer and not bev.is_pinned():
                bev = bev.pin_memory()
            bev = bev.to("cuda", non_blocking=self.pin_transfer)
        if self.pin_transfer and self.cursor + 1 < len(self.tokens):
            owner = getattr(self.get_bev, "__self__", None)
            next_token = self.tokens[self.cursor + 1]
            next_host = (
                owner.peek(next_token)
                if owner is not None and hasattr(owner, "peek")
                else None
            )
            if next_host is not None:
                self.transfer_stream = self.transfer_stream or torch.cuda.Stream()
                self.prefetch_host = (
                    next_host if next_host.is_pinned() else next_host.pin_memory()
                )
                with torch.cuda.stream(self.transfer_stream):
                    self.prefetched_bev = self.prefetch_host.to(
                        "cuda", non_blocking=True
                    )
                self.prefetched_token = next_token
        sensor = label["frame"]
        position_key = tuple(sensor["lidar2ego_rotation"]) + tuple(
            sensor["lidar2ego_translation"]
        )
        if position_key not in self.position_cache:
            if len(self.position_cache) >= 16:
                self.position_cache.clear()
            self.position_cache[position_key] = geometry(sensor, device="cuda")
        position = self.position_cache[position_key]
        correct = self.adapter(bev, self.text_cache[q["text"]], position)
        wrong = self.adapter(bev, self.text_cache[q["wrong_text"]], position)
        correct["source_bev"] = bev
        if self.stage == "A" and not torch.equal(correct["enhanced_bev"], bev):
            raise RuntimeError("预热破坏原始恒等")
        loss, parts = objective(self.model, correct, wrong, label, q, self.stage)
        task_grad = {}
        if self.accumulated == 0 and (
            self.updates % 200 == 0 or self.updates == self.budget_updates - 1
        ):
            named = [
                (n, p) for n, p in self.adapter.named_parameters() if p.requires_grad
            ]
            for key in ("detection", "segmentation", "attention"):
                if key not in parts:
                    continue
                gradients = torch.autograd.grad(
                    parts[key],
                    [p for _, p in named],
                    retain_graph=True,
                    allow_unused=True,
                )
                task_grad[key] = dict(
                    total_l2=math.sqrt(
                        sum(
                            float(g.detach().square().sum())
                            for g in gradients
                            if g is not None
                        )
                    ),
                    language_l2=math.sqrt(
                        sum(
                            float(g.detach().square().sum())
                            for (n, _), g in zip(named, gradients)
                            if g is not None and n.startswith("language.")
                        )
                    ),
                )
        group_start = self.cursor - self.accumulated
        divisor = min(4, len(self.tokens) - group_start)
        (loss / divisor).backward()
        if not torch.stack(
            [
                torch.isfinite(p.grad).all()
                for p in self.adapter.parameters()
                if p.grad is not None
            ]
        ).all():
            raise FloatingPointError("非有限梯度")
        if self.model is not None and any(
            p.grad is not None for p in self.model.parameters()
        ):
            raise RuntimeError("原作者出现梯度")
        self.accumulated += 1
        self.cursor += 1
        with torch.no_grad():
            query_cosine = torch.nn.functional.cosine_similarity(
                self.adapter.language(self.text_cache[q["text"]]),
                self.adapter.language(self.text_cache[q["wrong_text"]]),
            ).mean()
            residual_ratio = correct["residual"].detach().norm() / bev.norm()
            keys = list(parts)
            scalars = (
                torch.stack(
                    [parts[k].detach().reshape(()) for k in keys]
                    + [residual_ratio, query_cosine]
                )
                .cpu()
                .tolist()
            )
        record = dict(
            cursor=self.cursor,
            epoch=self.epoch,
            token=token,
            instruction_id=q["instruction_id"],
            stage=self.stage,
            losses=dict(zip(keys, scalars[: len(keys)])),
            task_gradients=task_grad,
            residual_ratio=scalars[-2],
            query_cosine=scalars[-1],
            attention_sha256=hashlib.sha256(
                correct["attention_logits"].detach().cpu().numpy().tobytes()
            ).hexdigest(),
            target_present=bool(q["selected"]),
            accumulation_divisor=divisor,
        )
        if self.accumulated == divisor:
            norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.adapter.parameters(), 5.0, error_if_nonfinite=True
                )
            )
            if norm <= 0:
                raise RuntimeError("优化器更新前梯度为零")
            gradients = (
                {
                    n: p.grad.detach().cpu().clone()
                    for n, p in self.adapter.named_parameters()
                    if p.grad is not None
                }
                if self.retain_audit_tensors
                else None
            )
            before = {n: p.detach().clone() for n, p in self.adapter.named_parameters()}
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.updates += 1
            self.accumulated = 0
            delta = float(
                torch.stack(
                    [
                        (p.detach() - before[n]).square().sum()
                        for n, p in self.adapter.named_parameters()
                    ]
                )
                .sum()
                .sqrt()
            )
            if delta <= 0:
                raise RuntimeError("参数更新为零")
            update = dict(update=self.updates, norm=norm, delta_l2=delta)
            if self.retain_audit_tensors:
                update.update(
                    gradients=gradients,
                    state={
                        n: p.detach().cpu().clone()
                        for n, p in self.adapter.named_parameters()
                    },
                )
            self.update_records.append(update)
            if not self.retain_audit_tensors and len(self.update_records) > 256:
                self.update_records.pop(0)
            self.optimizer.zero_grad(set_to_none=True)
            record.update(
                update=self.updates, gradient_l2=norm, parameter_delta_l2=delta
            )
        self.records.append(record)
        # scalar回传已等待本帧CUDA流，随后才释放异步传输的pinned源引用。
        del _held_host
        if not self.retain_audit_tensors and len(self.records) > 256:
            self.records.pop(0)
        if self._run_clock is None:
            self.wall_seconds += time.perf_counter() - start
        return record

    def run(
        self,
        microsteps: int | None = None,
        *,
        log: Path | None = None,
        checkpoint_dir: Path | None = None,
        cache: RollingCache | None = None,
    ) -> None:
        """执行有界片段或阶段；信号在micro边界保存，24小时到期标INCOMPLETE。"""
        remaining = microsteps
        if self._run_clock is not None:
            raise RuntimeError("拒绝嵌套训练计时")
        self._run_clock = time.perf_counter()
        stream = log.open("x") if log else None
        previous = {}
        from scripts.stage3.lg_bev_engineering_determinism import stable_native_backward

        deterministic = stable_native_backward()
        deterministic.__enter__()

        def stop(_signum, _frame):
            self.stop_requested = True

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            previous[signum] = signal.signal(signum, stop)
        try:
            while self.updates < self.budget_updates and (
                remaining is None or remaining > 0
            ):
                if self.should_stop():
                    break
                if self.cursor == len(self.tokens):
                    if self.accumulated:
                        raise RuntimeError("epoch尾残留梯度")
                    self.cursor = 0
                    self.epoch += 1
                record = self.microstep()
                record["learning_rate"] = (
                    self.optimizer.param_groups[0]["lr"]
                    if hasattr(self, "optimizer")
                    else None
                )
                if getattr(self, "progress_callback", None) is not None:
                    self.progress_callback(self, record)
                if stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                if remaining is not None:
                    remaining -= 1
                epoch_end = self.cursor == len(self.tokens)
                if checkpoint_dir is not None and (
                    epoch_end
                    or self.should_stop()
                    or record.get("update", 0) % 500 == 0
                    and "update" in record
                ):
                    frozen_check(self.model, self.encoder, self.baseline)
                    path = (
                        checkpoint_dir
                        / f"{self.stage}_u{self.updates:06d}_e{self.epoch}_m{self.cursor:06d}_{time.time_ns()}.pt"
                    )
                    digest = self.checkpoint(path, cache)
                    with (checkpoint_dir.parent / "control.jsonl").open("a") as control:
                        control.write(
                            json.dumps(
                                dict(
                                    event="checkpoint",
                                    stage=self.stage,
                                    updates=self.updates,
                                    cursor=self.cursor,
                                    path=str(path),
                                    sha256=digest,
                                )
                            )
                            + "\n"
                        )
                        control.flush()
                        os.fsync(control.fileno())
                    if getattr(self, "checkpoint_callback", None) is not None:
                        self.checkpoint_callback(self, path, digest)
                if self.should_stop():
                    break
        finally:
            try:
                deterministic.__exit__(None, None, None)
                if stream:
                    stream.close()
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
                frozen_check(self.model, self.encoder, self.baseline)
            finally:
                self.wall_seconds = self.elapsed_wall()
                self._run_clock = None


def main() -> None:
    """可执行正式入口；没有明确授权、全量标签和READY证据则在GPU构建前退出。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engineering-out",
        type=Path,
        default=ROOT / "results/lg_bev_engineering_20261027",
    )
    parser.add_argument("--data-report", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--authorize-full-training", action="store_true")
    parser.add_argument("--cache-gb", type=float, default=None)
    parser.add_argument("--prepare-data-if-needed", action="store_true")
    parser.add_argument("--data-run-id", default="formal_gt_20261027_01")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if not args.authorize_full_training:
        raise SystemExit("完整长训需用户另行授权，此命令未启动GPU")
    from scripts.stage3.lg_bev_engineering_full import full_training

    full_training(args)


if __name__ == "__main__":
    main()

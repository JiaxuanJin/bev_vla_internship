"""正式工程边界：归属安全、容量/pin/损坏/中断、GT差集和SDK计分。"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.stage3.lg_bev_engineering_cache import RollingCache, contained
from scripts.stage3.lg_bev_engineering_data import SceneData
from scripts.stage3.lg_bev_engineering_determinism import StableGrid
from scripts.stage3.lg_bev_engineering_evaluate import (
    sdk_map,
    macro_iou,
    detection_boxes,
)
from scripts.stage3.lg_bev_engineering_loss import (
    pair_ranking,
    map_roi,
    target_weighting,
)


@pytest.fixture
def cache(tmp_path):
    """只用字节级合成张量，不在E盘单测目录生成大型训练缓存。"""
    result = RollingCache(
        tmp_path / "owned", capacity=3_000_000, reserve=0, test_root=True
    )
    result.expected_shape = (1, 1, 2, 2)
    return result


def meta():
    return dict(
        scene_prefix=["frame"],
        regenerable=True,
        checkpoint_sha256="ck",
        config_sha256="cfg",
        preprocessing="synthetic test only",
        frames=["frame"],
        source_dtype="torch.float32",
        history_policy="from scene start",
    )


def test_cache_roundtrip_corruption_recognition_and_regeneration(cache):
    value = {"frame": torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)}
    cache.put("one", value, meta())
    assert torch.equal(cache.get("one")["frame"], value["frame"])
    path = contained(
        cache.root,
        cache.db.execute("SELECT path FROM shards WHERE key=?", ("one",)).fetchone()[0],
    )
    with path.open("r+b") as stream:
        stream.write(b"BAD!")
    assert cache.get("one") is None
    assert path.is_file()
    cache.put("one", value, meta())
    assert torch.equal(cache.get("one")["frame"], value["frame"])
    assert path.is_file()


def test_pinned_or_foreign_file_cannot_be_evicted(cache):
    value = {"frame": torch.ones(1, 1, 2, 2)}
    cache.put("one", value, meta())
    foreign = cache.root / "unowned_raw_keep.txt"
    foreign.write_text("keep")
    cache.capacity = cache.usage() + 2**20 - 1000
    with cache.pin("one"):
        with pytest.raises(OSError):
            cache.put("two", value, meta())
    cache.put("two", value, meta())
    assert cache.get("one") is None
    assert foreign.read_text() == "keep"


def test_reserve_blocks_write_before_deleting_anything(cache, monkeypatch):
    with cache.locked():
        pass
    before = set(cache.root.iterdir())
    monkeypatch.setattr(
        "scripts.stage3.lg_bev_engineering_cache.shutil.disk_usage",
        lambda _: SimpleNamespace(free=0),
    )
    with pytest.raises(OSError):
        cache.put("x", {"frame": torch.ones(1, 1, 2, 2)}, meta())
    assert set(cache.root.iterdir()) == before


@pytest.mark.parametrize(
    "path", ["../raw.pth", "/mnt/g/nus/checkpoint", "shards/../../raw"]
)
def test_path_escape_is_rejected(cache, path):
    with pytest.raises(ValueError):
        contained(cache.root, path)


def test_symlink_escape_is_rejected(cache, tmp_path):
    (cache.root / "bad").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        contained(cache.root, "bad/file")
    with pytest.raises(ValueError):
        cache.usage()


def test_authorized_capacity_cannot_expand(cache):
    with pytest.raises(ValueError):
        RollingCache(cache.root, capacity=30_000_000_001, test_root=True)
    bad = meta()
    bad["source_dtype"] = "torch.float16"
    with pytest.raises(ValueError):
        cache.put("x", {"frame": torch.ones(1, 1, 2, 2)}, bad)


def test_pending_complete_write_is_recovered(cache):
    tensor = torch.ones(1, 1, 2, 2)
    value = {"frame": tensor}
    metadata = meta()
    metadata["tensor_sha256"] = {
        "frame": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    }
    relative = "shards/crash.pt"
    path = contained(cache.root, relative)
    path.parent.mkdir()
    torch.save(value, path.with_name(path.name + ".pending"))
    cache.db.execute(
        "INSERT INTO shards VALUES(?,?,?,?,?,?,?,?)",
        ("crash", relative, "", 0, json.dumps(metadata), 0, 0, "pending"),
    )
    cache.db.commit()
    assert cache.recover()["recovered"] == 1
    assert torch.equal(cache.get("crash")["frame"], tensor)


def test_grounding_ranking_ignores_shared_target_and_handles_empty():
    c = torch.zeros(1, 1, 1, 3, requires_grad=True)
    w = torch.zeros_like(c, requires_grad=True)
    a = torch.tensor([[[[1, 1, 0]]]], dtype=torch.bool)
    b = torch.tensor([[[[0, 1, 1]]]], dtype=torch.bool)
    loss = pair_ranking(c, w, a, b, torch.ones_like(a))
    loss.backward()
    assert c.grad[0, 0, 0, 1] == 0 and w.grad[0, 0, 0, 1] == 0
    assert c.grad[0, 0, 0, 0] < 0 and w.grad[0, 0, 0, 2] < 0
    assert pair_ranking(c, w, a, a, torch.ones_like(a)) == 0


def test_roi_uses_physical_euclidean_distance():
    target = np.zeros((40, 40), bool)
    target[20, 20] = True
    roi = map_roi(target)
    assert roi[20, 33] and not roi[20, 34] and not roi[30, 30]
    assert not map_roi(np.zeros_like(target)).any()


def test_frozen_final_data_is_rejected_before_sdk_open():
    with pytest.raises(ValueError):
        SceneData([dict(split="cold_final")])
    with pytest.raises(ValueError):
        SceneData([dict(split="cold_final")], allowed_split="cold_final")


def test_stable_grid_cpu_gradient_matches_reference():
    value = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).requires_grad_()
    grid = torch.tensor([[[[-0.2, 0.1], [0.3, -0.4]]]])
    original = torch.nn.functional.grid_sample(value, grid, align_corners=False)
    expected = torch.autograd.grad(original.sum(), value)[0]
    output = StableGrid.apply(value, grid)
    actual = torch.autograd.grad(output.sum(), value)[0]
    assert torch.equal(output, original) and torch.equal(actual, expected)


def test_sdk_distance_strict_boundary_and_empty_target_fp():
    from nuscenes.eval.detection.data_classes import DetectionBox

    def box(token, x, score):
        return DetectionBox(
            sample_token=token,
            translation=(x, 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
            velocity=(0.0, 0.0),
            detection_name="car",
            detection_score=score,
        )

    gt = {"a": [box("a", 0.0, -1.0)], "empty": []}
    pred = {"a": [box("a", 0.5, 0.8)], "empty": [box("empty", 0.0, 0.9)]}
    score = sdk_map(gt, pred, ("car",))
    assert score["class_AP"]["car"]["0.5"] == 0
    assert 0 < score["class_AP"]["car"]["1.0"] < 1
    assert sdk_map(gt, pred, ("car", "pedestrian"))["mAP"] is None
    assert macro_iou(np.zeros((4, 4), int), (1, 2, 3))["mIoU"] is None


def test_sdk_range_and_zero_point_filter():
    frame = dict(
        lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0], lidar2ego_translation=[0.0, 0.0, 0.0]
    )
    boxes = np.array(
        [
            [50.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    assert (
        detection_boxes(
            boxes, np.array([0, 0]), np.array([0.5, 0.5]), frame, "x", np.array([1, 0])
        )
        == []
    )


def test_streamed_ap_matches_sdk_with_ties_duplicates_and_empty_queries(tmp_path):
    from nuscenes.eval.detection.data_classes import DetectionBox
    from scripts.stage3.lg_bev_engineering_metrics import MetricSpool

    rng = np.random.RandomState(17)
    gt = {}
    pred = {}
    spool = MetricSpool(tmp_path / "events.sqlite3")

    def box(token, x, score, cls):
        return DetectionBox(
            sample_token=token,
            translation=(float(x), 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
            velocity=(0.0, 0.0),
            detection_name=cls,
            detection_score=float(score),
        )

    for i in range(12):
        token = str(i)
        gt[token] = []
        pred[token] = []
        for cls in ("car", "pedestrian"):
            if i % 3:
                gt[token].append(box(token, 0.0, -1.0, cls))
            pred[token].extend(
                box(token, x, score, cls)
                for x, score in zip(
                    rng.choice([0.0, 0.5, 1.0, 2.0, 4.0], 5),
                    rng.choice([0.2, 0.5, 0.9], 5),
                )
            )
        spool.add(
            "original", "related", token, "scene" + str(i % 2), gt[token], pred[token]
        )
    expected = sdk_map(gt, pred, ("car", "pedestrian"))
    actual = spool.metric("original", "related", ("car", "pedestrian"))
    assert actual["class_AP"] == expected["class_AP"]
    assert actual["mAP"] == expected["mAP"]
    assert (
        spool.bootstrap("original", "related", ("car", "pedestrian"), replicates=8)[
            "valid_replicates"
        ]
        == 8
    )


def test_native_target_weight_wrapper_restores_after_exception():
    class Head:
        def _get_target_single(self):
            return (
                torch.tensor([0, 1]),
                torch.ones(2),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.ones(2, 2),
                torch.tensor([0, 1]),
                torch.empty(0, dtype=torch.long),
            )

    head = Head()
    with pytest.raises(RuntimeError):
        with target_weighting(head, torch.tensor([[3.0, 4.0]])):
            values = head._get_target_single()
            assert values[1].tolist() == [1.0, 3.0]
            assert values[3].tolist() == [[1.0, 1.0], [3.0, 3.0]]
            raise RuntimeError("deliberate")
    assert "_get_target_single" not in head.__dict__


def test_partial_data_cannot_start_full_budget(tmp_path, monkeypatch):
    from scripts.stage3.lg_bev_engineering_full import full_training
    from scripts.stage3.lg_bev_common import ROOT, sha, write

    write(tmp_path / "ready.json", dict(engineering_32frames="PASS"))
    write(tmp_path / "resource_admission.json", dict(budget_24h_pass=True))
    write(
        tmp_path / "fast_pipeline_equivalence.json",
        dict(
            status="PASS",
            source_sha256={
                str(p): sha(p)
                for p in (ROOT / "scripts/stage3").glob("lg_bev_engineering_*.py")
            },
        ),
    )
    write(
        tmp_path / "data.json",
        dict(status="PASS", frames=[dict(sample_token="sample32")], success=32),
    )
    monkeypatch.setattr(
        "scripts.stage3.lg_bev_engineering_full.formal_frames",
        lambda: [dict(sample_token="sample32", scene_token="s", timestamp=0)],
    )
    args = SimpleNamespace(
        authorize_full_training=True,
        engineering_out=tmp_path,
        prepare_data_if_needed=False,
        data_report=tmp_path / "data.json",
    )
    with pytest.raises(ValueError, match="19123"):
        full_training(args)
    args.authorize_full_training = False
    with pytest.raises(ValueError, match="授权"):
        full_training(args)


def test_nonterminal_checkpoint_cannot_open_development(tmp_path):
    from scripts.stage3.lg_bev_engineering_full import evaluate_terminal
    from scripts.stage3.lg_bev_common import write

    write(tmp_path / "terminal.json", dict(total_updates=8, status="SMOKE"))
    with pytest.raises(ValueError, match="禁止打开开发集"):
        evaluate_terminal(
            SimpleNamespace(training_run=tmp_path, checkpoint=tmp_path / "unused.pt")
        )
    assert not (tmp_path / "development_opened.json").exists()


def test_persisted_cache_policy_does_not_auto_expand(cache):
    cache.configure_capacity(2_000_000, user_requested=True)
    reloaded = RollingCache(cache.root, capacity=3_000_000, reserve=0, test_root=True)
    assert reloaded.capacity == 2_000_000
    with pytest.raises(ValueError, match="显式"):
        reloaded.configure_capacity(3_000_000)


def test_deterministic_conv_adjoint_matches_native_math():
    from scripts.stage3.lg_bev_engineering_determinism import StableConv

    torch.manual_seed(3)
    x = torch.randn(1, 2, 5, 5, requires_grad=True)
    w = torch.randn(3, 2, 3, 3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    expected = torch.nn.functional.conv2d(x, w, b, padding=1)
    actual = StableConv.apply(x, w, b, 1, 1, 1, 1)
    assert torch.equal(actual, expected)
    ga = torch.autograd.grad(actual.sum(), (x, w, b))
    gb = torch.autograd.grad(expected.sum(), (x, w, b))
    assert all(torch.allclose(a, b, atol=1e-6, rtol=1e-6) for a, b in zip(ga, gb))


def test_fixed_linear_adjoint_matches_grid_reference():
    """验证稀疏固定顺序归约，含越界/稀疏/零权重与多通道。"""
    from scripts.stage3.lg_bev_engineering_determinism import (
        sparse_adjoint,
        apply_adjoint,
    )

    generator = torch.Generator().manual_seed(17)
    value = torch.randn(1, 3, 7, 8, generator=generator, requires_grad=True)
    grid = torch.rand(1, 11, 13, 2, generator=generator) * 2.5 - 1.25
    result = torch.nn.functional.grid_sample(value, grid, align_corners=False)
    gradient = torch.randn(result.shape, generator=generator)
    expected = torch.autograd.grad(result, value, gradient)[0]
    x = ((grid[0, ..., 0] + 1) * 8 - 1) / 2
    y = ((grid[0, ..., 1] + 1) * 7 - 1) / 2
    actual = apply_adjoint(sparse_adjoint(x, y, 7, 8, "cpu"), gradient, 7, 8)
    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_failed_dynamic_cuda_cannot_enter_full_training(tmp_path, monkeypatch):
    from scripts.stage3.lg_bev_engineering_full import full_training

    monkeypatch.setenv("LG_BEV_DYNAMIC_CUDA", "ordered")
    with pytest.raises(ValueError, match="未通过数值"):
        full_training(SimpleNamespace(authorize_full_training=True))


def test_over_24h_budget_blocks_before_gpu(tmp_path):
    from scripts.stage3.lg_bev_engineering_full import full_training
    from scripts.stage3.lg_bev_common import write

    write(tmp_path / "ready.json", dict(engineering_32frames="PASS"))
    write(tmp_path / "resource_admission.json", dict(budget_24h_pass=False))
    with pytest.raises(ValueError, match="24小时"):
        full_training(
            SimpleNamespace(authorize_full_training=True, engineering_out=tmp_path)
        )


@pytest.mark.parametrize("align", [True, False])
def test_direct_cpu_interpolate_uses_identical_aten_backward(monkeypatch, align):
    """去掉无用CPU forward/输入复制，反向内核及FP32结果必须完全相同。"""
    from scripts.stage3.lg_bev_engineering_determinism import StableInterpolate

    monkeypatch.setenv("LG_BEV_UPSAMPLE_CUDA", "0")
    torch.manual_seed(3)
    value = torch.randn(1, 3, 5, 7, requires_grad=True)
    options = dict(
        size=None,
        scale_factor=2,
        mode="bilinear",
        align_corners=align,
        recompute_scale_factor=None,
        antialias=False,
    )
    gradient = torch.randn(1, 3, 10, 14)
    expected = torch.autograd.grad(
        torch.nn.functional.interpolate(value, **options), value, gradient
    )[0]
    monkeypatch.setenv("LG_BEV_CPU_DIRECT", "1")
    actual = torch.autograd.grad(
        StableInterpolate.apply(value, options), value, gradient
    )[0]
    assert torch.equal(actual, expected)


def test_run_wall_includes_checkpoint_and_frozen_checks(tmp_path, monkeypatch):
    """纯控制逻辑测试；微步2秒、两次校验各5秒、保存3秒都计入累计。"""
    from contextlib import nullcontext
    from scripts.stage3.lg_bev_engineering_train import Trainer

    clock = [0.0]
    monkeypatch.setattr(
        "scripts.stage3.lg_bev_engineering_train.time.perf_counter", lambda: clock[0]
    )
    monkeypatch.setattr(
        "scripts.stage3.lg_bev_engineering_determinism.stable_native_backward",
        nullcontext,
    )

    def check(*_args):
        clock[0] += 5

    monkeypatch.setattr("scripts.stage3.lg_bev_engineering_train.frozen_check", check)
    trainer = Trainer.__new__(Trainer)
    trainer.__dict__.update(
        wall_seconds=100.0,
        _run_clock=None,
        updates=0,
        budget_updates=1,
        tokens=["a"],
        cursor=0,
        accumulated=0,
        epoch=0,
        stop_requested=False,
        stage="A",
        model=None,
        encoder=None,
        baseline=None,
    )

    def microstep():
        clock[0] += 2
        trainer.cursor = trainer.updates = 1
        return dict(update=1)

    def checkpoint(*_args):
        clock[0] += 3
        return "digest"

    trainer.microstep, trainer.checkpoint = microstep, checkpoint
    trainer.run(1, log=tmp_path / "run.jsonl", checkpoint_dir=tmp_path / "checkpoints")
    assert trainer.wall_seconds == 115 and trainer._run_clock is None


def test_checkpoint_receipt_preserves_persist_cost_on_resume(tmp_path, monkeypatch):
    """小型CPU状态证明保存开销跟随checkpoint恢复，不重设累计计时。"""
    from scripts.stage3.lg_bev_engineering_train import Trainer
    from scripts.stage3.lg_bev_engineering_cache import atomic_save

    clock = [0.0]
    monkeypatch.setattr(
        "scripts.stage3.lg_bev_engineering_train.time.perf_counter", lambda: clock[0]
    )
    monkeypatch.setattr("scripts.stage3.lg_bev_engineering_train.rng_state", lambda: {})
    monkeypatch.setattr(
        "scripts.stage3.lg_bev_engineering_train.restore_rng", lambda _: None
    )

    def save(path, value):
        atomic_save(path, value)
        clock[0] += 3

    monkeypatch.setattr("scripts.stage3.lg_bev_engineering_train.atomic_save", save)

    def dummy():
        trainer = Trainer.__new__(Trainer)
        trainer.adapter = torch.nn.Linear(1, 1)
        trainer.optimizer = torch.optim.AdamW(trainer.adapter.parameters())
        trainer.__dict__.update(
            stage="A",
            scheduler=None,
            cursor=1,
            updates=0,
            accumulated=1,
            epoch=0,
            budget_updates=8,
            protocol_sha256="protocol",
            tokens=["a", "b"],
            baseline=None,
            wall_seconds=10.0,
            _run_clock=None,
        )
        return trainer

    trainer = dummy()
    trainer._run_clock = 0.0
    path = tmp_path / "state.pt"
    trainer.checkpoint(path)
    restored = dummy()
    restored.resume(path)
    assert restored.wall_seconds == 13.0
    assert all(
        torch.equal(a, b)
        for a, b in zip(trainer.adapter.parameters(), restored.adapter.parameters())
    )


def test_completed_warmup_gate_is_idempotent_and_endpoint_bound(tmp_path, monkeypatch):
    """A终点重启不再跑128帧，也不覆盖已封存结果；错误端点拒绝。"""
    from scripts.stage3.lg_bev_engineering_full import completed_warmup_gate

    clock, calls = [0.0], [0]
    monkeypatch.setattr(
        "scripts.stage3.lg_bev_engineering_full.time.perf_counter", lambda: clock[0]
    )
    trainer = SimpleNamespace(
        adapter=torch.nn.Linear(1, 1),
        protocol_sha256="same",
        updates=9562,
        wall_seconds=10.0,
    )
    trainer.elapsed_wall = lambda: trainer.wall_seconds

    def probe(_):
        calls[0] += 1
        clock[0] += 4
        return dict(status="PASS")

    monkeypatch.setattr("scripts.stage3.lg_bev_engineering_full.warmup_probe", probe)
    first = completed_warmup_gate(trainer, tmp_path)
    content = (tmp_path / "warmup_gate.json").read_bytes()
    trainer.wall_seconds = 10.0
    assert completed_warmup_gate(trainer, tmp_path) == first
    assert calls[0] == 1 and trainer.wall_seconds == 14.0
    assert (tmp_path / "warmup_gate.json").read_bytes() == content
    trainer.protocol_sha256 = "different"
    with pytest.raises(ValueError, match="同一A端点"):
        completed_warmup_gate(trainer, tmp_path)


def test_failed_warmup_counts_time_without_sealing_gate(tmp_path, monkeypatch):
    """探针异常不产生可复用成功报告，已消耗活跃时间不丢弃。"""
    from scripts.stage3.lg_bev_engineering_full import completed_warmup_gate

    clock = [0.0]
    monkeypatch.setattr(
        "scripts.stage3.lg_bev_engineering_full.time.perf_counter", lambda: clock[0]
    )
    trainer = SimpleNamespace(
        adapter=torch.nn.Linear(1, 1),
        protocol_sha256="same",
        updates=9562,
        wall_seconds=10.0,
    )
    trainer.elapsed_wall = lambda: trainer.wall_seconds

    def probe(_):
        clock[0] += 3
        raise RuntimeError("data failure")

    monkeypatch.setattr("scripts.stage3.lg_bev_engineering_full.warmup_probe", probe)
    with pytest.raises(RuntimeError, match="data failure"):
        completed_warmup_gate(trainer, tmp_path)
    assert trainer.wall_seconds == 13.0
    assert not (tmp_path / "warmup_gate.json").exists()

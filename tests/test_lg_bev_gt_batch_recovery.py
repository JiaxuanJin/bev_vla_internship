"""按场景GT原子块的恢复边界；数值来源另由真实全量/独立帧验收。"""

from pathlib import Path
import fcntl
import json

import pytest
import torch

from scripts.stage3.lg_bev_engineering_cache import RollingCache
from scripts.stage3.lg_bev_gt_completion import save_owned_block
from scripts.stage3.lg_bev_common import sha


def test_complete_pending_is_validated_and_recovered(tmp_path: Path) -> None:
    cache = RollingCache(tmp_path, capacity=2**24, reserve=0, test_root=True)
    path = tmp_path / "new_scene.pt"
    expected = {"train_token": {"boxes": torch.empty(0, 9)}}
    torch.save(expected, path.with_name(path.name + ".pending"))
    with cache.locked():
        save_owned_block(cache, path, expected, 10000)
    assert torch.load(path, weights_only=False)["train_token"]["boxes"].shape == (0, 9)


def test_bad_pending_is_retained_and_regenerated(tmp_path: Path) -> None:
    cache = RollingCache(tmp_path, capacity=2**24, reserve=0, test_root=True)
    path = tmp_path / "new_scene.pt"
    path.with_name(path.name + ".pending").write_bytes(b"interrupted")
    with cache.locked():
        save_owned_block(cache, path, {"value": torch.ones(2)}, 10000)
    assert len(list(tmp_path.glob("*.interrupted_*"))) == 1
    assert torch.equal(torch.load(path, weights_only=False)["value"], torch.ones(2))
    with cache.locked(), pytest.raises(ValueError, match="拒绝覆盖"):
        save_owned_block(cache, path, {"value": torch.zeros(2)}, 10000)


def compatibility_fixture(tmp_path, monkeypatch):
    from scripts.stage3 import lg_bev_gt_completion as gt

    monkeypatch.setattr(gt, "ROOT", tmp_path)
    source = tmp_path / "scripts/stage3"
    source.mkdir(parents=True)
    previous = tmp_path / "previous.py"
    previous.write_text(
        "def box_masks():\n    return 1\n\ndef verify_box_corners():\n    return 0\n"
    )
    helper = source / "lg_bev_data.py"
    helper.write_text(previous.read_text().replace("return 0", "return 2"))
    builder = source / "lg_bev_engineering_data.py"
    builder.write_text("# unchanged builder\n")
    (source / "lg_bev_gt_completion.py").write_text("# orchestrator\n")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(dict(builder_sha256=sha(builder), groups={"s": ["t"]})))
    regression = tmp_path / "regression.json"
    regression.write_text(json.dumps(dict(status="PASS", pixel_differences_old_new=0)))
    receipt = tmp_path / "compatibility.json"
    contract = dict(
        status="PASS_VALIDATOR_ONLY_COMPATIBILITY",
        supervision_semantics_changed=False,
        spec_sha256={str(spec): sha(spec)},
        previous_helper="previous.py",
        source_sha256={
            str(p.relative_to(tmp_path)): sha(p) for p in source.glob("*.py")
        },
        evidence_sha256={"regression.json": sha(regression)},
        regression_report="regression.json",
    )
    receipt.write_text(json.dumps(contract))
    monkeypatch.setattr(gt, "LEGACY_HELPER_SHA256", sha(previous))
    return gt, spec, receipt, helper, contract


def test_validator_only_contract_preserves_generation_and_exact_sources(
    tmp_path, monkeypatch
):
    gt, spec, receipt, _helper, _contract = compatibility_fixture(tmp_path, monkeypatch)
    assert gt.checked_config(spec, receipt)["groups"] == {"s": ["t"]}
    with pytest.raises(ValueError, match="必须提供"):
        gt.checked_config(spec, None)


def test_contract_cannot_authorize_changed_gt_even_after_rehash(tmp_path, monkeypatch):
    gt, spec, receipt, helper, contract = compatibility_fixture(tmp_path, monkeypatch)
    helper.write_text(helper.read_text().replace("return 1", "return 9"))
    contract["source_sha256"]["scripts/stage3/lg_bev_data.py"] = sha(helper)
    receipt.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="不得改变GT生成器"):
        gt.checked_config(spec, receipt)


def test_contract_rejects_source_or_task_changes(tmp_path, monkeypatch):
    gt, spec, receipt, helper, _contract = compatibility_fixture(tmp_path, monkeypatch)
    helper.write_text(helper.read_text() + "\n# changed after seal\n")
    with pytest.raises(ValueError, match="源码SHA"):
        gt.checked_config(spec, receipt)
    config = json.loads(spec.read_text())
    config["groups"] = {"another_scene": ["another_frame"]}
    spec.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="原分片绑定"):
        gt.checked_config(spec, receipt)


def test_real_coordinator_lock_rejects_second_start(tmp_path, monkeypatch):
    from scripts.stage3 import lg_bev_gt_completion as gt

    monkeypatch.setattr(gt, "RUN", tmp_path)
    monkeypatch.setattr(gt, "_build", lambda _: pytest.fail("不得启动第二份GT"))
    with (tmp_path / "coordinator.lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="拒绝重复启动"):
            gt.build()


def test_gt_writer_stops_at_capacity_without_requesting_eviction(tmp_path):
    sentinel = tmp_path / "existing_bev.pt"
    sentinel.write_bytes(b"must be retained")

    class FullCache:
        root = tmp_path

        def admit(self, incoming):
            raise OSError("capacity full")

        def reserve_for_artifact(self, *args):
            pytest.fail("GT不得请求淘汰已有文件")

    with pytest.raises(OSError, match="capacity full"):
        save_owned_block(FullCache(), tmp_path / "new_gt.pt", {}, 10000)
    assert sentinel.read_bytes() == b"must be retained"
    assert not (tmp_path / "new_gt.pt").exists()

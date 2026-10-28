"""按场景GT原子块的恢复边界；数值来源另由真实全量/独立帧验收。"""

from pathlib import Path

import pytest
import torch

from scripts.stage3.lg_bev_engineering_cache import RollingCache
from scripts.stage3.lg_bev_gt_completion import save_owned_block


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

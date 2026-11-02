"""CPU验收夹具；任何CUDA初始化或迁移都会立即失败。"""

import os
import builtins
import io
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import pytest
import torch

from core import Store, read


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    """包括误用torch.manual_seed隐式CUDA分支在内均拒绝。"""

    def denied(*args, **kwargs):
        raise AssertionError("CPU测试禁止CUDA初始化/迁移/种子")

    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    for name in ("_lazy_init", "init", "manual_seed", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, name, denied)
    monkeypatch.setattr(torch.Tensor, "cuda", denied)
    monkeypatch.setattr(torch.nn.Module, "cuda", denied)

    # 数据集原始表含多split GT，CPU验收整体拒绝访问原始数据集与开发缓存。
    def guarded_open(original):
        def wrapped(file, *args, **kwargs):
            if not isinstance(file, int):
                path = str(file)
                assert "/bev_seg_nuscenes/" not in path, "CPU验收禁止原始GT读取"
                assert "/cache/development_labels/" not in path, "CPU验收禁止开发GT读取"
            return original(file, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(builtins, "open", guarded_open(builtins.open))
    monkeypatch.setattr(io, "open", guarded_open(io.open))
    yield
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope="session")
def config():
    return read(HERE / "configs/main.json")


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path, test=True)

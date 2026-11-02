import os
import sys
from pathlib import Path
import builtins
import io
import pytest

REPORT=Path(__file__).resolve().parents[1]
ORIGINAL=REPORT.parents[1]
sys.path[:0]=[str(REPORT/'source'),str(ORIGINAL),str(ORIGINAL.parents[1])]
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch

@pytest.fixture(autouse=True)
def fence(monkeypatch):
    torch.set_num_threads(1)
    def denied(*a,**k):
        raise AssertionError('CPU preflight cannot initialize CUDA')
    monkeypatch.setattr(torch.cuda,'_lazy_init',denied)
    def guard(original):
        def wrapped(p,*a,**kw):
            if not isinstance(p,int):
                path=str(p)
                assert '/cache/development_labels/' not in path
                assert '/bev_seg_nuscenes/' not in path
            return original(p,*a,**kw)
        return wrapped
    monkeypatch.setattr(builtins,'open',guard(builtins.open))
    monkeypatch.setattr(io,'open',guard(io.open))

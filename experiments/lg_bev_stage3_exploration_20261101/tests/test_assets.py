"""只读实物迁移、冻结及合成双Head反向验收。"""

import copy
import subprocess
import sys

import pytest
import torch

from core import (
    HERE,
    OUTPUT,
    FORMAL,
    freeze,
    fresh_optimizer,
    group_bev,
    load_adapter,
    read,
    seed_cpu,
    selected_rows,
    tensor_sha,
    verify_sources,
)
from cli import validate_config


@pytest.mark.parametrize("name", ["a0", "a9562"])
def test_real_asset_identity(config, name):
    before = torch.get_rng_state().clone()
    adapter = load_adapter(config, name)
    assert tensor_sha(adapter.state_dict()) == config["assets"][name]["adapter_sha256"]
    assert torch.equal(before, torch.get_rng_state())
    assert all(p.grad is None and p.device.type == "cpu" for p in adapter.parameters())
    assert sum(p.numel() for p in adapter.up.parameters()) == 6400


@pytest.mark.parametrize(
    "field", ["file_sha256", "receipt_sha256", "execution_sha256", "adapter_sha256"]
)
def test_bad_migration_identity_rejected(config, field):
    changed = copy.deepcopy(config)
    changed["assets"]["a9562"][field] = "0" * 64
    with pytest.raises(ValueError):
        load_adapter(changed)


def inputs():
    seed_cpu(20261101)
    return torch.randn(1, 256, 4, 4), torch.randn(1, 3072), torch.randn(1, 5, 4, 4)


@pytest.mark.parametrize("name", ["a0", "a9562"])
@pytest.mark.parametrize("control", ["correct", "wrong", "zero"])
def test_zero_up_two_heads_identical(config, name, control):
    adapter = freeze(load_adapter(config, name))
    bev, text, position = inputs()
    text = (
        -text
        if control == "wrong"
        else torch.zeros_like(text) if control == "zero" else text
    )
    heads = freeze(
        torch.nn.Sequential(torch.nn.BatchNorm2d(256), torch.nn.Conv2d(256, 8, 1))
    )
    frozen = tensor_sha(heads.state_dict())
    original = group_bev("baseline", None, bev, text, position)
    enhanced = group_bev("a_only", adapter, bev, text, position)
    assert original["attention_logits"] is None
    assert torch.equal(original["enhanced_bev"], enhanced["enhanced_bev"])
    left, right = heads(bev), heads(enhanced["enhanced_bev"])
    assert torch.equal(left[:, :4], right[:, :4])
    assert torch.equal(left[:, 4:], right[:, 4:])
    assert tensor_sha(heads.state_dict()) == frozen
    assert not heads.training and not any(p.requires_grad for p in heads.parameters())


def test_only_adapter_updated_fresh_optimizer(config):
    adapter = load_adapter(config)
    optimizer, scheduler = fresh_optimizer(adapter, 256)
    assert not optimizer.state and scheduler.last_epoch == 0
    assert sum(p.numel() for p in adapter.parameters() if p.requires_grad) == 93866
    assert {id(p) for g in optimizer.param_groups for p in g["params"]} == {
        id(p) for p in adapter.parameters()
    }
    head = freeze(
        torch.nn.Sequential(torch.nn.BatchNorm2d(256), torch.nn.Conv2d(256, 4, 1))
    )
    before_head = tensor_sha(head.state_dict())
    before_adapter = tensor_sha(adapter.state_dict())
    bev, text, position = inputs()
    for update in range(2):
        output = adapter(bev, text, position)
        task = head(output["enhanced_bev"]).square().mean()
        parameters = dict(adapter.named_parameters())
        grads = torch.autograd.grad(task, tuple(parameters.values()), retain_graph=True)
        norms = {
            k: sum(
                float(g.abs().sum())
                for (n, _), g in zip(parameters.items(), grads)
                if n.startswith(k)
            )
            for k in ("up.", "language.")
        }
        assert norms["up."] > 0
        assert norms["language."] == 0 if update == 0 else norms["language."] > 0
        (task + output["attention_logits"].square().mean()).backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    assert tensor_sha(adapter.state_dict()) != before_adapter
    assert tensor_sha(head.state_dict()) == before_head
    assert all(p.grad is None for p in head.parameters())
    assert all(int(v["step"]) == 2 for v in optimizer.state.values())


@pytest.mark.parametrize(
    "profile,frames,updates", [("main", 512, 256), ("short", 256, 64)]
)
def test_preregistered_splits(profile, frames, updates):
    config = validate_config(HERE / "configs" / (profile + ".json"))
    train, dev, diagnostic = (
        selected_rows(config, r) for r in ("train", "development", "diagnostic")
    )
    assert [len(train), len(dev), len(diagnostic)] == [frames, 320, 128]
    assert config["updates"] == updates
    for first, second in ((train, dev), (train, diagnostic), (dev, diagnostic)):
        assert not {r["log_token"] for r in first} & {r["log_token"] for r in second}
        assert not {r["sample_token"] for r in first} & {
            r["sample_token"] for r in second
        }


def test_cold_final_denied_before_read(config, monkeypatch):
    import core

    monkeypatch.setattr(core, "read", lambda *_: pytest.fail("Cold Final触发读取"))
    with pytest.raises(ValueError, match="Cold Final"):
        selected_rows(config, "cold_final")


@pytest.mark.parametrize("damage", ["duplicate", "reorder", "log"])
def test_metadata_damage_rejected(config, tmp_path, damage):
    import json

    changed = copy.deepcopy(config)
    proposal = read(config["subset_manifest"])
    scene = proposal["训练候选"][0]
    if damage == "duplicate":
        scene["sample_tokens"][1] = scene["sample_tokens"][0]
    elif damage == "reorder":
        scene["sample_tokens"] = scene["sample_tokens"][::-1]
    else:
        scene["log_token"] = "错误日志"
    path = tmp_path / "subset.json"
    path.write_text(json.dumps(proposal))
    changed["subset_manifest"] = str(path)
    with pytest.raises(ValueError):
        selected_rows(changed, "train")


def test_formal_sources_and_gate_unchanged(config):
    verify_sources(config)
    status = read(FORMAL / "status.json")
    assert status["status"] == "FAILED"
    assert status["can_resume"] is False
    gate = read(FORMAL / "warmup_gate.json")
    assert "GROUNDING_NOT_READY" in str(gate)


@pytest.mark.parametrize("action", ["preflight", "train", "evaluate", "_worker"])
def test_gpu_entry_without_authorization_does_not_start(action):
    existed = OUTPUT.exists()
    result = subprocess.run(
        [sys.executable, "-B", str(HERE / "cli.py"), action],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 2 and "未授权GPU" in result.stderr
    assert OUTPUT.exists() == existed

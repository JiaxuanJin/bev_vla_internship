"""只读复核 smoke checkpoint 的真实双 Head 反向，不增加优化器更新。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.stage3.lg_bev_common import DEFAULT_OUT, read, sha, verify_protocol, write
from scripts.stage3.lg_bev_native import (
    dual_head,
    gradient_audit,
    load_native,
    module_sha,
    native_losses,
)
from scripts.stage3.lg_bev_prototype import (
    LanguageGroundedBEVAdapter,
    attention_supervision_loss,
)
from scripts.stage3.lg_bev_data import load_supervision
from scripts.stage3.lg_bev_train import encode, frame_input, training_frame


def verify(out: Path) -> None:
    """重新前向和反向，核查冻结不变量；禁止创建优化器或重新训练。"""
    protocol = verify_protocol(out)
    smoke = read(out / "smoke_test.json")
    checkpoint = Path(smoke["checkpoint"]["path"])
    assert sha(checkpoint) == smoke["checkpoint"]["sha256"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["protocol_sha256"] == sha(out / "protocol.json")
    assert saved["optimizer_updates"] == 2
    instruction_id = read(out / "smoke/schedule.json")[0]
    row = next(
        r
        for r in read(out / "manifest.json")["annotations"]
        if r["instruction_id"] == instruction_id
    )
    assert row["split"] == "train" and row["task"] == "object"
    model, _, encoder, dataset = load_native()
    before = dict(author=module_sha(model), text=module_sha(encoder))
    text = encode(encoder, [row])
    adapter = LanguageGroundedBEVAdapter(**protocol["model_config"]).cuda().eval()
    adapter.load_state_dict(saved["state_dict"])
    bev, position = frame_input(row)
    detail = adapter(bev, text[row["text"]], position)
    prediction = dual_head(model, detail["enhanced_bev"])
    gt, _ = training_frame(row)
    losses = native_losses(model, prediction, dataset.get_ann_info(row["index"]), gt)
    losses.pop("filter_stats")
    label = load_supervision(row, out, training=True)
    losses["mask"] = attention_supervision_loss(
        detail["attention_logits"],
        label["target"].cuda(),
        label["distractor"].cuda(),
        valid_mask=label["valid"].cuda(),
    )["total"]
    gradients = {name: gradient_audit(loss, adapter) for name, loss in losses.items()}
    for report in gradients.values():
        assert report["total_l2"] > 0 and report["all_finite"]
    sum(losses.values()).backward()
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    assert before == dict(author=module_sha(model), text=module_sha(encoder))
    write(
        out / "verification_backward.json",
        dict(
            status="PASS",
            optimizer_updates_this_verification=0,
            saved_checkpoint_updates=2,
            sample_token=row["sample_token"],
            gradients=gradients,
            frozen_sha256=before,
            checkpoint_sha256=sha(checkpoint),
            verification_source_sha256=sha(__file__),
            scope="one training frame, real cached BEV and native CUDA double heads; no pilot/dev evaluation",
        ),
    )
    print("LG_READONLY_BACKWARD_VERIFICATION_PASS updates=0", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    verify(parser.parse_args().out.resolve())

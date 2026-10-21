"""Step69：第二阶段统一多任务损失的单测和真实项目入口审计。

功能：
1. 验证检测头 Loss 格式、多类别分割、注意力监督及三路梯度。
2. 检查工作目录是否有作者检测模型/分割模型的相关证据。
3. 明确区分模块单测通过和真实检测+分割联合前向尚未验收。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2.multitask_training import (  # noqa: E402
    LossConfig,
    Stage2MultiTaskLoss,
    aggregate_detection_losses,
    supervised_attention_loss,
)

OUT = ROOT / "outputs/stage2_step69"
DOC = ROOT / "docs/stage2_step69_multitask_loss_CN.md"


def assert_raises(exc_type: type[Exception], call: Any) -> None:
    """测试异常输入被正确拒绝，而不是继续带病训练。"""
    try:
        call()
    except exc_type:
        return
    raise AssertionError(f"期待 {exc_type.__name__}，却没有抛出异常")


def self_test() -> dict[str, Any]:
    """CPU 合成测试：固定随机种子，所有测试项均可复现。"""
    torch.manual_seed(20261021)
    tests: dict[str, str] = {}
    detector = nn.Conv2d(3, 2, kernel_size=1)
    segmenter = nn.Conv2d(3, 4, kernel_size=1)
    attention = nn.Conv2d(3, 1, kernel_size=1)
    x = torch.randn(2, 3, 12, 16)

    # 真实 MMDetection 中 decoder loss 可能返回 loss_cls 的列表。
    det_out = detector(x)
    det_losses = {
        "loss_cls": [det_out.square().mean(), det_out.abs().mean()],
        "d0.loss_bbox": det_out.sum().square() / det_out.numel() ** 2,
        "acc": torch.tensor(0.98),
    }
    val = aggregate_detection_losses(det_losses)
    expected = (
        det_losses["loss_cls"][0]
        + det_losses["loss_cls"][1]
        + det_losses["d0.loss_bbox"]
    )
    assert torch.allclose(val, expected)
    tests["mmdet_loss_dict_and_metric_filter"] = "PASS"

    gt = torch.zeros(2, 12, 16, dtype=torch.long)
    gt[0, 2:4, 2:10] = 1
    gt[1, 4:9, 6:11] = 2
    logits = segmenter(x)
    attention_logits = attention(x)
    target_classes = torch.tensor([1, 2], dtype=torch.long)
    total, parts = Stage2MultiTaskLoss(LossConfig())(
        det_losses, logits, gt, attention_logits, target_classes
    )
    assert set(parts) == {
        "detection_loss",
        "segmentation_loss",
        "attention_loss",
        "total_loss",
    }
    assert all(torch.isfinite(part).all() for part in parts.values())
    total.backward()
    assert all(
        item.weight.grad is not None and item.weight.grad.abs().sum() > 0
        for item in (detector, segmenter, attention)
    )
    tests["three_branch_differentiability"] = "PASS"

    # 注意力维度错误必须报错，禁止直接 resize 当成物理配准。
    assert_raises(
        ValueError,
        lambda: supervised_attention_loss(
            torch.randn(2, 1, 8, 8), gt, target_classes, LossConfig()
        ),
    )
    tests["coordinate_size_mismatch_rejected"] = "PASS"

    assert_raises(ValueError, lambda: aggregate_detection_losses({"acc": val}))
    assert_raises(
        ValueError,
        lambda: aggregate_detection_losses({"loss_cls": torch.tensor(float("nan"))}),
    )
    tests["missing_or_nonfinite_detection_loss_rejected"] = "PASS"

    empty_gt = torch.zeros(2, 12, 16, dtype=torch.long)
    empty_att = supervised_attention_loss(
        torch.randn(2, 1, 12, 16, requires_grad=True),
        empty_gt,
        target_classes,
        LossConfig(),
    )
    assert abs(float(empty_att.detach())) == 0.0
    tests["empty_target_does_not_invent_supervision"] = "PASS"

    assert_raises(
        ValueError,
        lambda: Stage2MultiTaskLoss()(
            {}, logits.detach(), gt, attention_logits.detach(), target_classes
        ),
    )
    tests["no_fake_joint_training"] = "PASS"
    return {"status": "PASS", "tests": tests, "torch_version": torch.__version__}


def _digest(path: Path) -> str:
    """为参考权重计算 sha256；大模型只在存在时执行。"""
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def project_audit() -> dict[str, Any]:
    """只读核对现有实机链路：存在不代表能跑通联合训练。"""
    candidates = {
        "stage1_detection_config": ROOT / "BEVFormer/projects/configs/bevformer/bevformer_small.py",
        "stage1_detection_weights": ROOT / "BEVFormer/ckpts/bevformer_small_epoch_24.pth",
        "stage2_segmentation_config": ROOT / "BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py",
        "stage2_segmentation_weights": Path("/mnt/g/nus/bevformer_small_seg.pth"),
        "stage2_live_seg_report": ROOT / "outputs/stage2_step58/step58_author_live_smoke.json",
        "stage2_frozen_train_report": ROOT / "outputs/stage2_step59/step59_training_summary.json",
        "stage2_text_intervention_report": ROOT / "outputs/stage2_step68/step68_acceptance.json",
        "stage1_detection_eval": ROOT / "outputs/drivelm_check/det799.run.log",
    }
    files: dict[str, Any] = {}
    for name, path in candidates.items():
        item: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        if path.is_file() and path.suffix in {".py", ".json"}:
            item["sha256"] = _digest(path)
        files[name] = item
    return {
        "files": files,
        "unified_loss_interface": "testable",
        "real_detection_seg_attention_shared_forward": "NOT_VERIFIED",
        "real_joint_backprop": "NOT_VERIFIED",
        "reason": (
            "当前两条模型路径和独立权重不等于一个真实共享 BEV 联合训练模型；"
            "需在 Step70 将同帧检测头接入融合特征并由其返回实际 loss。"
        ),
    }


def write_report(result: dict[str, Any]) -> None:
    """将可追溯结果保存为 JSON 和中文报告。"""
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    path = OUT / "step69_loss_contract.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    tests = result["self_test"]["tests"]
    lines = [
        "# Step69：检测 + 分割 + 注意力多任务 Loss 模块与单元测试",
        "",
        "## 模块功能",
        "",
        "- 兼容 MMDetection 检测头 `loss_*`、`d0.loss_*` 和 decoder loss 列表。",
        "- 计算地图分割四类 CrossEntropy 与按类别指令构造的注意力 BCE / Dice。",
        "- 对检测、分割、注意力三条分支分别进行反向传播检查。",
        "- 注意力坐标尺寸不一致时拒绝执行，避免直接插值掩盖坐标问题。",
        "- 未提供检测头真实 loss 时拒绝伪造联合损失。",
        "",
        "## 单元测试结果（CPU 合成数据，非实机检测头）",
        "",
        "| 项目 | 结果 |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {status} |" for name, status in tests.items())
    lines += [
        "",
        "## 阶段二系统集成边界",
        "",
        "- 分割链路真实运行证据：Step58、Step59、Step68。",
        "- 现阶段检测与分割分别使用作者不同实验配置和权重。",
        "- 单元测试通过**仅说明统一损失接口功能正确**，不代表检测头已经同帧接入。",
        "- 真实共享 BEV 的检测头 + 分割头 + 注意力联合反传仍标记 NOT_VERIFIED。",
        "- 物理坐标与复杂自然语言语义验证未完成，不影响本接口的单元测试结论。",
        "",
        "## 代码与结果位置",
        "",
        "- `scripts/stage2/multitask_training.py`",
        "- `scripts/stage2/run_step69_loss_contract.py`",
        "- `outputs/stage2_step69/step69_loss_contract.json`",
        "- `docs/stage2_step69_multitask_loss_CN.md`",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Step69] JSON: {path}")
    print(f"[Step69] 中文报告: {DOC}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Step69 多任务损失接口验收")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--audit", action="store_true", help="检查本地代码与权重文件")
    args = parser.parse_args()
    result = {"self_test": self_test()}
    print(f"[Step69] {len(result['self_test']['tests'])} 项单元测试 PASS")
    if not args.self_test or args.audit:
        result["project_audit"] = project_audit()
        write_report(result)
        print("[Step69] 真实检测/分割/注意力联合前向：NOT_VERIFIED")


if __name__ == "__main__":
    main()

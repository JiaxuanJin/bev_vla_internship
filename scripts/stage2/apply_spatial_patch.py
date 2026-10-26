#!/usr/bin/env python3
"""为历史 Loss 增加 opt-in 空间标注分支；默认只检查，不修改文件。"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "scripts/stage2/multitask_training.py"

CHANGES = [
    (
        "    target_classes: Tensor,\n    cfg: LossConfig,\n) -> Tensor:\n",
        "    target_classes: Tensor,\n    cfg: LossConfig,\n    *,\n"
        "    spatial_targets: Sequence[Any] | None = None,\n) -> Tensor:\n",
        1,
    ),
    (
        '    if attention_logits.ndim != 4 or attention_logits.shape[1] != 1:\n',
        '    if spatial_targets is not None:\n'
        '        from .spatial_supervision import spatial_attention_loss\n\n'
        '        if target_classes.shape != (segmentation_gt.shape[0],):\n'
        '            raise ValueError("target_classes 必须逐帧指定")\n'
        '        if target_classes.dtype != torch.long or (\n'
        '            target_classes.device != attention_logits.device\n'
        '        ):\n'
        '            raise ValueError("指令类别 dtype/设备错误")\n'
        '        if len(spatial_targets) != segmentation_gt.shape[0]:\n'
        '            raise ValueError("实例标注数量必须等于 batch")\n'
        '        if any(\n'
        '            item.target_class != int(target_classes[i].item())\n'
        '            for i, item in enumerate(spatial_targets)\n'
        '        ):\n'
        '            raise ValueError("显式实例标签与指令类别不一致")\n'
        '        return spatial_attention_loss(\n'
        '            attention_logits, segmentation_gt, spatial_targets,\n'
        '            ignore_index=cfg.ignore_index,\n'
        '            dice_weight=cfg.attention_dice_weight,\n'
        '        )\n'
        '    if attention_logits.ndim != 4 or attention_logits.shape[1] != 1:\n',
        1,
    ),
    (
        "        target_classes: Tensor,\n        *,\n"
        "        regularization_logits: Tensor | None = None,\n",
        "        target_classes: Tensor,\n        *,\n"
        "        spatial_targets: Sequence[Any] | None = None,\n"
        "        regularization_logits: Tensor | None = None,\n",
        1,
    ),
    (
        "        att = supervised_attention_loss(\n"
        "            attention_logits, segmentation_gt, target_classes, self.config\n"
        "        )\n",
        "        att = supervised_attention_loss(\n"
        "            attention_logits, segmentation_gt, target_classes, self.config,\n"
        "            spatial_targets=spatial_targets,\n"
        "        )\n",
        1,
    ),
]


def main() -> None:
    """检查锚点与 Git 状态，通过显式 --apply 才执行修改。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true", help="修改代码并保留 .bak 文件")
    args = p.parse_args()
    if not TARGET.is_file():
        raise SystemExit(f"找不到目标源码：{TARGET}")
    source = TARGET.read_text(encoding="utf-8")
    if "spatial_targets: Sequence[Any] | None = None" in source:
        raise SystemExit("目标已经有空间监督扩展；停止重复修改")
    if not (ROOT / "scripts/stage2/spatial_supervision.py").is_file():
        raise SystemExit("必须先解压本补丁的 spatial_supervision.py")

    check = subprocess.run(
        ["git", "status", "--porcelain", "--", "scripts/stage2/multitask_training.py"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    if check.stdout.strip():
        raise SystemExit(
            "现有 multitask_training.py 有未提交修改，停止以防覆盖；先人工核对差异"
        )
    revised = source
    for old, new, count in CHANGES:
        if revised.count(old) != count:
            raise SystemExit(f"补丁锚点不唯一，原代码可能已变化：{old[:66]!r}")
        revised = revised.replace(old, new, count)
    print("[PASS] 历史 Loss 结构与所有补丁锚点匹配")
    print("[INFO] 不改变旧调用的默认类别级损失公式")
    if not args.apply:
        print("[CHECK ONLY] 若确认可修改，执行同一命令加 --apply")
        return
    backup = ROOT / ".git/stage2_20261026_multitask_training.py.bak"
    if backup.exists():
        raise SystemExit(f"备份文件已存在，拒绝覆盖：{backup}")
    if not backup.parent.is_dir():
        raise SystemExit("仓库 .git/ 目录不存在，不能安全保存备份")
    shutil.copy2(TARGET, backup)
    TARGET.write_text(revised, encoding="utf-8")
    print(f"[APPLIED] 更新：{TARGET}")
    print(f"[BACKUP] 备份：{backup}")
    print("[NEXT] 运行新增单测与既有回归；未做真实数据训练")


if __name__ == "__main__":
    main()

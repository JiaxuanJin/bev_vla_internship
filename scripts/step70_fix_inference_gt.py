#!/usr/bin/env python3
"""修复 Step70 真实推理时 GT 被误传入作者模型的问题。

只修改本项目 Step70 脚本，不接触第三方作者源码、checkpoint 和 GT。
"""

from __future__ import annotations

import ast
import argparse
from pathlib import Path


def patch_script(target: Path) -> None:
    """检查并修补 Step70 推理调用，同时保存原文件备份。

    Args:
        target: 当前项目中的 Step70 真实集成脚本。
    """
    if not target.is_file():
        raise FileNotFoundError(f"未找到脚本：{target}")

    original = target.read_text(encoding="utf-8")
    fixed_marker = 'inference_data.pop("semantic_indices", None)'
    if fixed_marker in original:
        print("[Step70 修复] 已处理 semantic_indices，无需重复修改")
        return

    old_prepare = (
        '            try:\n'
        '                with torch.inference_mode():\n'
        '                    result = wrapped(\n'
        '                        return_loss=False, rescale=True, **copy.deepcopy(data)\n'
        '                    )'
    )
    new_prepare = (
        '            try:\n'
        '                # GT 留在 data 中用于评估，不传给作者的 simple_test。\n'
        '                inference_data = copy.deepcopy(data)\n'
        '                inference_data.pop("semantic_indices", None)\n'
        '                with torch.inference_mode():\n'
        '                    result = wrapped(\n'
        '                        return_loss=False, rescale=True, **inference_data\n'
        '                    )'
    )
    if original.count(old_prepare) != 1:
        raise RuntimeError(
            "未找到预期的 Step70 推理代码或脚本已发生变更；"
            "为了安全没有执行自动修改。"
        )

    new_source = original.replace(old_prepare, new_prepare, 1)
    ast.parse(new_source, filename=str(target))
    backup = target.with_suffix(target.suffix + ".before_semantic_fix.bak")
    if backup.exists():
        raise FileExistsError(f"备份文件已经存在，不覆盖：{backup}")
    backup.write_text(original, encoding="utf-8")
    try:
        target.write_text(new_source, encoding="utf-8")
    except BaseException:
        target.write_text(original, encoding="utf-8")
        raise

    print("[Step70 修复] 已过滤推理参数 semantic_indices")
    print("[Step70 修复] GT 数据本身未修改")
    print(f"[Step70 修复] 脚本备份：{backup}")
    print(f"[Step70 修复] 脚本文件：{target}")


def main() -> None:
    """从项目根目录运行，定位并修改现有 Step70 脚本。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", type=Path,
        default=Path("scripts/stage2/run_step70_joint_integration.py"),
    )
    args = parser.parse_args()
    patch_script(args.target)


if __name__ == "__main__":
    main()

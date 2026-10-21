"""为旧 MMDetection3D 在 Step72 入口添加局部 NumPy 兼容别名。"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MARKER = "# STEP72_NUMPY_LEGACY_COMPAT"
COMPAT = '''\n# STEP72_NUMPY_LEGACY_COMPAT\n# 仅在别名缺失时补齐，适配作者旧版 MMDetection3D 数据流水线。\n# 不修改 NumPy 安装文件、第三方源码和模型权重。\nfor _legacy_name, _replacement in (\n    ("bool", np.bool_),\n    ("int", int),\n    ("float", float),\n    ("complex", complex),\n):\n    if _legacy_name not in np.__dict__:\n        setattr(np, _legacy_name, _replacement)\ndel _legacy_name, _replacement\n'''


def patch(target: Path) -> None:
    """只改 Step72 的 NumPy 导入附近，重复执行不会重复插入。"""
    if not target.is_file():
        raise FileNotFoundError(f"找不到 Step72 文件：{target}")
    source = target.read_text(encoding="utf-8")
    if MARKER in source:
        print("[Step72 修复] NumPy 兼容层已存在，跳过重复修改")
        return
    needle = "import numpy as np\n"
    if source.count(needle) != 1:
        raise RuntimeError("无法唯一定位 NumPy 导入语句，未修改源码")
    backup = target.with_suffix(target.suffix + ".before_numpy_fix.bak")
    if not backup.exists():
        shutil.copy2(target, backup)
    amended = source.replace(needle, needle + COMPAT, 1)
    compile(amended, str(target), "exec")
    target.write_text(amended, encoding="utf-8")
    print("[Step72 修复] NumPy 旧别名兼容层已安装")
    print(f"[Step72 修复] 原脚本备份：{backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("scripts/stage2/run_step72_real_joint_gradient.py"),
    )
    args = parser.parse_args()
    patch(args.target)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""只读审计根目录 stage2 包与 scripts/stage2 包的引用和重名文件。"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    """计算文件摘要，不能仅凭重名推断内容完全相同。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """枚举真正的 Python import 依赖及静态配置引用。"""
    a = ROOT / "stage2"
    b = ROOT / "scripts/stage2"
    print("[A] 早期兼容目录：", a)
    print("[B] 当前实验模块：", b)
    if not a.is_dir() or not b.is_dir():
        raise SystemExit("缺少目录，检查项目根路径")
    overlaps = sorted({p.name for p in a.glob("*.py")} & {p.name for p in b.glob("*.py")})
    for name in overlaps:
        print(f"重名文件 {name}: 内容相同={sha256(a / name) == sha256(b / name)}")
    matches: list[str] = []
    # 仅扫描项目自有源码，避免遍历 BEVFormer 第三方仓库与虚拟环境。
    own_paths = [
        *ROOT.glob("scripts/**/*.py"),
        *ROOT.glob("tests/**/*.py"),
        *ROOT.glob("stage2/**/*.py"),
    ]
    for path in sorted(set(own_paths)):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            if any(name == "stage2" or name.startswith("stage2.") for name in modules):
                matches.append(f"{path.relative_to(ROOT)}:{node.lineno} {modules}")
    print("\n根目录 stage2 的绝对导入：")
    print("\n".join(matches) if matches else "未在可解析 Python 源码中发现")
    p = ROOT / "scripts/stage2/run_stage2.py"
    if p.is_file() and "stage2/tokenizer_config.json" in p.read_text(encoding="utf-8"):
        print("[保留原因] scripts/stage2/run_stage2.py 引用了 stage2/tokenizer_config.json")
    print("[结论] 本命令只审计，不删除、合并或修改任何文件")


if __name__ == "__main__":
    main()

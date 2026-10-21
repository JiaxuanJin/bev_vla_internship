"""Step63：检查 Step62 输出的 NPZ 张量结构，为空间对齐诊断做准备。

脚本只读取文件，不改模型、GT、权重或历史报告。依赖 NumPy。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def inspect_array(key: str, value: np.ndarray) -> dict[str, Any]:
    """安全记录数组的名称、形状和数值范围，不输出原始数据。"""
    record: dict[str, Any] = {
        "name": key,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }
    if np.issubdtype(value.dtype, np.number) or np.issubdtype(
        value.dtype, np.bool_
    ):
        if value.size:
            sample = value.ravel()
            record["min"] = float(sample.min())
            record["max"] = float(sample.max())
            record["mean"] = float(sample.mean())
            if np.issubdtype(value.dtype, np.floating):
                record["all_finite"] = bool(np.isfinite(value).all())
    return record


def main() -> None:
    """打印三类样本中的张量结构并保存 JSON。"""
    parser = argparse.ArgumentParser(description="Step63 只读 NPZ 结构核查")
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/stage2_step62")
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    paths = sorted(root.glob("step62_*_sample.npz"))
    if not paths:
        raise FileNotFoundError(f"未发现 Step62 NPZ 文件：{root}")

    report: dict[str, Any] = {"source": str(root), "files": []}
    for path in paths:
        print(f"\n=== {path.name} ===", flush=True)
        entry: dict[str, Any] = {"file": path.name, "arrays": []}
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays.files:
                meta = inspect_array(key, arrays[key])
                entry["arrays"].append(meta)
                print(
                    f"  {key}: shape={meta['shape']} dtype={meta['dtype']} "
                    f"range=[{meta.get('min', 'N/A')}, {meta.get('max', 'N/A')}]",
                    flush=True,
                )
        report["files"].append(entry)

    output = root / "step63_npz_structure.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Step63] 结构审计已保存：{output}", flush=True)
    print("[Step63] 只读检查完成；没有进行调参、坐标校正或训练。")


if __name__ == "__main__":
    main()

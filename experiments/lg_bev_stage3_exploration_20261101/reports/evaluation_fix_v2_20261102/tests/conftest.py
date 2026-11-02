"""修复专用 CPU 测试，不初始化 CUDA，不读取 Development GT。"""

import os
from pathlib import Path
import sys

REPORT = Path(__file__).resolve().parents[1]
ORIGINAL = REPORT.parents[1]
sys.path[:0] = [
    str(REPORT / "source"),
    str(ORIGINAL),
    str(ORIGINAL.parents[1]),
]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

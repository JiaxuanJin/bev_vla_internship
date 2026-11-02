"""修复版本没有重新绑定授权时，必须在创建 Worker 前拒绝。"""

from pathlib import Path
import subprocess
import sys


def test_v2_development_stays_closed_without_new_authorization():
    report = Path(__file__).resolve().parents[1]
    assert not (report / "authorization.approved.json").exists()
    result = subprocess.run(
        [sys.executable, "-B", str(report / "source/entry.py"), "development"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "修复版本 v2 尚未重新绑定授权" in result.stderr

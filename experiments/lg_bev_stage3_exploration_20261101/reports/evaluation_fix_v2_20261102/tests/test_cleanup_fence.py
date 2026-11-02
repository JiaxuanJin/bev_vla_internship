"""真实 dir_fd 清理路径及越界拒绝测试；审计钩子仅装入子进程。"""

from pathlib import Path
import os
import subprocess
import sys


def test_exception_cleanup_and_directory_fd_fence(tmp_path):
    report = Path(__file__).resolve().parents[1]
    original = report.parents[1]
    script = r"""
import os
from pathlib import Path
import tempfile
import sys
from runtime_v2 import install_write_fence

base = Path(sys.argv[1])
allowed = base / "allowed"
outside = base / "outside"
allowed.mkdir()
outside.mkdir()
(outside / "sentinel").write_text("preserve")
outside_fd = os.open(outside, os.O_RDONLY | os.O_DIRECTORY)
install_write_fence(allowed)
try:
    with tempfile.TemporaryDirectory(dir=allowed) as directory:
        temporary = Path(directory)
        (temporary / "generated.py").write_text("pass")
        raise RuntimeError("模拟初始化异常")
except RuntimeError:
    pass
assert not temporary.exists(), "异常退出没有清理临时目录"
for path, descriptor in (("sentinel", outside_fd), ("../outside/sentinel", None)):
    try:
        if descriptor is None:
            os.unlink(allowed / path)
        else:
            os.unlink(path, dir_fd=descriptor)
    except ValueError:
        pass
    else:
        raise AssertionError("越界删除没有被拒绝")
assert (outside / "sentinel").read_text() == "preserve"
fd = os.open(allowed, os.O_RDONLY | os.O_DIRECTORY)
(allowed / "a").write_text("ok")
os.rename("a", "b", src_dir_fd=fd, dst_dir_fd=fd)
os.link("b", "c", src_dir_fd=fd, dst_dir_fd=fd)
os.unlink("b", dir_fd=fd)
os.unlink("c", dir_fd=fd)
os.close(fd)
os.close(outside_fd)
print("清理及边界验证通过")
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(tmp_path)],
        env=dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                (str(report / "source"), str(original), str(original.parents[1]))
            ),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Exception ignored" not in result.stderr

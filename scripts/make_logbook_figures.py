
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess
import sys
import textwrap

# 获取项目总目录
ROOT = Path(__file__).resolve().parents[1]
BEV = ROOT / "BEVFormer"
OUT = ROOT / "docs/images/2026-10-10"
LOG = ROOT / "build_logs"

OUT.mkdir(parents=True, exist_ok=True)
LOG.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable

# CUDA测试代码
cuda_code = """
import torch
print("PyTorch:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
x = torch.rand(100, 100, device="cuda")
y = x @ x
print("Result Device:", y.device)
print("CUDA GPU TEST PASSED")
"""

# nuScenes数据读取测试
nuscenes_code = """
from nuscenes.nuscenes import NuScenes
from pathlib import Path

nusc = NuScenes(
    version="v1.0-mini",
    dataroot="data/nuscenes",
    verbose=False
)
sample = nusc.sample[0]
token = sample["data"]["CAM_FRONT"]
path = nusc.get_sample_data_path(token)

print("Scenes:", len(nusc.scene))
print("Samples:", len(nusc.sample))
print("First Scene:", nusc.scene[0]["name"])
print("Image exists:", Path(path).is_file())

assert len(nusc.scene) == 10
assert len(nusc.sample) == 404
assert Path(path).is_file()
print("NUSCENES DATASET LOAD PASSED")
"""

# 五个真实测试
tests = [
    (
        "01_cuda_gpu",
        "PyTorch + CUDA GPU Test",
        ROOT,
        [PYTHON, "-c", cuda_code],
        "CUDA GPU TEST PASSED",
        ["PyTorch:", "CUDA Version:", "CUDA Available:",
         "GPU:", "Result Device:", "PASSED"]
    ),
    (
        "02_mmcv_attention",
        "MMCV CUDA Attention Test",
        ROOT,
        [PYTHON, "test_mmcv_cuda.py"],
        "MMCV CUDA FORWARD/BACKWARD PASSED",
        ["GPU:", "Forward shape:", "Output device:",
         "Max numerical error:", "Backward completed",
         "gradient:", "PASSED"]
    ),
    (
        "03_bevformer_build",
        "BEVFormer-Small Model Build",
        BEV,
        [PYTHON, "test_bevformer_build.py"],
        "BEVFORMER GPU TRANSFER PASSED",
        ["Model:", "Backbone:", "Neck:", "Detection head:",
         "Total parameters:", "Parameters (million):",
         "BEV grid:", "Model device:", "Allocated GPU",
         "PASSED"]
    ),
    (
        "04_checkpoint_loading",
        "BEVFormer Pretrained Weights",
        BEV,
        [PYTHON, "test_bevformer_weights.py"],
        "BEVFORMER WEIGHTS LOAD PASSED",
        ["Missing keys:", "Unexpected keys:",
         "Model parameters:", "Loaded parameters:",
         "Parameter coverage:", "Model device:",
         "GPU memory", "PASSED"]
    ),
    (
        "05_nuscenes_loading",
        "nuScenes Mini Dataset Test",
        BEV,
        [PYTHON, "-c", nuscenes_code],
        "NUSCENES DATASET LOAD PASSED",
        ["Scenes:", "Samples:", "First Scene:",
         "Image exists:", "PASSED"]
    )
]

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
try:
    font = ImageFont.truetype(font_path, 21)
    title_font = ImageFont.truetype(font_path, 27)
except OSError:
    font = ImageFont.load_default()
    title_font = font

for name, title, cwd, command, success, keywords in tests:
    print("Running:", title, flush=True)

    # 实际执行测试程序
    result = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace"
    )

    # 保存完整原始日志
    (LOG / f"{name}.log").write_text(
        result.stdout,
        encoding="utf-8"
    )

    # 只有真实测试成功，才生成成果图片
    if result.returncode != 0 or success not in result.stdout:
        print("FAILED - check log:", name)
        continue

    # 提取关键输出
    lines = [
        line.strip()
        for line in result.stdout.splitlines()
        if any(k in line for k in keywords)
    ]

    display_lines = []
    for line in lines:
        display_lines.extend(
            textwrap.wrap(line, width=80) or [""]
        )

    # 生成终端风格图片
    width = 1200
    height = max(250, 140 + len(display_lines) * 34)

    image = Image.new("RGB", (width, height), "#101820")
    draw = ImageDraw.Draw(image)

    draw.text((35, 28), title,
              font=title_font, fill="#78c6ff")
    draw.line((35, 80, width - 35, 80),
              fill="#52606d", width=2)

    y = 105
    for line in display_lines:
        color = "#6ee7a7" if "PASSED" in line else "#eeeeee"
        draw.text((35, y), line, font=font, fill=color)
        y += 34

    path = OUT / f"{name}.png"
    image.save(path)

    print("SAVED:", path)

print("===== ALL TESTS FINISHED =====")
print("Images:", OUT)
print("Logs:", LOG)

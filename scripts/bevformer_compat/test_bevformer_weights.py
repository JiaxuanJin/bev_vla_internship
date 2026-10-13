
import torch
from pathlib import Path
from mmcv import Config
from mmdet3d.models import build_model

# 1. 注册 BEVFormer 的自定义网络模块
import projects.mmdet3d_plugin

print("===== BEVFormer Weight Loading Test =====")

# 2. 定位项目目录和文件
root = Path(__file__).resolve().parent

config_path = root / "projects/configs/bevformer/bevformer_small.py"
checkpoint_path = root / "ckpts/bevformer_small_epoch_24.pth"

print("Checkpoint:", checkpoint_path)

# 3. 读取模型配置
cfg = Config.fromfile(str(config_path))

# 禁用额外的骨干网络预训练初始化
# 因为我们接下来加载完整的官方权重
cfg.model.pretrained = None
cfg.model.train_cfg = None

# 4. 创建完整 BEVFormer 网络
print("Building model...")

model = build_model(
    cfg.model,
    test_cfg=cfg.get("test_cfg")
)

# 5. 读取官方预训练权重到 CPU
print("Reading checkpoint...")

checkpoint = torch.load(
    str(checkpoint_path),
    map_location="cpu",
    weights_only=True
)

# 6. 提取真正的模型参数
if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# 兼容部分多 GPU 训练保存的参数前缀
state_dict = {
    name.removeprefix("module."): tensor
    for name, tensor in state_dict.items()
}

# 7. 把参数加载进模型
print("Loading model weights...")

result = model.load_state_dict(
    state_dict,
    strict=False
)

# 8. 检查参数匹配情况
missing = result.missing_keys
unexpected = result.unexpected_keys

print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

# 输出最多 10 条不匹配参数，方便排查
if missing:
    print("Missing examples:", missing[:10])

if unexpected:
    print("Unexpected examples:", unexpected[:10])

# 9. 计算成功匹配的可训练参数比例
parameters = dict(model.named_parameters())

total = sum(p.numel() for p in parameters.values())

loaded = sum(
    p.numel()
    for name, p in parameters.items()
    if name in state_dict
    and p.shape == state_dict[name].shape
)

coverage = loaded / total * 100

print("Model parameters:", total)
print("Loaded parameters:", loaded)
print("Parameter coverage:", round(coverage, 4), "%")

# 参数不完全匹配时停止，避免误判成功
if missing or unexpected:
    print("CHECKPOINT PARTIAL LOAD - REVIEW REQUIRED")
    raise SystemExit(1)

# 10. 把已加载权重的模型放到 RTX 5080
model = model.to("cuda")
model.eval()

torch.cuda.synchronize()

print("Model device:", next(model.parameters()).device)
print("GPU memory (MB):",
      round(torch.cuda.memory_allocated() / 1024**2, 2))

print("BEVFORMER WEIGHTS LOAD PASSED")


import torch
from mmcv import Config
from mmdet3d.models import build_model

# 第1步：导入 BEVFormer 自己定义的网络模块
# 这一步会把 BEVFormer 注册到模型构建器中
import projects.mmdet3d_plugin

print("===== BEVFormer-Small Build Test =====")
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# 第2步：读取 Small 模型配置
config_path = "projects/configs/bevformer/bevformer_small.py"
cfg = Config.fromfile(config_path)

print("Model type:", cfg.model["type"])

# 第3步：仅测试网络结构，不加载预训练权重
cfg.model.pretrained = None
cfg.model.train_cfg = None

# 第4步：真正创建完整神经网络
print("Building model...")
model = build_model(
    cfg.model,
    test_cfg=cfg.get("test_cfg")
)

# 第5步：切换到评估模式
model.eval()

# 第6步：查看创建出来的网络
print("Model:", type(model).__name__)
print("Backbone:", type(model.img_backbone).__name__)
print("Neck:", type(model.img_neck).__name__)
print("Detection head:", type(model.pts_bbox_head).__name__)

# 第7步：统计模型参数数量
total_params = sum(p.numel() for p in model.parameters())
print("Total parameters:", total_params)
print("Parameters (million):", round(total_params / 1e6, 2))

# 第8步：检查 BEV 网格大小
bev_h = model.pts_bbox_head.bev_h
bev_w = model.pts_bbox_head.bev_w

print("BEV grid:", bev_h, "x", bev_w)

assert type(model).__name__ == "BEVFormer"
assert bev_h == 150 and bev_w == 150

print("BEVFORMER SMALL MODEL BUILD PASSED")

# 第9步：把整个模型移动到 RTX 5080 GPU
print("Moving model to GPU...")

model = model.to("cuda")

# 等待 GPU 操作完成
torch.cuda.synchronize()

# 检查模型参数所在设备
device = next(model.parameters()).device

print("Model device:", device)

# 查看显存占用
memory_mb = torch.cuda.memory_allocated() / 1024**2
print("Allocated GPU memory (MB):", round(memory_mb, 2))

# 确认模型在 GPU 上
assert device.type == "cuda"

print("BEVFORMER GPU TRANSFER PASSED")

import torch
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction,
    multi_scale_deformable_attn_pytorch,
)

# 1. 指定 RTX 5080 GPU
device = "cuda"
torch.manual_seed(0)

# 2. 模拟图像特征
# [Batch, 特征位置数, 注意力头数, 每个头的维度]
value = torch.randn(
    1, 16, 2, 4,
    device=device,
    requires_grad=True
)

# 3. 一个 4x4 的特征图
spatial_shapes = torch.tensor(
    [[4, 4]], dtype=torch.long, device=device
)

level_start_index = torch.tensor(
    [0], dtype=torch.long, device=device
)

# 4. 五个 Query，每个头采样四个位置
sampling_locations = torch.rand(
    1, 5, 2, 1, 4, 2,
    device=device,
    requires_grad=True
)

# 5. 每个采样位置的注意力权重
attention_weights = torch.full(
    (1, 5, 2, 1, 4),
    0.25,
    device=device,
    requires_grad=True
)

# 6. 真正调用 CUDA 可变形注意力算子
output = MultiScaleDeformableAttnFunction.apply(
    value,
    spatial_shapes,
    level_start_index,
    sampling_locations,
    attention_weights,
    64
)

torch.cuda.synchronize()

print("GPU:", torch.cuda.get_device_name(0))
print("Forward shape:", output.shape)
print("Output device:", output.device)

# 7. 与纯 PyTorch 计算结果进行比较
reference = multi_scale_deformable_attn_pytorch(
    value.detach(),
    spatial_shapes.cpu(),
    sampling_locations.detach(),
    attention_weights.detach()
)

max_error = (output.detach() - reference).abs().max().item()
print("Max numerical error:", max_error)

# 8. 测试反向传播
loss = output.square().mean()
loss.backward()
torch.cuda.synchronize()

print("Backward completed")
print("Value gradient:", value.grad is not None)
print("Location gradient:", sampling_locations.grad is not None)
print("Weight gradient:", attention_weights.grad is not None)

assert torch.isfinite(output).all()
assert max_error < 1e-3

for grad in [value.grad, sampling_locations.grad, attention_weights.grad]:
    assert grad is not None
    assert torch.isfinite(grad).all()

print("MMCV CUDA FORWARD/BACKWARD PASSED")

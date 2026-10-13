"""
功能：BEVFormer-Small 单 GPU 推理兼容性修复
环境：PyTorch 2.8 + MMCV 1.7.2 + CUDA 12.8

修复内容：
1. 兼容新版 PyTorch 的 DDP Forward 接口
2. 修复 MMCV Scatter 传递整数 GPU 编号的问题

说明：
仅用于开发阶段的单 GPU 推理验证。
不修改原始 MMCV 安装文件及 BEVFormer 模型代码。
"""

import sys
import runpy
import torch

from mmcv.parallel import MMDistributedDataParallel
import mmcv.parallel._functions as mmcv_functions


# ==========================================
# 修复 1：兼容旧版 MMCV 的 DDP Forward
# ==========================================

def compatible_ddp_forward(self, *inputs, **kwargs):

    if self.device_ids:
        inputs, kwargs = self.to_kwargs(
            inputs, kwargs, self.device_ids[0]
        )

        return self.module(
            *inputs[0],
            **kwargs[0]
        )

    return self.module(*inputs, **kwargs)


MMDistributedDataParallel._run_ddp_forward = compatible_ddp_forward


# ==========================================
# 修复 2：GPU 编号转换为 torch.device
# ==========================================

original_get_stream = mmcv_functions._get_stream


def compatible_get_stream(device):

    # 旧版 MMCV 传入 int，例如 0
    # 新版 PyTorch 要求 torch.device("cuda:0")
    if isinstance(device, int):
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda", device)

    return original_get_stream(device)


mmcv_functions._get_stream = compatible_get_stream


# ==========================================
# 启动原始 BEVFormer 测试程序
# ==========================================

print("[PATCH 1] DDP Forward compatibility enabled.", flush=True)
print("[PATCH 2] GPU device compatibility enabled.", flush=True)

sys.argv[0] = "tools/test.py"

runpy.run_path("tools/test.py", run_name="__main__")

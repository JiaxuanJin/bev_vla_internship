"""从 BEVFormer-Small 的真实推理过程中提取 BEV Encoder 特征并生成可视化。"""

import json
import runpy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # WSL 无需图形桌面即可保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT / "outputs" / "bev_features"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
BEV_H, BEV_W, CHANNELS = 150, 150, 256
state = {"attempted": False, "success": False, "error": None}


def capture_encoder_output(module, inputs, output):
    """只捕获第一次真实 BEVFormerEncoder 前向计算，避免重复覆盖结果。"""
    if module.__class__.__name__ != "BEVFormerEncoder" or state["attempted"]:
        return
    state["attempted"] = True

    try:
        tensor = output
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Encoder 输出不是 Tensor：{type(tensor)}")

        original_shape = list(tensor.shape)
        # 可选的多层 Encoder 输出：取最后一层；Batch=1 时移除 Batch 维。
        while tensor.ndim > 2:
            tensor = tensor[0] if tensor.shape[0] == 1 else tensor[-1]

        if tuple(tensor.shape) == (BEV_H * BEV_W, CHANNELS):
            tensor = tensor.reshape(BEV_H, BEV_W, CHANNELS)
        elif tuple(tensor.shape) == (CHANNELS, BEV_H * BEV_W):
            tensor = tensor.T.reshape(BEV_H, BEV_W, CHANNELS)
        else:
            raise ValueError(f"意外的 Encoder 输出形状：{original_shape}，当前 {list(tensor.shape)}")

        feature = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
        if not np.isfinite(feature).all():
            raise ValueError("提取的 BEV 特征包含 NaN / Inf")

        # 真实特征数据（不是随机生成的模拟图）
        np.save(SAVE_DIR / "bev_feature.npy", feature)

        # 每个 BEV 网格位置的 256 维特征向量 L2 范数
        strength = np.linalg.norm(feature, axis=-1)
        plt.figure(figsize=(8, 7))
        plt.imshow(strength, origin="lower", cmap="viridis")
        plt.colorbar(label="Feature L2 norm")
        plt.xlabel("BEV grid X")
        plt.ylabel("BEV grid Y")
        plt.title("BEVFormer-Small: Real BEV Feature Strength")
        plt.tight_layout()
        plt.savefig(SAVE_DIR / "bev_feature_heatmap.png", dpi=200)
        plt.close()

        # 展示四个独立通道，说明同一 BEV 位置包含多维语义信息
        fig, axes = plt.subplots(2, 2, figsize=(10, 9))
        for ax, channel in zip(axes.flat, (0, 64, 128, 192)):
            im = ax.imshow(feature[:, :, channel], origin="lower", cmap="viridis")
            ax.set_title(f"Channel {channel}")
            fig.colorbar(im, ax=ax)
        fig.suptitle("BEVFormer-Small: Encoder Feature Channels")
        fig.tight_layout()
        fig.savefig(SAVE_DIR / "bev_feature_channels.png", dpi=200)
        plt.close(fig)

        info = {
            "model": "BEVFormer-Small",
            "capture_point": "BEVFormerEncoder forward output",
            "capture_note": "First encoder forward; last layer if intermediate layers are present",
            "original_shape": original_shape,
            "saved_shape": list(feature.shape),
            "dtype": str(feature.dtype),
            "min": float(feature.min()),
            "max": float(feature.max()),
            "mean": float(feature.mean()),
            "std": float(feature.std()),
        }
        (SAVE_DIR / "capture_info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        state["success"] = True
        print("\n========== BEV CAPTURE SUCCESS ==========", flush=True)
        print("Encoder 原始形状:", original_shape, flush=True)
        print("保存形状:", feature.shape, flush=True)
        print("结果目录:", SAVE_DIR, flush=True)
        print("=========================================\n", flush=True)
    except Exception as exc:
        state["error"] = str(exc)
        print(f"[BEV CAPTURE ERROR] {exc}", flush=True)


def main():
    compat = ROOT / "scripts" / "run_bevformer_test_compat.py"
    if not compat.is_file():
        raise FileNotFoundError(f"缺少昨天的兼容脚本：{compat}")

    # 全局前向 Hook，不改模型源码、不改预训练权重。
    hook = torch.nn.modules.module.register_module_forward_hook(capture_encoder_output)
    print("[BEV] Forward Hook registered.", flush=True)
    try:
        # 将命令行参数原封不动交给昨天已跑通的兼容推理入口。
        sys.argv[0] = str(compat)
        runpy.run_path(str(compat), run_name="__main__")
    finally:
        hook.remove()

    if not state["success"]:
        raise RuntimeError("BEV 特征尚未成功保存：" + str(state["error"]))


if __name__ == "__main__":
    main()

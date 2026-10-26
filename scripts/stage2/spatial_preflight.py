"""记录真实 GPU、算子与本地实验资产；不创建训练优化器。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    """流式计算资产 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """实测 CUDA 自定义算子的前向和反向，检查文件完整性。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    record = {"command": [sys.executable, *sys.argv], "optimizer_steps": 0}
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    record["nvidia_smi"] = {"exit_code": smi.returncode,
                             "output": smi.stdout + smi.stderr}
    record["meminfo"] = Path("/proc/meminfo").read_text()
    try:
        import torch
        import mmcv
        from mmcv.ops.multi_scale_deform_attn import (
            MultiScaleDeformableAttnFunction,
            multi_scale_deformable_attn_pytorch,
        )

        record.update(torch=torch.__version__, mmcv=mmcv.__version__,
                      cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            torch.manual_seed(20261026)
            free, total = torch.cuda.mem_get_info()
            record["gpu"] = dict(name=torch.cuda.get_device_name(0),
                                 free_bytes=free, total_bytes=total)
            value = torch.randn(1, 16, 2, 4, device="cuda", requires_grad=True)
            shape = torch.tensor([[4, 4]], device="cuda", dtype=torch.long)
            start = torch.tensor([0], device="cuda", dtype=torch.long)
            locations = torch.rand(1, 5, 2, 1, 4, 2, device="cuda",
                                   requires_grad=True)
            weights = torch.full((1, 5, 2, 1, 4), .25, device="cuda",
                                 requires_grad=True)
            result = MultiScaleDeformableAttnFunction.apply(
                value, shape, start, locations, weights, 64)
            reference = multi_scale_deformable_attn_pytorch(
                value, shape, locations, weights)
            error = float((result - reference).abs().max())
            result.square().sum().backward()
            torch.cuda.synchronize()
            record["cuda_operator"] = {
                "forward": True, "backward": True, "max_abs_error": error,
                "value_gradient_l2": float(value.grad.norm()),
                "finite_gradients": all(bool(torch.isfinite(t.grad).all())
                                         for t in (value, locations, weights)),
                "scope": "MMCV operator only; no BEVFormer model or optimizer",
            }
    except Exception:
        record["gpu_error"] = traceback.format_exc()
    data = Path("/mnt/g/BEV_VLA_Datasets/bev_seg_nuscenes")
    files = [Path("/mnt/g/nus/bevformer_small_joint_author.pth"),
             ROOT / "models/distilbert-base-uncased/model.safetensors",
             ROOT / "outputs/stage2_oom_fix_20261022_continue/continue/fusion_step020.pt",
             ROOT / "outputs/stage2_oom_fix_20261022_step1/train/fusion_step001.pt"]
    files += sorted((data / "maps/expansion").glob("*.json"))
    files += sorted((ROOT / "outputs/stage2_oom_fix_20261022_validation/validate").glob("val_*.npz"))
    record["assets"] = {str(p): {"exists": p.is_file(),
        "size": p.stat().st_size if p.is_file() else None,
        "sha256": sha256(p) if p.is_file() else None} for p in files}
    datasets = {}
    for split in ("train", "val"):
        path = data / f"nuscenes_infos_temporal_{split}.pkl"
        with path.open("rb") as stream:
            ann = pickle.load(stream)
        infos = ann["infos"]
        missing = [cam["data_path"] for info in infos
                   for cam in info["cams"].values()
                   if not Path(cam["data_path"]).is_file()]
        datasets[split] = {"frames": len(infos), "missing_cameras": missing,
                           "scenes": sorted({r["scene_token"] for r in infos}),
                           "metadata": ann["metadata"], "sha256": sha256(path)}
    record["datasets"] = datasets
    record["scene_overlap"] = sorted(set(datasets["train"]["scenes"]) &
                                      set(datasets["val"]["scenes"]))
    record["metadata_tables"] = {
        version: {p.name: p.stat().st_size for p in (data / version).glob("*.json")}
        for version in ("v1.0-mini", "v1.0-trainval")}
    with args.output.open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: record.get(k) for k in
                      ("cuda_available", "gpu", "cuda_operator", "gpu_error",
                       "scene_overlap")}, indent=2))
    print("Report:", args.output)


if __name__ == "__main__":
    main()

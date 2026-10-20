"""Step54: 作者 Mini Val 原始/真实标定 Slicer 的时序 A/B 对照。

- 与 Step50 使用相同的 Config、checkpoint、作者 single_gpu_test。
- 每组各自重建数据集和模型，防止 prev_bev / can_bus 跨实验污染。
- 校正组只在运行时替换 Slicer.forward 的采样网格，不改磁盘源码和权重。
- 采用与 Step53 一致的二维 LiDAR(z=0) -> Ego 假设；不能据此断言训练约定。
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """解析运行参数。"""
    parser = argparse.ArgumentParser(description="Step54 author Mini temporal A/B")
    parser.add_argument(
        "--author-root",
        type=Path,
        default=Path.home() / "projects/bevformer_author_clean_5080",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/mnt/g/nus/bevformer_small_seg.pth"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs/drivelm_check/root_cause/step54_author_mini_calibration_ab",
    )
    parser.add_argument(
        "--mode",
        choices=("baseline", "calibrated", "both"),
        default="both",
        help="默认依次独立运行 baseline 和 calibrated，分别重建模型和数据集",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    """分块计算校验和，防止错用 checkpoint。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_calibrated_grids(dataset: Any, head_cfg: Any) -> dict[str, Any]:
    """从实际 dataset 顺序提取真实标定，并预先计算 CPU 采样网格。

    使用 Step53 同一平面几何约定：源图是 head 翻转后的 [W,H] 特征，
    特征行 = W-1-X_index，特征列 = Y_index，目标是 Ego 坐标像素中心。
    """
    import torch
    from pyquaternion import Quaternion

    infos = dataset.data_infos
    src_h = int(head_cfg.bev_h)
    src_w = int(head_cfg.bev_w)
    det_grid = head_cfg.det_grid_conf
    map_grid = head_cfg.map_grid_conf
    x0, x1, _ = det_grid["xbound"]
    y0, y1, _ = det_grid["ybound"]
    tx0, tx1, dx = map_grid["xbound"]
    ty0, ty1, dy = map_grid["ybound"]
    gt_w = round((tx1 - tx0) / dx)
    gt_h = round((ty1 - ty0) / dy)
    if src_h != src_w:
        raise ValueError("Step53 标定脚本要求当前 BEV 正方形；本脚本未验证非正方形")

    dtype = torch.float64
    gx = tx0 + (torch.arange(gt_w, dtype=dtype) + 0.5) * dx
    gy = ty0 + (torch.arange(gt_h, dtype=dtype) + 0.5) * dy
    yy, xx = torch.meshgrid(gy, gx, indexing="ij")
    target_xy = torch.stack((xx, yy), dim=0).reshape(2, -1)
    grids: dict[str, Any] = {}
    for info in infos:
        token = str(info["token"])
        if token in grids:
            raise ValueError(f"重复 sample token: {token}")
        rot = torch.as_tensor(
            Quaternion(info["lidar2ego_rotation"]).rotation_matrix,
            dtype=dtype,
        )[:2, :2]
        trans = torch.as_tensor(info["lidar2ego_translation"], dtype=dtype)[:2]
        if abs(torch.linalg.det(rot).item()) < 0.9:
            raise ValueError(f"帧 {token} 水平旋转矩阵异常")
        lidar_xy = torch.linalg.solve(rot, target_xy - trans[:, None]).reshape(
            2, gt_h, gt_w
        )
        lidar_x, lidar_y = lidar_xy.unbind(dim=0)
        x_index = (lidar_x - x0) / ((x1 - x0) / src_w) - 0.5
        y_index = (lidar_y - y0) / ((y1 - y0) / src_h) - 0.5
        # 与 Step53 完全一致，输入特征实际排列 [B,C,W,H]。
        row = src_w - 1 - x_index
        col = y_index
        grid = torch.stack(
            (2 * col / (src_h - 1) - 1, 2 * row / (src_w - 1) - 1),
            dim=-1,
        ).to(torch.float32)
        valid = (grid[..., 0].abs() <= 1) & (grid[..., 1].abs() <= 1)
        if valid.float().mean().item() < 0.99:
            raise ValueError(f"帧 {token}: 有效采样比例不足 99%")
        grids[token] = grid
    return grids


def attach_calibrated_slicer(model: Any, dataset: Any, head_cfg: Any) -> dict[str, Any]:
    """只在内存中按实际样本 token 切换校准网格。"""
    import torch.nn.functional as functional

    head = model.pts_bbox_head
    cropper = head.feat_cropper
    grids = make_calibrated_grids(dataset, head_cfg)
    if not hasattr(cropper, "map_grid"):
        raise ValueError("作者 cropper 没有 map_grid；禁止盲目修补")
    shape = tuple(cropper.map_grid.shape[:2])
    expected = tuple(next(iter(grids.values())).shape[:2])
    if shape != expected:
        raise ValueError(f"采样网格目标尺寸不一致：作者 {shape} vs 校正 {expected}")

    state: dict[str, Any] = {"token": None, "calls": 0, "tokens": []}
    old_head_forward = head.forward

    def head_forward_with_token(*args: Any, **kwargs: Any) -> Any:
        """从真实模型的元数据中取 token，拒绝按伪造的帧索引映射。"""
        metas = args[1] if len(args) >= 2 else kwargs.get("img_metas")
        if not isinstance(metas, (list, tuple)) or len(metas) != 1:
            raise ValueError("只支持单帧单样本推理，img_metas 结构异常")
        token = str(metas[0]["sample_idx"])
        if token not in grids:
            raise KeyError(f"当前模型帧 token 不在 dataset：{token}")
        expected_token = str(dataset.data_infos[state["calls"]]["token"])
        if token != expected_token:
            raise ValueError(
                "时序顺序与 dataset 不符："
                f"{state['calls']} / actual={token} / expected={expected_token}"
            )
        state["token"] = token
        state["calls"] += 1
        state["tokens"].append(token)
        return old_head_forward(*args, **kwargs)

    def calibrated_slicer_forward(features: Any) -> Any:
        """只替换 Slicer 坐标查询，不改变输入 BEV 和分割 Decoder。"""
        token = state["token"]
        if token is None:
            raise RuntimeError("尚未得到 sample token，不允许启用校准网格")
        if tuple(features.shape[-2:]) != (150, 150):
            raise ValueError(f"运行时 BEV 特征尺寸异常: {tuple(features.shape)}")
        grid = grids[token].to(device=features.device, dtype=features.dtype)
        grid = grid.unsqueeze(0).expand(features.shape[0], -1, -1, -1)
        return functional.grid_sample(
            features,
            grid=grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    head.forward = head_forward_with_token
    cropper.forward = calibrated_slicer_forward
    return state


def read_author_score(path: Path, before_count: int) -> dict[str, Any]:
    """从作者 single_gpu_test 自己写出的结果读取 mIoU。"""
    if not path.is_file():
        raise FileNotFoundError(f"作者没有生成指标文件：{path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != before_count + 1:
        raise RuntimeError(
            f"作者指标文件变化行数异常：原始 {before_count}，运行后 {len(lines)}"
        )
    decoded = json.loads(lines[-1])
    score = ast.literal_eval(decoded) if isinstance(decoded, str) else decoded
    if not isinstance(score, dict) or "mIoU" not in score:
        raise ValueError("作者分割指标格式异常")
    return score


def run_arm(
    name: str,
    cfg: Any,
    checkpoint: Path,
    source: Path,
) -> dict[str, Any]:
    """每个实验分支重新创建 Dataset、DataLoader、Model、时序状态。"""
    import torch
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from projects.mmdet3d_plugin.apis.test import single_gpu_test
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    config = copy.deepcopy(cfg)
    config.model.pretrained = None
    config.model.train_cfg = None
    config.data.test.test_mode = True
    config.data.workers_per_gpu = 0
    samples_per_gpu = int(config.data.test.get("samples_per_gpu", 1))
    if samples_per_gpu != 1 or not bool(config.model.video_test_mode):
        raise ValueError("必须单样本且 video_test_mode=True")
    dataset = build_dataset(config.data.test)
    if len(dataset) != 81:
        raise ValueError(f"不是 Mini Val 81 帧：{len(dataset)}")
    tokens = [str(info["token"]) for info in dataset.data_infos]
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=config.data.nonshuffler_sampler,
    )
    model = build_model(config.model, test_cfg=config.get("test_cfg"))
    checkpoint_data = load_checkpoint(model, str(checkpoint), map_location="cpu")
    model.CLASSES = checkpoint_data.get("meta", {}).get("CLASSES", dataset.CLASSES)
    if "PALETTE" in checkpoint_data.get("meta", {}):
        model.PALETTE = checkpoint_data["meta"]["PALETTE"]

    state = None
    if name == "calibrated":
        state = attach_calibrated_slicer(model, dataset, config.model.pts_bbox_head)

    wrapped = MMDataParallel(model.cuda(), device_ids=[0])
    metric_file = source / "segmentation_result.json"
    before = (
        len(metric_file.read_text(encoding="utf-8").splitlines())
        if metric_file.is_file()
        else 0
    )
    print(f"\n===== Step54 {name}: 作者原版指标 + 81 帧时序 ON =====", flush=True)
    start = time.perf_counter()
    single_gpu_test(wrapped, loader, show=False, out_dir=None)
    score = read_author_score(metric_file, before)
    if state is not None:
        if state["calls"] != len(dataset) or state["tokens"] != tokens:
            raise RuntimeError("校准组没有逐帧按正确 token 更新标定")
    elapsed = time.perf_counter() - start
    print(f"[Step54] {name}: mIoU={score['mIoU']:.4%}; {elapsed:.1f}s", flush=True)
    del wrapped, model, loader, dataset
    torch.cuda.empty_cache()
    return {
        "variant": name,
        "metrics_author_rounded": score,
        "elapsed_seconds": round(elapsed, 2),
        "num_frames": 81,
        "tokens_in_order": tokens,
        "calibrated_slicer_calls": state["calls"] if state is not None else 0,
    }


def main() -> None:
    """执行完整的独立时序 A/B，保存机器可读与中文摘要。"""
    args = parse_args()
    root = args.author_root.expanduser().resolve()
    source = root / "BEVFormer-seg"
    config_path = source / "projects/configs/bevformer/bevformer_small_seg.py"
    checkpoint = args.checkpoint.expanduser().resolve()
    for path in (source, config_path, checkpoint):
        if not path.exists():
            raise FileNotFoundError(f"输入不存在：{path}")

    os.chdir(source)
    sys.path.insert(0, str(source))
    import torch
    from mmcv import Config

    importlib.import_module("projects.mmdet3d_plugin")
    cfg = Config.fromfile(str(config_path))
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境 CUDA 不可用")
    print(f"[Check] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[Check] ckpt SHA256: {sha256(checkpoint)}", flush=True)
    print(f"[Check] config: {config_path}", flush=True)
    print(f"[Check] mode: {args.mode}", flush=True)

    arm_names = ("baseline", "calibrated") if args.mode == "both" else (args.mode,)
    results = []
    for name in arm_names:
        item = run_arm(name, cfg, checkpoint, source)
        results.append(item)
        if name == "baseline" and args.mode == "both":
            baseline_miou = float(item["metrics_author_rounded"]["mIoU"])
            if not (0.055 <= baseline_miou <= 0.075):
                raise RuntimeError(
                    "本次作者 baseline 与 Step50 的 6.38% 明显不同，"
                    f"实测为 {baseline_miou:.2%}。先检查输入和环境，不启动校准组。"
                )
    if len(results) == 2 and results[0]["tokens_in_order"] != results[1]["tokens_in_order"]:
        raise RuntimeError("两个分支的 token 顺序不同，A/B 无效")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "purpose": "Step54 frozen-author-checkpoint Mini Val temporal geometry A/B",
        "scope": "81-frame full mini_val, temporal ON, author single_gpu_test",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config_path),
        "source": str(source),
        "variants": results,
        "notes": [
            "calibrated 使用 Step53 z_lidar=0 平面假设，仅推理时替换 feat_cropper.forward",
            "每个分支新建模型和数据集，避免 prev_bev / can_bus 时序污染",
            "author single_gpu_test 将舍入后的指标追加到源码目录 segmentation_result.json",
            "Step37 的 sparse/no-temporal 几何分支曾表现不稳定，本实验不能默认改善",
            "不能凭合成坐标零误差断言训练空间正确或正式修复完成",
        ],
    }
    if len(results) == 2:
        b = results[0]["metrics_author_rounded"]
        c = results[1]["metrics_author_rounded"]
        data["delta_miou_pp"] = round(100 * (float(c["mIoU"]) - float(b["mIoU"])), 4)
        data["delta_class_iou_pp"] = {
            key: round(100 * (float(c[key]) - float(b[key])), 4)
            for key in ("Divider", "Pred_Crossing", "Boundary")
        }
    (output / "results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Step54：作者 Mini Val 81 帧时序空间标定 A/B",
        "",
        "所有分支使用冻结作者权重、独立时序状态、相同 GT 与作者评估函数。",
        "校正分支仅使用 Step53 的平面标定假设进行推理时采样，并非训练修复。",
        "",
        "| 分支 | Divider | Pred Crossing | Boundary | 前景 mIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in results:
        score = item["metrics_author_rounded"]
        lines.append(
            f"| {item['variant']} | {float(score['Divider']):.2%} | "
            f"{float(score['Pred_Crossing']):.2%} | "
            f"{float(score['Boundary']):.2%} | {float(score['mIoU']):.2%} |"
        )
    if "delta_miou_pp" in data:
        lines += ["", f"mIoU 改变量：{data['delta_miou_pp']:+.2f} 个百分点。"]
    lines += [
        "", "## 结论边界", "",
        "- 同协议 81 帧 A/B 能检验该校正是否提高该 Mini Val 的指标。",
        "- 提升或下降均不能单独证实训练时实际空间坐标约定。",
        "- 不能拿此前独立帧 Step37 数值直接比较时序 Mini 指标。",
        "- 若三类指标下降或不稳，禁止合入此空间校正。",
        "",
    ]
    (output / "audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Step54] 结果: {output / 'results.json'}", flush=True)
    print(f"[Step54] 报告: {output / 'audit.md'}", flush=True)
    if "delta_miou_pp" in data:
        print(f"[Step54] 校正相对原始 ΔmIoU = {data['delta_miou_pp']:+.2f} pp", flush=True)


if __name__ == "__main__":
    main()

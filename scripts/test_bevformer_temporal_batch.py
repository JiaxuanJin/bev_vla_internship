#!/usr/bin/env python3
"""BEVFormer-Small：真实连续关键帧对的 Temporal ON 批量冒烟测试。

基于已通过的 test_bevformer_temporal_pair.py 推理方式扩展：
- 从原始 officialscene PKL 筛选真实 prev 关系、同场景、间隔 <= 0.6 秒的帧对；
- 从已有 samples 或 DriveLM ZIP 定向准备每帧六路相机图像；
- 模型与匹配检测权重仅加载一次，对每组帧对显式清空历史 BEV；
- 严格加载权重，检查第一帧不用历史、第二帧真正使用历史 BEV；
- 每对立即保存进度，附每帧耗时/候选框数量和失败信息。

本程序只做 Temporal 状态传递稳定性诊断，不计算官方 3D mAP / BEV mIoU。
默认工作目录：~/projects/bev_vla；不覆盖原有双帧脚本或 799 帧实验结果。

命令示例：
    python3 scripts/test_bevformer_temporal_batch.py --prepare-only
    python3 scripts/test_bevformer_temporal_batch.py --limit 2
    python3 scripts/test_bevformer_temporal_batch.py
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import importlib
import json
import os
import pickle
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer"
CONFIG = REPO / "projects/configs/bevformer/bevformer_small.py"
# 已通过双帧测试的检测权重，不能换成 seg 权重配检测配置。
DEFAULT_CHECKPOINT = REPO / "ckpts/bevformer_small_epoch_24.pth"
SOURCE_PKL = ROOT / "data/drivelm/nuscenes_infos_temporal_drivelm799_officialscene.pkl"
SAMPLES = Path("/mnt/g/BEV_VLA_Datasets/nuscenes/samples")
ZIPS = [Path(f"/mnt/g/nus/drivelm_nus_imgs_{part}.zip") for part in ("val", "train")]
OUTPUT = ROOT / "outputs/temporal_batch_smoke"


def camera_key(value):
    """提取 CAM_*/文件名，禁止将 ZIP 中任意路径写出目标目录。"""
    parts = Path(str(value).replace("\\", "/")).parts
    if len(parts) < 2:
        raise ValueError(f"无法解析相机路径: {value}")
    camera, name = parts[-2:]
    if not (camera.startswith("CAM_") and name.lower().endswith((".jpg", ".jpeg"))):
        raise ValueError(f"非法相机路径: {value}")
    if camera in (".", "..") or name in (".", ".."):
        raise ValueError(f"非法文件名: {value}")
    return f"{camera}/{name}"


def archive_index():
    """只索引 ZIP 目录，按需提取 JPG，不整体解压大型压缩包。"""
    members = {}
    for archive_path in ZIPS:
        if not archive_path.is_file():
            print(f"[提示] 跳过不存在的 ZIP: {archive_path}", flush=True)
            continue
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                parts = Path(member).parts
                if len(parts) >= 2 and parts[-2].startswith("CAM_") and member.lower().endswith(".jpg"):
                    key = f"{parts[-2]}/{parts[-1]}"
                    members.setdefault(key, (archive_path, member))
    return members


def find_pairs(infos, members):
    """优先挑选不重叠的真实相邻双帧序列，顺序固定便于复现。"""
    by_token = {str(info["token"]): info for info in infos}
    candidates = []
    for second in infos:
        first = by_token.get(str(second.get("prev", "")))
        if first is None or first.get("scene_token") != second.get("scene_token"):
            continue
        gap = (int(second["timestamp"]) - int(first["timestamp"])) / 1_000_000
        if not 0 < gap <= 0.6:
            continue
        try:
            keys = []
            for frame in (first, second):
                cams = frame.get("cams", {})
                if len(cams) != 6:
                    break
                keys.extend(camera_key(cam["data_path"]) for cam in cams.values())
        except (ValueError, KeyError, TypeError):
            continue
        if len(keys) != 12 or not all((SAMPLES / key).is_file() or key in members for key in keys):
            continue
        candidates.append((first, second, gap, keys))

    candidates.sort(key=lambda item: (int(item[0]["timestamp"]), str(item[0]["token"])))
    chosen, used_tokens = [], set()
    for first, second, gap, keys in candidates:
        tokens = {str(first["token"]), str(second["token"])}
        if tokens & used_tokens:
            continue  # 不重复使用某个关键帧，避免两组序列互相污染。
        used_tokens.update(tokens)
        chosen.append((first, second, gap, keys))
    return chosen, len(candidates)


def save_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def save_pickle_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, path)


def prepare(limit: int):
    """生成独立 pair PKL，并核查 JPG，原始 PKL 与已有图片不修改。"""
    if not SOURCE_PKL.is_file():
        raise FileNotFoundError(f"缺少官方场景测试索引: {SOURCE_PKL}")
    with SOURCE_PKL.open("rb") as stream:
        full = pickle.load(stream)  # 仅加载本地可信实验 PKL
    if not isinstance(full, dict) or not isinstance(full.get("infos"), list):
        raise ValueError("输入 PKL 结构不正确：需要包含 infos 列表")

    members = archive_index()
    pairs, total_candidates = find_pairs(full["infos"], members)
    if not pairs:
        raise RuntimeError("没有找到相机图像齐全的真实相邻帧对")
    if limit:
        pairs = pairs[:limit]
    print(f"[准备] 有效候选={total_candidates} 对；互不重叠={len(pairs)} 对（本次执行数量）", flush=True)

    pair_dir = OUTPUT / "pair_indices"
    pair_dir.mkdir(parents=True, exist_ok=True)
    all_keys = list(dict.fromkeys(key for _, _, _, keys in pairs for key in keys))
    copied = 0
    # 缺图时从 ZIP 指定成员提取，避免解压完整 3GB ZIP。
    for index, key in enumerate(all_keys, 1):
        target = SAMPLES / key
        if target.is_file():
            continue
        if key not in members:
            raise FileNotFoundError(f"缺失相机图像且 ZIP 中不存在: {key}")
        archive_path, member = members[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".batch_tmp")
        try:
            with zipfile.ZipFile(archive_path) as archive, archive.open(member) as source, temp.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            if not target.exists():
                os.replace(temp, target)
                copied += 1
        finally:
            temp.unlink(missing_ok=True)
        if index % 60 == 0:
            print(f"[准备] 已检查图像路径 {index}/{len(all_keys)}", flush=True)

    from PIL import Image
    for key in all_keys:
        with Image.open(SAMPLES / key) as image:
            image.verify()
    print(f"[准备] JPG 验证={len(all_keys)} 张，新提取={copied} 张", flush=True)

    prepared = []
    for number, (first, second, gap, keys) in enumerate(pairs, 1):
        path = pair_dir / f"pair_{number:03d}.pkl"
        fragment = dict(full)
        fragment["infos"] = [copy.deepcopy(first), copy.deepcopy(second)]
        save_pickle_atomic(path, fragment)
        prepared.append({
            "pair_index": number, "pkl_path": str(path.resolve()),
            "gap_seconds": round(gap, 6),
            "scene_token": str(first["scene_token"]),
            "tokens": [str(first["token"]), str(second["token"])],
        })

    manifest = {
        "source_pkl": str(SOURCE_PKL), "candidate_count": total_candidates,
        "prepared_pairs": len(prepared), "unique_images": len(all_keys),
        "newly_extracted_images": copied, "pairs": prepared,
        "note": "独立双帧对；未承诺此子集存在 3 帧以上的连续链条。",
    }
    save_json_atomic(OUTPUT / "prepare_manifest.json", manifest)
    return prepared


def unwrap(obj):
    """兼容已成功双帧脚本的 MMCV DataContainer 结构。"""
    from mmcv.parallel import DataContainer
    for _ in range(8):
        if isinstance(obj, DataContainer):
            obj = obj.data
        elif isinstance(obj, (list, tuple)) and len(obj) == 1:
            obj = obj[0]
        else:
            break
    return obj


def init_model(checkpoint):
    """严格核对并仅加载一次检测权重，不降级为 strict=False。"""
    import torch
    from mmcv import Config
    from mmdet3d.models import build_model

    cfg = Config.fromfile(str(CONFIG))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = True
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    package = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    weights = package.get("state_dict", package)
    if not isinstance(weights, dict) or not weights:
        raise RuntimeError("checkpoint 未包含有效 state_dict")
    if all(key.startswith("module.") for key in weights):
        weights = {key.removeprefix("module."): value for key, value in weights.items()}

    expected = model.state_dict()
    missing = sorted(set(expected) - set(weights))
    extra = sorted(set(weights) - set(expected))
    mismatched = [key for key in set(weights) & set(expected)
                  if getattr(weights[key], "shape", None) != getattr(expected[key], "shape", None)]
    if missing or extra or mismatched:
        print(f"[权重不匹配] 缺失={len(missing)}，多余={len(extra)}，尺寸冲突={len(mismatched)}", flush=True)
        for label, keys in (("缺失", missing), ("多余", extra), ("尺寸", mismatched)):
            for key in keys[:3]:
                print(f"  {label}: {key}", flush=True)
        raise RuntimeError("检测模型与权重不一致，请检查路径；不允许 strict=False")

    model.load_state_dict(weights, strict=True)
    del weights, package
    model = model.cuda().eval()
    if not getattr(model, "video_test_mode", False):
        raise RuntimeError("未启用 video_test_mode")
    if not hasattr(model, "prev_frame_info") or not hasattr(model, "simple_test"):
        raise RuntimeError("模型未提供历史 BEV 状态接口")
    print(f"[模型] 严格加载成功（仅一次）：{checkpoint}", flush=True)
    return model, cfg


def reset_history(model):
    """每组双帧开始前重置全部已知时序状态。"""
    model.prev_frame_info["prev_bev"] = None
    model.prev_frame_info["scene_token"] = None
    for name in ("prev_pos", "prev_angle"):
        if name in model.prev_frame_info:
            model.prev_frame_info[name] = 0


def load_pair_inputs(cfg, prepared):
    """每次只构建两帧数据集，保证原版 dataset 的排序/元信息校验不变。"""
    import torch
    from mmdet3d.datasets import build_dataset

    val_cfg = copy.deepcopy(cfg.data.test)
    val_cfg["ann_file"] = prepared["pkl_path"]
    val_cfg["test_mode"] = True
    for option in ("samples_per_gpu", "workers_per_gpu"):
        val_cfg.pop(option, None)
    dataset = build_dataset(val_cfg)
    if len(dataset) != 2:
        raise RuntimeError(f"数据集应有 2 帧，实际 {len(dataset)}")

    inputs = []
    for i, token in enumerate(prepared["tokens"]):
        sample = dataset[i]
        img, meta = unwrap(sample["img"]), unwrap(sample["img_metas"])
        if not torch.is_tensor(img) or img.ndim != 4 or tuple(img.shape[:2]) != (6, 3):
            raise RuntimeError(f"第 {i+1} 帧图像不是六路 RGB 张量")
        if not isinstance(meta, dict) or str(meta.get("sample_idx")) != token:
            raise RuntimeError(f"第 {i+1} 帧 Token 或数据顺序不匹配")
        if str(meta.get("scene_token")) != prepared["scene_token"]:
            raise RuntimeError(f"第 {i+1} 帧与上一帧不在同一场景")
        if len(meta.get("lidar2img", [])) != 6 or len(meta.get("can_bus", [])) < 18:
            raise RuntimeError(f"第 {i+1} 帧标定或 ego-motion 信息不完整")
        inputs.append((img, meta))
    return inputs


def run_pair(model, cfg, prepared, torch):
    """同一对第一帧无历史、第二帧有历史；使用内部 simple_test 参数探针。"""
    reset_history(model)
    inputs = load_pair_inputs(cfg, prepared)
    calls = []
    original = model.simple_test

    def inspected_simple_test(*args, **kwargs):
        calls.append(kwargs.get("prev_bev") is not None)
        return original(*args, **kwargs)

    model.simple_test = inspected_simple_test
    frames = []
    try:
        for i, (img, meta) in enumerate(inputs, 1):
            gpu_img = img.unsqueeze(0).cuda()
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                output = model(return_loss=False, rescale=True,
                               img=[gpu_img], img_metas=[[meta]])
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if not isinstance(output, (list, tuple)) or len(output) != 1:
                raise RuntimeError(f"第 {i} 帧检测结果结构不正确")
            preds = output[0].get("pts_bbox") or {}
            scores = preds.get("scores_3d")
            if not calls or len(calls) != i:
                raise RuntimeError(f"第 {i} 帧历史 BEV 探针调用次数异常: {len(calls)}")
            frames.append({
                "frame": i, "token": prepared["tokens"][i - 1],
                "scene_token": str(meta["scene_token"]),
                "prev_bev_used": calls[-1],
                "prev_bev_saved": model.prev_frame_info.get("prev_bev") is not None,
                "detections": int(scores.numel()) if scores is not None else None,
                "seconds": round(elapsed, 3),
            })
            del output, gpu_img
    finally:
        model.simple_test = original
        reset_history(model)

    ok = (calls == [False, True] and len(frames) == 2
          and all(frame["prev_bev_saved"] for frame in frames)
          and all(frame["detections"] is not None for frame in frames))
    return ok, frames


def write_results(records, checkpoint, total, stopped=False):
    """每对写入临时结果文件并原子替换，意外中断也保留前面已完成的进度。"""
    succeeded = sum(item["pass"] for item in records)
    failed = len(records) - succeeded
    summary = {
        "temporal_batch_pass": bool(records) and len(records) == total and failed == 0,
        "pairs_requested": total, "pairs_processed": len(records),
        "pairs_passed": succeeded, "pairs_failed": failed,
        "pair_pass_rate": round(succeeded / len(records), 4) if records else None,
        "stopped_early": stopped,
        "video_test_mode": True,
        "checkpoint": str(checkpoint), "config": str(CONFIG),
        "source_pkl": str(SOURCE_PKL), "pairs": records,
        "note": "27 对独立双帧 Temporal ON 功能冒烟测试，不是官方 3D mAP 或分割 mIoU。检测框为模型候选框数量。",
    }
    save_json_atomic(OUTPUT / "temporal_batch_result.json", summary)

    csv_path = OUTPUT / "temporal_batch_frames.csv"
    temp = csv_path.with_suffix(".csv.tmp")
    fields = ["pair_index", "pass", "frame", "token", "scene_token", "prev_bev_used",
              "prev_bev_saved", "detections", "seconds", "gap_seconds"]
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in records:
            for frame in item.get("frames", []):
                writer.writerow({**{k: item.get(k) for k in ("pair_index", "pass", "gap_seconds")},
                                 **frame})
    os.replace(temp, csv_path)
    return summary


def run_batch(prepared, checkpoint):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请切换到此前跑通双帧测试的环境")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"找不到检测 checkpoint: {checkpoint}")
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    plugin = importlib.import_module("projects.mmdet3d_plugin")
    if not Path(plugin.__file__).resolve().is_relative_to(REPO.resolve()):
        raise RuntimeError(f"导入了错误的插件目录: {plugin.__file__}")
    model, cfg = init_model(checkpoint)

    records = []
    total = len(prepared)
    stopped = False
    print(f"[开始] 计划处理 {total} 对连续帧，共 {2 * total} 帧", flush=True)
    try:
        for item in prepared:
            number = item["pair_index"]
            try:
                ok, frames = run_pair(model, cfg, item, torch)
                row = {**item, "pass": ok, "frames": frames}
                details = " / ".join(
                    f"F{frame['frame']}:历史={frame['prev_bev_used']},耗时={frame['seconds']}s"
                    for frame in frames
                )
                print(f"[第 {number:02d}/{total:02d} 对] {'PASS' if ok else 'FAIL'} | {details}", flush=True)
            except Exception as exc:
                row = {**item, "pass": False, "frames": [],
                       "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc(limit=5)}
                print(f"[第 {number:02d}/{total:02d} 对] FAIL | {row['error'][:350]}", flush=True)
                # GPU 发生 OOM/严重设备错误时，避免继续产生不可信测试记录。
                if "out of memory" in str(exc).lower() or "device-side assert" in str(exc).lower():
                    stopped = True
            records.append(row)
            summary = write_results(records, checkpoint, total, stopped)
            if stopped:
                print("[停止] GPU 资源/设备错误，已保留已完成测试的 JSON 与 CSV", flush=True)
                break
            if number % 5 == 0:
                gc.collect()
    except KeyboardInterrupt:
        stopped = True
        write_results(records, checkpoint, total, stopped=True)
        print("\n[中断] 结果已保存，未完成部分不能视为通过。", flush=True)
        raise

    summary = write_results(records, checkpoint, total, stopped)
    print(f"[汇总] 处理={summary['pairs_processed']}/{total}，通过={summary['pairs_passed']}，"
          f"失败={summary['pairs_failed']}，全部通过={summary['temporal_batch_pass']}", flush=True)
    print(f"[文件] {OUTPUT / 'temporal_batch_result.json'}", flush=True)
    print(f"[文件] {OUTPUT / 'temporal_batch_frames.csv'}", flush=True)
    if not summary["temporal_batch_pass"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="BEVFormer-Small 真实相邻双帧批量 Temporal ON 稳定性检查")
    parser.add_argument("--prepare-only", action="store_true", help="仅准备 JPG 和 pair PKL，不启动 GPU")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="匹配检测模型的 checkpoint（默认已验证的 epoch_24）")
    parser.add_argument("--limit", type=int, default=0, help="仅测试前 N 对，0 表示全部有效帧对")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit 必须 >= 0")
    prepared = prepare(args.limit)
    if args.prepare_only:
        print(f"[完成] 已准备 {len(prepared)} 对；输出: {OUTPUT / 'prepare_manifest.json'}", flush=True)
        return
    run_batch(prepared, args.checkpoint.expanduser().resolve())


if __name__ == "__main__":
    main()

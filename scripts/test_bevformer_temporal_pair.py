#!/usr/bin/env python3
"""BEVFormer-Small 双关键帧 Temporal ON 冒烟测试。

运行位置：~/projects/bev_vla
准备：python3 scripts/test_bevformer_temporal_pair.py --prepare-only
推理：python3 scripts/test_bevformer_temporal_pair.py

只将最多 12 张必要 JPG 添加到现有 samples 目录，绝不覆盖已有图片。
其余生成物均写入 outputs/temporal_pair_smoke/，不会修改原有 799 帧实验。
"""

import argparse
import copy
import json
import os
import pickle
import shutil
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "BEVFormer"
CONFIG = REPO / "projects/configs/bevformer/bevformer_small.py"
WEIGHTS = Path("/mnt/g/nus/bevformer_small_seg.pth")
OFFICIAL_PKL = ROOT / "data/drivelm/nuscenes_infos_temporal_drivelm799_officialscene.pkl"
SAMPLES = Path("/mnt/g/BEV_VLA_Datasets/nuscenes/samples")
ZIPS = [Path(f"/mnt/g/nus/drivelm_nus_imgs_{part}.zip") for part in ("val", "train")]
OUTPUT = ROOT / "outputs/temporal_pair_smoke"


def camera_key(path):
    """从已有 PKL 的相机路径提取 CAM_*/文件名，避免目录穿越。"""
    parts = Path(str(path).replace("\\", "/")).parts
    if len(parts) < 2:
        raise ValueError(f"无法解析相机路径: {path}")
    camera, name = parts[-2:]
    if not camera.startswith("CAM_") or not name.lower().endswith((".jpg", ".jpeg")):
        raise ValueError(f"不是合法的相机图片: {path}")
    return f"{camera}/{name}"


def archive_index():
    """只读取 ZIP 目录，不解压数 GB 的数据包。"""
    members = {}
    for archive_path in ZIPS:
        if not archive_path.is_file():
            print(f"[提示] ZIP 不存在，跳过: {archive_path}", flush=True)
            continue
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.namelist():
                parts = Path(member).parts
                if len(parts) >= 2 and parts[-2].startswith("CAM_") and member.lower().endswith(".jpg"):
                    key = f"{parts[-2]}/{parts[-1]}"
                    members.setdefault(key, (archive_path, member))
    return members


def choose_pair(infos, members):
    """筛选真实 prev 链、同场景、约 0.5 秒间隔且六相机有来源的两帧。"""
    by_token = {str(item["token"]): item for item in infos}
    candidates = []
    for second in infos:
        first = by_token.get(str(second.get("prev", "")))
        if first is None or first.get("scene_token") != second.get("scene_token"):
            continue
        gap = (int(second["timestamp"]) - int(first["timestamp"])) / 1_000_000
        if not 0 < gap <= 0.6:
            continue
        keys = []
        for frame in (first, second):
            cams = frame.get("cams", {})
            if len(cams) != 6:
                break
            keys.extend(camera_key(cam["data_path"]) for cam in cams.values())
        if len(keys) != 12:
            continue
        if all((SAMPLES / key).is_file() or key in members for key in keys):
            candidates.append((first, second, gap, keys))
    if not candidates:
        raise RuntimeError("没有找到可以读取六路相机的真实连续帧对，请检查 officialscene PKL 与 ZIP")
    return candidates[0], len(candidates)


def prepare():
    if not OFFICIAL_PKL.is_file():
        raise FileNotFoundError(f"缺少真实场景 PKL: {OFFICIAL_PKL}")
    with OFFICIAL_PKL.open("rb") as handle:
        full = pickle.load(handle)  # 仅加载自己的本地可信实验文件
    if not isinstance(full, dict) or not isinstance(full.get("infos"), list):
        raise ValueError("officialscene PKL 必须是带 infos 列表的字典")

    members = archive_index()
    (first, second, gap, keys), count = choose_pair(full["infos"], members)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"[准备] 可用真实连续帧对: {count}", flush=True)
    print(f"[准备] 前一帧: {first['token']}", flush=True)
    print(f"[准备] 当前帧: {second['token']}", flush=True)
    print(f"[准备] 时间间隔: {gap:.3f} 秒", flush=True)

    # 用原 PKL 数据结构生成两帧测试专用索引，不触碰原文件。
    pair = dict(full)
    pair["infos"] = [copy.deepcopy(first), copy.deepcopy(second)]
    pair_path = OUTPUT / "temporal_pair.pkl"
    with pair_path.open("wb") as handle:
        pickle.dump(pair, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 缺失图片从 ZIP 定向提取。使用临时文件 + 原子重命名，避免留下半张图片。
    copied = 0
    for key in dict.fromkeys(keys):
        target = SAMPLES / key
        if target.is_file():
            continue
        archive_path, member = members[key]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".temporal_tmp")
        try:
            with zipfile.ZipFile(archive_path) as archive, archive.open(member) as source, temporary.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            if target.exists():
                temporary.unlink(missing_ok=True)
            else:
                os.replace(temporary, target)
                copied += 1
        finally:
            temporary.unlink(missing_ok=True)

    # 确保图片实际可解码，不只检查名字和尺寸。
    from PIL import Image
    for key in keys:
        with Image.open(SAMPLES / key) as im:
            im.verify()
    print(f"[准备] 图像可解码: {len(keys)}/12，新提取: {copied} 张", flush=True)
    return pair_path, (first, second), gap


def unwrap(obj):
    """兼容 MMCV DataContainer / 单元素包装，复用旧脚本使用的推理输入结构。"""
    from mmcv.parallel import DataContainer
    for _ in range(8):
        if isinstance(obj, DataContainer):
            obj = obj.data
        elif isinstance(obj, (list, tuple)) and len(obj) == 1:
            obj = obj[0]
        else:
            break
    return obj


def run(pair_path, pair, gap, checkpoint):
    import torch
    import mmcv
    from mmcv import Config

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA；请确认处于此前跑通 BEVFormer 的 WSL Python 环境")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"权重不存在: {checkpoint}")
    print(f"[模型] 本次使用权重: {checkpoint}", flush=True)

    # 使用实际已验证的 BEVFormer 工程；将相对路径固定到 BEVFormer 根目录。
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    import importlib
    plugin = importlib.import_module("projects.mmdet3d_plugin")
    if not Path(plugin.__file__).resolve().is_relative_to(REPO.resolve()):
        raise RuntimeError(f"导入的插件不在本项目内: {plugin.__file__}")
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg = Config.fromfile(str(CONFIG))
    val_cfg = copy.deepcopy(cfg.data.test)
    val_cfg["ann_file"] = str(pair_path)
    val_cfg["test_mode"] = True
    for option in ("samples_per_gpu", "workers_per_gpu"):
        val_cfg.pop(option, None)
    dataset = build_dataset(val_cfg)
    if len(dataset) != 2:
        raise RuntimeError(f"预期加载两帧，实际加载 {len(dataset)} 帧")

    # 先检查真实场景元信息，防止无意间仍然读到了伪造独立帧。
    inputs = []
    for index, expected in enumerate(pair):
        sample = dataset[index]
        img = unwrap(sample["img"])
        meta = unwrap(sample["img_metas"])
        if not torch.is_tensor(img) or img.ndim != 4 or tuple(img.shape[:2]) != (6, 3):
            raise RuntimeError(f"第 {index+1} 帧图像张量异常: {getattr(img, 'shape', None)}")
        if not isinstance(meta, dict):
            raise RuntimeError(f"第 {index+1} 帧 meta 不是字典: {type(meta)}")
        if str(meta.get("sample_idx")) != str(expected["token"]):
            raise RuntimeError(f"第 {index+1} 帧 sample_idx 与原始 Token 不一致")
        if str(meta.get("scene_token")) != str(expected["scene_token"]):
            raise RuntimeError(f"第 {index+1} 帧场景 Token 不一致，禁止时序推理")
        if len(meta.get("lidar2img", [])) != 6 or len(meta.get("can_bus", [])) < 18:
            raise RuntimeError(f"第 {index+1} 帧相机标定或 can_bus 不完整")
        inputs.append((img, meta))
    print("[检查] 两帧数据管道、真实 scene_token、六相机标定和 can_bus 均通过", flush=True)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = True
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    package = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    weights = package.get("state_dict", package)
    if all(k.startswith("module.") for k in weights):
        weights = {k.removeprefix("module."): v for k, v in weights.items()}
    # 在 strict=True 之前先汇总差异，避免数百个键名刷屏。禁止静默跳过不匹配参数。
    expected = model.state_dict()
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    bad_shapes = [key for key in set(expected) & set(weights)
                  if tuple(expected[key].shape) != tuple(weights[key].shape)]
    if missing or unexpected or bad_shapes:
        print(f"[权重不匹配] 模型缺失={len(missing)}，额外权重={len(unexpected)}，形状不一致={len(bad_shapes)}", flush=True)
        for label, keys in (("模型需要但权重没有", missing),
                            ("权重含有但模型没有", unexpected),
                            ("权重形状不一致", bad_shapes)):
            for key in keys[:4]:
                print(f"  {label}: {key}", flush=True)
        raise RuntimeError("权重与配置不匹配。请选用对应模型的 checkpoint；不能使用 strict=False 强行跳过。")
    result = model.load_state_dict(weights, strict=True)
    del weights, package
    model = model.cuda().eval()
    print(f"[模型] 权重严格加载成功，video_test_mode={model.video_test_mode}", flush=True)

    if not hasattr(model, "prev_frame_info") or not hasattr(model, "simple_test"):
        raise RuntimeError("当前模型未暴露 BEVFormer Temporal 状态，停止测试")
    model.prev_frame_info["prev_bev"] = None
    model.prev_frame_info["scene_token"] = None

    # 探针直接记录模型内部 simple_test 真正收到的 prev_bev 参数。
    flags = []
    old_simple_test = model.simple_test
    def instrumented_simple_test(*args, **kwargs):
        flags.append(kwargs.get("prev_bev") is not None)
        return old_simple_test(*args, **kwargs)
    model.simple_test = instrumented_simple_test

    records = []
    for i, (img, meta) in enumerate(inputs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(
                return_loss=False, rescale=True,
                img=[img.unsqueeze(0).cuda()], img_metas=[[meta]],
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise RuntimeError(f"第 {i+1} 帧预测输出结构异常: {type(outputs)}")
        pts = outputs[0].get("pts_bbox") or {}
        scores = pts.get("scores_3d")
        boxes_count = int(scores.numel()) if scores is not None else None
        history_saved = model.prev_frame_info.get("prev_bev") is not None
        records.append({
            "frame": i+1, "token": str(pair[i]["token"]),
            "scene_token": str(meta["scene_token"]),
            "prev_bev_used": flags[-1] if len(flags) > i else None,
            "prev_bev_saved": history_saved,
            "detections": boxes_count, "seconds": round(elapsed, 3),
        })
        print(f"[推理 {i+1}/2] 历史BEV输入={flags[-1]}，输出状态={history_saved}，"
              f"检测框={boxes_count}，耗时={elapsed:.3f}s", flush=True)

    passed = flags == [False, True] and all(r["prev_bev_saved"] for r in records)
    summary = {
        "temporal_smoke_pass": passed,
        "video_test_mode": bool(model.video_test_mode),
        "gap_seconds": round(gap, 3),
        "checkpoint": str(checkpoint), "config": str(CONFIG),
        "pkl": str(pair_path), "frames": records,
        "note": "仅验证两帧 Temporal ON 运行与历史 BEV 传递，不代表 799 帧正式 mAP/mIoU。",
    }
    result_path = OUTPUT / "temporal_pair_result.json"
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[结果] {'PASS' if passed else 'FAIL'}，保存至 {result_path}", flush=True)
    if not passed:
        raise RuntimeError("历史 BEV 传递没有通过校验，请勿宣称 Temporal ON 成功")


def main():
    parser = argparse.ArgumentParser(description="BEVFormer-Small Temporal 双帧测试")
    parser.add_argument("--prepare-only", action="store_true", help="只准备图片与测试索引，不运行GPU")
    parser.add_argument("--checkpoint", type=Path, default=WEIGHTS,
                        help="显式指定匹配 BEVFormer-Small 配置的权重；结果将记录该路径")
    args = parser.parse_args()
    pair_path, pair, gap = prepare()
    if args.prepare_only:
        print("[完成] 数据准备成功。下一步去掉 --prepare-only 运行 GPU 双帧推理。", flush=True)
    else:
        run(pair_path, pair, gap, args.checkpoint.expanduser().resolve())


if __name__ == "__main__":
    main()

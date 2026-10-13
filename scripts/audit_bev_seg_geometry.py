#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BEVFormer-Small 分割空间错位的无破坏性数据检查。

目标：核对 Mini 验证帧的场景顺序、真实车身位姿、进入推理管线的 can_bus，
以及单帧预测与真值在数据管线中是否一致。

使用：cd ~/projects/bev_vla && python scripts/audit_bev_seg_geometry.py
只创建 outputs/bev_seg/seg_geometry_audit.json，不修改模型/原始数据/现有 mIoU。
"""

import copy
import importlib
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEG_REPO = ROOT / "BEVFormer-seg"
CONFIG = SEG_REPO / "projects/configs/bevformer/bevformer_small_seg.py"
OUTPUT = ROOT / "outputs/bev_seg/seg_geometry_audit.json"
INDICES = (0, 1, 2, 40, 80)


def num_list(x, digits=5):
    """浮点数转为可记录的简短数组（避免把全部 18 个状态打印在终端）。"""
    if x is None:
        return None
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    return [round(float(v), digits) for v in a]


def unwrap(value):
    """兼容 MMCV 的 DataContainer 以及单尺度测试增强产生的包装。"""
    for _ in range(12):
        if value.__class__.__name__ == "DataContainer" and hasattr(value, "data"):
            value = value.data
        elif isinstance(value, (tuple, list)) and len(value) == 1:
            value = value[0]
        else:
            break
    return value


def main():
    if not CONFIG.is_file():
        raise FileNotFoundError(f"模型配置不存在：{CONFIG}")

    # 该仓库使用 data/nuscenes 相对路径；先进入仓库再加载。
    os.chdir(SEG_REPO)
    sys.path.insert(0, str(SEG_REPO))
    from mmcv import Config
    from mmdet3d.datasets import build_dataset

    cfg = Config.fromfile(str(CONFIG))
    importlib.import_module("projects.mmdet3d_plugin")
    dc = copy.deepcopy(cfg.data.test)
    dc.test_mode = True
    dc.pop("samples_per_gpu", None)
    dc.pop("workers_per_gpu", None)
    ds = build_dataset(dc)
    n = len(ds)
    print(f"[验证集] 样本总数：{n}", flush=True)
    print(f"[时序] 配置 video_test_mode = {cfg.model.get('video_test_mode')}", flush=True)

    # 将数据集里的绝对位姿和相机位姿与其提供给模型的 can_bus 比较。
    counts = Counter(str(x.get("scene_token", "")) for x in ds.data_infos)
    print(f"[场景数] {len(counts)}；各场景样本数量 = {list(counts.values())}", flush=True)
    records = []
    for idx in INDICES:
        if idx >= n:
            continue
        info = ds.data_infos[idx]
        original_can_bus = num_list(info.get("can_bus"))
        raw_pose = num_list(info.get("ego2global_translation"))
        raw_rot = num_list(info.get("ego2global_rotation"))
        # 直接读取当前样本的标准 get_data_info（此处不会调用模型/GPU）。
        data = ds.get_data_info(idx)
        converted_can = num_list(data.get("can_bus"))
        pos = converted_can[:3] if converted_can and len(converted_can) >= 3 else None
        yaw = converted_can[-1] if converted_can else None
        same_pose = (pos is not None and raw_pose is not None
                     and np.allclose(pos, raw_pose, atol=1e-3))
        item = {
            "index": idx,
            "sample_token": str(info.get("token", "")),
            "scene_token": str(info.get("scene_token", "")),
            "frame_idx": info.get("frame_idx"),
            "timestamp": int(info.get("timestamp", 0)),
            "ego2global_translation": raw_pose,
            "ego2global_rotation": raw_rot,
            "can_bus_before_get_data_info": original_can_bus,
            "can_bus_after_get_data_info_position": pos,
            "can_bus_after_get_data_info_yaw_degrees": yaw,
            "can_bus_position_matches_ego_pose": bool(same_pose),
        }
        records.append(item)
        print(f"[帧 {idx:02d}] 场景={item['scene_token'][:10]}  ", end="")
        print(f"原始位置={raw_pose} -> can_bus位置={pos}；yaw={yaw}；位置一致={same_pose}", flush=True)

    # 两个相邻帧的实际 pipeline meta：验证图像与车辆状态是否传入。
    pipeline_meta = []
    for idx in (0, 1):
        if idx >= n:
            continue
        sample = ds[idx]
        meta = unwrap(sample.get("img_metas"))
        if not isinstance(meta, dict):
            raise TypeError(f"第 {idx} 帧 img_metas 类型不符合预期：{type(meta)}")
        bus = num_list(meta.get("can_bus"))
        entry = {
            "index": idx,
            "scene_token": str(meta.get("scene_token", "")),
            "can_bus_position": bus[:3] if bus and len(bus) >= 3 else None,
            "can_bus_yaw_degrees": bus[-1] if bus else None,
            "img_metas_keys": sorted(meta.keys()),
        }
        pipeline_meta.append(entry)
        print(f"[模型输入帧 {idx}] can_bus位置={entry['can_bus_position']}；yaw={entry['can_bus_yaw_degrees']}", flush=True)

    # 源码核对：不修改调用；模型测试入口是否支持时序状态保存。
    source = (SEG_REPO / "projects/mmdet3d_plugin/bevformer/detectors/bevformer.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run_bev_seg_minival.py").read_text(encoding="utf-8")
    flags = {
        "model_forward_test_has_prev_bev_state": "self.prev_frame_info['prev_bev'] = new_prev_bev" in source,
        "model_forward_test_handles_scene_switch": "img_metas[0][0]['scene_token']" in source,
        "evaluation_calls_model_test_entry": "model(return_loss=False" in runner,
        "evaluation_uses_dataset_order": "for index in range(total)" in runner,
    }
    print("[代码入口检查]", flags, flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "说明": "只检查数据/代码路径，不改动预测与 mIoU；翻转/平移优化不得直接当正式成绩。",
        "dataset_length": n,
        "scene_counts": dict(counts),
        "video_test_mode": bool(cfg.model.get("video_test_mode")),
        "records": records,
        "actual_pipeline_metas": pipeline_meta,
        "static_source_flags": flags,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 诊断结果已保存：{OUTPUT}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[失败] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        sys.exit(1)

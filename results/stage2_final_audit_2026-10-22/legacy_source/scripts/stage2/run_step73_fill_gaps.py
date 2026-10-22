#!/usr/bin/env python3
"""Step73：把第二阶段两项“基础完成”补齐为可复现功能和证据。

实现可审计的英文道路指令解析 + 动态 DistilBERT 编码，独立地图锚点核验。
这是功能与诊断，不声称开放词表语义理解、精度提升或坐标物理闭环。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2.semantic_instruction import (
    ego_direction_mask,
    calibrated_bev_direction_mask,
    parse_instruction,
    self_test as parser_test,
)

OUT = ROOT / "outputs/stage2_step73"
DOC = ROOT / "docs/stage2/15_语言与空间能力补充验收.md"

# 当前 DistilBERT 为英文 uncased 预训练模型。
COMMANDS = (
    "Focus on lane markings ahead",
    "Find the lane dividers in front of the car",
    "Locate pedestrian crossings ahead",
    "Show the zebra crossing on the front right",
    "Highlight the road boundaries on the left",
    "Find the curb on the rear right",
    "Find road edges and lane markings ahead",
    "Highlight road boundaries but ignore pedestrian crossings",
    "Do not ignore the lane lines",   # 明确不支持的复杂否定
    "Find the traffic lights ahead",  # 超出地图三类，明确不支持
)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def self_test() -> None:
    """可本地执行的结构化语言解析、坐标符号与栅格化自检。"""
    checked = parser_test()
    a = np.zeros((200, 400), dtype=np.uint8)
    x = np.array([-1.0, 1.0])
    y = np.array([1.0, -1.0])
    ins = parse_instruction("Locate the road edges on the front right")
    assert ego_direction_mask(x, y, ins).tolist() == [False, True]
    p = pixels_from_xy(np.array([0.0]), np.array([0.0]), (200, 400), 60, 30, "direct")
    assert p[0].tolist() == [100] and p[1].tolist() == [200]
    m = calibrated_bev_direction_mask(
        (6, 6), (-3, -3, -1, 3, 3, 1), np.eye(3), np.zeros(3), ins
    )
    assert int(m.sum()) == 9 and m.shape == (6, 6)
    assert np.isfinite(a).all() and len(checked) >= 5
    print("[Step73] 自检 PASS：多目标、否定、方向、歧义拒绝、米制坐标栅格")


def dynamic_language_and_bev(model_dir: Path, feature_file: Path) -> dict[str, Any]:
    """动态文本而非固定三条缓存，检查编码和 2D 映射的功能正确性。"""
    import torch

    from scripts.stage2.language_encoder import DistilBERTLanguageEncoder
    from scripts.stage2.bev_fusion import LanguageBEVFusion
    from scripts.stage2.run_step56_smoke import load_bev_feature

    valid = []
    rejected = []
    for prompt in COMMANDS:
        try:
            valid.append(parse_instruction(prompt))
        except ValueError as exc:
            rejected.append({"text": prompt, "reason": str(exc)})
    if len(valid) != 8 or len(rejected) != 2:
        raise AssertionError("固定受限语言测试集的成功/拒绝数量异常")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"真实 DistilBERT 本地目录不存在：{model_dir}")
    if not feature_file.is_file():
        raise FileNotFoundError(f"已有 BEV 特征不存在：{feature_file}")

    encoder = DistilBERTLanguageEncoder(
        model_name=str(model_dir), freeze=True, local_files_only=True
    ).to(device).eval()
    texts = [v.original for v in valid]
    with torch.no_grad():
        embeddings = encoder(texts, str(device))
    if tuple(embeddings.shape) != (8, 768) or not torch.isfinite(embeddings).all():
        raise AssertionError(f"编码输出形状或数值错误：{list(embeddings.shape)}")
    # 只用已保存真实 BEV；不允许找不到时偷偷退回合成随机张量。
    bev, source = load_bev_feature(feature_file, "real", device)
    if source != "real_npy":
        raise AssertionError("预期已保存的真实 BEV 特征")
    fusion = LanguageBEVFusion(text_dim=768, alpha_init=0.01).to(device).eval()
    with torch.no_grad():
        # 两条不同指令在相同 BEV 上运行，确保语言变量得到隔离。
        bev_pair = bev.expand(2, -1, -1, -1).contiguous()
        bypass, _ = fusion(bev_pair, embeddings[:2], enabled=False)
        enhanced, attention = fusion(bev_pair, embeddings[:2], enabled=True)
        if tuple(attention.shape) != (2, 1, 150, 150):
            raise AssertionError(f"BEV attention shape 错误：{list(attention.shape)}")
        if not torch.equal(bypass, bev_pair) or not torch.isfinite(enhanced).all():
            raise AssertionError("BEV 融合关闭回退/有限数值检查未通过")
        attention_delta = float((attention[0] - attention[1]).abs().mean())
        if attention_delta <= 1e-8:
            raise AssertionError("切换语言后 BEV 注意力没有数值变化")
        vec_norm = torch.nn.functional.normalize(embeddings, dim=-1)
        similarities = (vec_norm @ vec_norm.T).cpu().numpy()
        embeddings_delta = float((embeddings[0] - embeddings[1]).abs().mean())
        # 此处 projection/fusion 为随机初始化，只能 PASS 功能而不是语言定位。
    result = {
        "status": "PASS_FUNCTION_ONLY",
        "device": str(device),
        "encoder_path": str(model_dir),
        "feature_file": str(feature_file),
        "feature_origin": source,
        "parsed_commands": [entry.to_dict() for entry in valid],
        "explicitly_rejected": rejected,
        "supported_scope": "English, divider/ped_crossing/boundary, explicit directions and limited negation",
        "text_shape": list(embeddings.shape),
        "attention_shape": list(attention.shape),
        "different_text_embedding_mae": embeddings_delta,
        "different_text_attention_mae": attention_delta,
        "pairwise_embedding_cosine_similarity": similarities.round(6).tolist(),
        "bypass_exact_identity": True,
        "model_weights_trained_in_this_test": False,
        "spatial_attention_weights_trained": False,
        "no_semantic_accuracy_claim": True,
        "note": "动态真实 DistilBERT+二维 BEV 注意力功能通过；注意力头随机初始化，不可据此证明正确定位。",
    }
    del fusion, encoder, bev, embeddings
    return result


def pixels_from_xy(
    x: np.ndarray, y: np.ndarray, shape: tuple[int, int],
    patch_width: float, patch_height: float, method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """用明确米制中心和边界转换世界地图裁剪平面坐标到像素坐标。"""
    u, v = np.asarray(x), np.asarray(y)
    if method.startswith("swap"):
        u, v = v, u
        # 交换轴必须连真实物理尺度一起交换；仅用于发现坐标协议问题。
        x_range, y_range = patch_height, patch_width
    else:
        x_range, y_range = patch_width, patch_height
    if "flip_x" in method:
        u = -u
    if "flip_y" in method:
        v = -v
    row = np.rint((v / y_range + 0.5) * shape[0]).astype(np.int64)
    col = np.rint((u / x_range + 0.5) * shape[1]).astype(np.int64)
    return row, col


def geometry_to_points(geometry: Any, sample_step: float = 0.15) -> np.ndarray:
    """展开局部 shapely 线几何并沿折线等距采点，供与真实 GT 对照。"""
    parts = []
    geom_type = getattr(geometry, "geom_type", "")
    if getattr(geometry, "is_empty", True):
        return np.empty((0, 2), dtype=np.float64)
    if geom_type in {"LineString", "LinearRing"}:
        points = np.asarray(geometry.coords, dtype=np.float64)[:, :2]
        if len(points) < 2:
            return points
        diff = points[1:] - points[:-1]
        for origin, displacement in zip(points[:-1], diff):
            n = max(2, int(np.ceil(np.linalg.norm(displacement) / sample_step)) + 1)
            t = np.linspace(0, 1, n, dtype=np.float64)
            parts.append(origin[None, :] + t[:, None] * displacement[None, :])
    elif geom_type == "Polygon":
        parts.append(geometry_to_points(geometry.exterior, sample_step))
    elif hasattr(geometry, "geoms"):
        parts.extend(geometry_to_points(v, sample_step) for v in geometry.geoms)
    return np.concatenate(parts, axis=0) if parts else np.empty((0, 2), dtype=np.float64)


def raster_and_match(
    points: np.ndarray, gt: np.ndarray, patch_w: float, patch_h: float,
) -> list[dict[str, Any]]:
    """无学习参数的地图锚点对齐；比较不同方向以识别旋转/镜像歧义。"""
    from scipy.ndimage import binary_dilation

    height, width = gt.shape
    outputs = []
    methods = (
        "direct", "flip_x", "flip_y", "flip_x_flip_y",
        "swap", "swap_flip_x", "swap_flip_y", "swap_flip_x_flip_y",
    )
    gt_dilated = binary_dilation(gt.astype(bool), iterations=2)
    for name in methods:
        rows, cols = pixels_from_xy(points[:, 0], points[:, 1], gt.shape, patch_w, patch_h, name)
        keep = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        pred = np.zeros_like(gt, dtype=bool)
        pred[rows[keep], cols[keep]] = True
        pred_dilated = binary_dilation(pred, iterations=2)
        hits_pred = int((pred & gt_dilated).sum())
        hits_gt = int((gt.astype(bool) & pred_dilated).sum())
        precision = hits_pred / max(1, int(pred.sum()))
        recall = hits_gt / max(1, int(gt.sum()))
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        outputs.append({
            "transform": name, "f1_with_2px_tolerance": float(f1),
            "precision": float(precision), "recall": float(recall),
            "map_pixels": int(pred.sum()), "gt_pixels": int(gt.sum()),
        })
    return sorted(outputs, key=lambda item: item["f1_with_2px_tolerance"], reverse=True)


def geometric_anchor_check(map_root: Path, npz: Path) -> dict[str, Any]:
    """根据真实地图+独立 ego pose 投影道路分隔线，与 Step62 保存 GT 对照。"""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap
    from pyquaternion import Quaternion

    if not map_root.is_dir() or not npz.is_file():
        raise FileNotFoundError(f"地图目录或 Step62 NPZ 不存在：{map_root} / {npz}")
    with np.load(npz, allow_pickle=False) as payload:
        gt = np.asarray(payload["gt_mask"])
        token = str(payload["token"].item())
    if gt.shape != (200, 400):
        raise ValueError(f"真实 GT 必须为 200x400，实际 {gt.shape}")
    nusc = NuScenes(version="v1.0-mini", dataroot=str(map_root), verbose=False)
    sample = nusc.get("sample", token)
    scene = nusc.get("scene", sample["scene_token"])
    location = nusc.get("log", scene["log_token"])["location"]
    sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pose = nusc.get("ego_pose", sd["ego_pose_token"])
    calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    yaw_deg = float(np.rad2deg(Quaternion(pose["rotation"]).yaw_pitch_roll[0]))
    patch_w, patch_h = 60.0, 30.0
    # 不使用作者的 VectorizedLocalMap / RasterizeMapVectors GT 管道。
    # 直接从 nuScenesMap 取 GLOBAL 坐标 LineString，再独立转换为局部 Ego。
    nmap = NuScenesMap(dataroot=str(map_root), map_name=location)
    cx, cy = float(pose["translation"][0]), float(pose["translation"][1])
    radius = math.hypot(patch_w / 2, patch_h / 2) + 2.0
    records = nmap.get_records_in_patch(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        ["road_divider", "lane_divider"], mode="intersect"
    )
    sin_yaw, cos_yaw = math.sin(math.radians(yaw_deg)), math.cos(math.radians(yaw_deg))
    pieces: list[np.ndarray] = []
    for layer in ("road_divider", "lane_divider"):
        for record_token in records.get(layer, []):
            record = nmap.get(layer, record_token)
            line = nmap.extract_line(record["line_token"])
            points_world = geometry_to_points(line)
            if len(points_world) == 0:
                continue
            delta = points_world - np.array([cx, cy])[None, :]
            points_ego = np.column_stack((
                cos_yaw * delta[:, 0] + sin_yaw * delta[:, 1],
                -sin_yaw * delta[:, 0] + cos_yaw * delta[:, 1],
            ))
            pieces.append(points_ego)
    if not pieces:
        raise RuntimeError("地图范围内没有 Divider 矢量锚点，无法评估")
    pts = np.concatenate(pieces, axis=0)
    measurements = raster_and_match(pts, gt, patch_w, patch_h)
    direct = next(m for m in measurements if m["transform"] == "direct")
    best = measurements[0]
    # 不可从 F1 直接推出 LiDAR BEV 与 GT 已物理闭环；当前只做地图 ego 锚点检查。
    rotation = Quaternion(calib["rotation"]).rotation_matrix
    translation = np.asarray(calib["translation"], dtype=float)
    probe = np.array([2.0, -1.0, 0.0])
    probe_ego = rotation @ probe + translation
    round_trip = rotation.T @ (probe_ego - translation)
    calibration_error = float(np.max(np.abs(round_trip - probe)))
    if calibration_error > 1e-5:
        raise RuntimeError("真实 LiDAR/Ego 标定正逆变换未通过")
    # 显式把作者真实 pc_range 与 LiDAR→Ego 标定用于米制方向掩码。
    # 行/列排列必须由独立物理一致性实验确认后才能宣称最终部署已对齐。
    from mmcv import Config

    cfg_path = ROOT / "BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg_det.py"
    cfg = Config.fromfile(str(cfg_path))
    pc_range = tuple(float(v) for v in cfg.point_cloud_range)
    shape = (int(cfg.model.pts_bbox_head.bev_h), int(cfg.model.pts_bbox_head.bev_w))
    direction = parse_instruction("Show the crosswalk on the front right")
    directional_mask = calibrated_bev_direction_mask(
        shape, pc_range, rotation, translation, direction
    )
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "step73_front_right_metric_bev_mask.npy", directional_mask.astype(np.uint8))
    result = {
        "status": "MAP_ANCHOR_DIAGNOSTIC_ONLY",
        "source": str(npz), "sample_token": token, "map_name": location,
        "map_root": str(map_root), "ego_position": pose["translation"],
        "ego_yaw_deg": yaw_deg,
        "lidar2ego_translation": calib["translation"],
        "calibration_roundtrip_max_m": calibration_error,
        "author_point_cloud_range": list(pc_range),
        "bev_metric_grid_shape": list(shape),
        "front_right_ego_mask_pixel_ratio": float(directional_mask.mean()),
        "directional_mask_path": str(OUT / "step73_front_right_metric_bev_mask.npy"),
        "bev_axis_assumption": "row=lidar_y, col=lidar_x; independent axis convention not yet established",
        "gt_shape": list(gt.shape), "map_points": int(len(pts)),
        "direct_projection": direct, "best_projection": best,
        "candidate_transforms": measurements,
        "note": (
            "地图几何从官方 nuScenes Map API + ego pose 独立采点栅格化。"
            "检测可能的旋转/翻转和 Map GT 差异；尚未验证真实网络 BEV 投影矩阵"
            "与地图 GT 的完整空间物理闭环。"
        ),
    }
    return result


def report_md(payload: dict[str, Any]) -> str:
    language = payload.get("language", {})
    geometry = payload.get("map_geometry", {})
    lines = [
        "# 第二阶段 Step73：自然语言指令与语义-空间映射补充验收", "",
        "## 目标与范围", "",
        "- 对应师傅第二阶段：自然语言预处理/编码、语义到二维 BEV 映射的功能与证据。",
        "- 不修改作者权重、不训练、不读取 GT 做推理，不声称开放词表语言理解或分割精度提高。",
        "- 物理方向遵守 ego x 向前、y 向左；地图 GT 200×400 与模型 BEV 150×150 不可直接视为同一网格。", "",
        "## 1. 受约束自然语言解析与动态语言编码", "",
        f"- 状态：`{language.get('status', 'NOT_VERIFIED')}`",
        f"- 成功解析：{len(language.get('parsed_commands', []))} 条；明确拒绝：{len(language.get('explicitly_rejected', []))} 条。",
        f"- 文本输出：`{language.get('text_shape')}`；BEV attention：`{language.get('attention_shape')}`。",
        f"- 不同语言注意力 MAE：`{language.get('different_text_attention_mae')}`；关闭模块回退：`{language.get('bypass_exact_identity')}`。",
        "- 限制：英文道路三类关键词/有限否定与方向规则；复杂指代、自由指令和 learned grounding 未通过。",
        "", "## 2. 独立地图锚点与空间协议核验", "",
        f"- 状态：`{geometry.get('status', 'NOT_VERIFIED')}`。",
        f"- 真实 token：`{geometry.get('sample_token', 'N/A')}`；地图：`{geometry.get('map_name', 'N/A')}`。",
        f"- LiDAR→Ego→LiDAR 正逆最大误差：`{geometry.get('calibration_roundtrip_max_m', 'N/A')}` 米。",
    ]
    if geometry.get("direct_projection"):
        for key in ("direct_projection", "best_projection"):
            datum = geometry[key]
            lines.append(f"- {key}: `{datum['transform']}`，2px 容忍 F1={datum['f1_with_2px_tolerance']:.4f}，P={datum['precision']:.4f}，R={datum['recall']:.4f}。")
    if geometry.get("error"):
        lines.append(f"- 尚未完成原因：`{geometry['error']}`。")
    lines.extend([
        "", "## 任务书状态", "",
        "| 任务 | 状态 | 证据/边界 |", "|---|---|---|",
        "| 自然语言预处理与编码 | " + language.get("status", "NOT_VERIFIED") + " | 真实 DistilBERT + 结构化目标、否定、方向解析；不支持开放词表推理 |",
        "| 一维文本→二维 BEV 注意力 | " + language.get("status", "NOT_VERIFIED") + " | 真实 BEV 输入、融合回退、注意力变化；当前融合头为随机初始化功能检查 |",
        "| 地图/BEV 物理坐标 | NOT_VERIFIED | Map ego 锚点是必要诊断，不等于共享 BEV 与 GT 完整物理对齐 |",
        "| 不同指令是否正确定位目标 | NOT_VERIFIED | Step68 置乱文本没有明显变差，需有针对性的语言监督训练与独立验证 |",
        "", "## 文件", "",
        "- `scripts/stage2/semantic_instruction.py`",
        "- `scripts/stage2/run_step73_fill_gaps.py`",
        "- `outputs/stage2_step73/step73_gap_check.json`",
        "", "## 后续", "",
        "如果要求从受限语义指令升级成任意自然语言正确落到地图目标，"
        "需要配对语言-地图区域的训练样本和留出样本上的语义定位评估；"
        "不能仅凭 DistilBERT 768 维特征或二维 Attention shape 判定完成。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", type=Path, default=ROOT / "models/distilbert-base-uncased")
    parser.add_argument("--bev", type=Path, default=ROOT / "outputs/bev_features/bev_feature.npy")
    parser.add_argument("--map-root", type=Path, default=ROOT / "BEVFormer-seg/data/nuscenes")
    parser.add_argument("--gt", type=Path, default=ROOT / "outputs/stage2_step62/step62_divider_sample.npz")
    args = parser.parse_args()
    if not (args.self_test or args.live):
        parser.error("请指定 --self-test 或 --live")
    if args.self_test:
        self_test()
    if args.live:
        OUT.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {"step": 73, "scope": "functional_plus_independent_map_diagnostic"}
        try:
            result["language"] = dynamic_language_and_bev(args.model, args.bev)
            print("[Step73] 自然语言解析 + DistilBERT + 二维 BEV: PASS_FUNCTION_ONLY", flush=True)
        except Exception as exc:
            result["language"] = {"status": "NOT_VERIFIED", "error": str(exc)}
            print(f"[Step73] 语言功能 NOT_VERIFIED：{exc}", flush=True)
        try:
            result["map_geometry"] = geometric_anchor_check(args.map_root, args.gt)
            g = result["map_geometry"]
            print(f"[Step73] 地图锚点最佳 {g['best_projection']['transform']}，"
                  f"F1={g['best_projection']['f1_with_2px_tolerance']:.4f}; "
                  "物理 BEV/GT 尚未闭环", flush=True)
        except Exception as exc:
            result["map_geometry"] = {"status": "NOT_VERIFIED", "error": str(exc)}
            print(f"[Step73] 地图锚点 NOT_VERIFIED：{exc}", flush=True)
        save_json(OUT / "step73_gap_check.json", result)
        DOC.parent.mkdir(parents=True, exist_ok=True)
        DOC.write_text(report_md(result), encoding="utf-8")
        print("[Step73] JSON:", OUT / "step73_gap_check.json")
        print("[Step73] 中文报告:", DOC)
        if result["language"]["status"] != "PASS_FUNCTION_ONLY":
            raise SystemExit("语言功能未验收，请检查报告的缺失或错误")


if __name__ == "__main__":
    main()

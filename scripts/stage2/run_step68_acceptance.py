"""Step68：对第二阶段剩余证据缺口做只读验收与真实文本干预测试。

--quick: 不加载模型，汇总现有 Step58~67 证据并执行单元测试。
--live : 在已有 Mini Val 样本上运行原始/正常/置乱/常量文本的对照。

不修改模型、数据、权重、GT，也不训练或重新选验证集超参数。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/stage2_step68"
DOC = ROOT / "docs/stage2_acceptance_2026-10-21.md"
CONFIG = ROOT / "BEVFormer-seg/projects/configs/bevformer/bevformer_small_seg.py"
STEP58 = ROOT / "outputs/stage2_step58/step58_author_live_smoke.json"
STEP59 = ROOT / "outputs/stage2_step59/step59_training_summary.json"
STEP67 = ROOT / "outputs/stage2_step67/step67_ab_summary.json"
NAMES = ("Background", "Divider", "Ped Crossing", "Boundary")


def read_json(path: Path) -> dict[str, Any] | None:
    """只读取存在的 JSON，缺失文件保留为未验证。"""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    """流式生成文件 SHA256。"""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def average_precision(scores: np.ndarray, truth: np.ndarray) -> float | None:
    """按分数降序计算二分类非插值 AP，正样本为空返回 N/A。"""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=bool).reshape(-1)
    if len(scores) != len(truth) or not np.isfinite(scores).all():
        raise ValueError("AP 输入长度不符或有 NaN/Inf")
    positives = int(truth.sum())
    if positives == 0:
        return None
    # 稳定排序，对相等分数先合并再累计，降低 tie 对 AP 的偏差。
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    y = truth[order].astype(np.int64)
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(y) - 1]
    tp = np.cumsum(y)[ends]
    precision = tp / (ends + 1)
    recall_delta = np.diff(np.r_[0, tp]) / positives
    return float(np.sum(precision * recall_delta))


def confusion_metrics(matrix: np.ndarray) -> dict[str, Any]:
    """按 GT 行、预测列累计 4 类混淆矩阵；缺失类别不计为 0。"""
    result: dict[str, float | None] = {}
    for index, name in enumerate(NAMES):
        union = int(matrix[index, :].sum() + matrix[:, index].sum() - matrix[index, index])
        result[name] = float(matrix[index, index] / union) if union > 0 else None
    fg = [result[name] for name in NAMES[1:]]
    return {
        "per_class_iou": result,
        "foreground_miou": float(np.mean(fg)) if all(v is not None for v in fg) else None,
        "four_class_miou": (
            float(np.mean(list(result.values())))
            if all(v is not None for v in result.values()) else None
        ),
        "confusion_gt_rows_pred_cols": matrix.tolist(),
    }


def self_test() -> None:
    """验证 AP、混淆矩阵和合成多任务 Loss 的计算正确性。"""
    from scripts.stage2.multitask_loss import self_test as loss_test

    loss_test()
    ap = average_precision(np.array([0.95, 0.1, 0.8]), np.array([1, 0, 1]))
    assert ap is not None and abs(ap - 1.0) < 1e-12
    cm = np.diag([5, 4, 3, 2]).astype(np.int64)
    assert confusion_metrics(cm)["four_class_miou"] == 1.0
    print("[Step68] AP、四类 mIoU 单元测试 PASS")


def evidence() -> dict[str, Any]:
    """提取历史结果；仅依据现有文件，不推断不存在的实验。"""
    a = read_json(STEP58)
    b = read_json(STEP59)
    c = read_json(STEP67)
    sources = {
        "step58": str(STEP58),
        "step59": str(STEP59),
        "step67": str(STEP67),
    }
    checks: list[dict[str, str]] = []

    def add(name: str, status: str, reason: str) -> None:
        checks.append({"item": name, "status": status, "evidence": reason})

    step58_pass = (
        a is not None and a.get("bypass_equivalent_tolerance_1e-4") is True
        and a.get("real_pretrained_text_embeddings") is True
    )
    add("真实 DistilBERT + BEVFormer 分割链路",
        "PASS" if step58_pass else "MISSING",
        "Step58 冻结模型同帧真实前向、关闭模块回退" if step58_pass
        else "缺少或未通过 Step58 真实链路报告")
    replay_check = bool(
        b and b.get("train_distinct_frames", 0) > 0
        and b.get("val_distinct_frames", 0) > 0
    )
    add("真实 GT 训练与留出验证", "PASS" if replay_check else "MISSING",
        "Step59 有真实 Train/Val、分割 Loss 与权重" if replay_check
        else "缺少 Step59 原始训练总结")
    locked = (c or {}).get("locked_candidate", {})
    add("语言分割精度可靠提升", "NOT_VERIFIED",
        "Step67 校准最终选 original；既有 Adapter 未稳定改善前景与四类 IoU")
    add("文本语义有效性", "PENDING_LIVE",
        "Step67 最终锁定 original，必须对真实训练 Adapter 做同帧文字置乱/常量干预")
    add("BEV / GT 物理坐标一致性", "NOT_VERIFIED",
        "相同尺寸与 Decoder replay 不等于独立物理坐标验证；仍需地图锚点与实际投影对照")
    add("检测 + 分割 + 注意力真实联合训练", "NOT_INTEGRATED",
        "已实现损失聚合接口并有合成梯度测试，当前真实训练是地图分割链路")
    return {
        "checks": checks,
        "source_files": sources,
        "checkpoint_status": {
            "step67_locked": locked,
            "reason": (c or {}).get("selection_reason"),
        },
        "stage2_tests": {
            "module_source": str(ROOT / "scripts/stage2"),
            "config": str(CONFIG),
            "original_weights_unchanged": True,
        },
    }


def live_text_intervention(frame_count: int) -> dict[str, Any]:
    """在真实作者分割输出上对语言条件 Adapter 进行因果对照。"""
    if not torch.cuda.is_available():
        raise RuntimeError("缺少 CUDA，请切换 .venv-author")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    stage65 = importlib.import_module("scripts.stage2.run_step65_literature_adapter")
    stage59 = importlib.import_module("scripts.stage2.run_step59_train")
    stage60 = importlib.import_module("scripts.stage2.run_step60_extended_ab")
    stage67 = importlib.import_module("scripts.stage2.run_step67_three_in_one")
    old_report = read_json(STEP67)
    if old_report is None:
        raise FileNotFoundError(f"Step67 报告不存在：{STEP67}")
    saved = ROOT / "outputs/stage2_step67/adapter_step67.pt"
    input_paths = {
        "checkpoint": stage65.CHECKPOINT,
        "text_features": stage65.TEXT,
        "adapter_weights": saved,
    }
    for label, file_path in input_paths.items():
        if not file_path.is_file():
            raise FileNotFoundError(f"{label} 缺失：{file_path}")
    if sha256(stage65.CHECKPOINT) != old_report["original_checkpoint_sha256"]:
        raise RuntimeError("作者 checkpoint 哈希与 Step67 不一致")
    if sha256(saved) != old_report["adapter_step67_sha256"]:
        raise RuntimeError("Adapter 权重哈希与 Step67 不一致")

    sys.path.insert(0, str(stage65.AUTHOR))
    os.chdir(stage65.AUTHOR)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    torch.manual_seed(20261021)
    cfg = Config.fromfile(
        str(stage65.AUTHOR / "projects/configs/bevformer/bevformer_small_seg.py")
    )
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    cfg.data.workers_per_gpu = 0
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(stage65.CHECKPOINT), map_location="cpu")
    model.requires_grad_(False).eval()
    wrapped = MMDataParallel(model.cuda(), device_ids=[0])
    head = wrapped.module.pts_bbox_head
    head.eval()
    text = torch.from_numpy(np.load(stage65.TEXT, allow_pickle=False)).float().cuda()
    if text.shape != (3, 768):
        raise ValueError(f"文本缓存形状错误：{text.shape}")
    shuffled = text[[2, 0, 1]]
    constant = text.mean(dim=0, keepdim=True).expand_as(text).contiguous()
    adapter = stage65.TextConditionedLogitAdapter().cuda()
    state = stage67.safe_load(saved)["state_dict"]
    adapter.load_state_dict(state, strict=True)
    adapter.eval().requires_grad_(False)
    dataset = stage59.build_mini_dataset(cfg, build_dataset, "val")
    indices = list(old_report["new_mini_val_indices"])[:frame_count]
    if len(indices) != frame_count:
        raise ValueError("Step67 新 Mini Val 索引数量不足")
    subset = stage60.IndexedDataset(dataset, indices)
    samples = stage60.move_to_cpu(stage59.capture_samples(
        wrapped, head.transformer, head, subset,
        list(range(frame_count)), build_dataloader,
        cfg.data.nonshuffler_sampler, torch, "mini_val",
    ))
    del dataset, subset
    gc.collect()
    torch.cuda.empty_cache()
    matrices = {key: np.zeros((4, 4), dtype=np.int64) for key in (
        "original", "normal", "shuffled", "constant"
    )}
    attention_values = {k: {i: [] for i in range(3)} for k in ("normal", "shuffled")}
    masks = {i: [] for i in range(3)}
    changes = []
    with torch.inference_mode():
        for item in samples:
            bev, gt, ref = stage60.cuda_sample(item, torch)
            preds: dict[str, torch.Tensor] = {"original": ref}
            att_map: dict[str, torch.Tensor] = {}
            for name, txt in (("normal", text), ("shuffled", shuffled), ("constant", constant)):
                pred, att = adapter(bev, ref, txt, head.feat_cropper)
                preds[name] = pred
                att_map[name] = att
            for key, logits in preds.items():
                flat_gt = gt.flatten().long()
                flat_pred = logits.argmax(1).flatten().long()
                bincount = torch.bincount(4 * flat_gt + flat_pred, minlength=16)
                matrices[key] += bincount.reshape(4, 4).cpu().numpy()
            normal = preds["normal"]
            changes.append({
                "token": item.token,
                "normal_vs_shuffled_logit_mae": float((normal - preds["shuffled"]).abs().mean()),
                "normal_vs_constant_logit_mae": float((normal - preds["constant"]).abs().mean()),
                "argmax_changed_by_shuffle_ratio": float(
                    (normal.argmax(1) != preds["shuffled"].argmax(1)).float().mean()
                ),
            })
            for cls in range(3):
                masks[cls].append((gt[0] == cls + 1).cpu().numpy().reshape(-1))
                for mode in ("normal", "shuffled"):
                    attention_values[mode][cls].append(
                        att_map[mode][0, cls].detach().cpu().numpy().reshape(-1)
                    )
            print(f"[Step68] 文本干预：{len(changes)}/{len(samples)}", flush=True)
            del bev, gt, ref, preds, att_map
    evaluated = {k: confusion_metrics(v) for k, v in matrices.items()}
    attention_ap: dict[str, Any] = {}
    for i, name in enumerate(NAMES[1:]):
        labels = np.concatenate(masks[i])
        attention_ap[name] = {
            "gt_share": float(labels.mean()),
            "normal_ap": average_precision(np.concatenate(attention_values["normal"][i]), labels),
            "shuffled_ap": average_precision(np.concatenate(attention_values["shuffled"][i]), labels),
        }
    mean_logit_change = float(np.mean([v["normal_vs_shuffled_logit_mae"] for v in changes]))
    changed_pixels = float(np.mean([v["argmax_changed_by_shuffle_ratio"] for v in changes]))
    return {
        "samples": len(samples),
        "indices": indices,
        "tokens": [v["token"] for v in changes],
        "model": "step67_raw_adapter_not_locked_original",
        "normal_vs_shuffled_mean_abs_logit_delta": mean_logit_change,
        "argmax_changed_by_shuffle_fraction": changed_pixels,
        "has_numerical_text_effect": mean_logit_change > 1e-6,
        "segmentation": evaluated,
        "attention_ap": attention_ap,
        "per_frame": changes,
        "interpretation": (
            "能否检测到文本导致的输出变化，仅是文本依赖的必要条件；"
            "AP 高于置乱文本仍需更大样本与开放语义测试，才能支持语言定位结论。"
        ),
    }


def markdown(result: dict[str, Any]) -> str:
    """按证据强度生成中文验收报告，不把未证实项目判成 PASS。"""
    lines = [
        "# 第二阶段核心模块验收记录（Step68）", "",
        "## 验收范围", "",
        "- 项目：BEVFormer + DistilBERT + BEV 注意力与分割 Adapter。",
        "- 复用 Step58～67 原始日志及冻结权重；不修改 GT、checkpoint 或训练结果。",
        "- 时序 OFF 的单帧对照与 81 帧时序评估不得直接混用。", "",
        "## 验收清单", "",
        "| 检查项 | 状态 | 证据 / 缺口 |", "|---|---|---|",
    ]
    for check in result["checks"]:
        lines.append(f"| {check['item']} | {check['status']} | {check['evidence']} |")
    lines.extend(["", "## 真实文本干预（训练后 Raw Adapter）", ""])
    live = result.get("text_intervention")
    if live is None:
        lines.append("尚未运行 `--live`；只能说明数值链路存在，不可宣称语言语义已验证。")
    else:
        lines.extend([
            f"- 真实验证帧：{live['samples']}；原始评估不修改 GT 或权重。",
            f"- 正常 vs 置乱文本 logits MAE：{live['normal_vs_shuffled_mean_abs_logit_delta']:.8g}。",
            f"- 置乱文字导致 argmax 像素变化比例：{live['argmax_changed_by_shuffle_fraction']:.6%}。",
            "", "| 前景 mIoU | Original | Normal | Shuffled | Constant |",
            "|---|---:|---:|---:|---:|",
        ])
        row = live["segmentation"]
        lines.append("| 真实同帧结果 | " + " | ".join(
            "N/A" if row[m]["foreground_miou"] is None
            else f"{row[m]['foreground_miou']:.4%}"
            for m in ("original", "normal", "shuffled", "constant")
        ) + " |")
        lines.extend(["", "| Attention AP | GT 占比 | 正常文本 AP | 置乱 AP |",
                      "|---|---:|---:|---:|"])
        for name, entry in live["attention_ap"].items():
            fmt = lambda x: "N/A" if x is None else f"{x:.4%}"
            lines.append(
                f"| {name} | {fmt(entry['gt_share'])} | "
                f"{fmt(entry['normal_ap'])} | {fmt(entry['shuffled_ap'])} |"
            )
    lines.extend([
        "", "## 仍需独立完成的验证", "",
        "- **BEV / GT 物理坐标：** 必须在真实地图锚点/ego2global/sensor2lidar 上完成双路径投影一致性；Decoder replay 不能代替。",
        "- **检测多任务训练：** 当前提供可训练损失聚合接口，但未证明 BEVFormer 检测头、分割头和注意力可在同一批次联合反传。",
        "- **开放式语言：** 三类固定文本类别模板的测试不能证明复杂自然驾驶指令可正确落到目标区域。",
        "- **稳定精度提升：** Step67 锁定 original 是既定的安全回退，不得汇报成 Adapter 取得总体指标提升。",
        "", "## 主要输入及输出", "",
        f"- Step58 报告：`{STEP58}`",
        f"- Step59 报告：`{STEP59}`",
        f"- Step67 报告：`{STEP67}`",
        f"- 本次 JSON：`{OUT / 'step68_acceptance.json'}`",
        "", "## 结论", "",
        "完成了可追溯的分割路径验收和测试证据整理；未满足的项目保留真实状态。",
    ])
    return "\n".join(lines) + "\n"


def architecture_doc() -> str:
    """第二阶段架构说明：只描述当前源码实际支持的路径。"""
    return """# 第二阶段：语言引导 BEV 注意力模块设计说明

## 系统结构

```text
驾驶指令 → 本地冻结 DistilBERT → (3, 768) 类别文本向量
                                      │
六相机图像 → 冻结 BEVFormer → BEV (1, 256, 150, 150)
                                      │
              ┌───────────────────────┤
              ↓                       ↓
        原始分割 Decoder     Semantic projection / attention
              ↓                       ↓
     logits (1,4,200,400)    Attention (1,3,200,400)
              └───────────┬───────────┘
                          ↓
                   Logit Adapter（可训练）
                          ↓
                   最终四类语义地图
```

## 模块接口及限制

- 语言编码：`language_encoder.py`，预训练 DistilBERT 768 维文本特征。现有训练使用冻结、预先缓存的模板文本特征。
- 投影：`semantic_projection.py`，将 768 维输入映射到 BEV 通道空间。
- BEV 空间注意力：`language_bev_attention.py`，在 BEV 特征网格计算语言条件注意力。
- BEV 残差融合：`bev_fusion.py`，允许关闭模块后严格回退原始特征。
- 原始分割头集成：Step58 在 `transformer.forward` 输出后插入融合模块；Step65/67 使用冻结 BEV 特征上的四类 logit Adapter。
- 类别空间：Background / Divider / Ped Crossing / Boundary；GT 为 `(1,200,400)`。
- 评价：累积混淆矩阵与四类 argmax，独立报告前景三类 mIoU 和四类 mIoU。
- 检测联合损失：`multitask_loss.py` 仅提供聚合接口；当前并未验收检测 + 分割的真实联合推理与反传。

## 训练方案

- 真实 nuScenes v1.0-mini Mini Train 地图 GT；冻结 BEVFormer、分割 Decoder 和 DistilBERT，训练语言空间注意力与校正适配器。
- 有监督分割 Loss：CE / Focal CE / Soft-IoU / Tversky 实验变体；有监督 Attention BCE/Dice。
- Step65～67 建立 Bias-only、Text-shuffled、Original 等对照。
- Step67 Mini Train 校准无符合全部条件的模型，最后回退 Original。不得将 Raw Adapter 的局部收益视作已通过总体验收。

## 与阶段任务的边界

1. 真实 BEVFormer 分割链路及语言模块实现已完成；开放式自由驾驶指令定位尚未验证。
2. BEV / GT 物理坐标仍需独立锚点核验；重放 Decoder 一致仅证明内部张量排列正确。
3. 检测、分割、注意力三任务真正共同训练尚未完成；当前交付的是可扩展损失接口及其单元测试。
"""


def unit_test_doc(result: dict[str, Any]) -> str:
    """总结自测条件与真实试验、未通过事项，避免夸大。"""
    lines = [
        "# 第二阶段：模块与验收测试记录", "",
        "## 本轮独立单元测试", "",
        "| 测试 | 期待行为 | 结果 |", "|---|---|---|",
        "| 多任务损失接口 | 检测/分割/注意力梯度均可回传 | 合成数据 PASS |",
        "| AP | 完全正确排序得到 1 | 合成数据 PASS |",
        "| 四类 IoU | 完全预测匹配 GT 得到 1 | 合成数据 PASS |",
        "", "## 真实模型/数据验证", "",
        "| 项目 | 状态 |", "|---|---|",
    ]
    for item in result["checks"]:
        lines.append(f"| {item['item']} | {item['status']} |")
    lines.extend([
        "", "## 遗留项目（未通过或未核验）", "",
        "- Real 3-task detection + segmentation + attention joint backward：未验收。",
        "- nuScenes 地图 GT 与 LiDAR/ego/world 的独立物理锚点一致性：未验收。",
        "- 自由文本正确指代多个对象的语义定位：未验收。",
        "- 在完整 81 帧时序或官方 val 上稳定改善四类 mIoU：未验收。",
        "", "## 固定条件", "",
        "- Stage2 Step68 同帧文字干预：真实 Mini Val，时序 OFF，冻结作者模型和 Decoder。",
        "- 只保存报告与测试数据，不编辑作者代码、权重或数据。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    """先快速审计，然后按需执行真实模型文本干预，最终落盘报告。"""
    parser = argparse.ArgumentParser(description="Step68 第二阶段验收")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--frames", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.frames <= 12:
        parser.error("--frames 必须在 1~12")
    if args.self_test:
        self_test()
        return
    self_test()
    result = evidence()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.live:
        result["text_intervention"] = live_text_intervention(args.frames)
        check = next(item for item in result["checks"] if item["item"] == "文本语义有效性")
        check["status"] = "TEXT_DEPENDENCE_TESTED" if result["text_intervention"]["has_numerical_text_effect"] else "NO_MEASURABLE_EFFECT"
        check["evidence"] = "真实作者 BEV + Step67 Raw Adapter，正常/置乱/常量同帧干预；详见 AP 和 mIoU"
    (OUT / "step68_acceptance.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(markdown(result), encoding="utf-8")
    (ROOT / "docs/stage2_architecture_CN.md").write_text(
        architecture_doc(), encoding="utf-8"
    )
    (ROOT / "docs/stage2_unit_tests_CN.md").write_text(
        unit_test_doc(result), encoding="utf-8"
    )
    print("[Step68] JSON:", OUT / "step68_acceptance.json")
    print("[Step68] 中文验收报告:", DOC)
    print("[Step68] 状态：", ", ".join(
        f"{item['item']}={item['status']}" for item in result["checks"]
    ))


if __name__ == "__main__":
    main()

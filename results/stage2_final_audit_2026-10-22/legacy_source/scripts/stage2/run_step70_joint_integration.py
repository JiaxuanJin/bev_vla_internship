"""Step70：BEVFormer 联合检测/分割架构审计与共享 BEV 语言融合实测。

提供三种模式：
    --self-test：无需第三方模型/数据的模块形状和回退测试。
    --audit：读取当前本地配置与权重，严格审计双 Head 配置及权重结构。
    --live --joint-checkpoint PATH：加载真正联合训练权重，验证共享 BEV 注入。

本程序不会更改作者工程、GT 或 checkpoint，不运行真实联合反传。
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg"
LOCAL_SEG_REPO = ROOT / "BEVFormer-seg"
CONFIGS = {
    "det": ROOT / "BEVFormer/projects/configs/bevformer/bevformer_small.py",
    "seg": LOCAL_SEG_REPO / "projects/configs/bevformer/bevformer_small_seg.py",
    "joint": LOCAL_SEG_REPO / "projects/configs/bevformer/bevformer_small_seg_det.py",
}
WEIGHTS = {
    "det": ROOT / "BEVFormer/ckpts/bevformer_small_epoch_24.pth",
    "seg": Path("/mnt/g/nus/bevformer_small_seg.pth"),
}
OUT = ROOT / "outputs/stage2_step70"
DOC = ROOT / "docs/stage2_step70_joint_integration_CN.md"
JOINT_URL = (
    "https://drive.google.com/file/d/"
    "1a42LRJhI9FoIeOORTwEf0N2Vtu98wP9O/view?usp=share_link"
)


def sha256(path: Path) -> str:
    """流式读取大文件，生成内容校验值。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_inventory(path: Path, allow_legacy: bool = False) -> dict[str, Any]:
    """逐个打开 checkpoint，读取关键 Head 权重，避免两个大文件同时驻留内存。"""
    import torch

    # 权重只来自本机可信的作者 checkpoint；不执行任意外部代码。
    try:
        saved = torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        if not allow_legacy:
            raise RuntimeError(
                f"安全模式无法读取 {path}：{exc}。如果这是可信来源的官方权重，"
                "可以明确加 --allow-legacy-checkpoint 再试。"
            ) from exc
        print(f"[Step70] 警告：已显式允许传统 checkpoint 读取：{path}")
        saved = torch.load(str(path), map_location="cpu", weights_only=False)
    state = saved.get("state_dict", saved) if isinstance(saved, dict) else saved
    if not isinstance(state, dict):
        raise TypeError(f"{path} 不是有效 state_dict/checkpoint")
    keys = [str(key).removeprefix("module.") for key in state]
    useful = {
        "seg_decoder": "pts_bbox_head.seg_decoder.",
        "cls_branches": "pts_bbox_head.cls_branches.",
        "reg_branches": "pts_bbox_head.reg_branches.",
        "bev_encoder": "pts_bbox_head.transformer.encoder.",
    }
    counts = {
        name: sum(key.startswith(prefix) for key in keys)
        for name, prefix in useful.items()
    }
    result = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "total_state_keys": len(keys),
        "key_counts": counts,
        "both_heads_present": (
            counts["seg_decoder"] > 0
            and counts["cls_branches"] > 0
            and counts["reg_branches"] > 0
        ),
        "meta": {},
    }
    if isinstance(saved, dict) and isinstance(saved.get("meta"), dict):
        for name in ("epoch", "iter", "config", "mmdet_version"):
            value = saved["meta"].get(name)
            if isinstance(value, (str, int, float, bool)):
                result["meta"][name] = value if name != "config" else value[:300]
    del state, saved
    gc.collect()
    return result


def read_configs() -> dict[str, Any]:
    """用 MMCV 读取真实配置，禁止用文件名替代任务检查。"""
    from mmcv import Config

    result = {}
    for name, path in CONFIGS.items():
        if not path.is_file():
            result[name] = {"exists": False, "path": str(path)}
            continue
        cfg = Config.fromfile(str(path))
        head = cfg.model.pts_bbox_head
        transformer = head.transformer
        task = dict(head.get("task", {}))
        result[name] = {
            "exists": True,
            "path": str(path),
            "tasks": {"seg": bool(task.get("seg")), "det": bool(task.get("det"))},
            "bev_shape": [int(head.bev_h), int(head.bev_w)],
            "embed_dims": int(transformer.embed_dims),
            "seg_head_defined": bool(head.get("seg_encoder")),
            "decoder_defined": bool(transformer.get("decoder")),
            "test_pipeline_types": [
                str(item.get("type")) for item in cfg.test_pipeline
            ] if cfg.get("test_pipeline") else [],
            "test_ann_file": str(cfg.data.test.get("ann_file", "")),
            "map_grid_conf": dict(cfg.get("map_grid_conf", {})),
        }
    return result


def read_optional_joint(args: argparse.Namespace) -> Path | None:
    """只能使用用户显式提供的联合模型权重，不能拼接单任务模型冒充。"""
    path = args.joint_checkpoint
    if path is None:
        possible = ROOT / "models/bevformer_small_joint_author.pth"
        return possible if possible.is_file() else None
    return path.expanduser().resolve()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    """检查配置、预训练权重覆盖情况，保存可复现的审计报告。"""
    result: dict[str, Any] = {
        "step": 70,
        "mode": "read_only_joint_architecture_and_checkpoint_audit",
        "configs": read_configs(),
        "checkpoints": {},
        "joint_checkpoint_reference": JOINT_URL,
        "status": "NOT_VERIFIED",
    }
    for name, path in WEIGHTS.items():
        result["checkpoints"][name] = (
            checkpoint_inventory(path, args.allow_legacy_checkpoint) if path.is_file() else {"missing": str(path)}
        )
    joint = read_optional_joint(args)
    if joint is not None and joint.is_file():
        result["checkpoints"]["joint"] = checkpoint_inventory(joint, args.allow_legacy_checkpoint)
    else:
        result["checkpoints"]["joint"] = {
            "missing": str(joint) if joint else "尚未指定联合权重",
            "official_reference": JOINT_URL,
        }
    cfg = result["configs"].get("joint", {})
    joint_info = result["checkpoints"]["joint"]
    valid_cfg = (
        cfg.get("tasks") == {"seg": True, "det": True}
        and cfg.get("seg_head_defined")
        and cfg.get("decoder_defined")
    )
    result["joint_config_valid"] = bool(valid_cfg)
    result["joint_weight_has_both_heads"] = bool(
        joint_info.get("both_heads_present", False)
    )
    if valid_cfg and joint_info.get("both_heads_present"):
        result["status"] = "READY_FOR_REAL_JOINT_FORWARD"
    elif valid_cfg:
        result["status"] = "JOINT_ARCHITECTURE_FOUND_WEIGHT_REQUIRED"
    else:
        result["status"] = "JOINT_CONFIG_NOT_READY"

    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / "step70_audit.json", result)
    print("[Step70] 联合配置:", "PASS" if valid_cfg else "NOT_VERIFIED")
    for name, item in result["checkpoints"].items():
        print(f"[Step70] {name} 权重: {item.get('key_counts', item.get('missing'))}")
    print("[Step70] 当前状态:", result["status"])
    print("[Step70] 审计报告:", OUT / "step70_audit.json")
    return result


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """保存 JSON，不覆盖已有 checkpoint 或上游数据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def apply_language_fusion(bev: Any, fusion: Any, pooled: Any, enabled: bool) -> Any:
    """按 (B,HW,C) 空间顺序融合，保留原始形状及 dtype。"""
    import torch

    if not torch.is_tensor(bev) or tuple(bev.shape[1:]) != (22500, 256):
        raise ValueError(f"预期共享 BEV 为 (B,22500,256)，实际 {getattr(bev,'shape',None)}")
    if not enabled:
        return bev  # 精确回退，保留原 tensor 对象。
    bs = int(bev.shape[0])
    map_tensor = bev.reshape(bs, 150, 150, 256).permute(0, 3, 1, 2).float()
    text = pooled.to(device=bev.device).expand(bs, -1)
    enriched, _ = fusion(map_tensor, text, enabled=True)
    result = enriched.permute(0, 2, 3, 1).reshape(bs, 22500, 256)
    if not torch.isfinite(result).all():
        raise ValueError("语言增强 BEV 出现 NaN/Inf")
    return result.to(dtype=bev.dtype)


def self_test() -> None:
    """纯张量测试共享 BEV 前置插入和关闭后精确相等。"""
    import torch

    # 测试不依赖第三方模型，也不构成真实双 Head 已集成的证据。
    x = torch.randn(1, 22500, 256)
    text = torch.randn(1, 768)

    class DummyFusion(torch.nn.Module):
        def forward(self, fmap: Any, _text: Any, enabled: bool = True) -> Any:
            return fmap + 0.01 * fmap.sigmoid(), fmap[:, :1].sigmoid()

    fusion = DummyFusion()
    off = apply_language_fusion(x, fusion, text, False)
    on = apply_language_fusion(x, fusion, text, True)
    assert off.data_ptr() == x.data_ptr()
    assert on.shape == x.shape
    assert torch.isfinite(on).all()
    assert torch.max(torch.abs(on - x)).item() > 0
    print("[Step70] CPU 自检 PASS：150x150 共享 BEV 变换、开关回退、有限值")


def tensor_copy(value: Any, torch: Any) -> Any:
    """只保存推理头必要的张量，避免保留完整模型计算图。"""
    if value is None or not torch.is_tensor(value):
        return None
    return value.detach().float().cpu().contiguous().clone()


def compare_tensors(x: Any, y: Any, torch: Any) -> dict[str, Any]:
    """检测/分割两分支均要求维度一致且有限。"""
    if x is None or y is None:
        raise RuntimeError("真实 Head 缺少检测或分割输出")
    if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise RuntimeError("Head 输出维度或有限值检查失败")
    delta = (x - y).abs()
    return {"shape": list(x.shape), "mean_abs_difference": float(delta.mean()),
            "max_abs_difference": float(delta.max())}


def live(args: argparse.Namespace, report: dict[str, Any]) -> None:
    """使用作者联合 checkpoint 跑原始/关闭/增强的真实检测+分割前向。"""
    import torch

    if report["status"] != "READY_FOR_REAL_JOINT_FORWARD":
        raise RuntimeError("真实双 Head 推理需要有效联合权重；当前不可假装成功")
    if not torch.cuda.is_available():
        raise RuntimeError("检测不到 CUDA；请在作者 .venv-author 环境执行")
    if not AUTHOR.is_dir():
        raise FileNotFoundError(f"作者隔离工程不存在：{AUTHOR}")
    joint = read_optional_joint(args)
    assert joint is not None
    author_cfg = AUTHOR / "projects/configs/bevformer/bevformer_small_seg_det.py"
    if not author_cfg.is_file():
        raise FileNotFoundError(
            f"缺少作者隔离环境的联合配置 {author_cfg}；请先复制官方同版本配置，"
            "不要用不匹配插件的配置强行替代"
        )
    prompt_file = ROOT / "outputs/stage2_step58/distilbert_prompts.npy"
    if not prompt_file.is_file():
        raise FileNotFoundError(f"缺少真实 DistilBERT 缓存：{prompt_file}")
    prompts = np.load(prompt_file, allow_pickle=False)
    if prompts.ndim != 2 or prompts.shape[1] != 768 or prompts.shape[0] < 2:
        raise ValueError(f"DistilBERT 向量维度不合要求：{prompts.shape}")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(AUTHOR))
    os.chdir(AUTHOR)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from scripts.stage2.bev_fusion import LanguageBEVFusion

    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    cfg = Config.fromfile(str(author_cfg))
    task = dict(cfg.model.pts_bbox_head.task)
    if task != {"seg": True, "det": True}:
        raise ValueError(f"真实联合模型必须两个任务都启用：{task}")
    if (int(cfg.model.pts_bbox_head.bev_h), int(cfg.model.pts_bbox_head.bev_w)) != (150, 150):
        raise ValueError("仅支持作者 Small 150x150 BEV")
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model.video_test_mode = False
    cfg.data.test.test_mode = True
    cfg.data.test.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0

    cfg.data.test.pop("samples_per_gpu", None)
    dataset = build_dataset(cfg.data.test)
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"索引 {args.index} 不在数据集范围 0..{len(dataset)-1}")
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False,
        shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    it = iter(loader)
    data = None
    for _ in range(args.index + 1):
        data = next(it)
    token = str(dataset.data_infos[args.index]["token"])
    assert data is not None

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, str(joint), map_location="cpu")
    model.CLASSES = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model.eval()
    wrapped = MMDataParallel(model.cuda(), device_ids=[0])
    base = wrapped.module
    transformer = base.pts_bbox_head.transformer
    if not hasattr(transformer, "get_bev_features"):
        raise RuntimeError("作者 Transformer 没有共享 BEV 接口")

    torch.manual_seed(20261021)
    fusion = LanguageBEVFusion(text_dim=768, alpha_init=0.01).cuda().eval()
    pooled = torch.tensor(prompts[:2], device="cuda", dtype=torch.float32)
    state = {"mode": "original", "hook_calls": 0, "head_calls": 0,
             "raw": {}, "outputs": {}}

    original_bev_method = transformer.get_bev_features

    def injected_bev_method(*call_args: Any, **call_kwargs: Any) -> Any:
        """临时包裹 Python 方法：输出必须在检测 Decoder 执行前被修改。"""
        output = original_bev_method(*call_args, **call_kwargs)
        state["hook_calls"] += 1
        if state["mode"] == "bypass":
            return output
        if state["mode"] == "enhanced":
            return apply_language_fusion(output, fusion, pooled[:1], True)
        raise RuntimeError(f"非法共享 BEV 模式 {state['mode']}")

    def head_hook(_module: Any, _inputs: Any, output: Any) -> None:
        state["head_calls"] += 1
        if not isinstance(output, dict):
            raise RuntimeError("作者联合 Head 没有返回字典")
        keys = ("seg_preds", "all_cls_scores", "all_bbox_preds")
        state["raw"] = {key: tensor_copy(output.get(key), torch) for key in keys}
        if any(state["raw"][key] is None for key in keys):
            raise RuntimeError(f"Head 缺少任一检测或分割输出：{list(output)}")

    head_handle = base.pts_bbox_head.register_forward_hook(head_hook)
    try:
        def infer(name: str) -> dict[str, Any]:
            base.prev_frame_info = {
                "prev_bev": None, "scene_token": None,
                "prev_pos": 0, "prev_angle": 0,
            }
            state["mode"] = name
            state["hook_calls"] = 0
            state["head_calls"] = 0
            state["raw"] = {}
            if name != "original":
                transformer.get_bev_features = injected_bev_method
            try:
                # GT 留在 data 中用于评估，不传给作者的 simple_test。
                inference_data = copy.deepcopy(data)
                inference_data.pop("semantic_indices", None)
                with torch.inference_mode():
                    result = wrapped(
                        return_loss=False, rescale=True, **inference_data
                    )
            finally:
                transformer.get_bev_features = original_bev_method
            if state["head_calls"] != 1:
                raise RuntimeError(f"{name} Head 调用次数不等于 1")
            if name != "original" and state["hook_calls"] != 1:
                raise RuntimeError(f"{name} 共享 BEV Hook 调用次数不等于 1")
            if not isinstance(result, (list, tuple)) or len(result) != 1:
                raise RuntimeError(f"{name} 输出不是单样本结果")
            sample = result[0]
            if not isinstance(sample, dict) or sample.get("pts_bbox") is None:
                raise RuntimeError(f"{name} 没有产生真实 3D 检测输出")
            if sample.get("seg_preds") is None:
                raise RuntimeError(f"{name} 没有产生真实地图分割输出")
            return {key: value.clone() for key, value in state["raw"].items()}

        print("[Step70] 联合模型 GPU:", torch.cuda.get_device_name(0), flush=True)
        print("[Step70] sample_token:", token, flush=True)
        original = infer("original")
        bypass = infer("bypass")
        enhanced = infer("enhanced")
    finally:
        transformer.get_bev_features = original_bev_method
        head_handle.remove()

    metrics = {}
    for key in ("seg_preds", "all_cls_scores", "all_bbox_preds"):
        metrics[key] = {
            "bypass_vs_original": compare_tensors(original[key], bypass[key], torch),
            "enhanced_vs_original": compare_tensors(original[key], enhanced[key], torch),
        }
    bypass_ok = all(
        metrics[key]["bypass_vs_original"]["max_abs_difference"] <= 1e-4
        for key in metrics
    )
    result = {
        "step": 70,
        "status": "REAL_DUAL_HEAD_FORWARD_PASS" if bypass_ok else "BYPASS_FAILED",
        "sample_token": token,
        "joint_checkpoint": str(joint),
        "joint_sha256": report["checkpoints"]["joint"]["sha256"],
        "shared_bev_hook": "PerceptionTransformer.get_bev_features",
        "language_embedding": str(prompt_file),
        "language_fusion_weight_trained": False,
        "temporal_mode": "OFF; three isolated passes",
        "tensor_results": metrics,
        "joint_training_verified": False,
        "semantic_understanding_verified": False,
        "checkpoint_is_author_joint": "requires manual provenance validation",
        "note": "真实共享 BEV 双 Head 前向；语言融合随机初始化，仅为结构功能实验。",
    }
    save_json(OUT / "step70_joint_live.json", result)
    print("[Step70] 联合 Head 状态:", result["status"])
    for key, value in metrics.items():
        print("[Step70]", key, "关闭最大差:",
              value["bypass_vs_original"]["max_abs_difference"],
              "增强平均差:", value["enhanced_vs_original"]["mean_abs_difference"])
    if not bypass_ok:
        raise RuntimeError("关闭融合时检测/分割输出与原始输出不一致；不判定集成成功")


def write_review(audit_result: dict[str, Any]) -> None:
    """写中文审计结论，区分代码架构、权重和真实集成。"""
    DOC.parent.mkdir(parents=True, exist_ok=True)
    status = audit_result["status"]
    lines = [
        "# Step70：检测/分割联合架构与语言融合审计",
        "",
        f"- 当前状态：`{status}`",
        "- 已核对本地检测、分割、联合配置，检查 Head 任务开关和权重键。",
        "- 不能把单独检测/分割 checkpoint 拼成联合训练的证据。",
        "- 共享 BEV 的正确插入点为 `PerceptionTransformer.get_bev_features` 的输出，",
        "  该输出先传给检测 Decoder，同时为分割 Head 使用。",
        "- Step70 的前向实测（如执行 `--live`）只验证双 Head 接口和开关回退，",
        "  不代表真实联合反向传播或自然语言语义理解。",
        "- 真实检测+分割+注意力联合训练与物理坐标协议，需下一步专项验证。",
        "",
        f"- 作者联合 checkpoint 参考：{JOINT_URL}",
        "- 审计 JSON：`outputs/stage2_step70/step70_audit.json`",
        "- 可选实测 JSON：`outputs/stage2_step70/step70_joint_live.json`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """运行所选 Step70 模式。"""
    parser = argparse.ArgumentParser(description="Step70 真实双 Head 架构审计/联动检查")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--joint-checkpoint", type=Path, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    args = parser.parse_args()
    if not (args.self_test or args.audit or args.live):
        parser.error("请指定 --self-test、--audit 或 --live")
    if args.self_test:
        self_test()
    if args.audit or args.live:
        result = audit(args)
        write_review(result)
        if args.live:
            live(args, result)


if __name__ == "__main__":
    main()

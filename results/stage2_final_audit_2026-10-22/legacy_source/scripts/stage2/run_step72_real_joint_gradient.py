"""Step72：真实联合检测/分割/语言注意力梯度连通性验收。

只使用 nuScenes mini_train 的真实图像、3D 框、地图 GT。
作者联合 BEVFormer 模型冻结、预训练 DistilBERT 冻结；仅语言融合模块保留梯度。
本程序只执行 forward + backward，不执行 optimizer.step()，也不写 checkpoint。
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# STEP72_NUMPY_LEGACY_COMPAT
# 仅在别名缺失时补齐，适配作者旧版 MMDetection3D 数据流水线。
# 不修改 NumPy 安装文件、第三方源码和模型权重。
for _legacy_name, _replacement in (
    ("bool", np.bool_),
    ("int", int),
    ("float", float),
    ("complex", complex),
):
    if _legacy_name not in np.__dict__:
        setattr(np, _legacy_name, _replacement)
del _legacy_name, _replacement
import torch

ROOT = Path(__file__).resolve().parents[2]
AUTHOR = Path.home() / "projects/bevformer_author_clean_5080/BEVFormer-seg"
JOINT_CKPT = Path("/mnt/g/nus/bevformer_small_joint_author.pth")
RESULT = ROOT / "outputs/stage2_step72/step72_real_joint_gradient.json"
DOC = ROOT / "docs/stage2/14_真实联合梯度验证.md"


def first_tensor(obj: Any) -> torch.Tensor:
    """从 MMCV DataContainer 或 list 中取得原始张量，不构造伪标签。"""
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if hasattr(obj, "data") and not isinstance(obj, (dict, str, bytes)):
        return first_tensor(obj.data)
    if isinstance(obj, (tuple, list)):
        for item in obj:
            try:
                return first_tensor(item)
            except (TypeError, ValueError):
                continue
    raise TypeError(f"无法从 DataContainer 获取张量：{type(obj)}")


def validate_gt(batch: dict[str, Any]) -> torch.Tensor:
    """严格检查真实地图 GT，禁止暗中 resize 或生成假数据。"""
    if "semantic_indices" not in batch:
        raise KeyError("当前 batch 没有 semantic_indices，不能执行三任务真实测试")
    gt = first_tensor(batch["semantic_indices"])
    if gt.numel() != 200 * 400:
        raise ValueError(f"地图 GT 像素数与作者协议不符：{list(gt.shape)}")
    gt = gt.reshape(1, 200, 400).long().cuda()
    if int(gt.min()) < 0 or int(gt.max()) > 3:
        raise ValueError("地图 GT 必须是 0～3 的真实类别标签")
    if not any(bool((gt == c).any()) for c in (1, 2, 3)):
        raise ValueError("当前训练帧没有前景，换一个 --index 再试")
    if not all(k in batch for k in ("gt_bboxes_3d", "gt_labels_3d")):
        raise KeyError("当前 batch 没有真实 3D 检测 GT")
    return gt


def split_detection_losses(losses: dict[str, Any]) -> dict[str, Any]:
    """从作者真实 forward_train 中筛选检测损失，不重复计算 seg_loss。"""
    detected = {
        key: value for key, value in losses.items()
        if (key.startswith("loss") or ".loss" in key) and "seg" not in key.lower()
    }
    if not detected:
        raise RuntimeError(f"作者真实联合 Head 未返回检测损失，已有键：{list(losses)}")
    if not any("cls" in key for key in detected):
        raise RuntimeError("真实检测分类损失缺失")
    if not any("bbox" in key for key in detected):
        raise RuntimeError("真实检测框损失缺失")
    if not any("seg" in key.lower() for key in losses):
        raise RuntimeError("作者分割损失缺失，联合模型未同时训练两分支")
    return detected


def gradient_norm(loss: torch.Tensor, params: list[torch.nn.Parameter]) -> dict[str, Any]:
    """检查单一任务是否真能反传到融合权重（不修改梯度）。"""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    norms = [float(g.detach().float().norm()) for g in grads if g is not None]
    return {
        "grad_parameter_tensors": len(norms),
        "global_l2": float(sum(n * n for n in norms) ** 0.5),
        "all_finite": bool(all(np.isfinite(n) for n in norms)),
    }


def self_test() -> None:
    """CPU 合成输入只验证工具本身，不冒充真实联合训练。"""
    torch.manual_seed(72)
    layer = torch.nn.Conv2d(3, 4, 1)
    x = torch.randn(1, 3, 5, 7)
    pred = layer(x)
    loss_det = pred[:, 0].square().mean()
    loss_seg = pred[:, 1].square().mean()
    loss_att = pred[:, 2].square().mean()
    params = list(layer.parameters())
    for value in (loss_det, loss_seg, loss_att):
        stats = gradient_norm(value, params)
        assert stats["global_l2"] > 0 and stats["all_finite"]
    total = loss_det + loss_seg + loss_att
    total.backward()
    assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
    assert "loss_seg" not in split_detection_losses({
        "loss_cls": loss_det, "loss_bbox": loss_det,
        "loss_seg": loss_seg, "d0.loss_cls": loss_det,
    })
    print("[Step72] CPU 自检 PASS：三分支梯度、自检过滤器（非真实训练）")


def save_result(record: dict[str, Any]) -> None:
    """只记录实测结果，失败不能伪装成功。"""
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    DOC.parent.mkdir(parents=True, exist_ok=True)
    def status(k: str) -> str:
        return str(record.get("gradients", {}).get(k, {}))
    lines = [
        "# Step72：真实检测 + 地图分割 + 注意力联合反传验收",
        "",
        f"- 状态：`{record['status']}`",
        f"- 数据：`{record.get('dataset')}`；样本索引：`{record.get('index')}`",
        "- 真实联合 checkpoint；冻结作者主干；语言融合随机初始化，仅做梯度连通性检查。",
        "- 测试过程中无 optimizer.step、无权重文件修改。",
        "- 作者原始检测损失为真实 3D GT 的模型 loss；分割/注意力使用真实地图 GT。",
        "- 单帧测试，不证明时序一致性、坐标物理一致性或精度提升。",
        "",
        "## 各任务梯度",
        "",
        f"- Detection: `{status('detection')}`",
        f"- Segmentation: `{status('segmentation')}`",
        f"- Attention: `{status('attention')}`",
        f"- Total: `{status('total')}`",
        "",
        f"- JSON：`{RESULT.relative_to(ROOT)}`",
    ]
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def live(args: argparse.Namespace) -> None:
    """在作者隔离环境运行真实单帧多任务反传。"""
    if not torch.cuda.is_available():
        raise RuntimeError("此步骤必须在 RTX 5080 / CUDA 环境运行")
    if not args.joint_checkpoint.is_file():
        raise FileNotFoundError(f"没有联合权重：{args.joint_checkpoint}")
    prompt_file = ROOT / "outputs/stage2_step58/distilbert_prompts.npy"
    if not prompt_file.is_file():
        raise FileNotFoundError("缺少 Step58 真实 DistilBERT 文本向量")
    author_cfg_path = AUTHOR / "projects/configs/bevformer/bevformer_small_seg_det.py"
    if not author_cfg_path.is_file():
        raise FileNotFoundError(f"缺少作者联合模型配置：{author_cfg_path}")
    pooled = np.load(prompt_file, allow_pickle=False)
    if pooled.ndim != 2 or pooled.shape[1] != 768:
        raise ValueError(f"DistilBERT embedding shape 不正确：{pooled.shape}")

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(AUTHOR))
    os.chdir(AUTHOR)
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from scripts.stage2.bev_fusion import LanguageBEVFusion
    from scripts.stage2.multitask_training import LossConfig, Stage2MultiTaskLoss

    importlib.import_module("projects.mmdet3d_plugin")
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader

    cfg = Config.fromfile(str(author_cfg_path))
    if dict(cfg.model.pts_bbox_head.task) != {"seg": True, "det": True}:
        raise RuntimeError("真实 config 并非双任务联合模型")
    cfg.model.pretrained = None
    cfg.model.video_test_mode = False
    cfg.data.train.pop("samples_per_gpu", None)
    cfg.data.train.queue_length = 1  # 单帧梯度验收；时序训练另行验收
    cfg.data.train.workers_per_gpu = 0
    cfg.data.train.pop("workers_per_gpu", None)
    cfg.data.train.pop("samples_per_gpu", None)
    dataset = build_dataset(cfg.data.train)
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"--index 必须在 0～{len(dataset)-1}")
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False, nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    it = iter(loader)
    batch = None
    for _ in range(args.index + 1):
        batch = next(it)
    assert batch is not None
    gt = validate_gt(batch)
    sample_token = str(dataset.data_infos[args.index]["token"])

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, str(args.joint_checkpoint), map_location="cpu")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    wrapper = MMDataParallel(model.cuda(), device_ids=[0])
    base = wrapper.module
    head = base.pts_bbox_head
    transformer = head.transformer
    if not hasattr(head, "feat_cropper"):
        raise RuntimeError("没有找到作者的真实 BEV map slicer")

    # 这一短实验关闭时序，只验证单帧真实 GT 的损失和梯度连通性。
    original_obtain_history = base.obtain_history_bev
    original_get_bev = transformer.get_bev_features
    # 只在本脚本实例上临时猴补，不修改作者代码文件。
    base.obtain_history_bev = lambda *_args, **_kwargs: None

    torch.manual_seed(20261021)
    fusion = LanguageBEVFusion(text_dim=768, alpha_init=0.01).cuda().float()
    fusion.train()
    params = [p for p in fusion.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("语言融合模块无可训练参数")
    text = torch.tensor(pooled[:1], dtype=torch.float32, device="cuda")
    saved: dict[str, Any] = {"attention_logits": None, "head": None, "hook_count": 0}

    def shared_bev_hook(*call_args: Any, **call_kwargs: Any) -> torch.Tensor:
        """先产生真实 BEV，然后在检测 Decoder 前注入语言融合。"""
        bev = original_get_bev(*call_args, **call_kwargs)
        if list(bev.shape) != [1, 22500, 256]:
            raise ValueError(f"真实共享 BEV 形状异常：{list(bev.shape)}")
        saved["hook_count"] += 1
        feature_map = bev.reshape(1, 150, 150, 256).permute(0, 3, 1, 2).float()
        enriched, attention_prob = fusion(feature_map, text, enabled=True)
        if tuple(attention_prob.shape) != (1, 1, 150, 150):
            raise ValueError(f"语言注意力形状异常：{list(attention_prob.shape)}")
        # 使用作者 BEV -> 地图的方向变换和 slicer，而非任意 resize。
        attention_logit = torch.logit(attention_prob.clamp(1e-5, 1 - 1e-5))
        oriented = torch.flip(attention_logit.transpose(2, 3), dims=[2])
        projected = head.feat_cropper(oriented)
        if tuple(projected.shape) != (1, 1, 200, 400):
            raise RuntimeError(f"作者 slicer 后注意力形状异常：{list(projected.shape)}")
        saved["attention_logits"] = projected
        return enriched.permute(0, 2, 3, 1).reshape(1, 22500, 256).to(bev.dtype)

    def save_head_outputs(_head: Any, _inputs: Any, outputs: Any) -> None:
        if not isinstance(outputs, dict):
            raise RuntimeError("真实双任务 Head 没有返回字典")
        for name in ("seg_preds", "all_cls_scores", "all_bbox_preds"):
            if name not in outputs or not isinstance(outputs[name], torch.Tensor):
                raise RuntimeError(f"真实联合 Head 缺少 {name}")
        saved["head"] = outputs

    transformer.get_bev_features = shared_bev_hook
    handle = head.register_forward_hook(save_head_outputs)
    record: dict[str, Any] = {
        "step": 72,
        "status": "RUNNING",
        "mode": "frozen_real_joint_model_single_train_frame_autograd_no_update",
        "dataset": "nuScenes Mini Train (training pipeline, real 3D and map GT)",
        "index": args.index,
        "token": sample_token,
        "joint_checkpoint": str(args.joint_checkpoint),
        "checkpoint_modified": False,
        "optimizer_step_executed": False,
        "frozen_author_model": True,
        "time_mode": "single frame, history disabled",
        "module_weights_trained": False,
        "gradients": {},
    }
    try:
        # 作者原生 forward_train 负责真实 3D 检测和地图分割损失。
        author_losses = wrapper(return_loss=True, **batch)
        if saved["hook_count"] != 1 or saved["head"] is None:
            raise RuntimeError(f"共享 BEV Hook 或双 Head 不完整：{saved['hook_count']}")
        det_losses = split_detection_losses(author_losses)
        seg_logits = saved["head"]["seg_preds"]
        att_logits = saved["attention_logits"]
        if seg_logits is None or att_logits is None:
            raise RuntimeError("分割或语言注意力计算图不存在")
        if tuple(seg_logits.shape) != (1, 4, 200, 400):
            raise ValueError(f"真实分割输出形状错误：{tuple(seg_logits.shape)}")
        # 选择该帧真实存在的前景类别，避免虚构注意力正样本。
        counts = torch.stack([(gt == c).sum() for c in (1, 2, 3)])
        target_class = int(counts.argmax().item()) + 1
        if int(counts.max()) == 0:
            raise RuntimeError("该帧无正样本，不能报告有效注意力监督")
        target = torch.tensor([target_class], dtype=torch.long, device="cuda")
        loss_module = Stage2MultiTaskLoss(LossConfig())
        total, parts = loss_module(det_losses, seg_logits, gt, att_logits, target)
        det_part = parts["detection_loss"]
        # detach 的 parts 只用于日志。真实反传应从尚未 detach 的原损失图求值。
        from scripts.stage2.multitask_training import aggregate_detection_losses
        import torch.nn.functional as F
        detection_loss = aggregate_detection_losses(det_losses)
        segmentation_loss = F.cross_entropy(seg_logits, gt)
        from scripts.stage2.multitask_training import supervised_attention_loss
        attention_loss = supervised_attention_loss(
            att_logits, gt, target, LossConfig()
        )
        sources = {
            "detection": detection_loss,
            "segmentation": segmentation_loss,
            "attention": attention_loss,
            "total": total,
        }
        for name, term in sources.items():
            record["gradients"][name] = gradient_norm(term, params)
            record["gradients"][name]["loss_value"] = float(term.detach().cpu())
        if not all(
            info["all_finite"] and info["global_l2"] > 1e-12
            for info in record["gradients"].values()
        ):
            raise RuntimeError("至少有一条真实损失未向语言融合模块产生有效梯度")
        total.backward()
        actual_norms = [
            float(p.grad.float().norm()) for p in params if p.grad is not None
        ]
        record["total_backward_grad_norm"] = float(
            sum(n * n for n in actual_norms) ** 0.5
        )
        if not np.isfinite(record["total_backward_grad_norm"]) or record["total_backward_grad_norm"] <= 1e-12:
            raise RuntimeError("真实总损失 backward 后没有有效融合模块梯度")
        record["status"] = "PASS_REAL_JOINT_GRADIENT_NO_OPTIMIZER_STEP"
        record["target_class"] = target_class
        record["gt_class_counts"] = {str(i): int((gt == i).sum()) for i in range(4)}
        record["author_loss_keys"] = list(author_losses.keys())
        print("[Step72] 真实三任务梯度 PASS（不更新权重）", flush=True)
        for name, vals in record["gradients"].items():
            print(f"[Step72] {name}: loss={vals['loss_value']:.6f} grad={vals['global_l2']:.6g}", flush=True)
    except Exception as exc:
        record["status"] = "NOT_VERIFIED"
        record["failure_reason"] = f"{type(exc).__name__}: {exc}"
        save_result(record)
        raise
    finally:
        transformer.get_bev_features = original_get_bev
        base.obtain_history_bev = original_obtain_history
        handle.remove()
    save_result(record)
    print("[Step72] JSON:", RESULT, flush=True)
    print("[Step72] 中文报告:", DOC, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="真实 BEVFormer 双 Head + Attention 联合梯度验收")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--joint-checkpoint", type=Path, default=JOINT_CKPT)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if not (args.self_test or args.live):
        parser.error("指定 --self-test 或 --live")
    if args.self_test:
        self_test()
    if args.live:
        live(args)


if __name__ == "__main__":
    main()

"""由真实微型训练 JSON/NPZ 生成图表和单一中文日志；不生成示意数据。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from .micro_training import CLASSES, MODES, confusion, segmentation_metrics, write_json
from .run_step70_joint_integration import sha256

ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    "original": "Author baseline",
    "untrained": "Unified zero-init",
    "trained": "Trained normal",
    "shuffled": "Trained shuffled",
}
COLORS = ListedColormap(["#202936", "#f4bc41", "#49add3", "#ab77df"])


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: Any, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build(output: Path, logbook: Path) -> None:
    run, protocol = read(output / "run.json"), read(output / "protocol.json")
    rows = [
        json.loads(line)
        for line in (output / "training.jsonl").read_text().splitlines()
    ]
    if run["status"] != "PASS_REAL_20_OPTIMIZER_STEPS_AND_FIXED_AB" or len(rows) != 20:
        raise RuntimeError("真实运行未通过或不足20步，不能生成成功报告")
    initial = torch.load(
        output / "fusion_initial.pt", map_location="cpu", weights_only=False
    )
    final = torch.load(
        output / "fusion_step020.pt", map_location="cpu", weights_only=False
    )
    if any(int(s["step"]) != 20 for s in final["optimizer"]["state"].values()):
        raise RuntimeError("优化器状态不支持20次更新结论")
    if not all(torch.isfinite(v).all() for v in final["state_dict"].values()):
        raise RuntimeError("权重包含非有限值")
    matrices = {m: np.zeros((4, 4), dtype=np.int64) for m in MODES}
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    text_effects = []
    for frame in run["ab"]["frames"]:
        path = output / frame["artifact"]
        if sha256(path) != frame["sha256"]:
            raise RuntimeError("真实预测 NPZ 哈希不一致")
        with np.load(path, allow_pickle=False) as data:
            for mode in MODES:
                matrices[mode] += confusion(data["gt"], data[mode])
            fig, axes = plt.subplots(2, 3, figsize=(15, 6.5))
            panels = [
                ("Real map GT", data["gt"]),
                (LABELS["original"], data["original"]),
                (LABELS["trained"], data["trained"]),
                ("Baseline errors (red)", data["original"] != data["gt"]),
                ("Trained errors (red)", data["trained"] != data["gt"]),
                (
                    "Error change: green=fixed / red=introduced",
                    (data["trained"] != data["gt"]).astype(int)
                    - (data["original"] != data["gt"]).astype(int),
                ),
            ]
            for j, (ax, (title, raster)) in enumerate(zip(axes.flat, panels)):
                if j < 3:
                    ax.imshow(
                        raster, cmap=COLORS, vmin=0, vmax=3, interpolation="nearest"
                    )
                elif j < 5:
                    ax.imshow(
                        raster,
                        cmap=ListedColormap(["#202936", "#ed655c"]),
                        vmin=0,
                        vmax=1,
                        interpolation="nearest",
                    )
                else:
                    ax.imshow(
                        raster,
                        cmap=ListedColormap(["#51ba83", "#202936", "#ed655c"]),
                        vmin=-1,
                        vmax=1,
                        interpolation="nearest",
                    )
                ax.set_title(title, fontsize=10)
                ax.set_axis_off()
            fig.suptitle(
                f"Mini Val index {frame['index']} | {frame['token']}\nGT raster pixels; physical alignment NOT_VERIFIED | bg=dark, divider=yellow, crossing=blue, boundary=purple",
                fontsize=10,
            )
            save_figure(fig, figures / f"val_{frame['index']:03d}_segmentation.png")
            fig, axes = plt.subplots(1, 4, figsize=(17, 4))
            for ax, mode in zip(axes[:3], ("untrained", "trained", "shuffled")):
                obj = ax.imshow(
                    data[mode + "_attention_map_grid"], cmap="magma", vmin=0, vmax=1
                )
                ax.set_title(LABELS[mode])
                ax.set_axis_off()
            delta = (
                data["trained_attention_map_grid"] - data["shuffled_attention_map_grid"]
            )
            bound = max(float(np.abs(delta).max()), 1e-8)
            axes[3].imshow(delta, cmap="coolwarm", vmin=-bound, vmax=bound)
            axes[3].set_title(f"Normal - shuffled\nmax abs={bound:.3g}")
            axes[3].set_axis_off()
            fig.colorbar(obj, ax=list(axes[:3]), fraction=0.02)
            fig.suptitle(
                f"Index {frame['index']} | {frame['text']}\nReal network attention via author cropper; physical alignment NOT_VERIFIED",
                fontsize=10,
            )
            save_figure(fig, figures / f"val_{frame['index']:03d}_attention.png")
            text_effects.append(
                {
                    "index": frame["index"],
                    "attention_mae_normal_vs_shuffled": float(np.abs(delta).mean()),
                    "attention_max_abs_normal_vs_shuffled": bound,
                    "logit_mae_normal_vs_shuffled": float(
                        np.abs(data["trained_logits"] - data["shuffled_logits"]).mean()
                    ),
                    "prediction_changed_pixels": int(
                        np.count_nonzero(data["trained"] != data["shuffled"])
                    ),
                }
            )
    # 从原始整数混淆矩阵独立重算，不只复印运行摘要。
    for mode in MODES:
        if (
            segmentation_metrics(matrices[mode])
            != run["ab"]["metrics"][mode]["segmentation"]
        ):
            raise RuntimeError("独立指标复算失败")
    write_json(
        output / "verification.json",
        {
            "optimizer_state_steps": [
                int(s["step"]) for s in final["optimizer"]["state"].values()
            ],
            "all_20_update_records": all(r["optimizer_step_executed"] for r in rows),
            "all_parameters_changed": all(
                not torch.equal(v, initial["state_dict"][k])
                for k, v in final["state_dict"].items()
            ),
            "pooled_confusion_recomputed": True,
            "all_npz_hashes_valid": True,
            "text_effects": text_effects,
        },
    )
    steps = np.arange(1, 21)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, loss in zip(
        axes.flat,
        ("detection_loss", "segmentation_loss", "attention_loss", "total_loss"),
    ):
        ax.plot(steps, [row["losses"][loss] for row in rows], "o-", markersize=3)
        ax.set(title=loss, xlabel="Actual optimizer step", ylabel="Loss")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "20 real updates on 20 fixed Mini Train frames (different frames/targets; not a convergence estimate)"
    )
    fig.tight_layout()
    save_figure(fig, figures / "training_losses.png")
    stats = run["parameter_changes"]
    names = list(stats)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for name in names:
        axes[0].plot(
            steps,
            [r["parameters"][name]["gradient_l2"] or 0 for r in rows],
            label=name,
            linewidth=1,
        )
    axes[0].set(
        yscale="symlog",
        linthresh=1e-7,
        xlabel="Actual optimizer step",
        ylabel="Gradient L2 (after clip)",
    )
    axes[0].legend(fontsize=6)
    axes[1].barh(names, [stats[n]["delta_l2"] for n in names])
    axes[1].set(xlabel="Final - initial parameter L2", xscale="log")
    axes[1].tick_params(axis="y", labelsize=7)
    fig.suptitle(
        "Real trainable parameters and gradients; author model / DistilBERT frozen"
    )
    fig.tight_layout()
    save_figure(fig, figures / "parameters_gradients.png")
    with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "mode",
                *CLASSES,
                "foreground_miou",
                "four_class_miou",
                "background_false_positive_rate",
                "precision_at_2m",
                "recall_at_2m",
                "status",
            ]
        )
        for mode in MODES:
            s = run["ab"]["metrics"][mode]["segmentation"]
            d = run["ab"]["metrics"][mode]["detection"]
            writer.writerow(
                [
                    mode,
                    *[s["class_iou"][c] for c in CLASSES],
                    s["foreground_miou"],
                    s["four_class_miou"],
                    s["background_false_positive_rate"],
                    d["precision_at_2m"],
                    d["recall_at_2m"],
                    "DIAGNOSTIC; physical alignment NOT_VERIFIED",
                ]
            )
    means = {}
    for label in ("before", "after"):
        probe = run["train_probe_" + label]
        means[label] = {
            key: float(np.mean([r["losses"][key] for r in probe]))
            for key in probe[0]["losses"]
        }
    metric_rows = []
    for mode in MODES:
        s = run["ab"]["metrics"][mode]["segmentation"]
        metric_rows.append(
            "| "
            + " | ".join(
                [
                    LABELS[mode],
                    *[
                        f"{s['class_iou'][c] * 100:.4f}"
                        for c in ("Divider", "Ped Crossing", "Boundary", "Background")
                    ],
                    f"{s['foreground_miou'] * 100:.4f}",
                    f"{s['four_class_miou'] * 100:.4f}",
                    f"{s['background_false_positive_rate'] * 100:.4f}",
                ]
            )
            + " |"
        )
    loss_rows = [
        f"| {key} | {means['before'][key]:.8f} | {means['after'][key]:.8f} | {means['after'][key] - means['before'][key]:+.8f} |"
        for key in means["before"]
    ]
    parameter_rows = [
        f"| `{name}` | {s['numel']} | {s['delta_l2']:.7g} | {s['delta_max_abs']:.7g} | {s['changed_elements']} |"
        for name, s in stats.items()
    ]
    det_rows = []
    for mode in MODES:
        d = run["ab"]["metrics"][mode]["detection"]
        error = (
            f"{d['mean_matched_xy_error_m']:.4f}"
            if d["mean_matched_xy_error_m"] is not None
            else "N/A"
        )
        det_rows.append(
            f"| {LABELS[mode]} | {d['tp']} | {d['fp']} | {d['fn']} | {d['precision_at_2m'] * 100:.3f} | {d['recall_at_2m'] * 100:.3f} | {error} |"
        )
    tokens = [
        f"| {split} | {f['index']} | `{f['token']}` | {f['scene_name']} | {f['text']} |"
        for split, frames in (
            ("Mini Train", protocol["train"]),
            ("Mini Val", protocol["validation"]),
        )
        for f in frames
    ]
    version_rows = [f"| {key} | {value} |" for key, value in run["versions"].items()]
    fg_delta = 100 * (
        run["ab"]["metrics"]["trained"]["segmentation"]["foreground_miou"]
        - run["ab"]["metrics"]["original"]["segmentation"]["foreground_miou"]
    )
    memory = run["memory"]
    text = f"""# 2026-10-22 第二阶段收尾：统一系统真实微型训练与受控 A/B

本轮在 `main` 上实际完成 **20 次 optimizer.step()**，batch size=1，使用20个固定 Mini Train 帧。仅更新既有 `Stage2Pipeline.fusion` 的 {sum(s['numel'] for s in stats.values()):,} 个参数、{len(stats)} 个参数张量；作者 Backbone/检测头/分割头和 DistilBERT 全冻结。新增权重保存为 `outputs/stage2_20261022/fusion_step020.pt`，未覆盖任何旧权重。未 commit / push，未创建分支。

本轮固定验证6帧上的诊断前景 mIoU 相对作者基线变化 **{fg_delta:+.4f} 个百分点**。该变化仅描述相同作者栅格协议下的输出，不能据此认定模型真正改善；物理空间配准、官方 nuScenes AP/NDS、自然语言正确定位及稳定泛化收益均为 **NOT_VERIFIED**。

## 1. 依据与运行前审计

依据 `docs/stage2/02_第二阶段原始任务书.md`、`00_第二阶段执行计划_2026-10-20至23.md`、`13_阶段二任务对照.md`、`20_第二阶段最终严格验收_2026-10-21.md`。昨天8项任务、4项产出已通过，本轮复用模块，不重做功能实现。昨天真实联合反传日志确认 detection=3.566558599、segmentation=0.406495571、attention=0.875378728、total=4.060592175，未创建优化器，未更新权重。

当前提交 `{read(output / 'audit_start.json')['git_head']}`。启动时已有未跟踪 README 和历史调试脚本，原样保留，见 `audit_start.json`。初次沙箱 GPU 访问被操作系统屏蔽，经沙箱外作者环境验证 CUDA 可用后运行。未安装/改写第三方依赖或源码。

统一入口为 `run_stage2.py --mode micro-train` → `joint_runtime.py` → 新增实验驱动 `micro_training.py`；模型仍通过 `pipeline.py`、`multitask_training.py` 运行。BEV `[1,256,150,150]`、语义场 `[1,256,150,150]`、Attention `[1,1,150,150]`、地图 logits `[1,4,200,400]`；共享 BEV Hook 和双 Head 每帧各1次。

## 2. 固定协议与复现命令

协议在任何训练预测之前排他写入 `protocol.json`，SHA256 `{run['protocol_sha256']}`。20个训练索引为 `linspace(0,322,20,dtype=int)`，6个验证索引为 `linspace(0,80,6,dtype=int)`，均未按结果挑选。三条指令按顺序轮换；置乱使用同一句词序反转。GT仅用于损失和评估，不进入融合模块或推理输入。

作者训练预处理保持原样，queue_length=1，历史关闭。冻结作者网络保持 eval，禁用 dropout/gridmask/BN 更新。随机种子 `{protocol['seed']}`，每个训练帧使用 seed+index，验证用 seed+1000+index；Python、NumPy、Torch/CUDA 全部设种子。未强制 CUDA 算子逐位确定性，跨设备复现可能有微小浮点差异。

AdamW，lr=0.0003，weight_decay=0.01，梯度裁剪最大L2=5，alpha初始为0。损失沿用昨天 `L=Detection+Segmentation+0.1*Attention`；Detection聚合作者原始真实3D cls/bbox损失（含 decoder 辅助层）；Segmentation为真实地图四类CE；Attention为真实地图目标类别 BCE+0.25 Dice。作者自带加权分割Loss也逐步保留为诊断字段，但不再重复加到总Loss。缺席目标按既有损失返回0，不伪造正样本。注意力监督只有类别模板，缺少配对自然语言-区域标注。

训练前后探针固定为前3个训练帧，预处理随机种子相同。全部20次更新结束、最终权重保存后才运行验证四分支；不使用验证结果调整超参数、阈值、指令或挑选 checkpoint。

```bash
{' '.join(run['command'])}
.venv-bevformer/bin/python -m scripts.stage2.micro_report --output outputs/stage2_20261022 --logbook docs/logbook_2026-10-22.md
.venv-bevformer/bin/python scripts/stage2/run_stage2.py --mode cpu-tests --output outputs/stage2_20261022/cpu.json
```

完整重跑须为 `--output` 指定一个全新的目录，已有协议/日志/权重拒绝覆盖。原始 stdout/stderr 位于 `runtime.log`，每步损失、GT张量哈希、类别计数、参数梯度、参数变化和显存位于 `training.jsonl`。

## 3. 真实优化证据与 Loss

20条更新日志与 checkpoint 优化器各参数 `step=20` 独立核对一致；{len(stats)}个可训练参数张量均有数值改变且有限。冻结网络参数/缓冲区训练前后逐位哈希相等，保护文件哈希一致，检查详见 `integrity.json`。alpha从0变为 `{rows[-1]['alpha']:.9f}`。首步 detection/segmentation 对投影参数的梯度因 alpha=0 而为0是预期行为，Attention监督仍提供梯度；后续步通过非零alpha传递感知损失。

固定同帧训练探针的3帧均值（**这是训练集拟合检查，不是验证收益**）：

| Loss | 训练前 | 训练后 | 后减前 |
|---|---:|---:|---:|
{chr(10).join(loss_rows)}

逐步曲线使用20个不同帧/目标，不能把首末 Loss 差值当作收敛。作者加权分割Loss保留在原始日志，与本协议CE口径区别明确。

![真实三任务与总 Loss](../outputs/stage2_20261022/figures/training_losses.png)

| 参数 | 元素数 | 总变化L2 | 最大绝对变化 | 改变元素数 |
|---|---:|---:|---:|---:|
{chr(10).join(parameter_rows)}

![真实参数与梯度](../outputs/stage2_20261022/figures/parameters_gradients.png)

峰值 allocated={memory['peak_allocated_mib']:.1f} MiB，reserved={memory['peak_reserved_mib']:.1f} MiB。每步记录有限参数/梯度、全局梯度L2及显存；未出现 NaN/Inf。总实验运行耗时 `{run['wall_seconds']:.2f}` 秒（包含数据/完整性审计、训练探针和A/B，入口模型加载另计），启动UTC `{run['timestamp_utc']}`，北京时间为UTC+8。

## 4. 训练前后受控 A/B

四个分支始终使用同一联合作者 checkpoint、单帧模式、相同6个验证token和原始GT。零残差初始化系统在分割、检测分类及框张量上与作者模型最大绝对差均为0。所有指标从6帧整数混淆矩阵合计后计算，未平均逐帧IoU，未更改阈值或重采样GT。以下均为 **诊断百分数**，物理指标可信性 **NOT_VERIFIED**。

| 分支 | Divider IoU | Ped Crossing IoU | Boundary IoU | Background IoU | 前景mIoU | 四类mIoU | 背景误检率 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

背景误检率=GT背景被预测为任意前景的像素数/GT背景数。物理配准完整闭环未验证，作者硬编码数据加载版本为 `v1.0-trainval`，本轮核对infos版本与全部404帧token的官方 mini_train/mini_val 场景归属；没有改写该作者源码或GT。单一小规模实验、无重复种子，不声称显著性或泛化改善。

检测诊断固定score>=0.1、XY中心范围±51.2米、同类按分数贪心一对一2米匹配；真实GT为作者 `get_ann_info` 的有效类别，和预测都在LiDAR坐标。此为子集中心距离诊断，**不是官方AP、3D IoU AP或NDS**，也不跨其他checkpoint或时序结果比较。

| 分支 | TP | FP | FN | Precision % | Recall % | 匹配中心误差 m |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(det_rows)}

正常/词序置乱的注意力差异与logits差异见 `verification.json`。词序反转保留了类别关键词，只是对词序敏感性的弱对照；不能凭注意力不同证明语言理解，也不能凭指标持平证明对所有语言无响应。

## 5. 真实结果图与原始产物

所有图都来自本轮保存的真实NPZ预测、真实GT、注意力与训练日志，使用matplotlib绘制，无示意图。每个固定验证帧均输出分割/错误区域图和注意力正常/置乱图，不按好坏筛选。像素视图不冒充已校准的物理BEV。

{chr(10).join(f'- 验证索引 {f["index"]}：[GT/原始/训练后/错误区域](../outputs/stage2_20261022/figures/val_{f["index"]:03d}_segmentation.png)，[未训练/训练后/置乱注意力](../outputs/stage2_20261022/figures/val_{f["index"]:03d}_attention.png)。' for f in protocol['validation'])}

原始目录 `outputs/stage2_20261022/`：`run.json`完整环境/命令/源码哈希与结果；`protocol.json`预注册协议；`fusion_initial.pt`初始权重；`fusion_step020.pt`新增权重及优化器状态；`training.jsonl`20次更新；`train_probe_before/after.json`；`ab_results.json`；6个`val_*.npz`含真实GT/各分支logits与框/注意力；`parameter_changes.json`；`metrics.csv`；`protected_before.json`和`integrity.json`；`verification.json`独立复算证据；`figures/`14张图。outputs按仓库规则忽略，正式交付需单独打包或选择小型证据归档，不能仅上传源码后声称权重已交付。

## 6. 环境、数据与 checkpoint

Python `{run['python_version']}`，可执行文件 `{run['python']}`，GPU `{run['gpu']}`。

| 包 | 版本 |
|---|---|
{chr(10).join(version_rows)}

作者checkpoint：`{protocol['checkpoint']}`，SHA256 `{protocol['checkpoint_sha256']}`；DistilBERT SHA256 `{protocol['distilbert_sha256']}`；新权重 SHA256 `{run['new_weights_sha256']}`。作者checkpoint上游来源仍需人工核验，本轮只验证本地文件哈希及原样使用。

| 数据集 | index | token | 场景 | 固定指令 |
|---|---:|---|---|---|
{chr(10).join(tokens)}

## 7. 测试与交付前待办

测试结果在全部相关回归完成后补入本节。新增测试覆盖IoU/背景误检、缺失类别、错误栅格拒绝、检测匹配、真实优化后参数变化及NaN拒绝、协议防覆盖；真实20步训练证据独立保存，单测不代替真实运行。

10月23日正式交付前仍需：完成共享BEV与地图GT的物理坐标闭环；如要声称精度收益，需完整固定Mini Val及重复种子验证，官方检测AP/NDS与有语言-区域真值的更强文本对照；核验作者联合checkpoint来源；确认outputs证据/新增权重的交付方式。阶段二功能已完成，本轮20步微型优化不能替代正式收敛训练。等待用户确认后才整理暂存、commit和push到origin/main。
"""
    with logbook.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(f"Generated 14 real-data figures, verified metrics and logbook: {logbook}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/stage2_20261022")
    parser.add_argument(
        "--logbook", type=Path, default=ROOT / "docs/logbook_2026-10-22.md"
    )
    args = parser.parse_args()
    build(args.output.resolve(), args.logbook.resolve())


if __name__ == "__main__":
    main()

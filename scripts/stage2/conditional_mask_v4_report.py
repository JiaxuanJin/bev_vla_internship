"""V4 中文交付、可复现入口和增量白名单；不执行 Git 写操作。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import shutil
from statistics import median
import subprocess

import numpy as np

from scripts.stage2.conditional_mask_v4_data import (
    HISTORY,
    OUT,
    ROOT,
    inventory,
    region_mask,
    sha,
    write,
)
from scripts.stage2.conditional_mask_v4_evaluate import csv_write


def pct(value: float | None) -> str:
    """比例转为百分数，不将百分点写成相对百分比。"""
    return "N/A" if value is None else f"{100 * value:.6f}"


def table(headers: list[str], rows: list[list]) -> str:
    """生成独立可读的 Markdown 数值表。"""
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(str(v) for v in r) + " |" for r in rows],
        ]
    )


def asset_audit(manifest: dict) -> dict:
    """只读核实真实图像、LiDAR、标签和来源权重，不修改外部资产。"""
    from scripts.stage2.spatial_annotations import DATA

    with (ROOT / "outputs/stage2_spatial_v3_20261026/dataset_infos.pkl").open(
        "rb"
    ) as f:
        infos = {r["token"]: r for r in pickle.load(f)["infos"]}
    files = []
    for frame in manifest["frames"]:
        info = infos[frame["sample_token"]]
        paths = [Path(c["data_path"]) for c in info["cams"].values()] + [
            Path(info["lidar_path"])
        ]
        for path in paths:
            if not path.is_absolute():
                path = ROOT / path
            files.append(
                dict(
                    sample_token=frame["sample_token"],
                    path=str(path),
                    exists=path.is_file(),
                    bytes=path.stat().st_size if path.is_file() else 0,
                )
            )
        assert sha(frame["source_file"]) == frame["source_sha256"]
    expected = {
        Path(
            "/mnt/g/nus/bevformer_small_joint_author.pth"
        ): "1345919901785b8df7ee16838d62f9f7a00444c9d8061f0dbee712d1b1b67dcf",
        ROOT
        / "models/distilbert-base-uncased/model.safetensors": "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063",
    }
    weights = []
    for path, digest in expected.items():
        observed = sha(path)
        assert digest == observed
        weights.append(dict(path=str(path), sha256=observed, unchanged=True))
    metadata = [
        DATA / "v1.0-mini/scene.json",
        DATA / "v1.0-trainval/sample_annotation.json",
    ]
    maps = sorted((DATA / "maps/expansion").glob("*.json"))
    return dict(
        n_sensor_files=len(files),
        all_sensor_files_present=all(r["exists"] for r in files),
        sensor_files=files,
        weights=weights,
        map_metadata=[
            dict(
                path=str(p),
                exists=p.is_file(),
                bytes=p.stat().st_size if p.is_file() else 0,
            )
            for p in metadata + maps
        ],
        n_frames=len(manifest["frames"]),
        gt_integrity="all48 source NPZ SHA match; all derived target label SHA verified by GPU loaders",
    )


def ambiguity_audit(frames: list[dict]) -> dict:
    """列出所有候选方位，包括空目标和几何排序差不足的拒绝项。"""
    records = []
    for frame in frames:
        with np.load(frame["source_file"]) as f:
            masks, classes, xy = f["masks"], f["classes"], f["ego_xy"]
        for cls in (1, 2):
            for fb, lr in (
                ("front", "left"),
                ("front", "right"),
                ("rear", "left"),
                ("rear", "right"),
            ):
                region = region_mask(xy, fb, lr)
                ordered = sorted(
                    (float(np.linalg.norm(xy[m & region], axis=-1).min()), k)
                    for k, (m, c) in enumerate(zip(masks, classes))
                    if c == cls and (m & region).any()
                )
                near = ordered[1][0] - ordered[0][0] if len(ordered) > 1 else None
                far = ordered[-1][0] - ordered[-2][0] if len(ordered) > 1 else None
                records.append(
                    dict(
                        split=frame["split"],
                        sample_token=frame["sample_token"],
                        target_class=cls,
                        fb=fb,
                        lr=lr,
                        n_visible=len(ordered),
                        no_target=not ordered,
                        nearest_ambiguous=near is not None and near < 1,
                        farthest_ambiguous=far is not None and far < 1,
                        near_gap_m=near,
                        far_gap_m=far,
                        all_visible_record_ids=[
                            frame["instance_ids"][i] for _, i in ordered
                        ],
                    )
                )
    return dict(
        records=records,
        n_candidate_cells=len(records),
        counts={
            s: dict(
                no_target=sum(r["no_target"] for r in records if r["split"] == s),
                nearest_ambiguous=sum(
                    r["nearest_ambiguous"] for r in records if r["split"] == s
                ),
                farthest_ambiguous=sum(
                    r["farthest_ambiguous"] for r in records if r["split"] == s
                ),
            )
            for s in {r["split"] for r in records}
        },
        policy="all visible>=1px records first; ambiguous qualifiers rejected; no prediction-driven eligibility changes",
    )


def main() -> None:
    """报告反映终点实测，发布文档仅追加当天唯一 Logbook。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-docs", action="store_true")
    args = parser.parse_args()
    report = OUT / "report"
    metrics = json.loads((report / "metrics.json").read_text())
    training = json.loads((OUT / "training_summary.json").read_text())
    protocol = json.loads((OUT / "protocol.json").read_text())
    manifest = json.loads((OUT / "labels/manifest.json").read_text())
    diagnosis = json.loads((OUT / "fusion_diagnosis/metrics.json").read_text())
    latency = json.loads((OUT / "latency/metrics.json").read_text())
    train_audit = json.loads((OUT / "train_diagnosis/metrics.json").read_text())
    final = json.loads((report / "final_metrics.json").read_text())
    final_manifest = json.loads((OUT / "final_labels/manifest.json").read_text())
    write(
        OUT / "qualification_audit.json",
        ambiguity_audit(manifest["frames"] + final_manifest["frames"]),
    )
    old_record_ids = {i for f in manifest["frames"] for i in f["instance_ids"]}
    final_record_ids = {i for f in final_manifest["frames"] for i in f["instance_ids"]}
    shared_final_ids = sorted(old_record_ids & final_record_ids)
    write(
        OUT / "final_record_overlap.json",
        dict(
            shared_real_record_ids=shared_final_ids,
            n_shared=len(shared_final_ids),
            optimizer_updates_after_final_opening=0,
        ),
    )
    assets = asset_audit(manifest)
    assert assets["all_sensor_files_present"]
    write(OUT / "assets.json", assets)
    # 完整性核验涵盖前三轮及上一轮 stage3 审计，严格禁止历史结果回写。
    before = json.loads((OUT / "frozen_history.json").read_text())["files"]
    mismatches = [
        r["path"]
        for r in before
        if not Path(r["path"]).is_file()
        or Path(r["path"]).stat().st_size != r["bytes"]
        or sha(r["path"]) != r["sha256"]
    ]
    after_paths = {r["path"] for r in inventory(HISTORY)}
    added = sorted(after_paths - {r["path"] for r in before})
    assert not mismatches and not added
    write(
        OUT / "history_integrity.json",
        dict(
            n_files=len(before), mismatches=mismatches, added=added, status="UNCHANGED"
        ),
    )
    implementation = dict(
        training_sources_unchanged=all(
            sha(ROOT / p) == digest for p, digest in protocol["source_sha256"].items()
        ),
        shuffle_clarification="Protocol prose said permutation. Implementation sampled unrelated train texts WITH replacement independently each step. All384 target mini-batches are identical across branches; text mismatches are logged. No correction/retraining or hidden result overwrite.",
        optimizer_step_audit={},
        new_head_inference_inputs=[
            "frozen author BEV",
            "frozen DistilBERT embedding",
            "calibration sampling grid",
            "analytic Ego coordinates",
        ],
        no_gt_inference=True,
        test_checks=dict(
            command=".venv-bevformer/bin/python -m pytest -q tests/stage2/test_conditional_mask_v4_contract.py",
            result="2 passed in 0.98s",
            scope="CPU geometry/overlap contracts only",
        ),
    )
    schedules = []
    for branch, checkpoint in training["branches"].items():
        logs = [
            json.loads(line)
            for line in (OUT / branch / "training.jsonl").read_text().splitlines()
        ]
        assert len(logs) == 384 and all(
            r["step"] == i + 1
            and r["optimizer_state_steps"] == [i + 1]
            and r["gradient_l2_preclip"] > 0
            and r["parameter_step_delta_l2"] > 0
            for i, r in enumerate(logs)
        )
        schedules.append([r["instruction_ids"] for r in logs])
        implementation["optimizer_step_audit"][branch] = dict(
            updates=len(logs),
            min_gradient_l2=min(r["gradient_l2_preclip"] for r in logs),
            min_step_delta_l2=min(r["parameter_step_delta_l2"] for r in logs),
            total_parameter_delta_l2=checkpoint["parameter_delta_l2"],
        )
    assert all(s == schedules[0] for s in schedules)
    write(OUT / "implementation_audit.json", implementation)
    grounding, semantic = metrics["grounding"], metrics["segmentation"]
    primary = grounding["full_instruction"]["correct"]
    cf = grounding["full_instruction"]["same_direction_counterfactual"]
    instance_table = table(
        [
            "模型",
            "实例 IoU %",
            "Precision %",
            "Recall %",
            "F1 %",
            "同方位正确/错误 IoU %",
            "差值 pp",
        ],
        [
            [
                b,
                pct(v["correct"]["iou"]),
                pct(v["correct"]["precision"]),
                pct(v["correct"]["recall"]),
                pct(v["correct"]["f1"]),
                "/".join(
                    pct(v["same_direction_counterfactual"][k]["iou"])
                    for k in ("correct", "wrong")
                ),
                pct(v["same_direction_counterfactual"]["pooled_iou_gap"]),
            ]
            for b, v in grounding.items()
            if b != "v3_attention"
        ],
    )
    semantic_table = table(
        [
            "模型",
            "目标2m邻域语义 IoU %",
            "无关区域前景 mIoU %",
            "指令象限前景 mIoU %",
            "相对Original相关/无关 pp",
        ],
        [
            [
                b,
                pct(v["target_neighborhood_semantic_iou"]),
                pct(v["irrelevant"]["foreground_miou"]),
                pct(v["quadrant"]["foreground_miou"]),
                f"{pct(v['target_neighborhood_semantic_iou'] - semantic['original']['target_neighborhood_semantic_iou'])} / {pct(v['irrelevant']['foreground_miou'] - semantic['original']['irrelevant']['foreground_miou'])}",
            ]
            for b, v in semantic.items()
        ],
    )
    detection_table = table(
        ["模型", "Relevant AP@2m_xy %", "Irrelevant AP@2m_xy %", "相对Original pp"],
        [
            [
                b,
                pct(v["relevant"]["macro_ap"]),
                pct(v["irrelevant"]["macro_ap"]),
                pct(
                    v["relevant"]["macro_ap"]
                    - metrics["detection"]["metrics"]["original"]["relevant"][
                        "macro_ap"
                    ]
                ),
            ]
            for b, v in metrics["detection"]["metrics"].items()
        ],
    )
    latency_table = table(
        [
            "模型",
            "GPU缓存均值 ms",
            "GPU六相机均值/中位/p95 ms",
            "六相机峰值 MiB",
            "CPU缓存均值 ms",
        ],
        [
            [
                b,
                f"{x['cached_gpu']['mean_ms']:.3f}",
                "/".join(
                    f"{x['full_six_camera_gpu'][k]:.3f}"
                    for k in ("mean_ms", "median_ms", "p95_ms")
                ),
                f"{x['full_six_camera_gpu']['peak_allocated_mib']:.3f}",
                f"{latency['cpu_cached_native_heads'][b]['mean_ms']:.3f}",
            ]
            for b, x in latency["gpu"].items()
        ],
    )
    scalars = [r for r in diagnosis["records"] if r["branch"] == "scalar_v3"]
    train_scalar = [r for r in scalars if r["split"] == "train"]
    body = f"""# 第三阶段 V4：条件实例输出与感知收益独立验证（2026-10-26）

已创建并真实训练独立的 Language-Conditioned Target Mask Head。执行 **{training['optimizer_updates']} 次 CUDA optimizer.step**：单个真实 nearest/farthest 对过拟合256次，四个等预算分支各384次。只更新新增322,609参数；原作者主干、双 Head、DistilBERT及历史V3全部冻结。训练是**真实缓存BEV上的GPU前反向**，不是六相机主干训练；另有真实六相机前向与缓存Head等价验证。没有 commit/push。

本轮有限 Grounding 改善，**没有 Segmentation/Detection 精度收益**。全指令实例IoU {pct(primary['iou'])}%、Precision {pct(primary['precision'])}%，仍远未能可靠输出指定实例。新Head与原感知路径解耦，原语义与检测输出按设计不变，不以新增任务指标替代师傅的感知验收。

## 数据、实例定义与隔离

种子 {protocol['seed']}；训练276条/40帧/5场景；新Head开发66条/8帧/scene-0655。同方位不同实例42条有向、21对；两侧真实地图record ID交集0。原训练集场景内按固定元数据随机划出新开发场景；没有使用 scene-0061、0553、0001 的GT或预测做本轮训练/评价。历史适配器训练过这六个场景，所以V3在本开发集不是无暴露模型；V4用原作者BEV、新Head从零训练，V3只作冻结参考，不能据此比较新旧泛化能力。

所有可见>=1像素真实record先参与排序，最近/最远按“该record可见像素至Ego的最小半径”定义，相邻序差>=1m；这不代表record最远点或全局完整地图的最近记录。选中后<8像素拒绝，绝不换用其他record；重叠目标正例优先、同类干扰排除共享目标像素。unique只有一个可见record。清单保存真实ID、标定、候选次序、目标像素及所有拒绝项。无目标指令本轮NOT_EVALUABLE；正例训练不包含空目标，不能宣称具备拒识能力。

{assets['n_sensor_files']}个六相机/LiDAR文件全部存在；48份原GT NPZ SHA核验、缓存SHA核验；对象GT关联annotation/instance token和真实标定，<=1mm对应。物理网格150×150 LiDAR +/-51.2m，200×400 Ego x[-30,30] y[-15,15]、0.15m像素中心；Ego z=0经完整LiDAR标定，bilinear align_corners=False。未修改作者cropper、GT或权重；作者cropper与物理采样约0.899625m差异仍未修正，V4采用独立物理网格，与原Head保持独立。

严格最终泛化 **BLOCKED**：没有可溯源且排除作者预训练/历史评价暴露的新场景。本轮没有再次开启旧留出，也没有将新开发写成独立最终集。所有重复计数指标同时给指令/帧/场景分母，不能当成66张独立地图。

## Grounding：独立实例任务

GT=selected真实record ∩语言方位；其他同类实例是条件Mask负例，仍是原语义GT正例。模型推理仅输入冻结原BEV、冻结文本、标定网格与解析几何坐标，不输入GT/instance ID/候选mask/GT距离，也不输入历史语言Attention，以保护无语言消融。视觉1×1降维24通道、米制位置映射、冻结3072维文本FiLM、两层局部3×3解码输出单通道。该架构没有显式候选实例级全局排序，不将局部语言调制自动解释为关系推理。

主阈值0.5训练前固定，所有分支取终点384，不做开发最优阈值或早停选择。下面66条指令/8帧/1场景按汇总TP/union计算，反事实只取42条同帧同类同方位且不同record的严格配对；未配对项的“wrong”是左右互换，另列逐指令，不能混入同实例限定词结论。

{instance_table}

完整指令相对无语言 +{pct(primary['iou'] - grounding['no_language']['correct']['iou'])} pp；相对打乱文本训练 +{pct(primary['iou'] - grounding['shuffled_text']['correct']['iou'])} pp；相对仅类别 +{pct(primary['iou'] - grounding['class_only']['correct']['iou'])} pp。同方位反事实胜 {cf['wins']}/42、平 {cf['ties']}/42；正确/错误IoU差 {pct(cf['pooled_iou_gap'])} pp。语序置乱仍保留nearest/farthest词，不能当作语义完全错误对照；其结果在JSON/CSV单独列出。

完整指令TP={primary['tp']}、GT={primary['gt']}、预测={primary['pred']}、并集={primary['union']}；全图背景FP={primary['background_fp']}/{primary['background_pixels']}（{pct(primary['background_fp_rate'])}%），同象限其他同类record FP={primary['distractor_fp']}/{primary['distractor_pixels']}（{pct(primary['distractor_fp_rate'])}%）；目标2m ROI外FP={primary['outside_roi_fp']}/{primary['outside_roi_negative']}。每个指令重复计数，没有只裁ROI隐藏背景错误。阈值0.2/0.35/0.5/0.65/0.8的实例IoU依次为 {' / '.join(pct(x['iou']) for x in grounding['full_instruction']['threshold_sensitivity'].values())}%；仅报告敏感性，没有改用更有利阈值。

冻结V3 Attention IoU={pct(grounding['v3_attention']['correct']['iou'])}%，它在这些场景已有训练暴露，且阈值任务不同；本轮新Mask {pct(primary['iou'])}%不能声称超过它。V4没有另一个Attention输出，V4 Attention IoU=N/A，禁止把实例Mask重命名为Attention。

单对训练诊断：正确/反事实IoU={pct(training['diagnostic']['pooled_correct_iou'])}/{pct(training['diagnostic']['pooled_wrong_iou'])}%，证明模型有文本条件容量，仅为记忆证据。多场景训练终点IoU={pct(train_audit['actual_bev']['iou'])}%、Precision={pct(train_audit['actual_bev']['precision'])}%、Recall={pct(train_audit['actual_bev']['recall'])}%；训练和开发都低Precision，存在整体欠拟合/稀疏目标损失校准问题，不能归咎于开发集。训练零BEV IoU={pct(train_audit['zero_bev']['iou'])}%，说明有视觉依赖；零输入分布外，不能证明真实实例关系推理已成立。384次×batch2相当于276条训练指令约2.78次随机暴露，不能推断单纯增加步数必然解决。

## Segmentation：原作者4类语义 Head

相关指标采用目标真实实例的**2m欧氏邻域∩指令象限**，GT保留邻域内所有正确同类语义；汇总target-class交并比。无关指标为象限外前景三类IoU均值；另给象限及全图混淆矩阵，避免将全局mIoU代替相关指标。nearest/farthest不改变语义GT。

{semantic_table}

V4解耦评估复用相同Original原生语义/检测预测，不是独立重跑的全数据双Head输出，也没有以Mask阈值图替代语义预测；无感知梯度连接的因果诊断和真实V4六相机计时另行保存。V4与Original目标邻域交集/并集={semantic['original']['intersection']}/{semantic['original']['union']}，逐像素输出相同；相关/无关精度变化均0.000000pp。新Mask改进没有传给原感知路径。V3相对Original变化来自历史标量融合，不能归功于新增Head。完整原始混淆矩阵、GT/预测/union和ROI外语义FP见dev_predictions/metrics.json与semantic_per_instruction.csv。

## Detection：真实对象协议，迁移诊断

64条car/pedestrian对象指令/8帧/1场景，空相关目标提示{metrics['detection']['n_empty_prompts']}条，保留FP。相同class与Ego象限分区后逐帧匹配未占用GT，LiDAR XY距离<=2m，遍历全部native top300分数，AP=各TP处precision之和/GT，再宏平均car/pedestrian。原始分数/box、真实annotation token匹配、逐类GT/TP/FP/FN和PR曲线全部保存。这是**自定义非插值AP@2m_xy，不是nuScenes官方AP/mAP**，也不是固定阈值Precision；对象指令是区域对象集合，不是地图record对应的3D物体。

{detection_table}

V4对象条件训练任务 **NOT_EVALUABLE**：本轮没有训练对象条件检测分支；上述真实AP为控制/迁移诊断。新地图Mask不进入Detection Head，相关AP增益0.000000pp，不宣称检测改善。

## Fusion/Head 根因与单变量零更新诊断

固定5个训练场景首对/8个开发帧首对，共13探针；无融合/历史标量融合/零初始化通道残差均0更新，同输入/GT/Head/alpha，无cropper修改。训练探针标量融合BEV相对L2中位={median(r['relative_bev_l2'] for r in train_scalar):.8f}；感知fusion梯度非零，Det/Seg中位范数={median(r['fusion_parameter_grad_l2']['detection'] for r in train_scalar):.8f}/{median(r['fusion_parameter_grad_l2']['segmentation'] for r in train_scalar):.8f}，但对独立Mask Head的Det/Seg梯度均严格0。

零初始化通道残差初始输出不变，up层有真实任务梯度、down/text门在初始化时梯度0，证明可训练入口而非训练效果；没有用放大alpha制造指标。新Head只收到Grounding损失，独立Head本身无法把感知损失或定位改善送入旧双Head，这是本轮无感知增益的确定架构原因。后续需在新的训练/开发协议中比较受任务损失约束的残差/可训练原Head，保持其他同类GT为正；该路线本轮**未训练、收益未证明**，不把可行梯度当成改善证据。

训练侧Loss将极稀疏正例和大面积负例分别取均值；终点真实高Recall/低Precision与这种偏向召回的目标及局部解码器缺少全局实例排序一致，但各因素的因果贡献尚未单独消融，不能断言唯一根因。保留失败结果，不追加基于已看到开发数字的试错。

## GPU、CPU、时延与计算量

硬件RTX5080 Laptop，torch CUDA可用、MMCV扩展导入且真实原生Head/六相机自定义算子执行。四路Head训练峰值约{max(v['peak_allocated_mib'] for v in training['branches'].values()):.3f}MiB；梯度诊断峰值高于仅训练，见原日志。下面同一真实帧、FP32、batch1、CUDA每次前后同步；缓存5预热/30计时，完整六相机3预热/10计时；CPU缓存2预热/10计时、4线程，不含BERT/图像主干，未运行CPU六相机。完整GPU均包含真实图像scatter、BEV主干、双Head/decode，V3/V4包含冻结BERT；不含IO/GT/固定标定生成。

{latency_table}

新模型=原模型+冻结文本+322,609参数Mask Head；完整参数分解：{latency['parameters']}。没有计算GFLOPs，标 **NOT_VERIFIED**。不按CPU均值偶然降低宣称加速；单帧、固定顺序、小样本计时受热态影响。V4完整GPU相对Original增加 {latency['gpu']['v4_independent_head']['full_six_camera_gpu']['mean_ms'] - latency['gpu']['original']['full_six_camera_gpu']['mean_ms']:.3f}ms。两个非原始架构不同，不将它们的时延差单独归因于Mask Head。

48帧训练/开发使用历史真实train-pipeline增强及float16缓存，所有A/B相同；完整六相机等价核验使用新FP32 test-pipeline，同capturedBEV的原生boxes/scores/labels最大误差0，保存新Head真实logits。这证明缓存Head数值链，不等于所有开发指令均重新进行图像主干前向或测试管线精度验证。

## 归档、SHA与复现

全部输出 `{OUT}`；dev_predictions逐指令NPZ保存正确/错误/语序乱序logits及mask、真实GT/目标/干扰物、Original/V3/V4原语义预测，CSV含所有阈值原始计数；comparison_00..08.png为每开发帧首对及最差失败样本，均标明token/指令/分支/SHA。diagnostic_pair、train_diagnosis、fusion_diagnosis、dev_objects、latency各自独立协议和数值。

训练完整终点SHA及路径见下表；所有模型保留AdamW状态与实际更新次数。

{table(['分支','更新数','checkpoint SHA256'], [[b, v['optimizer_updates'], v['sha256']] for b,v in [('diagnostic_pair',training['diagnostic']),*training['branches'].items()]])}

checkpoint路径为相应分支 `target_mask_step0256.pt` 或 `target_mask_step0384.pt`。source_snapshot保存新增代码与本地依赖，execution_manifest含逐文件SHA，protocol绑定三份训练源码未变；2537份历史文件SHA/大小/路径均未变化，作者/BERT原SHA未变化。`.gitignore`已排除outputs、NPZ、PT；白名单仅新增源码、两个契约检查、V4报告及当天Logbook追加，不默认包含历史未跟踪文件或大型结果。

实现差异公开：预注册文字将打乱训练称作permutation，实际使用固定随机种子的**每步无关文本有放回抽样**，每步used_text均保存，目标mini-batch在四分支完全一致；没有悄悄改协议或重跑覆盖。

```bash
bash {OUT}/reproduce_v4.sh /tmp/bev_v4_reproduction
```

该脚本拒绝已存在目的地，依次冻结真实标签、GPU训练、固定开发评估、因果梯度/六相机/性能、训练侧审计、报告归档；无需旧留出集。首次实际命令与GPU stdout/stderr见commands.txt、gpu_training_run.log、gpu_evaluation_run.log、gpu_diagnostics_run.log、gpu_train_audit_run.log。新增2项CPU契约检查用于米制斜坡采样/重叠实例正负例，不能代替算法指标。

本轮结论：可训练条件实例输出已实现，语言对照有小幅差距，但实例精度低且误激活严重，尚未证明稳定可靠实例识别；分割与检测收益均未实现；严格新场景泛化受数据暴露限制BLOCKED。没有通过改变ROI/阈值、修改GT或重用旧留出来制造提升。
"""
    fg = final["grounding"]
    fs = final["segmentation"]
    fd = final["detection"]["metrics"]
    final_grounding_table = table(
        [
            "模型",
            "最终实例 IoU %",
            "Precision %",
            "Recall %",
            "同方位正确/错误 IoU %",
            "差值 pp",
        ],
        [
            [
                b,
                pct(v["correct"]["iou"]),
                pct(v["correct"]["precision"]),
                pct(v["correct"]["recall"]),
                "/".join(
                    pct(v["same_direction_counterfactual"][k]["iou"])
                    for k in ("correct", "wrong")
                ),
                pct(v["same_direction_counterfactual"]["pooled_iou_gap"]),
            ]
            for b, v in fg.items()
            if b != "v3_attention"
        ],
    )
    final_perception_table = table(
        [
            "模型",
            "相关2m语义 IoU %",
            "无关语义 mIoU %",
            "Relevant AP@2m_xy %",
            "Irrelevant AP@2m_xy %",
        ],
        [
            [
                b,
                pct(v["target_neighborhood_semantic_iou"]),
                pct(v["irrelevant"]["foreground_miou"]),
                pct(fd[b]["relevant"]["macro_ap"]),
                pct(fd[b]["irrelevant"]["macro_ap"]),
            ]
            for b, v in fs.items()
        ],
    )
    fp = fg["full_instruction"]["correct"]
    fcf = fg["full_instruction"]["same_direction_counterfactual"]
    final_section = f"""## 补充：封存后一次性适配场景留出（scene-0002）

初始protocol中严格预训练未见泛化BLOCKED没有被改写。进一步核查完整officialTrain传感器资产后发现scene-0002可用，在所有权重、384预算终点与0.5阈值封存后，通过final_protocol.json追加预注册，固定时间顺序第0/13/26/39帧。既往输出搜索只命中DriveLM官方full test/test_eval/test_llama预处理，未发现适配训练/预测/选择记录；预处理暴露公开，不能证明没有未归档使用。作者预训练可能见过officialTrain，因此它是**新适配场景评价**，不是预训练未见泛化；严格泛化仍BLOCKED。

实际26条地图指令/4帧/1场景，严格同方位20条有向/10对；32条真实对象指令/4帧/1场景。所有可见record排序规则、拒绝项、2m语义ROI、非官方AP、阈值及四路权重保持固定；最终与训练/开发全部真实record ID交集={len(shared_final_ids)}。四帧均实际执行真实六相机GPU前向，原GT和作者test-pipeline语义GT逐像素相等，captured FP32 BEV与完整原生检测输出最大误差0。没有使用训练侧增强缓存替代最终test图像。评价后0次optimizer更新。

{final_grounding_table}

完整指令相对无语言 +{pct(fp['iou'] - fg['no_language']['correct']['iou'])} pp；相对打乱文本训练 +{pct(fp['iou'] - fg['shuffled_text']['correct']['iou'])} pp；相对仅类别 +{pct(fp['iou'] - fg['class_only']['correct']['iou'])} pp。同方位正确/错误差{pct(fcf['pooled_iou_gap'])}pp，胜{fcf['wins']}/20、平{fcf['ties']}/20；全图TP/GT/预测/union={fp['tp']}/{fp['gt']}/{fp['pred']}/{fp['union']}，背景FP={fp['background_fp']}/{fp['background_pixels']}，同象限其他实例FP={fp['distractor_fp']}/{fp['distractor_pixels']}。原始值见report/final_metrics.json、final_predictions/mask_per_instruction.csv。冻结V3 Attention IoU={pct(fg['v3_attention']['correct']['iou'])}%，仍高于新Head；因此不能声称本轮相对历史系统定位精度整体提升，有限正向证据是相同新Head容量的语言消融差距。

{final_perception_table}

相关/无关语义和对象AP对Original均0.000000pp；相对V3的差异来自是否保留历史融合，不归因于Mask改进。新Head对感知精度无贡献、对象条件训练NOT_EVALUABLE的结论未变。最终数据不用于重新选阈值/模型，也没有因为数字低而更换场景/筛帧。新留出仅一个适配场景、4帧，不能证明跨区域或预训练未见泛化。final_predictions/comparison_*.png采用可配对帧首条指令与最差失败样本生成真实对照，final_cache、final_labels、final_objects及gpu_final_run.log保留全部输入/输出和匹配依据。

"""
    body = body.replace("## 归档、SHA与复现", final_section + "## 归档、SHA与复现")
    body = body.replace("这两个契约检查", "这两项契约检查")
    (report / "V4_中文实验报告.md").write_text(body)
    csv_write(
        report / "final_summary.csv",
        [
            dict(
                task="grounding",
                branch=b,
                metric="target_instance_iou",
                value=v["correct"]["iou"],
                n_instructions=26,
                n_frames=4,
                n_scenes=1,
            )
            for b, v in fg.items()
        ]
        + [
            dict(
                task="segmentation",
                branch=b,
                metric="relevant_semantic_iou",
                value=v["target_neighborhood_semantic_iou"],
                n_instructions=26,
                n_frames=4,
                n_scenes=1,
            )
            for b, v in fs.items()
        ]
        + [
            dict(
                task="detection_transfer",
                branch=b,
                metric="relevant_ap_2m_xy",
                value=v["relevant"]["macro_ap"],
                n_instructions=32,
                n_frames=4,
                n_scenes=1,
            )
            for b, v in fd.items()
        ],
    )
    metrics["final_evaluation"] = final
    (report / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)
    )
    csv_write(
        report / "summary.csv",
        [
            dict(
                task="grounding",
                branch=b,
                metric="target_instance_iou",
                value=v["correct"]["iou"],
                n_instructions=66,
                n_frames=8,
                n_scenes=1,
            )
            for b, v in grounding.items()
        ]
        + [
            dict(
                task="segmentation",
                branch=b,
                metric=k,
                value=value,
                n_instructions=66,
                n_frames=8,
                n_scenes=1,
            )
            for b, v in semantic.items()
            for k, value in (
                (
                    "target_neighborhood_semantic_iou",
                    v["target_neighborhood_semantic_iou"],
                ),
                ("irrelevant_semantic_miou", v["irrelevant"]["foreground_miou"]),
            )
        ]
        + [
            dict(
                task="detection_transfer",
                branch=b,
                metric=k + "_ap_2m_xy",
                value=v[k]["macro_ap"],
                n_instructions=64,
                n_frames=8,
                n_scenes=1,
            )
            for b, v in metrics["detection"]["metrics"].items()
            for k in ("relevant", "irrelevant")
        ],
    )
    sources = sorted((ROOT / "scripts/stage2").glob("*.py"))
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir()
    for p in sources:
        shutil.copy2(p, snapshot / p.name)
    shutil.copy2(
        ROOT / "tests/stage2/test_conditional_mask_v4_contract.py",
        snapshot / "test_conditional_mask_v4_contract.py",
    )
    shutil.copy2(
        ROOT / "stage3_v4_codex_task_20261026.md",
        snapshot / "stage3_v4_codex_task_20261026.md",
    )
    commands = (
        "\n".join(
            f"/home/jjx2003520/projects/bevformer_author_clean_5080/.venv-author/bin/python -m scripts.stage2.conditional_mask_v4_{name}"
            for name in (
                "data",
                "experiment",
                "evaluate",
                "diagnostics",
                "training_audit",
                "final",
                "report --publish-docs",
            )
        )
        + "\n"
    )
    (OUT / "commands.txt").write_text(commands)
    reproduce = f"""#!/usr/bin/env bash
set -euo pipefail
cd {ROOT}
export BEV_V4_OUTPUT="${{1:?provide a NEW output directory}}"
if [ -e "$BEV_V4_OUTPUT" ]; then exit 1; fi
V4_PYTHON=/home/jjx2003520/projects/bevformer_author_clean_5080/.venv-author/bin/python
"$V4_PYTHON" -m scripts.stage2.conditional_mask_v4_data
"$V4_PYTHON" -m scripts.stage2.conditional_mask_v4_experiment > "$BEV_V4_OUTPUT/gpu_training_run.log" 2>&1
"$V4_PYTHON" -m scripts.stage2.conditional_mask_v4_evaluate > "$BEV_V4_OUTPUT/gpu_evaluation_run.log" 2>&1
"$V4_PYTHON" -m scripts.stage2.conditional_mask_v4_diagnostics > "$BEV_V4_OUTPUT/gpu_diagnostics_run.log" 2>&1
"$V4_PYTHON" -m scripts.stage2.conditional_mask_v4_training_audit > "$BEV_V4_OUTPUT/gpu_train_audit_run.log" 2>&1
"$V4_PYTHON" -m scripts.stage2.conditional_mask_v4_final > "$BEV_V4_OUTPUT/gpu_final_run.log" 2>&1
/home/jjx2003520/projects/bev_vla/.venv-bevformer/bin/python -m pytest -q tests/stage2/test_conditional_mask_v4_contract.py > "$BEV_V4_OUTPUT/cpu_contract_tests.log" 2>&1
"$V4_PYTHON" -m scripts.stage2.conditional_mask_v4_report
"""
    (OUT / "reproduce_v4.sh").write_text(reproduce)
    whitelist = [
        str(p.relative_to(ROOT))
        for p in sources
        if p.name.startswith("conditional_mask_v4")
    ]
    whitelist += [
        "tests/stage2/test_conditional_mask_v4_contract.py",
        "docs/stage3/02_语言条件目标Mask与感知收益验证_2026-10-26.md",
        "docs/logbook_2026-10-26.md (append V4 section only)",
    ]
    (OUT / "git_whitelist.txt").write_text("\n".join(whitelist) + "\n")
    if args.publish_docs:
        doc = ROOT / "docs/stage3/02_语言条件目标Mask与感知收益验证_2026-10-26.md"
        with doc.open("x") as f:
            f.write(body)
        logbook = ROOT / "docs/logbook_2026-10-26.md"
        marker = "## 第三阶段 V4：独立条件实例 Head 与三任务验收"
        assert marker not in logbook.read_text()
        with logbook.open("a") as f:
            f.write(
                f"\n\n{marker}\n\n真实RTX5080累计1792次CUDA更新（诊断256，四路各384），原BEV/双Head/BERT冻结。新训练276指令/40帧/5场景，新Head开发66指令/8帧/scene-0655，同方位反事实42条；共享record ID为0，不访问旧scene-0061/0553/0001。\n\n{instance_table}\n\n{semantic_table}\n\n{detection_table}\n\n补充封存后新适配场景scene-0002，固定4帧/26地图指令/20条实例反事实及32对象指令，评价后0更新；作者预训练可能见过，不能声称预训练未见泛化。\n\n{final_grounding_table}\n\n{final_perception_table}\n\n结论：完整指令实例IoU开发 {pct(primary['iou'])}%、新适配场景 {pct(fp['iou'])}%，低Precision且背景扩散；语言消融有有限证据，未超过V3 Attention定位IoU，原语义和真实对象AP相对Original均0pp，独立Mask与感知Head不连通。对象条件检测训练NOT_EVALUABLE；严格预训练未见泛化BLOCKED。数据、GPU更新、SHA、CPU/GPU时延和失败图详见 `{doc.relative_to(ROOT)}` 与 `{OUT.relative_to(ROOT)}`。2537历史文件完整性验证无变化，保留全部前序结果，不commit/push。\n"
            )
    write(
        OUT / "execution_manifest.json",
        dict(
            actual_cuda_optimizer_updates=1792,
            native_head_updates=0,
            backbone_updates=0,
            frozen_history_files=len(before),
            sources=[dict(path=str(p), sha256=sha(p)) for p in sources],
            checkpoints={
                b: v
                for b, v in [
                    ("diagnostic_pair", training["diagnostic"]),
                    *training["branches"].items(),
                ]
            },
            files=inventory([OUT]),
            final_generalization="BLOCKED",
            adaptation_scene_final=dict(
                status="COMPLETED",
                scene="scene-0002",
                n_frames=4,
                map_instructions=26,
                object_instructions=32,
                optimizer_updates_after_opening=0,
            ),
        ),
    )
    print("V4_DELIVERED", str(doc) if args.publish_docs else str(report), flush=True)


if __name__ == "__main__":
    main()

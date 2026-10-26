"""V5 有限探索终止、第三阶段收尾、第四阶段验收与只读 Git 交付。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.stage2.coupled_v5_data import BRANCHES, OUT, V4, inventory, sha, write
from scripts.stage2.conditional_mask_v4_evaluate import csv_write


def pct(value: float | None) -> str:
    """比例转为百分数；差值转为百分点时使用同一倍数。"""
    return "N/A" if value is None else f"{100*value:.6f}"


def table(headers: list[str], rows: list[list]) -> str:
    """生成可直接审阅的实测表格。"""
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows],
        ]
    )


def verify_history() -> dict:
    """校验所有旧结果及旧Python源码，V4不得被新实验回写。"""
    files = json.loads((OUT / "frozen_history.json").read_text())["files"]
    bad = [
        r["path"]
        for r in files
        if not Path(r["path"]).is_file()
        or Path(r["path"]).stat().st_size != r["bytes"]
        or sha(r["path"]) != r["sha256"]
    ]
    old = json.loads((OUT / "frozen_sources.json").read_text())["files"]
    source_bad = [r["path"] for r in old if sha(r["path"]) != r["sha256"]]
    for p, h in json.loads((OUT / "protocol.json").read_text())[
        "source_sha256"
    ].items():
        assert sha(ROOT / p) == h, p
    assert not bad and not source_bad
    return dict(
        n_historical_files=len(files),
        n_unchanged_old_python_sources=len(old),
        mismatches=bad,
        source_mismatches=source_bad,
        v4_frozen=True,
    )


def audit_training() -> dict:
    """从逐步AdamW状态确认更新与梯度，不能用CUDA前向替代训练。"""
    result = {}
    schedules = []
    for branch in BRANCHES:
        rows = [
            json.loads(s)
            for s in (OUT / "train" / branch / "pilot_training.jsonl")
            .read_text()
            .splitlines()
        ]
        assert len(rows) == 96
        for i, r in enumerate(rows, 1):
            assert r["step"] == i and r["optimizer_state_steps"] == [i]
            assert r["parameter_step_delta_l2"] > 0 and all(
                v > 0 for v in r["task_parameter_gradient_l2"].values()
            )
            assert 0 <= r["relative_bev_l2"] <= 0.050001
        schedules.append([r["instruction_id"] for r in rows])
        result[branch] = dict(
            updates=len(rows),
            tasks=dict(Counter(r["task"] for r in rows)),
            min_detection_gradient=min(
                r["task_parameter_gradient_l2"]["detection"] for r in rows
            ),
            min_segmentation_gradient=min(
                r["task_parameter_gradient_l2"]["segmentation"] for r in rows
            ),
            median_detection_gradient=median(
                r["task_parameter_gradient_l2"]["detection"] for r in rows
            ),
            median_segmentation_gradient=median(
                r["task_parameter_gradient_l2"]["segmentation"] for r in rows
            ),
            negative_task_gradient_cosines=sum(
                r["task_gradient_cosine"] < 0 for r in rows
            ),
            median_task_gradient_cosine=median(r["task_gradient_cosine"] for r in rows),
            min_task_gradient_cosine=min(r["task_gradient_cosine"] for r in rows),
            max_relative_feature_l2=max(r["relative_bev_l2"] for r in rows),
            weighted_object_positive_assignments=sum(
                t["relevant_positive_queries"]
                for r in rows
                for t in r["relevant_positive_assignments"]
            ),
            minimum_parameter_step_delta=min(
                r["parameter_step_delta_l2"] for r in rows
            ),
        )
    assert all(s == schedules[0] for s in schedules)
    return dict(
        total_optimizer_updates=288,
        branches=result,
        identical_target_schedule=True,
        continued_training=False,
        new_architecture_exploration_closed=True,
    )


def scope_drivelm() -> dict:
    """区分真实QA预处理、通用VQA和受约束几何指令，不伪造QA分数。"""
    official_path = ROOT / "outputs/drivelm_official_preprocess/full/summary.json"
    qa_path = ROOT / "outputs/drivelm_vqa_baseline/prepare_summary_20.json"
    official = json.loads(official_path.read_text())
    qa = json.loads(qa_path.read_text())
    qa_records = [
        json.loads(s)
        for s in (ROOT / "outputs/drivelm_vqa_baseline/qa_20.jsonl")
        .read_text()
        .splitlines()
    ]
    prediction_files = sorted(
        str(p) for p in (ROOT / "outputs/drivelm_vqa_baseline").glob("*prediction*")
    )
    return dict(
        official_preprocess=dict(
            scene_count=official["scene_count"],
            keyframe_count=official["keyframe_count"],
            qa_record_count=official["qa_record_count"],
            missing_image_paths=official["missing_image_paths"],
            source=str(official_path),
            sha256=sha(official_path),
        ),
        qa20=dict(
            selected=len(qa_records),
            distinct_scenes=qa["distinct_scenes"],
            categories=qa["qa_by_category"],
            prediction_files=prediction_files,
        ),
        grounding="nuScenes real map-vector and3DGT constrained English instructions, NOT original DriveLM QA grounding labels",
        vqa_status="NOT_EVALUABLE: V5 has no generative answer decoder and no DriveLM QA predictions/answer scores; prepared QA20 is data only",
        instruction_ap_scope="real car/pedestrian templates with traceable3Dannotation; cannot extrapolate to DriveLM open referents, forecasting/planning/behavior",
        historical799="perception-only sparse DriveLM-linked nuScenes val subset; historical aggregate mAP/mIoU are not V5 instruction metrics or DriveLM QA scores",
    )


def checkpoint_catalog(training: dict) -> list[dict]:
    """区分原始基线、冻结读出和失败候选，所有权重保留SHA。"""
    original = Path("/mnt/g/nus/bevformer_small_joint_author.pth")
    bert = ROOT / "models/distilbert-base-uncased/model.safetensors"
    assert (
        sha(original)
        == "1345919901785b8df7ee16838d62f9f7a00444c9d8061f0dbee712d1b1b67dcf"
    )
    assert (
        sha(bert) == "5e3f1108e3cb34ee048634875d8482665b65ac713291a7e32396fb18f6ff0063"
    )
    old = ROOT / "outputs/stage2_spatial_v3_20261026/train/combined/fusion_step176.pt"
    v4_record = json.loads((V4 / "training_summary.json").read_text())["branches"][
        "full_instruction"
    ]
    catalog = [
        dict(
            name="original_author",
            path=str(original),
            sha256=sha(original),
            status="FROZEN_BASELINE_RETAINED",
            optimizer_updates_this_round=0,
        ),
        dict(
            name="distilbert",
            path=str(bert),
            sha256=sha(bert),
            status="FROZEN_TEXT",
            optimizer_updates_this_round=0,
        ),
        dict(
            name="v3_historical",
            path=str(old),
            sha256=sha(old),
            status="FROZEN_HISTORICAL_COMPARATOR",
            optimizer_updates_this_round=0,
        ),
        dict(
            name="v4_mask_readout",
            path=v4_record["checkpoint"],
            sha256=v4_record["sha256"],
            status="FROZEN_GROUNDING_READOUT",
            optimizer_updates_this_round=0,
        ),
    ]
    for branch, r in training["branches"].items():
        for p in r["checkpoints"]:
            assert sha(p["checkpoint"]) == p["sha256"]
            catalog.append(
                dict(
                    name=branch,
                    path=p["checkpoint"],
                    sha256=p["sha256"],
                    optimizer_updates=p["optimizer_updates"],
                    status="RESEARCH_ONLY_NOT_ACCEPTED_FOR_PERCEPTION_RELEASE",
                    terminal=p["optimizer_updates"] == 96,
                )
            )
    return catalog


def git_candidates(incremental: list[str], small: Path, stage4: Path) -> dict:
    """聚合既有审查白名单，排除忽略资产和遗留无关文件，不暂存。"""
    candidates = set(incremental)
    prior = [
        ROOT / "results/stage2_delivery_2026-10-23/git_whitelist.txt",
        ROOT / "results/test_layout_2026-10-26/git_whitelist.txt",
        ROOT / "outputs/stage2_spatial_20261026/git_whitelist.txt",
        ROOT / "outputs/stage2_spatial_v2_20261026/git_whitelist.txt",
        ROOT / "outputs/stage2_spatial_v3_20261026/git_whitelist.txt",
        ROOT / "outputs/stage3_metric_review_20261026/git_whitelist.txt",
        V4 / "git_whitelist.txt",
    ]
    inputs = []
    for path in prior:
        if not path.is_file():
            continue
        inputs.append(dict(path=str(path), sha256=sha(path)))
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split(" (append", 1)[0].rstrip("/")
            if not line.startswith(("/", "..")) and not any(
                c in line for c in ("*", "?")
            ):
                candidates.add(line)
    candidates.update(str(p.relative_to(ROOT)) for p in small.glob("*") if p.is_file())
    generated_names = {
        str((small / "git_whitelist.txt").relative_to(ROOT)),
        str((small / "git_candidates.json").relative_to(ROOT)),
    }
    candidates.update(generated_names)
    tracked = set(
        subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    )
    selected, excluded = [], []
    for name in sorted(candidates):
        path = ROOT / name
        ignored = (
            subprocess.run(
                ["git", "check-ignore", "-q", "--", name], cwd=ROOT
            ).returncode
            == 0
        )
        if path.is_dir():
            excluded.append(
                dict(path=name, reason="directory entry requires explicit files")
            )
        elif ignored:
            excluded.append(
                dict(path=name, reason="ignored local artifact; archived separately")
            )
        elif not path.is_file() and name not in tracked and name not in generated_names:
            excluded.append(
                dict(path=name, reason="missing historical nontracked path")
            )
        else:
            selected.append(name)
    result = dict(
        paths=selected,
        excluded=excluded,
        input_whitelists=inputs,
        policy="review candidate union, not automatically approved/staged. Historical dirty changes remain for owner review; source files not named by reviewed lists stay excluded. Existing tracked test deletions included only from migration whitelist. No add/commit/push.",
    )
    (small / "git_whitelist.txt").write_text("\n".join(selected) + "\n")
    write(small / "git_candidates.json", result)
    write(stage4 / "git_candidates.json", result)
    return result


def main(publish_docs: bool) -> None:
    """封存有限预算失败结果，交付状态与精度未达标状态同时如实记录。"""
    decision = json.loads((OUT / "pilot_decision.json").read_text())
    assert (
        decision["decision"] == "STOP_ARCHITECTURE_EXPLORATION_AND_CLOSE"
        and not (OUT / "expand_training_summary.json").exists()
    )
    m = json.loads((OUT / "pilot/metrics.json").read_text())
    training = json.loads((OUT / "pilot_training_summary.json").read_text())
    perf = json.loads((OUT / "runtime/metrics.json").read_text())
    diag = json.loads((OUT / "diagnosis/metrics.json").read_text())
    integrity = verify_history()
    audit = audit_training()
    scope = scope_drivelm()
    catalog = checkpoint_catalog(training)
    report = OUT / "report"
    report.mkdir()
    write(OUT / "history_integrity.json", integrity)
    write(report / "training_audit.json", audit)
    write(
        report / "checkpoint_catalog.json",
        dict(
            models=catalog,
            release_default="original frozen native perception baseline; V5 candidates not promoted",
            best_improved_checkpoint=None,
        ),
    )
    write(report / "drivelm_scope.json", scope)
    models = m["maps"]["metrics"]
    objects = m["objects"]["metrics"]
    summary = []
    for b, v in models.items():
        summary.append(
            dict(
                branch=b,
                target_mask_iou=v["target_mask"].get("iou"),
                relevant_segmentation_iou=v["relevant_segmentation_iou"],
                irrelevant_segmentation_iou=v["irrelevant_segmentation"][
                    "foreground_miou"
                ],
                relevant_detection_ap=objects[b]["relevant"]["macro_ap"],
                irrelevant_detection_ap=objects[b]["irrelevant"]["macro_ap"],
                sameclass_distractor_semantic_iou=v["distractor_semantic_iou"],
                sameclass_distractor_mask_fp=v["target_mask"].get("distractor_fp"),
                sameclass_distractor_mask_pixels=v["target_mask"].get(
                    "distractor_pixels"
                ),
                mask_background_fp=v["target_mask"].get("background_fp"),
                semantic_changed_pixels=v["semantic_changed_pixels"],
                map_instructions=66,
                object_instructions=64,
                independent_frames=8,
                independent_scenes=1,
            )
        )
    csv_write(report / "metrics.csv", summary)
    metrics = dict(
        summary=summary,
        maps=m["maps"],
        objects=m["objects"],
        decision=decision,
        training_updates=288,
        continued_training=False,
        final_evaluation="NOT_RUN: failed preregistered development gate, no final set opened",
        strict_pretraining_unseen_generalization="BLOCKED",
        architecture_research_closed=True,
    )
    write(report / "metrics.json", metrics)
    base = next(r for r in summary if r["branch"] == "original")
    v5 = next(r for r in summary if r["branch"] == "targeted_language")
    ctl = next(r for r in summary if r["branch"] == "targeted_no_language")
    results_table = table(
        [
            "模型",
            "Target Mask IoU %",
            "Relevant Seg IoU %",
            "Irrelevant Seg IoU %",
            "Relevant AP@2m_xy %",
            "Irrelevant AP@2m_xy %",
        ],
        [
            [
                r["branch"],
                *[
                    pct(r[k])
                    for k in (
                        "target_mask_iou",
                        "relevant_segmentation_iou",
                        "irrelevant_segmentation_iou",
                        "relevant_detection_ap",
                        "irrelevant_detection_ap",
                    )
                ],
            ]
            for r in summary
        ],
    )
    timing_table = table(
        [
            "模型",
            "缓存GPU均值 ms",
            "六相机GPU均值/中位/p95 ms",
            "六相机峰值 MiB",
            "缓存CPU均值 ms",
        ],
        [
            [
                b,
                f"{v['cached_gpu']['mean_ms']:.3f}",
                " / ".join(
                    f"{v['full_six_camera_gpu'][k]:.3f}"
                    for k in ("mean_ms", "median_ms", "p95_ms")
                ),
                f"{v['full_six_camera_gpu']['peak_allocated_mib']:.3f}",
                f"{perf['cpu_cached'][b]['mean_ms']:.3f}",
            ]
            for b, v in perf["gpu"].items()
        ],
    )
    grad = audit["branches"]["targeted_language"]
    mask = v5["target_mask_iou"]
    delta = {
        k: 100 * (v5[k] - base[k])
        for k in (
            "relevant_segmentation_iou",
            "irrelevant_segmentation_iou",
            "relevant_detection_ap",
            "irrelevant_detection_ap",
        )
    }
    native_roi_fp = models["targeted_language"]["outside_roi_semantic_fp"]
    objs = m["objects"]
    ap_breakdown = {
        b: {
            region: {
                c: {k: v[k] for k in ("ap", "gt", "tp", "fp", "fn")}
                for c, v in values["classes"].items()
            }
            for region, values in regions.items()
        }
        for b, regions in objects.items()
    }
    write(report / "detection_failure_counts.json", ap_breakdown)
    checkpoint_table = table(
        ["分支", "终点更新", "checkpoint SHA256"],
        [
            [b, 96, v["selected_terminal"]["sha256"]]
            for b, v in training["branches"].items()
        ],
    )
    report_text = f"""# V5：语言条件 BEV 真实耦合与最后一轮探索结论（2026-10-26）

已完成最小Language-Conditioned BEV Residual Adapter与三路等预算真实GPU训练。**288次CUDA optimizer.step（三路各96，其中各48地图/48对象），按预注册门槛停止；不扩训、不换阈值、不再建立V6/V7。** 新增213,488参数；作者主干/原生双Head、DistilBERT、V3以及V4Mask读出全部冻结。只训练适配参数，缓存边界为真实原作者BEV，不是六相机主干训练；另有真实六相机完整V5前向和双Head数值等价验证。

核心结论：真实耦合已经成立，但**没有同时获得指令相关分割与检测收益，最终精度目标未达标**。V5相对Original相关语义IoU变化{delta['relevant_segmentation_iou']:+.6f}pp、无关语义{delta['irrelevant_segmentation_iou']:+.6f}pp、相关AP{delta['relevant_detection_ap']:+.6f}pp、无关AP{delta['irrelevant_detection_ap']:+.6f}pp。没有“最佳改进模型”可晋级，保留所有失败候选，原始感知基线保持默认。

## 数据与有限预算协议

只用V4原训练/开发划分：地图训练276指令/40帧/5场景，开发66指令/8帧/scene-0655；对象训练320指令/40帧，开发64指令/8帧/同一场景。对象相关空提示{objs['empty_prompts']}条照常计算FP。scene-0655已公开作为开发，不称独立最终集；V3历史权重见过它，V3只是冻结参考，三路V5从同一个新适配器零残差初始化。V4冻结Mask权重只用训练五场景学得。所有A/B使用完全相同真实BEV、标定、原GT、指令顺序与seed=20261031。

未读取scene-0553、0001、0002留出预测做本轮选择。最终评价 **NOT_RUN**：小试失败，不为失败模型开启新最终集。严格预训练未见泛化 **BLOCKED**：没有可核验排除作者预训练暴露的新场景；这不表示官方Train没有其他物理可用场景。

预注册：三路先各96步；仅当有语言分支在相关IoU和AP两项均胜过两路同预算控制至少0.01pp、胜过Original且两项无关指标不降低超过0.1pp，才允许三路一起到256步，总上限768。实际仅288；检测门槛失败，未用Mask提升来放行扩训。代码和门槛见protocol.json与pilot_decision.json，记录未经回写。

## 最小真实耦合结构

冻结原生BEV[1,256,150,150]经1×1卷积降至16通道，与仅由LiDAR/Ego标定生成的5通道几何相加。冻结DistilBERT3072维读出通过FiLM调制通道，经零初始化16→256投影和tanh有界残差返回原生BEV。每通道残差幅度<=原输入通道RMS×0.05；所有控制相同固定上界，没有调大alpha追逐结果。**同一个增强BEV进入真实原作者Detection与Segmentation Head**；前向接口没有GT、实例ID或候选掩码。

本轮没有新增Mask Head，也没有用地图Mask指导汽车/行人。TargetMaskIoU仅通过同一冻结V4读出在增强BEV上实测，用于诊断，不能替代原生感知指标。无语言控制仅去掉V5适配器文本，辅助V4读出仍接收正确文本，因此其Mask数值不是完整系统无语言Grounding消融。Original/V3没有这个Mask输出标N/A，V3Attention不是实例Mask。

三路：global_control=相同适配器、零文本、全局双任务损失；targeted_no_language=相同适配器、零文本、与语言分支完全相同的目标加权；targeted_language=同一容量/预算/目标权重、真实文本。由此分开比较全局适配、目标权重和文本的贡献，不能把历史模型训练预算当作等预算对照。

地图损失=全局真实DetectionLoss +全图4类CE +目标2m邻域CE +0.5象限外CE +0.1象限外Original软概率KL。完整GT保留所有正确同类实例，绝不把非目标实例涂背景。对象损失为独立车辆/行人文本的真实3DGT：作者Hungarian分配不变，相关同类/方位正query分类/回归权重3，其余正例和负query权重1；另外保持全图地图CE与无关区域KL。真实annotation/instance token中心匹配<1mm；48帧原生GT过滤与作者ObjectRangeFilter/ObjectNameFilter/limit_yaw逐项完全一致。逐层权重分配轨迹和原始GT保存，地图实例标签从未充当3D对象GT。

## 同协议原始数值

{results_table}

分割Relevant=真实record目标的2m欧氏缓冲∩指令象限内target-class交并比，GT包含邻域内所有正确同类语义；Irrelevant=象限外三前景类IoU均值。统一150×150 LiDAR物理BEV与200×400 Ego GT，通过完整标定、像素中心、bilinear align_corners=False映射；未修改作者cropper及其约0.899625m历史采样差异。

所有指标先汇总像素交并，再求比值；66指令仅来自8帧/1场景，不能当66张独立地图。V5相关语义TP/union={models['targeted_language']['intersection']}/{models['targeted_language']['union']}，Original={models['original']['intersection']}/{models['original']['union']}；V5目标ROI外语义FP={native_roi_fp}，同类其他record的2m邻域语义IoU={pct(models['targeted_language']['distractor_semantic_iou'])}%。原始GT/预测/union、区域外FP、完整混淆矩阵均在pilot/maps/metrics.json和逐指令CSV。

相关AP=自定义非插值AP@2m_xy，逐帧同class/固定Ego象限分区后按分数贪心匹配未占用真实GT，LiDAR XY距离<=2m；全native top300分数参与，AP为TP处precision之和/GT，car/pedestrian宏平均，空目标保留FP。**不是nuScenes官方AP/mAP、不是固定阈值Precision**。相关GT car253/ped26是本帧/指令匹配计数，不是279条独立物理轨迹。原始box/score/GT-token匹配/PR曲线完整保留。

## 耦合证据与失败诊断

三路每一步原生DetectionLoss和SegmentationLoss对新增参数梯度都有限非零，AdamW状态步数逐步1..96且参数位移>0；主干/原Head/BERT/V4读出更新数0。语言分支最小Det/Seg梯度={grad['min_detection_gradient']:.9f}/{grad['min_segmentation_gradient']:.9f}，任务梯度余弦{grad['negative_task_gradient_cosines']}/96次为负，最小{grad['min_task_gradient_cosine']:.6f}。训练特征扰动最大相对L2={grad['max_relative_feature_l2']:.6f}，低于固定5%边界。完整GPU日志和逐步JSONL记录所有损失、梯度、真实更新、实际匹配与特征幅度。

与V4不同，新适配器确实改变原Head logits、box和预测像素；真实六相机中间BEV注入后的双Head，与同一增强BEV缓存路径各原生logit最大差0。V4完整图像路径与Original原生logits最大差0，确认先前解耦基线；这次没有靠复用Original预测冒充V5原生评价。

但是语言分支相对目标加权无语言控制，相关IoU仅{100*(v5['relevant_segmentation_iou']-ctl['relevant_segmentation_iou']):+.6f}pp，相关AP为{100*(v5['relevant_detection_ap']-ctl['relevant_detection_ap']):+.6f}pp；正确文本全开发相关语义IoU{pct(v5['relevant_segmentation_iou'])}%反而略低于wrong分支{pct(models['targeted_language_wrong']['relevant_segmentation_iou'])}%，对象AP也未优于wrong文本。这意味着观察到的微小语义增益不能可靠归因于正确指令定位。

AP下降在三路新适配控制均出现：Original相关car TP206/FP977/FN47，V5为TP205/FP987/FN48；ped TP21保持但FP374→422。定位/置信度排序退化与冻结Head输入分布扰动相符，相关car AP从75.777437%降至74.869426%。训练对匹配正例加权的损失与自定义AP排序不是同一个目标；梯度非零不保证AP提高。52/96次任务冲突、较大的有语言特征扰动、同指令正确/错误缺少感知优势提供可验证失败证据，不能单独断言唯一根因。GT过滤与真实ID核验未发现足以使本试验无效的错误。

小试只有各48地图/48对象步，不能证明更长训练永远无效；但本项目有限预算收敛决定已经明确：**停止架构探索，不以此继续增加步数或新版本**。未进行阈值/ROI搜索、删样本、换最终场景或选择正向结果。

## CPU/GPU性能及推理

{timing_table}

同一真实六相机帧，FP32/batch1，图像[1,6,3,736,1280]；缓存5预热/30次，完整图像3预热/10次，每次CUDA前后同步；完整范围含scatter、原图像BEV主干、原生双Head/decode，以及增强分支BERT；V4/V5额外含同一冻结Mask读出。CPU缓存4线程、2预热/10次，不含图像主干/BERT；CPU完整六相机未测，GFLOPs=NOT_VERIFIED。单帧固定顺序的小样本时延，不宣称CPU微小波动是加速。

V5六相机比Original均值增加{perf['gpu']['v5_coupled']['full_six_camera_gpu']['mean_ms']-perf['gpu']['original']['full_six_camera_gpu']['mean_ms']:.3f}ms；训练峰值{max(r['peak_allocated_mib'] for r in training['branches'].values()):.3f}MiB，诊断峰值见原日志。参数分解{perf['protocol']['parameters']}。没有证明效率改善。runtime/real_six_camera_prediction.npz是真实新V5的语义logits、Mask和原生3D解码，不是历史图或合成结果。

## 权重、复现与冻结

{checkpoint_table}

三路目录均保存step001/048/096，含AdamW与协议SHA；当前研究候选未晋级为“最佳改善模型”。脚本完整再现固定budget/停训门槛，目标输出目录必须新建：

```bash
bash {OUT}/reproduce_v5.sh /tmp/bev_v5_reproduction
# 已完成训练的真实六相机单次研究推理，不训练、不使用GT：
/home/jjx2003520/projects/bevformer_author_clean_5080/.venv-author/bin/python \\
  -m scripts.stage2.coupled_v5_runtime --mode infer \\
  --output-dir /tmp/bev_v5_single_inference
```

完整训练/评价/诊断日志见gpu_pilot_training.log、gpu_pilot_evaluation.log、gpu_runtime_diagnosis.log；逐指令NPZ/CSV在pilot/maps与pilot/objects，8帧真实对照图comparison_00..07.png，全部样本保留。中文报告、结果JSON/CSV、训练审计、外部资产与所有权重SHA在report；源码快照在source_snapshot。

封存旧实验{integrity['n_historical_files']}文件和旧Python源码{integrity['n_unchanged_old_python_sources']}份SHA全部不变，V4没有追加训练或改写。当天唯一Logbook仅追加V5节；不commit/push。第三阶段从此进入性能/可评价范围/复现交付收尾，第四阶段精度验收标NOT_MET，不能以功能通过冒充项目最终精度通过。
"""
    (report / "V5_中文技术报告.md").write_text(report_text)
    stage3_close = f"""# 第三阶段收尾：性能、DriveLM范围与权重（2026-10-26）

V5是最后一轮探索。实际288次新适配器CUDA更新已封存；相关语义IoU{delta['relevant_segmentation_iou']:+.6f}pp但相关AP{delta['relevant_detection_ap']:+.6f}pp，未通过预注册门槛，停止扩大训练和新架构研究。此前所有失败/改善结果都保留，不回写V1至V4。当前没有同时改善原生指令相关Detection与Segmentation的合法增强模型，默认保留Original原生感知基线。

{results_table}

地图66指令、对象64指令都来自同8帧/scene-0655，属于开发证据；V3有该场景历史训练暴露，不能作为干净泛化基准。早期各轮使用不同场景/ROI/任务，禁止把其总体mAP/mIoU或不同分母数值拼成递增精度曲线。旧独立留出0553/0001/0002未参与本轮调参；V5失败后没有开启新最终集。

## 性能和可复现边界

{timing_table}

上述为同时间窗口、同输入、真实GPU数据，不拿旧轮时延跨时段对比。缓存/完整图像、BERT与辅助Mask开销详见V5报告。六相机V5比原模型更慢，没有速度提升证据；GFLOPs未测。所有候选模型和原基线SHA见V5report/checkpoint_catalog.json。V5候选只能标研究失败模型，不能宣传已优化模型。

## DriveLM能评价什么

真实官方预处理完成696场景/4072关键帧/29450条QA，24432条图像路径当时缺失0；本轮只读核对该原始summary，未重跑全量转换。另有跨20场景各一条QA、感知/预测/规划/行为各5条的准备数据，未发现VQA预测/评分文件。

V5仅有受约束英文地图record最近/最远/方位及真实car/pedestrian区域集合指令；真实几何标签不等于原DriveLM自然语言QA的实例对应。模型没有生成式回答decoder，未实施原QA引用对象→3Dannotation的可靠协议。因此DriveLM开放QA、预测/规划/行为准确率全部NOT_EVALUABLE，不能以29,450条预处理、799帧整体感知指标或有限模板正确性替代QA验收。历史799帧感知是稀疏独立帧、不同checkpoint/协议，不代表V5语言任务收益。

## 收尾交付

新源码coupled_v5_*，原作者工程/数据/第三方权重不修改；结果、完整日志、checkpoint、SHA、原生推理输出、源码快照和复现脚本已归档。新适配器213,488参数、V4读出322,609参数都单独计数，不声称训练主干。执行reproduce_v5.sh使用新目录；实际单次六相机推理入口coupled_v5_runtime --mode infer，不含训练或GT推理。

下一步是第四阶段综合验收与Git审阅交付，精度未达标事实保留；除严重无效错误，不重新开始架构探索。详见 ../stage4/01_综合验收与Git交付_2026-10-26.md。
"""
    acceptance = [
        dict(
            item="真实语言条件BEV进入原生双Head",
            status="VERIFIED",
            evidence="288真实CUDA更新、每步Det/Seg非零梯度、真实六相机完整链",
        ),
        dict(
            item="真实对象语言监督与GT对应",
            status="VERIFIED",
            evidence="car/pedestrian真实annotation/instance token<1mm；48帧作者GT过滤完全一致",
        ),
        dict(
            item="相关分割精度稳定受正确指令提升",
            status="NOT_MET",
            evidence="IoU微增但wrong指令略优，未胜历史V3，开发单场景",
        ),
        dict(
            item="相关Detection AP提升且无关区域受保护",
            status="NOT_MET",
            evidence=f"相关{delta['relevant_detection_ap']:+.6f}pp，无关{delta['irrelevant_detection_ap']:+.6f}pp，未通过门槛",
        ),
        dict(
            item="最终独立泛化",
            status="BLOCKED",
            evidence="无预训练未见来源核验；V5失败后未开启新最终集",
        ),
        dict(
            item="DriveLM原QA问答/预测/规划/行为",
            status="NOT_EVALUABLE",
            evidence="无回答decoder、无原QA对象到3DGT对应/预测评分",
        ),
        dict(
            item="真实推理和性能",
            status="VERIFIED_WITH_LIMITS",
            evidence="CPU缓存/GPU完整图像实际计时；无加速证据，GFLOPs未测",
        ),
        dict(
            item="前序资产保护",
            status="VERIFIED",
            evidence=f"{integrity['n_historical_files']}旧结果、{integrity['n_unchanged_old_python_sources']}源码不变",
        ),
        dict(
            item="有限预算终止与本地交付",
            status="PREPARED",
            evidence="停止288更新，源码/结果/权重/复现和Git候选白名单；未暂存/提交/推送",
        ),
    ]
    write(
        report / "acceptance.json",
        dict(
            items=acceptance,
            overall_algorithm_accuracy="NOT_MET",
            engineering_handoff="PREPARED_LOCAL",
            no_more_architecture_rounds=True,
        ),
    )
    stage4 = Path(
        os.environ.get(
            "BEV_V5_DELIVERY_OUTPUT", ROOT / "outputs/stage4_delivery_20261026"
        )
    ).resolve()
    stage4.mkdir(parents=True, exist_ok=False)
    small = ROOT / "results/stage4_delivery_2026-10-26"
    if not publish_docs:
        small = stage4 / "git_evidence"
    small.mkdir(parents=True, exist_ok=False)
    doc3 = ROOT / "docs/stage3/03_V5语言条件耦合与最终探索结论_2026-10-26.md"
    docclose = ROOT / "docs/stage3/04_第三阶段收尾与DriveLM可评价范围_2026-10-26.md"
    doc4 = ROOT / "docs/stage4/01_综合验收与Git交付_2026-10-26.md"
    stage4_text = f"""# 第四阶段综合验收与Git交付（2026-10-26）

第三阶段探索已按预算收敛，V5仅288次CUDA更新，没有继续V6/V7。**工程结果可交付审阅，算法最终精度验收NOT_MET**。不因原生Head梯度贯通或功能检查通过而声称感知目标完成。

{table(['验收项','状态','真实证据/限制'],[[r['item'],r['status'],r['evidence']] for r in acceptance])}

## 同协议算法对照

{results_table}

分割66指令/8帧/1开发场景，对象64指令/相同8帧。AP@2m_xy为自定义非官方非插值AP，真实对象独立监督；TargetMask来自冻结V4读出，不替代原语义或检测指标。没有同时改善相关IoU/AP的新最佳模型可交付上线；Original保留为原生默认基线，V5仅research-only。所有失败候选权重及原始数据均保留，绝不删除不利样本。

## 推理与复现

{timing_table}

单帧真实六相机FP32、同步预热与逐次计时，包含增强分支BERT；V4/V5均含辅助Mask。缓存/完整范围不同须分别看。不是部署吞吐/完整CPU图像benchmark，GFLOPs未验证。直接命令：

```bash
bash {OUT}/reproduce_v5.sh /tmp/bev_v5_reproduction
/home/jjx2003520/projects/bevformer_author_clean_5080/.venv-author/bin/python \\
  -m scripts.stage2.coupled_v5_runtime --mode infer --output-dir /tmp/bev_v5_single_inference
```

依赖原作者RTX5080环境、外部author/BERT权重与真实nuScenes文件、已SHA绑定的V4标签/原始BEV缓存。精确外部资产索引在本地artifact清单；包不含第三方大权重和数据集，不能在缺失这些依赖时声称可完全离线重训。

## DriveLM范围

官方29450条QA预处理不等于QA精度。模型仅评估受约束的真实地图/车辆/行人几何指令；无生成式回答Head，也未建立原QA指代到3DGT协议，所以开放QA、预测、规划、行为评分NOT_EVALUABLE。历史799帧整体感知不是本轮指令相关验收数据。

## Git与本地资产交付

小型可入库证据：results/stage4_delivery_2026-10-26/；其中git_whitelist.txt是前序审查白名单与本轮增量的候选聚合，仅供审阅，不代表替用户批准所有历史dirty差异。已按.gitignore排除outputs、NPZ、checkpoint和数据集；保留迁移明确列出的旧测试删除项，不删除未列入的历史测试或遗留文件。

本地完整证据和研究权重归档：{stage4}，包含独立artifact压缩包、包SHA256、文件清单及外部依赖说明。源码/报告/训练日志/原始预测/9个V5checkpoint均归档，不把大型资产默认入Git。与此前阶段的旧输出逐SHA验证保持原样。原始数据和作者权重不修改。

Git HEAD和index只读核验，未git add/commit/push，远端未更新。本轮不自动生成或执行推送；先按候选白名单逐项审阅，特别是会话开始前已有修改。中文Logbook仅追加当天同一文件。

截至本次交付：第三阶段有限探索、性能/范围/复现收尾已完成；第四阶段综合验收已记录真实不达标项并准备本地Git交付材料。算法精度验收不能标通过；除严重使实验无效的错误外，不再次启动架构研究。
"""
    for path, body in (
        (report / "stage3_closure.md", stage3_close),
        (stage4 / "acceptance_report.md", stage4_text),
    ):
        path.write_text(body)
    for src in (
        report / "metrics.csv",
        report / "acceptance.json",
        report / "checkpoint_catalog.json",
        report / "drivelm_scope.json",
        report / "training_audit.json",
        OUT / "history_integrity.json",
        OUT / "pilot_decision.json",
    ):
        shutil.copy2(src, small / src.name)
    write(
        small / "metrics_summary.json",
        dict(
            rows=summary,
            n_map_instructions=66,
            n_object_instructions=64,
            n_frames=8,
            n_scenes=1,
            optimizer_updates=288,
            decision=decision,
            external_raw_outputs=str(OUT),
        ),
    )
    source_files = sorted((ROOT / "scripts/stage2").glob("coupled_v5*.py"))
    incremental = [str(p.relative_to(ROOT)) for p in source_files] + [
        str(p.relative_to(ROOT))
        for p in (doc3, docclose, doc4, ROOT / "docs/logbook_2026-10-26.md")
    ]
    (OUT / "git_whitelist_incremental.txt").write_text("\n".join(incremental) + "\n")
    reproduce = f"""#!/usr/bin/env bash
set -euo pipefail
cd {ROOT}
export BEV_V5_OUTPUT="${{1:?provide a NEW output directory}}"
if [ -e "$BEV_V5_OUTPUT" ]; then exit 1; fi
export BEV_V5_DELIVERY_OUTPUT="$BEV_V5_OUTPUT/delivery"
V5_PYTHON=/home/jjx2003520/projects/bevformer_author_clean_5080/.venv-author/bin/python
"$V5_PYTHON" -m scripts.stage2.coupled_v5_data
"$V5_PYTHON" -m scripts.stage2.coupled_v5_train --phase pilot > "$BEV_V5_OUTPUT/gpu_pilot_training.log" 2>&1
"$V5_PYTHON" -m scripts.stage2.coupled_v5_evaluate --phase pilot > "$BEV_V5_OUTPUT/gpu_pilot_evaluation.log" 2>&1
# Original V5 failed its fixed gate; reproduction verifies that result without new tuning.
"$V5_PYTHON" -c 'import json;from scripts.stage2.coupled_v5_data import OUT;assert not json.loads((OUT/"pilot_decision.json").read_text())["expand"]'
"$V5_PYTHON" -m scripts.stage2.coupled_v5_runtime > "$BEV_V5_OUTPUT/gpu_runtime_diagnosis.log" 2>&1
"$V5_PYTHON" -m scripts.stage2.coupled_v5_delivery
"""
    (OUT / "reproduce_v5.sh").write_text(reproduce)
    commands = (
        "\n".join(
            [
                ".venv-bevformer/bin/python -m scripts.stage2.coupled_v5_data",
                *[
                    "/home/jjx2003520/projects/bevformer_author_clean_5080/.venv-author/bin/python -m scripts.stage2."
                    + name
                    for name in (
                        "coupled_v5_train --phase pilot",
                        "coupled_v5_evaluate --phase pilot",
                        "coupled_v5_runtime",
                    )
                ],
                ".venv-bevformer/bin/python -m scripts.stage2.coupled_v5_delivery --publish-docs",
            ]
        )
        + "\n"
    )
    (OUT / "commands.txt").write_text(commands)
    (small / "README.md").write_text(
        "# 第四阶段本地交付证据\n\n算法精度NOT_MET，工程材料已准备；288次CUDA更新后停止。metrics_summary.json为同协议指令相关数值，checkpoint_catalog.json列SHA与失败模型状态，acceptance.json保留不达标/不可评价项。原始NPZ、日志和模型在本地outputs独立包，不默认推入Git；git_whitelist.txt仅候选审阅清单，未暂存/commit/push。\n"
    )
    if publish_docs:
        doc4.parent.mkdir(exist_ok=True)
        for path, body in (
            (doc3, report_text),
            (docclose, stage3_close),
            (doc4, stage4_text),
        ):
            with path.open("x") as f:
                f.write(body)
        logbook = ROOT / "docs/logbook_2026-10-26.md"
        prefix = json.loads((OUT / "logbook_prefix.json").read_text())
        raw = logbook.read_bytes()
        assert (
            len(raw) == prefix["bytes"]
            and hashlib.sha256(raw).hexdigest() == prefix["sha256"]
        )
        marker = "## V5最后一轮探索与第三/四阶段收尾"
        assert marker not in raw.decode()
        with logbook.open("a") as f:
            f.write(
                f"\n\n{marker}\n\n三路各96步，共288次真实CUDA optimizer更新（各48地图/48对象），新增213,488参数进入真实原生双Head；每步Det/Seg梯度非零，作者主干/Head/BERT/V4Mask保持冻结。\n\n{results_table}\n\n开发地图66指令、对象64指令/同8帧/scene-0655。V5相关IoU {delta['relevant_segmentation_iou']:+.6f}pp，相关AP {delta['relevant_detection_ap']:+.6f}pp，无关AP {delta['irrelevant_detection_ap']:+.6f}pp，固定门槛失败；停止扩训，不再V6/V7。正确文本没有稳定原生感知优势。旧留出0553/0001/0002未作本轮选择，失败后未开启新最终集；严格预训练未见泛化BLOCKED。\n\n真实六相机V5比Original增加{perf['gpu']['v5_coupled']['full_six_camera_gpu']['mean_ms']-perf['gpu']['original']['full_six_camera_gpu']['mean_ms']:.3f}ms；DriveLM原QA/VQA未实现，29450条QA仅预处理。第三阶段收尾、第四阶段精度NOT_MET及Git候选材料已归档；旧{integrity['n_historical_files']}结果文件和{integrity['n_unchanged_old_python_sources']}份Python源码不变，不暂存/commit/push。详见 {doc3.relative_to(ROOT)}、{docclose.relative_to(ROOT)}、{doc4.relative_to(ROOT)}。\n"
            )
    # 复现模式只生成本地报告，不能写现有项目Logbook/文档。
    if not publish_docs:
        incremental = [str(p.relative_to(ROOT)) for p in source_files]
    if publish_docs:
        git = git_candidates(incremental, small, stage4)
    else:
        git = dict(
            paths=[],
            excluded=[],
            policy="local reproduction; no new repository Git publication",
        )
        write(stage4 / "git_candidates.json", git)
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir()
    for p in sorted((ROOT / "scripts/stage2").glob("*.py")):
        shutil.copy2(p, snapshot / p.name)
    (OUT / "git_status_final.txt").write_text(
        subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    )
    index = ROOT / ".git/index"
    git_state = dict(
        head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        index_sha256=sha(index) if index.is_file() else None,
        staged_paths=subprocess.check_output(
            ["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True
        ).splitlines(),
        performed_git_mutations=False,
        remote_updated=False,
        candidate_paths=len(git["paths"]),
    )
    write(stage4 / "git_state.json", git_state)
    write(
        stage4 / "external_assets.json",
        dict(
            weights=catalog[:4],
            train_dev_manifest=str(V4 / "labels/manifest.json"),
            train_dev_manifest_sha256=sha(V4 / "labels/manifest.json"),
            datasets="real nuScenes images/LiDAR/map metadata and author mini dataset are external; immutable48-frame source/cache hashes retained in V4 manifest",
            author_python="/home/jjx2003520/projects/bevformer_author_clean_5080/.venv-author/bin/python",
        ),
    )
    write(
        OUT / "execution_manifest.json",
        dict(
            optimizer_updates=288,
            phase="pilot_only",
            architecture_research_closed=True,
            continued_training=False,
            source_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_files},
            integrity=integrity,
            files=inventory([OUT]),
        ),
    )
    # 本地大文件独立打包；包在新stage4目录，不含外部GT/权重或旧实验目录。
    archive = stage4 / "stage3_v5_evidence_and_weights.tar.gz"
    members = [p for p in OUT.rglob("*") if p.is_file() and stage4 not in p.parents]
    if publish_docs:
        members.extend([doc3, docclose, doc4])
    with tarfile.open(archive, "x:gz", compresslevel=1) as tar:
        for p in sorted(members):
            tar.add(
                p,
                arcname=(
                    str(p.relative_to(ROOT))
                    if p.is_relative_to(ROOT)
                    else str(p.relative_to(OUT))
                ),
                recursive=False,
            )
    write(
        stage4 / "artifact_manifest.json",
        dict(
            archive=str(archive),
            sha256=sha(archive),
            bytes=archive.stat().st_size,
            members=len(members),
            scope="V5 raw predictions/plots/logs/all9newcheckpoints/source snapshot/reproduction/docs; external author/BERT/data and earlier results remain separate",
            overall_accuracy="NOT_MET",
            git_handoff="PREPARED_NOT_STAGED",
        ),
    )
    print(
        "V5_AND_STAGE4_DELIVERED",
        dict(
            optimizer_updates=288,
            no_more_training=True,
            accuracy="NOT_MET",
            archive=str(archive),
            sha256=sha(archive),
            git_candidates=len(git["paths"]),
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-docs", action="store_true")
    main(parser.parse_args().publish_docs)

"""只读整理固定诊断结果；不导入模型、不读取 GT 或 Checkpoint。"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments/lg_bev_stage3_exploration_20261101"
MODELS = {"a0": "A0初始化候选", "a_only": "A9562", "enhanced": "StageB256"}
TASKS = {"map": "地图", "object": "对象"}
CLASSES = {
    ("map", 1): "lane",
    ("map", 2): "crosswalk",
    ("object", 0): "car",
    ("object", 8): "pedestrian",
}


def read(path):
    """仅读取明确指定的现有 JSON。"""
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(rows, key):
    """同协议已存标量的算术平均；无有效分母时不补零。"""
    values = [r[key] for r in rows if r[key] is not None]
    return fmean(values) if values else "NOT_VERIFIED"


def write_csv(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/stage4_delivery_20261103"
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("输出必须在本仓库内的新目录")
    if out.exists():
        raise FileExistsError("不覆盖既有结果；复核时指定新的输出目录")
    audit_path = EXP / "reports/delivery_gap_audit_20261102/核查记录.json"
    audit = read(audit_path)
    metrics_path = Path(audit["metrics_json"])
    if not metrics_path.parent.name.endswith("_diagnostic"):
        raise ValueError("只允许读取已完成的固定诊断目录")
    if sha(metrics_path) != audit["metrics_json_sha256"]:
        raise ValueError("既有指标 SHA 与 11/02 身份核查不符")
    metrics = read(metrics_path)
    subset_path = (
        ROOT
        / "experiments/lg_bev_stage3_feasibility_audit_20261101/proposed_subset_metadata.json"
    )
    subset = read(subset_path)
    protocol_path = EXP / "reports/evaluation_20261102/protocol.json"
    protocol = read(protocol_path)
    prior = EXP / "reports/stage_a_acceptance_20261102"
    source_paths = [audit_path, metrics_path, subset_path, protocol_path]
    source_paths += sorted(prior.glob("*.csv")) + [prior / "核心验收汇总.json"]
    input_before = {str(p): sha(p) for p in source_paths}
    scenes = {r["scene_token"]: r for r in subset["训练内诊断候选"]}
    scene_ids = sorted({k.split("/")[0] for k in metrics["per_scene"]})
    if len(scene_ids) != 4 or not set(scene_ids) <= scenes.keys():
        raise ValueError("固定四场景身份不符")
    scene_attention, scene_perception, query_rows = [], [], []
    instruction_groups = defaultdict(list)
    tokens_by_scene = {}
    for scene_id in scene_ids:
        info = scenes[scene_id]
        saved = metrics["per_scene"][scene_id + "/a_only/correct"]
        tokens = {
            r["query_id"].split("/")[0]
            for task in ("map", "object")
            for r in saved["Attention"][task]["rows"]
        }
        if len(tokens) != 2 or not tokens <= set(protocol["diagnostic_tokens"]):
            raise ValueError("每场景应有固定两帧")
        tokens_by_scene[scene_id] = sorted(tokens)
        base = {
            "场景": info["scene_name"],
            "场景token": scene_id,
            "地点": info["location"],
            "日志token": info["log_token"],
            "场景帧数": len(tokens),
            "帧token": "|".join(sorted(tokens)),
        }
        for group, model in MODELS.items():
            key = scene_id + "/" + group + "/correct"
            for task, task_name in TASKS.items():
                a = metrics["per_scene"][key]["Attention"][task]
                scene_attention.append(
                    {
                        **base,
                        "模型": model,
                        "任务": task_name,
                        "指令数": a["queries"],
                        "正目标数": a["positive_queries"],
                        "空目标数": a["empty_queries"],
                        "有效正目标配对数": sum(
                            r["positive"] and r["counterfactual_valid"]
                            for r in a["rows"]
                        ),
                        "正目标macro_IoU": (
                            a["positive_macro_IoU"]
                            if a["positive_macro_IoU"] is not None
                            else "NOT_VERIFIED"
                        ),
                        "正目标pooled_IoU": (
                            a["positive_pooled_IoU"]
                            if a["positive_pooled_IoU"] is not None
                            else "NOT_VERIFIED"
                        ),
                        "空目标激活率": (
                            a["empty_active_fraction"]
                            if a["empty_active_fraction"] is not None
                            else "NOT_VERIFIED"
                        ),
                        "来源字段": "per_scene/" + key + "/Attention/" + task,
                    }
                )
                for row in a["rows"]:
                    query_id = row["query_id"]
                    suffix = query_id.split("/")[1].split("_")
                    cls = int(suffix[2])
                    qualifier = suffix[-1]
                    item = {
                        **base,
                        "模型": model,
                        "任务": task_name,
                        "类别": CLASSES[task, cls],
                        "限定词": qualifier,
                        "方位": "_".join(suffix[3:-1]),
                        "指令": query_id,
                        "样本token": query_id.split("/")[0],
                        "正目标": row["positive"],
                        "有效配对": row["counterfactual_valid"],
                        "正确IoU": row["IoU"],
                        "错误IoU": row["wrong_IoU"],
                        "目标Logit差": row["logit_gap"],
                        "目标概率差": row["probability_gap"],
                        "空目标激活率": row["empty_active_fraction"],
                        "背景激活像素": row["background_FP"],
                        "同类干扰激活像素": row["distractor_FP"],
                        "峰值在目标内": row["peak_in_target"],
                        "目标概率质量占比": row["target_probability_mass_fraction"],
                        "来源字段": "per_scene/" + key + "/Attention/" + task + "/rows",
                    }
                    query_rows.append(item)
                    instruction_groups[group, task, cls, qualifier].append(row)
        for group, model in (
            ("baseline", "Baseline"),
            ("a_only", "A9562"),
            ("enhanced", "StageB256"),
        ):
            key = scene_id + "/" + group + "/correct"
            v = metrics["per_scene"][key]
            item = {
                **base,
                "模型": model,
                "总指令数": 4,
                "对象指令数": 2,
                "地图指令数": 2,
            }
            for field, chinese, metric in [
                ("Relevant_Detection_AP", "相关检测AP@2m_xy", "AP@2m_xy"),
                ("Irrelevant_Detection_AP", "无关检测AP@2m_xy", "AP@2m_xy"),
                ("Relevant_Segmentation_IoU", "相关分割IoU", "mIoU"),
                ("Irrelevant_Segmentation_IoU", "无关分割IoU", "mIoU"),
            ]:
                item[chinese] = (
                    v[field][metric] if v[field][metric] is not None else "NOT_VERIFIED"
                )
                item[chinese + "_原状态"] = v[field]["status"]
                if metric == "AP@2m_xy":
                    for cls in ("car", "pedestrian"):
                        item[chinese + "_" + cls + "_GT"] = v[field]["classes"][cls][
                            "2.0"
                        ]["GT"]
                else:
                    for cls in ("1", "2"):
                        item[chinese + "_类别" + cls + "_GT像素"] = v[field]["classes"][
                            cls
                        ]["GT_positive_pixels"]
            item["来源字段"] = "per_scene/" + key
            scene_perception.append(item)
    grouped = []
    for (group, task, cls, qualifier), rows in sorted(instruction_groups.items()):
        positive = [r for r in rows if r["positive"]]
        paired = [r for r in positive if r["counterfactual_valid"]]
        grouped.append(
            {
                "模型": MODELS[group],
                "任务": TASKS[task],
                "类别": CLASSES[task, cls],
                "限定词": qualifier,
                "指令数": len(rows),
                "帧数": len({r["query_id"].split("/")[0] for r in rows}),
                "正目标数": len(positive),
                "空目标数": len(rows) - len(positive),
                "有效正目标配对数": len(paired),
                "正目标macro_IoU": mean(positive, "IoU"),
                "有效配对正确IoU": mean(paired, "IoU"),
                "有效配对错误IoU": mean(paired, "wrong_IoU"),
                "有效配对目标Logit差": mean(paired, "logit_gap"),
                "空目标激活率": mean(rows, "empty_active_fraction"),
                "指令列表": "|".join(r["query_id"] for r in rows),
                "来源字段": "groups/" + group + "/correct/Attention/" + task + "/rows",
            }
        )
    out.mkdir(parents=True)
    tables = {
        "分场景Attention.csv": scene_attention,
        "分场景感知.csv": scene_perception,
        "逐指令与背景激活.csv": query_rows,
        "分指令类型汇总.csv": grouped,
    }
    for filename, rows in tables.items():
        write_csv(out / filename, rows)
    record = {
        "日期": "2026-11-03",
        "数据集": "固定8帧工程诊断，非DriveLM正式测试，非320帧Development",
        "帧数": 8,
        "场景数": 4,
        "日志数": 4,
        "指令数": 16,
        "地图指令数": 8,
        "地图正目标数": 4,
        "地图空目标数": 4,
        "对象指令数": 8,
        "对象正目标数": 5,
        "对象空目标数": 3,
        "场景及固定帧": tokens_by_scene,
        "来源": [
            {
                "路径": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                "SHA256": sha(p),
            }
            for p in source_paths
        ],
        "汇总方法": "分场景指标逐字段摘录既有per_scene；逐指令背景项逐字段摘录；仅缺少的任务×类别×限定词分组对同协议已存IoU/Logit标量求均值，不生成预测或重计AP。",
        "分母规则": "正目标IoU排除空目标；正确/错误仅在同一非空且有效配对集合；空目标激活单列。不可验证项为NOT_VERIFIED，保留原NOT_EVALUABLE，不填零。",
        "背景计数规则": "background_FP/distractor_FP是Attention阈值激活像素，不是检测框FP；不同网格/指令不作总像素平均。",
        "独立性限制": "nuScenes原train来源的预登记诊断场景；不是原任务DriveLM-v1正式测试；不证明Stage A或作者预训练未见，单seed和模板指令不支持稳定泛化。",
        "源结果保持": all(sha(p) == input_before[str(p)] for p in source_paths),
        "执行边界": {
            "新推理": False,
            "训练": False,
            "GT及Checkpoint读取": False,
            "ColdFinal读取": False,
            "回归测试执行": False,
            "模型模块导入": False,
        },
        "输出": [
            {"文件": name, "行数": len(rows), "SHA256": sha(out / name)}
            for name, rows in tables.items()
        ],
    }
    # 来源摘要只绑定输入，不把文件路径或目录名称当作新的实验授权。
    (out / "来源与统计口径.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"输出目录": str(out), "表格行数": {k: len(v) for k, v in tables.items()}},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

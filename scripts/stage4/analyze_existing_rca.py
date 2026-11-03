"""只读既有八帧预测和 StageB256 标量日志，整理 RCA 缺少的配对证据。

不导入模型、不推理、不调用评分器、不读取 Checkpoint 或 Cold Final。
原始预测只接受核心验收清单内且 SHA256 一致的本地文件。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/stage4_rca_20261103"
REPORTS = ROOT / "experiments/lg_bev_stage3_exploration_20261101/reports"
DIAGNOSTIC = Path(
    "/mnt/g/BEV_VLA_Datasets/lg_bev_stage3_exploration_20261101/"
    "lg_bev_explore_main_20261101_unified_eval_20261102_diagnostic"
)
TRAIN = DIAGNOSTIC.parent / "lg_bev_explore_main_20261101"
INPUTS: dict[str, str] = {}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    INPUTS[str(path)] = digest(data)
    return data


def read_json(path: Path):
    return json.loads(read_bytes(path))


def describe(values):
    values = list(values)
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def write_json(name, value):
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    OUT = parser.parse_args().output_dir.resolve()
    if OUT.exists():
        raise FileExistsError("不覆盖既有 RCA 输出")
    core = read_json(REPORTS / "stage_a_acceptance_20261102/核心验收汇总.json")
    existing = read_json(DIAGNOSTIC / "metrics.json")
    assert INPUTS[str(DIAGNOSTIC / "metrics.json")] == (
        "515de677142087dbf7ffeb2b05a0da883754714d71f295de46777146b70ec64c"
    )
    gate_path = Path(core["historical_A128"]["path"])
    gate = read_json(gate_path)
    assert INPUTS[str(gate_path)] == core["historical_A128"]["sha256"]
    assert gate == core["historical_A128"]["result"]
    attention_rows = {
        (g, task, row["query_id"]): row
        for g in ("a_only", "enhanced")
        for task in ("map", "object")
        for row in existing["groups"][g + "/correct"]["Attention"][task]["rows"]
    }
    pairs, outputs = [], []
    for item in core["original_predictions"]:
        path = Path(item["路径"])
        assert path.parent == DIAGNOSTIC / "predictions"
        raw = read_bytes(path)
        assert digest(raw) == item["SHA256"]
        # 仅反序列化已绑定 SHA 的本地 NumPy 预测，不执行模型代码。
        frame = pickle.loads(gzip.decompress(raw))
        groups = {}
        for row in frame["records"]:
            groups.setdefault(row["query_id"], {})[row["group"]] = row
        for qid, models in groups.items():
            a, b, base = [models[g] for g in ("a_only", "enhanced", "baseline")]
            assert np.array_equal(a["target"], b["target"])
            assert np.array_equal(a["valid"], b["valid"])
            assert a["instruction"] == b["instruction"]
            target, valid = a["target"] & a["valid"], a["valid"]
            pair = {
                "场景token": a["scene"],
                "样本token": frame["token"],
                "指令": qid,
                "任务": a["task"],
                "目标像素": int(target.sum()),
                "有效域像素": int(valid.sum()),
                "目标面积占比": float(target.sum() / valid.sum()),
                "有效配对": bool(a["paired"]),
            }
            for name, row, group in [
                ("A9562", a, "a_only"),
                ("StageB256", b, "enhanced"),
            ]:
                saved = attention_rows[group, a["task"], qid]
                for key in [
                    "IoU",
                    "precision",
                    "recall",
                    "logit_gap",
                    "background_FP",
                    "distractor_FP",
                    "peak_in_target",
                    "empty_active_fraction",
                ]:
                    pair[name + "_" + key] = saved[key]
                logits = row["details"]["correct"]
                for region, mask in [
                    ("目标", target),
                    ("背景", row["background"] & valid),
                    ("同类干扰", row["distractor"] & valid),
                ]:
                    values = logits[mask]
                    pair[name + "_" + region + "Logit均值"] = (
                        float(values.mean()) if values.size else None
                    )
                    pair[name + "_" + region + "Logit中位数"] = (
                        float(np.median(values)) if values.size else None
                    )
                pair[name + "_目标错误Logit均值"] = (
                    float(row["details"]["wrong"][target].mean())
                    if target.any() and a["paired"]
                    else None
                )
            pair["IoU变化_百分点"] = (
                100 * (pair["StageB256_IoU"] - pair["A9562_IoU"])
                if target.any()
                else None
            )
            pairs.append(pair)
            p0, pa, pb = [r["predictions"]["correct"] for r in (base, a, b)]
            d0, da, db = [p["decoded_native"] for p in (p0, pa, pb)]
            assert all(np.array_equal(d0[k], da[k]) for k in d0)
            assert np.array_equal(p0["semantic"], pa["semantic"])
            output = {
                "指令": qid,
                "任务": a["task"],
                "样本token": frame["token"],
                "TopK": len(db["scores"]),
            }
            output["A9562与Baseline解码输出逐数组一致"] = True
            for key in ("boxes", "scores", "labels"):
                assert d0[key].shape == db[key].shape
                output[key + "_与Baseline完全一致"] = bool(
                    np.array_equal(d0[key], db[key])
                )
                output[key + "_相同解码序位最大绝对差"] = float(
                    np.max(np.abs(d0[key] - db[key]))
                )
            output["相同解码序位XY最大坐标差米_非物理位移"] = float(
                np.linalg.norm(d0["boxes"][:, :2] - db["boxes"][:, :2], axis=1).max()
            )
            distances = []
            for label in np.unique(db["labels"]):
                bx = db["boxes"][db["labels"] == label, :2]
                ax = d0["boxes"][d0["labels"] == label, :2]
                assert len(ax), "增强后出现原集合无对应类别"
                distances.extend(
                    np.linalg.norm(bx[:, None] - ax[None], axis=2).min(axis=1).tolist()
                )
            output["B到Baseline同类最近中心距离中位数米"] = statistics.median(distances)
            output["B到Baseline同类最近中心距离最大值米"] = max(distances)
            output["三百框各类别数量一致"] = all(
                int((db["labels"] == k).sum()) == int((d0["labels"] == k).sum())
                for k in set(db["labels"]) | set(d0["labels"])
            )
            output["分割改变像素"] = int(
                np.count_nonzero(p0["semantic"] != pb["semantic"])
            )
            output["分割像素总数"] = p0["semantic"].size
            output["正确错误文本增强分割改变像素"] = int(
                np.count_nonzero(
                    pb["semantic"] != b["predictions"]["wrong"]["semantic"]
                )
            )
            outputs.append(output)
    assert len(pairs) == len(outputs) == 16
    steps, step_paths = {}, {}
    for path in sorted((TRAIN / "sessions").glob("*/steps/*.json")):
        step = int(path.stem)
        assert step not in steps, "存在重复步，需先核对恢复谱系"
        steps[step] = read_json(path)
        step_paths[step] = str(path)
    assert set(steps) == set(range(1, 1025))
    assert [v["update"] for v in steps.values() if "update" in v] == list(range(1, 257))
    sampling = []
    for task in ("map", "object"):
        for present in (True, False):
            rows = [
                v
                for v in steps.values()
                if v["instruction_id"].split("_")[1] == task
                and v["target_present"] == present
            ]
            sampling.append(
                {
                    "任务": task,
                    "目标非空": present,
                    "微步数": len(rows),
                    "比例_本任务512微步": len(rows) / 512,
                }
            )
    gradients = [
        {
            "微步": k,
            "来源": step_paths[k],
            "任务": v["instruction_id"].split("_")[1],
            "梯度": v["task_gradients"],
        }
        for k, v in steps.items()
        if v["task_gradients"]
    ]
    loss_by_task = {}
    for task in ("map", "object"):
        rows = [v for v in steps.values() if v["instruction_id"].split("_")[1] == task]
        loss_by_task[task] = {
            key: describe(v["losses"][key] for v in rows) for key in rows[0]["losses"]
        }
    task_summary = {}
    for task in ("map", "object"):
        rows = [r for r in pairs if r["任务"] == task and r["目标像素"] > 0]
        task_summary[task] = {
            "正目标数": len(rows),
            "B相对A_IoU降低数": sum(r["IoU变化_百分点"] < 0 for r in rows),
            "B相对A_IoU不变数": sum(r["IoU变化_百分点"] == 0 for r in rows),
            "B相对A_IoU升高数": sum(r["IoU变化_百分点"] > 0 for r in rows),
            "A目标面积占比范围": describe(r["目标面积占比"] for r in rows),
            "A峰值命中目标数": sum(r["A9562_peak_in_target"] for r in rows),
            "B峰值命中目标数": sum(r["StageB256_peak_in_target"] for r in rows),
        }
    source_paths = [
        "scripts/stage3/lg_bev_prototype.py",
        "scripts/stage3/lg_bev_engineering_loss.py",
        "scripts/stage3/lg_bev_engineering_train.py",
        "scripts/stage3/lg_bev_supervision_space.py",
        "scripts/stage3/lg_bev_engineering_data.py",
        "scripts/stage3/lg_bev_data.py",
        "scripts/stage3/lg_bev_native.py",
        "scripts/stage2/spatial_annotations.py",
        "scripts/stage2/bev_coordinates.py",
    ]
    execution = read_json(gate_path.parent / "execution.json")
    bound = {}
    for rel in source_paths:
        path = ROOT / rel
        actual = digest(read_bytes(path))
        expected = execution["daily_source_sha256"].get(str(path)) or execution[
            "source_sha256"
        ].get(str(path))
        bound[rel] = {
            "SHA256": actual,
            "正式执行清单中的SHA": expected,
            "一致": actual == expected if expected else None,
        }
        assert expected is None or actual == expected
    result = {
        "范围": "只读固定8帧原预测与StageB256既存1024条标量日志；无重新评分、训练、推理或Checkpoint读取",
        "A128原始门槛结果": {
            k: gate[k]
            for k in [
                "status",
                "positive_macro_IoU",
                "target_correct_wrong_logit_gap",
                "empty_active_fraction",
            ]
        },
        "逐指令变化概要": task_summary,
        "感知输出变化概要": {
            "query数": 16,
            "B解码boxes数值有变化query数": sum(
                not r["boxes_与Baseline完全一致"] for r in outputs
            ),
            "B解码scores数值有变化query数": sum(
                not r["scores_与Baseline完全一致"] for r in outputs
            ),
            "B解码labels数值有变化query数": sum(
                not r["labels_与Baseline完全一致"] for r in outputs
            ),
            "B分割有变化query数": sum(r["分割改变像素"] > 0 for r in outputs),
            "同解码序位XY坐标差_非物理位移": describe(
                r["相同解码序位XY最大坐标差米_非物理位移"] for r in outputs
            ),
            "同类最近中心距离_各query中位数米范围": describe(
                r["B到Baseline同类最近中心距离中位数米"] for r in outputs
            ),
            "同类最近中心距离_各query最大值米范围": describe(
                r["B到Baseline同类最近中心距离最大值米"] for r in outputs
            ),
            "三百框各类别数量一致query数": sum(
                r["三百框各类别数量一致"] for r in outputs
            ),
            "分割改变像素范围_每query80000": describe(
                r["分割改变像素"] for r in outputs
            ),
            "说明": "解码序位可能重排，其坐标差不是物理位移；同类最近中心是B对Baseline单向、多对一集合比较，不使用GT、不重计AP，也不是检测匹配。未保存的Head原始logits为NOT_VERIFIED。",
        },
        "既有AP完整字典相等": {
            part: existing["groups"]["baseline/correct"][part]
            == existing["groups"]["enhanced/correct"][part]
            for part in ["Relevant_Detection_AP", "Irrelevant_Detection_AP"]
        },
        "训练采样": sampling,
        "训练残差相对L2": describe(v["residual_ratio"] for v in steps.values()),
        "训练初始4微步残差均零": all(
            steps[k]["residual_ratio"] == 0 for k in range(1, 5)
        ),
        "训练非零残差微步数": sum(v["residual_ratio"] > 0 for v in steps.values()),
        "256次参数位移L2": describe(
            v["parameter_delta_l2"] for v in steps.values() if "update" in v
        ),
        "按任务Loss分布_非成对收敛曲线": loss_by_task,
        "保存的逐任务梯度探针": gradients,
        "梯度方向余弦": "NOT_VERIFIED：只有范数，不能恢复向量夹角",
        "源码身份核对": bound,
    }
    OUT.mkdir(parents=True)
    write_csv("A9562_StageB256逐指令变化.csv", pairs)
    write_csv("既有解码输出变化.csv", outputs)
    write_csv("StageB256任务与空目标采样.csv", sampling)
    write_json("根因只读分析.json", result)
    write_json(
        "输入来源与SHA256.json",
        {
            "输入": INPUTS,
            "分析脚本SHA256": digest(Path(__file__).read_bytes()),
            "输出": {p.name: digest(p.read_bytes()) for p in sorted(OUT.iterdir())},
        },
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "逐指令变化概要",
                    "感知输出变化概要",
                    "训练采样",
                    "训练残差相对L2",
                    "保存的逐任务梯度探针",
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

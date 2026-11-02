"""只读复核既有 8 帧原始预测，导出 Stage A 核心证据，不重新推理。"""

import csv
import gzip
import json
from pathlib import Path
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

REPORT = Path(__file__).resolve().parent
ORIGINAL = REPORT.parents[1]
sys.path[:0] = [
    str(ORIGINAL / "reports/evaluation_20261102/source"),
    str(ORIGINAL),
    str(ORIGINAL.parents[1]),
]
from core import FORMAL, OUTPUT, SHARED, read, sha
from lifecycle import verified_checkpoint
from core import Store

RUN_ID = "lg_bev_explore_main_20261101_unified_eval_20261102_diagnostic"
RUN = OUTPUT / RUN_ID
PROTOCOL = ORIGINAL / "reports/evaluation_20261102/protocol.json"
protocol = read(PROTOCOL)
identity = dict(
    schema="LG_BEV_UNIFIED_EVAL_20261102_V1",
    action="diagnostic",
    protocol_sha256=sha(PROTOCOL),
)
state, end, checkpoint = verified_checkpoint(
    Store(), RUN_ID, identity, require_outcome="PREFLIGHT_PASS"
)
assert state["cursor"] == 8 and end["check_count"] == 288
assert end["all_a_only_heads_bitwise_equal"]
assert end["adapter_digests"]["a_only"] == protocol["a9562"]["adapter_sha256"]
manifest = read(RUN / "raw_manifest.json")
metrics = read(RUN / "metrics.json")
assert len(manifest["artifacts"]) == 8
historical = read(FORMAL / "warmup_gate.json")
assert historical["status"] == "GROUNDING_NOT_READY"
assert historical["identity"]["adapter_sha256"] == end["adapter_digests"]["a_only"]


def probability(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -60, 60)))


def csv_file(name, rows):
    with (REPORT / name).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


main_rows, paired_rows = [], []
for task, chinese in (("map", "地图"), ("object", "对象")):
    for group in ("a0", "a_only", "enhanced"):
        value = metrics["groups"][group + "/correct"]["Attention"][task]
        main_rows.append(
            dict(
                任务=chinese,
                模型=group,
                正目标宏平均IoU=value["positive_macro_IoU"],
                正目标合并IoU=value["positive_pooled_IoU"],
                正目标条数=value["positive_queries"],
                空目标条数=value["empty_queries"],
                空目标激活比例=value["empty_active_fraction"],
                正确减错误Logit差=value["mean_logit_gap"],
            )
        )
    rows = metrics["groups"]["a_only/correct"]["Attention"][task]["rows"]
    paired = [r for r in rows if r["positive"] and r["counterfactual_valid"]]
    paired_rows.append(
        dict(
            任务=chinese,
            有效正目标配对数=len(paired),
            正确IoU=float(np.mean([r["IoU"] for r in paired])),
            错误IoU=float(np.mean([r["wrong_IoU"] for r in paired])),
            正确减错误Logit差=float(np.mean([r["logit_gap"] for r in paired])),
            正确减错误概率差=float(np.mean([r["probability_gap"] for r in paired])),
        )
    )
csv_file("Attention对照.csv", main_rows)
csv_file("有效指令配对.csv", paired_rows)

from scripts.stage3.lg_bev_engineering_data import LabelStore

labels = LabelStore(read(SHARED / "runs/formal_gt_20261027_v1/data_report.json"))
instance_rows, query_rows, sources = [], [], []
examples = {}
for artifact in manifest["artifacts"]:
    path = OUTPUT / artifact["path"]
    assert sha(path) == artifact["sha256"]
    with gzip.open(path, "rb") as stream:
        frame = pickle.load(stream)
    assert frame["token"] == protocol["diagnostic_tokens"][frame["index"]]
    sources.append(dict(索引=frame["index"], 路径=str(path), SHA256=artifact["sha256"]))
    grouped = {(r["task"], r["group"]): r for r in frame["records"]}
    for task in ("object", "map"):
        row = grouped[task, "a_only"]
        assert row["group"] == "a_only"
        target = row["target"].astype(bool) & row["valid"].astype(bool)
        distractor = row["distractor"].astype(bool) & row["valid"].astype(bool)
        correct = row["details"]["correct"]
        wrong = row["details"]["wrong"]
        entry = dict(
            样本=frame["token"],
            指令=row["query_id"],
            任务=task,
            真实目标像素=int(target.sum()),
            同类干扰像素=int(distractor.sum()),
            配对有效=row["paired"],
            目标正确Logit=float(correct[target].mean()) if target.any() else None,
            目标错误Logit=float(wrong[target].mean()) if target.any() else None,
            干扰正确Logit=(
                float(correct[distractor].mean()) if distractor.any() else None
            ),
            干扰错误Logit=float(wrong[distractor].mean()) if distractor.any() else None,
        )
        query_rows.append(entry)
        # 按固定帧序选择第一个有目标/干扰的例子，不参考模型分数。
        if task not in examples and target.any() and distractor.any():
            examples[task] = (frame["token"], grouped, row["instruction"])
        if task != "object":
            continue
        label = labels.get(frame["token"])
        cls = row["target_class"]
        chosen = set(row["selected_ids"])
        for index, (token, mask, class_id) in enumerate(
            zip(
                label["annotation_tokens"],
                label["object_masks"].numpy(),
                label["labels"],
            )
        ):
            if int(class_id) != cls:
                continue
            mask = mask.astype(bool) & row["valid"]
            if not mask.any():
                continue
            instance_rows.append(
                dict(
                    样本=frame["token"],
                    指令=row["query_id"],
                    实例=token,
                    实例序号=index,
                    类别="car" if cls == 0 else "pedestrian",
                    是否指令目标=token in chosen,
                    像素数=int(mask.sum()),
                    正确平均概率=float(probability(correct)[mask].mean()),
                    错误平均概率=float(probability(wrong)[mask].mean()),
                    正确减错误Logit差=float((correct - wrong)[mask].mean()),
                )
            )
csv_file("逐指令定位.csv", query_rows)
csv_file("同类不同位置实例.csv", instance_rows)

eligible = [
    r
    for r in query_rows
    if r["任务"] == "object"
    and r["配对有效"]
    and r["真实目标像素"]
    and r["同类干扰像素"]
]
spatial = dict(
    可比较对象指令数=len(eligible),
    正确目标强于同类干扰数=sum(
        r["目标正确Logit"] > r["干扰正确Logit"] for r in eligible
    ),
    正确相对错误增强目标数=sum(
        r["目标正确Logit"] > r["目标错误Logit"] for r in eligible
    ),
    同类实例记录数=len(instance_rows),
    说明="像素均值及逐实例记录是定位诊断，不构成唯一实例泛化证明",
)

font = Path("/mnt/c/Windows/Fonts/msyh.ttc")
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=str(font)
    ).get_name()
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
for axis_row, task in zip(axes, ("object", "map")):
    token, grouped, query = examples[task]
    target = grouped[task, "a_only"]["target"]
    distractor = grouped[task, "a_only"]["distractor"]
    for axis, (group, control, title) in zip(
        axis_row,
        (
            ("a0", "correct", "初始化／正确指令"),
            ("a_only", "correct", "A9562／正确指令"),
            ("a_only", "wrong", "A9562／错误指令"),
            (None, None, "真实目标与同类干扰"),
        ),
    ):
        if group:
            display = probability(grouped[task, group]["details"][control])
            im = axis.imshow(display, vmin=0, vmax=1, cmap="magma")
        else:
            display = np.zeros((*target.shape, 3))
            display[target] = [0.1, 0.9, 0.2]
            display[distractor] = [0.9, 0.2, 0.2]
            axis.imshow(display)
        if group:
            axis.contour(target, levels=[0.5], colors=["lime"], linewidths=0.7)
            axis.contour(distractor, levels=[0.5], colors=["cyan"], linewidths=0.5)
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("栅格列索引")
    axis_row[0].set_ylabel("对象原生栅格行" if task == "object" else "地图 GT 栅格行")
fig.suptitle("固定诊断样本的空间注意力：绿线为目标，青线为同类干扰；颜色为激活概率")
fig.colorbar(im, ax=axes[:, :3].ravel().tolist(), shrink=0.5, label="Attention 概率")
for suffix in ("png", "pdf"):
    fig.savefig(REPORT / ("空间注意力对照." + suffix), dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for axis, task, name in zip(axes, ("map", "object"), ("地图", "对象")):
    vals = [
        metrics["groups"][g + "/correct"]["Attention"][task]["positive_macro_IoU"]
        for g in ("a0", "a_only", "enhanced")
    ]
    axis.bar(
        ["初始化", "A9562", "StageB256"],
        np.array(vals) * 100,
        color=["#999999", "#2878B5", "#F28E2B"],
    )
    axis.set_title(name + "正目标 Attention 宏平均 IoU")
    axis.set_ylabel("IoU（%）")
    axis.set_ylim(0, max(vals) * 125)
    for i, value in enumerate(vals):
        axis.text(i, value * 100, f"{value * 100:.2f}%", ha="center", va="bottom")
fig.suptitle("既有固定 8 帧诊断结果：不属于 Development，不能替代 A128")
for suffix in ("png", "pdf"):
    fig.savefig(REPORT / ("Attention指标对照." + suffix), dpi=180)
plt.close(fig)

summary = dict(
    scope="复用已通过的8帧GPU诊断原始预测，仅新增统计与图表，无新模型推理",
    original_protocol_sha256=sha(PROTOCOL),
    checkpoint=str(checkpoint),
    checkpoint_sha256=sha(checkpoint),
    a9562_checkpoint_sha256=protocol["a9562"]["file_sha256"],
    a9562_adapter_sha256=end["adapter_digests"]["a_only"],
    historical_A128=dict(
        path=str(FORMAL / "warmup_gate.json"),
        sha256=sha(FORMAL / "warmup_gate.json"),
        result=historical,
    ),
    diagnostic_attention=main_rows,
    paired_instruction_response=paired_rows,
    same_class_localization=spatial,
    diagnostic_A9562_minus_A0=metrics["A9562_minus_A0"],
    baseline_head_identity=True,
    original_predictions=sources,
    visualization_examples={
        task: dict(token=x[0], instruction=x[2]["instruction_id"])
        for task, x in examples.items()
    },
    formal_acceptance="GROUNDING_NOT_READY，未通过；保持原状态",
    development_inference_frames=0,
)
with (REPORT / "核心验收汇总.json").open("x") as stream:
    json.dump(summary, stream, ensure_ascii=False, indent=2)
print(
    json.dumps(
        dict(完成=True, 正式验收=summary["formal_acceptance"], 定位诊断=spatial),
        ensure_ascii=False,
    )
)

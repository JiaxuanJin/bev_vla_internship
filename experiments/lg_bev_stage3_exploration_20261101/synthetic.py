"""仅CPU的合成三组验收样本；绝不读取真实GT，不是研究结果。"""

from copy import deepcopy

import numpy as np

from evaluation import CONTROLS, aggregate


def records():
    """两个类别、两个驾驶日志；错误文本仅改预测，不改评分GT。"""
    output = []
    for index in range(2):
        semantic = np.ones((40, 40), int)
        semantic[:, 1::2] = 2
        target = np.zeros((40, 40), bool)
        target[0, index] = True
        gt_boxes = [
            {
                "id": "target",
                "xy": [1.0, 1.0],
                "class": "car" if index == 0 else "pedestrian",
            },
            {
                "id": "other",
                "xy": [-1.0, -1.0],
                "class": "car" if index == 0 else "pedestrian",
            },
        ]
        pred_boxes = [{**box, "score": 0.9} for box in gt_boxes]
        for task in ("map", "object"):
            for group in ("baseline", "a_only", "enhanced", "a0"):
                row = {
                    "query_id": f"synthetic_{index}/{task}",
                    "task": task,
                    "group": group,
                    "log": f"synthetic_log_{index}",
                    "scene": f"synthetic_scene_{index}",
                    "target_class": (
                        1 + index if task == "map" else 0 if index == 0 else 8
                    ),
                    "target": target.copy(),
                    "valid": np.ones_like(target),
                    "gt": semantic.copy(),
                    "fb": "front",
                    "lr": "left",
                    "paired": True,
                    "selected_ids": ["target"],
                    "distractor": np.zeros_like(target),
                    "background": ~target,
                    "details": {
                        control: (
                            np.where(target, 1.0, -1.0)
                            if control == "correct"
                            else np.where(target, -1.0, 1.0)
                        )
                        for control in CONTROLS
                    },
                    "predictions": {
                        control: {
                            "gt_boxes": deepcopy(gt_boxes),
                            "pred_boxes": deepcopy(pred_boxes),
                            "semantic": semantic.copy(),
                        }
                        for control in CONTROLS
                    },
                }
                if group == "baseline":
                    row["details"] = {control: None for control in CONTROLS}
                if group == "a0":
                    row["predictions"] = {}
                output.append(row)
    return output


def report():
    """报告明确标记合成输入，不能替代GPU或开发评价。"""
    return {
        "scope": "CPU_SYNTHETIC_NOT_MODEL_RESULTS",
        "development_GT_read": False,
        "cuda_initialized": False,
        "metrics": aggregate(records()),
    }

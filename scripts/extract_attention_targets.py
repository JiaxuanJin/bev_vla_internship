"""
功能：DriveLM 语言语义解析（规则法原型）
输入：data/drivelm/train_sample.json
输出：outputs/drivelm_attention_targets.json

主要功能：
1. 提取目标相对于自车的空间方向
2. 识别目标类别和运动状态
3. 避免方向重复和 ego car 误识别
4. 保存结构化结果，为后续 BEV 注意力生成提供输入
"""

import json
import re
from pathlib import Path

# 优先匹配更具体的空间方向
DIRECTIONS = [
    "front left", "front right",
    "back left", "back right",
    "rear left", "rear right",
    "front", "back", "left", "right"
]

# 将单复数统一成标准类别
OBJECT_PATTERNS = {
    "construction vehicle": r"\bconstruction vehicles?\b",
    "barrier": r"\bbarriers?\b",
    "truck": r"\btrucks?\b",
    "trailer": r"\btrailers?\b",
    "car": r"\bcars?\b",
    "bus": r"\bbus(?:es)?\b",
    "bicycle": r"\bbicycles?\b",
    "pedestrian": r"\bpedestrians?\b",
}

STATUS_PATTERNS = {
    "moving": r"\bmoving\b",
    "parked": r"\bparked\b",
    "stopped": r"\bstopped\b",
    "standing": r"\bstanding\b",
}

def extract_direction(text):
    """提取最具体的空间方向，避免重复匹配。"""
    text = text.lower()
    for direction in DIRECTIONS:
        if re.search(rf"\b{re.escape(direction)}\b", text):
            return direction.replace("rear", "back")
    return None

def extract_keywords(text, patterns):
    """根据关键词规则识别并标准化目标或状态。"""
    text = text.lower()

    # 去掉 ego car，防止把自车当成关注目标
    text = re.sub(r"\bego car\b", " ", text)

    return [
        name for name, pattern in patterns.items()
        if re.search(pattern, text)
    ]

# 读取 DriveLM 标注
with open("data/drivelm/train_sample.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

# 逐场景、逐关键帧解析语言问答
for scene_id, scene in data.items():
    for frame_id, frame in scene.get("key_frames", {}).items():
        qa_list = frame.get("QA", {}).get("perception", [])

        for qa in qa_list:
            question = qa.get("Q", "")
            answer = qa.get("A", "")

            if not isinstance(question, str) or not isinstance(answer, str):
                continue

            result = {
                "scene_id": scene_id,
                "frame_id": frame_id,
                "question": question,
                "answer": answer,
                "direction": extract_direction(question + " " + answer),
                "objects": extract_keywords(answer, OBJECT_PATTERNS),
                "states": extract_keywords(answer, STATUS_PATTERNS)
            }

            results.append(result)

# 保存结构化结果，方便后续注意力模块调用
output = Path("outputs/drivelm_attention_targets.json")
output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 展示前五条解析结果
print("========== DriveLM 语义解析结果 ==========")

for i, item in enumerate(results[:5], start=1):
    print(f"\nQA {i}")
    print("关注方向:", item["direction"])
    print("关注目标:", item["objects"])
    print("目标状态:", item["states"])

print(f"\n总计解析: {len(results)} 条问答")
print("保存位置:", output)

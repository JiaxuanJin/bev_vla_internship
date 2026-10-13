"""
功能：检查 DriveLM-v1 示例数据结构
输入：data/drivelm/train_sample.json
用途：分析场景描述、关键帧及语言标注字段
"""

import json

# 1. 读取 DriveLM 官方示例数据
with open("data/drivelm/train_sample.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("========== DriveLM 数据分析 ==========")
print("场景数量:", len(data))

# 2. 逐个检查场景及关键帧数量
for scene_id, scene in data.items():
    print("\n场景 ID:", scene_id)
    print("场景字段:", list(scene.keys()))

    frames = scene.get("key_frames", {})

    print("关键帧数量:", len(frames))

    # 3. 只展示第一个关键帧，避免输出过长
    if frames:
        first_id, first_frame = next(iter(frames.items()))

        print("第一个关键帧 ID:", first_id)
        print("关键帧数据类型:", type(first_frame).__name__)

        if isinstance(first_frame, dict):
            print("关键帧包含字段:", list(first_frame.keys()))

            # 展示部分数据，观察问答和标注结构
            print("关键帧内容预览:")
            print(json.dumps(
                first_frame,
                ensure_ascii=False,
                indent=2
            )[:2500])

    # 只详细检查第一个场景
    break

print("\n========== 分析完成 ==========")

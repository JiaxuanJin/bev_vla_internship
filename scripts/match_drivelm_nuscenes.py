"""
功能：匹配 DriveLM-v1 与 nuScenes Mini 的场景及关键帧
输入：DriveLM 示例 JSON、nuScenes Mini
输出：匹配数量与匹配示例
"""

import json
from nuscenes.nuscenes import NuScenes

# 1. 加载两个数据集
with open("data/drivelm/train_sample.json", "r", encoding="utf-8") as f:
    drivelm = json.load(f)

nusc = NuScenes(
    version="v1.0-mini",
    dataroot="data/nuscenes",
    verbose=False
)

# 2. 建立 nuScenes 场景和关键帧索引
scene_tokens = {scene["token"] for scene in nusc.scene}
sample_tokens = {sample["token"] for sample in nusc.sample}

# 3. 提取 DriveLM 场景与关键帧 ID
drivelm_scenes = set(drivelm.keys())
drivelm_frames = set()

for scene in drivelm.values():
    frames = scene.get("key_frames", {})

    if isinstance(frames, dict):
        drivelm_frames.update(frames.keys())

# 4. 计算两个数据集的交集
matched_scenes = drivelm_scenes & scene_tokens
matched_frames = drivelm_frames & sample_tokens

print("========== 数据匹配结果 ==========")
print("DriveLM 示例场景数量:", len(drivelm_scenes))
print("nuScenes Mini 场景数量:", len(scene_tokens))
print("场景匹配数量:", len(matched_scenes))

print("DriveLM 示例关键帧数量:", len(drivelm_frames))
print("nuScenes Mini 关键帧数量:", len(sample_tokens))
print("关键帧匹配数量:", len(matched_frames))

# 5. 输出部分匹配结果
if matched_frames:
    print("\n匹配成功的关键帧:")
    for token in sorted(matched_frames)[:5]:
        print(token)
else:
    print("\n当前未找到匹配关键帧。")
    print("需要进一步检查完整数据集或 ID 映射关系。")

print("\n========== 检查完成 ==========")

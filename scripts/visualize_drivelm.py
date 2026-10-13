"""
功能：DriveLM 多模态数据可视化
输入：DriveLM 示例 JSON 和对应的六路摄像头图像
输出：六路图像、目标标注框及视觉问答示例
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# 1. 设置数据路径
json_path = Path("data/drivelm/train_sample.json")
image_root = Path(
    "data/drivelm/DriveLM-official/challenge/"
    "llama_adapter_v2_multimodal7b/data/nuscenes"
)

# 2. 加载第一个场景的第一个关键帧
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

scene_id, scene = next(iter(data.items()))
frame_id, frame = next(iter(scene["key_frames"].items()))

cameras = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"
]

# 3. 创建六路摄像头画布
fig, axes = plt.subplots(2, 3, figsize=(18, 9))

for ax, camera in zip(axes.flat, cameras):
    # 根据 JSON 的 image_paths 寻找对应图像
    raw_path = frame["image_paths"][camera]
    relative_path = raw_path.split("samples/", 1)[1]
    image_path = image_root / "samples" / relative_path

    if not image_path.exists():
        ax.text(0.5, 0.5, "Image missing",
                ha="center", va="center")
        ax.axis("off")
        print("缺少图片:", image_path)
        continue

    with Image.open(image_path) as img:
        ax.imshow(img)

    # 4. 绘制 DriveLM 提供的 2D 目标标注框
    for obj_id, obj in frame["key_object_infos"].items():
        if f",{camera}," not in obj_id:
            continue

        bbox = obj.get("2d_bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="red", linewidth=2
        )
        ax.add_patch(rect)

    ax.set_title(camera)
    ax.axis("off")

# 5. 展示当前关键帧的一组视觉问答
qa = frame.get("QA", {}).get("perception", [])
if qa:
    question = qa[0].get("Q", "")
    answer = qa[0].get("A", "")
    fig.suptitle(
        f"DriveLM Multimodal Example\n"
        f"Q: {question}\nA: {answer}",
        fontsize=12
    )

plt.tight_layout(rect=[0, 0, 1, 0.90])

# 6. 保存结果，保留原始数据不变
output = Path("outputs/drivelm_multimodal_example.png")
output.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(output, dpi=150, bbox_inches="tight")
plt.close()

print("Scene:", scene_id)
print("Frame:", frame_id)
print("Saved:", output.resolve())

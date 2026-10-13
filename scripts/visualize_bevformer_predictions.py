"""
功能：BEVFormer-Small 真实 3D 预测结果可视化
输入：官方生成的 results_nusc.json + nuScenes Mini
输出：相机预测框和 BEV 鸟瞰预测框

说明：
仅绘制模型 Prediction，不使用 Ground Truth 代替预测。
坐标转换：全局坐标 -> 自车坐标 -> 相机坐标。
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import box_in_image, BoxVisibility


# 1. 自动寻找最近生成的官方预测结果
files = list(Path("test/bevformer_small").glob(
    "*/pts_bbox/results_nusc.json"
))
if not files:
    raise FileNotFoundError("未找到 results_nusc.json")

result_path = max(files, key=lambda p: p.stat().st_mtime)

with open(result_path, "r", encoding="utf-8") as f:
    predictions = json.load(f)["results"]

# 2. 加载 nuScenes 数据集
nusc = NuScenes(
    version="v1.0-mini",
    dataroot="data/nuscenes",
    verbose=False
)

# 置信度阈值：只展示相对可靠的预测
SCORE_THRESHOLD = 0.35

COLORS = {
    "car": "red",
    "truck": "orange",
    "bus": "yellow",
    "pedestrian": "lime",
    "bicycle": "cyan",
    "motorcycle": "magenta",
    "traffic_cone": "blue",
}


def make_box(pred):
    """将 nuScenes 预测记录转换为 3D Box。"""
    return Box(
        center=pred["translation"],
        size=pred["size"],
        orientation=Quaternion(pred["rotation"]),
        name=pred["detection_name"],
        score=pred["detection_score"]
    )


def to_ego(box, ego_pose):
    """全局坐标转换到自车坐标。"""
    box.translate(-np.array(ego_pose["translation"]))
    box.rotate(Quaternion(ego_pose["rotation"]).inverse)
    return box


def to_camera(box, calibration):
    """自车坐标转换到相机坐标。"""
    box.translate(-np.array(calibration["translation"]))
    box.rotate(Quaternion(calibration["rotation"]).inverse)
    return box


def get_frame_boxes(token):
    """获取某帧的相机、位姿和可见预测框。"""
    sample = nusc.get("sample", token)
    cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    ego = nusc.get("ego_pose", cam["ego_pose_token"])
    calib = nusc.get(
        "calibrated_sensor", cam["calibrated_sensor_token"]
    )
    intrinsic = np.array(calib["camera_intrinsic"])
    size = (cam["width"], cam["height"])

    camera_boxes = []
    ego_boxes = []

    for pred in predictions[token]:
        if pred["detection_score"] < SCORE_THRESHOLD:
            continue

        box_ego = to_ego(make_box(pred), ego)
        ego_boxes.append(box_ego)

        box_cam = to_camera(box_ego.copy(), calib)

        if box_in_image(
            box_cam, intrinsic, size, vis_level=BoxVisibility.ANY
        ):
            camera_boxes.append(box_cam)

    return cam, intrinsic, camera_boxes, ego_boxes


# 3. 自动选取前视摄像头可见预测较多的一帧
best_token = None
best_count = -1

for token in predictions:
    _, _, boxes, _ = get_frame_boxes(token)
    if len(boxes) > best_count:
        best_count = len(boxes)
        best_token = token

cam, intrinsic, camera_boxes, ego_boxes = get_frame_boxes(best_token)

# 4. 创建相机与 BEV 双视角画布
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 左侧：在真实相机图像中绘制预测 3D 框
image_path = nusc.get_sample_data_path(cam["token"])

with Image.open(image_path) as image:
    axes[0].imshow(image)

for box in camera_boxes:
    color = COLORS.get(box.name, "white")
    box.render(
        axes[0],
        view=intrinsic,
        normalize=True,
        colors=(color, color, color),
        linewidth=1.8
    )

# 固定相机像素坐标范围，避免画框时坐标轴自动扩大
axes[0].set_xlim(0, cam["width"])
axes[0].set_ylim(cam["height"], 0)
axes[0].set_aspect("equal")

axes[0].set_title(
    f"BEVFormer Prediction - CAM_FRONT ({len(camera_boxes)} boxes)"
)
axes[0].axis("off")

# 右侧：将预测框绘制在自车 BEV 鸟瞰坐标系中
ax = axes[1]
ax.set_facecolor("#151923")

for box in ego_boxes:
    # 提取 3D 框底部四角，形成鸟瞰矩形
    corners = box.bottom_corners()
    order = [0, 1, 2, 3, 0]

    # 自车坐标 x=前方、y=左侧
    # 绘图坐标：横轴向右为正，纵轴向前为正
    plot_x = -corners[1, order]
    plot_y = corners[0, order]

    color = COLORS.get(box.name, "white")
    ax.plot(plot_x, plot_y, color=color, linewidth=1.5)

# 自车位置
ax.scatter(0, 0, color="cyan", marker="^", s=150, label="Ego")

ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_aspect("equal")
ax.set_title(f"BEVFormer Prediction - BEV ({len(ego_boxes)} boxes)")
ax.set_xlabel("Right (+) / Left (-), metres")
ax.set_ylabel("Forward (+) / Backward (-), metres")
ax.grid(alpha=0.2)
ax.legend()

# 5. 保存图片
output = Path("../outputs/bevformer_predictions_visualized.png")
output.parent.mkdir(parents=True, exist_ok=True)

plt.tight_layout()
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.close()

print("========== BEVFormer 预测可视化 ==========")
print("使用结果:", result_path)
print("Sample Token:", best_token)
print("前视相机可见预测框:", len(camera_boxes))
print("BEV 范围内用于绘图的预测框:", len(ego_boxes))
print("输出图片:", output.resolve())

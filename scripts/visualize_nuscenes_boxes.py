from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

root = Path("data/nuscenes")
nusc = NuScenes(version="v1.0-mini", dataroot=str(root), verbose=True)

sample = nusc.sample[0]
camera = "CAM_FRONT"
cam_token = sample["data"][camera]

_, boxes, camera_intrinsic = nusc.get_sample_data(cam_token)

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
nusc.render_sample_data(cam_token, with_anns=True, ax=ax, verbose=False)
ax.set_title(f"{camera} with 3D boxes")

output = Path("outputs/nuscenes_cam_front_boxes.png")
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved visualization: {output.resolve()}")
print(f"Number of 3D boxes in {camera}: {len(boxes)}")

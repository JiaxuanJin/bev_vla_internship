from pathlib import Path
from nuscenes.nuscenes import NuScenes
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("data/nuscenes")
nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(root),
    verbose=True
)

sample = nusc.sample[0]

cameras = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

fig, axes = plt.subplots(2, 3, figsize=(18, 8))

for ax, camera in zip(axes.flat, cameras):
    token = sample["data"][camera]
    image_path = nusc.get_sample_data_path(token)

    with Image.open(image_path) as img:
        ax.imshow(img)

    ax.set_title(camera)
    ax.axis("off")

plt.tight_layout()

output = Path("outputs/nuscenes_6cameras.png")
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved visualization: {output.resolve()}")

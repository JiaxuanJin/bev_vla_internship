from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(
    version="v1.0-mini",
    dataroot="data/nuscenes",
    verbose=False
)

sample = nusc.sample[0]
lidar_token = sample["data"]["LIDAR_TOP"]

fig, ax = plt.subplots(figsize=(10, 10))

nusc.render_sample_data(
    lidar_token,
    with_anns=True,
    nsweeps=1,
    axes_limit=40,
    ax=ax,
    verbose=False
)

ax.set_title("nuScenes LiDAR - Bird's Eye View")

output = Path("outputs/nuscenes_lidar_bev.png")
fig.savefig(output, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output.resolve()}")

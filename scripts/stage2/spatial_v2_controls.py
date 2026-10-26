"""补充同一封存留出协议的第一轮错误方位 Attention，禁止重新选模型。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.micro_training import forward_payload, get_batch
from scripts.stage2.spatial_annotations import load_target, project_attention
from scripts.stage2.spatial_experiment import write_json
from scripts.stage2.spatial_v2_experiment import V1, final_dataset, swap_direction
from scripts.stage2.spatial_v2_experiment import configure_v2
from scripts.stage2.pipeline import Stage2Pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--axis", choices=("lr", "fb"), default="lr")
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    args.predictions = args.predictions.resolve()
    args.annotations = args.annotations.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.selection:
        args.selection = args.selection.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(args.predictions.read_text())
    model, wrapped, pipeline, dataset = final_dataset()
    pipeline.fusion.load_state_dict(torch.load(V1, map_location="cpu", weights_only=False)["state_dict"])
    pipeline.eval()
    second = None
    if args.axis == "fb":
        selected = json.loads(args.selection.read_text())["selected"]
        second = configure_v2(Stage2Pipeline(pipeline.encoder, pipeline.grid).cuda())
        second.fusion.load_state_dict(torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)["state_dict"])
        second.eval()
    bevs = {}
    records = []
    for number, row in enumerate(metadata["annotations"]):
        if row["index"] not in bevs:
            captured = {}
            original = model.pts_bbox_head.transformer.get_bev_features
            def capture(*a, **kw):
                value = original(*a, **kw)
                captured["bev"] = BEVGrid().to_map(value).detach()
                return value
            model.pts_bbox_head.transformer.get_bev_features = capture
            model.prev_frame_info = {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}
            data = get_batch(dataset, row["index"], 20261027 + row["index"])
            try:
                with torch.inference_mode():
                    wrapped(return_loss=False, rescale=True, **forward_payload(data, inference=True))
                bevs[row["index"]] = captured["bev"]
            finally:
                del model.pts_bbox_head.transformer.get_bev_features
        _, target, grid = load_target(row, args.annotations, "cuda")
        with torch.inference_mode():
            wrong_text = (swap_direction(row["text"]) if args.axis == "lr" else
                          row["text"].replace("front", "REAR").replace("rear", "front").replace("REAR", "rear"))
            correct = pipeline(bevs[row["index"]], [row["text"]], strict=False)["attention"]
            wrong = pipeline(bevs[row["index"]], [wrong_text], strict=False)["attention"]
            pa = project_attention(correct, grid)[0, 0].cpu().numpy()
            pb = project_attention(wrong, grid)[0, 0].cpu().numpy()
            if second is not None:
                second.fusion.set_calibration(row)
                second_wrong = second(bevs[row["index"]], [wrong_text], strict=False)["attention"]
                second_wrong = project_attention(second_wrong, grid)[0, 0].cpu().numpy()
        with np.load(args.predictions.parent / row["prediction_file"]) as existing:
            delta = float(np.abs(pa - existing["historical_attention"]).max())
        assert delta < 1e-6, delta
        filename = f"control_{number:03d}.npz"
        np.savez_compressed(args.output_dir / filename, v1_incorrect_attention=pb)
        records.append({"instruction_id": row["instruction_id"], "file": filename,
                        "v1_correct_attention_reproduction_max_abs": delta})
        if second is not None:
            with np.load(args.predictions.parent / row["prediction_file"]) as existing:
                truth = existing["target"]
                gap1 = float((existing["historical_attention"][truth] - pb[truth]).mean())
                gap2 = float((existing["spatial_attention"][truth] - second_wrong[truth]).mean())
            records[-1].update(v1_target_probability_gap=gap1, v2_target_probability_gap=gap2,
                               v1_correct_wins=gap1 > 0, v2_correct_wins=gap2 > 0)
    write_json(args.output_dir / "controls.json", {"records": records,
        "axis": args.axis,
        "scope": "same sealed checkpoint; supplementary diagnosis, no model selection or further training"})
    print("V1_WRONG_CONTROLS", len(records), "exact same-BEV checks passed")


if __name__ == "__main__":
    main()

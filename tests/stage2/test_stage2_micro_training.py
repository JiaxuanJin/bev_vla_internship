"""微型训练的指标、参数更新及异常拒绝回归；不冒充真实 GPU 训练。"""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage2.micro_training import (
    confusion,
    detection_counts,
    parameter_stats,
    segmentation_metrics,
    write_json,
)


def test_pooled_iou_and_background_false_positives():
    matrix = confusion(np.array([0, 0, 1, 2, 3]), np.array([0, 1, 1, 2, 0]))
    result = segmentation_metrics(matrix)
    assert result["class_iou"] == {
        "Background": 1 / 3,
        "Divider": 1 / 2,
        "Ped Crossing": 1.0,
        "Boundary": 0.0,
    }
    assert result["background_false_positive_rate"] == 0.5
    assert result["foreground_miou"] == 0.5
    assert result["physical_alignment"] == "NOT_VERIFIED"


def test_missing_class_and_invalid_raster_cannot_create_scores():
    result = segmentation_metrics(confusion(np.zeros((2, 3)), np.zeros((2, 3))))
    assert result["class_iou"]["Divider"] is None
    assert result["foreground_miou"] is None
    with pytest.raises(ValueError, match="resize"):
        confusion(np.zeros((2, 3)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="类别"):
        confusion(np.array([4]), np.array([0]))


def test_detection_matching_is_classwise_one_to_one_and_score_filtered():
    result = detection_counts(
        np.array([[0, 0], [0.1, 0], [5, 0], [0, 0], [100, 0]]),
        np.array([0.9, 0.8, 0.05, 0.7, 0.99]),
        np.array([0, 0, 1, 1, 0]),
        np.array([[0, 0], [5, 0]]),
        np.array([0, 1]),
    )
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 1)


def test_parameter_stats_detect_real_update_and_nonfinite_gradient():
    module = torch.nn.Linear(2, 1)
    before = {k: v.detach().clone() for k, v in module.state_dict().items()}
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01)
    module(torch.ones(1, 2)).square().sum().backward()
    optimizer.step()
    stats = parameter_stats(module, before)
    assert all(x["delta_l2"] > 0 and x["gradient_l2"] > 0 for x in stats.values())
    module.weight.grad.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        parameter_stats(module, before)


def test_protocol_is_exclusive_and_rejects_nan(tmp_path):
    path = tmp_path / "protocol.json"
    write_json(path, {"steps": 20})
    with pytest.raises(FileExistsError):
        write_json(path, {"steps": 10})
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"loss": float("nan")})


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 101, 1024])
def test_stream_original_records_across_chunk_and_unicode_boundaries(
    tmp_path, chunk_size
):
    import json
    from scripts.stage2.low_memory_dataset import json_records

    rows = [
        {"token": str(i), "raw": '原始 GT, [ ] " \\ \n' * (i + 1)} for i in range(12)
    ]
    path = tmp_path / "records.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    assert list(json_records(path, chunk_size)) == rows
    path.write_text("[]")
    assert list(json_records(path, chunk_size)) == []


@pytest.mark.parametrize("contents", ["[{}", "[{},]", "[{} {}]", "[{}] junk", "{}"])
def test_stream_rejects_corrupt_metadata(tmp_path, contents):
    from scripts.stage2.low_memory_dataset import json_records

    path = tmp_path / "bad.json"
    path.write_text(contents)
    with pytest.raises(ValueError):
        list(json_records(path, 3))


def test_forward_payload_shares_images_preserves_gt_and_isolates_mutable_metadata():
    from scripts.stage2.micro_training import forward_payload

    image, gt = torch.ones(4), torch.arange(4)
    data = {
        "img": image,
        "semantic_indices": gt,
        "img_metas": [{"can_bus": np.arange(4, dtype=float)}],
    }
    payload = forward_payload(data)
    assert payload["img"] is image and payload["semantic_indices"] is gt
    payload["img_metas"][0]["can_bus"][:] = 0
    np.testing.assert_array_equal(data["img_metas"][0]["can_bus"], np.arange(4))
    assert "semantic_indices" not in forward_payload(data, inference=True)


def test_memory_guard_fails_before_allocations(monkeypatch):
    import scripts.stage2.runtime_memory as memory

    monkeypatch.setattr(memory, "memory_snapshot", lambda: {"MemAvailable_mib": 100})
    with pytest.raises(MemoryError, match="LOW_HOST_MEMORY"):
        memory.memory_event("test", min_available_mib=2048)


def test_resume_requires_matching_immutable_protocol_and_step(tmp_path):
    from scripts.stage2.micro_training import load_resume
    from scripts.stage2.run_step70_joint_integration import sha256

    protocol = tmp_path / "protocol.json"
    protocol.write_text('{"steps": 20}')
    checkpoint = tmp_path / "step.pt"
    torch.save({"steps": 1, "protocol_sha256": sha256(protocol)}, checkpoint)
    assert load_resume(checkpoint, protocol)["steps"] == 1
    protocol.write_text('{"steps": 19}')
    with pytest.raises(ValueError, match="protocol hash"):
        load_resume(checkpoint, protocol)


def test_fresh_process_rejects_twenty_updates_before_launch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scripts.stage2.micro_process as processes

    def unexpected(*args, **kwargs):
        pytest.fail("Must reject before launching a child")

    monkeypatch.setattr(processes, "run_worker", unexpected)
    args = SimpleNamespace(
        output=tmp_path / "new/result.json",
        micro_steps=20,
        micro_phase="step",
        protocol=None,
        resume=None,
    )
    with pytest.raises(ValueError, match="one step"):
        processes.run_micro_stages(args)

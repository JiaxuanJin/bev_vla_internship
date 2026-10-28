"""未改作者原生六相机时序推理与FP32分片的按需重建。"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import time
from typing import Iterator

import numpy as np
import torch

from scripts.stage2.bev_coordinates import BEVGrid
from scripts.stage2.micro_training import forward_payload, get_batch
from scripts.stage2.spatial_annotations import AUTHOR, DATA
from scripts.stage3.lg_bev_common import ROOT, sha
from scripts.stage3.lg_bev_engineering_cache import RollingCache
from scripts.stage3.lg_bev_engineering_data import SceneData, initialize_author

CHECKPOINT = Path("/mnt/g/nus/bevformer_small_joint_author.pth")


def source_stats(paths: list[str]) -> dict:
    """每传感器目录解析一次真实路径，每文件仅一次lstat，识别单文件symlink。"""
    parents = {parent: parent.resolve() for parent in {Path(p).parent for p in paths}}
    result = {}
    for name in paths:
        path = Path(name)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            resolved, info = path.resolve(), path.stat()
        else:
            resolved = parents[path.parent] / path.name
        result[name] = [str(resolved), info.st_size, info.st_mtime_ns]
    return result


def verified_scene_sources(
    cache: RollingCache, data: SceneData, expected: dict | None = None
) -> tuple[dict, bool]:
    """原始文件首次真实SHA，后续校验规范路径/size/mtime及manifest自身SHA。

    数据集必须保持只读；来源变化或manifest损坏立即停止，不静默接受新数据。
    仅缓存完整性元数据，不缓存/量化BEV，不向模型传GT。
    """
    from scripts.stage3.lg_bev_engineering_cache import contained

    paths = sorted(
        {
            str(DATA / media["filename"])
            for row in data.rows
            for media in row["media"].values()
        }
    )
    stats = source_stats(paths)
    path = contained(
        cache.root, "source_manifests/" + data.rows[0]["scene_token"] + ".json"
    )
    if path.exists():
        saved = json.loads(path.read_text())
        checksum = saved.pop("checksum")
        if (
            hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()
            != checksum
        ):
            raise ValueError("原始来源manifest损坏")
        if saved["stats"] != stats or sorted(saved["sources"]) != paths:
            raise ValueError("原始来源路径/size/mtime改变，停止而非接受新数据")
        sources = saved["sources"]
        hit = True
    else:
        sources = {p: sha(Path(p)) for p in paths}
        after = source_stats(paths)
        if after != stats:
            raise ValueError("原始文件在SHA读取期间变化")
        saved = dict(
            version=1,
            stats=stats,
            sources=sources,
            policy="immutable raw sources; actual SHA first, stat+manifest SHA on reuse",
        )
        saved["checksum"] = hashlib.sha256(
            json.dumps(saved, sort_keys=True).encode()
        ).hexdigest()
        with cache.locked():
            cache.reserve_for_artifact(len(json.dumps(saved).encode()) + 8192)
            path.parent.mkdir(exist_ok=True)
            pending = path.with_name(path.name + ".pending")
            with pending.open("x") as stream:
                json.dump(saved, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists():
                raise ValueError("来源manifest发生并发写入，禁止覆盖")
            os.rename(pending, path)
        hit = False
    if expected is not None and sources != expected:
        raise ValueError("原始来源SHA与真实GT来源不符")
    return sources, hit


@contextmanager
def preserve_rng() -> Iterator[None]:
    """缓存miss不得改变训练的Python/NumPy/CPU/CUDA随机状态。"""
    state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        torch.cuda.get_rng_state_all(),
    )
    try:
        yield
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        torch.cuda.set_rng_state_all(state[3])


class NativeRuntime:
    """冻结完整作者模型，video_test_mode=True，保留原生history和scene reset。"""

    def __init__(self, *, fast_images: bool = False) -> None:
        from mmcv.parallel import MMDataParallel
        from mmcv.runner import load_checkpoint
        from mmdet3d.models import build_model

        if not torch.cuda.is_available():
            raise RuntimeError("真实验收需要本机CUDA")
        self.cfg = initialize_author()
        checkpoint_digest = sha(CHECKPOINT)
        if (
            checkpoint_digest
            != "1345919901785b8df7ee16838d62f9f7a00444c9d8061f0dbee712d1b1b67dcf"
        ):
            raise ValueError("原作者checkpoint与封存协议不一致")
        self.fast_images = fast_images
        self.cfg.model.pretrained = None
        self.cfg.model.video_test_mode = True
        self.model = build_model(self.cfg.model, test_cfg=self.cfg.get("test_cfg"))
        saved = load_checkpoint(self.model, str(CHECKPOINT), map_location="cpu")
        self.model.CLASSES = saved["meta"]["CLASSES"]
        del saved
        self.model.requires_grad_(False).eval().cuda()
        self.wrapped = MMDataParallel(self.model, device_ids=[0])
        self.state_scene = None
        self.state_index = -1
        self.raw_calls = 0
        self.raw_seconds = 0.0
        self.lineage = dict(
            checkpoint_sha256=checkpoint_digest,
            config_sha256=sha(
                AUTHOR / "projects/configs/bevformer/bevformer_small_seg_det.py"
            ),
            preprocessing="author test_pipeline; fixed scale0.8, normalize,pad32,no photometric; video_test_mode=True; FP32",
            source_dtype="torch.float32",
            history_policy="native unenhanced full scene chronological recurrent history; reset only atscene start",
            environment=dict(
                torch=torch.__version__,
                cuda=torch.version.cuda,
                tf32=torch.backends.cuda.matmul.allow_tf32,
            ),
        )
        self.lineage["effective_config_sha256"] = hashlib.sha256(
            self.cfg.pretty_text.encode()
        ).hexdigest()
        self.lineage["data_implementation_sha256"] = sha(
            ROOT / "scripts/stage3/lg_bev_engineering_data.py"
        )
        self.lineage["fast_image_only_unused_GT_removed"] = fast_images

    def reset(self) -> None:
        """只重置原生运行时状态，不修改作者参数或buffer。"""
        self.model.prev_frame_info = dict(
            prev_bev=None, scene_token=None, prev_pos=0, prev_angle=0
        )
        self.state_scene = None
        self.state_index = -1

    def forward(self, data: SceneData, index: int) -> tuple[torch.Tensor, dict, dict]:
        """真实六相机前向，hook完整Head输出；任何非连续访问必须先从0重建。"""
        row = data.rows[index]
        if index == 0:
            self.reset()
        elif self.state_scene != row["scene_token"] or self.state_index != index - 1:
            raise RuntimeError("拒绝在错误temporal history下生成BEV")
        captured = []
        with preserve_rng():
            preprocessing_start = time.perf_counter()
            if self.fast_images:
                old_pipeline = data.dataset.pipeline
                data.dataset.pipeline = data.image_pipeline
                data.dataset.get_map_ann_info = lambda _info: []
                try:
                    batch = get_batch(data.dataset, index, 20261027)
                finally:
                    data.dataset.pipeline = old_pipeline
                    del data.dataset.get_map_ann_info
            else:
                batch = get_batch(data.dataset, index, 20261027)
            preprocessing_seconds = time.perf_counter() - preprocessing_start
            from scripts.stage2.joint_runtime import _first_tensor

            gt_digest = (
                hashlib.sha256(
                    _first_tensor(batch["semantic_indices"])
                    .reshape(200, 400)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                    .tobytes()
                ).hexdigest()
                if not self.fast_images
                else None
            )
            history_used = self.model.prev_frame_info["prev_bev"] is not None
            handle = self.model.pts_bbox_head.register_forward_hook(
                lambda _m, _a, out: captured.append(out)
            )
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                with torch.no_grad():
                    result = self.wrapped(
                        return_loss=False,
                        rescale=True,
                        **forward_payload(batch, inference=True),
                    )
            finally:
                handle.remove()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        if len(captured) != 1:
            raise RuntimeError("完整Head调用次数异常")
        out = captured[0]
        sequence = out["bev_embed"]
        if sequence.dtype != torch.float32:
            raise TypeError("原生输出非FP32，禁止回转伪造")
        # 作者transformer.forward在decoder前把[B,HW,C]显式permute为[HW,B,C]。
        if sequence.shape != (22500, 1, 256):
            raise ValueError(f"未知原生Head BEV布局:{sequence.shape}")
        bev = BEVGrid().to_map(sequence.permute(1, 0, 2)).detach().cpu().contiguous()
        prediction = {
            k: out[k].detach().cpu().contiguous()
            for k in ("all_cls_scores", "all_bbox_preds", "seg_preds")
        }
        self.state_scene = row["scene_token"]
        self.state_index = index
        self.raw_calls += 1
        self.raw_seconds += elapsed
        trace = dict(
            sample_token=row["sample_token"],
            scene_token=row["scene_token"],
            timestamp=row["timestamp"],
            index=index,
            history_used=history_used,
            source_dtype=str(sequence.dtype),
            bev_shape=list(bev.shape),
            seconds=elapsed,
            decoded_boxes=len(result[0]["pts_bbox"]["scores_3d"]),
        )
        trace["map_gt_sha256"] = gt_digest
        trace["preprocessing_seconds"] = preprocessing_seconds
        return bev, prediction, trace


class NativeCacheProvider:
    """按8帧分片访问；miss重建整个原生前缀，缓存命中不伪造history状态。"""

    def __init__(
        self,
        cache: RollingCache,
        runtime: NativeRuntime,
        scenes: dict[str, SceneData],
        shard_frames: int = 8,
        *,
        admit_disk: bool = True,
        expected_sources: dict[str, dict] | None = None,
    ) -> None:
        self.cache, self.runtime, self.scenes = cache, runtime, scenes
        self.shard_frames = shard_frames
        self.admit_disk = admit_disk
        self.index = {
            row["sample_token"]: (scene, i)
            for scene, data in scenes.items()
            for i, row in enumerate(data.rows)
        }
        self.fingerprints = {}
        self.metrics = []
        self.source_hashes = {}
        self.resident_key = None
        self.resident = {}
        self.source_manifest_hits = {}
        for scene, data in scenes.items():
            sources, hit = verified_scene_sources(
                cache,
                data,
                expected_sources.get(scene) if expected_sources is not None else None,
            )
            self.source_manifest_hits[scene] = hit
            encoded = json.dumps(
                dict(lineage=runtime.lineage, sources=sources), sort_keys=True
            )
            self.fingerprints[scene] = hashlib.sha256(encoded.encode()).hexdigest()
            self.source_hashes[scene] = sources

    def key(self, token: str) -> str:
        """key绑定完整场景原始来源，时间索引和模型协议。"""
        scene, index = self.index[token]
        return f"{self.fingerprints[scene]}:{scene}:{index//self.shard_frames}"

    def generate(self, token: str) -> tuple[dict, dict]:
        """如果可安全接续原生状态就顺序推进，否则从真实场景0重建。"""
        scene, index = self.index[token]
        data = self.scenes[scene]
        start = index // self.shard_frames * self.shard_frames
        end = min(start + self.shard_frames, len(data.rows))
        initial = (
            start
            if self.runtime.state_scene == scene
            and self.runtime.state_index == start - 1
            else 0
        )
        tensors = {}
        traces = []
        began = time.perf_counter()
        for i in range(initial, end):
            bev, _pred, trace = self.runtime.forward(data, i)
            if i >= start:
                tensors[data.rows[i]["sample_token"]] = bev
                traces.append(trace)
        metadata = dict(
            self.runtime.lineage,
            regenerable=True,
            scene_prefix=[r["sample_token"] for r in data.rows[:end]],
            frames=[
                dict(
                    sample_token=r["sample_token"],
                    scene_token=scene,
                    timestamp=r["timestamp"],
                    source=r["media"],
                )
                for r in data.rows[start:end]
            ],
            scene_source_sha256=self.fingerprints[scene],
            raw_sources_sha256=self.source_hashes[scene],
            traces=traces,
        )
        self.metrics.append(
            dict(
                scene=scene,
                start=start,
                end=end,
                raw_forwards=end - initial,
                seconds=time.perf_counter() - began,
            )
        )
        return tensors, metadata

    def get(self, token: str) -> torch.Tensor:
        """读写容量和完整性由RollingCache统一执行。"""
        key = self.key(token)
        if self.resident_key != key:
            self.resident = {}
            if self.admit_disk:
                self.resident = self.cache.fetch(key, lambda: self.generate(token))
            else:
                self.resident, metadata = self.generate(token)
                self.cache.record_ephemeral(key, self.resident, metadata)
            self.resident_key = key
        return self.resident[token]

    def peek(self, token: str) -> torch.Tensor | None:
        """只看当前RAM块，不触发I/O或原生时序前向，允许安全预拷贝。"""
        return self.resident.get(token)

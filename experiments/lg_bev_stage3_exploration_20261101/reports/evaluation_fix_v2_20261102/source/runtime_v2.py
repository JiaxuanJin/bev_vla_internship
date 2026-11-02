"""授权后的原生模型和数据接入；本模块导入不初始化GPU或读取开发GT。"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import random
import socket
import sys
import tempfile

import numpy as np
import torch

from core import ROOT, SHARED, freeze, no_symlinks, read, require, selected_rows, sha


def install_write_fence(root):
    """子进程审计钩子拒绝根目录外写入；第三方缓存同样受限。"""
    root = no_symlinks(root)

    def allowed(path, dir_fd=None):
        if isinstance(path, int) or path is None:
            return
        path = Path(os.fsdecode(path))
        if not path.is_absolute() and dir_fd not in (None, -1):
            anchor = Path(os.readlink(f"/proc/self/fd/{dir_fd}"))
            require(anchor.is_absolute(), "目录描述符没有可核验的绝对路径")
            require(anchor.is_dir(), "描述符没有指向存活目录")
            path = anchor / path
        path = no_symlinks(path)
        require(
            path.is_relative_to(root) and ".." not in path.parts,
            "拒绝实验外写入：" + str(path),
        )

    def hook(event, args):
        if event == "open":
            path, mode, flags = args
            writing = any(c in (mode or "") for c in "wax+") or flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            )
            if writing:
                allowed(path)
        elif event in ("os.remove", "os.rmdir"):
            allowed(args[0], args[1])
        elif event in ("os.mkdir", "os.chmod"):
            allowed(args[0], args[2])
        elif event == "os.utime":
            allowed(args[0], args[3])
        elif event in ("os.rename", "os.link"):
            allowed(args[0], args[2])
            allowed(args[1], args[3])
        elif event == "os.symlink":
            raise PermissionError("实验禁止创建符号链接")
        elif event == "sqlite3.connect":
            database = os.fsdecode(args[0])
            if database != ":memory:":
                allowed(database)

    sys.addaudithook(hook)


def environment(store):
    """所有缓存写到新根；禁止网络，禁用动态CUDA编译扩展。"""
    for key in (
        "TMPDIR",
        "TEMP",
        "TMP",
        "HF_HOME",
        "TORCH_HOME",
        "XDG_CACHE_HOME",
        "TORCH_EXTENSIONS_DIR",
        "MPLCONFIGDIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        path = store.path("runtime/" + key.lower())
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    tempfile.tempdir = None
    torch.set_num_threads(2)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        WANDB_MODE="offline",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        LG_BEV_DYNAMIC_CUDA="0",
        LG_BEV_UPSAMPLE_CUDA="0",
        LG_BEV_FIXED_CUDA="1",
        LG_BEV_CPU_DIRECT="1",
    )
    original = socket.socket.connect

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise PermissionError("探索实验仅使用本地资产")
        return original(sock, address)

    socket.socket.connect = connect
    install_write_fence(store.root)


def private_scene_module(store):
    """单独加载已封存SceneData，重绑定其私有写入依赖，不修改原模块。"""
    path = ROOT / "scripts/stage3/lg_bev_engineering_data.py"
    name = "_lg_bev_exploration_scene_private"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    from metadata_context import velocity_subset_nuscenes_class

    module.subset_nuscenes_class = velocity_subset_nuscenes_class
    module.CACHE_ROOT = store.path("cache/evaluation_fix_v2_20261102")
    module.RollingCache = lambda: store

    def atomic_save(path, value):
        return store.tensor(str(Path(path).relative_to(store.root)), value)

    module.atomic_save = atomic_save
    module.CACHE_ROOT.joinpath("metadata").mkdir(parents=True, exist_ok=True)
    return module


class Runtime:
    """一帧/一个场景驻留；顺序原生history，不写共享catalog或正式run。"""

    def __init__(self, config, store, *, development_authorized=False):
        from scripts.stage3.lg_bev_engineering_native import NativeRuntime
        from scripts.stage3.lg_bev_engineering_data import LabelStore
        from scripts.stage2.language_encoder import DistilBERTLanguageEncoder
        from scripts.stage2.spatial_v3_model import InstanceReadout

        self.config, self.store = config, store
        self.device = torch.device("cuda")
        self.development_authorized = development_authorized
        self.module = private_scene_module(store)
        self.native = NativeRuntime(fast_images=True)
        self.model = freeze(self.native.model)
        self.encoder = freeze(
            InstanceReadout(
                DistilBERTLanguageEncoder(
                    str(ROOT / "models/distilbert-base-uncased"), local_files_only=True
                )
            ).cuda()
        )
        self.labels = LabelStore(
            read(SHARED / "runs/formal_gt_20261027_v1/data_report.json")
        )
        self.rows = {}
        for role in ("train", "diagnostic"):
            self.rows.update(
                {r["sample_token"]: r for r in selected_rows(config, role)}
            )
        if development_authorized:
            self.rows.update(
                {r["sample_token"]: r for r in selected_rows(config, "development")}
            )
        self.scene = None
        self.data = None
        self.current_token = None
        self.current_bev = None
        self.current_prediction = None

    def _scene(self, token):
        require(token in self.rows, "访问未授权帧")
        row = self.rows[token]
        require(row["split"] == "train" or self.development_authorized, "禁止开发GT")
        if self.scene != row["scene_token"]:
            rows = sorted(
                [
                    r
                    for r in self.rows.values()
                    if r["scene_token"] == row["scene_token"]
                ],
                key=lambda r: r["timestamp"],
            )
            # 只读复用通过摘要验证的训练索引，不连接共享catalog。
            source = SHARED / "metadata" / (row["scene_token"] + ".pt")
            relative = (
                "cache/evaluation_fix_v2_20261102/metadata/"
                + row["scene_token"]
                + ".pt"
            )
            if (
                row["split"] == "train"
                and source.exists()
                and not self.store.path(relative).exists()
            ):
                digest = source.with_suffix(".sha256").read_text().strip()
                require(sha(source) == digest, "既有训练索引损坏")
                self.store.bytes(relative, source.read_bytes())
                self.store.bytes(relative[:-3] + ".sha256", digest.encode())
            py, np_state, cpu = (
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
            )
            cuda = torch.cuda.get_rng_state_all()
            try:
                self.data = self.module.SceneData(rows, allowed_split=row["split"])
                from scripts.stage3.lg_bev_engineering_native import (
                    verified_scene_sources,
                )

                expected = None
                if row["split"] == "train":
                    expected = {}
                    for selected in rows:
                        expected.update(
                            self.labels.get(selected["sample_token"])["source_hashes"]
                        )
                verified_scene_sources(self.store, self.data, expected)
            finally:
                random.setstate(py)
                np.random.set_state(np_state)
                torch.set_rng_state(cpu)
                torch.cuda.set_rng_state_all(cuda)
            self.scene = row["scene_token"]
            self.native.reset()
            self.current_token = None
        return row

    def get_label(self, token):
        row = self._scene(token)
        if row["split"] == "train":
            return self.labels.get(token)
        require(self.development_authorized, "禁止开发GT")
        relative = (
            "cache/evaluation_fix_v2_20261102/development_labels/" + token + ".pt"
        )
        path = self.store.path(relative)
        if not path.exists():
            value = self.data.label(token)
            self.store.tensor(relative, value)
            self.store.json(relative + ".sha.json", {"sha256": sha(path)})
        require(sha(path) == read(str(path) + ".sha.json")["sha256"], "开发标签损坏")
        return torch.load(path, map_location="cpu", weights_only=False)

    def get(self, token):
        """仅重建必要前缀；同帧多组共享BEV，不保存多份大型磁盘缓存。"""
        self._scene(token)
        if self.current_token == token:
            return self.current_bev
        index = self.data.index[token]
        start = (
            self.native.state_index + 1
            if (
                self.native.state_scene == self.scene
                and self.native.state_index < index
            )
            else 0
        )
        # 场景构建与原生前向不得改变Adapter训练随机序列。
        py, np_state, cpu = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
        )
        cuda = torch.cuda.get_rng_state_all()
        try:
            for position in range(start, index + 1):
                bev, prediction, trace = self.native.forward(self.data, position)
            self.current_token, self.current_bev = token, bev
            self.current_prediction = prediction
            self.last_trace = trace
        finally:
            random.setstate(py)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state_all(cuda)
        return self.current_bev

    def validate_label(self, token):
        """锁定几何/实例表，投影越界时停止，不自动变更GT。"""
        from scripts.stage2.spatial_annotations import metric_grid

        label = self.get_label(token)
        require(label["gt"].shape == (200, 400), "地图GT尺寸错误")
        require(label["object_masks"].shape[-2:] == (150, 150), "对象GT尺寸错误")
        require(label["map_masks"].shape[-2:] == (200, 400), "实例地图尺寸错误")
        require(bool((metric_grid(label["frame"])[0].abs() < 1).all()), "地图投影越界")
        for query in label["queries"]:
            ids = label["annotation_tokens" if query["task"] == "object" else "map_ids"]
            require(
                query["target_instance_ids"] == [ids[i] for i in query["selected"]],
                "目标实例ID与索引错配",
            )
            if query["pair_id"] is not None:
                paired = [
                    q
                    for q in label["queries"]
                    if q["instruction_id"] == query["pair_id"]
                ]
                require(
                    len(paired) == 1 and paired[0]["text"] == query["wrong_text"],
                    "错误指令配对身份不符",
                )
        return label


class TrainLabels:
    """向原数学Trainer提供仅读取的已授权train标签。"""

    def __init__(self, runtime):
        self.runtime = runtime

    def get(self, token):
        require(self.runtime.rows[token]["split"] == "train", "训练读取非train")
        return self.runtime.validate_label(token)

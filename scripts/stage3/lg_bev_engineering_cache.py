"""G盘独立FP32分片缓存：归属、容量、原子提交、校验、pin及崩溃恢复。"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import tempfile
from typing import Callable, Iterator
import uuid

import torch
from torch import Tensor

from scripts.stage3.lg_bev_common import sha

CACHE_ROOT = Path("/mnt/g/BEV_VLA_Datasets/lg_bev_rolling_cache")
MAX_BYTES = 30_000_000_000
RESERVE_BYTES = 250_000_000_000
OWNER = "lg_bev_rolling_cache_v1_20261027"


def contained(root: Path, relative: str) -> Path:
    """规范化且拒绝符号链接、越界路径；只操作明确属于本管理器的文件。"""
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("拒绝越界路径")
    path = root / relative
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise ValueError("缓存路径不允许符号链接")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("路径逃逸缓存根目录")
    return path


def atomic_save(path: Path, value: object) -> None:
    """在同目录写临时文件并fsync后rename；已存在的正式文件拒绝覆盖。"""
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    try:
        with temporary.open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # 单写进程持有外层锁；rename前再次拒绝覆盖。
        if path.exists():
            raise FileExistsError(path)
        os.rename(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class RollingCache:
    """30GB/250GB双门槛的归属分片缓存，重建只依赖原生场景起点。

    Args:
        root: 正式运行必须为指定G盘根目录；测试允许独立临时目录。
        capacity: 仅允许不超过已授权30GB，不能自动扩容。
        reserve: 正式G盘至少250GB；测试可显式设置0。
        test_root: 仅小型单测使用，不应在生产CLI开放。
    """

    def __init__(
        self,
        root: Path = CACHE_ROOT,
        capacity: int = MAX_BYTES,
        reserve: int = RESERVE_BYTES,
        *,
        test_root: bool = False,
    ) -> None:
        self.root = root.absolute()
        if not test_root and self.root != CACHE_ROOT:
            raise ValueError("正式缓存只能位于已授权G盘目录")
        if not 0 < capacity <= MAX_BYTES:
            raise ValueError("容量超过授权上限或无效")
        if not test_root and reserve < RESERVE_BYTES:
            raise ValueError("不能降低250GB剩余空间门槛")
        self.capacity, self.reserve = capacity, reserve
        self.expected_shape = (1, 256, 150, 150)
        self.root.mkdir(parents=True, exist_ok=True)
        if not test_root:
            (self.root / "tmp").mkdir(exist_ok=True)
            for key in ("TMPDIR", "TEMP", "TMP"):
                os.environ[key] = str(self.root / "tmp")
            tempfile.tempdir = str(self.root / "tmp")
        contained(self.root, "owner.json")
        owner_path = self.root / "owner.json"
        if not owner_path.exists():
            with owner_path.open("x") as stream:
                json.dump(
                    dict(owner=OWNER, max_bytes=MAX_BYTES, reserve_bytes=RESERVE_BYTES),
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
        ownership = json.loads(owner_path.read_text())
        if ownership["owner"] != OWNER:
            raise ValueError("已有目录不属于LG-BEV缓存管理器")
        self.capacity = min(self.capacity, ownership.get("capacity_bytes", MAX_BYTES))
        self.lock_path = contained(self.root, "cache.lock")
        self.db_path = contained(self.root, "catalog.sqlite3")
        self.db = sqlite3.connect(self.db_path, timeout=60)
        self.db.execute("PRAGMA temp_store=MEMORY")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS shards "
            "(key TEXT PRIMARY KEY,path TEXT UNIQUE,digest TEXT,bytes INTEGER,"
            "metadata TEXT,access REAL,pins INTEGER DEFAULT 0,state TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events "
            "(time REAL,event TEXT,key TEXT,detail TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS leases "
            "(key TEXT,pid INTEGER,nonce TEXT PRIMARY KEY)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS ephemeral "
            "(key TEXT PRIMARY KEY,metadata TEXT,access REAL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS labels "
            "(run TEXT,token TEXT,record TEXT,PRIMARY KEY(run,token))"
        )
        self.db.commit()

    @contextmanager
    def locked(self) -> Iterator[None]:
        """跨进程排他锁覆盖容量判断、写入和淘汰。"""
        with self.lock_path.open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def event(self, event: str, key: str, detail: str = "") -> None:
        """在同一G盘数据库记录管理事件。"""
        self.db.execute(
            "INSERT INTO events VALUES(?,?,?,?)", (time.time(), event, key, detail)
        )
        self.db.commit()

    def configure_capacity(
        self, capacity: int, *, user_requested: bool = False
    ) -> None:
        """显式配置不超过30GB；所有后续进程读取持久化上限，不能自动扩容。"""
        if not 0 < capacity <= MAX_BYTES:
            raise ValueError("超出授权30GB")
        with self.locked():
            owner_path = contained(self.root, "owner.json")
            owner = json.loads(owner_path.read_text())
            previous = owner.get("capacity_bytes", MAX_BYTES)
            if capacity > previous and not user_requested:
                raise ValueError("扩容需要用户显式配置")
            old_limit = self.capacity
            self.capacity = capacity
            try:
                self.reserve_for_artifact(4096)
            except Exception:
                self.capacity = old_limit
                raise
            pending = owner_path.with_name("owner.json.new-" + uuid.uuid4().hex)
            with pending.open("x") as stream:
                json.dump({**owner, "capacity_bytes": capacity}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, owner_path)
            self.event(
                "capacity", "owner", f"{previous}->{capacity};explicit={user_requested}"
            )

    def usage(self) -> int:
        """把根目录内标签、checkpoint、参考缓存及临时写入也纳入30GB。"""
        total = 0
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            for name in dirs + files:
                if (Path(directory) / name).is_symlink():
                    raise ValueError("缓存目录出现符号链接，停止写入")
            for name in files:
                try:
                    total += (Path(directory) / name).stat().st_size
                except FileNotFoundError:
                    pass
        return total

    def admit(self, incoming: int) -> None:
        """任何新增大型数据前检查实际磁盘，不删除GT或历史文件腾空间。"""
        if shutil.disk_usage(self.root).free - incoming < self.reserve:
            raise OSError("G盘剩余空间将低于250GB，停止新增写入")
        if self.usage() + incoming > self.capacity:
            raise OSError("缓存容量不足且无安全可淘汰分片")

    def _evict_for(self, incoming: int, protected: str = "") -> None:
        while self.usage() + incoming > self.capacity:
            row = self.db.execute(
                "SELECT key,path,digest,metadata FROM shards WHERE pins=0 "
                "AND state='verified' AND key!=? ORDER BY access LIMIT 1",
                (protected,),
            ).fetchone()
            if row is None:
                raise OSError("全部分片被pin或容量用于不可淘汰数据，停止写入")
            key, relative, digest, encoded = row
            metadata = json.loads(encoded)
            if not metadata.get("regenerable") or not metadata.get("scene_prefix"):
                raise RuntimeError("分片无法安全重建，不允许淘汰")
            path = contained(self.root, relative)
            if not relative.startswith("shards/") or not path.is_file():
                raise RuntimeError("归属目录或文件异常，拒绝清理")
            if sha(path) != digest:
                self.db.execute("UPDATE shards SET state='corrupt' WHERE key=?", (key,))
                self.event("corrupt", key, "淘汰前摘要不符；保留损坏文件")
                continue
            path.unlink()
            self.db.execute("UPDATE shards SET state='evicted' WHERE key=?", (key,))
            self.event("evict", key)

    def reserve_for_artifact(self, incoming: int, protected: str = "") -> None:
        """调用方持锁，新增GT/checkpoint也只能安全淘汰BEV，不删除其他文件。"""
        if shutil.disk_usage(self.root).free - incoming < self.reserve:
            raise OSError("G盘剩余空间将低于250GB，停止新增写入")
        self._evict_for(incoming, protected)
        self.admit(incoming)

    def put(self, key: str, values: dict[str, Tensor], metadata: dict) -> dict:
        """只接收原始FP32张量，提交前后校验；metadata绑定来源和完整前缀。"""
        if (
            not values
            or not metadata.get("scene_prefix")
            or not metadata.get("regenerable")
        ):
            raise ValueError("缺少可重建场景前缀")
        for value in values.values():
            if value.dtype != torch.float32 or value.shape != self.expected_shape:
                raise ValueError("必须为真实原生FP32 BEV [1,256,150,150]")
            if not torch.isfinite(value).all():
                raise ValueError("BEV包含非有限值")
        for required in (
            "checkpoint_sha256",
            "config_sha256",
            "preprocessing",
            "frames",
            "source_dtype",
            "history_policy",
        ):
            if required not in metadata:
                raise ValueError(f"缺少分片来源字段:{required}")
        if metadata["source_dtype"] != "torch.float32":
            raise ValueError("拒绝FP16回转的伪FP32缓存")
        with self.locked():
            row = self.db.execute(
                "SELECT path,digest,state FROM shards WHERE key=?", (key,)
            ).fetchone()
            if row and row[2] == "verified":
                if sha(contained(self.root, row[0])) == row[1]:
                    return dict(hit=True, digest=row[1])
                self.db.execute("UPDATE shards SET state='corrupt' WHERE key=?", (key,))
                self.event("corrupt", key, "读取前摘要不符")
            size = sum(v.numel() * v.element_size() for v in values.values()) + 2**20
            self.reserve_for_artifact(size, key)
            relative = (
                "shards/"
                + hashlib.sha256(key.encode()).hexdigest()
                + "-"
                + uuid.uuid4().hex
                + ".pt"
            )
            metadata = {
                **metadata,
                "tensor_sha256": {
                    t: hashlib.sha256(
                        v.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest()
                    for t, v in values.items()
                },
            }
            # pending归属先登记：崩溃后仅恢复此登记文件，绝不清理未归属路径。
            self.db.execute(
                "INSERT OR REPLACE INTO shards VALUES(?,?,?,?,?,?,?,?)",
                (
                    key,
                    relative,
                    "",
                    size,
                    json.dumps(metadata),
                    time.time(),
                    0,
                    "pending",
                ),
            )
            self.db.commit()
            path = contained(self.root, relative)
            atomic_save(
                path, {t: v.detach().cpu().contiguous() for t, v in values.items()}
            )
            digest = sha(path)
            check = torch.load(path, map_location="cpu", weights_only=True)
            if any(not torch.equal(check[t], v.cpu()) for t, v in values.items()):
                raise RuntimeError("分片写入校验失败")
            self.db.execute(
                "UPDATE shards SET digest=?,bytes=?,state='verified' WHERE key=?",
                (digest, path.stat().st_size, key),
            )
            self.event("commit", key)
            return dict(hit=False, digest=digest, bytes=path.stat().st_size)

    def get(self, key: str) -> dict[str, Tensor] | None:
        """读取前做文件和逐tensor摘要校验；损坏返回miss并保留证据。"""
        with self.locked():
            row = self.db.execute(
                "SELECT path,digest,metadata,state FROM shards WHERE key=?", (key,)
            ).fetchone()
            if not row or row[3] != "verified":
                return None
            path = contained(self.root, row[0])
            if not path.is_file() or sha(path) != row[1]:
                self.db.execute("UPDATE shards SET state='corrupt' WHERE key=?", (key,))
                self.event("corrupt", key, "file digest")
                return None
            values = torch.load(path, map_location="cpu", weights_only=True)
            expected = json.loads(row[2])["tensor_sha256"]
            for token, value in values.items():
                digest = hashlib.sha256(
                    value.contiguous().numpy().tobytes()
                ).hexdigest()
                if value.dtype != torch.float32 or digest != expected[token]:
                    raise RuntimeError("分片内部tensor摘要不符")
            self.db.execute(
                "UPDATE shards SET access=? WHERE key=?", (time.time(), key)
            )
            self.db.commit()
            return values

    @contextmanager
    def pin(self, key: str) -> Iterator[None]:
        """活跃分片不允许被淘汰；checkpoint不依赖cache存在来恢复。"""
        nonce = uuid.uuid4().hex
        with self.locked():
            self.db.execute("UPDATE shards SET pins=pins+1 WHERE key=?", (key,))
            self.db.execute(
                "INSERT INTO leases VALUES(?,?,?)", (key, os.getpid(), nonce)
            )
            self.db.commit()
        try:
            yield
        finally:
            with self.locked():
                self.db.execute(
                    "UPDATE shards SET pins=MAX(0,pins-1) WHERE key=?", (key,)
                )
                self.db.execute("DELETE FROM leases WHERE nonce=?", (nonce,))
                self.db.commit()

    def recover(self) -> dict:
        """恢复已登记pending分片，不扫描删除未知文件；pin保留供显式会话审查。"""
        count = 0
        with self.locked():
            for key, pid, nonce in self.db.execute(
                "SELECT key,pid,nonce FROM leases"
            ).fetchall():
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    self.db.execute(
                        "UPDATE shards SET pins=MAX(0,pins-1) WHERE key=?", (key,)
                    )
                    self.db.execute("DELETE FROM leases WHERE nonce=?", (nonce,))
                except PermissionError:
                    pass
            for key, relative, encoded in self.db.execute(
                "SELECT key,path,metadata FROM shards WHERE state='pending'"
            ).fetchall():
                path = contained(self.root, relative)
                pending = path.with_name(path.name + ".pending")
                candidate = path if path.is_file() else pending
                if candidate.is_file():
                    try:
                        values = torch.load(
                            candidate, map_location="cpu", weights_only=True
                        )
                        expected = json.loads(encoded)["tensor_sha256"]
                        valid = set(values) == set(expected) and all(
                            v.dtype == torch.float32
                            and hashlib.sha256(v.numpy().tobytes()).hexdigest()
                            == expected[t]
                            for t, v in values.items()
                        )
                    except Exception:
                        valid = False
                    state = "verified" if valid else "corrupt"
                    if valid and candidate == pending:
                        os.rename(pending, path)
                    self.db.execute(
                        "UPDATE shards SET state=?,digest=?,bytes=? WHERE key=?",
                        (
                            state,
                            sha(path if valid else candidate),
                            (
                                candidate.stat().st_size
                                if not valid
                                else path.stat().st_size
                            ),
                            key,
                        ),
                    )
                else:
                    self.db.execute(
                        "UPDATE shards SET state='evicted' WHERE key=?", (key,)
                    )
                count += 1
            self.db.commit()
        return dict(
            recovered=count,
            usage_bytes=self.usage(),
            capacity_bytes=self.capacity,
            free_bytes=shutil.disk_usage(self.root).free,
        )

    def fetch(self, key: str, generate: Callable[[], tuple[dict, dict]]) -> dict:
        """miss时调用确定性原生场景前缀生成器；不复用错误的history。"""
        value = self.get(key)
        if value is None:
            tensors, metadata = generate()
            self.put(key, tensors, metadata)
            value = self.get(key)
        if value is None:
            raise RuntimeError("缓存提交后仍不可读")
        return value

    def record_ephemeral(
        self, key: str, values: dict[str, Tensor], metadata: dict
    ) -> None:
        """流式冷场景只持有一个RAM分片，G盘仍登记完整FP32来源和逐tensor校验值。"""
        if metadata["source_dtype"] != "torch.float32" or not metadata.get(
            "regenerable"
        ):
            raise ValueError("RAM分片也必须是真实可再生FP32")
        metadata = {
            **metadata,
            "storage": "RAM_ONLY_REGENERABLE_SCENE_PREFIX",
            "tensor_sha256": {
                token: hashlib.sha256(value.numpy().tobytes()).hexdigest()
                for token, value in values.items()
            },
        }
        encoded = json.dumps(metadata)
        with self.locked():
            self.reserve_for_artifact(len(encoded.encode()) + 65536)
            self.db.execute(
                "INSERT OR REPLACE INTO ephemeral VALUES(?,?,?)",
                (key, encoded, time.time()),
            )
            self.event("ephemeral", key)

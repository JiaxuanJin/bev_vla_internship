"""真实FP32分片的fsync后进程退出、恢复、损坏识别与原生重建验收。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from scripts.stage3.lg_bev_common import write
from scripts.stage3.lg_bev_engineering_cache import RollingCache, contained


def audit_recovery(
    cache: RollingCache, provider, reference, token: str, run: Path
) -> dict:
    """子进程在原子rename前硬退出，恢复不依赖内存张量或未提交catalog指针。"""
    source = provider.get(token)
    row = cache.db.execute(
        "SELECT metadata FROM shards WHERE key=?", (provider.key(token),)
    ).fetchone()
    metadata = json.loads(row[0])
    metadata["frames"] = metadata["frames"][:1]
    metadata["scene_prefix"] = metadata["scene_prefix"][:1]
    recipe = run / "fault_recipe.json"
    key = "fault_recovery:" + run.name
    write(
        recipe,
        dict(
            reference=reference.lookup[token]["path"],
            token=token,
            metadata=metadata,
            key=key,
            capacity=cache.capacity,
        ),
    )
    code = """
import json,os,sys,torch
from pathlib import Path
from scripts.stage3.lg_bev_engineering_cache import RollingCache
import scripts.stage3.lg_bev_engineering_cache as module
r=json.loads(Path(sys.argv[1]).read_text())
values=torch.load(r['reference'],map_location='cpu',weights_only=True)
cache=RollingCache(capacity=r['capacity'])
def terminate(*args): os._exit(73)
module.os.rename=terminate
cache.put(r['key'],{r['token']:values[r['token']]},r['metadata'])
raise RuntimeError('fault was not triggered')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(recipe)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 73:
        raise RuntimeError(f"故障注入未按预期退出:{result.returncode}:{result.stderr}")
    restored = cache.recover()
    recovered = cache.get(key)
    if recovered is None or not torch.equal(recovered[token], source):
        raise RuntimeError("fsync后硬退出恢复失败")
    relative = cache.db.execute(
        "SELECT path FROM shards WHERE key=?", (key,)
    ).fetchone()[0]
    path = contained(cache.root, relative)
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        last = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([last[0] ^ 1]))
        stream.flush()
    if cache.get(key) is not None:
        raise RuntimeError("损坏缓存未被识别")
    began = time.perf_counter()
    value = cache.fetch(key, lambda: provider.generate(token))[token]
    elapsed = time.perf_counter() - began
    if not torch.equal(value, reference.get(token)):
        raise RuntimeError("损坏后原生重建不一致")
    return dict(
        status="PASS",
        child_exit_code=73,
        exit_point="after_file_fsync_before_rename",
        recovered_pending=restored["recovered"],
        actual_FP32_frame_shape=list(source.shape),
        recovered_bitwise=True,
        corruption_recognized=True,
        corrupt_owned_file_preserved=path.is_file(),
        regenerated_via_native_scene_prefix=True,
        regenerated_bitwise=True,
        regeneration_seconds=elapsed,
    )

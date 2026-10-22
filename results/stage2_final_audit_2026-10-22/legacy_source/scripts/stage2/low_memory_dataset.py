"""Read original nuScenes tables in bounded chunks for single-frame micro runs.

Only the process-local dataset module's constructor is temporarily substituted.
No files, annotations, map geometry, record fields or author pipelines are edited.
This subset is deliberately unsuitable for temporal traversal or official eval.
"""

from contextlib import contextmanager
import importlib
import json
from pathlib import Path

from .runtime_memory import memory_event


def json_records(path, chunk_size=2**20):
    """Stream a top-level array of objects, including pretty-printed JSON."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding="utf-8") as stream:
        buffer, pos, eof = "", 0, False

        def refill():
            nonlocal buffer, pos, eof
            chunk = stream.read(chunk_size)
            buffer, pos = buffer[pos:] + chunk, 0
            eof = not chunk

        def skip_space():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    return
                refill()

        refill()
        skip_space()
        if pos == len(buffer) or buffer[pos] != "[":
            raise ValueError(f"Expected JSON array: {path}")
        pos += 1
        first = True
        while True:
            skip_space()
            if pos < len(buffer) and buffer[pos] == "]":
                pos += 1
                if (buffer[pos:] + stream.read()).strip():
                    raise ValueError(f"Trailing JSON content: {path}")
                return
            if not first:
                if pos == len(buffer) or buffer[pos] != ",":
                    raise ValueError(f"Missing JSON comma: {path}")
                pos += 1
                skip_space()
            while True:
                try:
                    record, end = decoder.raw_decode(buffer, pos)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    refill()
            if not isinstance(record, dict):
                raise ValueError("Metadata rows must be objects")
            yield record
            pos, first = end, False


def subset_nuscenes_class(base, tokens):
    class SingleFrameNuScenes(base):
        def __init__(self, version, dataroot, **kwargs):
            root = Path(dataroot) / version
            memory_event("metadata.streaming.before")
            selected_data = [
                r
                for r in json_records(root / "sample_data.json")
                if r["sample_token"] in tokens and r["is_key_frame"]
            ]
            selected_ann = [
                r
                for r in json_records(root / "sample_annotation.json")
                if r["sample_token"] in tokens
            ]
            self._selected = {
                "sample_data": selected_data,
                "sample_annotation": selected_ann,
            }
            self._wanted = {
                "sample": tokens,
                "ego_pose": {r["ego_pose_token"] for r in selected_data},
                "calibrated_sensor": {
                    r["calibrated_sensor_token"] for r in selected_data
                },
                "instance": {r["instance_token"] for r in selected_ann},
            }
            super().__init__(version=version, dataroot=dataroot, **kwargs)
            if {r["token"] for r in self.sample} != tokens:
                raise ValueError(
                    "Requested dataset tokens missing from original metadata"
                )
            del self._selected, self._wanted
            memory_event("metadata.streaming.after")

        def __load_table__(self, table_name):
            if table_name in self._selected:
                return self._selected[table_name]
            wanted = self._wanted.get(table_name)
            return [
                r
                for r in json_records(Path(self.table_root) / (table_name + ".json"))
                if wanted is None or r["token"] in wanted
            ]

    return SingleFrameNuScenes


@contextmanager
def bounded_metadata(tokens):
    module = importlib.import_module(
        "projects.mmdet3d_plugin.datasets.nuscenes_dataset"
    )
    original = module.NuScenes
    module.NuScenes = subset_nuscenes_class(original, set(tokens))
    try:
        yield
    finally:
        module.NuScenes = original

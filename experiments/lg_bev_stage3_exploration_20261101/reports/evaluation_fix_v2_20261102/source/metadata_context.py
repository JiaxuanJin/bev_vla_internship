"""仅补齐评分帧速度所需的一阶标注及时间戳，不改变官方 SDK 算法。"""

from pathlib import Path

from scripts.stage2.low_memory_dataset import json_records


def velocity_subset_nuscenes_class(base, tokens):
    """上下文仅供 box_velocity 使用；评分清单仍由调用方独立固定。"""
    requested = set(tokens)
    if not requested:
        raise ValueError("评分帧清单不能为空")

    class VelocityContextNuScenes(base):
        """保留原始 prev/next，显式检查依赖，禁止补零和丢帧。"""

        def __init__(self, version, dataroot, **kwargs):
            root = Path(dataroot) / version
            annotations = [
                row
                for row in json_records(root / "sample_annotation.json")
                if row["sample_token"] in requested
            ]
            scoring = {row["token"]: row for row in annotations}
            wanted = {
                row[key]
                for row in annotations
                for key in ("prev", "next")
                if row[key] and row[key] not in scoring
            }
            context = (
                [
                    row
                    for row in json_records(root / "sample_annotation.json")
                    if row["token"] in wanted
                ]
                if wanted
                else []
            )
            if {row["token"] for row in context} != wanted:
                raise ValueError("原始相邻标注依赖不完整，拒绝构造速度上下文")
            all_annotations = {**scoring, **{row["token"]: row for row in context}}
            sample_tokens = requested | {row["sample_token"] for row in context}
            samples = [
                row
                for row in json_records(root / "sample.json")
                if row["token"] in sample_tokens
            ]
            sample_index = {row["token"]: row for row in samples}
            if set(sample_index) != sample_tokens:
                raise ValueError("相邻标注缺少真实 sample 时间戳")
            scenes = {sample_index[token]["scene_token"] for token in requested}
            if len(scenes) != 1:
                raise ValueError("一次只允许一个已授权场景的速度上下文")
            if any(row["scene_token"] not in scenes for row in samples):
                raise ValueError("相邻标注跨越授权场景，停止读取")
            for row in annotations:
                for key, reciprocal in (("prev", "next"), ("next", "prev")):
                    if not row[key]:
                        continue
                    neighbor = all_annotations[row[key]]
                    if (
                        neighbor["instance_token"] != row["instance_token"]
                        or neighbor[reciprocal] != row["token"]
                    ):
                        raise ValueError("原始实例链接不互反或实例身份不一致")
            data = [
                row
                for row in json_records(root / "sample_data.json")
                if row["sample_token"] in requested and row["is_key_frame"]
            ]
            self._selected = {
                "sample": samples,
                "sample_annotation": annotations + context,
                "sample_data": data,
            }
            self._wanted = {
                "ego_pose": {row["ego_pose_token"] for row in data},
                "calibrated_sensor": {row["calibrated_sensor_token"] for row in data},
                "instance": {row["instance_token"] for row in annotations + context},
            }
            self.velocity_context = {
                "requested_sample_tokens": sorted(requested),
                "context_sample_tokens": sorted(sample_tokens - requested),
                "context_annotation_tokens": sorted(wanted),
                "scoring_annotations": len(annotations),
            }
            super().__init__(version=version, dataroot=dataroot, **kwargs)
            del self._selected, self._wanted

        def __load_table__(self, table_name):
            if table_name in self._selected:
                return self._selected[table_name]
            wanted = self._wanted.get(table_name)
            return [
                row
                for row in json_records(Path(self.table_root) / (table_name + ".json"))
                if wanted is None or row["token"] in wanted
            ]

    return VelocityContextNuScenes

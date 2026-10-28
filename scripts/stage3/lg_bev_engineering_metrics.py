"""G盘流式AP事件表：有限内存、SDK完全同口径的匹配/插值及场景bootstrap。"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np

from scripts.stage3.lg_bev_engineering_cache import RollingCache


class MetricSpool:
    """每query匹配独立，confidence全局排序存G盘，避免保留数百万DetectionBox。"""

    def __init__(self, path: Path, cache: RollingCache | None = None) -> None:
        if path.exists():
            raise FileExistsError(path)
        if cache is not None and not path.resolve().is_relative_to(
            cache.root.resolve()
        ):
            raise ValueError("正式评价事件表只能写G盘")
        self.cache = cache
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA temp_store=MEMORY")
        self.db.execute(
            "CREATE TABLE counts (branch TEXT,kind TEXT,class TEXT,scene TEXT,n INTEGER)"
        )
        self.db.execute(
            "CREATE TABLE events (seq INTEGER PRIMARY KEY,branch TEXT,kind TEXT,class TEXT,scene TEXT,d REAL,score REAL,tp INTEGER)"
        )
        self.db.execute(
            "CREATE TABLE queries (branch TEXT,kind TEXT,sample TEXT,scene TEXT)"
        )
        self.db.execute(
            "CREATE INDEX event_lookup ON events(branch,kind,class,d,score DESC,seq DESC,scene,tp)"
        )
        self.db.execute(
            "CREATE INDEX count_lookup ON counts(branch,kind,class,scene,n)"
        )
        self.db.commit()

    def add(
        self, branch: str, kind: str, sample: str, scene: str, gt: list, pred: list
    ) -> None:
        """先在query内按SDK tie规则匹配，再将TP/FP存盘；空GT预测必须保留。"""
        if self.cache:
            with self.cache.locked():
                self.cache.reserve_for_artifact(2**20)
        self.db.execute(
            "INSERT INTO queries VALUES(?,?,?,?)", (branch, kind, sample, scene)
        )
        classes = sorted({b.detection_name for b in gt + pred})
        rows = []
        for cls in classes:
            truth = [b for b in gt if b.detection_name == cls]
            forecasts = [b for b in pred if b.detection_name == cls]
            self.db.execute(
                "INSERT INTO counts VALUES(?,?,?,?,?)",
                (branch, kind, cls, scene, len(truth)),
            )
            # 插入原始预测顺序，SQL score DESC,seq DESC与SDK索引倒序tie一致。
            order = sorted(
                range(len(forecasts)),
                key=lambda i: (forecasts[i].detection_score, i),
                reverse=True,
            )
            for distance in (0.5, 1.0, 2.0, 4.0):
                taken = set()
                hits = {}
                for index in order:
                    box = forecasts[index]
                    best = np.inf
                    chosen = None
                    for i, target in enumerate(truth):
                        if i in taken:
                            continue
                        delta = np.linalg.norm(
                            np.array(box.translation[:2]) - target.translation[:2]
                        )
                        if delta < best:
                            best = delta
                            chosen = i
                    match = best < distance
                    if match:
                        taken.add(chosen)
                    hits[index] = int(match)
                rows.extend(
                    (
                        branch,
                        kind,
                        cls,
                        scene,
                        distance,
                        float(b.detection_score),
                        hits[i],
                    )
                    for i, b in enumerate(forecasts)
                )
        self.db.executemany(
            "INSERT INTO events(branch,kind,class,scene,d,score,tp) VALUES(?,?,?,?,?,?,?)",
            rows,
        )
        self.db.commit()

    def curve(
        self,
        branch: str,
        kind: str,
        cls: str,
        distance: float,
        scene: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
        """只加载一个class/distance的轻量事件向量，不实例化框或全部分支。"""
        where = "branch=? AND kind=? AND class=?"
        args = [branch, kind, cls]
        if scene is not None:
            where += " AND scene=?"
            args.append(scene)
        counts = {}
        for s, n in self.db.execute(
            "SELECT scene,SUM(n) FROM counts WHERE " + where + " GROUP BY scene", args
        ):
            counts[s] = int(n)
        query = (
            "SELECT score,tp,scene FROM events WHERE "
            + where
            + " AND d=? ORDER BY score DESC,seq DESC"
        )
        n = self.db.execute(
            "SELECT COUNT(*) FROM events WHERE " + where + " AND d=?", args + [distance]
        ).fetchone()[0]
        scores = np.empty(n, dtype=np.float64)
        tp = np.empty(n, dtype=np.float64)
        scenes = []
        for i, (score, hit, s) in enumerate(self.db.execute(query, args + [distance])):
            scores[i] = score
            tp[i] = hit
            scenes.append(s)
        return scores, tp, scenes, counts

    @staticmethod
    def ap(tp: np.ndarray, npos: float, weights: np.ndarray | None = None) -> float:
        """SDK101点插值+calc_ap，权重用于固定tie顺序的场景多项式bootstrap。"""
        from nuscenes.eval.detection.algo import calc_ap

        if npos <= 0 or not tp.any():
            return 0.0
        weight = np.ones_like(tp) if weights is None else weights
        keep = weight > 0
        hits = np.cumsum((tp * weight)[keep])
        total = np.cumsum(weight[keep])
        if not keep.any():
            return 0.0
        precision = hits / total
        recall = hits / npos
        interpolated = np.interp(np.linspace(0, 1, 101), recall, precision, right=0.0)
        return float(calc_ap(SimpleNamespace(precision=interpolated), 0.1, 0.1))

    def metric(
        self, branch: str, kind: str, classes: tuple[str, ...], scene: str | None = None
    ) -> dict:
        """缺正类保持NOT_EVALUABLE，不能改变分母或称官方全val。"""
        values = {}
        missing = []
        for cls in classes:
            data = {}
            for d in (0.5, 1.0, 2.0, 4.0):
                _scores, tp, _scenes, counts = self.curve(branch, kind, cls, d, scene)
                if not sum(counts.values()):
                    missing.append(cls)
                    data = None
                    break
                data[str(d)] = self.ap(tp, sum(counts.values()))
            values[cls] = data
        return dict(
            status="NOT_EVALUABLE" if missing else "OK",
            mAP=(
                None
                if missing
                else float(np.mean([v for c in values.values() for v in c.values()]))
            ),
            class_AP=values,
            missing_positive_classes=missing,
            official_full_validation_mAP=False,
        )

    def bootstrap(
        self, branch: str, kind: str, classes: tuple[str, ...], replicates: int = 1000
    ) -> dict:
        """以scene多项式权重重采样，预测tie顺序固定；单场景拒绝伪造区间。"""
        scenes = sorted(
            r[0]
            for r in self.db.execute(
                "SELECT DISTINCT scene FROM queries WHERE branch=? AND kind=?",
                (branch, kind),
            )
        )
        if len(scenes) < 2:
            return dict(status="NOT_MEANINGFUL_ONE_SCENE", scenes=len(scenes))
        rng = np.random.RandomState(20261027)
        samples = rng.multinomial(
            len(scenes), np.full(len(scenes), 1 / len(scenes)), size=replicates
        )
        lookup = {s: i for i, s in enumerate(scenes)}
        values = np.zeros(replicates)
        valid = np.ones(replicates, bool)
        for cls in classes:
            for distance in (0.5, 1.0, 2.0, 4.0):
                _scores, tp, ids, counts = self.curve(branch, kind, cls, distance)
                positions = np.array([lookup[s] for s in ids], dtype=int)
                positives = np.array([counts.get(s, 0) for s in scenes])
                for i, multiplicity in enumerate(samples):
                    npos = float(multiplicity @ positives)
                    if not npos:
                        valid[i] = False
                        continue
                    values[i] += self.ap(tp, npos, multiplicity[positions]) / (
                        4 * len(classes)
                    )
        return dict(
            status="SCENE_CLUSTER_BOOTSTRAP",
            replicates=replicates,
            valid_replicates=int(valid.sum()),
            seed=20261027,
            percentile95=(
                np.quantile(values[valid], [0.025, 0.975]).tolist()
                if valid.any()
                else None
            ),
            method="scene multinomial weights, original confidence-tie order fixed; no prompt-level resampling",
        )

"""Move association (ARCHITECTURE §6.5): pair a removed cluster with an added cluster that looks like the same object.

For each (removed, added) pair the score combines
  * size:      the bounding boxes' dimensions agree within `dim_tol` (spec: ~15 %). Height is compared directly; the two
               horizontal sides are compared sorted, so an object turned 90 degrees still matches;
  * points:    point-count ratio (min/max): a moved object shows a similar amount of surface;
  * intensity: mean intensity difference (the same material reflects the same).
Greedy best-first matching above `min_score` gives `moved` events; unmatched clusters stay `added` / `removed`.

Viewpoint caveat: a moved object can show different faces to a fixed sensor (one side was hidden before, another is
hidden now), so the size test is a tolerance, not an equality, and height (most viewpoint-stable) weighs most.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Moved:
    removed: object                  # geo_diff.Cluster (baseline points)
    added: object                    # geo_diff.Cluster (current points)
    score: float
    parts: dict

    @property
    def displacement(self) -> np.ndarray:
        return self.added.centroid - self.removed.centroid

    def as_dict(self) -> dict:
        return {"score": round(self.score, 3), "parts": {k: round(v, 3) for k, v in self.parts.items()},
                "from": [round(float(v), 4) for v in (self.removed.bbox_min + self.removed.bbox_max) / 2],
                "to": [round(float(v), 4) for v in (self.added.bbox_min + self.added.bbox_max) / 2],
                "distance_m": round(float(np.linalg.norm(self.displacement[:2])), 4)}


@dataclass
class Association:
    moved: list = field(default_factory=list)
    added: list = field(default_factory=list)       # unmatched
    removed: list = field(default_factory=list)     # unmatched


def _size_score(a_size: np.ndarray, b_size: np.ndarray, tol: float) -> float:
    """1 when the sizes match, falling linearly to 0 at `tol` relative difference (per dimension, worst one wins)."""
    ah, bh = a_size[2], b_size[2]
    a_xy, b_xy = np.sort(a_size[:2]), np.sort(b_size[:2])
    rel = [abs(ah - bh) / max(ah, bh, 1e-6)] + [abs(x - y) / max(x, y, 1e-6) for x, y in zip(a_xy, b_xy)]
    # height is the most viewpoint-stable dimension: full weight; the footprint gets twice the tolerance
    worst = max(rel[0] / tol, rel[1] / (2 * tol), rel[2] / (2 * tol))
    return float(max(0.0, 1.0 - worst))


def _robust_size_score(r, a, tol: float, abs_tol: float) -> float:
    """Top height + footprint, when the clusters carry them (geo_diff does). A change cluster's z EXTENT is not the
    object's height: the bottom ~tau of any object sits within tau of the old table points and is cut off (a 6 cm crate
    left 2.9 cm before and 2.3 cm after a move: 21 %). Its top height (median z of the cluster) and FOOTPRINT do not.
    Each comparison allows max(relative tolerance, abs_tol): on a 6 cm object 15 % is 9 mm, finer than the sensor's
    noise; viewed straight down vs at an angle the same crate's top read 8.2 vs 7.1 cm (DH-15)."""
    def frac(x, y, rel):
        return abs(x - y) / max(rel * max(abs(x), abs(y)), abs_tol)
    worst = max([frac(r.top_z, a.top_z, tol)] + [frac(x, y, 2 * tol) for x, y in zip(r.footprint, a.footprint)])
    return float(max(0.0, 1.0 - worst))


def pair_score(removed, added, cfg: dict) -> tuple[float, dict]:
    a = cfg.get("association", {})
    tol = float(a.get("dim_tol", 0.15))
    if np.isfinite(removed.top_z) and np.isfinite(added.top_z) and removed.footprint and added.footprint:
        size = _robust_size_score(removed, added, tol, float(a.get("dim_abs_tol_m", 0.02)))
    else:
        size = _size_score(removed.size, added.size, tol)
    count = min(removed.n_points, added.n_points) / max(removed.n_points, added.n_points)
    inten = max(0.0, 1.0 - abs(removed.mean_intensity - added.mean_intensity) / float(a.get("intensity_tol", 0.15)))
    w = a.get("weights", {"size": 0.5, "count": 0.25, "intensity": 0.25})
    score = w["size"] * size + w["count"] * count + w["intensity"] * inten
    parts = {"size": size, "count": count, "intensity": inten}
    if size == 0.0:                      # a different-sized thing is never "the same object moved"
        score = 0.0
    max_move = a.get("max_move_m")
    if max_move is not None and np.linalg.norm((added.centroid - removed.centroid)[:2]) > float(max_move):
        score = 0.0
    return float(score), parts


def associate(removed: list, added: list, cfg: dict) -> Association:
    """Greedy best-first: take the highest-scoring (removed, added) pair above min_score, remove both, repeat."""
    min_score = float(cfg.get("association", {}).get("min_score", 0.6))
    cand = []
    for i, r in enumerate(removed):
        for j, a in enumerate(added):
            s, parts = pair_score(r, a, cfg)
            if s >= min_score:
                cand.append((s, i, j, parts))
    cand.sort(key=lambda c: -c[0])
    used_r, used_a, out = set(), set(), Association()
    for s, i, j, parts in cand:
        if i in used_r or j in used_a:
            continue
        used_r.add(i)
        used_a.add(j)
        out.moved.append(Moved(removed[i], added[j], s, parts))
    out.removed = [r for i, r in enumerate(removed) if i not in used_r]
    out.added = [a for j, a in enumerate(added) if j not in used_a]
    return out

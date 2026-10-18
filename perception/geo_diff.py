"""Geometric diff of an aligned current scan against the baseline (ARCHITECTURE §6.4), with visibility checks.

added    current points farther than tau from any baseline point, EXCEPT those the baseline could not have seen
         (its own geometry was in the way): those are newly REVEALED, e.g. the table under a crate that was removed.
removed  baseline points farther than tau from any current point, EXCEPT those the current scan could not have seen
         (current geometry in the way): those are OCCLUDED, e.g. the table behind a drum that was added.
The spec (§6.4) asks for the second check; the first is its mirror image, added because without it every removed
object also produces an "added" patch of revealed table (DH-13).

Visibility is a coarse voxel-grid ray march from the sensor origin to the point (§6.4: "a coarse voxel-grid ray march
is enough"). Clusters come from DBSCAN, then the min-points / min-size filters from the config.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN


@dataclass
class Cluster:
    kind: str                     # "added" | "removed"
    idx: np.ndarray               # indices into the current (added) or baseline (removed) cloud
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    n_points: int
    mean_intensity: float
    top_z: float = float("nan")          # MEDIAN z of the change cluster = its top sliver (the bottom ~tau is cut away);
                                         # the median was 3x steadier across viewpoints than the 95th percentile (DH-15)
    footprint: tuple = ()                # (p95 - p5) extent in x and y, sorted ascending

    @property
    def size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min

    def as_dict(self) -> dict:
        return {"kind": self.kind, "n_points": self.n_points, "mean_intensity": round(self.mean_intensity, 3),
                "center": [round(float(v), 4) for v in (self.bbox_min + self.bbox_max) / 2],
                "size": [round(float(v), 4) for v in self.size]}


@dataclass
class GeoDiff:
    added: list = field(default_factory=list)
    removed: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _keys(xyz: np.ndarray, vox: float) -> np.ndarray:
    """Voxel indices packed into one int64 (21 bits per axis, offset so negatives fit)."""
    k = np.floor(np.asarray(xyz) / vox).astype(np.int64) + (1 << 20)
    return (k[..., 0] << 42) | (k[..., 1] << 21) | k[..., 2]


def occluded(points: np.ndarray, origin, occupied_xyz: np.ndarray, vox: float = 0.02,
             margin_voxels: float = 2.0, chunk: int = 4000) -> np.ndarray:
    """True where the straight line from `origin` to a point passes through an occupied voxel of `occupied_xyz`
    before reaching the point (stopping `margin_voxels` short of it, so the point's own surface does not count)."""
    if len(points) == 0:
        return np.zeros(0, bool)
    occ = np.unique(_keys(occupied_xyz, vox))
    o = np.asarray(origin, dtype=np.float64)
    out = np.zeros(len(points), bool)
    step = vox / 2.0
    for s0 in range(0, len(points), chunk):
        p = np.asarray(points[s0:s0 + chunk], dtype=np.float64)
        v = p - o
        r = np.linalg.norm(v, axis=1)
        d = v / np.maximum(r, 1e-9)[:, None]
        reach = r - margin_voxels * vox
        n_steps = int(np.ceil(max(reach.max(), 0) / step)) + 1
        s = np.arange(1, n_steps) * step
        samples = o + d[:, None, :] * s[None, :, None]
        valid = s[None, :] < reach[:, None]
        k = _keys(samples, vox)
        pos = np.searchsorted(occ, k)
        hit = (pos < len(occ)) & (occ[np.minimum(pos, len(occ) - 1)] == k) & valid
        out[s0:s0 + chunk] = hit.any(axis=1)
    return out


def _clusters(kind: str, xyz: np.ndarray, intensity: np.ndarray, cand: np.ndarray, cfg: dict) -> list:
    if cand.sum() == 0:
        return []
    idx = np.where(cand)[0]
    labels = DBSCAN(eps=float(cfg.get("cluster_eps_m", 0.02)), min_samples=int(cfg.get("dbscan_min_samples", 5))).fit_predict(xyz[idx])
    out = []
    min_pts, min_size = int(cfg.get("min_cluster_points", 50)), float(cfg.get("min_cluster_size_m", 0.04))
    for lab in sorted(set(labels) - {-1}):
        members = idx[labels == lab]
        pts = xyz[members]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        if len(members) < min_pts or (hi - lo).max() < min_size:
            continue
        p5, p95 = np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)
        out.append(Cluster(kind, members, pts.mean(axis=0), lo, hi, len(members), float(intensity[members].mean()),
                           float(np.median(pts[:, 2])), tuple(sorted((float(p95[0] - p5[0]), float(p95[1] - p5[1]))))))
    return out


def geo_diff(base_xyz: np.ndarray, base_int: np.ndarray, cur_xyz: np.ndarray, cur_int: np.ndarray, cfg: dict,
             base_origin=(0.0, 0.0, 0.0), cur_origin=(0.0, 0.0, 0.0)) -> GeoDiff:
    """Both clouds in the baseline frame (current already aligned). Origins = sensor positions in that frame."""
    tau = float(cfg.get("diff_threshold_m", 0.04))
    vox = float(cfg.get("visibility_voxel_m", 0.02))
    d_cur, _ = cKDTree(base_xyz).query(cur_xyz, distance_upper_bound=tau)
    d_base, _ = cKDTree(cur_xyz).query(base_xyz, distance_upper_bound=tau)
    far_cur, far_base = ~np.isfinite(d_cur), ~np.isfinite(d_base)
    revealed = np.zeros(len(cur_xyz), bool)
    hidden = np.zeros(len(base_xyz), bool)
    revealed[far_cur] = occluded(cur_xyz[far_cur], base_origin, base_xyz, vox)
    hidden[far_base] = occluded(base_xyz[far_base], cur_origin, cur_xyz, vox)
    added_c = far_cur & ~revealed
    removed_c = far_base & ~hidden
    diff = GeoDiff(_clusters("added", cur_xyz, cur_int, added_c, cfg), _clusters("removed", base_xyz, base_int, removed_c, cfg))
    diff.stats = {"n_far_current": int(far_cur.sum()), "n_revealed": int(revealed.sum()), "n_added_points": int(added_c.sum()),
                  "n_far_baseline": int(far_base.sum()), "n_occluded": int(hidden.sum()), "n_removed_points": int(removed_c.sum())}
    return diff

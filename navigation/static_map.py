"""Static occupancy map from the prior lidar map (docs/SCOUT_NAV.md §6; DN-40, DN-41).

    sm = StaticMap.from_points(points (N,3) map, cfg=None)
    sm.voxel_m, sm.origin (3,), sm.occ (nx,ny,nz) bool, sm.bounds_lo / bounds_hi (3,), sm.floor_z, sm.ceiling_z
    sm.occupied(p) -> bool | (N,) bool                 p (3,) or (N,3); outside the grid = False (unknown)
    sm.terrain_z(x, y, z_below, obstacles=None) -> float
        surface height under (x, y): the highest occupied voxel whose mean point z is below z_below (the voxel's
        mean, not its top, so a crate top at 1.00 m reads ~1.00, DN-41); obstacles with top_seen raise it to their
        top; floor_z if nothing. This is what a downward rangefinder sees (§3 height reference).
    sm.distance_along(p, direction, max_d, obstacles=None) -> float
        ray march (half-voxel steps) to the first occupied voxel, known obstacle box or the edge of the mapped grid
        (an unknown boundary counts as a surface, DN-40); max_d if nothing.
    sm.distance_along_rays(P (N,3), direction, max_d, obstacles=None) -> (N,)
        the same for parallel rays from N starts (the follower's hull corridor, DN-57); a ray starting inside an
        occupied voxel / box ignores that run / box.
    sm.band(cfg) -> (z_lo, z_hi)                       floor + min_alt_m .. ceiling - ceiling_margin_m
    sm.geofence(cfg) -> (lo_xy (2,), hi_xy (2,))       map bounds shrunk by geofence_margin_m
    sm.clear_cylinder(p, radius, half_height) -> bool  no map point inside the vertical cylinder
    sm.nearest_points(p, r) -> (M,3)                   map points within r of p (KD-tree)

Floor / ceiling: the lowest / highest voxel layer whose occupied count is >= surface_layer_frac of the fullest
layer; the height is the median z of the points in that layer and the next one (a surface straddles voxel borders).
Bounds: 0.1 / 99.9 percentiles of the points that fall in occupied voxels (stray points do not grow the fence).
"""
from __future__ import annotations

import numpy as np

from .nav_config import shared_defaults

DEFAULTS = dict(shared_defaults())
DEFAULTS.update({
    "voxel_m": 0.1,
    "voxel_min_points": 3,
    "surface_layer_frac": 0.3,
})


def ray_box_distance(p, d, lo, hi) -> float:
    """Distance along the unit ray p + t d (t >= 0) to the axis-aligned box [lo, hi]; inf if missed; 0 if inside."""
    p, d, lo, hi = (np.asarray(a, dtype=np.float64) for a in (p, d, lo, hi))
    if np.all(p >= lo) and np.all(p <= hi):
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1, t2 = (lo - p) * inv, (hi - p) * inv
    tmin = np.where(np.isnan(t1), -np.inf, np.minimum(t1, t2))
    tmax = np.where(np.isnan(t2), np.inf, np.maximum(t1, t2))
    # a zero direction component: inside the slab -> (-inf, inf), outside -> empty
    zero = d == 0
    if np.any(zero & ((p < lo) | (p > hi))):
        return float("inf")
    tmin = np.where(zero, -np.inf, tmin)
    tmax = np.where(zero, np.inf, tmax)
    t0, t1_ = float(np.max(tmin)), float(np.min(tmax))
    if t1_ < max(t0, 0.0):
        return float("inf")
    return max(t0, 0.0)


class StaticMap:
    def __init__(self, occ, origin, voxel_m, zmean, points, floor_z, ceiling_z, bounds_lo, bounds_hi, cfg):
        self.occ = occ
        self.origin = np.asarray(origin, dtype=np.float64)
        self.voxel_m = float(voxel_m)
        self.zmean = zmean
        self.points = points
        self.floor_z = float(floor_z)
        self.ceiling_z = float(ceiling_z)
        self.bounds_lo = np.asarray(bounds_lo, dtype=np.float64)
        self.bounds_hi = np.asarray(bounds_hi, dtype=np.float64)
        self.cfg = cfg
        self._tree = None

    # ---------------------------------------------------------------------------------------------------------
    @classmethod
    def from_points(cls, points, cfg: dict | None = None) -> "StaticMap":
        c = dict(DEFAULTS)
        c.update(cfg or {})
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) < 100:
            raise ValueError(f"map has {len(pts)} points: too few for a static map")
        res = float(c["voxel_m"])
        origin = np.floor(pts.min(axis=0) / res) * res - res               # one voxel of padding
        shape = (np.floor((pts.max(axis=0) - origin) / res).astype(int) + 2)
        idx = np.floor((pts - origin) / res).astype(np.int64)
        lin = np.ravel_multi_index(idx.T, shape)
        n = int(np.prod(shape))
        counts = np.bincount(lin, minlength=n)
        zsum = np.bincount(lin, weights=pts[:, 2], minlength=n)
        occ_flat = counts >= int(c["voxel_min_points"])
        zmean = np.full(n, np.nan, np.float32)
        zmean[occ_flat] = (zsum[occ_flat] / counts[occ_flat]).astype(np.float32)
        occ = occ_flat.reshape(shape)
        if not occ.any():
            raise ValueError("no voxel reaches voxel_min_points: map too sparse for the voxel size")

        layer = occ.sum(axis=(0, 1))
        full = layer >= float(c["surface_layer_frac"]) * layer.max()
        k_floor, k_ceil = int(np.argmax(full)), int(len(full) - 1 - np.argmax(full[::-1]))
        in_occ = occ_flat[lin]
        z = pts[:, 2]
        z_floor_lo = origin[2] + k_floor * res
        sel = in_occ & (z >= z_floor_lo) & (z < z_floor_lo + 2 * res)
        floor_z = float(np.median(z[sel])) if sel.any() else z_floor_lo + res / 2
        z_ceil_hi = origin[2] + (k_ceil + 1) * res
        sel = in_occ & (z < z_ceil_hi) & (z >= z_ceil_hi - 2 * res)
        ceiling_z = float(np.median(z[sel])) if sel.any() else z_ceil_hi - res / 2
        if ceiling_z <= floor_z + 0.5:
            raise ValueError(f"floor {floor_z:.2f} m / ceiling {ceiling_z:.2f} m not separable: is this an indoor map?")
        good = pts[in_occ]
        lo = np.percentile(good, 0.1, axis=0)
        hi = np.percentile(good, 99.9, axis=0)
        lo[2], hi[2] = floor_z, ceiling_z
        return cls(occ, origin, res, zmean.reshape(shape), pts.astype(np.float32), floor_z, ceiling_z, lo, hi, c)

    # ---------------------------------------------------------------------------------------------------------
    @property
    def shape(self):
        return self.occ.shape

    def index(self, p):
        """(N,3) points -> (N,3) int voxel indices and an (N,) in-grid mask."""
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        idx = np.floor((p - self.origin) / self.voxel_m).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.asarray(self.occ.shape)), axis=1)
        return idx, ok

    def occupied(self, p):
        p = np.asarray(p, dtype=np.float64)
        single = p.ndim == 1
        idx, ok = self.index(p)
        out = np.zeros(len(idx), bool)
        out[ok] = self.occ[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        return bool(out[0]) if single else out

    def terrain_z(self, x, y, z_below, obstacles=None) -> float:
        idx, ok = self.index([x, y, self.origin[2]])
        best = -np.inf
        if ok[0]:
            col = self.zmean[idx[0, 0], idx[0, 1], :]
            col = col[np.isfinite(col) & (col < z_below)]
            if col.size:
                best = float(col.max())
        for ob in obstacles or ():
            lo, hi = ob.lo, ob.hi
            if ob.top_seen and lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and hi[2] < z_below:
                best = max(best, float(hi[2]))
        return best if np.isfinite(best) else self.floor_z

    def distance_along(self, p, direction, max_d, obstacles=None) -> float:
        p = np.asarray(p, dtype=np.float64).reshape(3)
        d = np.asarray(direction, dtype=np.float64).reshape(3)
        nd = np.linalg.norm(d)
        if nd < 1e-9 or max_d <= 0:
            return float(max_d)
        d = d / nd
        step = self.voxel_m / 2
        t = np.arange(0.0, max_d + 1e-9, step)
        idx, ok = self.index(p + t[:, None] * d)
        hit = ~ok
        hit[ok] = self.occ[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        best = float(t[np.argmax(hit)]) if hit.any() else float(max_d)
        for ob in obstacles or ():
            hi = ob.hi.copy()
            if not ob.top_seen:
                hi[2] = max(hi[2], self.ceiling_z)
            best = min(best, ray_box_distance(p, d, ob.lo, hi))
        return float(min(best, max_d))

    def distance_along_rays(self, P, direction, max_d, obstacles=None) -> np.ndarray:
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        d = np.asarray(direction, dtype=np.float64).reshape(3)
        nd = np.linalg.norm(d)
        if nd < 1e-9 or max_d <= 0 or not len(P):
            return np.full(len(P), float(max_d))
        d = d / nd
        t = np.arange(0.0, max_d + 1e-9, self.voxel_m / 2)
        idx, ok = self.index((P[:, None, :] + t[None, :, None] * d).reshape(-1, 3))
        hit = ~ok
        hit[ok] = self.occ[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        hit = hit.reshape(len(P), len(t))
        hit &= ~np.cumprod(hit, axis=1).astype(bool)            # the occupied run a ray starts in is beside the hull
        best = np.where(hit.any(axis=1), t[np.argmax(hit, axis=1)], float(max_d))
        for ob in obstacles or ():
            hi = ob.hi.copy()
            if not ob.top_seen:
                hi[2] = max(hi[2], self.ceiling_z)
            for i, p in enumerate(P):
                if not (np.all(p >= ob.lo) and np.all(p <= hi)):
                    best[i] = min(best[i], ray_box_distance(p, d, ob.lo, hi))
        return np.minimum(best, float(max_d))

    def band(self, cfg: dict | None = None) -> tuple:
        c = dict(DEFAULTS)
        c.update(cfg or {})
        lo, hi = self.floor_z + float(c["min_alt_m"]), self.ceiling_z - float(c["ceiling_margin_m"])
        if hi <= lo:
            raise ValueError(f"empty flight band: {lo:.2f}..{hi:.2f} m")
        return lo, hi

    def geofence(self, cfg: dict | None = None) -> tuple:
        c = dict(DEFAULTS)
        c.update(cfg or {})
        m = float(c["geofence_margin_m"])
        return self.bounds_lo[:2] + m, self.bounds_hi[:2] - m

    # ---------------------------------------------------------------------------------------------------------
    def _kdtree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.points)
        return self._tree

    def nearest_points(self, p, r) -> np.ndarray:
        ii = self._kdtree().query_ball_point(np.asarray(p, dtype=np.float64).reshape(3), float(r))
        return self.points[np.asarray(ii, dtype=np.int64)].astype(np.float64)

    def nearest_distance(self, p) -> float:
        d, _ = self._kdtree().query(np.asarray(p, dtype=np.float64).reshape(3))
        return float(d)

    def clear_cylinder(self, p, radius, half_height) -> bool:
        p = np.asarray(p, dtype=np.float64).reshape(3)
        near = self.nearest_points(p, float(np.hypot(radius, half_height)))
        if not len(near):
            return True
        inside = (np.hypot(near[:, 0] - p[0], near[:, 1] - p[1]) <= radius) & (np.abs(near[:, 2] - p[2]) <= half_height)
        return not bool(inside.any())

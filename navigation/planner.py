"""2.5-D planner: over or around (docs/SCOUT_NAV.md §6; DN-43..DN-49).

    planner = Planner(static_map, cfg=None)          precomputes the per-layer inflated static grids + edge list
    planner.plan(start_xyz, goal: GoalPose, obstacles: list[Obstacle] = ()) -> PlanResult
    planner.layer_z (L,), planner.free_mask(obstacles) -> (L, nx, ny) bool, planner.segment_free(p, q, free) -> bool

Search space: (x, y, layer). xy cells = the static map's voxel grid (0.1 m); layers every layer_dz_m from the band
bottom (floor + min_alt_m) up to the band top (ceiling - ceiling_margin_m). A layer's cell is blocked when
  - a mapped voxel overlapping [z - (hull_half_height + static_vmargin_m), z + ...] lies within
    hull_radius + static_inflate_extra_m (+ one cell, discretisation) horizontally, or
  - an obstacle box overlapping [z - (hull_half_height + dynamic_vmargin_m), z + ...] lies within
    hull_radius + dynamic_inflate_extra_m (+ half a cell diagonal); top_seen=False boxes reach +inf (never over), or
  - the cell is outside the geofence (map bounds - geofence_margin_m), so the safety gate never clamps a plan.
Edges: 8-connected in a layer (cost = length), vertical between adjacent layers (cost = climb_cost_factor x dz).
"Over" vs "around" falls out of that cost. Search: Dijkstra on a scipy sparse graph (C speed; DN-43).
Then greedy farthest-visible line-of-sight shortcutting in 3D (a sample between layers must be free in both).

Start inside inflation (the vehicle is already there): hop to the nearest free node within max_start_escape_m whose
straight segment does not touch hull-only clearance (DN-48). Goal inside inflation or outside the band: ok=False.
PlanResult.shape: "direct" (one straight segment) | "over" (climbs above both endpoints) | "around" (lateral
deviation > around_tol_m, or dips under something) | "over+around". info: plan_time_s, over_ids / around_ids
(obstacles whose inflated footprint the straight line crosses, split by whether the plan passes above them), ...
"""
from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .static_map import DEFAULTS as MAP_DEFAULTS
from .static_map import StaticMap
from .types import GoalPose, Obstacle, PlanResult

DEFAULTS = dict(MAP_DEFAULTS)
DEFAULTS.update({
    "layer_dz_m": 0.2,
    "climb_cost_factor": 3.0,
    "static_vmargin_m": 0.2,
    "dynamic_vmargin_m": 0.3,
    "max_start_escape_m": 1.2,
    "around_tol_m": 0.25,
})

_OFFSETS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _pair_slices(n, d):
    """index ranges (src, dst) along one axis for a neighbour offset d."""
    return (slice(0, n - d), slice(d, n)) if d >= 0 else (slice(-d, n), slice(0, n + d))


class Planner:
    def __init__(self, static_map: StaticMap, cfg: dict | None = None):
        c = dict(DEFAULTS)
        c.update(cfg or {})
        self.cfg = c
        self.map = sm = static_map
        self.res = res = sm.voxel_m
        nx, ny, nz = sm.occ.shape
        self.nx, self.ny = nx, ny
        self.x0, self.y0 = float(sm.origin[0]), float(sm.origin[1])
        self.z_lo, self.z_hi = sm.band(c)
        self.dz = float(c["layer_dz_m"])
        L = int(np.floor((self.z_hi - self.z_lo) / self.dz + 1e-9)) + 1
        self.layer_z = self.z_lo + self.dz * np.arange(L)
        self.L = L
        self.xc = self.x0 + (np.arange(nx) + 0.5) * res
        self.yc = self.y0 + (np.arange(ny) + 0.5) * res
        hull = float(c["hull_radius_m"])
        self.R_static = hull + float(c["static_inflate_extra_m"])
        self.R_dyn = hull + float(c["dynamic_inflate_extra_m"])
        self.R_hard = hull
        self.v_static = float(c["hull_half_height_m"]) + float(c["static_vmargin_m"])
        self.v_dyn = float(c["hull_half_height_m"]) + float(c["dynamic_vmargin_m"])
        self.climb = float(c["climb_cost_factor"])

        flo, fhi = sm.geofence(c)
        fence = ((self.xc[:, None] >= flo[0]) & (self.xc[:, None] <= fhi[0])
                 & (self.yc[None, :] >= flo[1]) & (self.yc[None, :] <= fhi[1]))
        zc = sm.origin[2] + (np.arange(nz) + 0.5) * res
        self.static_free = np.zeros((L, nx, ny), bool)
        self.static_hard = np.zeros((L, nx, ny), bool)          # within hull radius only: real contact
        for k, z in enumerate(self.layer_z):
            sel = (zc + res / 2 > z - self.v_static) & (zc - res / 2 < z + self.v_static)
            col = sm.occ[:, :, sel].any(axis=2)
            dist = distance_transform_edt(~col) * res if col.any() else np.full((nx, ny), np.inf)
            self.static_free[k] = (dist > self.R_static + res) & fence
            self.static_hard[k] = dist <= self.R_hard + res
        self.N = L * nx * ny
        ids = np.arange(self.N).reshape(L, nx, ny)
        us, vs, ws = [], [], []
        for di, dj in _OFFSETS:
            si, di_ = _pair_slices(nx, di)
            sj, dj_ = _pair_slices(ny, dj)
            m = self.static_free[:, si, sj] & self.static_free[:, di_, dj_]
            us.append(ids[:, si, sj][m])
            vs.append(ids[:, di_, dj_][m])
            ws.append(np.full(int(m.sum()), res * np.hypot(di, dj)))
        m = self.static_free[:-1] & self.static_free[1:]
        us.append(ids[:-1][m])
        vs.append(ids[1:][m])
        ws.append(np.full(int(m.sum()), self.climb * self.dz))
        self._u = np.concatenate(us)
        self._v = np.concatenate(vs)
        self._w = np.concatenate(ws)
        self._G_static = csr_matrix((self._w, (self._u, self._v)), shape=(self.N, self.N))

    # ---------------------------------------------------------------------------------------------------------
    def _cell(self, p):
        i = int(np.floor((p[0] - self.x0) / self.res))
        j = int(np.floor((p[1] - self.y0) / self.res))
        return i, j, (0 <= i < self.nx and 0 <= j < self.ny)

    def _layer(self, z):
        return int(np.argmin(np.abs(self.layer_z - np.clip(z, self.z_lo, self.z_hi))))

    def _node_xyz(self, nid):
        k, rem = divmod(int(nid), self.nx * self.ny)
        i, j = divmod(rem, self.ny)
        return np.array([self.xc[i], self.yc[j], self.layer_z[k]])

    def _box_mask(self, ob: Obstacle, R, vmargin):
        """(L, nx, ny) cells within R of the box footprint on layers its z-range (+- vmargin) overlaps."""
        lo, hi = np.asarray(ob.lo, float), np.asarray(ob.hi, float)
        top = hi[2] if ob.top_seen else np.inf
        layers = (top > self.layer_z - vmargin) & (lo[2] < self.layer_z + vmargin)
        out = np.zeros((self.L, self.nx, self.ny), bool)
        if not layers.any():
            return out
        dx = np.maximum(np.maximum(lo[0] - self.xc, self.xc - hi[0]), 0.0)
        dy = np.maximum(np.maximum(lo[1] - self.yc, self.yc - hi[1]), 0.0)
        foot = np.hypot(dx[:, None], dy[None, :]) <= R + self.res * 0.71
        out[layers] = foot
        return out

    def free_mask(self, obstacles=()) -> np.ndarray:
        free = self.static_free.copy()
        for ob in obstacles or ():
            free &= ~self._box_mask(ob, self.R_dyn, self.v_dyn)
        return free

    def _hard_mask(self, obstacles=()):
        hard = self.static_hard.copy()
        for ob in obstacles or ():
            hard |= self._box_mask(ob, self.R_hard, float(self.cfg["hull_half_height_m"]))
        return hard

    def _samples_ok(self, pts, free):
        """(M,3) points -> (M,) bool: in grid, in band, free on both bracketing layers."""
        i = np.floor((pts[:, 0] - self.x0) / self.res).astype(np.int64)
        j = np.floor((pts[:, 1] - self.y0) / self.res).astype(np.int64)
        fk = (pts[:, 2] - self.z_lo) / self.dz
        ok = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny) & (fk >= -1e-6) & (fk <= self.L - 1 + 1e-6)
        i, j = np.clip(i, 0, self.nx - 1), np.clip(j, 0, self.ny - 1)
        k0 = np.clip(np.floor(fk + 1e-6).astype(np.int64), 0, self.L - 1)
        k1 = np.clip(np.ceil(fk - 1e-6).astype(np.int64), 0, self.L - 1)
        return ok & free[k0, i, j] & free[k1, i, j]

    def _seg_samples(self, p, Q):
        """samples along p -> each row of Q; -> (points (M,3), segment index (M,))."""
        step = self.res / 2
        lens = np.linalg.norm(Q - p, axis=1)
        n = np.maximum(2, np.ceil(lens / step).astype(np.int64) + 1)
        seg = np.repeat(np.arange(len(Q)), n)
        start = np.concatenate([[0], np.cumsum(n)[:-1]])
        t = (np.arange(seg.size) - np.repeat(start, n)) / np.repeat(n - 1, n)
        return p + t[:, None] * (Q[seg] - p), seg

    def segment_free(self, p, q, free) -> bool:
        pts, _ = self._seg_samples(np.asarray(p, float), np.asarray(q, float).reshape(1, 3))
        return bool(self._samples_ok(pts, free).all())

    def _shortcut(self, P, free):
        out = [P[0]]
        i, n = 0, len(P)
        while i < n - 1:
            nxt = i + 1
            if i + 2 < n:
                cand = np.arange(i + 2, n)
                pts, seg = self._seg_samples(P[i], P[cand])
                bad = np.bincount(seg[~self._samples_ok(pts, free)], minlength=len(cand))
                vis = np.nonzero(bad == 0)[0]
                if vis.size:
                    nxt = int(cand[vis.max()])
            out.append(P[nxt])
            i = nxt
        return np.asarray(out)

    def _cost(self, P):
        d = np.diff(P, axis=0)
        return float(np.sum(np.hypot(d[:, 0], d[:, 1])) + self.climb * np.sum(np.abs(d[:, 2])))

    # ---------------------------------------------------------------------------------------------------------
    def plan(self, start_xyz, goal: GoalPose, obstacles=()) -> PlanResult:
        t0 = time.perf_counter()
        obstacles = list(obstacles or ())
        start = np.asarray(start_xyz, dtype=np.float64).reshape(3)
        gpos = np.asarray(goal.position, dtype=np.float64).reshape(3)
        info = {"n_obstacles": len(obstacles), "layers": len(self.layer_z)}

        def fail(reason):
            info["plan_time_s"] = time.perf_counter() - t0
            return PlanResult(False, None, reason, float("inf"), "", info)

        if not (self.z_lo - 1e-6 <= gpos[2] <= self.z_hi + 1e-6):
            return fail(f"goal z {gpos[2]:.2f} outside the flight band {self.z_lo:.2f}..{self.z_hi:.2f}")
        gi, gj, gin = self._cell(gpos)
        si, sj, sin_ = self._cell(start)
        if not gin:
            return fail("goal outside the map grid")
        if not sin_:
            return fail("start outside the map grid")
        free = self.free_mask(obstacles)
        kg = self._layer(gpos[2])
        if not free[kg, gi, gj]:
            why = "static map" if not self.static_free[kg, gi, gj] else "an obstacle"
            return fail(f"goal inside the inflated {why} (or outside the geofence)")
        g = (kg * self.nx + gi) * self.ny + gj

        ks = self._layer(start[2])
        escape = 0.0
        if free[ks, si, sj]:
            s = (ks * self.nx + si) * self.ny + sj
        else:
            s, escape = self._escape(start, free, self._hard_mask(obstacles))
            if s is None:
                return fail(f"start inside inflation with no free node within {self.cfg['max_start_escape_m']} m")
        info["start_escape_m"] = escape

        if obstacles:
            keep = free.reshape(-1)[self._u] & free.reshape(-1)[self._v]
            G = csr_matrix((self._w[keep], (self._u[keep], self._v[keep])), shape=(self.N, self.N))
        else:
            G = self._G_static
        dist, pred = dijkstra(G, directed=False, indices=s, return_predecessors=True)
        if not np.isfinite(dist[g]):
            return fail("no path: the goal is not reachable in the flight band (boxed in)")
        nodes = [g]
        while nodes[-1] != s:
            nodes.append(int(pred[nodes[-1]]))
        nodes.reverse()
        raw = np.vstack([start] + [self._node_xyz(n) for n in nodes] + [gpos])
        P = self._shortcut(raw, free)
        cost = self._cost(P)
        shape, extra = self._shape(P, start, gpos, obstacles)
        info.update(extra)
        info.update({"raw_nodes": len(nodes), "graph_cost": float(dist[g]), "plan_time_s": time.perf_counter() - t0})
        return PlanResult(True, P, "", cost, shape, info)

    def _escape(self, start, free, hard):
        r = float(self.cfg["max_start_escape_m"])
        ks = self._layer(start[2])
        best, best_d = None, 0.0
        w = int(np.ceil(r / self.res)) + 1
        si, sj, _ = self._cell(start)
        i0, i1 = max(0, si - w), min(self.nx, si + w + 1)
        j0, j1 = max(0, sj - w), min(self.ny, sj + w + 1)
        clear = ~hard
        cands = []
        for k in range(max(0, ks - 1), min(self.L, ks + 2)):
            ii, jj = np.nonzero(free[k, i0:i1, j0:j1])
            if not ii.size:
                continue
            pts = np.column_stack([self.xc[ii + i0], self.yc[jj + j0], np.full(ii.size, self.layer_z[k])])
            d = np.linalg.norm(pts - start, axis=1)
            for m in np.nonzero(d <= r)[0]:
                cands.append((d[m], k, ii[m] + i0, jj[m] + j0, pts[m]))
        for d, k, i, j, p in sorted(cands, key=lambda t: t[0]):
            pts, _ = self._seg_samples(start, p.reshape(1, 3))
            ii = np.clip(np.floor((pts[:, 0] - self.x0) / self.res).astype(int), 0, self.nx - 1)
            jj = np.clip(np.floor((pts[:, 1] - self.y0) / self.res).astype(int), 0, self.ny - 1)
            kk = np.array([self._layer(z) for z in pts[:, 2]])
            ok = clear[kk, ii, jj]
            # all clear, or the vehicle is already inside the hard keep-out (e.g. stopped by a looming emergency
            # close to an obstacle) and the hop only LEAVES it: blocked samples form a prefix, never re-entered
            leaving = (not ok[0]) and ok[-1] and bool(np.all(np.diff(ok.astype(np.int8)) >= 0))
            if ok.all() or leaving:
                best, best_d = (k * self.nx + i) * self.ny + j, d
                break
        return best, (float(best_d) if best is not None else 0.0)

    def _shape(self, P, start, gpos, obstacles):
        zs = float(np.clip(start[2], self.z_lo, self.z_hi))
        over = P[:, 2].max() > max(zs, gpos[2]) + 0.5 * self.dz
        under = P[:, 2].min() < min(zs, gpos[2]) - 0.5 * self.dz
        a, b = start[:2], gpos[:2]
        ab = b - a
        L2 = float(ab @ ab)
        # dense samples of the plan, for lateral deviation and per-obstacle relations
        S = [P[0]]
        for p, q in zip(P[:-1], P[1:]):
            n = max(2, int(np.ceil(np.linalg.norm(q - p) / (self.res / 2))) + 1)
            S.append(p + np.linspace(0, 1, n)[1:, None] * (q - p))
        S = np.vstack(S)
        if L2 > 1e-12:
            t = np.clip((S[:, :2] - a) @ ab / L2, 0, 1)
            lat = np.linalg.norm(S[:, :2] - (a + t[:, None] * ab), axis=1)
        else:
            lat = np.linalg.norm(S[:, :2] - a, axis=1)
        around = lat.max() > float(self.cfg["around_tol_m"]) or under
        over_ids, around_ids = [], []
        line = a + np.linspace(0, 1, 200)[:, None] * ab
        for ob in obstacles:
            lo, hi = np.asarray(ob.lo, float), np.asarray(ob.hi, float)
            dx = np.maximum(np.maximum(lo[0] - line[:, 0], line[:, 0] - hi[0]), 0)
            dy = np.maximum(np.maximum(lo[1] - line[:, 1], line[:, 1] - hi[1]), 0)
            if not (np.hypot(dx, dy) <= self.R_dyn).any():
                continue
            inside = np.all((S[:, :2] >= lo[:2]) & (S[:, :2] <= hi[:2]), axis=1)
            (over_ids if inside.any() and np.all(S[inside, 2] > hi[2]) else around_ids).append(ob.id)
        if len(P) == 2:
            shape = "direct"
        elif over and around:
            shape = "over+around"
        elif over:
            shape = "over"
        elif around:
            shape = "around"
        else:
            shape = "direct"
        return shape, {"over_ids": over_ids, "around_ids": around_ids, "max_z": float(P[:, 2].max()),
                       "max_lateral_m": float(lat.max()), "under": bool(under)}

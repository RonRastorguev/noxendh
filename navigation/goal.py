"""Wall goal: where to hover to look at a wall point B (docs/SCOUT_NAV.md §6; DN-42).

    wall_goal(static_map, point_b, cfg=None) -> GoalPose          raises ValueError(reason) when B is unusable

1. B must be on a mapped surface: >= goal_min_points map points within goal_pca_radius_m and the nearest within
   goal_max_b_dist_m.
2. PCA of those points: normal = smallest eigenvector; lambda_min/lambda_mid <= goal_max_planarity (a plane, not a
   corner); |elevation of the normal| <= goal_max_normal_tilt_deg (a wall, not a floor / crate top / ceiling).
3. Orientation into free space: each sign of the HORIZONTAL normal is tried; it is valid when the hover point is
   inside the geofence, the sight line from the hover point to B (stepped 1.5 voxels off the wall) is clear of
   mapped surfaces, and the hover point has static clearance (hull radius + static_inflate_extra_m horizontally, hull_half_height +
   static_vmargin_m vertically: the planner's clearance). Both valid -> the one facing the map centre.
4. Hover point = B + standoff_m x horizontal normal, z clamped into the flight band; yaw = the camera faces B
   (horizontally; the camera's own pitch is not compensated).
"""
from __future__ import annotations

import numpy as np

from .static_map import StaticMap
from .types import GoalPose

DEFAULTS = {
    "standoff_m": 0.8,
    "goal_pca_radius_m": 0.25,
    "goal_min_points": 20,
    "goal_max_b_dist_m": 0.10,
    "goal_max_normal_tilt_deg": 30.0,
    "goal_max_planarity": 0.2,
    "static_vmargin_m": 0.2,
}


def _cfg(cfg):
    from .static_map import DEFAULTS as MAP_DEFAULTS
    c = dict(MAP_DEFAULTS)
    c.update(DEFAULTS)
    c.update(cfg or {})
    return c


def wall_goal(static_map: StaticMap, point_b, cfg: dict | None = None) -> GoalPose:
    c = _cfg(cfg)
    b = np.asarray(point_b, dtype=np.float64).reshape(3)
    near = static_map.nearest_points(b, c["goal_pca_radius_m"])
    if len(near) < int(c["goal_min_points"]):
        raise ValueError(f"B {np.round(b, 2).tolist()} is not on a mapped surface: {len(near)} map points within "
                         f"{c['goal_pca_radius_m']} m (need {c['goal_min_points']})")
    dmin = static_map.nearest_distance(b)
    if dmin > c["goal_max_b_dist_m"]:
        raise ValueError(f"B is {dmin:.2f} m from the nearest map point (max {c['goal_max_b_dist_m']}): not on a surface")
    q = near - near.mean(axis=0)
    w, v = np.linalg.eigh(q.T @ q / len(q))                          # ascending eigenvalues
    planarity = float(w[0] / max(w[1], 1e-12))
    if planarity > c["goal_max_planarity"]:
        raise ValueError(f"the map around B is not a plane (planarity {planarity:.2f} > {c['goal_max_planarity']}): "
                         "corner or clutter; pick a point on a flat wall")
    n = v[:, 0] / np.linalg.norm(v[:, 0])
    tilt = float(np.degrees(np.arcsin(min(1.0, abs(n[2])))))
    if tilt > c["goal_max_normal_tilt_deg"]:
        raise ValueError(f"the surface at B is not a wall: normal elevation {tilt:.0f} deg "
                         f"(max {c['goal_max_normal_tilt_deg']})")
    nh = np.array([n[0], n[1], 0.0])
    nh /= np.linalg.norm(nh)

    z_lo, z_hi = static_map.band(c)
    fence_lo, fence_hi = static_map.geofence(c)
    standoff = float(c["standoff_m"])
    R = float(c["hull_radius_m"]) + float(c["static_inflate_extra_m"])
    H = float(c["hull_half_height_m"]) + float(c["static_vmargin_m"])
    center = (static_map.bounds_lo + static_map.bounds_hi) / 2
    cands, why = [], []
    for sgn in (1.0, -1.0):
        nn = sgn * nh
        hover = b + standoff * nn
        hover[2] = float(np.clip(b[2], z_lo, z_hi))
        if not (np.all(hover[:2] >= fence_lo) and np.all(hover[:2] <= fence_hi)):
            why.append(f"{'+' if sgn > 0 else '-'}n: hover outside the geofence")
            continue
        off = 1.5 * static_map.voxel_m                                 # step off the surface's own voxels
        p0 = b + nn * off
        need = float(np.linalg.norm(hover - p0))
        free = static_map.distance_along(p0, hover - p0, need)
        if free < need - 1e-6:                                         # the camera's sight line hover -> B
            why.append(f"{'+' if sgn > 0 else '-'}n: mapped surface {free + off:.2f} m from B on the sight line")
            continue
        if not static_map.clear_cylinder(hover, R, H):
            why.append(f"{'+' if sgn > 0 else '-'}n: hover point within {R:.2f} m of the map")
            continue
        cands.append((float(nn[:2] @ (center[:2] - b[:2])), nn, hover))
    if not cands:
        raise ValueError("no free hover point in front of B: " + "; ".join(why))
    _, nn, hover = max(cands, key=lambda t: t[0])
    normal = n if n @ nn > 0 else -n
    yaw = float(np.arctan2(b[1] - hover[1], b[0] - hover[0]))
    return GoalPose(position=hover, yaw=yaw, target=b, normal=normal)

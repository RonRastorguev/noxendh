"""Automatic camera refinement: line the lidar's object outlines up with the photo's edges (DH-30).

Ron, 2026-10-18: "can you just calibrate it for me? you have the scan and the image". Starting from a rough
calibration (clicked point pairs or corners), this refines the camera pose, and optionally the focal length, with
no clicks:
- **Lidar outlines**: seen from the camera, points where the depth jumps (an object in front of what is behind it)
  are the near side of a silhouette. Those 3D points are fixed for one pass.
- **Photo edges**: Canny edges of the photo, turned into a distance map (pixels to the nearest edge, clipped).
- **Cost**: the mean clipped distance of the projected outline points. The pose that minimises it puts the lidar's
  silhouettes on the photo's edges. Coarse to fine (a wide clip first, then a tight one), and the outlines are
  recomputed from the improved pose between passes.

Texture edges in the photo (floor planks, print) have no lidar counterpart. The clip keeps them from pulling, and
the rough calibration keeps the search local. The result is checked: the camera must stay near the lidar and the
cost must drop, or the start is kept.
"""
from __future__ import annotations

import numpy as np


def _project(P, K, T, dist=None):
    """Pinhole + radial lens (k1, k2: OpenCV's first two terms; the 102° DECXIN has clear barrel distortion)."""
    c = P @ T[:3, :3].T + T[:3, 3]
    z = c[:, 2]
    x, y = c[:, 0] / np.maximum(z, 1e-6), c[:, 1] / np.maximum(z, 1e-6)
    if dist is not None and (dist[0] or dist[1]):
        r2 = x * x + y * y
        f = 1 + dist[0] * r2 + dist[1] * r2 * r2
        x, y = x * f, y * f
    return np.stack([K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]], axis=1), z


def _lens_ok(K, dist, image_size) -> bool:
    """The radial map must keep growing out to the image corners (else the model folds back on itself)."""
    w, h = image_size
    rmax = np.hypot(max(K[0, 2], w - K[0, 2]) / K[0, 0], max(K[1, 2], h - K[1, 2]) / K[1, 1]) * 1.1
    r = np.linspace(0, rmax, 200)
    return bool(np.all(np.diff(r * (1 + dist[0] * r ** 2 + dist[1] * r ** 4)) > 0))


def outline_points(xyz, K, T, image_size, *, dist=None, cell_px: int = 2, jump_frac: float = 0.08, jump_m: float = 0.04,
                   min_depth: float = 0.15) -> np.ndarray:
    """3D points on the near side of depth jumps, as seen from the camera (K, T: scan frame -> camera)."""
    from scipy.ndimage import maximum_filter
    w, h = image_size
    uv, z = _project(np.asarray(xyz, float), K, T, dist)
    ok = (z > min_depth) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    idx = np.flatnonzero(ok)
    gw, gh = (w + cell_px - 1) // cell_px, (h + cell_px - 1) // cell_px
    cell = (uv[idx, 1] // cell_px).astype(int) * gw + (uv[idx, 0] // cell_px).astype(int)
    depth = np.full(gw * gh, np.inf)
    np.minimum.at(depth, cell, z[idx])
    near = np.where(np.isfinite(depth), depth, 0.0).reshape(gh, gw)
    # the farthest neighbour depth: a cell is an outline if something around it is clearly farther behind it
    far = maximum_filter(near, size=3)
    has = near > 0
    edge = has & (far > near * (1 + jump_frac) + jump_m)
    # keep, per outline cell, the nearest point in it
    order = np.lexsort((z[idx], cell))
    cs = cell[order]
    firsts = order[np.r_[True, cs[1:] != cs[:-1]]]
    on_edge = edge.ravel()[cell[firsts]]
    return np.asarray(xyz, float)[idx[firsts[on_edge]]]


def edge_distance(image_bgr, clip_px: float, blur: int = 5) -> np.ndarray:
    import cv2
    g = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (blur, blur), 0)
    v = float(np.median(g))
    edges = cv2.Canny(g, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))
    dt = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    return np.minimum(dt, clip_px).astype(np.float32)


def _cost(P, K, T, dt, clip_px, dist=None, wts=None):
    h, w = dt.shape
    uv, z = _project(P, K, T, dist)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    inside = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    d = np.full(len(P), clip_px, np.float32)               # a point that leaves the picture costs the most
    d[inside] = dt[v[inside], u[inside]]
    return float(d.mean()) if wts is None else float((d * wts).sum() / wts.sum())


def _compose(T0, p):
    import cv2
    R, _ = cv2.Rodrigues(np.asarray(p[:3], float))
    D = np.eye(4)
    D[:3, :3], D[:3, 3] = R, p[3:6]
    return D @ T0                                          # a small motion of the camera, applied after T0


def refine(xyz, image_bgr, K0, T0, sensor_pose, *, dist0=None, fit_focal: bool = True, fit_distortion: bool = False,
           focus=None, focus_weight: float = 4.0,
           max_cam_lidar_m: float = 0.6, max_shift_m: float = 0.15, max_turn_deg: float = 8.0) -> dict:
    """Refine (K0, T0, dist0) (scan frame -> camera) against one scan + its photo. Returns {ok, K, dist,
    T_cam_from_frame, cost_start, cost_end, n_outline, camera_to_lidar_m, focal_px, warnings} or {ok: False, reason}.
    `focus` (a function xyz -> bool mask, e.g. the crop box) weights the outlines that matter `focus_weight` times,
    while the rest of the picture still holds the camera's position (DH-31: on Ron's scan the whole-picture fit was
    dominated by chairs and people who moved during the scan, and left his bottle ~20 px off; the crop region
    alone was too few outlines and moved the camera 7 cm). fit_distortion solves the lens's radial k1, k2 too; it
    is OFF by default: on Ron's scan it made the fit worse and the lens terms stayed near zero."""
    from scipy.optimize import minimize

    h, w = image_bgr.shape[:2]
    K0, T0 = np.asarray(K0, float), np.asarray(T0, float)
    d0 = np.zeros(2) if dist0 is None else np.asarray(dist0, float)[:2]
    lidar = np.asarray(sensor_pose, float)[:3, 3]
    K, T, d = K0.copy(), T0.copy(), d0.copy()
    report = []
    dt15 = edge_distance(image_bgr, 15.0)
    P0 = outline_points(xyz, K0, T0, (w, h), dist=d0)
    start_cost = _cost(P0, K0, T0, dt15, 15.0, d0, None if focus is None else np.where(focus(P0), focus_weight, 1.0))
    nf = 1 if fit_focal else 0
    nd = 2 if fit_distortion else 0
    for clip, rounds in ((40.0, 2), (15.0, 2), (6.0, 2)):
        dt = edge_distance(image_bgr, clip)
        for _ in range(rounds):
            P = outline_points(xyz, K, T, (w, h), dist=d)
            wts = None if focus is None else np.where(focus(P), focus_weight, 1.0)
            if len(P) < 200:
                return {"ok": False, "reason": f"too few lidar outlines in view ({len(P)}): is the camera calibration "
                                               "roughly right, and is anything with clear edges in front of it?"}

            def unpack(p, K=K, d=d):
                Kp = K.copy()
                if fit_focal:
                    Kp[0, 0] = Kp[1, 1] = K[0, 0] * np.exp(p[6])
                dp = d + (p[6 + nf:6 + nf + nd] if fit_distortion else 0)
                return Kp, dp

            def f(p, P=P, dt=dt, clip=clip, wts=wts):
                Kp, dp = unpack(p)
                if fit_distortion and not _lens_ok(Kp, dp, (w, h)):
                    return clip                                  # a folded lens model is never a solution
                return _cost(P, Kp, _compose(T, p[:6]), dt, clip, dp, wts)

            x0 = np.zeros(6 + nf + nd)
            step = [0.01] * 6 + [0.02] * nf + [0.05] * nd
            r = minimize(f, x0, method="Powell", options={"xtol": 1e-4, "ftol": 1e-4, "maxfev": 6000, "direc": np.diag(step)})
            K, d = unpack(r.x)
            T = _compose(T, r.x[:6])
            report.append((clip, len(P), round(float(r.fun), 2)))
    P = outline_points(xyz, K, T, (w, h), dist=d)
    end_cost = _cost(P, K, T, dt15, 15.0, d, None if focus is None else np.where(focus(P), focus_weight, 1.0))
    cam = -T[:3, :3].T @ T[:3, 3]
    cam0 = -T0[:3, :3].T @ T0[:3, 3]
    shift = float(np.linalg.norm(cam - cam0))
    dR = T[:3, :3] @ T0[:3, :3].T
    turn = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    d_lidar = float(np.linalg.norm(cam - lidar))
    warnings = []
    if d_lidar > max_cam_lidar_m or shift > max_shift_m or turn > max_turn_deg or not 0.8 <= K[0, 0] / K0[0, 0] <= 1.25             or not _lens_ok(K, d, (w, h)):
        return {"ok": False, "reason": f"the refinement wandered (moved {shift * 100:.0f} cm, turned {turn:.1f}°, focal x"
                f"{K[0, 0] / K0[0, 0]:.2f}, lens k1 {d[0]:.2f} k2 {d[1]:.2f}): the starting calibration is probably too "
                "far off; redo the point pairs first", "passes": report}
    if end_cost >= start_cost * 0.98:
        warnings.append("the refinement found no clear improvement; the starting calibration was kept close")
    return {"ok": True, "K": K, "dist": [float(d[0]), float(d[1]), 0.0, 0.0, 0.0], "T_cam_from_frame": T,
            "cost_start": round(start_cost, 2), "cost_end": round(end_cost, 2), "n_outline": int(len(P)),
            "camera_to_lidar_m": round(d_lidar, 3), "moved_m": round(shift, 3), "turned_deg": round(turn, 2),
            "focal_px": round(float(K[0, 0]), 1), "k1": round(float(d[0]), 4), "k2": round(float(d[1]), 4),
            "passes": report, "warnings": warnings}

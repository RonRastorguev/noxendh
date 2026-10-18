"""Render a depth image of a map point cloud from a camera pose, with a z-buffer (§8.2, and §8.4 step 3).

A lidar map is sparse: projected points leave holes, and through the holes of a near surface the far surface behind
it shows up. Each point is therefore splatted as a disc about as wide as the map's point spacing at its depth
(radius = splat_scale * 0.5 * f * spacing / z pixels), the nearest depth wins. Points are grouped by splat radius
and each group is drawn with a min filter (grayscale erosion) of that size.

Returned depth is the camera z (not the range along the ray); pixels with no map data are +inf.
"""
from __future__ import annotations

import cv2
import numpy as np

from .camera import CameraModel

_BIG = np.float32(1e6)


def project_points(points_map, T_map_cam, cam: CameraModel, min_z=0.05, max_z=None):
    """-> (uv (M, 2) float, z (M,) float, idx (M,) into points_map) for points in front of and inside the view."""
    T_cam_map_R = np.asarray(T_map_cam)[:3, :3].T
    t = np.asarray(T_map_cam)[:3, 3]
    X = (np.asarray(points_map, dtype=np.float64) - t) @ T_cam_map_R.T
    z = X[:, 2]
    keep = z > min_z
    if max_z is not None:
        keep &= z <= max_z
    idx = np.nonzero(keep)[0]
    X = X[idx]
    xn, yn = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
    if cam.has_distortion:                       # drop off-view points before the polynomial can fold them back in
        x0, x1, y0, y1 = cam.normalized_bounds()
        m = (xn > x0 - 0.05) & (xn < x1 + 0.05) & (yn > y0 - 0.05) & (yn < y1 + 0.05)
        idx, X = idx[m], X[m]
    uv = cam.project(X)
    inside = (uv[:, 0] > -0.5) & (uv[:, 0] < cam.width - 0.5) & (uv[:, 1] > -0.5) & (uv[:, 1] < cam.height - 0.5)
    return uv[inside], X[inside, 2], idx[inside]


def render_depth(points_map, T_map_cam, cam: CameraModel, spacing_m: float, splat_scale: float = 1.0,
                 max_radius_px: int = 8, min_z: float = 0.05, max_z=None) -> np.ndarray:
    uv, z, _ = project_points(points_map, T_map_cam, cam, min_z, max_z)
    depth = np.full((cam.height, cam.width), _BIG, np.float32)
    if len(z) == 0:
        return np.full_like(depth, np.inf)
    u = np.clip(np.rint(uv[:, 0]).astype(np.int64), 0, cam.width - 1)
    v = np.clip(np.rint(uv[:, 1]).astype(np.int64), 0, cam.height - 1)
    f = 0.5 * (cam.fx + cam.fy)
    r = np.clip(np.rint(splat_scale * 0.5 * f * spacing_m / z), 0, max_radius_px).astype(np.int64)
    for rad in np.unique(r):
        m = r == rad
        lin = v[m] * cam.width + u[m]
        zz = z[m].astype(np.float32)
        order = np.lexsort((zz, lin))            # by pixel, nearest first
        lin, zz = lin[order], zz[order]
        first = np.ones(len(lin), bool)
        first[1:] = lin[1:] != lin[:-1]
        layer = np.full(cam.height * cam.width, _BIG, np.float32)
        layer[lin[first]] = zz[first]
        layer = layer.reshape(cam.height, cam.width)
        if rad > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(rad) + 1, 2 * int(rad) + 1))
            layer = cv2.erode(layer, k)
        np.minimum(depth, layer, out=depth)
    depth[depth >= _BIG] = np.inf
    return depth


def depth_at(depth, uv, window_px: int = 2, max_rel_spread: float = 0.05, min_valid_frac: float = 0.6):
    """Robust depth at sub-pixel keypoints: median of the (2w+1)^2 window. NaN where the window is mostly empty
    or straddles a depth edge ((max - min) / median > max_rel_spread): a keypoint on an occlusion boundary has
    no single depth, and guessing one is how wrong landmarks get made."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    h, w = depth.shape
    out = np.full(len(uv), np.nan)
    if len(uv) == 0:
        return out
    u = np.rint(uv[:, 0]).astype(np.int64)
    v = np.rint(uv[:, 1]).astype(np.int64)
    pad = np.pad(depth, window_px, constant_values=np.inf)
    offs = np.arange(-window_px, window_px + 1)
    du, dv = np.meshgrid(offs, offs)
    win = pad[(v[:, None] + window_px + dv.reshape(1, -1)), (u[:, None] + window_px + du.reshape(1, -1))]
    finite = np.isfinite(win)
    frac = finite.mean(axis=1)
    win = np.where(finite, win, np.nan)
    ok = frac >= min_valid_frac
    if not np.any(ok):
        return out
    med = np.nanmedian(win[ok], axis=1)
    spread = (np.nanmax(win[ok], axis=1) - np.nanmin(win[ok], axis=1)) / med
    good = spread <= max_rel_spread
    idx = np.nonzero(ok)[0]
    out[idx[good]] = med[good]
    return out

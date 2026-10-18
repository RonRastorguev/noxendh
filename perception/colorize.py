"""Colour lidar points from the camera frame of the same scan (DH-24).

Each point is projected into the image with the camera calibration ({K, dist, T_cam_from_sensor, image_size}, the
dict fuse.py reads) and takes that pixel's colour. A point the camera cannot see, because something nearer to the
camera covers the same pixels, keeps no colour, so an object's colour never bleeds onto the table behind it. That
is a coarse z-buffer: per block of pixels, only points within a small depth margin of the nearest one are visible.
"""
from __future__ import annotations

import numpy as np


def colorize(xyz: np.ndarray, image_bgr: np.ndarray | None, cam: dict | None, *, block_px: int = 4,
             depth_tol_m: float = 0.04, depth_tol_frac: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """(rgb uint8 (N, 3), seen bool (N,)). Without an image or a calibration, nothing is seen."""
    n = len(xyz)
    rgb = np.zeros((n, 3), np.uint8)
    seen = np.zeros(n, bool)
    if image_bgr is None or not cam or n == 0:
        return rgb, seen
    import cv2

    K = np.asarray(cam["K"], float)
    T = np.asarray(cam["T_cam_from_sensor"], float)
    dist = np.asarray(cam.get("dist", [0, 0, 0, 0, 0]), float)
    h, w = image_bgr.shape[:2]
    cw, ch = cam.get("image_size", [w, h])
    if (cw, ch) != (w, h):                       # calibrated at another resolution: scale the intrinsics
        K = K.copy()
        K[0] *= w / cw
        K[1] *= h / ch
    c = np.asarray(xyz, float) @ T[:3, :3].T + T[:3, 3]
    front = np.flatnonzero(c[:, 2] > 0.02)
    if len(front) == 0:
        return rgb, seen
    uv, _ = cv2.projectPoints(c[front], np.zeros(3), np.zeros(3), K, dist)
    uv = uv.reshape(-1, 2)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx, u, v, z = front[inside], u[inside], v[inside], c[front[inside], 2]
    # coarse z-buffer: the nearest depth per block of pixels
    bx, by = u // block_px, v // block_px
    nbx = (w + block_px - 1) // block_px
    cell = by * nbx + bx
    zmin = np.full(((h + block_px - 1) // block_px) * nbx, np.inf)
    np.minimum.at(zmin, cell, z)
    vis = z <= zmin[cell] + depth_tol_m + depth_tol_frac * z
    smooth = cv2.GaussianBlur(image_bgr, (3, 3), 0)
    rgb[idx[vis]] = smooth[v[vis], u[vis]][:, ::-1]
    seen[idx[vis]] = True
    return rgb, seen

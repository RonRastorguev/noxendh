"""Top-down map image of a cloud for the dashboard (ARCHITECTURE §7.2 map panel; tools/make_topdown.py wraps it)."""
from __future__ import annotations

import numpy as np


def topdown(xyz: np.ndarray, resolution_m: float = 0.004, pad_m: float = 0.02):
    """Height-coloured PNG-ready BGR image + the pixel<->map transform.
    Returns (image, meta) with meta = {origin_xy (map coords of pixel (0,0) corner), resolution_m, width, height};
    pixel (u, v) <-> map (x = origin_x + u*res, y = origin_y + (height - v)*res): image up = map +y."""
    import cv2
    lo, hi = xyz[:, :2].min(axis=0) - pad_m, xyz[:, :2].max(axis=0) + pad_m
    w, h = (np.ceil((hi - lo) / resolution_m).astype(int) + 1)
    u = ((xyz[:, 0] - lo[0]) / resolution_m).astype(int)
    v = (h - 1 - ((xyz[:, 1] - lo[1]) / resolution_m)).astype(int)
    zmap = np.full((h, w), -np.inf, np.float32)
    np.maximum.at(zmap, (v, u), xyz[:, 2])
    filled = np.isfinite(zmap)
    z = np.where(filled, zmap, 0.0)
    zn = np.clip((z - np.percentile(xyz[:, 2], 2)) / max(np.ptp(xyz[:, 2]), 1e-6), 0, 1)
    # cividis (navy -> yellow): colour-blind safe, and the dashboard's coral / teal status pins stand out on it
    img = cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_CIVIDIS)
    # small sampling gaps are closed; large ones are real lidar shadows (never observed) and stay visible, in slate
    closed = cv2.morphologyEx(filled.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
    img = cv2.inpaint(img, ((~filled) & closed).astype(np.uint8) * 255, 2, cv2.INPAINT_TELEA)
    img[~closed] = (52, 46, 42)
    meta = {"origin_xy": [float(lo[0]), float(lo[1])], "resolution_m": resolution_m, "width": int(w), "height": int(h)}
    return img, meta


def map_to_pixel(meta: dict, x: float, y: float) -> tuple[float, float]:
    return ((x - meta["origin_xy"][0]) / meta["resolution_m"], meta["height"] - 1 - (y - meta["origin_xy"][1]) / meta["resolution_m"])

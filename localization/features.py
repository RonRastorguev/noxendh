"""ORB features for mapping and localization (same code both sides, so descriptors are comparable).

Two additions to plain `cv2.ORB`:
- CLAHE contrast normalization first: the mapping camera and the Scout camera see the room at different exposures
  (and different times); local contrast normalization makes FAST + BRIEF less sensitive to that.
- grid bucketing: ORB's strongest corners cluster on the busiest texture; PnP is better conditioned with points
  spread over the image. Each cell keeps its strongest corners, then the rest is filled by response.
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULTS = {
    "n_features": 1500,          # kept after bucketing
    "oversample": 2.0,           # ORB is asked for n_features * oversample, then bucketed down
    "scale_factor": 1.2,
    "n_levels": 8,
    "edge_threshold": 19,
    "fast_threshold": 15,        # provisional; real indoor scenes may need lower
    "grid": [8, 6],              # bucketing cells (cols, rows); [1, 1] disables bucketing
    "clahe_clip": 2.0,           # 0 disables CLAHE
    "clahe_tiles": 8,
}


def _cfg(cfg):
    out = dict(DEFAULTS)
    out.update(cfg or {})
    return out


def to_gray(image) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize(gray, cfg=None) -> np.ndarray:
    c = _cfg(cfg)
    if c["clahe_clip"] and c["clahe_clip"] > 0:
        t = int(c["clahe_tiles"])
        return cv2.createCLAHE(clipLimit=float(c["clahe_clip"]), tileGridSize=(t, t)).apply(gray)
    return gray


def _bucket(kps, n, grid, width, height):
    cols, rows = int(grid[0]), int(grid[1])
    if len(kps) <= n or cols * rows <= 1:
        return list(range(len(kps)))
    resp = np.array([k.response for k in kps])
    xy = np.array([k.pt for k in kps])
    cell = (np.minimum((xy[:, 1] * rows / height).astype(int), rows - 1) * cols
            + np.minimum((xy[:, 0] * cols / width).astype(int), cols - 1))
    per_cell = max(1, n // (cols * rows))
    order = np.lexsort((-resp, cell))                      # by cell, strongest first within a cell
    chosen = []
    taken = np.zeros(len(kps), bool)
    start = 0
    for c in np.unique(cell[order]):
        end = start
        while end < len(order) and cell[order[end]] == c:
            end += 1
        sel = order[start:start + min(per_cell, end - start)]
        chosen.extend(sel.tolist())
        taken[sel] = True
        start = end
    if len(chosen) < n:
        rest = np.where(~taken)[0]
        rest = rest[np.argsort(-resp[rest])][: n - len(chosen)]
        chosen.extend(rest.tolist())
    return chosen[:n]


def detect(image, cfg=None, mask=None) -> tuple:
    """-> (keypoints list, descriptors (N, 32) uint8 or empty (0, 32)), keypoints in input-image pixels."""
    c = _cfg(cfg)
    gray = normalize(to_gray(image), c)
    orb = cv2.ORB_create(nfeatures=int(c["n_features"] * c["oversample"]), scaleFactor=float(c["scale_factor"]),
                         nlevels=int(c["n_levels"]), edgeThreshold=int(c["edge_threshold"]),
                         patchSize=int(c["edge_threshold"]), fastThreshold=int(c["fast_threshold"]))
    kps, des = orb.detectAndCompute(gray, mask)
    if des is None or len(kps) == 0:
        return [], np.zeros((0, 32), np.uint8)
    keep = _bucket(kps, int(c["n_features"]), c["grid"], gray.shape[1], gray.shape[0])
    return [kps[i] for i in keep], des[keep]

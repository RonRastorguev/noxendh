"""Preprocess a scan before comparison (ARCHITECTURE §6.2): ROI crop, voxel downsample, statistical outlier removal.

numpy + scipy only (no Open3D), so it runs in the Windows Python that runs the table demo and in WSL alike.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _oriented(roi: dict) -> bool:
    return roi.get("center") is not None and roi.get("size") is not None


def roi_mask(xyz: np.ndarray, roi: dict | None):
    """Boolean mask of the points inside the ROI, or None when there is no crop. Two forms (sensor/map frame):
    - axis-aligned {min: [x,y,z], max: [x,y,z]};
    - rotated {center: [x, y], size: [w, l], yaw_deg, z: [lo, hi]}: a box turned about the vertical by yaw_deg
      (counter-clockwise from +x), so a crop can follow a diorama that sits at an angle (DH-33)."""
    if not roi:
        return None
    p = np.asarray(xyz, dtype=np.float64)
    if _oriented(roi):
        cx, cy = roi["center"]
        w, l = roi["size"]
        a = np.radians(float(roi.get("yaw_deg", 0.0)))
        dx, dy = p[:, 0] - cx, p[:, 1] - cy
        u = dx * np.cos(a) + dy * np.sin(a)             # along the box's own x (width)
        v = -dx * np.sin(a) + dy * np.cos(a)            # along its y (length)
        zlo, zhi = roi.get("z", (-np.inf, np.inf))
        return (np.abs(u) <= w / 2) & (np.abs(v) <= l / 2) & (p[:, 2] >= zlo) & (p[:, 2] <= zhi)
    if roi.get("min") is None or roi.get("max") is None:
        return None
    lo, hi = np.asarray(roi["min"], dtype=np.float64), np.asarray(roi["max"], dtype=np.float64)
    return np.all((p >= lo) & (p <= hi), axis=1)


def roi_corners(roi: dict | None):
    """The ROI box's 8 corners (8, 3), or None (no crop)."""
    if not roi:
        return None
    if _oriented(roi):
        cx, cy = roi["center"]
        w, l = roi["size"]
        a = np.radians(float(roi.get("yaw_deg", 0.0)))
        zlo, zhi = roi["z"]
        ex, ey = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        return np.array([[*(np.array([cx, cy]) + su * w / 2 * ex + sv * l / 2 * ey), z]
                         for su in (-1, 1) for sv in (-1, 1) for z in (zlo, zhi)])
    if roi.get("min") is None or roi.get("max") is None:
        return None
    lo, hi = roi["min"], roi["max"]
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def roi_bounds(roi: dict | None):
    """Axis-aligned (lo, hi) enclosing the ROI box, or None."""
    c = roi_corners(roi)
    return None if c is None else (c.min(axis=0), c.max(axis=0))


def crop_roi(xyz: np.ndarray, intensity: np.ndarray, roi: dict | None):
    """Keep points inside the ROI (axis-aligned or rotated box, see roi_mask). None or nulls = no crop.
    This is what keeps the judges' hands and bodies out of the diff (§6.2)."""
    keep = roi_mask(xyz, roi)
    if keep is None:
        return xyz, intensity
    return xyz[keep], intensity[keep]


def voxel_downsample(xyz: np.ndarray, intensity: np.ndarray, voxel: float):
    """One point per occupied voxel: the mean position and mean intensity of the points in it."""
    if len(xyz) == 0 or voxel <= 0:
        return xyz, intensity
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse, xyz)
    isum = np.zeros(len(counts))
    np.add.at(isum, inverse, intensity)
    return (sums / counts[:, None]).astype(np.float32), (isum / counts).astype(np.float32)


def remove_outliers(xyz: np.ndarray, intensity: np.ndarray, k: int = 16, std_ratio: float = 2.0):
    """Statistical outlier removal: drop points whose mean distance to their k neighbours is more than
    std_ratio standard deviations above the cloud-wide mean of that quantity."""
    if len(xyz) <= k:
        return xyz, intensity
    dist, _ = cKDTree(xyz).query(xyz, k=k + 1)
    mean_d = dist[:, 1:].mean(axis=1)
    keep = mean_d <= mean_d.mean() + std_ratio * mean_d.std()
    return xyz[keep], intensity[keep]


def for_registration(xyz: np.ndarray, intensity: np.ndarray, cfg: dict, seed: int = 0):
    """The cloud ICP should see: ROI crop + outlier removal + a RANDOM subsample (cfg reg_max_points, default 40 000).
    Not the voxel grid: two clouds snapped to the same 1 cm lattice have a false minimum where the lattices line up
    (synthetic test: knocked 2-3 cm, ICP stuck 7.6 mm off; random subsample 2.9-3.6 mm). DH-14."""
    xyz, intensity = crop_roi(xyz, intensity, cfg.get("roi"))
    xyz, intensity = remove_outliers(xyz, intensity, int(cfg.get("outlier_k", 16)), float(cfg.get("outlier_std", 2.0)))
    n = int(cfg.get("reg_max_points", 40_000))
    if len(xyz) > n:
        idx = np.sort(np.random.default_rng(seed).choice(len(xyz), n, replace=False))
        xyz, intensity = xyz[idx], intensity[idx]
    return xyz, intensity


def preprocess(xyz: np.ndarray, intensity: np.ndarray, cfg: dict):
    """The §6.2 chain with the table-demo config keys (roi, voxel_m; outlier_k / outlier_std optional)."""
    xyz, intensity = crop_roi(xyz, intensity, cfg.get("roi"))
    xyz, intensity = voxel_downsample(xyz, intensity, float(cfg.get("voxel_m", 0.01)))
    xyz, intensity = remove_outliers(xyz, intensity, int(cfg.get("outlier_k", 16)), float(cfg.get("outlier_std", 2.0)))
    return xyz, intensity


def veil_mask(xyz: np.ndarray, origin, *, cos_thr: float = 0.2, k: int = 16) -> np.ndarray:
    """Mask of lidar 'veil' points: returns that land in the gap behind an object's edge, where one beam hit both the
    object and what is behind it (mixed pixels). They lie almost edge-on to the beam, so the angle between the beam
    and the local surface is grazing (|cos| below cos_thr ~ 11°). Measured on Ron's clear bottle (DH-31): 25 % of its
    points, nearly all in the fringe past its edge, which took the floor's colour in the 3D view. Display only: the
    comparison keeps every point."""
    from perception.register import estimate_normals
    p = np.asarray(xyz, dtype=np.float64)
    if len(p) < k + 1:
        return np.zeros(len(p), bool)
    n = estimate_normals(p, k=k)
    ray = p - np.asarray(origin, dtype=np.float64)
    ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-9)
    return np.abs(np.sum(n * ray, axis=1)) < cos_thr

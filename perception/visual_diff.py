"""Appearance-only change detection from the camera (ARCHITECTURE §6.6): rust, burns, stains that geometry misses.

Pipeline (fixed camera):
  1. align: ECC on grayscale (Euclidean), so a nudged camera or a re-seated rig does not light up every edge;
  2. normalise lighting: match the after-image's Lab lightness median/spread and colour-cast (a/b medians) to the
     before-image. Robust statistics (median, MAD), so the change itself barely moves them. The venue's lighting
     WILL drift between baseline and rescan (§6.6); this is the first line of defence, re-baselining the second;
  3. per-pixel colour difference in CIE Lab (Delta E 1976), Gaussian blur, threshold `delta_e_threshold`;
  4. morphological open/close, then connected components >= `min_area_px`;
  5. `mask` (255 = consider) removes everything outside the projected ROI and every region already explained by a
     geometric event (projected bounding boxes): §6.6.
Back-projection of a region to the map (depth of the nearest lidar points) belongs to fuse.py: it needs the camera
calibration (§6.9), which does not exist yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Region:
    bbox: tuple                   # (x, y, w, h) pixels in the BEFORE image
    area_px: int
    centroid: tuple               # (x, y) pixels
    mean_delta_e: float

    def as_dict(self) -> dict:
        return {"bbox": [int(v) for v in self.bbox], "area_px": int(self.area_px),
                "centroid": [round(float(v), 1) for v in self.centroid], "mean_delta_e": round(self.mean_delta_e, 1)}


@dataclass
class VisualDiff:
    regions: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _lab(bgr: np.ndarray) -> np.ndarray:
    """float32 CIE Lab: L 0..100, a/b about -128..127."""
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def _robust(x: np.ndarray) -> tuple[float, float]:
    med = float(np.median(x))
    return med, float(np.median(np.abs(x - med)) * 1.4826 + 1e-6)


def align_ecc(before: np.ndarray, after: np.ndarray, iterations: int = 60, eps: float = 1e-5):
    """Warp `after` onto `before` (Euclidean ECC on blurred grayscale). Returns (warped, valid_mask, warp, ok)."""
    g0 = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    g1 = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    h, w = g0.shape
    try:
        _, warp = cv2.findTransformECC(g0, g1, warp, cv2.MOTION_EUCLIDEAN,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps), None, 5)
        ok = True
    except cv2.error:
        ok = False
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    warped = cv2.warpAffine(after, warp, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.full((h, w), 255, np.uint8), warp, (w, h), flags=flags, borderValue=0)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))           # drop the replicated border band
    return warped, valid, warp, ok


def normalize_to(before_lab: np.ndarray, after_lab: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Match after's L median/spread and a/b medians to before's, over the valid pixels."""
    m = valid > 0
    out = after_lab.copy()
    l0, s0 = _robust(before_lab[..., 0][m])
    l1, s1 = _robust(after_lab[..., 0][m])
    out[..., 0] = (after_lab[..., 0] - l1) * (s0 / s1) + l0
    for c in (1, 2):
        out[..., c] = after_lab[..., c] - np.median(after_lab[..., c][m]) + np.median(before_lab[..., c][m])
    return out


def visual_diff(before_bgr: np.ndarray, after_bgr: np.ndarray, cfg: dict, mask: np.ndarray | None = None) -> VisualDiff:
    v = cfg.get("visual", {})
    thr = float(v.get("delta_e_threshold", 18))
    min_area = int(v.get("min_area_px", 400))
    blur = int(v.get("blur_px", 5)) | 1
    morph = int(v.get("morph_px", 5))
    h, w = before_bgr.shape[:2]
    stats = {}
    if v.get("ecc", True):
        after_bgr, valid, warp, ok = align_ecc(before_bgr, after_bgr)
        stats.update(ecc_ok=ok, ecc_shift_px=[round(float(warp[0, 2]), 2), round(float(warp[1, 2]), 2)],
                     ecc_rot_deg=round(float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))), 3))
    else:
        valid = np.full((h, w), 255, np.uint8)
    if mask is not None:
        valid = cv2.bitwise_and(valid, mask.astype(np.uint8))
    lab0, lab1 = _lab(before_bgr), _lab(after_bgr)
    if v.get("normalize", True):
        lab1 = normalize_to(lab0, lab1, valid)
    de = np.linalg.norm(lab0 - lab1, axis=2)
    de = cv2.GaussianBlur(de, (blur, blur), 0)
    binary = ((de > thr) & (valid > 0)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
    binary = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, labels, cc, cent = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions = []
    for i in range(1, n):
        x, y, bw, bh, area = cc[i]
        if area < min_area:
            continue
        regions.append(Region((x, y, bw, bh), int(area), (float(cent[i][0]), float(cent[i][1])),
                              float(de[labels == i].mean())))
    stats.update(n_components=int(n - 1), n_regions=len(regions), delta_e_p99=round(float(np.percentile(de[valid > 0], 99)), 1))
    return VisualDiff(sorted(regions, key=lambda r: -r.area_px), stats)


def bbox_mask(shape, boxes, pad: int = 8) -> np.ndarray:
    """255 everywhere except inside the given (x, y, w, h) boxes (padded): to exclude regions geometry already explains."""
    m = np.full(shape[:2], 255, np.uint8)
    for x, y, w, h in boxes:
        cv2.rectangle(m, (int(x) - pad, int(y) - pad), (int(x + w) + pad, int(y + h) + pad), 0, thickness=-1)
    return m

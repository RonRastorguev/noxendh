"""Synthetic fixed-rig scans for developing and testing the pipeline without hardware.

Ray-casts an L2-like scan of a diorama: a table plane (z = 0) plus axis-aligned boxes, from a sensor above it.
Because it is ray-cast, occlusion is real: an added object hides the table behind it, and a removed one reveals
table that was never observed. That is exactly what `geo_diff`'s visibility checks must handle.

The noise model is a stand-in until the noise test on the real L2 replaces it (CALIBRATION, DH-12):
Gaussian range noise (default sigma 1 cm; the L2 datasheet says +-2 cm) and a non-repetitive-like random ray pattern.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Box:
    lo: tuple            # (x, y, z) min corner, metres, sensor/map frame
    hi: tuple            # (x, y, z) max corner
    intensity: float = 0.6
    name: str = ""


@dataclass
class Scene:
    boxes: list = field(default_factory=list)
    table_intensity: float = 0.3

    def with_box(self, box: Box) -> "Scene":
        return Scene(self.boxes + [box], self.table_intensity)

    def without(self, name: str) -> "Scene":
        return Scene([b for b in self.boxes if b.name != name], self.table_intensity)

    def moved(self, name: str, dx: float, dy: float) -> "Scene":
        out = []
        for b in self.boxes:
            if b.name == name:
                b = Box((b.lo[0] + dx, b.lo[1] + dy, b.lo[2]), (b.hi[0] + dx, b.hi[1] + dy, b.hi[2]), b.intensity, b.name)
            out.append(b)
        return Scene(out, self.table_intensity)


# Ron's diorama (2026-10-17): a 0.6 x 1.0 m rectangle, long axis = map y; "top" of the rectangle = +y.
# Two shelves along the top end, an aisle down the middle, a tank (gets rust -> flagged), a crate (moved -> routine),
# and debris that gets dropped in the aisle (-> flagged). 1:10 scale. Real dimensions get measured on Saturday.
TABLE_X = (-0.30, 0.30)
TABLE_Y = (-0.50, 0.50)
ROI = {"min": [-0.33, -0.53, -0.03], "max": [0.33, 0.53, 0.40]}
DEBRIS = Box((-0.045, 0.10, 0.0), (0.045, 0.16, 0.06), 0.80, "debris")      # a dropped drum lying on its side in the aisle;
# must stand clear of the change threshold tau (4 cm): a 4 cm-tall object read as "table" (DH-20)
CRATE_MOVE = (0.0, 0.18)                                                      # crate slid 18 cm along the right side


def diorama() -> Scene:
    return Scene([
        Box((-0.27, 0.38, 0.0), (-0.06, 0.46, 0.20), 0.55, "shelf_left"),
        Box((0.06, 0.38, 0.0), (0.27, 0.46, 0.20), 0.55, "shelf_right"),
        Box((-0.24, -0.05, 0.0), (-0.12, 0.07, 0.13), 0.45, "tank"),
        Box((0.12, -0.25, 0.0), (0.20, -0.17, 0.06), 0.70, "crate"),
        Box((-0.25, -0.43, 0.0), (-0.13, -0.33, 0.05), 0.60, "pallet"),
    ])


def ray_cast(scene: Scene, origin, directions: np.ndarray, plane_z: float | None = 0.0):
    """Nearest hit of each unit ray against the boxes and the table plane (an infinite plane at `plane_z`; None = no
    plane, e.g. when the scene models a finite table and a floor as boxes). Returns (t, intensity); t = inf = miss."""
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    n = len(d)
    t_best = np.full(n, np.inf)
    inten = np.zeros(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        for b in scene.boxes:
            lo, hi = np.asarray(b.lo), np.asarray(b.hi)
            t1, t2 = (lo - o) * inv, (hi - o) * inv
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 0) & (tmin < t_best)
            t_best[hit] = tmin[hit]
            inten[hit] = b.intensity
        if plane_z is not None:
            tz = (plane_z - o[2]) / d[:, 2]
            hit = (d[:, 2] < 0) & (tz > 0) & (tz < t_best)
            t_best[hit] = tz[hit]
            inten[hit] = scene.table_intensity
    return t_best, inten


def scan(scene: Scene, *, origin=(0.0, 0.0, 0.7), n_rays: int = 150_000, half_angle_deg: float = 45.0,
         range_sigma: float = 0.01, intensity_sigma: float = 0.03, seed: int = 0, plane_z: float | None = 0.0):
    """One accumulated fixed-rig scan: (xyz float32 (N,3), intensity float32 (N,)) in the sensor/map frame."""
    rng = np.random.default_rng(seed)
    # random directions in a downward cone (uniform in solid angle), like an accumulated non-repetitive pattern
    cos_max = np.cos(np.radians(half_angle_deg))
    cz = rng.uniform(cos_max, 1.0, n_rays)
    phi = rng.uniform(0, 2 * np.pi, n_rays)
    sz = np.sqrt(1 - cz ** 2)
    d = np.stack([sz * np.cos(phi), sz * np.sin(phi), -cz], axis=1)
    t, inten = ray_cast(scene, origin, d, plane_z)
    ok = np.isfinite(t)
    t = t[ok] + rng.normal(0.0, range_sigma, ok.sum())
    xyz = np.asarray(origin) + d[ok] * t[:, None]
    inten = np.clip(inten[ok] + rng.normal(0.0, intensity_sigma, ok.sum()), 0, 1)
    return xyz.astype(np.float32), inten.astype(np.float32)


# --- synthetic camera, consistent with the scans above (for UI / pipeline development without hardware) -----------

def camera_model(origin=(0.0, 0.0, 0.7), width: int = 480, height: int = 640, hfov_deg: float = 64.0) -> dict:
    """A pinhole camera at the lidar origin looking straight down (camera z = -map z, camera x = map x, camera y = -map y).
    Portrait (480 x 640, 64 deg horizontal) so the whole 0.6 x 1.0 m diorama is in frame from 0.7 m.
    Same dict shape fuse.py reads from a real calibration: K, dist, T_cam_from_sensor, image_size."""
    fx = (width / 2) / np.tan(np.radians(hfov_deg) / 2)
    K = [[fx, 0.0, width / 2], [0.0, fx, height / 2], [0.0, 0.0, 1.0]]
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    t = -R @ np.asarray(origin, float)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return {"K": K, "dist": [0, 0, 0, 0, 0], "T_cam_from_sensor": T.tolist(), "image_size": [width, height]}


_COLORS = {"shelf_left": (150, 150, 150), "shelf_right": (150, 150, 150), "crate": (60, 110, 160),
           "tank": (205, 205, 210), "pallet": (90, 140, 175), "debris": (40, 60, 170)}


def render_image(scene: Scene, cam: dict, *, rust_on: str | None = None, seed: int = 0, gain: float = 1.0) -> np.ndarray:
    """BGR image of the scene from `cam`: textured table, box tops drawn far-to-near (painter's algorithm).
    `rust_on` paints an orange-brown patch on that box's top (an appearance-only change)."""
    import cv2
    w, h = cam["image_size"]
    K, T = np.asarray(cam["K"]), np.asarray(cam["T_cam_from_sensor"])
    rng = np.random.default_rng(seed)
    grain = cv2.GaussianBlur(np.random.default_rng(7).normal(0, 16, (h, w)).astype(np.float32), (0, 0), 3)
    img = np.empty((h, w, 3), np.float32)
    img[:] = (52, 54, 58)                                      # floor around the diorama
    corners = [(TABLE_X[0], TABLE_Y[0], 0), (TABLE_X[1], TABLE_Y[0], 0), (TABLE_X[1], TABLE_Y[1], 0), (TABLE_X[0], TABLE_Y[1], 0)]
    table = np.zeros((h, w), np.uint8)

    def proj(p):
        c = T[:3, :3] @ np.asarray(p, float) + T[:3, 3]
        uv = K @ c
        return (uv[:2] / uv[2]).astype(np.float32)

    cv2.fillConvexPoly(table, np.array([proj(p) for p in corners], np.int32), 255)
    img[table > 0] = (120, 140, 160)
    img += grain[..., None]
    for b in sorted(scene.boxes, key=lambda b: b.hi[2]):
        top = [(b.lo[0], b.lo[1], b.hi[2]), (b.hi[0], b.lo[1], b.hi[2]), (b.hi[0], b.hi[1], b.hi[2]), (b.lo[0], b.hi[1], b.hi[2])]
        poly = np.array([proj(p) for p in top], np.int32)
        col = _COLORS.get(b.name, (90 + 160 * b.intensity, 90 + 120 * b.intensity, 120))
        # visible sides: the silhouette of all 8 corners, shaded darker, then the top face over it
        allc = [(x, y, z) for x in (b.lo[0], b.hi[0]) for y in (b.lo[1], b.hi[1]) for z in (b.lo[2], b.hi[2])]
        hull = cv2.convexHull(np.array([proj(p) for p in allc], np.int32))
        side = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(side, hull, 255)
        img[side > 0] = np.asarray(col, np.float32) * 0.62 + 0.4 * grain[side > 0][:, None]
        layer = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(layer, poly, 255)
        m = layer > 0
        img[m] = np.asarray(col, np.float32) + 0.5 * grain[m][:, None]
        if rust_on == b.name:
            cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
            rw, rh = (poly[:, 0].max() - poly[:, 0].min()) * 0.45, (poly[:, 1].max() - poly[:, 1].min()) * 0.45
            patch = np.zeros((h, w), np.uint8)
            cv2.ellipse(patch, (int(cx), int(cy)), (int(rw), int(rh)), 15, 0, 360, 255, -1)
            pm = (patch > 0) & m
            img[pm] = 0.3 * img[pm] + 0.7 * np.array((35, 75, 150), np.float32)
    img = img * gain + rng.normal(0, 2.5, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


SCENARIOS = ("no_change", "crate_moved", "rust_tank", "dropped_in_aisle", "everything")


def scenario(name: str):
    """(scene, rust_on) for the demo scenarios (ARCHITECTURE §11, Ron's diorama 2026-10-17)."""
    base = diorama()
    if name in ("baseline", "no_change"):
        return base, None
    if name == "crate_moved":
        return base.moved("crate", *CRATE_MOVE), None
    if name == "rust_tank":
        return base, "tank"
    if name == "dropped_in_aisle":
        return base.with_box(DEBRIS), None
    if name == "everything":
        return base.moved("crate", *CRATE_MOVE).with_box(DEBRIS), "tank"
    raise ValueError(f"unknown scenario {name!r}; one of {SCENARIOS}")

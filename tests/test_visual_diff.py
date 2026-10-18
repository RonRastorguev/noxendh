"""Visual diff (§6.6) on synthetic camera frames: a textured diorama image, a rust patch, lighting drift, a nudge.
Runs under pytest or `python tests/test_visual_diff.py`."""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from perception.visual_diff import bbox_mask, visual_diff  # noqa: E402

CFG = {"visual": {"delta_e_threshold": 18, "min_area_px": 400}}
RUST = (35, 75, 150)          # BGR: orange-brown


def scene(seed=0, w=640, h=480):
    """A camera frame of the diorama: wood-ish table texture, grey racks, a pale tank, a crate."""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)
    img[:] = (120, 140, 160)
    grain = cv2.GaussianBlur(rng.normal(0, 18, (h, w)).astype(np.float32), (0, 0), 3)
    img += grain[..., None]
    for (x, y, bw, bh, col) in ((60, 60, 220, 40, (150, 150, 150)), (360, 60, 220, 40, (150, 150, 150)),
                                (430, 260, 120, 120, (200, 205, 210)), (250, 280, 70, 70, (80, 110, 150))):
        img[y:y + bh, x:x + bw] = col
        img[y:y + bh, x:x + bw] += grain[y:y + bh, x:x + bw, None] * 0.5
    return np.clip(img, 0, 255).astype(np.uint8)


def noisy(img, seed, sigma=3.0):
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def rusted(img, x=460, y=290, w=40, h=35):
    out = img.copy()
    patch = out[y:y + h, x:x + w].astype(np.float32)
    out[y:y + h, x:x + w] = np.clip(0.3 * patch + 0.7 * np.array(RUST, np.float32), 0, 255).astype(np.uint8)
    return out


def test_no_change_gives_no_regions():
    base = scene()
    for s in range(3):
        d = visual_diff(noisy(base, 2 * s), noisy(base, 2 * s + 1), CFG)
        assert d.regions == [], (s, [r.as_dict() for r in d.regions], d.stats)


def test_rust_patch_gives_one_region_at_the_patch():
    base = scene()
    d = visual_diff(noisy(base, 1), noisy(rusted(base), 2), CFG)
    assert len(d.regions) == 1, ([r.as_dict() for r in d.regions], d.stats)
    cx, cy = d.regions[0].centroid
    assert abs(cx - 480) < 10 and abs(cy - 307) < 10, d.regions[0].as_dict()


def test_lighting_drift_alone_gives_no_regions():
    base = scene()
    dim = np.clip(base.astype(np.float32) * 0.78 + 8, 0, 255).astype(np.uint8)          # darker, lower contrast
    warm = np.clip(base.astype(np.float32) * np.array([0.92, 1.0, 1.08]), 0, 255).astype(np.uint8)  # warmer cast
    for after in (dim, warm):
        d = visual_diff(noisy(base, 1), noisy(after, 2), CFG)
        assert d.regions == [], ([r.as_dict() for r in d.regions], d.stats)


def test_lighting_drift_plus_rust_gives_exactly_the_rust():
    base = scene()
    after = np.clip(rusted(base).astype(np.float32) * 0.8 + 6, 0, 255).astype(np.uint8)
    d = visual_diff(noisy(base, 1), noisy(after, 2), CFG)
    assert len(d.regions) == 1, ([r.as_dict() for r in d.regions], d.stats)


def test_camera_nudge_is_aligned_away():
    base = scene()
    M = np.float32([[1, 0, 5.0], [0, 1, -3.0]])                    # the rig nudged 5 px right, 3 px up
    nudged = cv2.warpAffine(base, M, (base.shape[1], base.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    d = visual_diff(noisy(base, 1), noisy(nudged, 2), CFG)
    assert d.regions == [] and d.stats["ecc_ok"], ([r.as_dict() for r in d.regions], d.stats)
    assert abs(d.stats["ecc_shift_px"][0] - 5.0) < 0.2 and abs(d.stats["ecc_shift_px"][1] + 3.0) < 0.2, d.stats
    # negative: without alignment the shifted edges become false regions (2 in this scene)
    raw = visual_diff(noisy(base, 1), noisy(nudged, 2), {"visual": dict(CFG["visual"], ecc=False)})
    assert len(raw.regions) > 0, raw.stats


def test_small_speck_is_ignored():
    base = scene()
    d = visual_diff(noisy(base, 1), noisy(rusted(base, w=12, h=12), 2), CFG)
    assert d.regions == [], [r.as_dict() for r in d.regions]


def test_region_explained_by_geometry_is_masked_out():
    base = scene()
    m = bbox_mask(base.shape, [(450, 280, 60, 55)])
    d = visual_diff(noisy(base, 1), noisy(rusted(base), 2), CFG, mask=m)
    assert d.regions == [], [r.as_dict() for r in d.regions]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"  [ok] {name}")
    print(f"{len(tests)}/{len(tests)} passed")

"""Milestone 1 pipeline pieces on SYNTHETIC ray-cast scans (perception/synthetic.py): preprocess, ICP, geometric diff.

The spec's gates (ARCHITECTURE §10/§11) are on recorded real scans; these tests prove the logic first, including the
cases that only a ray-cast scene exposes: an added object must not create "removed" table behind it (occlusion), and a
removed object must not create an "added" patch of newly revealed table. Runs under pytest or `python tests/test_geo_diff.py`.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from perception import synthetic as syn  # noqa: E402
from perception.geo_diff import geo_diff  # noqa: E402
from perception.preprocess import crop_roi, for_registration, preprocess, remove_outliers, voxel_downsample  # noqa: E402
from perception.register import apply, register  # noqa: E402

ORIGIN = (0.0, 0.0, 0.7)
CFG = {"roi": syn.ROI, "voxel_m": 0.01, "diff_threshold_m": 0.04,
       "cluster_eps_m": 0.02, "min_cluster_points": 50, "min_cluster_size_m": 0.04,
       "icp": {"max_rmse_m": 0.01, "min_fitness": 0.8}}
DEBRIS = syn.DEBRIS                     # something dropped in the aisle (Ron's diorama)


def _scan(scene, seed):
    xyz, inten = syn.scan(scene, origin=ORIGIN, seed=seed)
    return preprocess(xyz, inten, CFG)


def _diff(base_scene, cur_scene, seeds=(1, 2)):
    b = _scan(base_scene, seeds[0])
    c = _scan(cur_scene, seeds[1])
    return geo_diff(b[0], b[1], c[0], c[1], CFG, base_origin=ORIGIN, cur_origin=ORIGIN)


def _near(cluster, box, tol=0.03):
    c = (cluster.bbox_min + cluster.bbox_max) / 2
    bc = (np.asarray(box.lo) + np.asarray(box.hi)) / 2
    return np.linalg.norm(c[:2] - bc[:2]) < tol + 0.5 * max(box.hi[0] - box.lo[0], box.hi[1] - box.lo[1])


def test_preprocess_crop_voxel_outliers():
    rng = np.random.default_rng(0)
    xyz = np.vstack([rng.uniform(-0.3, 0.3, (5000, 3)), [[5, 5, 5]]]).astype(np.float32)
    inten = np.ones(len(xyz), np.float32)
    c, ci = crop_roi(xyz, inten, {"min": [-1, -1, -1], "max": [1, 1, 1]})
    assert len(c) == 5000                                      # the far point is cropped
    v, vi = voxel_downsample(c, ci, 0.05)
    assert len(v) < len(c) and np.allclose(vi, 1.0)
    cloud = np.vstack([rng.normal(0, 0.01, (2000, 3)), [[0.5, 0.5, 0.5]]]).astype(np.float32)
    r, _ = remove_outliers(cloud, np.ones(len(cloud), np.float32))
    assert len(r) < len(cloud) and not np.any(np.all(np.isclose(r, [0.5, 0.5, 0.5]), axis=1))


def test_no_change_gives_no_clusters():
    """The false-alarm test (§11): two independent noisy scans of the same scene -> zero events."""
    for seeds in ((1, 2), (3, 4), (5, 6)):
        d = _diff(syn.diorama(), syn.diorama(), seeds)
        assert d.added == [] and d.removed == [], (seeds, [c.as_dict() for c in d.added + d.removed], d.stats)


def test_debris_dropped_gives_one_added_cluster_and_no_removed():
    """§10 Milestone 1 gate shape: object dropped -> exactly 1 added cluster, 0 false clusters (3 trials)."""
    for seeds in ((1, 2), (3, 4), (5, 6)):
        d = _diff(syn.diorama(), syn.diorama().with_box(DEBRIS), seeds)
        assert len(d.added) == 1 and _near(d.added[0], DEBRIS), ([c.as_dict() for c in d.added], d.stats)
        assert d.removed == [], ([c.as_dict() for c in d.removed], d.stats)


def test_tall_object_near_the_edge_shadows_the_table_without_false_removed():
    """Occlusion (§6.4): a tall stack set down off-centre hides a patch of table from the sensor. Those baseline table
    points are missing from the scan but must count as OCCLUDED, never as removed. (The aisle debris sits nearly under
    the sensor and casts almost no shadow, so it cannot prove this.)"""
    stack = syn.Box((0.18, -0.40, 0.0), (0.26, -0.32, 0.14), 0.65, "stack")
    for seeds in ((1, 2), (3, 4)):
        d = _diff(syn.diorama(), syn.diorama().with_box(stack), seeds)
        assert len(d.added) == 1 and _near(d.added[0], stack), ([c.as_dict() for c in d.added], d.stats)
        assert d.removed == [], ([c.as_dict() for c in d.removed], d.stats)
        assert d.stats["n_occluded"] > 0, d.stats


def test_crate_removed_gives_one_removed_cluster_and_no_added():
    base = syn.diorama()
    crate = next(b for b in base.boxes if b.name == "crate")
    for seeds in ((1, 2), (3, 4)):
        d = _diff(base, base.without("crate"), seeds)
        assert len(d.removed) == 1 and _near(d.removed[0], crate), ([c.as_dict() for c in d.removed], d.stats)
        assert d.added == [], ([c.as_dict() for c in d.added], d.stats)          # the table under it is revealed
        assert d.stats["n_revealed"] > 0


def test_crate_moved_gives_one_removed_and_one_added():
    """Input to move association (§6.5, Milestone 2): a move shows as one removed + one added cluster."""
    base = syn.diorama()
    d = _diff(base, base.moved("crate", *syn.CRATE_MOVE))
    assert len(d.removed) == 1 and len(d.added) == 1, ([c.as_dict() for c in d.added + d.removed], d.stats)
    a, r = d.added[0], d.removed[0]
    assert abs(a.n_points - r.n_points) / max(a.n_points, r.n_points) < 0.5      # similar size: associate-able


def _reg_scan(seed):
    xyz, inten = syn.scan(syn.diorama(), origin=ORIGIN, seed=seed)
    return for_registration(xyz, inten, CFG, seed=seed)


def _knock(dx, deg=1.0):
    th = np.radians(deg)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    T[:3, 3] = [dx, -0.7 * dx, 0.003]
    return T


def _errors(T_est, T_true):
    E = T_est @ T_true
    return np.linalg.norm(E[:3, 3]), np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))


def test_icp_identity_on_a_fixed_rig_does_not_drift():
    reg = register(_reg_scan(2)[0], _reg_scan(1)[0], CFG)
    err_t, err_r = _errors(reg.T, np.eye(4))
    assert reg.ok and err_t < 0.002 and err_r < 0.2, (err_t, err_r, reg.as_dict())


def test_icp_recovers_a_knock_and_accepts():
    """Knocked 0.5-3 cm + 1 deg. Registration uses the random subsample, not the voxel grid (DH-14)."""
    b, c = _reg_scan(1)[0], _reg_scan(2)[0]
    for dx in (0.005, 0.01, 0.02, 0.03):
        T_true = _knock(dx)
        reg = register(apply(T_true, c), b, CFG)
        err_t, err_r = _errors(reg.T, T_true)
        assert reg.ok and err_t < 0.005 and err_r < 0.2, (dx, err_t, err_r, reg.as_dict())


FLAT_SCENE = syn.Scene([       # the first, plane-dominated test layout (low racks, few vertical faces)
    syn.Box((-0.30, -0.25, 0.0), (-0.10, -0.20, 0.15), 0.55, "rack_a"),
    syn.Box((0.10, -0.25, 0.0), (0.30, -0.20, 0.15), 0.55, "rack_b"),
    syn.Box((-0.05, 0.05, 0.0), (0.03, 0.13, 0.06), 0.70, "crate"),
    syn.Box((0.18, 0.08, 0.0), (0.28, 0.18, 0.12), 0.45, "tank"),
])


def test_voxel_grid_registration_has_the_lattice_minimum():
    """Negative test: documents WHY registration avoids the voxel grid. On a plane-dominated scene, two clouds snapped
    to the same 1 cm lattice stick ~7.6 mm off for a 2 cm knock (DH-14). Ron's diorama has tall shelves and does not
    show it, but the real one may be flatter, so registration keeps the random subsample."""
    flat_cfg = dict(CFG, roi={"min": [-0.45, -0.35, -0.03], "max": [0.45, 0.35, 0.40]})
    b = preprocess(*syn.scan(FLAT_SCENE, origin=ORIGIN, seed=1), flat_cfg)[0]
    c = preprocess(*syn.scan(FLAT_SCENE, origin=ORIGIN, seed=2), flat_cfg)[0]
    T_true = _knock(0.02)
    err_t, _ = _errors(register(apply(T_true, c), b, flat_cfg).T, T_true)
    assert err_t > 0.005, err_t


def test_icp_rejects_a_badly_aligned_scan():
    """Rotated 90 deg about the vertical: not recoverable from identity, must be refused, never diffed (§6.3)."""
    b, c = _reg_scan(1)[0], _reg_scan(2)[0]
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    reg = register(apply(T, c), b, CFG)
    assert not reg.ok, reg.as_dict()


if __name__ == "__main__":
    import time
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        t0 = time.time()
        fn()
        print(f"  [ok] {name} ({time.time() - t0:.1f} s)")
    print(f"{len(tests)}/{len(tests)} passed")

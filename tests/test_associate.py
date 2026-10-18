"""Move association (§6.5, spec §11: "synthetic clusters for move matching") plus end-to-end on ray-cast scenes.
Runs under pytest or `python tests/test_associate.py`."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from perception import synthetic as syn  # noqa: E402
from perception.associate import associate  # noqa: E402
from perception.geo_diff import Cluster, geo_diff  # noqa: E402
from perception.preprocess import preprocess  # noqa: E402

CFG = {"roi": syn.ROI, "voxel_m": 0.01, "diff_threshold_m": 0.04,
       "cluster_eps_m": 0.02, "min_cluster_points": 50, "min_cluster_size_m": 0.04,
       "association": {"dim_tol": 0.15, "min_score": 0.6}}
ORIGIN = (0.0, 0.0, 0.7)


def C(kind, center, size, n, inten):
    c, s = np.asarray(center, float), np.asarray(size, float)
    return Cluster(kind, np.arange(n), c, c - s / 2, c + s / 2, n, inten)


def test_same_object_moved_is_paired():
    r = C("removed", (0, 0, 0.03), (0.08, 0.08, 0.06), 120, 0.70)
    a = C("added", (-0.2, 0.1, 0.03), (0.08, 0.075, 0.058), 110, 0.69)
    out = associate([r], [a], CFG)
    assert len(out.moved) == 1 and not out.added and not out.removed
    assert abs(out.moved[0].displacement[0] + 0.2) < 1e-9


def test_rotated_90_degrees_still_paired():
    r = C("removed", (0, 0, 0.05), (0.20, 0.05, 0.10), 200, 0.55)
    a = C("added", (0.1, 0.2, 0.05), (0.05, 0.20, 0.10), 190, 0.55)
    assert len(associate([r], [a], CFG).moved) == 1


def test_different_size_is_not_a_move():
    r = C("removed", (0, 0, 0.03), (0.08, 0.08, 0.06), 120, 0.70)
    a = C("added", (0.2, 0.0, 0.05), (0.08, 0.08, 0.10), 120, 0.70)       # 67 % taller: a different object
    out = associate([r], [a], CFG)
    assert not out.moved and len(out.added) == 1 and len(out.removed) == 1


def test_different_material_lowers_the_score():
    r = C("removed", (0, 0, 0.03), (0.08, 0.08, 0.06), 120, 0.20)
    a = C("added", (0.2, 0.0, 0.03), (0.08, 0.08, 0.06), 120, 0.80)       # same box shape, very different reflectance
    out = associate([r], [a], CFG)
    assert out.moved and out.moved[0].parts["intensity"] == 0.0 and out.moved[0].score < 0.8


def test_greedy_pairs_each_object_with_its_twin():
    crate_r = C("removed", (0, 0, 0.03), (0.08, 0.08, 0.06), 120, 0.70)
    drum_r = C("removed", (0.2, 0.2, 0.04), (0.06, 0.06, 0.08), 90, 0.85)
    crate_a = C("added", (-0.2, 0.0, 0.03), (0.08, 0.08, 0.06), 118, 0.71)
    drum_a = C("added", (0.25, -0.1, 0.04), (0.06, 0.06, 0.08), 92, 0.84)
    out = associate([crate_r, drum_r], [drum_a, crate_a], CFG)
    pairs = {(m.removed.mean_intensity, m.added.mean_intensity) for m in out.moved}
    assert pairs == {(0.70, 0.71), (0.85, 0.84)}


def test_extra_new_object_stays_added():
    crate_r = C("removed", (0, 0, 0.03), (0.08, 0.08, 0.06), 120, 0.70)
    crate_a = C("added", (-0.2, 0.0, 0.03), (0.08, 0.08, 0.06), 118, 0.71)
    drum_a = C("added", (0.25, -0.1, 0.04), (0.06, 0.06, 0.08), 92, 0.84)
    out = associate([crate_r], [crate_a, drum_a], CFG)
    assert len(out.moved) == 1 and len(out.added) == 1 and out.added[0].mean_intensity == 0.84 and not out.removed


def _diff(base, cur, seeds=(1, 2)):
    b = preprocess(*syn.scan(base, origin=ORIGIN, seed=seeds[0]), CFG)
    c = preprocess(*syn.scan(cur, origin=ORIGIN, seed=seeds[1]), CFG)
    return geo_diff(b[0], b[1], c[0], c[1], CFG, base_origin=ORIGIN, cur_origin=ORIGIN)


def test_scene_crate_moved_gives_one_moved_event_three_trials():
    """Milestone 2 gate shape (§10): crate move -> 1 moved event, 3/3 trials."""
    base = syn.diorama()
    for seeds in ((1, 2), (3, 4), (5, 6)):
        d = _diff(base, base.moved("crate", *syn.CRATE_MOVE), seeds)
        out = associate(d.removed, d.added, CFG)
        assert len(out.moved) == 1 and not out.added and not out.removed, (seeds, [m.as_dict() for m in out.moved],
                                                                           [c.as_dict() for c in out.added + out.removed])
        assert abs(out.moved[0].as_dict()["distance_m"] - np.hypot(*syn.CRATE_MOVE)) < 0.03


def test_scene_crate_moved_and_debris_dropped():
    base = syn.diorama()
    d = _diff(base, base.moved("crate", *syn.CRATE_MOVE).with_box(syn.DEBRIS))
    out = associate(d.removed, d.added, CFG)
    assert len(out.moved) == 1 and len(out.added) == 1 and not out.removed, ([m.as_dict() for m in out.moved],
                                                                             [c.as_dict() for c in out.added + out.removed])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"  [ok] {name}")
    print(f"{len(tests)}/{len(tests)} passed")

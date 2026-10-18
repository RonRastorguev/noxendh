"""run_scan_comparison end to end on synthetic scan folders (the same folders the server's synthetic source writes).
Runs under pytest or `python tests/test_pipeline.py`."""
from __future__ import annotations

import collections
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import yaml  # noqa: E402

from perception import io as scan_io  # noqa: E402
from perception.classify.cached import RulesClassifier  # noqa: E402
from perception.classify.chain import ClassifierChain  # noqa: E402
from perception.pipeline import run_scan_comparison  # noqa: E402
from perception.register import apply  # noqa: E402
from perception import synthetic as syn  # noqa: E402
from server import sources  # noqa: E402

with open(os.path.join(ROOT, "config", "table_demo.yaml"), encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
CFG["roi"] = sources.SYNTH_ROI
CAM = sources.synthetic_camera()
CHAIN = ClassifierChain([RulesClassifier()], timeout_s=2)


def _run(scenario, seed=11, mutate=None):
    d = tempfile.mkdtemp(prefix="nx_pipe_")
    base = scan_io.load_scan(sources.capture_synthetic(d, "baseline", is_baseline=True, seed=1))
    cur = scan_io.load_scan(sources.capture_synthetic(d, scenario, is_baseline=False, seed=seed))
    if mutate:
        cur = mutate(cur)
    stages = []
    res = run_scan_comparison(base, cur, CFG, crops_dir=os.path.join(d, "crops"), camera=CAM, classifier=CHAIN,
                              progress=lambda stage, detail=None: stages.append(stage))
    return res, stages


def kinds(res):
    return dict(collections.Counter(e["kind"] for e in res.events))


def test_no_change_gives_no_events():
    for seed in (11, 12, 13):
        res, _ = _run("no_change", seed)
        assert res.error is None and res.events == [], (seed, kinds(res), res.stats)


def test_crate_moved_is_one_routine_moved_event_with_crops():
    res, stages = _run("crate_moved")
    assert kinds(res) == {"moved": 1}, (kinds(res), res.stats)
    ev = res.events[0]
    assert ev["classification"]["routine"] is True and ev["classification"]["provider"] == "rules"
    assert ev["crop_located"] and os.path.exists(ev["crop_before"]) and os.path.exists(ev["crop_after"])
    assert abs(ev["geo_score"] - 100 * np.hypot(*syn.CRATE_MOVE)) < 3, ev["geo_score"]
    assert stages[:2] == ["aligning", "comparing"] and "classifying" in stages and stages[-1] == "done"


def test_debris_dropped_in_aisle_is_one_flagged_added_event():
    res, _ = _run("dropped_in_aisle")
    assert kinds(res) == {"added": 1}, (kinds(res), res.stats)
    assert res.events[0]["classification"]["routine"] is False


def test_rust_is_one_appearance_event_placed_on_the_tank():
    res, _ = _run("rust_tank")
    assert kinds(res) == {"appearance": 1}, (kinds(res), res.stats)
    bm = res.events[0]["bbox_map"]
    tank = next(b for b in syn.diorama().boxes if b.name == "tank")
    cx, cy = (tank.lo[0] + tank.hi[0]) / 2, (tank.lo[1] + tank.hi[1]) / 2
    assert bm is not None and abs(bm["center"][0] - cx) < 0.05 and abs(bm["center"][1] - cy) < 0.05, (bm, cx, cy)
    assert abs(bm["center"][2] - tank.hi[2]) < 0.04, bm        # on the tank's top, not the table


def test_everything_scenario():
    res, _ = _run("everything")
    assert kinds(res) == {"moved": 1, "added": 1, "appearance": 1}, (kinds(res), res.stats)


def test_misaligned_scan_is_refused_not_diffed():
    def rotate(scan):
        T = np.eye(4)
        T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        scan.xyz = apply(T, scan.xyz)
        return scan
    res, _ = _run("no_change", mutate=rotate)
    assert res.error and "not aligned" in res.error and res.events == [], (res.error, res.registration)


def _flat_scans(tilt_current=(0.0, 0.0, 9.8)):
    """Two scans of a bare floor from an untouched (or tilted) fixed lidar."""
    # the only structure is one object that is somewhere else in each scan (like people walking through on Ron's
    # floor crop): ICP's structure check cannot be satisfied, although the lidar never moved
    spots = {1: syn.Box((0.10, 0.20, 0.0), (0.25, 0.35, 0.30), 0.6, "visitor"),
             2: syn.Box((-0.25, -0.35, 0.0), (-0.10, -0.20, 0.30), 0.6, "visitor")}
    pose = np.eye(4)
    pose[:3, 3] = sources.SYNTH_ORIGIN
    scans = []
    for seed, accel in ((1, (0.0, 0.0, 9.8)), (2, tilt_current)):
        xyz, inten = syn.scan(syn.Scene([spots[seed]], 0.3), origin=sources.SYNTH_ORIGIN, seed=seed)
        meta = scan_io.make_meta(f"flat{seed}", accumulation_s=10, is_baseline=seed == 1, frame="table",
                                 sensor_pose=pose, roi=CFG["roi"], table_calib_id="table-test",
                                 lidar={"imu_accel_mean": list(accel)})
        scans.append(scan_io.Scan(f"flat{seed}", xyz, inten, None, meta))
    return scans


def test_fixed_rig_keeps_its_pose_when_icp_cannot_confirm_and_the_lidar_is_unmoved():
    base, cur = _flat_scans()
    off = dict(CFG, registration={"fixed_rig_known_pose": False})
    res_off = run_scan_comparison(base, cur, off, crops_dir=tempfile.mkdtemp(), classifier=CHAIN)
    assert res_off.error and "not aligned" in res_off.error, res_off.registration
    on = dict(CFG, registration={"fixed_rig_known_pose": True})
    res_on = run_scan_comparison(base, cur, on, crops_dir=tempfile.mkdtemp(), classifier=CHAIN)
    assert not res_on.error and res_on.registration["ok"] and "known pose" in res_on.registration["reason"], res_on.registration
    assert np.allclose(res_on.registration["T"], np.eye(4)) and res_on.events, res_on.events      # the visitor is flagged


def test_fixed_rig_fallback_refuses_a_moved_lidar():
    base, cur = _flat_scans(tilt_current=(0.0, 1.0, 9.75))               # ~6 degrees of tilt: the rig was moved
    res = run_scan_comparison(base, cur, dict(CFG, registration={"fixed_rig_known_pose": True}),
                              crops_dir=tempfile.mkdtemp(), classifier=CHAIN)
    assert res.error and "not aligned" in res.error, res.registration


if __name__ == "__main__":
    import time
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        t0 = time.time()
        fn()
        print(f"  [ok] {name} ({time.time() - t0:.1f} s)")
    print(f"{len(tests)}/{len(tests)} passed")

"""Scout camera relocalization (localization/, ARCHITECTURE §8.2-8.3) on SYNTHETIC data: no hardware data exists yet.

A textured synthetic room is rendered exactly (localization/synthetic_room.py); its map cloud stands in for the
Point-LIO map, a handheld "mapping walk" gives the keyframes, and a separate Scout flight is the held-out test.
The thresholds below are regression bounds on the synthetic numbers (docs/SCOUT_LOC.md has the measured values),
not real-world accuracy claims. Runs under pytest or `python tests/test_scout_loc.py` (~2 min on this laptop).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from localization import synthetic_eval as ev  # noqa: E402
from localization import synthetic_room as sr  # noqa: E402
from localization.camera import CameraModel  # noqa: E402
from localization.config import body_from_camera, build_localizer, camera_from_config, load_configs  # noqa: E402
from localization.depth_render import depth_at, render_depth  # noqa: E402
from localization.fix import PoseFix  # noqa: E402
from localization.geometry import (R_to_quat_xyzw, interpolate_pose, inv_T, look_at, make_T, pose_error,  # noqa: E402
                                   quat_xyzw_to_R, rot_z)
from localization.landmarks import LandmarkDB  # noqa: E402
from localization.localizer import Localizer  # noqa: E402

_DS = {}


def _dataset():
    if "ds" not in _DS:
        _DS["ds"] = sr.build_dataset(0)
    return _DS["ds"]


# --- units ---------------------------------------------------------------------------------------------------

def test_camera_nominal_model_from_diagonal_fov():
    cam = CameraModel.from_fov(640, 480, 102.0, "diagonal")
    assert abs(cam.fov_deg()["diagonal"] - 102.0) < 1e-6
    assert abs(cam.fx - 323.91) < 0.01                      # 400 px half-diagonal / tan(51 deg)
    big = cam.scaled(2)
    assert abs(big.fov_deg()["diagonal"] - 102.0) < 0.05
    d = CameraModel(640, 480, 330, 331, 321, 238, np.array([-0.3, 0.1, 0.001, -0.001, -0.02]))
    X = np.array([[0.3, -0.2, 1.0], [-0.5, 0.4, 2.0], [0.0, 0.0, 3.0]])
    xy = d.normalized(d.project(X))
    assert np.allclose(xy, X[:, :2] / X[:, 2:3], atol=1e-4)


def test_geometry_quaternion_and_interpolation():
    R = rot_z(0.7) @ look_at((0, 0, 0), (1, 2, 0.5))[:3, :3]
    assert np.allclose(quat_xyzw_to_R(R_to_quat_xyzw(R)), R, atol=1e-9)
    T0, T1 = make_T(np.eye(3), (0, 0, 0)), make_T(rot_z(np.pi / 2), (2, 0, 0))
    Tm = interpolate_pose([0.0, 1.0], [T0, T1], 0.5)
    pe, re = pose_error(Tm, make_T(rot_z(np.pi / 4), (1, 0, 0)))
    assert pe < 1e-9 and re < 1e-6
    assert interpolate_pose([0.0, 1.0], [T0, T1], 1.5) is None
    assert np.allclose(inv_T(T1) @ T1, np.eye(4))


def test_zbuffer_near_surface_wins_and_depth_edges_are_rejected():
    """A near square in front of a far wall: through the near one's point gaps the wall must not show."""
    cam = CameraModel.from_fov(320, 240, 90.0, "horizontal")
    s = 0.02
    g = np.arange(-1.0, 1.0, s)
    wall = np.array([[x, y, 3.0] for x in np.arange(-4, 4, s * 3) for y in np.arange(-3, 3, s * 3)])
    near = np.array([[x, y, 1.0] for x in g[g < 0] for y in g])          # covers the left half of the view
    depth = render_depth(np.vstack([wall, near]), np.eye(4), cam, spacing_m=s * 3)
    left = depth[60:180, 20:140]
    right = depth[60:180, 180:300]
    assert np.all(np.abs(left - 1.0) < 0.01), np.unique(np.round(left, 2))   # no wall seen through the gaps
    fin = np.isfinite(right)                                  # small holes are allowed (= no data, never wrong data)
    assert fin.mean() > 0.9 and np.all(np.abs(right[fin] - 3.0) < 0.01)
    d = depth_at(depth, [[80, 120], [250, 120], [cam.cx, 120]], window_px=3)
    assert abs(d[0] - 1.0) < 0.01 and abs(d[1] - 3.0) < 0.01
    assert np.isnan(d[2])                                     # on the occlusion edge: no depth, no landmark


class _Scripted:
    """Estimator that returns pre-scripted fixes (None = no fix)."""
    def __init__(self, poses):
        self.poses = list(poses)

    def estimate(self, image, T_prior=None):
        T = self.poses.pop(0)
        if T is None:
            return PoseFix(False, None, 3, 10, "scripted", reason="below threshold")
        return PoseFix(True, T, 100, 120, "scripted")


def test_localizer_state_machine():
    P = lambda x: make_T(np.eye(3), (x, 0.0, 0.0))          # noqa: E731
    seq = ([P(0.00), P(0.01), P(0.02)] + [None] * 10          # 3 fixes, then 1.0 s without
           + [None, P(0.10), P(5.0), P(0.12)]                  # > 1 s -> lost; reacquire; a 5 m jump breaks it
           + [P(0.13), P(0.14)])
    loc = Localizer([_Scripted(seq)], {"reacquire_fixes": 2, "loc_lost_hold_s": 1.0})
    outs = [loc.process(None, 0.1 * k) for k in range(len(seq))]
    ok = [o.loc_ok for o in outs]
    assert ok[0] is False and ok[1] and ok[2]                 # needs 2 consistent fixes first
    assert all(ok[3:13])                                      # held for exactly 1.0 s after the fix at 0.2 s
    assert ok[13] is False and outs[13].state == "lost"       # 1.1 s without a fix
    assert ok[14] is False                                    # reacquiring 1/2
    assert ok[15] is False and "inconsistent" in outs[15].reason
    assert ok[16] is False                                    # 1/2 again after the jump
    assert ok[17] and ok[18]
    assert np.allclose(outs[12].T_map_cam, P(0.02))           # held pose between fixes
    assert outs[12].age_s > 0.99 and not outs[12].fresh


def test_localizer_rejects_implausible_jump_while_tracking():
    P = lambda x: make_T(np.eye(3), (x, 0.0, 0.0))          # noqa: E731
    loc = Localizer([_Scripted([P(0), P(0.01), P(3.0), P(0.03)])], {"reacquire_fixes": 2})
    outs = [loc.process(None, 0.1 * k) for k in range(4)]
    assert outs[2].loc_ok and not outs[2].fresh and "implausible" in outs[2].reason
    assert outs[3].fresh and np.allclose(outs[3].T_map_cam, P(0.03))


# --- synthetic room end to end -----------------------------------------------------------------------------

def test_landmark_db_depth_accuracy_and_round_trip():
    ds = _dataset()
    assert ds.db.n_keyframes > 100 and len(ds.db) > 50_000
    e = ev.landmark_depth_error(ds)
    assert e["rel_med"] < 0.01 and e["frac_over_5pct"] < 0.02, e
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "lm.npz")
        ds.db.save(p)
        db2 = LandmarkDB.load(p)
    assert np.array_equal(db2.descriptors, ds.db.descriptors) and np.allclose(db2.points, ds.db.points)
    assert np.allclose(db2.kf_poses, ds.db.kf_poses) and db2.camera.to_dict() == ds.cam.to_dict()


def test_tracking_on_held_out_flight():
    s = ev.summarize(ev.tracking(_dataset()))
    print("    tracking:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}))
    assert s["fixes"] >= 0.95 * s["frames"], s
    assert s["pos_med_m"] < 0.03 and s["pos_p95_m"] < 0.06, s
    assert s["rot_med_deg"] < 0.5 and s["rot_p95_deg"] < 1.0, s


def test_global_relocalization_without_prior():
    s = ev.summarize(ev.kidnapped(_dataset()))
    print("    kidnapped:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()}))
    assert s["fixes"] >= 0.9 * s["frames"], s
    assert s["pos_med_m"] < 0.03 and s["pos_max_m"] < 0.10, s


def test_loc_ok_goes_false_on_textureless_view_and_recovers():
    recs, (h0, h1) = ev.blank_wall(_dataset())
    last_fix = None
    for r in recs:
        if r["fresh"]:
            last_fix = r["stamp"]
        if h0 <= r["stamp"] < h1:
            assert not r["fresh"], r                          # never a fix on the plain wall
        if last_fix is not None:
            age = r["stamp"] - last_fix
            assert r["loc_ok"] == (age <= 1.0 + 1e-6), (r, age)
    first_false = next(r["stamp"] for r in recs if r["stamp"] > 1.0 and not r["loc_ok"])
    fix_before = max(r["stamp"] for r in recs if r["fresh"] and r["stamp"] < first_false)
    assert 1.0 < first_false - fix_before <= 1.1 + 1e-6
    assert first_false < h1                                   # it did go false while facing the wall
    assert all(r["loc_ok"] for r in recs[-8:])                # and recovered once texture was back


def test_apriltag_fallback_keeps_loc_ok_on_the_plain_wall():
    recs, (h0, h1) = ev.blank_wall(_dataset(), with_tag=True)
    hold = [r for r in recs if h0 <= r["stamp"] < h1]
    assert all(r["loc_ok"] and r["fresh"] and r["method"] == "apriltag" for r in hold), hold[:3]
    # one 0.20 m tag at 0.6 m, face-on: rotation is the weak part of a single-square pose (~1-2 deg here)
    assert max(r["pos_err_m"] for r in hold) < 0.05 and max(r["rot_err_deg"] for r in hold) < 3.0,         [(round(r["pos_err_m"], 3), round(r["rot_err_deg"], 2)) for r in hold]


def test_config_files_build_a_localizer():
    cfgs = load_configs()
    cam = camera_from_config(cfgs)
    assert abs(cam.fov_deg()["diagonal"] - float(cfgs["hw"]["camera_front"]["fov_deg_nominal"])) < 1e-6
    assert "NOMINAL" in cam.source
    T = body_from_camera(cfgs)
    assert np.allclose(T[:3, 2], [1, 0, 0])                   # optical axis = body forward
    loc = build_localizer(cfgs, db=_dataset().db, camera=_dataset().cam)
    assert loc.cfg["loc_lost_hold_s"] == float(cfgs["flight"]["safety"]["loc_lost_hold_s"])


def test_build_landmark_map_tool_end_to_end():
    """The CLI on files: PCD map + keyframes.json (lidar poses + extrinsics) + intrinsics -> landmarks.npz."""
    import cv2
    import yaml
    from perception.io import write_pcd
    ds = _dataset()
    with tempfile.TemporaryDirectory() as tmp:
        write_pcd(os.path.join(tmp, "map.pcd"), ds.map_xyz, np.zeros(len(ds.map_xyz), np.float32))
        T_lc = sr.T_LIDAR_CAM_SYNTH
        kfs = []
        for kf in ds.keyframes[:12:3]:
            name = f"kf_{kf.kf_id:03d}.png"
            cv2.imwrite(os.path.join(tmp, name), kf.image)
            kfs.append({"id": kf.kf_id, "image": name, "stamp": kf.stamp,
                        "pose": (kf.T_map_cam @ inv_T(T_lc)).tolist()})
        with open(os.path.join(tmp, "keyframes.json"), "w") as f:
            json.dump({"pose_kind": "T_map_lidar", "keyframes": kfs}, f)
        with open(os.path.join(tmp, "ext.yaml"), "w") as f:
            yaml.safe_dump({"T_cam_lidar": inv_T(T_lc).tolist()}, f)
        with open(os.path.join(tmp, "cam.yaml"), "w") as f:
            yaml.safe_dump(ds.cam.to_dict(), f)
        out = os.path.join(tmp, "lm.npz")
        p = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "build_landmark_map.py"), "--map",
                            os.path.join(tmp, "map.pcd"), "--keyframes", os.path.join(tmp, "keyframes.json"),
                            "--extrinsics", os.path.join(tmp, "ext.yaml"), "--intrinsics",
                            os.path.join(tmp, "cam.yaml"), "--out", out], capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        summary = json.loads(p.stdout)
        db = LandmarkDB.load(out)
    assert summary["keyframes"] == 4 and db.n_keyframes == 4 and len(db) > 1000
    assert np.allclose(db.kf_poses[0], ds.keyframes[0].T_map_cam, atol=1e-9)
    ref = ds.db.landmarks_of([0])                              # same keyframe, same code -> same landmarks
    assert np.allclose(np.sort(db.points[db.kf_index == 0], axis=0), np.sort(ds.db.points[ref], axis=0), atol=1e-4)


def test_ros_adapter_conversions_without_ros():
    """The node's message conversions live in a ROS-free module; check them here (the node itself needs rclpy)."""
    sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "nx_scout_loc"))
    from nx_scout_loc import convert
    rng = np.random.default_rng(3)
    bgr = rng.integers(0, 256, (4, 5, 3), dtype=np.uint8)
    rgb_padded = np.zeros((4, 16), np.uint8)                   # step 16 > width * 3 = 15
    rgb_padded[:, :15] = bgr[..., ::-1].reshape(4, 15)
    assert np.array_equal(convert.image_to_bgr("rgb8", 4, 5, 16, rgb_padded.tobytes()), bgr)
    assert np.array_equal(convert.image_to_bgr("bgr8", 4, 5, 15, bgr.tobytes()), bgr)
    try:
        convert.image_to_bgr("yuv422", 4, 5, 10, bytes(40))
        raise AssertionError("unsupported encoding accepted")
    except ValueError:
        pass
    T = make_T(rot_z(0.4), (1.0, 2.0, 3.0))
    pos, q = convert.pose_fields(T)
    assert np.allclose(pos, (1, 2, 3)) and np.allclose(quat_xyzw_to_R(q), T[:3, :3])
    assert abs(convert.stamp_to_sec(12, 500_000_000) - 12.5) < 1e-12


if __name__ == "__main__":
    import time
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        t0 = time.time()
        fn()
        print(f"  [ok] {name} ({time.time() - t0:.1f} s)")
    print(f"{len(tests)}/{len(tests)} passed")

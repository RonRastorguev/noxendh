"""perception.table_frame: find the diorama's tabletop from a scan in the SENSOR frame, for any mount.

Synthetic: Ron's diorama on a finite 0.6 x 1.0 m table with a floor 0.75 m below, ray-cast from a lidar over the
near end, then rotated into an arbitrary sensor mount (dome down, dome tilted, any heading). The fit must recover the
sensor's pose in the table frame from the cloud plus the IMU's "up" alone."""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from perception import synthetic as syn  # noqa: E402
from perception import table_frame as tf  # noqa: E402


def rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def scene_with_table_and_floor(base=None):
    sc = base or syn.diorama()
    table = syn.Box((syn.TABLE_X[0], syn.TABLE_Y[0], -0.02), (syn.TABLE_X[1], syn.TABLE_Y[1], 0.0), 0.3, "table")
    floor = syn.Box((-3, -3, -0.77), (3, 3, -0.75), 0.2, "floor")
    return syn.Scene(sc.boxes + [table, floor], sc.table_intensity)


def sensor_cloud(R_ts, origin, seed=0, imu_tilt_deg=1.0):
    """Scan in the TABLE frame from `origin`, returned in the sensor frame whose axes are R_ts (sensor -> table)."""
    xyz_t, inten = syn.scan(scene_with_table_and_floor(), origin=origin, n_rays=150_000, half_angle_deg=75,
                            seed=seed, plane_z=None)
    xyz_s = (xyz_t - np.asarray(origin)) @ R_ts           # = R_ts.T @ (p - o)
    up_s = R_ts.T @ (rot("x", imu_tilt_deg) @ np.array([0, 0, 1.0]))   # the IMU's "up", slightly off
    return xyz_s.astype(np.float32), inten, up_s * 9.81


def check_mount(R_ts, origin, label):
    xyz_s, _, accel = sensor_cloud(R_ts, origin)
    t0 = time.time()
    fit = tf.fit_table(xyz_s, accel)
    dt = time.time() - t0
    assert fit.ok, f"{label}: {fit.reason}"
    T = fit.T_table_sensor
    pos_err = np.linalg.norm(T[:3, 3] - np.asarray(origin))
    dR = T[:3, :3] @ R_ts.T
    ang_err = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    assert pos_err < 0.02, f"{label}: sensor position off by {pos_err * 100:.1f} cm"
    assert ang_err < 1.0, f"{label}: orientation off by {ang_err:.2f}°"
    W, L = fit.measured_size_m
    assert abs(W - 0.6) < 0.04 and L > 0.85, f"{label}: measured {W} x {L}"
    print(f"  {label}: pos err {pos_err * 1000:.1f} mm, rot err {ang_err:.2f}°, measured {W:.2f} x {L:.2f} m, "
          f"plane rms {fit.quality['plane_rms_m'] * 1000:.1f} mm, {dt:.2f} s, warnings {len(fit.warnings)}")
    return fit


def test_dome_down_over_near_end():
    check_mount(rot("x", 180), (0.0, -0.75, 0.6), "dome down, heading 0")


def test_dome_down_any_heading_and_tilt():
    check_mount(rot("z", 37) @ rot("x", 180) @ rot("y", 12), (0.05, -0.80, 0.55), "dome down, heading 37°, tilt 12°")


def test_beside_the_long_side_no_warning():
    # Ron's recommended rig: the lidar beside the middle of a 1 m edge, dome tilted toward the table's centre
    fit = check_mount(rot("z", 90) @ rot("x", 135), (-0.50, 0.0, 0.55), "beside the long side, tilted 45°")
    assert not fit.warnings, fit.warnings


def test_tilted_forward_mount():
    check_mount(rot("z", -120) @ rot("x", 130), (-0.1, -0.85, 0.5), "dome tilted 130°, heading -120°")


def test_points_land_in_the_roi():
    R_ts, origin = rot("z", 37) @ rot("x", 180), (0.0, -0.75, 0.6)
    xyz_s, _, accel = sensor_cloud(R_ts, origin, seed=3)
    fit = tf.fit_table(xyz_s, accel)
    xyz_t = tf.transform(xyz_s, fit.T_table_sensor)
    lo, hi = np.array(fit.roi["min"]), np.array(fit.roi["max"])
    inside = np.all((xyz_t >= lo) & (xyz_t <= hi), axis=1)
    on_tank = inside & (np.abs(xyz_t[:, 0] + 0.18) < 0.05) & (np.abs(xyz_t[:, 1] - 0.01) < 0.05) & (xyz_t[:, 2] > 0.1)
    assert on_tank.sum() > 50, "the tank is not where the table frame says it is"
    assert not np.any(inside & (xyz_t[:, 2] < -0.05)), "floor points inside the crop"


def test_dome_up_on_the_table_is_refused_with_a_reason():
    # the bring-up setup: L2 dome up sitting level with the tabletop; it sees almost nothing below itself
    rng = np.random.default_rng(0)
    xyz = rng.uniform([-3, -3, -0.04], [3, 3, 2.5], (50_000, 3)).astype(np.float32)
    fit = tf.fit_table(xyz, [0, 0, 9.8])
    assert not fit.ok and "below the lidar" in fit.reason, fit.reason


def test_floor_alone_gives_a_level_frame_and_a_crop_warning():
    # no table under the lidar: the floor is taken as the level frame (Ron: allow it, he crops by hand)
    floor_only = syn.Scene([syn.Box((-3, -3, -0.77), (3, 3, -0.75), 0.2, "floor")], 0.3)
    xyz_t, _ = syn.scan(floor_only, origin=(0, 0, 0.1), n_rays=80_000, half_angle_deg=75, plane_z=None)
    xyz_s = (xyz_t - np.array([0, 0, 0.1])) @ rot("x", 180)
    fit = tf.fit_table(xyz_s.astype(np.float32), rot("x", 180).T @ np.array([0, 0, 9.8]))
    assert fit.ok, fit.reason
    assert any("Crop" in w for w in fit.warnings), fit.warnings
    assert abs(fit.quality["lidar_height_above_table_m"] - 0.85) < 0.03, fit.quality


def test_apply_without_calibration_leaves_cfg_alone():
    cfg = {"roi": {"min": [-3, -3, -0.3], "max": [3, 3, 3]}, "table_frame_file": "does/not/exist.yaml"}
    out = tf.apply(cfg, ROOT)
    assert out["roi"] == cfg["roi"] and out["table_calib_id"] is None and "frame" not in out


def test_save_load_apply_roundtrip(tmp=None):
    import tempfile
    R_ts, origin = rot("x", 180), (0.0, -0.75, 0.6)
    xyz_s, _, accel = sensor_cloud(R_ts, origin, seed=5)
    fit = tf.fit_table(xyz_s, accel)
    d = tempfile.mkdtemp()
    path = os.path.join(d, "table_frame.yaml")
    tf.save(path, fit, scan_id="20261017T000000Z_test", accel_mean=accel)
    out = tf.apply({"table_frame_file": path}, ROOT)
    assert out["frame"] == "table" and out["table_calib_id"] == "table-20261017T000000Z_test"
    assert np.allclose(np.array(out["sensor_pose"]), fit.T_table_sensor, atol=1e-5)


def test_hand_crop_replaces_the_roi_only_in_its_own_frame():
    import tempfile
    d = tempfile.mkdtemp()
    cfg = {"roi": {"min": [-3, -3, -0.3], "max": [3, 3, 3]}, "table_frame_file": os.path.join(d, "none.yaml"),
           "crop_file": os.path.join(d, "crop.yaml")}
    box = {"min": [-0.4, 0.2, 0.7], "max": [0.3, 1.3, 1.2]}
    tf.save_crop(cfg, ROOT, box, frame="sensor")
    out = tf.apply(cfg, ROOT)
    assert out["roi"] == box and out["crop_source"] == "hand"
    # after a table calibration the frame changes: the sensor-frame crop no longer applies
    R_ts, origin = rot("x", 180), (0.0, -0.75, 0.6)
    xyz_s, _, accel = sensor_cloud(R_ts, origin, seed=7)
    tf.save(os.path.join(d, "tf.yaml"), tf.fit_table(xyz_s, accel), scan_id="s1")
    out = tf.apply(dict(cfg, table_frame_file=os.path.join(d, "tf.yaml")), ROOT)
    assert out["crop_source"] == "table" and out["roi"] != box
    try:
        tf.save_crop(cfg, ROOT, {"min": [0, 0, 1], "max": [1, 1, 0]}, frame="sensor")
        raise AssertionError("an inverted box was accepted")
    except ValueError:
        pass


def test_rotated_crop_follows_a_turned_diorama():
    from perception.preprocess import crop_roi, roi_bounds, roi_mask
    # a 0.6 x 1.0 m tabletop turned 30° about the vertical, centred at (0.2, -0.1)
    a = np.radians(30)
    ex, ey = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    rng = np.random.default_rng(0)
    uv = rng.uniform([-0.3, -0.5], [0.3, 0.5], (4000, 2))
    xy = np.array([0.2, -0.1]) + uv[:, :1] * ex + uv[:, 1:] * ey
    table = np.column_stack([xy, np.zeros(len(xy))])
    outside = np.column_stack([rng.uniform(-2, 2, (4000, 2)), np.zeros(4000)])
    outside = outside[~roi_mask(outside, {"center": [0.2, -0.1], "size": [0.62, 1.02], "yaw_deg": 30, "z": [-0.05, 0.5]})]
    pts = np.vstack([table, outside]).astype(np.float32)
    roi = {"center": [0.2, -0.1], "size": [0.62, 1.02], "yaw_deg": 30, "z": [-0.05, 0.5]}
    kept, _ = crop_roi(pts, np.zeros(len(pts), np.float32), roi)
    assert len(kept) == len(table), (len(kept), len(table))
    # the same box unrotated would lose the turned corners
    flat = dict(roi, yaw_deg=0)
    assert roi_mask(table, flat).sum() < 0.95 * len(table)
    lo, hi = roi_bounds(roi)
    assert np.all(table.min(axis=0)[:2] >= lo[:2] - 1e-6) and np.all(table.max(axis=0)[:2] <= hi[:2] + 1e-6)
    import tempfile
    d = tempfile.mkdtemp()
    cfg = {"table_frame_file": os.path.join(d, "none.yaml"), "crop_file": os.path.join(d, "crop.yaml")}
    doc = tf.save_crop(cfg, ROOT, dict(roi, yaw_deg=390), frame="sensor")
    assert doc["roi"]["yaw_deg"] == 30 and tf.apply(cfg, ROOT)["roi"]["center"] == [0.2, -0.1]


def test_crop_is_carried_into_a_new_calibration_of_an_unmoved_lidar():
    from perception.preprocess import roi_mask
    R_ts, origin = rot("z", 37) @ rot("x", 180), (0.0, -0.75, 0.6)
    xyz_s, _, accel = sensor_cloud(R_ts, origin, seed=11)
    old = {"T_table_sensor": tf.fit_table(xyz_s, accel).T_table_sensor.tolist()}
    # a second calibration of the SAME scan with a different (bigger) expected size: same surface, frame shifted
    new_fit = tf.fit_table(xyz_s, accel, expected_size_m=(0.6, 1.3))
    new = {"T_table_sensor": new_fit.T_table_sensor.tolist()}
    box = {"center": [0.05, 0.1], "size": [0.5, 0.7], "yaw_deg": 15.0, "z": [-0.02, 0.3]}
    carried = tf.carry_crop({"roi": box}, old, new)
    pts_old = tf.transform(xyz_s, old["T_table_sensor"])
    pts_new = tf.transform(xyz_s, new["T_table_sensor"])
    a, b = roi_mask(pts_old, box), roi_mask(pts_new, carried)
    assert a.sum() > 1000 and (a == b).mean() > 0.999, ((a != b).sum(), a.sum())


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            ok += 1
            print(f"  [ok] {t.__name__} ({time.time() - t0:.1f} s)")
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
    print(f"{ok}/{len(tests)} passed")
    sys.exit(0 if ok == len(tests) else 1)

"""perception.camera_calib (four clicked table corners -> camera pose) and perception.colorize (points take the
camera's colour, occluded points stay uncoloured), on a synthetic tilted camera beside a dome-down lidar."""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from perception import camera_calib as cc  # noqa: E402
from perception import colorize as col  # noqa: E402
from perception import synthetic as syn  # noqa: E402
from test_table_frame import rot  # noqa: E402

IMG = (1280, 960)
LIDAR = (0.0, -0.75, 0.6)
S = np.eye(4)
S[:3, :3], S[:3, 3] = rot("x", 180), LIDAR                  # dome-down lidar over the near end, in the table frame
TABLE_DOC = {"size_m": [0.6, 1.0], "measured_size_m": [0.6, 1.0]}


def look_at(C, target, fov_diag=102.0):
    C, target = np.asarray(C, float), np.asarray(target, float)
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, -R @ C
    K = cc.intrinsics_from_fov(IMG, fov_diag)
    return {"K": K.tolist(), "dist": [0] * 5, "T_cam_from_sensor": T.tolist(), "image_size": list(IMG)}


CAM_TRUE = look_at((0.10, -0.85, 0.65), (0.0, 0.05, 0.0))


def clicks_for(cam, noise_px=1.5, seed=0):
    import cv2
    K, T = np.asarray(cam["K"]), np.asarray(cam["T_cam_from_sensor"])
    corners = cc.table_corners(TABLE_DOC)
    c = corners @ T[:3, :3].T + T[:3, 3]
    uv = (c @ K.T)[:, :2] / (c @ K.T)[:, 2:3]
    rng = np.random.default_rng(seed)
    uv = uv + rng.normal(0, noise_px, uv.shape)
    return uv[rng.permutation(4)]                              # Ron clicks in any order


def pose_err(res, cam):
    T = np.asarray(cam["T_cam_from_sensor"])
    C_true = -T[:3, :3].T @ T[:3, 3]
    d = np.linalg.norm(np.asarray(res["camera_in_frame_m"]) - C_true)
    dR = res["T_cam_from_frame"][:3, :3] @ T[:3, :3].T
    a = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    return d, a


def test_corners_recover_the_camera_pose():
    res = cc.solve_from_corners(clicks_for(CAM_TRUE), cc.table_corners(TABLE_DOC), IMG, 102.0, S)
    assert res["ok"], res
    d, a = pose_err(res, CAM_TRUE)
    print(f"    pos {d * 100:.1f} cm, rot {a:.2f}°, reproj {res['reproj_px']} px, f {res['focal_px']} ({res['focal_from']})")
    assert d < 0.03 and a < 1.5, (d, a)


def test_wrong_fov_guess_is_corrected_by_the_focal_fit():
    res = cc.solve_from_corners(clicks_for(CAM_TRUE, seed=1), cc.table_corners(TABLE_DOC), IMG, 92.0, S)
    assert res["ok"], res
    d, a = pose_err(res, CAM_TRUE)
    f_true = CAM_TRUE["K"][0][0]
    print(f"    guess 92° -> f {res['focal_px']} (true {f_true:.1f}), pos {d * 100:.1f} cm, rot {a:.2f}°")
    assert d < 0.06 and a < 3.0, (d, a)


def test_far_from_lidar_is_refused():
    far = np.eye(4)
    far[:3, 3] = (2.0, 2.0, 1.0)                              # a lidar nowhere near the camera
    res = cc.solve_from_corners(clicks_for(CAM_TRUE), cc.table_corners(TABLE_DOC), IMG, 102.0, far)
    assert not res["ok"] and "lidar" in res["reason"]


def test_saved_calibration_survives_a_table_recalibration():
    import tempfile
    res = cc.solve_from_corners(clicks_for(CAM_TRUE), cc.table_corners(TABLE_DOC), IMG, 102.0, S)
    p = os.path.join(tempfile.mkdtemp(), "cam.yaml")
    doc = cc.save(p, res, image_size=IMG, sensor_pose=S, scan_id="x", table_calib_id="table-x", fov_diag_deg=102.0)
    # a new table frame shifted 5 cm and turned 3°: the camera dict must follow the lidar, not the old table
    D = np.eye(4)
    D[:3, :3], D[:3, 3] = rot("z", 3), (0.05, 0.0, 0.0)
    cam_new = cc.camera_for_frame(cc.load(p), D @ S)
    T_expect = np.asarray(res["T_cam_from_frame"]) @ np.linalg.inv(D)
    assert np.allclose(np.asarray(cam_new["T_cam_from_sensor"]), T_expect, atol=1e-6)
    assert doc["table_calib_id"] == "table-x"


def test_colorize_takes_the_pixel_colour_and_skips_occluded_points():
    scene = syn.diorama()
    img = syn.render_image(scene, CAM_TRUE, seed=0)
    xyz, _ = syn.scan(scene, origin=LIDAR, n_rays=120_000, half_angle_deg=75, seed=0)
    rgb, seen = col.colorize(xyz, img, CAM_TRUE)
    t0 = time.time()
    col.colorize(xyz, img, CAM_TRUE)
    print(f"    {seen.mean() * 100:.0f} % of {len(xyz)} points coloured, {time.time() - t0:.2f} s")
    tank = syn.diorama().boxes[[b.name for b in syn.diorama().boxes].index("tank")]
    on_top = seen & np.all((xyz[:, :2] >= np.array(tank.lo[:2]) + 0.01) & (xyz[:, :2] <= np.array(tank.hi[:2]) - 0.01), axis=1) \
        & (np.abs(xyz[:, 2] - tank.hi[2]) < 0.02)
    assert on_top.sum() > 30
    tank_bgr = np.array(syn._COLORS["tank"], float)
    assert np.abs(rgb[on_top].astype(float).mean(axis=0) - tank_bgr[::-1]).max() < 25, rgb[on_top].mean(axis=0)
    # the table right behind the tank (as seen from the camera) is hidden by it: those points stay uncoloured
    behind = (np.abs(xyz[:, 2]) < 0.01) & (xyz[:, 0] > tank.lo[0] + 0.02) & (xyz[:, 0] < tank.hi[0] - 0.02) \
        & (xyz[:, 1] > tank.hi[1] + 0.01) & (xyz[:, 1] < tank.hi[1] + 0.06)
    if behind.sum() > 10:
        tank_like = np.all(np.abs(rgb[behind & seen].astype(float) - tank_bgr[::-1]) < 25, axis=1)
        assert tank_like.mean() < 0.2, f"{tank_like.mean():.0%} of hidden table points took the tank's colour"


def _pairs(cam, n, noise_px=2.0, noise_m=0.01, seed=0):
    """n spots a person would pick: tops and corners of the diorama's boxes and the table's corners."""
    rng = np.random.default_rng(seed)
    spots = [(x, y, b.hi[2]) for b in syn.diorama().boxes for x in (b.lo[0], b.hi[0]) for y in (b.lo[1], b.hi[1])]
    spots += [tuple(c) for c in cc.table_corners(TABLE_DOC)]
    obj = np.asarray(spots, float)[rng.permutation(len(spots))[:n]]
    K, T = np.asarray(cam["K"]), np.asarray(cam["T_cam_from_sensor"])
    c = obj @ T[:3, :3].T + T[:3, 3]
    uv = (c @ K.T)[:, :2] / (c @ K.T)[:, 2:3]
    return uv + rng.normal(0, noise_px, uv.shape), obj + rng.normal(0, noise_m, obj.shape)


def scene_err_px(res, cam):
    """What colour accuracy depends on: where the scene's points land in the image, fitted camera vs true camera
    (p90 over diorama points in view). With 1 cm picking error on spots ~0.2 m apart the POSE itself is poorly
    conditioned (focal and distance trade off), while the scene reprojection stays usable; DH-26 sweep: p90
    ~15-17 px median over seeds at 1280 x 960 with the right lens figure, ~16-25 px with a wrong one and 12 pairs."""
    xyz, _ = syn.scan(syn.diorama(), origin=LIDAR, n_rays=20000, half_angle_deg=60, seed=1)

    def proj(K, T):
        c = xyz @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]
        return (c @ np.asarray(K).T)[:, :2] / (c @ np.asarray(K).T)[:, 2:3]
    uvt = proj(cam["K"], cam["T_cam_from_sensor"])
    m = (uvt[:, 0] > 0) & (uvt[:, 0] < IMG[0]) & (uvt[:, 1] > 0) & (uvt[:, 1] < IMG[1])
    return float(np.percentile(np.linalg.norm(proj(res["K"], res["T_cam_from_frame"])[m] - uvt[m], axis=1), 90))


def test_point_pairs_colour_lands_within_a_few_cm():
    # one random set of clicks can be unlucky; what Ron sees on average is the median over click sets
    for n in (6, 8, 12):
        errs = []
        for seed in range(8):
            px, xyz = _pairs(CAM_TRUE, n, seed=300 + seed)
            res = cc.solve_from_pairs(px, xyz, IMG, 102.0, S)
            assert res["ok"], (n, res)
            errs.append(scene_err_px(res, CAM_TRUE))
        print(f"    {n} pairs: scene p90 median {np.median(errs):.1f} px, worst {max(errs):.1f} px over 8 click sets")
        assert np.median(errs) < 30, (n, errs)


def test_point_pairs_with_a_wrong_focal_guess():
    px, xyz = _pairs(CAM_TRUE, 12, seed=4)
    res = cc.solve_from_pairs(px, xyz, IMG, 90.0, S)
    assert res["ok"], res
    e = scene_err_px(res, CAM_TRUE)
    print(f"    guess 90° -> f {res['focal_px']} (true {CAM_TRUE['K'][0][0]:.1f}, {res['focal_from']}), scene p90 {e:.1f} px")
    assert e < 55, e


def test_a_mismatched_pair_is_named():
    px, xyz = _pairs(CAM_TRUE, 8, seed=2)
    px[3] += (160, -120)                                     # one image click on the wrong spot
    res = cc.solve_from_pairs(px, xyz, IMG, 102.0, S)
    assert res["ok"] and any("pair 4" in w and "left out" in w for w in res["warnings"]), res
    assert scene_err_px(res, CAM_TRUE) < 45


def test_fewer_than_six_pairs_is_refused():
    px, xyz = _pairs(CAM_TRUE, 5)
    res = cc.solve_from_pairs(px, xyz, IMG, 102.0, S)
    assert not res["ok"] and "6" in res["reason"]


def test_veil_filter_drops_edge_smear_not_surfaces():
    from perception.preprocess import veil_mask
    rng = np.random.default_rng(0)
    origin = np.array([0.0, 0.0, 0.6])
    floor = np.column_stack([rng.uniform(-0.5, 0.5, 3000), rng.uniform(0.3, 1.0, 3000), np.zeros(3000)])
    # a box face square to the beam, and a smear strip lying along the beams behind its edge (mixed pixels)
    face = np.column_stack([rng.uniform(-0.1, 0.1, 800), np.full(800, 0.5), rng.uniform(0.0, 0.2, 800)])
    t = rng.uniform(0.0, 1.0, 400)
    edge = np.array([0.1, 0.5, 0.1])
    beam = (edge - origin) / np.linalg.norm(edge - origin)
    smear = edge + np.outer(0.05 + 0.25 * t, beam) + rng.normal(0, 0.002, (400, 3)) + np.column_stack([np.zeros(400), np.zeros(400), rng.uniform(-0.08, 0.08, 400)])
    pts = np.vstack([floor, face, smear])
    v = veil_mask(pts, origin)
    print(f"    veil flagged: floor {v[:3000].mean():.0%}, face {v[3000:3800].mean():.0%}, smear {v[3800:].mean():.0%}")
    assert v[3000:3800].mean() < 0.1 and v[3800:].mean() > 0.6


def test_no_calibration_means_no_colour():
    rgb, seen = col.colorize(np.zeros((10, 3), np.float32), np.zeros((4, 4, 3), np.uint8), None)
    assert not seen.any()


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

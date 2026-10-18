"""Looming obstacle detection (avoidance/, docs/SCOUT_NAV.md §5) on SYNTHETIC data: no hardware data exists yet.

The textured synthetic room (localization/synthetic_room.py) is rendered at the TRUE camera pose of a Scout flying a
scripted trajectory at 10 Hz (pitch / roll from the acceleration like a multicopter). The detector gets what the
real one would get: the image, the FC's odometry (an odom frame that drifts away from map, attitude noise, velocity
noise) and, optionally, a noisy PnP map pose. The prior map is the DEFAULT room's cloud; obstacles under test are
extra boxes appended AFTER the default ones (shared surfaces keep identical textures) and are NOT in the map.

Bounds here are regression bounds on synthetic numbers (provisional), not real-world claims.
Runs under pytest or `py -3.9 tests/test_looming.py` (prints the measured numbers).
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from avoidance import DEFAULTS, LoomingDetector, load_config  # noqa: E402
from localization import synthetic_room as sr  # noqa: E402
from localization.camera import CameraModel  # noqa: E402
from localization.geometry import body_pose, make_T, rot_x, rot_y, rot_z  # noqa: E402

SEED = 0
RATE = 10.0
G = 9.81
_CACHE = {}
RESULTS = {}


def _cam():
    return CameraModel.from_fov(640, 480, 102.0, "diagonal")


def _map():
    if "map" not in _CACHE:
        _CACHE["map"] = sr.map_cloud(sr.make_room(SEED), 0.03, 0.01, SEED + 1)
    return _CACHE["map"]


def _renderer(extra=()):
    key = ("r",) + tuple((n, tuple(lo), tuple(hi)) for n, lo, hi in extra)
    if key not in _CACHE:
        room = sr.make_room(SEED, boxes=list(sr.DEFAULT_BOXES) + list(extra))
        _CACHE[key] = sr.Renderer(room, _cam(), supersample=2)       # 2x: optics-like anti-aliasing
    return _CACHE[key]


# ------------------------------------------------------------------------------------------------------------------
# trajectories: dense (100 Hz) positions + yaw, attitude from acceleration, sampled at 10 Hz

def _dense_path(waypoints, speed_fn, dt=0.01, t_max=120.0):
    """Fly a polyline at speed_fn(s_remaining, s) (m/s, smoothed). -> times, positions (N, 3), headings (N,)."""
    wp = np.asarray(waypoints, dtype=np.float64)
    seg = np.diff(wp, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    s, v, t = 0.0, 0.0, 0.0
    ts, ss = [], []
    while s < cum[-1] - 1e-4 and t < t_max:
        vt = float(speed_fn(cum[-1] - s, s))
        v += np.clip(vt - v, -1.0 * dt, 1.0 * dt)                 # 1 m/s^2 acceleration limit
        s = min(cum[-1], s + v * dt)
        ts.append(t)
        ss.append(s)
        t += dt
        if v < 1e-3 and vt < 1e-3:
            break
    ss = np.asarray(ss)
    i = np.clip(np.searchsorted(cum, ss, side="right") - 1, 0, len(L) - 1)
    a = (ss - cum[i]) / L[i]
    pos = wp[i] + seg[i] * a[:, None]
    head = np.arctan2(seg[i, 1], seg[i, 0])
    return np.asarray(ts), pos, head


def _frames(ts, pos, yaw):
    """10 Hz samples with velocity and multicopter tilt from acceleration. -> list of dict(t, T_map_body, v_map)."""
    dt = ts[1] - ts[0]
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    k = max(1, int(round(0.15 / dt)))                               # smooth the tilt like a real attitude loop
    kern = np.ones(k) / k
    acc = np.stack([np.convolve(acc[:, j], kern, mode="same") for j in range(3)], 1)
    out = []
    step = int(round(1.0 / RATE / dt))
    for n in range(0, len(ts), step):
        c, s = np.cos(yaw[n]), np.sin(yaw[n])
        a_f = c * acc[n, 0] + s * acc[n, 1]
        a_l = -s * acc[n, 0] + c * acc[n, 1]
        pitch, roll = np.arctan2(a_f, G), -np.arctan2(a_l, G)
        out.append({"t": float(ts[n]), "T_map_body": body_pose(pos[n], yaw[n], pitch, roll), "v_map": vel[n]})
    return out


def straight(p0, p1, z, speed=0.4, stop_at=None, ttc_floor=None, hull=0.20):
    """Straight flight p0 -> p1 at height z, cruise `speed`. `ttc_floor`: decelerate so that the hull-to-surface
    time-to-contact never drops below it (the follower's TTC speed law), surface = the path end + `stop_at`."""
    wps = [(p0[0], p0[1], z), (p1[0], p1[1], z)]

    def sp(rem, s):
        if ttc_floor is None:
            return speed
        d_surface = rem + stop_at                                   # body to the surface ahead
        return max(0.0, min(speed, (d_surface - hull) / ttc_floor)) if rem > 1e-3 else 0.0
    ts, pos, head = _dense_path(wps, sp)
    return _frames(ts, pos, head)


def _noise_rot(rng, sigma_deg):
    a = np.radians(rng.normal(0, sigma_deg, 3))
    return rot_z(a[2]) @ rot_y(a[1]) @ rot_x(a[0])


def run(frames, extra=(), appear_t=None, with_map=True, seed=1, detector=None, cfg=None, reset_at=None,
        pass_counter=False):
    """Render + detect. `extra` boxes are in the world (not the map) from `appear_t` on (always if None)."""
    rng = np.random.default_rng(seed)
    cam = _cam()
    det = detector or LoomingDetector(cam, cfg, map_points=_map() if with_map else None, map_spacing_m=0.03,
                                      T_body_cam=sr.T_BODY_CAM_SYNTH)
    r_world = _renderer(extra)
    r_plain = _renderer(())
    # odom = map moved by an offset and a yaw that drifts (optical flow + gyro-only yaw), with a +5 % scale error
    # on translation AND velocity (flow scale) and a velocity bias; attitude noise. PnP pose: sigma 3 cm / 0.5 deg.
    yaw0, off = np.radians(25.0), np.array([-1.3, 0.7, 0.0])
    scale, v_bias = 1.05, np.array([0.02, -0.02, 0.0])
    res = []
    for f in frames:
        t = f["t"]
        T_odom_map = make_T(rot_z(yaw0 + np.radians(0.5) * t / 60.0), off + np.array([0.002, -0.001, 0.0]) * t)
        counter = 0
        if reset_at is not None and t >= reset_at - 1e-9:          # PX4 local-frame reset: odom jumps
            T_odom_map = make_T(rot_z(np.radians(5.0)), (0.3, -0.2, 0.0)) @ T_odom_map
            counter = 1
        Tb = f["T_map_body"]
        T_odom_body = T_odom_map @ Tb
        T_odom_body[:3, 3] *= scale
        T_odom_body[:3, :3] = T_odom_body[:3, :3] @ _noise_rot(rng, 0.03)
        v_odom = scale * (T_odom_map[:3, :3] @ f["v_map"]) + v_bias + rng.normal(0, 0.02, 3)
        T_pnp = None
        if with_map:
            T_pnp = Tb.copy()
            T_pnp[:3, :3] = T_pnp[:3, :3] @ _noise_rot(rng, 0.5 / np.sqrt(3))
            T_pnp[:3, 3] += rng.normal(0, 0.03, 3)
        rend = r_world if (appear_t is None or t >= appear_t - 1e-9) else r_plain
        img, _ = rend.render(Tb @ sr.T_BODY_CAM_SYNTH, noise_sigma=2.0, rng=rng)
        t0 = time.perf_counter()
        out = det.process(img, t, T_odom_body, v_odom, T_pnp, reset_counter=counter if pass_counter else None)
        res.append({"t": t, "T_map_body": Tb, "out": out, "ms": 1000.0 * (time.perf_counter() - t0)})
    return res


def _timing(res):
    ms = np.array([r["ms"] for r in res[3:]])
    return {"mean_ms": float(ms.mean()), "p95_ms": float(np.percentile(ms, 95)), "max_ms": float(ms.max())}


def _obstacles(res, source=None):
    return [(r, o) for r in res for o in r["out"].obstacles if source is None or o.source == source]


def _face_error(o, box_lo, box_hi, approach_axis=0):
    """Distance between the detected front-face centre (feature centroid horizontally, box z-mid) and the true
    face facing the approach (lo along `approach_axis`)."""
    lo, hi = np.asarray(box_lo, float), np.asarray(box_hi, float)
    true_face = (lo + hi) / 2
    true_face[approach_axis] = lo[approach_axis]
    det = np.array(o.info["face_center"], float)
    det[2] = (o.lo[2] + o.hi[2]) / 2 if o.top_seen else det[2]
    if not o.top_seen:                                              # top unknown: compare horizontally only
        return float(np.linalg.norm((det - true_face)[:2]))
    return float(np.linalg.norm(det - true_face))


# ------------------------------------------------------------------------------------------------------------------
# tests

LOW_BOX = ("unmapped_low", (4.5, 3.3, 0.0), (5.9, 4.7, 0.7))
TALL_BOX = ("unmapped_tall", (4.5, 3.75, 0.0), (5.0, 4.25, 1.9))


def _approach_box(box, z_cam_body=1.1):
    frames = straight((0.8, 4.0), (box[1][0] - 1.0, 4.0), z_cam_body, speed=0.4)
    res = run(frames, extra=[box])
    lo, hi = np.array(box[1]), np.array(box[2])
    dets = []
    for r, o in _obstacles(res, "looming_map"):
        x_cam = (r["T_map_body"] @ sr.T_BODY_CAM_SYNTH)[0, 3]
        dets.append({"t": r["t"], "range": float(lo[0] - x_cam), "face_err": _face_error(o, lo, hi),
                     "center_err": float(np.linalg.norm(o.center - (lo + hi) / 2)), "top_seen": o.top_seen,
                     "hi_z": float(o.hi[2]), "id": o.id, "n": o.info["n_features"]})
    return res, dets


def test_low_box_detected_early_and_placed():
    res, dets = _approach_box(LOW_BOX)
    assert dets, "the unmapped low box was never detected"
    first = dets[0]
    errs = [d["face_err"] for d in dets]
    RESULTS["low_box"] = {"first_range_m": first["range"], "n_detections": len(dets),
                          "face_err_median_m": float(np.median(errs)), "face_err_max_m": float(np.max(errs)),
                          "center_err_median_m": float(np.median([d["center_err"] for d in dets])),
                          "top_seen_frac": float(np.mean([d["top_seen"] for d in dets])),
                          "hi_z_median": float(np.median([d["hi_z"] for d in dets])),
                          "emergencies": int(sum(r["out"].emergency for r in res)), **_timing(res)}
    assert first["range"] >= 1.5, first
    assert np.median(errs) <= 0.3 and np.percentile(errs, 90) <= 0.35, errs
    seen = [d for d in dets if d["top_seen"]]
    assert len(seen) >= 0.8 * len(dets), RESULTS["low_box"]
    assert all(d["hi_z"] >= 0.65 for d in seen), [d["hi_z"] for d in seen]           # at most 5 cm under the true 0.7
    # over-estimated tops are the conservative side. Within 2.6 m (where detection began before DN-36) <= 1.2; the
    # earlier detections that the odometry-scale correction adds (2.7-2.85 m) bound the top from fewer, longer rays
    assert all(d["hi_z"] <= 1.2 for d in seen if d["range"] <= 2.6), [(d["range"], d["hi_z"]) for d in seen]
    assert all(d["hi_z"] <= 1.5 for d in seen), [(d["range"], d["hi_z"]) for d in seen]


def test_tall_box_detected_top_not_underestimated():
    res, dets = _approach_box(TALL_BOX)
    assert dets, "the unmapped tall box was never detected"
    errs = [d["face_err"] for d in dets]
    near = [d for d in dets if d["range"] < 2.4]
    RESULTS["tall_box"] = {"first_range_m": dets[0]["range"], "n_detections": len(dets),
                           "face_err_median_m": float(np.median(errs)), "face_err_max_m": float(np.max(errs)),
                           "center_err_median_m": float(np.median([d["center_err"] for d in dets])),
                           "top_seen_frac": float(np.mean([d["top_seen"] for d in dets])),
                           "top_seen_frac_below_2.4m": float(np.mean([d["top_seen"] for d in near])) if near else None,
                           "min_hi_z": float(min(d["hi_z"] for d in dets)), **_timing(res)}
    assert dets[0]["range"] >= 1.5, dets[0]
    assert np.median(errs) <= 0.3, errs
    # never under-estimates a CLAIMED top. An unseen top (top_seen False: the planner never flies over it, DN-46) is
    # the highest feature + top_unseen_pad_m; with own-excess map pairs (DN-35) the box is first boxed at 3.35 m
    # instead of 1.99 m, where that placeholder can sit a few cm under the true 1.9 m
    assert all(d["hi_z"] >= 1.85 for d in dets if d["top_seen"]), [d["hi_z"] for d in dets if d["top_seen"]]
    assert all(d["hi_z"] >= 1.80 for d in dets), [(d["range"], d["hi_z"], d["top_seen"]) for d in dets]
    assert near and not any(d["top_seen"] for d in near), near


def test_mapped_wall_and_box_head_on_no_false_alarm():
    """The false-alarm test: the goal wall (and a mapped box) approached head-on to 0.8 m with the follower's TTC
    speed law (hull-to-surface TTC >= 2 s) must give zero obstacles and no emergency."""
    stats = {}
    # wall_xL at x = 8, along y = 2.0, passing the mapped pillar 0.8 m to the side; body stops 0.8 m from the wall
    wall = straight((2.5, 2.0), (7.2, 2.0), 1.1, speed=0.5, stop_at=0.8, ttc_floor=2.0)
    # mapped crate_a (x 1.0-1.8, y 0.6-1.4, 1.0 m tall) head-on from +x, flying at 0.8 m (camera sees its face)
    crate = [dict(f) for f in straight((5.0, 1.0), (2.6, 1.0), 0.8, speed=0.5, stop_at=0.8, ttc_floor=2.0)]
    for name, frames in (("wall", wall), ("mapped_box", crate)):
        res = run(frames)
        n_obs = len(_obstacles(res))
        n_em = sum(r["out"].emergency for r in res)
        finite = [r["out"].min_ttc_s for r in res if np.isfinite(r["out"].min_ttc_s)]
        stats[name] = {"frames": len(res), "obstacles": n_obs, "emergencies": n_em,
                       "min_ttc_s": float(min(finite)) if finite else None, **_timing(res)}
    RESULTS["head_on_mapped"] = stats
    for name, s in stats.items():
        assert s["obstacles"] == 0 and s["emergencies"] == 0, (name, s)


def test_yaw_in_place_nothing():
    ts = np.arange(0, 8.0, 0.01)
    pos = np.stack([4.0 + 0.02 * np.sin(1.3 * ts), 1.6 + 0.02 * np.cos(0.9 * ts), 1.1 + 0.01 * np.sin(2 * ts)], 1)
    yaw = np.radians(45.0) * ts
    res = run(_frames(ts, pos, yaw))
    n_obs, n_em = len(_obstacles(res)), sum(r["out"].emergency for r in res)
    RESULTS["yaw_in_place"] = {"frames": len(res), "obstacles": n_obs, "emergencies": n_em, **_timing(res)}
    assert n_obs == 0 and n_em == 0


def test_box_beside_corridor_no_emergency():
    side = ("unmapped_side", (5.5, 2.7, 0.0), (6.0, 3.2, 1.5))           # 0.8 m from the path's centreline
    res = run(straight((1.0, 4.0), (6.8, 4.0), 1.1, speed=0.5), extra=[side])
    n_em = sum(r["out"].emergency for r in res)
    finite = [r["out"].min_ttc_s for r in res if np.isfinite(r["out"].min_ttc_s)]
    RESULTS["beside_corridor"] = {"frames": len(res), "emergencies": n_em,
                                  "map_obstacles": len(_obstacles(res, "looming_map")),
                                  "min_ttc_s": float(min(finite)) if finite else None, **_timing(res)}
    assert n_em == 0, RESULTS["beside_corridor"]


def _late(gap_m):
    frames = straight((2.5, 2.0), (7.0, 2.0), 1.1, speed=0.5)
    i = next(n for n, f in enumerate(frames) if f["T_map_body"][0, 3] >= 4.6)   # at full speed, past the pillar
    x_b = frames[i]["T_map_body"][0, 3]
    t_app = frames[i]["t"]
    box = ("late", (x_b + gap_m, 1.6, 0.0), (x_b + gap_m + 0.5, 2.4, 1.6))
    frames = [f for f in frames if f["t"] <= t_app + 1.0 + 1e-9]
    res = run(frames, extra=[box], appear_t=t_app, with_map=False)
    first = next((r["t"] for r in res if r["t"] >= t_app - 1e-9 and r["out"].emergency), None)
    before = sum(r["out"].emergency for r in res if r["t"] < t_app - 1e-9)
    return {"gap_body_m": gap_m, "appear_t": t_app, "latency_s": None if first is None else first - t_app,
            "emergencies_before": int(before),
            "true_contact_ttc_at_appear_s": (gap_m - 0.20) / 0.5, **_timing(res)}


def test_late_obstacle_emergency_without_map_pose():
    s = _late(0.9)
    s10 = _late(1.0)
    RESULTS["late_obstacle"] = {"gap_0.9m": s, "gap_1.0m": s10}
    assert s["emergencies_before"] == 0 and s10["emergencies_before"] == 0
    assert s["latency_s"] is not None and s["latency_s"] <= 0.3 + 1e-6, s
    assert s10["latency_s"] is not None and s10["latency_s"] <= 0.6 + 1e-6, s10


def test_long_flight_past_mapped_content_no_obstacles():
    th = np.linspace(0, 2 * np.pi * 1.3, 2000)
    c, rad = (4.1, 2.9), (1.8, 1.25)
    wps = np.stack([c[0] + rad[0] * np.cos(th), c[1] + rad[1] * np.sin(th), 1.1 + 0.2 * np.sin(2 * th)], 1)
    ts, pos, head = _dense_path(wps, lambda rem, s: 0.4)
    yaw = head + np.radians(8.0) * np.sin(0.7 * ts)
    res = run(_frames(ts, pos, yaw))
    n_obs, n_em = len(_obstacles(res)), sum(r["out"].emergency for r in res)
    RESULTS["long_flight"] = {"frames": len(res), "duration_s": float(ts[-1]), "obstacles": n_obs,
                              "emergencies": n_em, **_timing(res)}
    assert n_obs == 0 and n_em == 0, RESULTS["long_flight"]


def test_odom_frame_reset_no_false_alarm():
    """A PX4 local-frame reset (0.3 m, -0.2 m, 5 deg) mid-flight: detected from the odom jump alone, and from
    the reset counter; either way nothing is measured across it (DN-34)."""
    frames = straight((2.5, 2.0), (6.0, 2.0), 1.1, speed=0.5)
    stats = {}
    for name, pc in (("jump_check", False), ("reset_counter", True)):
        res = run(frames, reset_at=3.0, pass_counter=pc)
        stats[name] = {"obstacles": len(_obstacles(res)), "emergencies": int(sum(r["out"].emergency for r in res)),
                       "motion_resets": [r["t"] for r in res if "motion_reset" in r["out"].debug]}
    RESULTS["odom_reset"] = stats
    for s_ in stats.values():
        assert s_["obstacles"] == 0 and s_["emergencies"] == 0 and s_["motion_resets"], stats


def test_own_depth_at_a_silhouette():
    """DN-35, exact geometry: a near box (Z 0.6 m) beside far background (Z 4 m). A pair projected on the tangent at
    its anchor measures the NEIGHBOUR's depth, so the old per-feature value (median over the feature's own pairs) is
    its neighbours' depth and a box feature at the silhouette reads background; the own-depth median (over the pairs
    that measure the feature) is exact for every feature."""
    from scipy.spatial import cKDTree
    from avoidance.looming import _wmedian, _wmedian_own
    det = LoomingDetector(_cam(), load_config())
    c = det.cfg
    tau, e = 0.1, np.array([0.0, -300.0])
    gx, gy = np.meshgrid(np.arange(-150.0, 151.0, 20.0), np.arange(-100.0, 101.0, 20.0))
    p_now = np.column_stack([gx.ravel(), gy.ravel()]) + np.random.default_rng(0).uniform(-4, 4, (gx.size, 2))
    Z = np.where(p_now[:, 0] < 0.0, 0.6, 4.0)
    p_old = (p_now - e) * (Z / (Z + tau))[:, None] + e
    K = int(c["neighbours_k"])
    dist, nb = cKDTree(p_now).query(p_now, k=K + 1, distance_upper_bound=c["max_pair_px"])
    dist, nb = dist[:, 1:], nb[:, 1:]
    ok = np.isfinite(dist)
    nb = np.where(ok, nb, 0)
    (r,), good, wt = det._pairs(p_now, [p_old], e, nb, ok, c["min_proj_px"])
    w_own = _wmedian_own(r, good, wt, nb, len(p_now), int(c["min_pairs"]))
    w_nb = _wmedian(r, good, wt, int(c["min_pairs"]))
    fin = np.isfinite(w_own)
    edge = (p_now[:, 0] < 0.0) & (p_now[:, 0] > -45.0)                      # box features next to the silhouette
    err_own = np.abs(tau / w_own[fin] / Z[fin] - 1.0)
    err_nb = np.abs(tau / w_nb[edge] / Z[edge] - 1.0)
    RESULTS["own_depth_silhouette"] = {"features": int(len(p_now)), "own_valid": int(fin.sum()),
                                       "own_max_rel_err": float(err_own.max()),
                                       "neighbour_max_rel_err_at_edge": float(np.nanmax(err_nb))}
    assert fin.sum() >= 0.9 * len(p_now) and err_own.max() < 1e-6, RESULTS["own_depth_silhouette"]
    assert np.nanmax(err_nb) > 0.5, RESULTS["own_depth_silhouette"]        # the old estimate fails there


def _follower_flight(path, yaw0, nav_over=None, t_max=40.0, dt=0.01):
    """Fly `path` with the REAL follower (navigation/follower.py) on the static map of this room: kinematic vehicle
    (first-order position lag 0.3 s, yaw rate <= 60 deg/s), follower at 20 Hz. -> times, positions, yaws (dense)."""
    from navigation.follower import Follower, wrap
    from navigation.nav_config import load_nav_config
    from navigation.static_map import StaticMap
    if "sm" not in _CACHE:
        _CACHE["nav_cfg"] = load_nav_config()
        _CACHE["sm"] = StaticMap.from_points(_map(), _CACHE["nav_cfg"])
    f = Follower(dict(_CACHE["nav_cfg"], **(nav_over or {})), _CACHE["sm"])
    f.set_path(np.asarray(path, float), 0.0)
    p, yaw, v = np.asarray(path[0], float).copy(), float(yaw0), np.zeros(3)
    sp, ysp = p.copy(), yaw
    ts, ps, ys = [], [], []
    for k in range(int(t_max / dt)):
        t = k * dt
        if k % 5 == 0:
            sp, ysp, done, _ = f.step(t, body_pose(p, yaw), v)
            if done:
                break
        p_new = p + (sp - p) * (dt / 0.3)
        v = (p_new - p) / dt
        p = p_new
        yaw = float(wrap(yaw + np.clip(wrap(ysp - yaw) * min(1.0, dt / 0.2), -np.radians(60) * dt, np.radians(60) * dt)))
        ts.append(t)
        ps.append(p.copy())
        ys.append(yaw)
    return np.asarray(ts), np.asarray(ps), np.asarray(ys)


def test_oblique_wall_approach_by_the_follower_no_false_alarm():
    """The closed-loop false alarm (SCOUT_NAV.md §10): the route around the low box approaches the MAPPED north wall
    diagonally (31 deg) to a vertex 0.65 m from it, then runs along it. Flown by the real follower with the corridor
    TTC law (DN-57), the looming detector must give no emergency and no obstacle."""
    path = [[2.0, 4.2, 1.1], [3.9, 5.35, 1.1], [5.2, 5.35, 1.1]]
    ts, pos, yaw = _follower_flight(path, float(np.arctan2(1.15, 1.9)))
    res = run(_frames(ts, pos, yaw))
    finite = [r["out"].min_ttc_s for r in res if np.isfinite(r["out"].min_ttc_s)]
    RESULTS["oblique_wall_follower"] = {"frames": len(res), "obstacles": len(_obstacles(res)),
                                        "emergencies": int(sum(r["out"].emergency for r in res)),
                                        "min_ttc_s": float(min(finite)) if finite else None, **_timing(res)}
    assert RESULTS["oblique_wall_follower"]["obstacles"] == 0, RESULTS["oblique_wall_follower"]
    assert RESULTS["oblique_wall_follower"]["emergencies"] == 0, RESULTS["oblique_wall_follower"]


def test_no_closing_motion_no_output_and_config_reads_shared_values():
    cfg = load_config()
    assert cfg["ttc_emergency_s"] == 1.5 and cfg["hull_radius_m"] == 0.20
    assert abs(cfg["top_clear_m"] - 0.70) < 1e-9
    assert set(DEFAULTS) <= set(cfg)
    ts = np.arange(0, 1.5, 0.01)                                     # hover
    pos = np.tile([3.0, 4.0, 1.1], (len(ts), 1))
    res = run(_frames(ts, pos, np.zeros(len(ts))), extra=[LOW_BOX])
    assert all(r["out"].reason == "no_closing_motion" and not r["out"].obstacles for r in res)


if __name__ == "__main__":
    import json
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {fn.__name__} ({time.perf_counter() - t0:.1f} s)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(json.dumps(RESULTS, indent=1, default=float))
    sys.exit(1 if failed else 0)

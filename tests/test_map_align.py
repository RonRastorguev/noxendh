"""Map <-> odom alignment (localization/map_align.py, docs/SCOUT_NAV.md §4) on SYNTHETIC odom + fixes, numpy only.

A small truth model stands in for PX4: the odom frame drifts against the map (x/y drift + random walk, yaw drift),
with a rangefinder height reference the odom z is the height above the surface PX4 is referenced to (first-order
lag `px4_tau` after a terrain step), `dist_bottom_m` is the raw range, frame resets jump the odom frame. Fixes are the
true body pose at the IMAGE stamp plus noise, delivered 50-200 ms later, some of them gross outliers.
The bounds are regression bounds on this synthetic setup, not real-world accuracy claims.
Runs under pytest or `py -3.9 tests/test_map_align.py`.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from localization.geometry import body_pose, inv_T, make_T, rot_z  # noqa: E402
from localization.map_align import MapOdomAligner, wrap, yaw_of  # noqa: E402
from navigation.types import OdomSample  # noqa: E402

DT = 0.02                 # odom 50 Hz
FIX_DT = 0.1              # camera 10 Hz
FC_H = 0.0559             # scout_hw.yaml derived.fc_height_at_rest_m (read there; repeated here only as test data)


def flat(x, y, z):
    return 0.0


def box_terrain(x, y, z, lo=(2.0, -1.0), hi=(3.4, 1.0), top=0.7):
    return top if (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and z > top) else 0.0


class World:
    """Truth: T_map_body(t); the FC's odom = A_true^-1 (map -> odom) with A_true drifting; z per height ref."""

    def __init__(self, seed=0, height_ref="range", terrain=flat, px4_tau=0.0, drift_v=(0.0, 0.0), drift_yaw_dps=0.0,
                 rw_xy=0.0, psi0=0.0, t0=(0.0, 0.0), beta=0.0, rng_noise=0.004, dist_bottom=True,
                 fix_sigma=0.015, fix_sigma_deg=0.5, beam_r=0.0):
        self.rng = np.random.default_rng(seed)
        self.height_ref, self.terrain, self.px4_tau = height_ref, terrain, px4_tau
        self.drift_v, self.drift_yaw = np.asarray(drift_v, float), math.radians(drift_yaw_dps)
        self.rw_xy, self.rng_noise, self.with_dist = rw_xy, rng_noise, dist_bottom
        self.psi, self.t, self.beta = float(psi0), np.asarray(t0, float).copy(), float(beta)
        self.T_px4 = None
        self.t_last = None
        self.p_odom_prev = None
        self.rc = 0
        self.fix_sigma, self.fix_sigma_deg = fix_sigma, fix_sigma_deg
        self.T_map_body = None
        self.beam_r = beam_r              # > 0: a finite footprint near an edge returns either surface or a mix

    def offset(self):
        return (self.T_px4 if self.height_ref == "range" else 0.0) + self.beta

    def step(self, t, T_map_body) -> OdomSample:
        dt = 0.0 if self.t_last is None else t - self.t_last
        self.t_last = t
        self.T_map_body = np.asarray(T_map_body, float)
        p = self.T_map_body[:3, 3]
        # odom drift: rotate about the vehicle (as a gyro yaw drift does), plus x/y drift + random walk
        if dt > 0:
            dpsi = self.drift_yaw * dt
            pm_xy = self.map_xy(self.odom_xy_of(p))
            self.psi += dpsi
            self.t = pm_xy - rot_z(self.psi)[:2, :2] @ self.odom_xy_of_prev
            self.t += self.drift_v * dt + self.rng.normal(0, self.rw_xy * math.sqrt(dt), 2)
        terr = self.beam_surface(p)          # PX4 fuses the same range reading we get
        if self.T_px4 is None:
            self.T_px4 = terr
        elif self.px4_tau <= 0:
            self.T_px4 = terr
        else:
            self.T_px4 += (1 - math.exp(-dt / self.px4_tau)) * (terr - self.T_px4)
        T_odom_map = inv_T(make_T(rot_z(self.psi), (self.t[0], self.t[1], self.offset())))
        T_ob = T_odom_map @ self.T_map_body
        self.odom_xy_of_prev = T_ob[:2, 3].copy()
        v = np.zeros(3) if self.p_odom_prev is None or dt <= 0 else (T_ob[:3, 3] - self.p_odom_prev) / dt
        self.p_odom_prev = T_ob[:3, 3].copy()
        dist = None
        if self.with_dist:
            dist = float(p[2] - terr + self.rng.normal(0, self.rng_noise))
        return OdomSample(stamp=t, T_odom_body=T_ob, v_odom=v, reset_counter=self.rc, dist_bottom_m=dist)

    def beam_surface(self, p):
        h0 = self.terrain(p[0], p[1], p[2])
        if self.beam_r <= 0:
            return h0
        hs = [self.terrain(p[0] + dx, p[1] + dy, p[2]) for dx, dy in
              ((self.beam_r, 0), (-self.beam_r, 0), (0, self.beam_r), (0, -self.beam_r))]
        if max(hs + [h0]) - min(hs + [h0]) < 1e-9:
            return h0
        u = self.rng.random()                 # footprint straddles an edge: flicker between surfaces, or a mix
        if u < 0.4:
            return float(self.rng.choice(hs + [h0]))
        if u < 0.6:
            return float(self.rng.uniform(min(hs), max(hs)))
        return h0

    # helpers for the drift update (odom xy of the body under the CURRENT alignment)
    def odom_xy_of(self, p):
        if not hasattr(self, "odom_xy_of_prev"):
            self.odom_xy_of_prev = rot_z(-self.psi)[:2, :2] @ (p[:2] - self.t)
        return self.odom_xy_of_prev

    def map_xy(self, p_odom_xy):
        return rot_z(self.psi)[:2, :2] @ p_odom_xy + self.t

    def reset(self, dxy, dz, dyaw_deg):
        """FC frame reset: new odom = D old odom. A_true' = A_true D^-1."""
        dpsi = math.radians(dyaw_deg)
        self.psi -= dpsi
        self.t = self.t - rot_z(self.psi)[:2, :2] @ np.asarray(dxy, float)
        self.beta -= dz
        self.odom_xy_of_prev = rot_z(dpsi)[:2, :2] @ self.odom_xy_of_prev + np.asarray(dxy, float)
        self.p_odom_prev = None
        self.rc += 1

    def fix(self, T_true, outlier=False):
        T = np.asarray(T_true, float).copy()
        dp = self.rng.normal(0, self.fix_sigma, 3)
        dyaw = math.radians(self.rng.normal(0, self.fix_sigma_deg))
        if outlier:
            d = self.rng.normal(0, 1, 3)
            dp += d / np.linalg.norm(d) * self.rng.uniform(0.4, 1.5)
            dyaw += math.radians(self.rng.uniform(8, 30)) * self.rng.choice([-1, 1])
        T[:3, :3] = rot_z(dyaw) @ T[:3, :3]
        T[:3, 3] += dp
        return T

    def to_map_xyz(self, p_odom):
        """Where a commanded odom point really is in map (PX4 tracks it on its own estimate)."""
        xy = self.map_xy(np.asarray(p_odom)[:2])
        return np.array([xy[0], xy[1], p_odom[2] + self.offset()])


def run(world, al, traj, t_end, t0=0.0, fix_rate=FIX_DT, latency=(0.05, 0.2), outlier_p=0.0, fixes_off=None,
        on_tick=None, events=None):
    """Drive world + aligner. traj(t) -> T_map_body. Fixes stamped at the image time, delivered after a latency."""
    rng = np.random.default_rng(1234)
    pending = []                     # (deliver_t, stamp, T_fix, is_outlier)
    out = {"err_xy": [], "err_z": [], "err_yaw": [], "t": [], "fix": [], "outliers": []}
    next_img = t0
    last_outlier = False
    n = int(round((t_end - t0) / DT))
    for i in range(n + 1):
        t = t0 + i * DT
        if events:
            for (te, fn) in events:
                if abs(t - te) < DT / 2:
                    fn(world, al, t)
        T_true = traj(t)
        al.add_odom(world.step(t, T_true))
        while t >= next_img - 1e-9:
            off = fixes_off is not None and fixes_off[0] <= next_img < fixes_off[1]
            if not off:
                is_out = (not last_outlier) and rng.random() < outlier_p
                last_outlier = is_out
                Tf = world.fix(traj(next_img), outlier=is_out)
                lat = rng.uniform(*latency) if isinstance(latency, tuple) else latency
                pending.append((next_img + lat, next_img, Tf, is_out))
            next_img += fix_rate
        pending.sort(key=lambda q: q[0])
        while pending and pending[0][0] <= t + 1e-9:
            _, st, Tf, is_out = pending.pop(0)
            r = al.add_fix(st, Tf)
            out["fix"].append((st, r, is_out))
        pm = al.pose_map(t)
        if pm is not None:
            Tm = pm[0]
            out["t"].append(t)
            out["err_xy"].append(float(np.linalg.norm(Tm[:2, 3] - T_true[:2, 3])))
            out["err_z"].append(float(Tm[2, 3] - T_true[2, 3]))
            out["err_yaw"].append(math.degrees(abs(wrap(yaw_of(Tm[:3, :3]) - yaw_of(T_true[:3, :3])))))
        if on_tick:
            on_tick(world, al, t, T_true)
    for k in ("err_xy", "err_z", "err_yaw", "t"):
        out[k] = np.asarray(out[k])
    return out


def loop_traj(t, c=(4.0, 3.0), r=(2.2, 1.6), w=0.2, z0=1.0):
    x, y = c[0] + r[0] * math.cos(w * t), c[1] + r[1] * math.sin(w * t)
    vx, vy = -r[0] * w * math.sin(w * t), r[1] * w * math.cos(w * t)
    return body_pose((x, y, z0 + 0.3 * math.sin(0.3 * t)), math.atan2(vy, vx), pitch=0.05 * math.sin(1.3 * t),
                     roll=0.04 * math.sin(0.9 * t))


def _init_at(world, al, traj, t=0.0):
    al.add_odom(world.step(t, traj(t)))
    al.init_known_pose(traj(t), t)


# ------------------------------------------------------------------------------------------------------- tests
def test_init_known_pose_on_ground():
    T_pad = body_pose((1.0, 2.0, FC_H), math.radians(30.0))
    for href in ("range", "baro"):
        w = World(height_ref=href, psi0=math.radians(10.0), t0=(0.3, -0.2), beta=0.0 if href == "range" else 0.42)
        al = MapOdomAligner()
        al.set_height_ref(href)
        al.set_terrain(flat)
        al.init_known_pose(T_pad, 0.0)                     # before any odom: applied at the first sample
        assert not al.valid(0.0)
        s = w.step(0.0, T_pad)
        if href == "range":
            assert abs(s.T_odom_body[2, 3] - FC_H) < 1e-9  # range height ref: odom z on the ground = fc height
        al.add_odom(s)
        for i in range(1, 50):
            al.add_odom(w.step(i * DT, T_pad))
        Tm, v = al.pose_map(49 * DT)
        assert np.allclose(Tm, T_pad, atol=1e-3), (href, Tm)          # range noise: < 1 mm
        assert np.allclose(v, 0.0)
        assert al.valid(49 * DT)
        p_o, y_o = al.setpoint_to_odom(T_pad[:3, 3], math.radians(30.0))
        assert np.allclose(p_o, s.T_odom_body[:3, 3], atol=1e-3) and abs(wrap(y_o - yaw_of(s.T_odom_body))) < 1e-9
        T_true = make_T(rot_z(w.psi), (w.t[0], w.t[1], w.offset()))
        assert np.allclose(al.T_map_odom(), T_true, atol=1e-3)
        st = al.status(1.0)
        assert st["initialized"] and st["valid"] and st["height_ref"] == href


def test_init_from_fixes_without_known_pose():
    w = World(psi0=0.3, t0=(1.0, 0.5))
    al = MapOdomAligner()
    al.set_terrain(flat)
    T = body_pose((2.0, 1.0, 1.0), 0.5)
    for i in range(10):
        al.add_odom(w.step(i * DT, T))
    r1 = al.add_fix(0.1, w.fix(T))
    assert not r1["accepted"] and not al.valid(0.18)
    r2 = al.add_fix(0.15, w.fix(T))
    assert r2["accepted"] and r2["reason"] == "init_from_fixes"
    for i in range(10, 20):
        al.add_odom(w.step(i * DT, T))
    Tm, _ = al.pose_map()
    assert np.linalg.norm(Tm[:3, 3] - T[:3, 3]) < 0.04 and al.valid(0.38)


def test_drift_tracking_with_noisy_latent_fixes():
    w = World(seed=1, drift_v=(0.015, -0.010), drift_yaw_dps=0.05, rw_xy=0.003, psi0=0.2, t0=(0.5, -0.3))
    al = MapOdomAligner()
    al.set_terrain(flat)
    _init_at(w, al, loop_traj)
    o = run(w, al, loop_traj, 90.0, t0=DT)
    m = o["t"] > 5.0
    exy, ez, eyaw = o["err_xy"][m], np.abs(o["err_z"][m]), o["err_yaw"][m]
    print(f"    drift: xy err median {np.median(exy) * 100:.1f} cm, p95 {np.percentile(exy, 95) * 100:.1f}, max "
          f"{exy.max() * 100:.1f}; z max {ez.max() * 100:.1f} cm; yaw max {eyaw.max():.2f} deg; true drift "
          f"accumulated {np.linalg.norm(w.t - (0.5, -0.3)):.2f} m / {math.degrees(w.psi - 0.2):.1f} deg")
    assert np.median(exy) < 0.02 and exy.max() < 0.05
    assert ez.max() < 0.04 and eyaw.max() < 0.6
    acc = sum(1 for _, r, _ in o["fix"] if r["accepted"])
    assert acc >= 0.98 * len(o["fix"]) and al.status(90.0)["n_reanchors"] == 0


def test_outliers_rejected():
    w = World(seed=2, drift_v=(0.01, 0.01), drift_yaw_dps=0.05, rw_xy=0.003)
    al = MapOdomAligner()
    al.set_terrain(flat)
    _init_at(w, al, loop_traj)
    o = run(w, al, loop_traj, 60.0, t0=DT, outlier_p=0.12)
    outl = [r for _, r, is_out in o["fix"] if is_out]
    inl = [r for _, r, is_out in o["fix"] if not is_out]
    n_out_acc = sum(r["accepted"] for r in outl)
    print(f"    outliers: {len(outl)} injected, {n_out_acc} accepted; inliers {sum(r['accepted'] for r in inl)}"
          f"/{len(inl)} accepted; xy err max {o['err_xy'][o['t'] > 5].max() * 100:.1f} cm")
    assert len(outl) >= 30 and n_out_acc == 0
    assert sum(r["accepted"] for r in inl) >= 0.97 * len(inl)
    assert o["err_xy"][o["t"] > 5].max() < 0.05 and al.status(60.0)["n_reanchors"] == 0

    # one gross outlier while hovering, no other fixes: the frame must not move
    T = body_pose((3.0, 2.0, 1.2), 0.4)
    w2 = World(seed=3)
    al2 = MapOdomAligner()
    al2.set_terrain(flat)
    al2.add_odom(w2.step(0.0, T))
    al2.init_known_pose(T, 0.0)
    for i in range(1, 25):
        al2.add_odom(w2.step(i * DT, T))
    before = al2.pose_map()[0]
    bad = T.copy()
    bad[:3, 3] += (0.6, -0.4, 0.3)
    bad[:3, :3] = rot_z(0.3) @ bad[:3, :3]
    r = al2.add_fix(0.4, bad)
    for i in range(25, 100):
        al2.add_odom(w2.step(i * DT, T))
    after = al2.pose_map()[0]
    assert not r["accepted"] and "rejected" in r["reason"]
    assert np.allclose(before, after, atol=1e-3)                  # range noise only (< 1 mm)


def test_latency_fix_at_image_stamp():
    """Fast motion (0.5 m/s, 30 deg/s yaw) with 200 ms camera latency: stamped at the image time the fixes fit;
    stamped at arrival they would not."""
    def traj(t):
        return body_pose((1.0 + 0.5 * t, 2.0, 1.0), math.radians(30.0) * t)
    w = World(seed=4)
    al = MapOdomAligner()
    al.set_terrain(flat)
    _init_at(w, al, traj)
    o = run(w, al, traj, 8.0, t0=DT, latency=0.2)
    res = np.array([r["residual_m"] for _, r, _ in o["fix"] if r["residual_m"] is not None])
    resd = np.array([abs(r["residual_deg"]) for _, r, _ in o["fix"] if r["residual_deg"] is not None])
    assert all(r["accepted"] for _, r, _ in o["fix"])
    assert res.max() < 0.06 and resd.max() < 2.0 and o["err_xy"].max() < 0.05
    # the same fix applied at its ARRIVAL stamp misfits by ~10 cm / 6 deg and is rejected
    t_img = 7.0
    r_late = al.add_fix(t_img + 0.2, traj(t_img))
    assert not r_late["accepted"], r_late
    # older than the odom buffer -> rejected with a reason, frame untouched
    r_old = al.add_fix(1.0, traj(1.0))
    assert not r_old["accepted"] and r_old["reason"] == "stamp_outside_odom_buffer"
    print(f"    latency 200 ms: residual max {res.max() * 100:.1f} cm / {resd.max():.2f} deg; at arrival stamp "
          f"{r_late['residual_m'] * 100:.0f} cm / {abs(r_late['residual_deg']):.1f} deg -> rejected")


def test_reanchor_after_alignment_jump_and_smoothness():
    """The odom frame jumps 0.4 m / 6 deg WITHOUT a reset counter (e.g. a flow glitch): fixes are rejected at first,
    a consistent run re-anchors; the odom setpoint of a fixed map point never moves > 3 cm per 50 ms tick."""
    def traj(t):
        return body_pose((4.0 + 0.3 * math.sin(0.4 * t), 3.0 + 0.2 * math.cos(0.3 * t), 1.2), 0.3 * math.sin(0.2 * t))
    w = World(seed=5, drift_v=(0.01, -0.01), drift_yaw_dps=0.05, rw_xy=0.003)
    al = MapOdomAligner()
    al.set_terrain(flat)
    _init_at(w, al, traj)
    target = np.array([4.5, 3.0, 1.2])
    sps = []

    def jump(world, aligner, t):
        world.psi += math.radians(6.0)
        world.t += (0.3, -0.25)

    def tick(world, aligner, t, T_true):              # every odom sample (20 ms)
        p, y = aligner.setpoint_to_odom(target, 0.2)
        sps.append((t, p, y))
    o = run(w, al, traj, 40.0, t0=DT, outlier_p=0.05, on_tick=tick, events=[(20.0, jump)])
    st = al.status(40.0)
    assert st["n_reanchors"] == 1, st
    ra = [st_ for st_, r, _ in o["fix"] if r["reason"].startswith("reanchor")][0]
    assert 20.0 < ra < 21.5, ra
    # smoothness of the commanded odom setpoint (from alignment updates; the vehicle hovers near `target`)
    # worst change over any 60 ms window (>= one 50 ms setpoint tick)
    dp = [float(np.linalg.norm(b[1] - a[1])) for a, b in zip(sps, sps[3:])]
    dy = [math.degrees(abs(wrap(b[2] - a[2]))) for a, b in zip(sps, sps[3:])]
    m_after = o["t"] > 24.0
    print(f"    re-anchor at {ra - 20.0:.2f} s after the jump; max setpoint step {max(dp) * 100:.2f} cm / "
          f"{max(dy):.2f} deg per 60 ms; xy err after 4 s max {o['err_xy'][m_after].max() * 100:.1f} cm")
    assert max(dp) <= 0.03 and max(dy) <= 0.5
    assert o["err_xy"][m_after].max() < 0.05 and o["err_yaw"][m_after].max() < 0.6
    m_before = (o["t"] > 5.0) & (o["t"] < 20.0)
    assert o["err_xy"][m_before].max() < 0.05


def test_frame_reset_reanchors_without_jump():
    w = World(seed=6, drift_v=(0.01, 0.0), drift_yaw_dps=0.05, rw_xy=0.002, beta=0.0)
    al = MapOdomAligner()
    al.set_terrain(flat)
    _init_at(w, al, loop_traj)
    rec = {}

    def do_reset(world, aligner, t):
        rec["before"] = aligner.pose_map()[0]
        world.reset((1.5, -0.8), 0.25, 35.0)

    def tick(world, aligner, t, T_true):
        if "before" in rec and "after" not in rec:
            rec["after"] = aligner.pose_map()[0]
            rec["T_true_step"] = T_true

    o = run(w, al, loop_traj, 40.0, t0=DT, on_tick=tick, events=[(15.0, do_reset)])
    st = al.status(40.0)
    assert st["n_resets"] == 1
    lr = st["last_reset"]
    assert abs(lr["dxy_m"][0] - 1.5) < 0.02 and abs(lr["dxy_m"][1] + 0.8) < 0.02, lr
    assert abs(lr["dz_m"] - 0.25) < 0.02 and abs(lr["dyaw_deg"] - 35.0) < 0.5, lr
    # map pose continuous across the reset (one 20 ms step of real motion, ~1 cm at 0.4 m/s)
    jump = float(np.linalg.norm(rec["after"][:3, 3] - rec["before"][:3, 3]))
    assert jump < 0.03, jump
    late = [r for st_, r, _ in o["fix"] if st_ > 14.8]
    assert all(r["accepted"] for r in late), [r["reason"] for r in late if not r["accepted"]][:3]
    assert o["err_xy"].max() < 0.05 and np.abs(o["err_z"]).max() < 0.04 and st["n_reanchors"] == 0
    print(f"    reset: delta {lr['dxy_m']} m, {lr['dz_m']:.3f} m, {lr['dyaw_deg']:.2f} deg; map pose step {jump * 100:.1f}"
          f" cm; xy err max {o['err_xy'].max() * 100:.1f} cm; all {len(late)} fixes across it accepted")


def _terrain_closed_loop(px4_tau, use_dist, model_up=0.5, model_down=0.5, seed=7, map_fn=box_terrain,
                         true_fn=box_terrain, x_of_t=None, y=0.0, beam_r=0.0, t_end=20.0):
    """Take off at (0.5, 0), hover at 1.2 m, from 6 s fly x 0.5 -> 5.0 at 0.4 m/s over a 0.7 m box (x 2.0-3.4), PX4-like z
    loop on ITS estimate (P 1.0, +1.0 / -0.7 m/s). Returns the vehicle's true map z history and command errors.
    `map_fn` is what the aligner believes (the static map), `true_fn` the world; `x_of_t` overrides the path."""
    cfg = {"use_dist_bottom": use_dist, "hgt_model_tau_up_s": model_up, "hgt_model_tau_down_s": model_down}
    w = World(seed=seed, terrain=true_fn, px4_tau=px4_tau, drift_v=(0.005, 0.0), rw_xy=0.002,
              dist_bottom=use_dist, beam_r=beam_r)
    al = MapOdomAligner(cfg)
    al.set_height_ref("range")
    al.set_terrain(map_fn)
    z = FC_H
    x = 0.5
    T0 = body_pose((x, y, z), 0.0)
    al.add_odom(w.step(0.0, T0))
    al.init_known_pose(T0, 0.0)
    rng = np.random.default_rng(seed)
    pending, next_img = [], 0.1
    zs, cmd_err, ts = [], [], []
    al.test_fix_log, al.test_pose_z_err = [], []
    t = 0.0
    while t < t_end:
        t += DT
        z_sp_map = 1.2 if t > 0.5 else FC_H
        if x_of_t is not None:
            x = x_of_t(t)
        elif t > 6.0:
            x = min(5.0, x + 0.4 * DT)
        p_odom, _ = al.setpoint_to_odom((x, y, z_sp_map), 0.0)
        z_odom_now = z - w.offset()                      # what PX4 believes (its own odom z)
        vz = float(np.clip(1.0 * (p_odom[2] - z_odom_now), -0.7, 1.0))
        z = max(FC_H, z + vz * DT)
        T = body_pose((x, y, z), 0.0)
        al.add_odom(w.step(t, T))
        if t >= next_img:
            pending.append((next_img + rng.uniform(0.05, 0.2), next_img, w.fix(T)))
            next_img += FIX_DT
        while pending and pending[0][0] <= t:
            _, st, Tf = pending.pop(0)
            al.test_fix_log.append(al.add_fix(st, Tf))
        if t > 6.0:                                       # takeoff settled (P 1.0: e^-5)
            zs.append(z)
            ts.append(t)
            cmd_err.append(w.to_map_xyz(al.setpoint_to_odom((x, y, 1.2), 0.0)[0])[2] - 1.2)
            al.test_pose_z_err.append(al.pose_map(t)[0][2, 3] - z)
    return np.array(ts), np.array(zs), np.array(cmd_err), al


def test_terrain_step_range_height_ref():
    """Crossing a 0.7 m box with a rangefinder height reference: the vehicle's true map z stays at 1.2 m for a fast
    and a slow PX4 step lag when dist_bottom is available (no lag assumption); without it, it stays right when the
    lag model matches, and the 'conservative' model (slow up, instant down) only ever errs upwards."""
    rows = []
    for tau in (0.05, 1.5):
        ts, zs, ce, al = _terrain_closed_loop(tau, True)
        rows.append(("dist_bottom", tau, zs, ce))
        assert np.abs(zs - 1.2).max() < 0.05, (tau, np.abs(zs - 1.2).max())
        assert al.status(20.0)["height_source"] == "dist_bottom"
    ts, zs, ce, al = _terrain_closed_loop(0.5, False, 0.5, 0.5)
    rows.append(("lag model, matched 0.5", 0.5, zs, ce))
    assert np.abs(zs - 1.2).max() < 0.05 and al.status(20.0)["height_source"] == "lag_model"
    for tau in (0.05, 1.5):
        ts, zs, ce, al = _terrain_closed_loop(tau, False, 3.0, 0.0)
        rows.append(("lag model, conservative", tau, zs, ce))
        assert (zs - 1.2).min() > -0.03, (tau, (zs - 1.2).min())
    ts, zs, ce, al = _terrain_closed_loop(1.5, False, 0.5, 0.5)
    rows.append(("lag model, mismatched 0.5", 1.5, zs, ce))
    for name, tau, zs, ce in rows:
        print(f"    {name:26s} PX4 tau {tau:4.2f} s: true z {(zs - 1.2).min() * 100:+5.1f} .. "
              f"{(zs - 1.2).max() * 100:+5.1f} cm; instantaneous command error max {np.abs(ce).max() * 100:5.1f} cm")


def _edge_path(t):
    """0.5 -> 1.95 (hover 3 s between the dilated map edge 1.9 and the true edge 2.0), -> 3.45 (hover 3 s past the
    true far edge 3.4, before the map's 3.5), -> 5.0; 0.4 m/s."""
    legs = [(6.0, 0.5), (None, 1.95), (3.0, None), (None, 3.45), (3.0, None), (None, 5.0)]
    x, tt = 0.5, 0.0
    for dur, goal in legs:
        if goal is None:
            seg = dur
            if t <= tt + seg:
                return x
        else:
            seg = abs(goal - x) / 0.4 if dur is None else dur
            if t <= tt + seg:
                return x + (goal - x) * (t - tt) / seg if dur is None else x
            x = goal
        tt += seg
    return x


def test_mapped_box_edge_beam_vs_map_disagree():
    """The low_box sim bug: near a MAPPED box edge the voxel map says "box top" where the rangefinder still sees the
    floor (and vice versa), with a finite beam that flickers / mixes on the edge. The terrain under the vehicle must
    follow what the range sees: true z stays at 1.2 m, the map z estimate stays right, no PnP z fix is rejected and
    nothing re-anchors; for a fast and a slow PX4 step lag, a dilated and an eroded map edge."""
    dil = lambda x, y, z: box_terrain(x, y, z, lo=(1.9, -1.1), hi=(3.5, 1.1))        # noqa: E731
    ero = lambda x, y, z: box_terrain(x, y, z, lo=(2.1, -0.9), hi=(3.3, 0.9))        # noqa: E731
    for name, mfn in (("dilated map", dil), ("eroded map", ero)):
        for tau in (0.05, 1.5):
            ts, zs, ce, al = _terrain_closed_loop(tau, True, map_fn=mfn, x_of_t=_edge_path, y=0.3, beam_r=0.06,
                                                  t_end=24.0, seed=11)
            pz = np.abs(np.asarray(al.test_pose_z_err))
            n_zrej = sum(1 for r in al.test_fix_log if not r["accepted"] and "z residual" in r["reason"])
            st = al.status(24.0)
            print(f"    {name:11s} PX4 tau {tau:4.2f} s: true z {(zs - 1.2).min() * 100:+5.1f} .. "
                  f"{(zs - 1.2).max() * 100:+5.1f} cm; |pose z err| max {pz.max() * 100:4.1f} cm (p99 "
                  f"{np.percentile(pz, 99) * 100:4.1f}); z-rejected fixes {n_zrej}; reanchors {st['n_reanchors']};"
                  f" surface steps {st['n_surface_steps']}")
            assert np.abs(zs - 1.2).max() < 0.05, (name, tau)
            assert np.percentile(pz, 99) < 0.05 and pz.max() < 0.12, (name, tau, pz.max())
            assert n_zrej == 0 and st["n_reanchors"] == 0, (name, tau, n_zrej, st["n_reanchors"])


def test_baro_height_ref_is_plain_offset():
    def traj(t):
        return body_pose((0.5 + 0.4 * t, 0.0, 1.2), 0.0)
    w = World(seed=8, height_ref="baro", terrain=box_terrain, beta=0.37, drift_v=(0.005, 0.0))
    al = MapOdomAligner()
    al.set_height_ref("baro")
    al.set_terrain(box_terrain)                       # ignored with a baro height reference
    _init_at(w, al, traj)
    zs = []

    def tick(world, aligner, t, T_true):
        zs.append(aligner.setpoint_to_odom((T_true[0, 3], 0.0, 1.2), 0.0)[0][2])
    o = run(w, al, traj, 10.0, t0=DT, on_tick=tick)
    assert np.ptp(zs) < 0.02 and abs(zs[-1] - (1.2 - 0.37)) < 0.02
    assert np.abs(o["err_z"]).max() < 0.03 and al.status(10.0)["height_source"] == "offset"


def test_validity_goes_false_without_fixes():
    def traj(t):
        return body_pose((0.5 + 0.5 * t, 1.0, 1.0), 0.0)
    w = World(seed=9, drift_v=(0.015, 0.0), rw_xy=0.002)
    al = MapOdomAligner()
    al.set_terrain(flat)
    _init_at(w, al, traj)
    flags = {}

    def tick(world, aligner, t, T_true):
        for q in (9.9, 12.0, 13.0, 16.0, 19.9, 20.4):
            if abs(t - q) < DT / 2:
                flags[q] = aligner.valid(t)
                if q == 16.0:
                    flags["status16"] = aligner.status(t)
    o = run(w, al, traj, 21.0, t0=DT, fixes_off=(10.0, 20.0), on_tick=tick)
    st = flags.pop("status16")
    print(f"    validity while moving 0.5 m/s, fixes off 10-20 s: {flags}; at 16 s: {st['valid_reason']}")
    assert flags[9.9] and flags[12.0] and flags[13.0]        # odom-only for a few seconds is fine
    assert not flags[16.0] and not flags[19.9]               # drift budget exceeded
    assert flags[20.4]                                       # the first fixes after the gap pass the widened gate
    # hovering: the drift budget lasts longer than max_fix_age_s (time drift only), then it also goes false
    T = body_pose((2.0, 2.0, 1.0), 0.0)
    w2 = World(seed=10)
    al2 = MapOdomAligner()
    al2.add_odom(w2.step(0.0, T))
    al2.init_known_pose(T, 0.0)
    for i in range(1, 501):
        al2.add_odom(w2.step(i * DT, T))
    assert al2.valid(10.0) and al2.status(10.0)["fix_age_s"] > 3.0      # 10 s hover without a fix: still valid
    assert not al2.valid(10.6)                              # odom stale (newest sample at 10.0 s)
    assert st["n_fixes_accepted"] > 0 and st["fix_age_s"] > 5.0



def test_candidates_keep_the_surface_under_the_body_not_a_low_edge_voxel():
    """A 0.5 m pad: the edge voxels' point mean reads 0.355, the top 0.40. The merged candidate must be the surface
    straight under the body (sim hover_test: keeping the lowest biased map z by -5.5 cm)."""
    a = MapOdomAligner()
    a.set_height_ref("range")

    def terrain(x, y, z):
        r = max(abs(x - 1.0), abs(y - 4.0))
        return 0.40 if r < 0.2 else (0.355 if r <= 0.25 + 1e-9 else 0.0)
    a.set_terrain(terrain)
    c = a._candidates(np.array([1.0, 4.0, 2.456]), 0.0)       # uninitialised: identity alignment
    assert 0.40 in np.round(c, 3) and 0.355 not in np.round(c, 3), c

if __name__ == "__main__":
    import time
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        t0 = time.time()
        fn()
        print(f"  [ok] {name} ({time.time() - t0:.1f} s)")
    print(f"{len(tests)}/{len(tests)} passed")

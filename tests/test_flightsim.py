"""Closed-loop flight sim (flightsim/, docs/SCOUT_NAV.md §8): vehicle model, estimator, world, loop plumbing.

None of these tests needs the other flight-software pieces (aligner, looming, navigation, mission): the scripted
flight uses a dummy mission through run_scenario's `mission_factory`. Runs under pytest or
`py -3.9 tests/test_flightsim.py` (~1 min).
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flightsim import loop, metrics, scenarios  # noqa: E402
from flightsim.render import FastRenderer  # noqa: E402
from flightsim.vehicle_sim import SimVehicle, yaw_of  # noqa: E402
from flightsim.world import SimWorld, check_texture_identity  # noqa: E402
from localization import synthetic_room as sr  # noqa: E402
from localization.geometry import body_pose  # noqa: E402
from navigation.types import Setpoint  # noqa: E402

NO_DRIFT = {"estimator": {"init_offset_sigma_m": 0.0, "init_yaw": "heading", "bias_sigma_mps": 0.0,
                          "vel_scale_sigma": 0.0, "yaw_rate_bias_sigma_dps": 0.0, "yaw_rw_dps_sqrt_s": 0.0,
                          "hgt_noise_m": 0.0, "hgt_step_tau_s": 0.0}}
_W = {}


def _world():
    if "w" not in _W:
        _W["w"] = SimWorld()
    return _W["w"]


def _merge(a, b):
    return loop._merge(a, b)


def _vehicle(world=None, cfg=None, seed=0, at=(2.0, 4.0), yaw=0.0):   # 1 m off the raised pad at A
    w = world or _world()
    return SimVehicle(w, body_pose((at[0], at[1], 0.0), yaw), _merge(NO_DRIFT, cfg), seed=seed)


def _run(veh, seconds, sp_fn=None, every=5, on_step=None):
    """Advance `seconds`; stream sp_fn(veh) -> (p_odom, yaw_odom) every `every` steps (20 Hz) if given."""
    n = int(round(seconds / veh.dt))
    out = []
    for i in range(n):
        if sp_fn is not None and i % every == 0:
            p, yaw = sp_fn(veh)
            veh.send_setpoint(Setpoint(veh.t, np.asarray(p, float), float(yaw)))
        veh.step()
        if on_step:
            out.append(on_step(veh))
    return out


def _airborne(veh, z=1.0):
    """Stream a hold, engage offboard, arm, climb to odom z. -> the odom hold position."""
    o = veh.odometry()
    p0, y0 = o.T_odom_body[:3, 3].copy(), yaw_of(o.T_odom_body)
    _run(veh, 1.1, lambda v: (p0, y0))
    veh.request_offboard()
    veh.request_arm()
    assert veh.armed and veh.mode == "offboard"
    tgt = np.array([p0[0], p0[1], z])
    _run(veh, 4.0, lambda v: (tgt, y0))
    assert not veh.on_ground and abs(veh.odometry().T_odom_body[2, 3] - z) < 0.03
    return tgt, y0


# --- vehicle dynamics -------------------------------------------------------------------------------------------

def test_step_response_and_tilt():
    veh = _vehicle()
    tgt, y0 = _airborne(veh, 1.2)
    p_start = veh.p.copy()
    goal = tgt + np.array([1.0, 0.0, 0.0])
    rec = _run(veh, 6.0, lambda v: (goal, y0), on_step=lambda v: (v.p[0], v.pitch, v.v[0]))
    x = np.array([r[0] for r in rec]) - p_start[0]
    pitch = np.array([r[1] for r in rec])
    t = np.arange(1, len(x) + 1) * veh.dt
    assert abs(x[-1] - 1.0) < 0.02, x[-1]
    assert x.max() < 1.0 + 0.05, x.max()                          # no big overshoot (P + first-order lag)
    t63 = t[np.argmax(x >= 0.632)]
    assert 0.9 < t63 < 2.0, t63                                     # ~1/MPC_XY_P + the velocity lag
    assert pitch.max() > math.radians(1.0)                          # accelerating forward pitches the nose DOWN
    assert pitch.min() < -math.radians(0.5)                         # braking pitches it up
    assert abs(veh.p[1] - p_start[1]) < 1e-6 and abs(veh.roll) < 1e-3


def test_velocity_caps_and_yaw_rate():
    veh = _vehicle(at=(1.0, 2.1))                                    # clear lane south of the pillar
    tgt, y0 = _airborne(veh, 1.0)
    far = tgt + np.array([5.0, 0.0, 0.0])
    sp = _run(veh, 3.0, lambda v: (far, y0), on_step=lambda v: math.hypot(v.v[0], v.v[1]))
    assert 1.45 < max(sp) <= 1.5 + 1e-6, max(sp)
    assert not veh.collisions
    up = far + np.array([0.0, 0.0, 1.6])
    xy = lambda v: v.odometry().T_odom_body[:2, 3]  # noqa: E731
    vz = _run(veh, 2.0, lambda v: (np.r_[xy(v), up[2]], y0),
              on_step=lambda v: v.v[2])
    assert 0.95 < max(vz) <= 1.0 + 1e-6, max(vz)
    vz = _run(veh, 2.5, lambda v: (np.r_[xy(v), 0.6], y0),
              on_step=lambda v: v.v[2])
    assert -0.7 - 1e-6 <= min(vz) < -0.65, min(vz)
    y_start = veh.yaw
    yaws = _run(veh, 1.0, lambda v: (v.odometry().T_odom_body[:3, 3], y0 + math.pi * 0.9), on_step=lambda v: v.yaw)
    rate = np.abs(np.diff(np.unwrap([y_start] + yaws))) / veh.dt
    assert rate.max() <= math.radians(60.0) + 1e-6 and rate.max() > math.radians(59.0)


def test_land_mode_disarms_on_touchdown():
    veh = _vehicle()
    _airborne(veh, 1.0)
    veh.request_land()
    assert veh.mode == "land"
    vz = _run(veh, 1.5, on_step=lambda v: v.v[2])
    assert abs(min(vz) + 0.6) < 0.01
    _run(veh, 1.0)
    assert veh.on_ground and veh.landed and veh.armed               # touched down; COM_DISARM_LAND not yet elapsed
    _run(veh, 2.1)
    assert not veh.armed and not veh.collisions
    assert abs(veh.odometry().T_odom_body[2, 3] - veh.rest_h) < 1e-6


def test_offboard_loss_holds_position():
    veh = _vehicle()
    tgt, y0 = _airborne(veh, 1.0)
    p = veh.p.copy()
    _run(veh, 0.9)                                                   # stream stopped, still under COM_OF_LOSS_T
    assert veh.mode == "offboard" and not veh.failsafe
    _run(veh, 0.2)
    assert veh.mode == "position" and veh.failsafe and not veh.status().offboard
    _run(veh, 3.0)
    assert np.linalg.norm(veh.p - p) < 0.03 and veh.armed and not veh.on_ground
    assert any(e["event"] == "offboard_loss" for e in veh.events)


def test_arming_rules():
    veh = _vehicle()
    veh.request_arm()
    assert not veh.armed                                             # not in offboard
    veh.request_offboard()
    assert veh.mode != "offboard"                                    # no setpoint stream at all
    o = veh.odometry()
    hold = lambda v: (o.T_odom_body[:3, 3], yaw_of(o.T_odom_body))  # noqa: E731
    _run(veh, 0.6, hold)
    veh.request_offboard()
    assert veh.mode != "offboard"                                    # streamed < 1 s
    _run(veh, 0.5, hold)
    veh.request_offboard()
    assert veh.mode == "offboard"
    veh.request_arm()
    assert veh.armed
    _run(veh, 10.1, hold)                                            # armed, never commanded up
    assert not veh.armed and veh.on_ground                           # COM_DISARM_PRFLT
    refused = [e for e in veh.events if e["event"] in ("arm_refused", "offboard_refused")]
    assert len(refused) == 3
    # the simulated pilot arms once offboard is engaged; nothing arms without offboard
    v2 = _vehicle(cfg={"fc": {"pilot_arm_after_s": 0.5}})
    _run(v2, 2.0)
    assert not v2.armed
    _run(v2, 1.2, hold)
    v2.request_offboard()
    _run(v2, 0.6, hold)
    assert v2.armed and any(e.get("by") == "pilot" for e in v2.events)


# --- estimator ----------------------------------------------------------------------------------------------------

def test_range_height_reference_over_a_box():
    w = SimWorld(obstacles=[("slab", (3.0, 2.5, 0.0), (5.0, 4.5, 0.7))])
    assert abs(w.terrain_z(4.0, 3.5) - 0.7) < 1e-9 and w.terrain_z(2.0, 3.5) == 0.0
    res = {}
    for tau in (0.0, 1.5):
        veh = _vehicle(world=w, cfg={"estimator": {"hgt_step_tau_s": tau}}, at=(1.5, 3.5))
        o = veh.odometry()
        assert abs(o.T_odom_body[2, 3] - veh.rest_h) < 1e-9 and abs(o.dist_bottom_m - veh.rest_h) < 1e-9
        tgt, y0 = _airborne(veh, 1.2)
        over = tgt + np.array([2.5, 0.0, 0.0])                      # odom x 2.5 m ahead = the middle of the slab
        rec = _run(veh, 10.0, lambda v: (over, y0),
                   on_step=lambda v: (v.p[0], v.p[2], v.odometry().T_odom_body[2, 3], v._height_true()))
        a = np.array(rec)
        assert abs(a[-1, 1] - (0.7 + 1.2)) < 0.03, a[-1]           # same odom z -> climbed by the slab height
        assert abs(a[-1, 2] - 1.2) < 0.03 and abs(a[-1, 3] - 1.2) < 0.03
        k = int(np.argmax(a[:, 0] >= 3.0))                           # crossing the slab edge
        res[tau] = float(np.max(np.abs(a[k:k + 100, 2] - a[k:k + 100, 3])))   # odom z vs true height, 1 s
        assert not veh.collisions
    assert res[0.0] < 0.02 and res[1.5] > 0.3, res                   # instant follow vs a slow (lagged) step


def test_drift_statistics():
    """Gauss-Markov velocity bias: the TRUE position of a vehicle holding an odom setpoint drifts with the analytic
    variance; same seed -> same drift; no-drift config -> no drift; yaw drift ~ rate bias * T."""
    sig, tau, T = 0.05, 30.0, 12.0
    var_th = 2 * sig ** 2 * tau * (T - 2 * tau * (1 - math.exp(-T / tau)) + tau / 2 * (1 - math.exp(-2 * T / tau)))
    cfg = {"estimator": {"bias_sigma_mps": sig, "bias_tau_s": tau, "yaw_rate_bias_sigma_dps": 0.2}}
    d, dy = [], []
    for seed in range(24):
        veh = _vehicle(cfg=cfg, seed=seed)
        tgt, y0 = _airborne(veh, 1.0)
        veh.bias[:] = 0.0                                            # start the clock with a zero bias
        p0, psi0 = veh.p[:2].copy(), veh.dpsi
        _run(veh, T, lambda v: (tgt, y0))
        d.append(veh.p[:2] - p0)
        dy.append(veh.dpsi - psi0)
    d = np.array(d)
    ratio = float(np.std(d) / math.sqrt(var_th))
    assert 0.65 < ratio < 1.5, (ratio, np.std(d), math.sqrt(var_th))
    ry = float(np.std(dy) / math.radians(0.2 * T))
    assert 0.5 < ry < 1.6, ry
    a = _vehicle(cfg=cfg, seed=3)
    b = _vehicle(cfg=cfg, seed=3)
    for v in (a, b):
        tgt, y0 = _airborne(v, 1.0)
        _run(v, 3.0, lambda vv: (tgt, y0))
    assert np.array_equal(a.p, b.p) and a.dpsi == b.dpsi
    z = _vehicle(seed=5)
    T0 = z.T_map_odom_true()
    tgt, y0 = _airborne(z, 1.0)
    _run(z, 3.0, lambda vv: (tgt + np.array([1.0, 0.5, 0.0]), y0 + 0.5))
    T1 = z.T_map_odom_true()
    assert np.abs(T1[:2, 3] - T0[:2, 3]).max() < 1e-6 and abs(yaw_of(T1) - yaw_of(T0)) < 1e-9
    # default drift: the odom frame is NOT the map frame (random origin + yaw at power-on)
    r = SimVehicle(_world(), body_pose((1.0, 4.0, 0.0), 0.0), seed=1)
    Tr = r.T_map_odom_true()
    assert np.linalg.norm(Tr[:2, 3] - [1.0, 4.0]) > 1e-3 or abs(yaw_of(Tr)) > 1e-3


def test_local_frame_reset_is_reported():
    veh = _vehicle(cfg={"estimator": {"reset_at_s": 6.0, "reset_delta": [0.4, -0.2, 0.0, 10.0]}})
    tgt, y0 = _airborne(veh, 1.0)
    assert veh.t < 6.0
    rc0 = veh.odometry().reset_counter
    _run(veh, 6.0 - veh.t - 0.02, lambda v: (tgt, y0))
    o0, p0 = veh.odometry().T_odom_body[:3, 3].copy(), veh.p.copy()
    _run(veh, 0.04, lambda v: (tgt, y0))
    o = veh.odometry()
    assert o.reset_counter == rc0 + 1
    lr = veh.status().info["last_reset"]
    assert lr["delta_xy"] == [0.4, -0.2] and abs(lr["delta_yaw_rad"] - math.radians(10)) < 1e-12
    assert np.linalg.norm(o.T_odom_body[:2, 3] - o0[:2] - [0.4, -0.2]) < 0.01      # the odom estimate jumped
    _run(veh, 6.0, lambda v: (tgt, y0))                                           # same odom setpoint ...
    expect = veh._R_e().T @ -np.array([0.4, -0.2])                                # (odom axes rotated by the yaw jump)
    assert np.linalg.norm(veh.p[:2] - p0[:2] - expect) < 0.03                     # ... so the TRUE vehicle moved


# --- world ----------------------------------------------------------------------------------------------------

def test_collision_detection_and_clearance():
    w = _world()
    r, below, above = 0.2, 0.0559, 0.06
    c = w.hull_clearance(np.array([0.5, 3.0, 1.0]), r, below, above)
    assert abs(c["lateral"] - 0.3) < 1e-9 and c["lateral_what"] == "wall_x0"
    c = w.hull_clearance(np.array([4.0, 3.6, 1.0]), r, below, above)   # 0.4 m north of the pillar face
    assert abs(c["lateral"] - 0.2) < 1e-9 and c["lateral_what"] == "pillar_d"
    c = w.hull_clearance(np.array([1.4, 1.0, 1.5]), r, below, above)   # above crate_a (top 1.0)
    assert c["below_what"] == "crate_a_top" and abs(c["below"] - (1.5 - below - 1.0)) < 1e-9
    c = w.hull_clearance(np.array([4.0, 3.3, 1.0]), r, below, above)
    assert c["lateral"] < 0                                            # penetrates the pillar
    veh = _vehicle(at=(3.0, 3.0))
    tgt, y0 = _airborne(veh, 1.0)
    _run(veh, 4.0, lambda v: (tgt + np.array([2.0, 0.0, 0.0]), y0))   # straight into the pillar
    assert veh.collisions and veh.crashed and "pillar_d" in veh.collisions[0]["what"]
    assert veh.min_lateral < 0


def test_world_texture_identity_and_fast_renderer():
    w = SimWorld(obstacles=[("box1", (4.4, 3.0, 0.0), (5.8, 4.4, 0.7)), ("late", (0, 0, 0), (0.6, 0.6, 2.0), False)])
    chk = check_texture_identity(w)
    assert chk["identical"], chk
    room = sr.make_room(0)
    cam = w.cam
    T = body_pose((1.2, 4.1, 1.1), 0.2, 0.05, -0.03) @ sr.T_BODY_CAM_SYNTH
    a, _ = sr.Renderer(room, cam).render(T, gain=0.85, noise_sigma=2.0, rng=np.random.default_rng(4))
    b = FastRenderer(room, cam).render(T, gain=0.85, noise_sigma=2.0, rng=np.random.default_rng(4))
    assert int(np.abs(a.astype(int) - b.astype(int)).max()) <= 2
    # the late obstacle: invisible and not solid until placed, then both
    view = body_pose((2.0, 1.9, 1.0), 0.0) @ sr.T_BODY_CAM_SYNTH
    before = w.render(view)
    assert w.terrain_z(3.3, 1.9) == 0.0
    w.place_obstacle("late", (3.0, 1.6, 0.0), stamp=12.0)
    after = w.render(view)
    assert np.abs(before.astype(int) - after.astype(int)).max() > 50
    assert abs(w.terrain_z(3.3, 1.9) - 2.0) < 1e-9 and w.obstacle("late").appear_at == 12.0


def test_metrics_matching_and_bars():
    truths = [{"name": "box", "lo": [4.0, 3.0, 0.0], "hi": [5.0, 4.0, 0.7], "appear_at": None}]
    assert metrics.match_obstacle({"lo": [3.8, 2.9, 0.0], "hi": [4.3, 3.5, 1.0], "first_seen": 1.0}, truths) == "box"
    assert metrics.match_obstacle({"lo": [7.3, 3.0, 0.0], "hi": [7.9, 3.5, 3.0], "first_seen": 1.0}, truths) is None
    rows = [{"airborne": 1, "x": x, "y": 3.5} for x in np.linspace(1, 7, 50)]
    assert metrics.geometric_pass(rows, truths[0], np.array([1.0, 3.5]), np.array([7.0, 3.5])) == "over"
    rows = [{"airborne": 1, "x": x, "y": 4.6} for x in np.linspace(1, 7, 50)]
    assert metrics.geometric_pass(rows, truths[0], np.array([1.0, 3.5]), np.array([7.0, 3.5])) == "around"
    m = {"outcome": "arrived", "collision": False, "collisions": [], "arrival": {"pos_err_m": 0.1, "yaw_err_deg": 3},
         "obstacles_created": 0, "detections": {"n": 0}, "emergencies": 0, "plan_shapes": ["direct"]}
    ok, checks = metrics.evaluate(m, scenarios.SCENARIOS["clear"].expect)
    assert ok, checks
    ok, checks = metrics.evaluate(m, scenarios.SCENARIOS["low_barrier"].expect)
    assert not ok and {c["check"] for c in checks if not c["ok"]} == {"plan_shape", "geometric_low_barrier"}
    ok, checks = metrics.evaluate(m, scenarios.SCENARIOS["low_box"].expect)     # over OR around: either passes
    assert ok, checks
    assert scenarios.ORDER == ["clear", "low_box", "low_barrier", "tall_box", "late_obstacle", "drift",
                               "round_trip", "hover_test", "wrong_takeoff"]


# --- the loop with a dummy (scripted, open-loop) mission ---------------------------------------------------------

class DummyMission:
    """Scripted: stream a hold 1.2 s, offboard, arm, climb to odom z 1.0, fly 1.5 m forward, land. No camera use
    except recording what arrives."""

    def __init__(self, vehicle, log):
        self.v, self.log = vehicle, log
        self.state = "PREFLIGHT"
        self.images = []
        self.last_setpoint = None
        self._t_state = 0.0

    def on_image(self, image, stamp):
        self.images.append((self.v.t, stamp, image.shape))

    def _to(self, s, now):
        self.log({"t": now, "event": "transition", "to": s})
        self.state, self._t_state = s, now

    def tick(self, now):
        o = self.v.odometry()
        st = self.v.status()
        if self.state == "PREFLIGHT":
            self.p0, self.y0 = o.T_odom_body[:3, 3].copy(), yaw_of(o.T_odom_body)
            self._to("ARMING", now)
        if self.state in ("DONE",):
            return
        sp = self.p0.copy()
        if self.state == "ARMING" and now - self._t_state > 1.2:
            self.v.request_offboard()
            self.v.request_arm()
            if st.armed:
                self._to("TAKEOFF", now)
        if self.state == "TAKEOFF":
            sp[2] = 1.0
            if abs(o.T_odom_body[2, 3] - 1.0) < 0.03:
                self._to("NAV", now)
        if self.state == "NAV":
            sp[2] = 1.0
            sp[0] += 1.5
            if np.linalg.norm(o.T_odom_body[:3, 3] - sp) < 0.03:
                self._to("LAND", now)
                self.v.request_land()
        if self.state == "LAND":
            if st.landed and not st.armed:
                self._to("DONE", now)
            return
        self.last_setpoint = Setpoint(now, sp, self.y0)
        self.v.send_setpoint(self.last_setpoint)

    def summary(self):
        return {"plan_shapes": [], "emergencies": 0}


def test_scripted_flight_through_the_loop():
    holder = {}

    def factory(world, vehicle, scenario, sim_cfg, log, recorder):
        holder["m"] = DummyMission(vehicle, log)
        return holder["m"], {}

    scen = scenarios.Scenario("scripted", "dummy open-loop flight", timeout_s=30.0,
                              expect={"outcome": "done", "no_collision": True})
    cfg = loop.load_sim_config(overrides={"loop": {"stop_at": "done", "post_terminal_s": 0.5},
                                          "vehicle": {"estimator": {"init_offset_sigma_m": 0.3, "init_yaw": "heading"}}})
    with tempfile.TemporaryDirectory() as d:
        m = loop.run_scenario(scen, 0, d, sim_cfg=cfg, mission_factory=factory, world=_world())
        assert m["passed"], m["checks"]
        assert m["outcome"] == "done" and not m["collision"] and m["ever_armed"]
        assert m["states_visited"] == ["ARMING", "TAKEOFF", "NAV", "LAND", "DONE"]
        run_dir = m["run_dir"]
        for f in ("events.jsonl", "trajectory.csv", "metrics.json"):
            assert os.path.getsize(os.path.join(run_dir, f)) > 0
        evs = [json.loads(line) for line in open(os.path.join(run_dir, "events.jsonl"), encoding="utf-8")]
        kinds = [e["event"] for e in evs]
        for k in ("transition", "offboard_engaged", "armed", "liftoff", "land_mode", "touchdown", "disarmed"):
            assert k in kinds, k
        assert json.load(open(os.path.join(run_dir, "metrics.json"), encoding="utf-8"))["outcome"] == "done"
    imgs = holder["m"].images
    lat = np.array([d - s for d, s, _ in imgs])
    assert np.allclose(lat, 0.1, atol=1e-6)                          # delivered 0.1 s late, stamped at capture
    assert np.allclose(np.diff([s for _, s, _ in imgs]), 0.1, atol=1e-6) and imgs[0][2] == (480, 640, 3)
    assert m["max_speed_xy_mps"] <= 1.5 and 0.85 < m["min_height_above_surface_m"] < 1.0
    assert m["min_clearance_m"] > 0.5                                # (1, 4) -> (2.5, 4) at 1 m: nothing near
    tm = m["timing"]
    assert tm["frames"] >= 10 * tm["sim_s"] - 1 and tm["render_per_sim_s"] < 2.0


if __name__ == "__main__":
    import time
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        t0 = time.time()
        fn()
        print(f"  [ok] {name} ({time.time() - t0:.1f} s)")
    print(f"{len(tests)}/{len(tests)} passed")

"""Mission state machine (navigation/mission.py) against FAKE components: the logic only, no images.

Run: py -3.9 tests/test_mission.py  (also under pytest). The closed-loop tests with the real components are in
tests/test_flightsim.py / tools/run_scout_sim.py.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from localization.geometry import make_T, rot_z  # noqa: E402
from navigation.mission import Mission  # noqa: E402
from navigation.types import GoalPose, LoomingOutput, Obstacle, OdomSample, PlanResult, VehicleStatus  # noqa: E402

PAD = (1.0, 3.0, 0.056)
B = np.array([8.0, 3.0, 1.2])


class FakeVehicle:
    """Kinematic vehicle: odom = map shifted by a fixed offset; follows setpoints at <= 1 m/s."""

    def __init__(self, start=PAD, yaw=0.0, offset=(5.0, -2.0, 0.0)):
        self.p = np.array(start, float)
        self.yaw = yaw
        self.off = np.array(offset, float)
        self.armed = self.offboard = False
        self.landed = True
        self.failsafe = False
        self.sp = None
        self.sp_t = None
        self.t = 0.0
        self.land_mode = False
        self.arm_requests = 0
        self.n_sp = 0
        self.ground = PAD[2]                   # where the gear rests (raised pad tests lift it)

    def step(self, dt):
        self.t += dt
        if self.land_mode:
            self.p[2] = max(self.ground, self.p[2] - 0.6 * dt)
            if self.p[2] <= self.ground + 1e-6:
                self.landed, self.armed = True, False
            return
        if self.armed and self.offboard and self.sp is not None:
            d = (self.sp.position + self.off * 0) - (self.p + self.off)
            n = np.linalg.norm(d)
            step = min(n, 1.0 * dt)
            if n > 1e-9:
                self.p += d / n * step
            dy = (self.sp.yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += max(-1.0 * dt, min(1.0 * dt, dy))
            self.landed = self.p[2] <= self.ground + 0.01

    def odometry(self):
        return OdomSample(self.t, make_T(rot_z(self.yaw), self.p + self.off), np.zeros(3))

    def status(self):
        return VehicleStatus(self.t, self.armed, self.offboard, self.landed, self.failsafe)

    def send_setpoint(self, sp):
        self.sp, self.sp_t = sp, self.t
        self.n_sp += 1

    def request_arm(self):
        self.arm_requests += 1
        if self.offboard:
            self.armed = True
            self.landed = False

    def request_offboard(self):
        if self.n_sp >= 10:
            self.offboard = True

    def request_land(self):
        self.land_mode, self.offboard = True, False

    def capabilities(self):
        return {"height_ref": "baro"}


class FakeLoc:
    """Fresh fix every frame equal to the true body pose (+ offset), unless `blind`."""

    def __init__(self, veh, offset=(0, 0, 0), yaw_off=0.0):
        self.veh, self.off, self.yaw_off, self.blind = veh, np.array(offset, float), yaw_off, False

    def process(self, image, stamp, T_prior=None):
        class O:
            pass
        o = O()
        o.fresh = not self.blind
        o.T_map_body = make_T(rot_z(self.veh.yaw + self.yaw_off), self.veh.p + self.off) if o.fresh else None
        o.n_inliers, o.method, o.loc_ok = 500, "fake", o.fresh
        return o


class FakeAligner:
    """Exact alignment = the vehicle's known offset; invalid when no fix for 2 s."""

    def __init__(self, veh):
        self.veh, self.init, self.last_fix, self.now = veh, False, None, 0.0

    def add_odom(self, s):
        self.now = s.stamp

    def set_terrain(self, fn):
        pass

    def set_height_ref(self, k):
        pass

    def init_known_pose(self, T, stamp):
        self.init, self.last_fix = True, stamp

    def add_fix(self, stamp, T, info=None):
        self.last_fix = stamp
        return {"accepted": True}

    def valid(self, now):
        return self.init and self.last_fix is not None and now - self.last_fix <= 2.0

    def T_map_odom(self):
        return make_T(np.eye(3), -self.veh.off) if self.init else None

    def pose_map(self, now=None):
        return make_T(rot_z(self.veh.yaw), self.veh.p.copy()), np.zeros(3)

    def body_pose_map_at(self, stamp):
        return make_T(rot_z(self.veh.yaw), self.veh.p.copy())

    def setpoint_to_odom(self, p, yaw):
        return np.asarray(p) + self.veh.off, yaw


class FakeMap:
    floor_z = 0.0

    def terrain_z(self, x, y, z):
        return 0.0


class FakePlanner:
    cfg = None

    def __init__(self):
        self.calls = 0

    def plan(self, start, goal, obstacles):
        self.calls += 1
        shape = "over" if obstacles else "direct"
        wps = np.array([start, goal.position]) if not obstacles else np.array(
            [start, [start[0], start[1], 1.8], [goal.position[0], goal.position[1], 1.8], goal.position])
        return PlanResult(True, wps, shape=shape, cost=1.0)


class FakeFollower:
    def set_path(self, wps, yaw):
        self.wps, self.yaw, self.i = np.asarray(wps, float), yaw, 1

    def step(self, now, T, v, obstacles):
        p = T[:3, 3]
        while self.i < len(self.wps) - 1 and np.linalg.norm(self.wps[self.i] - p) < 0.05:
            self.i += 1
        tgt = self.wps[self.i]
        d = tgt - p
        n = np.linalg.norm(d)
        sp = p + (d / n * min(n, 0.25) if n > 1e-9 else 0)
        done = self.i == len(self.wps) - 1 and n < 0.05
        return sp, self.yaw if n < 1.0 else math.atan2(d[1], d[0]) if np.hypot(d[0], d[1]) > 0.05 else self.yaw, \
            done, {}


class FakeSafety:
    emergency_hover_s = 1.0

    def filter(self, now, p, yaw, T, valid, emergency):
        return p, yaw, ("hold" if (emergency or not valid) else "pass"), ""


class FakeMemory:
    def __init__(self):
        self.obs, self.version = [], 0

    def update(self, obs, now):
        if obs:
            self.obs = list(obs)
            self.version += 1

    def active(self, now):
        return self.obs


class FakeLooming:
    def __init__(self):
        self.fire_at = None
        self.fired = False

    def process(self, image, stamp, T_odom_body, v, T_map_body=None, reset_counter=None):
        if self.fire_at is not None and stamp >= self.fire_at and not self.fired:
            self.fired = True
            ob = Obstacle(1, np.array([4.0, 3.0, 0.35]), np.array([0.7, 0.7, 0.35]), True, stamp, stamp,
                          "looming_map", 0.9)
            return LoomingOutput(stamp, True, 1.2, [ob])
        return LoomingOutput(stamp, False, float("inf"), [])


def goal_fn(static_map, b):
    return GoalPose(np.array([b[0] - 0.8, b[1], b[2]]), 0.0, np.asarray(b, float), np.array([-1.0, 0, 0]))


def make(cfg=None, loc_offset=(0, 0, 0), loc_yaw=0.0):
    veh = FakeVehicle()
    c = {"takeoff": {"position_map": list(PAD), "yaw_deg": 0.0}, "goal": {"point_b": B.tolist()},
         "auto_arm": True, "dwell_s": 1.0, "preflight_timeout_s": 3.0}
    c.update(cfg or {})
    m = Mission(c, veh, FakeLoc(veh, loc_offset, loc_yaw), FakeAligner(veh), FakeLooming(), FakeMap(),
                FakePlanner(), FakeFollower(), FakeSafety(), FakeMemory(), wall_goal_fn=goal_fn)
    return m, veh


def run(m, veh, seconds, image_every=2, until=None):
    t = 0.0
    for k in range(int(seconds * 20)):
        veh.step(0.05)
        t = veh.t
        if k % image_every == 0:
            m.on_image(None, t)
        m.tick(t)
        if until is not None and m.state == until:
            break
    return t


def states(m):
    return [e["to"] for e in m.events if e["event"] == "transition"]


def test_full_mission_to_wall_and_hold():
    m, veh = make()
    run(m, veh, 40, until="ARRIVED")
    run(m, veh, 2)
    s = states(m)
    assert s[:5] == ["ARMING", "TAKEOFF", "SETTLE", "NAV", "ARRIVED"], s
    assert np.linalg.norm(veh.p - (B - [0.8, 0, 0])) < 0.1, veh.p
    assert m.state == "ARRIVED"


def test_return_and_land():
    m, veh = make({"on_arrival": "return_land"})
    run(m, veh, 60, until="DONE")
    s = states(m)
    assert s[-3:] == ["RETURN", "LAND", "DONE"], s
    assert np.linalg.norm(veh.p[:2] - np.array(PAD[:2])) < 0.2, veh.p


def test_refuses_off_pad():
    m, veh = make(loc_offset=(0.5, 0.0, 0.0), loc_yaw=math.radians(20))
    run(m, veh, 5)
    assert m.state == "REFUSED", (m.state, m.reason)
    assert not veh.armed and veh.arm_requests == 0


def test_no_auto_arm_waits_for_pilot():
    m, veh = make({"auto_arm": False, "arm_timeout_s": 5.0})
    run(m, veh, 3)
    assert m.state == "ARMING" and not veh.armed
    veh.armed, veh.landed = True, False        # the pilot arms
    run(m, veh, 2)
    assert m.state in ("TAKEOFF", "SETTLE"), m.state


def test_hold_then_land_when_loc_lost():
    m, veh = make({"loc_lost_abort_s": 3.0})
    run(m, veh, 40, until="NAV")
    run(m, veh, 1)
    m.localizer.blind = True
    run(m, veh, 8)
    s = states(m)
    assert "HOLD" in s and s[-1] in ("LAND", "DONE"), s


def test_emergency_hover_then_replan_over():
    m, veh = make()
    run(m, veh, 40, until="NAV")
    m.looming.fire_at = veh.t + 0.5
    run(m, veh, 60, until="ARRIVED")
    assert m.state == "ARRIVED", (m.state, states(m))
    assert any(e["event"] == "emergency" for e in m.events)
    assert "over" in m.plan_shapes, m.plan_shapes


def test_abort_when_pilot_takes_over():
    m, veh = make()
    run(m, veh, 40, until="NAV")
    veh.offboard = False                     # the pilot flips the offboard switch
    run(m, veh, 1)
    assert m.state == "ABORTED", m.state
    n = veh.n_sp
    run(m, veh, 1)
    assert veh.n_sp == n, "setpoints must stop once aborted"


class StepAligner(FakeAligner):
    """Range height reference: the odom z is height above the surface PX4 refers to (`href`, map z)."""

    def __init__(self, veh):
        super().__init__(veh)
        self.href = 0.0

    def height_reference_z(self):
        return self.href

    def setpoint_to_odom(self, p, yaw):
        q = np.asarray(p, float) + self.veh.off
        return q - np.array([0.0, 0.0, self.href]), yaw


def test_feed_forward_ignores_a_terrain_step():
    """DN-63: crossing onto a 0.7 m box re-references PX4's z (the odom setpoint drops 0.7 m in 0.3 s); that is not
    motion, so the vertical feed-forward stays ~0 while the horizontal one still follows the moving setpoint."""
    m, veh = make()
    m.aligner = StepAligner(veh)
    m._hold_odom = None
    ffz, ffx, spz = [], [], []
    for k in range(40):
        m._now = 0.05 * k
        m.aligner.href = 0.7 * min(max((m._now - 0.5) / 0.3, 0.0), 1.0)    # the step, spread over 0.3 s
        m._send_map([2.0 + 0.5 * m._now, 4.0, 1.2], 0.0)                   # level, 0.5 m/s along x
        sp = veh.sp
        spz.append(sp.position[2])
        if sp.velocity_ff is not None:
            ffz.append(sp.velocity_ff[2])
            ffx.append(sp.velocity_ff[0])
    assert spz[0] - spz[-1] > 0.69, "the odom setpoint itself must still follow the new height reference"
    assert max(abs(v) for v in ffz) < 1e-9, max(abs(v) for v in ffz)
    assert abs(ffx[-1] - 0.5) < 0.01, ffx[-1]



class PadMap(FakeMap):
    """A raised pad box (top `h`) around the pad, floor elsewhere."""

    def __init__(self, h):
        self.h = h

    def terrain_z(self, x, y, z):
        return self.h if abs(x - PAD[0]) <= 0.25 and abs(y - PAD[1]) <= 0.25 and z >= self.h else 0.0


def make_pad(h, pad_z=None, cfg=None):
    m, veh = make(cfg)
    m.static_map = PadMap(h)
    z = h + 0.056 if pad_z is None else pad_z
    veh.p[2] = veh.ground = z
    m.T_map_pad[2, 3] = z
    return m, veh


def test_raised_pad_climbs_relative_to_pad_top():
    m, veh = make_pad(0.8)                         # a tall pad: floor + 1.0 would be only 0.2 m above it
    run(m, veh, 40, until="SETTLE")
    assert m.state == "SETTLE", (m.state, m.reason)
    assert abs(veh.p[2] - (0.8 + 0.5)) < 0.1, veh.p      # pad top + takeoff_min_climb_m
    m2, veh2 = make_pad(0.4)                       # a low pad: floor + takeoff_alt_m (1.0) wins
    run(m2, veh2, 40, until="SETTLE")
    assert abs(veh2.p[2] - 1.0) < 0.1, veh2.p


def test_refuses_pad_pose_not_on_the_map():
    m, veh = make_pad(0.4, pad_z=1.2)              # configured pad pose floats 0.8 m above the pad top
    run(m, veh, 3)
    assert m.state == "REFUSED" and "pad" in m.reason, (m.state, m.reason)
    assert not veh.armed


def test_return_lands_on_raised_pad():
    m, veh = make_pad(0.4, cfg={"on_arrival": "return_land"})
    run(m, veh, 80, until="DONE")
    assert states(m)[-3:] == ["RETURN", "LAND", "DONE"], states(m)



def _hover_mission(ceiling=None, pad_h=0.0):
    m, veh = make_pad(pad_h, cfg={"mode": "hover_test", "goal": {"point_b": None}})
    if ceiling is not None:
        m.static_map.ceiling_z = ceiling
    return m, veh


def test_hover_test_rises_2m_hovers_10s_lands_same_spot():
    m, veh = _hover_mission(pad_h=0.4)
    z0 = veh.p[2]
    zs, xy, t_hover = [], [], []

    def watch():
        if m.state == "HOVER":
            zs.append(veh.p[2]), xy.append(veh.p[:2].copy()), t_hover.append(veh.t)
    t = 0.0
    for k in range(int(60 * 20)):
        veh.step(0.05)
        if k % 2 == 0:
            m.on_image(None, veh.t)
        m.tick(veh.t)
        watch()
        if m.state == "DONE":
            break
    assert states(m) == ["ARMING", "TAKEOFF", "HOVER", "LAND", "DONE"], states(m)
    assert m.goal is None                                   # no wall goal needed
    assert abs(np.mean(zs) - (z0 + 2.0)) < 0.05, (np.mean(zs), z0)
    assert t_hover[-1] - t_hover[0] >= 10.0 - 0.06
    assert max(np.linalg.norm(p - np.array(PAD[:2])) for p in xy) < 0.05
    assert np.linalg.norm(veh.p[:2] - np.array(PAD[:2])) < 0.05 and abs(veh.p[2] - z0) < 1e-6   # back on the pad


def test_hover_test_capped_below_ceiling():
    m, veh = _hover_mission(ceiling=2.2)                    # floor pad: 0.056 + 2.0 would come within 0.5 m
    run(m, veh, 30, until="HOVER")
    run(m, veh, 2)
    assert m.state == "HOVER"
    assert abs(veh.p[2] - (2.2 - 0.5)) < 0.05, veh.p
    assert any(e["event"] == "hover_height_capped" for e in m.events)


def test_unknown_mode_rejected():
    try:
        make({"mode": "loop_the_loop"})
    except ValueError as e:
        assert "mode" in str(e)
    else:
        raise AssertionError("an unknown mode must be refused")



class WallMap(FakeMap):
    """A mapped wall at x <= 0.1 (occupied), floor elsewhere."""
    ceiling_z = 3.0

    def occupied(self, P):
        P = np.atleast_2d(P)
        return P[:, 0] <= 0.1


def test_raw_emergency_box_on_a_mapped_wall_is_not_remembered():
    m, veh = make()
    m.static_map = WallMap()
    run(m, veh, 40, until="NAV")
    wall_box = Obstacle(5, np.array([0.33, 4.36, 1.5]), np.array([0.33, 0.28, 1.52]), False, veh.t, veh.t,
                        "looming_raw", 0.8)
    real_box = Obstacle(6, np.array([4.0, 3.0, 1.0]), np.array([0.3, 0.3, 1.0]), False, veh.t, veh.t,
                        "looming_raw", 0.8)
    m.looming.process = lambda *a, **k: LoomingOutput(veh.t, True, 1.2, [wall_box, real_box])
    m.on_image(None, veh.t)
    assert [o.id for o in m.memory.active(veh.t)] == [6]
    assert any(e["event"] == "obstacle_explained_by_map" for e in m.events)
    assert any(e["event"] == "emergency" for e in m.events)      # the hover itself is not suppressed


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    print(f"{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)

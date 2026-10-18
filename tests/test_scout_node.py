"""The `scout_mission` node's ROS-free timing logic (ros2_ws/src/nx_scout_nav/nx_scout_nav/node_logic.py).

Both bugs came from the node's first PX4 SITL runs (docs/SCOUT_NAV.md §10, 2026-10-18): the offboard setpoint stream
stopped for 1.1 s (a camera frame held the mission lock; later a planning tick ran long), past PX4's COM_OF_LOSS_T,
and PX4 went to RTL; and frames were stamped at callback time, up to a processing time late.

Run: py -3.9 tests/test_scout_node.py  (also under pytest)
"""
from __future__ import annotations

import ast
import os
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "nx_scout_nav"))

from navigation.types import Setpoint  # noqa: E402
from nx_scout_nav.node_logic import (DurationStats, StreamKeeper, frame_stamp, mission_overrides,  # noqa: E402
                                     mission_streaming)

NODE = os.path.join(ROOT, "ros2_ws", "src", "nx_scout_nav", "nx_scout_nav", "mission_node.py")


class FakeMission:
    def __init__(self, state="NAV"):
        self.state = state
        self._stream = True


class FakeVehicle:
    def __init__(self):
        self.sent = []

    def send_setpoint(self, sp):
        self.sent.append((time.monotonic(), sp))

    def odometry(self):
        return "odom"

    def capabilities(self):
        return {"height_ref": "baro"}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


SP = Setpoint(1.0, np.array([1.0, 2.0, 1.2]), 0.5, np.array([0.4, 0.0, 0.0]))


def test_frame_stamp_uses_reception_time():
    now = 1000.0
    s, src = frame_stamp(int(999.4 * 1e9), now, 0.12)          # received 0.6 s before its callback ran
    assert src == "received" and abs(s - (999.4 - 0.12)) < 1e-6
    # old behaviour (callback time) would have been 0.6 s late: the SITL residuals
    assert abs((now - 0.12) - s - 0.6) < 1e-6


def test_frame_stamp_falls_back():
    now = 1000.0
    for rx in (None, 0, int(990.0 * 1e9), int(1000.5 * 1e9)):  # missing, 0, another clock / stale, future
        s, src = frame_stamp(rx, now, 0.12)
        assert src == "callback" and abs(s - 999.88) < 1e-9, rx


def test_keeper_passes_the_vehicle_through():
    v = FakeVehicle()
    k = StreamKeeper(v)
    assert k.odometry() == "odom" and k.capabilities() == {"height_ref": "baro"}
    k.send_setpoint(SP)
    assert v.sent[-1][1] is SP and k.n_sent == 1


def test_keeper_resends_after_a_gap_only_while_streaming():
    v, c, m = FakeVehicle(), Clock(), FakeMission()
    k = StreamKeeper(v, lambda: mission_streaming(m), max_gap_s=0.1, clock=c)
    assert not k.check(0.0)                                   # nothing sent yet: nothing to keep alive
    k.send_setpoint(SP)
    c.t = 0.05
    assert not k.check(5.0)                                   # fresh
    c.t = 0.15
    assert k.check(7.0)
    sp = v.sent[-1][1]
    assert sp.stamp == 7.0 and sp.velocity_ff is None         # no feed-forward on a re-sent hold
    assert np.allclose(sp.position, SP.position) and sp.yaw == SP.yaw and sp.position is not SP.position
    c.t = 0.2
    assert not k.check(7.1)                                   # the keepalive itself counts as a send
    for state in ("PREFLIGHT", "LAND", "DONE", "REFUSED", "ABORTED"):
        m.state = state
        c.t += 1.0
        assert not k.check(9.0), state
    m.state, m._stream = "NAV", False
    assert not k.check(9.0)
    assert k.n_keepalive == 1


def test_stream_never_gaps_while_the_mission_stalls_1s():
    """The mission sends at 20 Hz, then stalls 1.2 s (a frame holding the lock, or a long planning tick: the SITL
    worst cases). With the watchdog thread running, what the vehicle receives never gaps long; well inside PX4's
    COM_OF_LOSS_T = 1 s (< 0.3 s: Windows timer granularity). Without it the gap is the stall."""
    v, m = FakeVehicle(), FakeMission()
    k = StreamKeeper(v, lambda: mission_streaming(m), max_gap_s=0.1)
    k.start(time.monotonic, period_s=0.05)
    try:
        for _ in range(5):
            k.send_setpoint(SP)
            time.sleep(0.05)
        time.sleep(1.2)                                       # the stall
        for _ in range(3):
            k.send_setpoint(SP)
            time.sleep(0.05)
    finally:
        k.stop()
    ts = np.array([t for t, _ in v.sent])
    assert k.n_keepalive >= 6, k.n_keepalive
    assert float(np.max(np.diff(ts))) < 0.3, float(np.max(np.diff(ts)))
    assert k.gap_summary()["max_gap_ms"] < 300.0


def test_mission_overrides_wall_goal_unchanged():
    over = mission_overrides(on_arrival="return_land", auto_arm=True, takeoff=[1.0, 4.0, 0.0559, 0.0],
                             goal=[8.0, 3.6, 1.2])
    assert over == {"on_arrival": "return_land", "auto_arm": True, "mode": "wall_goal",
                    "takeoff": {"position_map": [1.0, 4.0, 0.0559], "yaw_deg": 0.0},
                    "goal": {"point_b": [8.0, 3.6, 1.2]}}
    assert "goal" not in mission_overrides(goal=[0.0]) and "takeoff" not in mission_overrides(takeoff=[0.0])


def test_mission_overrides_hover_test():
    over = mission_overrides(takeoff=[1.0, 4.0, 0.0559, 0.0], mode="hover_test", hover_climb_m=1.5, hover_s=8.0)
    assert over["mode"] == "hover_test" and "goal" not in over               # no goal needed
    assert over["hover_test"] == {"climb_m": 1.5, "hover_s": 8.0}
    assert "hover_test" not in mission_overrides(mode="hover_test")           # unset -> scout_mission.yaml's values
    assert mission_overrides(mode="hover_test", hover_s=3.0)["hover_test"] == {"hover_s": 3.0}
    try:
        mission_overrides(mode="hover")
    except ValueError as e:
        assert "hover_test" in str(e)
    else:
        raise AssertionError("an unknown mode must be refused")


def test_duration_stats():
    s = DurationStats()
    assert s.summary() == {"n": 0}
    for v in (0.1, 0.2, 0.3):
        s.add(v)
    out = s.summary()
    assert out["n"] == 3 and out["max_ms"] == 300.0 and out["p50_ms"] == 200.0
    assert s.summary() == {"n": 0}                                      # reset


def test_node_wiring():
    """rclpy passes MessageInfo only when the callback REQUIRES two arguments (subscription.py binds one argument
    first): `info=None` would silently stamp every frame at callback time again. And the mission must get the
    keeper, never the raw vehicle (or the stream depends on the mission's timing again)."""
    src = open(NODE, encoding="utf-8").read()
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in ("_on_image", "_on_compressed"):
        a = fns[name].args
        assert [x.arg for x in a.args] == ["self", "msg", "info"] and not a.defaults, name
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "build_flight_stack"]
    assert len(calls) == 1 and ast.unparse(calls[0].args[0]) == "self.keeper"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} tests passed")

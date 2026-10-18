"""The `scout_mission` node's timing logic, with NO ROS imports (unit-tested in plain Python: tests/test_scout_node.py).

Both pieces came out of the node's first PX4 SITL runs (docs/SCOUT_NAV.md §10, 2026-10-18):
- `frame_stamp`: a frame is stamped from when the laptop RECEIVED it (the DDS reception time), not from when its
  callback finally ran. The image callback processes one frame at a time (localization + looming: 0.4-0.8 s in WSL
  next to PX4 and the camera), and the newest frame waits in the reader meanwhile; stamped at callback time it was
  up to a whole processing time late, and the map<->odom alignment rejected the fix (xy residual 0.18-0.35 m at
  0.5 m/s).
- `StreamKeeper`: the offboard setpoint stream never depends on the mission's own timing. It wraps the vehicle; a
  watchdog thread re-sends the last setpoint (position + yaw, no velocity feed-forward) whenever nothing was sent for
  `max_gap_s` while the mission is streaming. In SITL the stream stopped for 1.1 s twice: once while a camera frame
  held the mission lock, once while a tick planned; both past PX4's COM_OF_LOSS_T (1 s, also on the real Scout), and
  PX4 switched to RTL and the mission aborted on the failsafe. A keepalive inside the tick thread (the first fix)
  covered the first case only.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from navigation.mission import AIRBORNE
from navigation.types import Setpoint

STREAM_STATES = ("ARMING",) + tuple(AIRBORNE)       # the states in which Mission streams setpoints


def frame_stamp(received_ns, now_s: float, latency_s: float, max_skew_s: float = 2.0):
    """-> (stamp, source). `received_ns`: the DDS reception time (rclpy MessageInfo "received_timestamp", system
    clock ns; 0 / None when the RMW does not fill it). One clock for odometry and images (DN-05): the laptop's
    receive time minus the configured latency. A reception time that is missing, in the future, or more than
    `max_skew_s` old (a different clock) falls back to the callback time."""
    if received_ns:
        r = float(received_ns) * 1e-9
        if now_s - max_skew_s <= r <= now_s + 0.05:
            return r - float(latency_s), "received"
    return float(now_s) - float(latency_s), "callback"


MISSION_MODES = ("wall_goal", "hover_test")


def mission_overrides(on_arrival="hold", auto_arm=False, takeoff=(0.0,), goal=(0.0,), mode="wall_goal",
                      hover_climb_m=-1.0, hover_s=-1.0) -> dict:
    """The node's parameters -> overrides merged over config/scout_mission.yaml (navigation.build).
    takeoff [x, y, z, yaw_deg] / goal [x, y, z]: any other length = not set (the config's value). mode: wall_goal (fly to
    the wall point B) or hover_test (climb hover_climb_m above the pad, hold hover_s, land on the pad; no goal needed).
    hover_climb_m / hover_s <= 0 = not set."""
    mode = str(mode)
    if mode not in MISSION_MODES:
        raise ValueError(f"parameter 'mode' must be one of {MISSION_MODES}, got {mode!r}")
    over = {"on_arrival": str(on_arrival), "auto_arm": bool(auto_arm), "mode": mode}
    tk = list(takeoff)
    if len(tk) == 4:
        over["takeoff"] = {"position_map": [float(v) for v in tk[:3]], "yaw_deg": float(tk[3])}
    g = list(goal)
    if len(g) == 3:
        over["goal"] = {"point_b": [float(v) for v in g]}
    ht = {}
    if float(hover_climb_m) > 0:
        ht["climb_m"] = float(hover_climb_m)
    if float(hover_s) > 0:
        ht["hover_s"] = float(hover_s)
    if ht:
        over["hover_test"] = ht
    return over


def mission_streaming(mission) -> bool:
    return mission.state in STREAM_STATES and bool(getattr(mission, "_stream", True))


class StreamKeeper:
    """A VehicleInterface wrapper (navigation/vehicle.py): every call passes through; `send_setpoint` is recorded,
    and `check()` (the watchdog, `start()` runs it every `period_s` in its own thread) re-sends the last setpoint
    when the last send is older than `max_gap_s` and `is_streaming()` is true."""

    def __init__(self, vehicle, is_streaming=None, max_gap_s: float = 0.1, clock=time.monotonic):
        self.vehicle = vehicle
        self.is_streaming = is_streaming or (lambda: False)
        self.max_gap_s = float(max_gap_s)
        self.clock = clock
        self._lock = threading.Lock()
        self._last = None                      # (clock time, Setpoint)
        self.n_sent = 0
        self.n_keepalive = 0
        self.max_gap_seen_s = 0.0
        self._stop = threading.Event()
        self._thread = None

    def __getattr__(self, name):               # odometry, status, request_*, capabilities
        return getattr(self.vehicle, name)

    def _send(self, sp):
        t = self.clock()
        if self._last is not None:
            self.max_gap_seen_s = max(self.max_gap_seen_s, t - self._last[0])
        self.vehicle.send_setpoint(sp)
        self._last = (t, sp)

    def send_setpoint(self, sp) -> None:
        with self._lock:
            self._send(sp)
            self.n_sent += 1

    def check(self, stamp: float) -> bool:
        with self._lock:
            if self._last is None or self.clock() - self._last[0] < self.max_gap_s:
                return False
            if not self.is_streaming():
                return False
            sp = self._last[1]
            self._send(Setpoint(float(stamp), np.asarray(sp.position, float).copy(), float(sp.yaw), None))
            self.n_keepalive += 1
            return True

    def gap_summary(self, reset: bool = True) -> dict:
        with self._lock:
            out = {"sent": self.n_sent, "keepalives": self.n_keepalive, "max_gap_ms": round(self.max_gap_seen_s * 1e3, 1)}
            if reset:
                self.max_gap_seen_s = 0.0
        return out

    def start(self, stamp_fn, period_s: float = 0.05):
        def loop():
            while not self._stop.wait(period_s):
                try:
                    self.check(stamp_fn())
                except Exception:  # noqa: BLE001 - the watchdog must outlive one bad send
                    pass
        self._thread = threading.Thread(target=loop, name="setpoint_keeper", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class DurationStats:
    """Rolling durations (s) -> {"n", "p50_ms", "p95_ms", "max_ms"} since the last `summary(reset=True)`."""

    def __init__(self):
        self._v = []

    def add(self, s: float) -> None:
        self._v.append(float(s))

    def summary(self, reset: bool = True) -> dict:
        v = np.asarray(self._v, float)
        if reset:
            self._v = []
        if len(v) == 0:
            return {"n": 0}
        return {"n": int(len(v)), "p50_ms": round(float(np.percentile(v, 50)) * 1e3, 1),
                "p95_ms": round(float(np.percentile(v, 95)) * 1e3, 1), "max_ms": round(float(v.max()) * 1e3, 1)}

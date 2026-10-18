#!/usr/bin/env python3
"""SITL check of `Px4ScoutVehicle` against real PX4 firmware (SIH, no Gazebo). SITL ONLY: this script ARMS.

Never point it at the real Scout (DN-07: an agent never arms the real aircraft). It refuses to start unless
`--confirm-sitl` is given AND `NOXEN_SCOUT_SITL` names the PX4 SITL process the runner booted.

Run it through `run_sitl_check.sh` (takes the laptop's simulator lock, boots PX4 SIH + the XRCE agent, tears down, then runs
`sitl_ulog_check.py` on PX4's own flight log):

    wsl.exe -d Ubuntu-24.04 bash /mnt/c/Users/Ron/Desktop/noxen-dh26/drivers/px4_scout/bench/run_sitl_check.sh flight

Modes
  flight    stream setpoints -> offboard -> arm -> climb to 1.0 m (odom) -> hold 5 s -> turn to face +x odom ->
            move +1 m in x odom -> hold 5 s -> land; records every 50 ms and checks the frame conventions against
            PX4's own NED odometry, command acceptance (vehicle_command_ack), the status flags and the
            distance_sensor -> dist_bottom_m path.
  fallback  PX4 with its distance sensor silenced (runner sets SIH_DISTSNSR_OVR): the adapter must come up
            without range data and report dist_bottom_m = None.

Checks are GATING (the adapter's own behaviour: frames, flags, commands) or REPORTED (`gating: false`: numbers that
measure SIH's estimator / controller, whose EKF wanders 0.3-0.5 m on default SITL sensors per the ulog ground truth).
Tolerances are this bench's, not flight thresholds.

Output: <out-dir>/result.json (checks, acks, phases, sent setpoints) and <out-dir>/samples.jsonl. Exit 0 all gating
checks pass / 1 a gating check failed / 2 could not evaluate (no PX4, timeouts before flight).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

from drivers.px4_scout import frames as F  # noqa: E402
from navigation.types import Setpoint  # noqa: E402

RATE_HZ = 20.0
CLIMB_Z = 1.0              # odom, m (task)
MOVE_DX = 1.0              # odom +x, m (task)
V_CLIMB = 0.4              # m/s setpoint ramps (the real mission caps at 0.5 m/s)
V_MOVE = 0.5
YAW_RATE = math.radians(30.0)
VEHICLE_CMD_RESULT_ACCEPTED = 0
TOL_HOLD_Z = 0.10          # m   (reported)
TOL_HOLD_XY = 0.15         # m   (reported)
TOL_YAW = math.radians(5.0)
TOL_RANGE = 0.10           # m   (reported: dist_bottom vs EKF height)


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def yaw_of(T):
    return math.atan2(T[1, 0], T[0, 0])


class Raw:
    """PX4's own messages, recorded beside the adapter (the reference the adapter is checked against)."""

    def __init__(self, node, find_topic):
        from px4_msgs.msg import DistanceSensor, VehicleCommandAck, VehicleOdometry
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=20)
        self.lock = threading.Lock()
        self.odom = None
        self.range = None                       # latest raw DistanceSensor
        self.odom_rx = []                       # receive times (s), for the rate / jitter
        self.acks = []
        self.t0 = time.monotonic()
        node.create_subscription(VehicleOdometry, find_topic(node, "/fmu/out/vehicle_odometry"), self._on_odom, be)
        node.create_subscription(VehicleCommandAck, find_topic(node, "/fmu/out/vehicle_command_ack"), self._on_ack, be)
        try:
            node.create_subscription(DistanceSensor, find_topic(node, "/fmu/out/distance_sensor", tries=2),
                                     self._on_range, be)
        except RuntimeError:
            pass

    def _on_odom(self, m):
        with self.lock:
            self.odom = m
            if len(self.odom_rx) < 20000:
                self.odom_rx.append(time.monotonic())

    def _on_range(self, m):
        with self.lock:
            self.range = m

    def _on_ack(self, m):
        with self.lock:
            self.acks.append({"t": round(time.monotonic() - self.t0, 3), "command": int(m.command),
                              "result": int(m.result), "target_system": int(m.target_system)})


class Check:
    def __init__(self):
        self.items = []

    def add(self, name, ok, value=None, expect=None, note="", gating=True):
        self.items.append({"name": name, "pass": bool(ok), "gating": bool(gating), "value": value, "expect": expect,
                           "note": note})
        tag = ("PASS" if ok else "FAIL") if gating else ("ok" if ok else "off") + " (reported)"
        print(f"[check] {tag} {name}: {value} (expect {expect}) {note}", flush=True)
        return ok

    @property
    def all_pass(self):
        return all(c["pass"] for c in self.items if c["gating"])


class Bench:
    def __init__(self, node, veh, raw, out_dir):
        self.node, self.veh, self.raw = node, veh, raw
        self.out = open(os.path.join(out_dir, "samples.jsonl"), "w")
        self.t0 = time.monotonic()
        self.phase = "init"
        self.phases = []
        self.sp = None                          # Setpoint being streamed (None: nothing streamed)
        self.sent = None                        # the hold setpoints (odom x, y, z, yaw) -> sitl_ulog_check.py
        self.samples = []
        self.ck = Check()
        self.pfc_ok = lambda: True

    def t(self):
        return time.monotonic() - self.t0

    def set_phase(self, name):
        self.phase = name
        self.phases.append({"phase": name, "t": round(self.t(), 3)})
        print(f"[phase] {self.t():7.2f} s  {name}", flush=True)

    def tick(self):
        """One 20 Hz step: stream the current setpoint (if any), record one sample."""
        if self.sp is not None:
            self.sp.stamp = time.time()
            self.veh.send_setpoint(self.sp)
        od, st = self.veh.odometry(), self.veh.status()
        with self.raw.lock:
            r, rr = self.raw.odom, self.raw.range
        s = {"t": round(self.t(), 3), "phase": self.phase,
             "armed": st.armed, "offboard": st.offboard, "landed": st.landed, "failsafe": st.failsafe,
             "nav_state": st.info.get("nav_state"), "arming_state": st.info.get("arming_state"),
             "status_received": bool(st.info.get("status_received")),
             "stale_status": bool(st.info.get("stale_status", False))}
        if self.sp is not None:
            s["sp"] = [round(float(v), 4) for v in self.sp.position] + [round(self.sp.yaw, 4)]
        if od is not None:
            T = od.T_odom_body
            s.update(p=[round(float(v), 4) for v in T[:3, 3]], yaw=round(yaw_of(T), 5),
                     zaxis_z=round(float(T[2, 2]), 5), xaxis=[round(float(v), 4) for v in T[:3, 0]],
                     v=[round(float(v), 4) for v in od.v_odom], reset=od.reset_counter,
                     dist_bottom=None if od.dist_bottom_m is None else round(od.dist_bottom_m, 4),
                     stamp=od.stamp)
        if r is not None:
            q = [float(v) for v in r.q]
            s.update(ned_p=[round(float(v), 4) for v in r.position], ned_q=[round(v, 6) for v in q],
                     ned_v=[round(float(v), 4) for v in r.velocity], ned_yaw=round(math.atan2(
                         2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)), 5))
            # same-instant frame check: convert PX4's raw message with frames.py and compare with what the adapter
            # stored (catches a field mix-up in the adapter, e.g. q order, a velocity / position swap)
            if od is not None:
                T_ref = F.ned_to_odom_pose(r.position, r.q)
                s["frame_err_p"] = float(np.max(np.abs(T_ref[:3, 3] - od.T_odom_body[:3, 3])))
                s["frame_err_R"] = float(np.max(np.abs(T_ref[:3, :3] - od.T_odom_body[:3, :3])))
        if rr is not None:
            s["rng_raw"] = round(float(rr.current_distance), 4)
            if od is not None and od.dist_bottom_m is not None:
                # the adapter's dist_bottom vs the raw reading, tilt-corrected (range_below_body_m = 0 here)
                s["rng_err"] = abs(od.dist_bottom_m - float(rr.current_distance) * float(od.T_odom_body[2, 2]))
        self.samples.append(s)
        self.out.write(json.dumps(s) + "\n")
        return od, st

    def run_for(self, secs, update=None):
        end = time.monotonic() + secs
        while time.monotonic() < end:
            t_next = time.monotonic() + 1.0 / RATE_HZ
            if update is not None:
                update()
            self.tick()
            time.sleep(max(0.0, t_next - time.monotonic()))

    def wait_until(self, cond, timeout, retry=None, retry_s=1.0, update=None):
        """Tick until cond(odom, status) is true; call retry() every retry_s. -> seconds waited or None."""
        t_start = time.monotonic()
        t_retry = t_start
        if retry is not None:
            retry()
        while time.monotonic() - t_start < timeout:
            t_next = time.monotonic() + 1.0 / RATE_HZ
            if update is not None:
                update()
            od, st = self.tick()
            if cond(od, st):
                return round(time.monotonic() - t_start, 3)
            if retry is not None and time.monotonic() - t_retry >= retry_s:
                retry()
                t_retry = time.monotonic()
            time.sleep(max(0.0, t_next - time.monotonic()))
        return None

    def window(self, phase, key):
        return [s for s in self.samples if s["phase"] == phase and key in s]


def ramp(start, end, speed, t_start):
    """Position on a straight ramp from start to end at `speed`, and the feed-forward velocity."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    d = end - start
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return end.copy(), np.zeros(3)
    s = speed * (time.monotonic() - t_start) / L
    if s >= 1.0:
        return end.copy(), np.zeros(3)
    return start + s * d, d / L * speed


def flight(b: Bench, ck: Check, pre_flight_timeout):
    veh = b.veh
    b.set_phase("wait_ready")
    w = b.wait_until(lambda od, st: od is not None and st.info.get("status_received") and b.pfc_ok(),
                     pre_flight_timeout)
    if w is None:
        return 2, "PX4 never reported odometry + status + pre_flight_checks_pass"
    od, st = b.tick()
    ck.add("pre_arm_status", (not st.armed) and st.landed and (not st.offboard) and (not st.failsafe),
           {"armed": st.armed, "landed": st.landed, "offboard": st.offboard, "failsafe": st.failsafe},
           "disarmed, landed, not offboard, no failsafe")
    ck.add("ground_z_near_zero", abs(od.T_odom_body[2, 3]) < 0.3, round(float(od.T_odom_body[2, 3]), 3),
           "|z_odom| < 0.3 m on the ground", "odom z is floor-referenced at the EKF origin")
    p0 = od.T_odom_body[:3, 3].copy()
    yaw0 = yaw_of(od.T_odom_body)
    reset0 = od.reset_counter
    ground = p0.copy()
    x0, y0 = float(p0[0]), float(p0[1])
    b.sent = {"hold_1m": [x0, y0, CLIMB_Z, yaw0], "turned": [x0, y0, CLIMB_Z, 0.0],
              "goal": [x0 + MOVE_DX, y0, CLIMB_Z, 0.0]}

    b.set_phase("stream")
    b.sp = Setpoint(time.time(), ground.copy(), yaw0)
    b.run_for(1.5)

    b.set_phase("offboard")
    w = b.wait_until(lambda od, st: st.offboard, 6.0, retry=veh.request_offboard)
    if not ck.add("offboard_accepted", w is not None, w, "status.offboard within 6 s"):
        return 1, "offboard refused"

    b.set_phase("arm")
    w = b.wait_until(lambda od, st: st.armed, 6.0, retry=veh.request_arm)
    if not ck.add("arm_accepted", w is not None, w, "status.armed within 6 s"):
        return 1, "arm refused"

    b.set_phase("climb")
    top = np.array([x0, y0, CLIMB_Z])
    t_c = time.monotonic()

    def upd_climb():
        b.sp.position, b.sp.velocity_ff = ramp(ground, top, V_CLIMB, t_c)
    w = b.wait_until(lambda od, st: od is not None and abs(od.T_odom_body[2, 3] - CLIMB_Z) < 0.05
                     and np.array_equal(b.sp.position, top), 20.0, update=upd_climb)
    ck.add("climb_reached", w is not None, w, "z_odom within 5 cm of 1.0 m within 20 s", gating=False)
    b.sp.position, b.sp.velocity_ff = top.copy(), None

    b.set_phase("hold_1m")
    b.run_for(5.0)
    hs = b.window("hold_1m", "p")
    z = np.array([s["p"][2] for s in hs])
    ned_z = np.array([s["ned_p"][2] for s in hs if "ned_p" in s])
    xy = np.array([s["p"][:2] for s in hs])
    ck.add("odom_z_up_vs_ned_down", abs(ned_z.mean() + z.mean()) < 0.01 and z.mean() > 0.5,
           {"odom_z": round(float(z.mean()), 3), "ned_z": round(float(ned_z.mean()), 3)},
           "climbed: odom z > 0.5, PX4 NED z = -odom z")
    ck.add("hold_z_matches_setpoint", abs(z.mean() - CLIMB_Z) < TOL_HOLD_Z, round(float(z.mean()), 3),
           f"1.0 ± {TOL_HOLD_Z} m", gating=False)
    dxy = float(np.max(np.linalg.norm(xy - p0[:2], axis=1)))
    ck.add("hold_xy_held", dxy < TOL_HOLD_XY, round(dxy, 3), f"< {TOL_HOLD_XY} m from takeoff xy", gating=False)
    ck.add("airborne_flags", all(s["armed"] and s["offboard"] and not s["landed"] and not s["failsafe"]
                                 for s in hs), None, "armed, offboard, not landed, no failsafe over the hold")
    db = [s for s in hs if s.get("dist_bottom") is not None]
    re_ = [s["rng_err"] for s in hs if "rng_err" in s]
    ck.add("dist_bottom_valid_in_hold", len(db) >= 0.9 * len(hs), f"{len(db)}/{len(hs)}",
           ">= 90 % of hold samples carry dist_bottom_m")
    ck.add("dist_bottom_is_tilt_corrected_range", bool(re_) and float(np.percentile(re_, 90)) < 0.02,
           {"p90_err_m": round(float(np.percentile(re_, 90)), 4) if re_ else None,
            "max_err_m": round(max(re_), 4) if re_ else None},
           "p90 |dist_bottom - raw range * cos(tilt)| < 2 cm (residual: the raw range sample is newer)")
    if db:
        err = max(abs(s["dist_bottom"] - s["p"][2]) for s in db)
        ck.add("dist_bottom_vs_ekf_height", err < TOL_RANGE, {"max_err_m": round(err, 3)},
               f"|dist_bottom - z_odom| < {TOL_RANGE} m over flat SIH ground", gating=False)

    b.set_phase("turn_to_plus_x")
    t_y = time.monotonic()
    dyaw = wrap(0.0 - yaw0)

    def upd_yaw():
        f = min(1.0, YAW_RATE * (time.monotonic() - t_y) / max(abs(dyaw), 1e-9))
        b.sp.yaw = wrap(yaw0 + f * dyaw)
    b.run_for(abs(dyaw) / YAW_RATE + 0.5, upd_yaw)
    b.sp.yaw = 0.0
    b.run_for(3.0)
    ys = b.window("turn_to_plus_x", "yaw")[-20:]
    yo = float(np.mean([s["yaw"] for s in ys]))
    yn = float(np.mean([s["ned_yaw"] for s in ys]))
    xa = np.mean([s["xaxis"] for s in ys], axis=0)
    ck.add("yaw_odom_zero_faces_east", abs(wrap(yo)) < TOL_YAW and abs(wrap(yn - math.pi / 2)) < TOL_YAW,
           {"yaw_odom_deg": round(math.degrees(yo), 2), "yaw_ned_deg": round(math.degrees(yn), 2),
            "body_x_in_odom": [round(float(v), 3) for v in xa]},
           "odom yaw 0 = PX4 yaw +90 deg (nose east), body x = +x odom")

    b.set_phase("move_plus_x")
    a = top.copy()
    goal = a + np.array([MOVE_DX, 0.0, 0.0])
    ned_a = np.array(b.samples[-1]["ned_p"])
    t_m = time.monotonic()

    def upd_move():
        b.sp.position, b.sp.velocity_ff = ramp(a, goal, V_MOVE, t_m)
    b.run_for(MOVE_DX / V_MOVE + 0.2, upd_move)
    b.sp.position, b.sp.velocity_ff = goal.copy(), None
    mv = [s for s in b.window("move_plus_x", "v") if s["v"][0] > 0.2]
    if mv:
        vm = np.mean([s["v"][:2] for s in mv], axis=0)
        vdir = math.atan2(vm[1], vm[0])
        ym = float(np.mean([s["yaw"] for s in mv]))
        face = max(abs(wrap(s["yaw"] - math.atan2(s["v"][1], s["v"][0]))) for s in mv)
        ck.add("moving_plus_x_faces_motion", abs(wrap(ym - vdir)) < math.radians(30) and abs(vdir) < math.radians(30),
               {"mean_velocity_dir_deg": round(math.degrees(vdir), 2), "mean_yaw_deg": round(math.degrees(ym), 2),
                "max_heading_vs_velocity_deg": round(math.degrees(face), 2), "n": len(mv)},
               "mean odom velocity along +x and body x along it, < 30 deg: sign / axis test (SIH's GPS noise)")
    else:
        ck.add("moving_plus_x_faces_motion", False, "no sample with v_x > 0.2", "")

    b.set_phase("hold_after_move")
    b.run_for(5.0)
    hs = b.window("hold_after_move", "p")
    pm = np.mean([s["p"] for s in hs], axis=0)
    d_ned = np.mean([s["ned_p"] for s in hs], axis=0) - ned_a
    ck.add("move_reached_goal", float(np.linalg.norm(pm - goal)) < TOL_HOLD_XY,
           {"p": [round(float(v), 3) for v in pm], "goal": [round(float(v), 3) for v in goal]},
           f"within {TOL_HOLD_XY} m", gating=False)
    ck.add("plus_x_odom_is_east_ekf", abs(d_ned[1] - MOVE_DX) < TOL_HOLD_XY and abs(d_ned[0]) < TOL_HOLD_XY,
           {"d_north": round(float(d_ned[0]), 3), "d_east": round(float(d_ned[1]), 3),
            "d_down": round(float(d_ned[2]), 3)}, "+1 m odom x = +1 m east in PX4 NED, ~0 north",
           "PX4's own estimate; the ground-truth version is in the ulog check", gating=False)
    ck.add("reset_counter_constant", all(s["reset"] == reset0 for s in b.samples if "reset" in s), reset0,
           "no local-frame reset during the flight")

    b.set_phase("land")
    w = b.wait_until(lambda od, st: st.info.get("nav_state") == F.NAVIGATION_STATE_AUTO_LAND, 5.0,
                     retry=veh.request_land)
    ck.add("land_accepted", w is not None, w, "nav_state AUTO_LAND (18) within 5 s")
    b.sp = None                                 # the FC flies the descent; stop streaming, as the mission does
    b.set_phase("descend")
    w = b.wait_until(lambda od, st: st.landed and not st.armed, 40.0)
    ck.add("landed_and_disarmed", w is not None, w, "status.landed and disarmed within 40 s (auto-disarm)")
    b.set_phase("post")
    b.run_for(2.0)
    od, st = b.tick()
    ck.add("post_landing_flags", st.landed and not st.armed and not st.failsafe,
           {"landed": st.landed, "armed": st.armed, "offboard": st.offboard, "failsafe": st.failsafe,
            "nav_state": st.info.get("nav_state")}, "landed, disarmed, no failsafe",
           "PX4 sets nav_state back to OFFBOARD after the land-mode disarm: `offboard` is not a 'flying' flag")
    ck.add("post_landing_z", od is not None and abs(od.T_odom_body[2, 3]) < 0.3,
           None if od is None else round(float(od.T_odom_body[2, 3]), 3), "|z_odom| < 0.3 m back on the ground",
           gating=False)
    return None, ""


def fallback(b: Bench, ck: Check, timeout):
    b.set_phase("wait_odom")
    w = b.wait_until(lambda od, st: od is not None and st.info.get("status_received"), timeout)
    if w is None:
        return 2, "PX4 never reported odometry + status"
    b.set_phase("observe")
    b.run_for(5.0)
    os_ = b.window("observe", "p")
    ck.add("no_range_message", b.veh._range is None and b.raw.range is None, b.veh._range,
           "no distance_sensor message ever received",
           "the topic may still be advertised: the XRCE client creates its writers at session start")
    ck.add("dist_bottom_none", len(os_) > 50 and all(s["dist_bottom"] is None for s in os_),
           f"{sum(s['dist_bottom'] is None for s in os_)}/{len(os_)} None", "all None (lag model in map_align)")
    return None, ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm-sitl", action="store_true", help="required: this script ARMS the vehicle")
    ap.add_argument("--mode", choices=["flight", "fallback"], default="flight")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ready-timeout", type=float, default=120.0)
    a = ap.parse_args()
    if not a.confirm_sitl or not os.environ.get("NOXEN_SCOUT_SITL"):
        print("refusing: SITL only (needs --confirm-sitl and NOXEN_SCOUT_SITL from run_sitl_check.sh)")
        return 2
    os.makedirs(a.out_dir, exist_ok=True)

    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from px4_msgs.msg import VehicleStatus as PxStatus
    from drivers.px4_scout.px4_scout_vehicle import Px4ScoutVehicle, find_topic

    rclpy.init()
    node = Node("scout_sitl_check")
    t_c = time.monotonic()
    result = {"schema_version": 1, "tool": "sitl_offboard_check", "mode": a.mode,
              "runid": os.environ.get("NOXEN_RUNID", ""), "px4": os.environ.get("NOXEN_SCOUT_SITL"),
              "px4_params": os.environ.get("NOXEN_SCOUT_PX4_PARAMS", "")}
    rc, why = None, ""
    ex, th = None, None
    try:
        veh = Px4ScoutVehicle(node, {"height_ref": os.environ.get("NOXEN_SCOUT_HGT_REF", "baro")},
                              odom_latency_s=0.03, range_below_body_m=0.0)
        result["construct_s"] = round(time.monotonic() - t_c, 2)
        result["topics"] = sorted(n for n, _ in node.get_topic_names_and_types() if n.startswith("/fmu/"))
        raw = Raw(node, find_topic)
        pfc = {"ok": False}
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        node.create_subscription(PxStatus, find_topic(node, "/fmu/out/vehicle_status"),
                                 lambda m: pfc.update(ok=bool(m.pre_flight_checks_pass)), be)
        ex = SingleThreadedExecutor()
        ex.add_node(node)
        stop = threading.Event()

        def spin():
            while not stop.is_set():
                ex.spin_once(timeout_sec=0.05)
        th = threading.Thread(target=spin, daemon=True)
        th.start()
        b = Bench(node, veh, raw, a.out_dir)
        b.pfc_ok = lambda: pfc["ok"]
        ck = b.ck
        rc, why = (flight if a.mode == "flight" else fallback)(b, ck, a.ready_timeout)
        with raw.lock:
            acks = list(raw.acks)
            rx = np.diff(np.array(raw.odom_rx)) if len(raw.odom_rx) > 2 else np.array([])
        result["acks"] = acks
        if a.mode == "flight" and rc != 2:
            want = {F.VEHICLE_CMD_DO_SET_MODE: "offboard", F.VEHICLE_CMD_COMPONENT_ARM_DISARM: "arm",
                    F.VEHICLE_CMD_NAV_LAND: "land"}
            for cid, name in want.items():
                mine = [x for x in acks if x["command"] == cid]
                ck.add(f"ack_{name}_accepted", any(x["result"] == VEHICLE_CMD_RESULT_ACCEPTED for x in mine),
                       [x["result"] for x in mine], "vehicle_command_ack result 0 (ACCEPTED)")
        fr = [s for s in b.samples if "frame_err_p" in s]
        if fr:
            # the adapter's sample and the raw message can be one message apart (two callbacks); where they are the
            # same message the adapter's pose must equal frames.py(raw) to float precision
            same = [s for s in fr if s["frame_err_p"] < 1e-6]
            ck.add("adapter_pose_is_frames_of_px4_ned",
                   len(same) >= 0.8 * len(fr) and max((s["frame_err_R"] for s in same), default=1.0) < 1e-9,
                   {"same_message_samples": f"{len(same)}/{len(fr)}",
                    "max_err_R_same": max((s["frame_err_R"] for s in same), default=None),
                    "max_err_p_any": round(max(s["frame_err_p"] for s in fr), 4)},
                   "adapter T_odom_body == frames.ned_to_odom_pose(PX4 msg) on >= 80 % of samples")
        st_samples = [s for s in b.samples if s.get("stale_status")]
        ck.add("status_never_stale", not st_samples, len(st_samples),
               "0 samples with the adapter's 1 s status-timeout failsafe, incl. before PX4's first vehicle_status "
               "(the startup bug fixed 2026-10-17)")
        if len(rx) > 10:
            result["odom_rx"] = {"n": int(len(rx) + 1), "rate_hz": round(float(1.0 / rx.mean()), 1),
                                 "dt_p50_ms": round(float(np.percentile(rx, 50) * 1e3), 1),
                                 "dt_p99_ms": round(float(np.percentile(rx, 99) * 1e3), 1),
                                 "dt_max_ms": round(float(rx.max() * 1e3), 1)}
        result["phases"] = b.phases
        result["sent_setpoints"] = b.sent
        result["checks"] = ck.items
        if rc is None:
            rc = 0 if ck.all_pass else 1
        b.out.close()
    except Exception as e:                      # noqa: BLE001 -- a bench: record and exit 2
        import traceback
        traceback.print_exc()
        rc, why = 2, f"{type(e).__name__}: {e}"
    result["exit"] = rc
    result["result"] = {0: "PASS", 1: "FAIL", 2: "ERROR"}[rc]
    result["reason"] = why
    with open(os.path.join(a.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result[k] for k in ("result", "exit", "reason", "mode")}), flush=True)
    try:                                        # stop the spin thread first: rclpy torn down under a spinning
        if th is not None:                      # executor aborts the process ("terminate called...", run 3)
            stop.set()
            th.join(timeout=2.0)
            ex.shutdown(timeout_sec=1.0)
        node.destroy_node()
        rclpy.try_shutdown()
    except Exception:                           # noqa: BLE001
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Post-flight half of the SITL check: PX4's own flight log (.ulg) vs what sitl_offboard_check.py sent.

    python3 sitl_ulog_check.py <artifact dir>        (run by run_sitl_check.sh after PX4 has exited)

Two questions the DDS side cannot answer:
  1. Did the FIRMWARE receive the setpoints the adapter meant? For each hold setpoint the bench sent (odom x, y, z,
     yaw; result.json "sent_setpoints"), PX4's logged `trajectory_setpoint` must hold NED (y, x, -z) and yaw
     pi/2 - yaw. Written out here independently of frames.py, so a sign / axis slip in frames.py would show.
  2. Did the AIRCRAFT go where odom said? SIH's ground truth (vehicle_local_position_groundtruth,
     vehicle_attitude_groundtruth) over the last 3 s of each hold: up is up, odom +x is east, odom yaw 0 is nose east.
     Gated on direction and sign; the position errors are reported (they are SIH's estimator, not the adapter).

Merges its checks into <dir>/result.json (source "ulog") and recomputes result / exit (0 / 1 / 2).
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

TOL_SP = 1e-3              # m / rad: the logged setpoint is the float32 copy of what was sent
HOLD_TAIL_S = (3.5, 0.5)   # use [last - 3.5 s, last - 0.5 s] of the time PX4 held a setpoint


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def topic(u, name):
    for d in u.data_list:
        if d.name == name and d.multi_id == 0:
            return d.data
    return None


def main(art):
    from pyulog import ULog
    res_p = os.path.join(art, "result.json")
    res = json.load(open(res_p))
    checks = res.setdefault("checks", [])

    def add(name, ok, value=None, expect=None, note="", gating=True):
        checks.append({"name": name, "pass": bool(ok), "gating": gating, "value": value, "expect": expect,
                       "note": note, "source": "ulog"})
        tag = ("PASS" if ok else "FAIL") if gating else ("ok" if ok else "off") + " (reported)"
        print(f"[ulog] {tag} {name}: {value} (expect {expect}) {note}", flush=True)

    logs = sorted(glob.glob(os.path.join(art, "rootfs", "log", "*", "*.ulg")))
    sent = res.get("sent_setpoints")
    if not logs or not sent:
        add("ulog_available", False, {"logs": len(logs), "sent_setpoints": bool(sent)}, "a .ulg and sent setpoints")
    else:
        u = ULog(logs[-1])
        ts, gt, ga = topic(u, "trajectory_setpoint"), topic(u, "vehicle_local_position_groundtruth"), \
            topic(u, "vehicle_attitude_groundtruth")
        t_sp = ts["timestamp"] * 1e-6
        sp = np.stack([ts["position[0]"], ts["position[1]"], ts["position[2]"]], axis=1)
        tg = gt["timestamp"] * 1e-6
        pg = np.stack([gt["x"], gt["y"], gt["z"]], axis=1)
        ta = ga["timestamp"] * 1e-6
        q = np.stack([ga[f"q[{i}]"] for i in range(4)], axis=1)
        yaw_g = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        est = topic(u, "vehicle_local_position")
        if est is not None:                                 # how good is SIH's own estimator? (explains the wander)
            te = est["timestamp"] * 1e-6
            air = est["z"] < -0.3
            err = {}
            for k_e, k_g in (("x", "x"), ("y", "y"), ("z", "z"), ("vx", "vx"), ("vy", "vy"), ("vz", "vz")):
                e = np.abs(est[k_e] - np.interp(te, tg, gt[k_g]))[air]
                err[k_e] = round(float(np.percentile(e, 95)), 3) if len(e) else None
            add("sih_ekf_error_p95_airborne", True, err, "PX4 estimate - SIH ground truth, p95 (m, m/s)",
                "context, not a check: the vehicle chases this error", gating=False)
        truth = {}
        for name, (x, y, z, yaw) in sent.items():
            ned = np.array([y, x, -z])                       # odom (x east, y north, z up) -> NED, by hand
            yaw_ned = wrap(math.pi / 2 - yaw)
            m = (np.max(np.abs(sp - ned), axis=1) < TOL_SP) & \
                (np.abs(np.arctan2(np.sin(ts["yaw"] - yaw_ned), np.cos(ts["yaw"] - yaw_ned))) < TOL_SP)
            add(f"fw_received_setpoint_{name}", int(m.sum()) > 0,
                {"n_logged": int(m.sum()), "ned": [round(float(v), 4) for v in ned],
                 "yaw_ned_deg": round(math.degrees(yaw_ned), 2)},
                "PX4's trajectory_setpoint holds NED (y, x, -z), yaw pi/2 - yaw")
            if m.sum() == 0:
                continue
            t1 = float(t_sp[m].max())
            w = (tg > t1 - HOLD_TAIL_S[0]) & (tg < t1 - HOLD_TAIL_S[1])
            wa = (ta > t1 - HOLD_TAIL_S[0]) & (ta < t1 - HOLD_TAIL_S[1])
            truth[name] = {"p_ned": pg[w].mean(axis=0), "yaw_ned": float(np.arctan2(np.sin(yaw_g[wa]).mean(),
                                                                                     np.cos(yaw_g[wa]).mean())),
                           "ned_sp": ned, "t_first": float(t_sp[m].min()), "t_last": t1, "mask": m,
                           "wander_m": [round(float(v), 3) for v in pg[w][:, :2].std(axis=0)]}
        if "hold_1m" in truth:
            up = -float(truth["hold_1m"]["p_ned"][2])
            add("truth_odom_z_is_up", up > 0.6, round(up, 3), "truth height > 0.6 m while odom z setpoint is +1.0")
            add("truth_hold_height", abs(up - 1.0) < 0.15, round(up, 3), "1.0 ± 0.15 m (SIH estimator + control)",
                gating=False)
        if "turned" in truth and "goal" in truth:
            # SIH's GPS carries 0.2 m white position noise (sensor_gps_sim) and the TRUE hover wanders up to ±0.7 m over a
            # hold (2026-10-17 runs), so hold-to-hold means cannot resolve a 1 m move. The gating test is the true
            # displacement over the 2 s ramp (+1 s for the vehicle to follow): last 'turned' setpoint -> first 'goal'
            # setpoint + 1 s. It is a sign / axis test (an x/y swap reads 90 deg, a sign flip 180 deg): 45 deg, because
            # SIH's EKF velocity is itself off by up to ~0.6 m/s here (reported below) and the true ramp direction read
            # -9, -12 and -32 deg over three runs.
            mt = truth["turned"]["mask"] & (t_sp < truth["goal"]["t_first"])
            t_a, t_b = float(t_sp[mt].max()), truth["goal"]["t_first"] + 1.0
            i_a, i_b = np.searchsorted(tg, [t_a, t_b])
            i_b = min(i_b, len(tg) - 1)
            d = pg[i_b] - pg[i_a]
            ang = math.degrees(math.atan2(d[0], d[1]))      # 0 = east, + = towards north
            mag = float(np.hypot(d[0], d[1]))
            add("truth_plus_x_odom_is_east", abs(ang) < 45.0 and 0.5 < mag < 1.5,
                {"d_north": round(float(d[0]), 3), "d_east": round(float(d[1]), 3), "dir_deg_from_east": round(ang, 1),
                 "dist_m": round(mag, 3), "window_s": round(t_b - t_a, 2)},
                "true motion over the +1 m odom-x ramp is east: direction < 45 deg, 0.5-1.5 m")
            dh = truth["goal"]["p_ned"] - truth["turned"]["p_ned"]
            add("truth_move_error_holds", abs(dh[1] - 1.0) < 0.15 and abs(dh[0]) < 0.15,
                {"d_north": round(float(dh[0]), 3), "d_east": round(float(dh[1]), 3),
                 "hover_wander_std_NE_m": {k: truth[k]["wander_m"] for k in ("turned", "goal")}},
                "hold means 1 m east ± 0.15 m (SIH estimator + control)", gating=False)
            yg = truth["goal"]["yaw_ned"]
            # sign / axis test again (a convention slip reads 90 or 180 deg off): SIH's own yaw estimate was 6 deg off
            # truth in one run (truth 85.0, estimate 91.0); the tight 5 deg check vs PX4's estimate is on the DDS side
            add("truth_yaw_odom_zero_is_east", abs(wrap(yg - math.pi / 2)) < math.radians(15.0),
                round(math.degrees(yg), 2), "true heading 90 ± 15 deg (east) while odom yaw setpoint is 0")
    gating = [c for c in checks if c.get("gating", True)]
    ok = all(c["pass"] for c in gating)
    if res.get("exit") == 2:
        pass                                                # the flight itself could not be evaluated
    else:
        res["exit"] = 0 if ok else 1
        res["result"] = "PASS" if ok else "FAIL"
    res["n_checks"] = {"gating": len(gating), "gating_failed": sum(not c["pass"] for c in gating),
                       "reported": len(checks) - len(gating)}
    json.dump(res, open(res_p, "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("result", "exit", "n_checks")}), flush=True)
    return res["exit"]


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))

#!/usr/bin/env python3
"""SITL ONLY: watch the `scout_mission` node's events (scout_mission/state, JSON) until it ends (docs/SCOUT_NAV.md §10).

    sitl_mission_monitor.py --out-dir ART [--timeout-s 420]

Writes ART/mission_events.jsonl (every event) and ART/result.json (the state sequence with times, plans, obstacles,
emergencies, the terminal state and its reason; plus the camera's true track against the goal when
ART/camera_truth.jsonl exists: SIH truth, or the EKF pose with --camera-pose odom). Exit 0 = the mission reached DONE / 1 = it ended REFUSED or ABORTED / 2 = no terminal
state before the timeout (or no events at all).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

TERMINAL = ("DONE", "REFUSED", "ABORTED")


def track_by_state(rows, trans, camera_pose="truth"):
    """The camera track (camera_truth.jsonl: ros_t, body p_map) cut at the mission's transitions (same clock: the
    node stamps events with ROS time = system time). The first row is the aircraft resting on the pad (the camera's
    anchor). Per state visit: duration, height range, horizontal distance from the takeoff spot; plus the touchdown
    (last row) against the takeoff spot."""
    pad = rows[0]["p_map"]
    edges = [(None, "PREFLIGHT")] + [(t["t"], t["to"]) for t in trans]
    out = []
    for i, (t0, st) in enumerate(edges):
        t1 = edges[i + 1][0] if i + 1 < len(edges) else None
        seg = [r for r in rows if (t0 is None or r["ros_t"] >= t0) and (t1 is None or r["ros_t"] < t1)]
        e = {"state": st, "t0": t0, "t1": t1, "duration_s": None if t0 is None or t1 is None else round(t1 - t0, 2),
             "frames": len(seg)}
        if seg:
            z = [r["p_map"][2] for r in seg]
            dxy = [math.hypot(r["p_map"][0] - pad[0], r["p_map"][1] - pad[1]) for r in seg]
            e.update(z_min=round(min(z), 3), z_mean=round(sum(z) / len(z), 3), z_max=round(max(z), 3),
                     xy_from_pad_mean=round(sum(dxy) / len(dxy), 3), xy_from_pad_max=round(max(dxy), 3))
        out.append(e)
    last = rows[-1]["p_map"]
    return {"pose_source": camera_pose, "takeoff_spot": pad, "states": out,
            "touchdown": {"p_map": last, "xy_from_takeoff_m": round(math.hypot(last[0] - pad[0], last[1] - pad[1]), 3),
                          "z_minus_takeoff_m": round(last[2] - pad[2], 3)}}


def summarize(events, out_dir, timeout_hit, camera_pose="truth"):
    trans = [{"t": e.get("t"), "from": e.get("state"), "to": e.get("to"), "reason": e.get("reason", "")}
             for e in events if e.get("event") == "transition"]
    states = ["PREFLIGHT"] + [t["to"] for t in trans]
    final = states[-1] if trans else ("PREFLIGHT" if events else None)
    kinds = {}
    for e in events:
        kinds[e.get("event")] = kinds.get(e.get("event"), 0) + 1
    goal = next((e for e in events if e.get("event") == "goal"), None)
    res = {"schema_version": 1, "tool": "sitl_mission_monitor", "final_state": final,
           "final_reason": trans[-1]["reason"] if trans else "", "states": states, "transitions": trans,
           "event_counts": kinds, "goal": goal,
           "plans": [{"t": e.get("t"), "shape": e.get("shape"), "n_wp": e.get("n_wp")} for e in events
                     if e.get("event") == "plan"],
           "timeout": bool(timeout_hit)}
    ct = os.path.join(out_dir, "camera_truth.jsonl")
    if goal and os.path.exists(ct):
        rows = [json.loads(line) for line in open(ct, encoding="utf-8") if line.strip()]
        gp = goal["position"]
        if rows:
            d = [math.dist(r["p_map"], gp) for r in rows]
            k = min(range(len(d)), key=d.__getitem__)
            res["camera_track"] = {"pose_source": camera_pose, "frames": len(rows), "closest_to_goal_m": round(d[k], 3),
                            "closest_p_map": rows[k]["p_map"], "closest_yaw_deg": rows[k]["yaw_map_deg"],
                            "goal_yaw_deg": goal.get("yaw_deg"), "max_z_m": max(r["p_map"][2] for r in rows),
                            "last_p_map": rows[-1]["p_map"],
                            "note": ("SIH ground truth" if camera_pose == "truth" else
                                     "PX4 EKF pose (odom), NOT truth") + " of every published frame, in map; tracking "
                                    "precision is reported, never gated (SIH EKF wanders 0.3-0.7 m)"}
    if os.path.exists(ct):
        rows = [json.loads(line) for line in open(ct, encoding="utf-8") if line.strip()]
        if rows:
            res["track_by_state"] = track_by_state(rows, trans, camera_pose)
    # DONE also follows a land-in-place (localization lost, no path, time limit): say how it got there
    res["arrived"] = "ARRIVED" in states
    res["land_reason"] = next((t["reason"] for t in trans if t["to"] == "LAND"), None)
    res["result"] = "PASS" if final == "DONE" else ("FAIL" if final in TERMINAL else "ERROR")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--timeout-s", type=float, default=420.0)
    ap.add_argument("--topic", default="scout_mission/state")
    ap.add_argument("--camera-pose", default="truth", help="what sitl_camera_sim.py rendered at (truth | odom)")
    a = ap.parse_args()
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("scout_sitl_mission_monitor")
    events = []
    f = open(os.path.join(a.out_dir, "mission_events.jsonl"), "w", encoding="utf-8")

    def on_ev(msg):
        try:
            ev = json.loads(msg.data)
        except ValueError:
            ev = {"event": "unparsable", "raw": msg.data}
        events.append(ev)
        f.write(json.dumps(ev) + "\n")
        f.flush()
        if ev.get("event") == "transition":
            print(f"[monitor] t={ev.get('t')} {ev.get('state')} -> {ev.get('to')}  {ev.get('reason', '')}", flush=True)

    node.create_subscription(String, a.topic, on_ev, 100)
    t_end = time.monotonic() + a.timeout_s
    timeout_hit = True
    try:
        while time.monotonic() < t_end:
            rclpy.spin_once(node, timeout_sec=0.2)
            last = next((e for e in reversed(events) if e.get("event") == "transition"), None)
            if last and last.get("to") in TERMINAL:
                t_drain = time.monotonic() + 1.0
                while time.monotonic() < t_drain:
                    rclpy.spin_once(node, timeout_sec=0.1)
                timeout_hit = False
                break
    except KeyboardInterrupt:
        pass
    f.close()
    res = summarize(events, a.out_dir, timeout_hit, a.camera_pose)
    with open(os.path.join(a.out_dir, "result.json"), "w", encoding="utf-8") as fo:
        json.dump(res, fo, indent=1)
    print(json.dumps({k: res[k] for k in ("result", "final_state", "final_reason", "arrived", "land_reason", "states")}),
          flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return {"PASS": 0, "FAIL": 1}.get(res["result"], 2)


if __name__ == "__main__":
    sys.exit(main())

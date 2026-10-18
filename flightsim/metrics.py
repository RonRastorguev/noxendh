"""Scenario metrics + pass/fail (docs/SCOUT_NAV.md §8). Plain data in, plain data out (JSON-able).

`compute(run)` turns what loop.py recorded into metrics; `evaluate(metrics, expect)` applies a scenario's pass bars
(PROVISIONAL, from the brief: no collision; arrival <= 0.15 m and <= 10 deg; scenario-specific bars).
Everything measured against the TRUE world (the vehicle sim's truth), never against the flight software's own
belief, except where a metric is explicitly "vs the mission's goal".
"""
from __future__ import annotations

import math

import numpy as np

MATCH_TOL_M = 0.3          # a detected box within this of a true unmapped obstacle counts as that obstacle


def _stats(a) -> dict:
    a = np.asarray([v for v in a if v is not None and np.isfinite(v)], float)
    if len(a) == 0:
        return {"n": 0, "median": None, "p95": None, "max": None}
    return {"n": int(len(a)), "median": round(float(np.median(a)), 4), "p95": round(float(np.percentile(a, 95)), 4),
            "max": round(float(np.max(a)), 4)}


def box_gap(lo_a, hi_a, lo_b, hi_b) -> float:
    """Largest axis separation between two boxes (<= 0: they overlap)."""
    lo_a, hi_a, lo_b, hi_b = (np.asarray(v, float) for v in (lo_a, hi_a, lo_b, hi_b))
    return float(np.max(np.maximum(lo_b - hi_a, lo_a - hi_b)))


def match_obstacle(det: dict, truths: list, tol=MATCH_TOL_M):
    """Name of the true unmapped obstacle a detected box corresponds to, or None (a false obstacle)."""
    best, gbest = None, math.inf
    for tr in truths:
        if tr.get("appear_at") is not None and det.get("first_seen") is not None \
                and det["first_seen"] < tr["appear_at"] - 0.05:
            continue
        g = box_gap(det["lo"], det["hi"], tr["lo"], tr["hi"])
        if g <= tol and g < gbest:
            best, gbest = tr["name"], g
    return best


def emergency_episodes(looming: list, gap_s=0.5) -> int:
    n, last = 0, None
    for r in looming:
        if r.get("emergency"):
            if last is None or r["stamp"] - last > gap_s:
                n += 1
            last = r["stamp"]
    return n


def geometric_pass(rows: list, ob: dict, a_xy, goal_xy) -> str:
    """How the TRUE trajectory got past an obstacle: "over" (entered its footprint above it), "around" (passed it
    without entering the footprint), "not_passed"."""
    lo, hi = np.asarray(ob["lo"], float), np.asarray(ob["hi"], float)
    d = np.asarray(goal_xy, float) - np.asarray(a_xy, float)
    L = float(np.linalg.norm(d))
    u = d / L if L > 0 else np.array([1.0, 0.0])
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [lo[0], hi[1]], [hi[0], hi[1]]])
    far = float(np.max((corners - a_xy) @ u))
    over = passed = False
    for r in rows:
        if not r.get("airborne"):
            continue
        x, y = r["x"], r["y"]
        if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]:
            over = True
        if (np.array([x, y]) - a_xy) @ u > far:
            passed = True
    return "over" if over else ("around" if passed else "not_passed")


def compute(run: dict) -> dict:
    rows = run.get("rows", [])
    truths = run.get("true_obstacles", [])
    m = {"scenario": run.get("scenario"), "seed": run.get("seed"), "outcome": run.get("outcome"),
         "final_state": run.get("final_state"), "reason": run.get("reason"),
         "states_visited": run.get("states_visited", [])}
    col = run.get("collisions", [])
    m["collision"] = bool(col)
    m["collisions"] = col
    m["min_clearance_m"] = run.get("min_lateral_clearance_m")
    m["min_clearance_what"] = run.get("min_lateral_what")
    m["min_height_above_surface_m"] = run.get("min_below_clearance_m")
    m["arrival"] = run.get("arrival")
    m["landing"] = run.get("landing")
    m["hover"] = run.get("hover")
    m["max_speed_xy_mps"] = run.get("max_speed_xy_mps")
    m["max_tilt_deg"] = run.get("max_tilt_deg")
    m["ever_armed"] = bool(run.get("ever_armed"))

    looming = run.get("looming", [])
    m["emergencies"] = emergency_episodes(looming)
    m["emergency_frames"] = int(sum(1 for r in looming if r.get("emergency")))
    if run.get("mission_emergencies") is not None:
        m["mission_emergencies"] = int(run["mission_emergencies"])
    dets = [d for r in looming for d in r.get("obstacles", [])]
    m["detections"] = {"n": len(dets), "true": sum(1 for d in dets if match_obstacle(d, truths)),
                       "false": sum(1 for d in dets if not match_obstacle(d, truths))}
    obs = run.get("obstacles_final", [])
    matched = [match_obstacle(o, truths) for o in obs]
    m["obstacles_created"] = len(obs)
    m["obstacles_true"] = sum(1 for x in matched if x)
    m["obstacles_false"] = sum(1 for x in matched if not x)
    m["obstacles"] = [{"id": o.get("id"), "lo": o["lo"], "hi": o["hi"], "source": o.get("source"),
                       "top_seen": o.get("top_seen"), "matches": x} for o, x in zip(obs, matched)]
    m["true_obstacles_detected"] = sorted({x for x in matched if x})

    plans = run.get("plans", [])
    shapes = run.get("plan_shapes")
    if shapes is None:
        shapes = [p["shape"] for p in plans if p.get("ok")]
    m["plan_shapes"] = shapes
    m["plans"] = {"n": len(plans), "failed": sum(1 for p in plans if not p.get("ok")),
                  "max_ms": _stats([p.get("ms") for p in plans])["max"]}

    air = [r for r in rows if r.get("airborne")]
    m["alignment"] = {"xy_err_m": _stats([r.get("align_xy_err_m") for r in air]),
                      "yaw_err_deg": _stats([r.get("align_yaw_err_deg") for r in air]),
                      "pose_xy_err_m": _stats([r.get("pose_xy_err_m") for r in air]),
                      "pose_z_err_m": _stats([r.get("pose_z_err_m") for r in air])}
    loc = run.get("loc", [])
    fixes = [r for r in loc if r.get("fresh")]
    afix = run.get("align_fixes", [])
    m["localization"] = {"frames": len(loc), "fixes": len(fixes),
                         "fix_rate": round(len(fixes) / len(loc), 3) if loc else None,
                         "fix_pos_err_m": _stats([r.get("pos_err_m") for r in fixes]),
                         "fix_rot_err_deg": _stats([r.get("rot_err_deg") for r in fixes]),
                         "ms": _stats([r.get("ms") for r in loc]),
                         "aligner_accepted": sum(1 for r in afix if r.get("accepted")),
                         "aligner_rejected": sum(1 for r in afix if not r.get("accepted"))}

    a_xy, g_xy = run.get("takeoff_xy"), run.get("goal_xy")
    geo = {}
    if a_xy is not None and g_xy is not None:
        for tr in truths:
            geo[tr["name"]] = geometric_pass(rows, tr, np.asarray(a_xy, float), g_xy)
    m["geometric"] = geo
    m["events"] = {"late_obstacle": run.get("late_obstacle_event"), "reset": run.get("reset_event")}
    tm = run.get("timing", {})
    sim_s = tm.get("sim_s") or 0.0
    m["timing"] = dict(tm)
    if sim_s > 0:
        for k in ("wall_s", "render_s", "mission_s", "mission_tick_s", "mission_image_s", "physics_s"):
            if k in tm:
                m["timing"][k.replace("_s", "") + "_per_sim_s"] = round(tm[k] / sim_s, 3)
    return m


def evaluate(m: dict, expect: dict) -> tuple:
    """-> (passed, [ {check, ok, value, bar} ])."""
    checks = []

    def add(name, ok, value, bar):
        checks.append({"check": name, "ok": bool(ok), "value": value, "bar": bar})

    out = expect.get("outcome")
    if out:
        add("outcome", m.get("outcome") == out, m.get("outcome"), out)
    if expect.get("no_collision"):
        add("no_collision", not m.get("collision"), len(m.get("collisions", [])), 0)
    arr = m.get("arrival") or {}
    if "max_pos_err_m" in expect:
        v = arr.get("pos_err_m")
        add("arrival_pos_err_m", v is not None and v <= expect["max_pos_err_m"], v, expect["max_pos_err_m"])
    if "max_yaw_err_deg" in expect:
        v = arr.get("yaw_err_deg")
        add("arrival_yaw_err_deg", v is not None and v <= expect["max_yaw_err_deg"], v, expect["max_yaw_err_deg"])
    if "max_obstacles" in expect:
        add("obstacles_created", m.get("obstacles_created", 0) <= expect["max_obstacles"],
            m.get("obstacles_created"), f"<= {expect['max_obstacles']}")
        add("detections", m["detections"]["n"] <= expect["max_obstacles"], m["detections"]["n"],
            f"<= {expect['max_obstacles']}")
    if "max_emergencies" in expect:
        add("emergencies", m.get("emergencies", 0) <= expect["max_emergencies"], m.get("emergencies"),
            f"<= {expect['max_emergencies']}")
    if "min_emergencies" in expect:
        add("emergencies", m.get("emergencies", 0) >= expect["min_emergencies"], m.get("emergencies"),
            f">= {expect['min_emergencies']}")
    if "shape_contains" in expect:
        s = expect["shape_contains"]
        add("plan_shape", any(s in (sh or "") for sh in m.get("plan_shapes", [])), m.get("plan_shapes"),
            f"some plan shape contains {s!r}")
    for name, want in (expect.get("geometric") or {}).items():
        got = m.get("geometric", {}).get(name)
        add(f"geometric_{name}", got == want, got, want)
    if expect.get("final_state"):
        add("final_state", m.get("final_state") == expect["final_state"], m.get("final_state"), expect["final_state"])
    if expect.get("landed_on_pad"):
        ld = m.get("landing") or {}
        add("landed_on_pad", bool(ld.get("on_pad")), ld.get("on_pad"), True)
    if "max_pad_xy_err_m" in expect:
        v = (m.get("landing") or {}).get("pad_xy_err_m")
        add("pad_xy_err_m", v is not None and v <= expect["max_pad_xy_err_m"], v, expect["max_pad_xy_err_m"])
    hv = m.get("hover") or {}
    if "min_hover_s" in expect:
        v = hv.get("duration_s")
        add("hover_s", v is not None and v >= expect["min_hover_s"] - 0.06, v, f">= {expect['min_hover_s']}")
    if "max_hover_xy_dev_m" in expect:
        v = hv.get("xy_dev_max_m")
        add("hover_xy_dev_m", v is not None and v <= expect["max_hover_xy_dev_m"], v, expect["max_hover_xy_dev_m"])
    if "max_hover_z_err_m" in expect:
        v = hv.get("z_err_max_m")
        add("hover_z_err_m", v is not None and v <= expect["max_hover_z_err_m"], v, expect["max_hover_z_err_m"])
    if expect.get("never_armed"):
        add("never_armed", not m.get("ever_armed"), m.get("ever_armed"), False)
    return all(c["ok"] for c in checks), checks

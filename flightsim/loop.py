"""Closed-loop runner: SimVehicle + camera + the real flight software (navigation.mission.Mission), one scenario.

    run_scenario(scenario, seed, out_dir, ...) -> metrics dict   (also writes <out_dir>/<runid>/{events.jsonl,
                                                                   trajectory.csv, metrics.json})

Clock: one sim clock for everything (odometry stamps, image stamps, mission.tick(now)). Per physics step (100 Hz):
scenario hooks (late obstacle, frame reset) -> camera capture every 1/rate_hz (rendered at the TRUE camera pose,
queued for delivery after latency_s, stamped at capture) -> deliver due images to mission.on_image(img, stamp) ->
mission.tick(now) at tick_hz -> vehicle.step().

The flight software is built by `build_mission` (imports are lazy: this module imports without the other pieces;
a missing piece makes run_scenario raise ComponentsMissing -> the CLI exits 2). Tests and experiments can pass their
own `mission_factory` with the same signature.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import deque

import numpy as np

from localization import synthetic_room as sr
from localization.geometry import pose_error

from . import metrics as M
from . import scenarios as S
from .vehicle_sim import SimVehicle, wrap, yaw_of
from .world import ROOT, SimWorld

TERMINAL = ("DONE", "REFUSED", "ABORTED")

DEFAULTS = {
    "world": {},                         # flightsim.world DEFAULTS overrides
    "vehicle": {"fc": {"pilot_arm_after_s": 0.5}},   # SimVehicle overrides (the sim pilot arms 0.5 s after offboard)
    "camera": {"rate_hz": 10.0, "latency_s": 0.1, "gain": 0.85, "noise_sigma": 2.0, "phase_s": 0.0},
    "loop": {"tick_hz": 20.0, "stop_at": "arrived", "arrived_dwell_s": 3.0, "arrival_window_s": 1.0,
             "post_terminal_s": 2.0},
    "mission": {},                       # Mission cfg overrides on top of config/scout_mission.yaml
    "goal_standoff_m": None,             # None = navigation config standoff_m (0.8 fallback)
}


class ComponentsMissing(RuntimeError):
    """A flight-software piece (mission, aligner, looming, navigation) is not importable: cannot run."""


def _merge(base: dict, over: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_sim_config(path=None, overrides=None) -> dict:
    p = path or os.path.join(ROOT, "config", "scout_sim.yaml")
    y = {}
    if os.path.exists(p):
        import yaml
        with open(p, encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
    return _merge(_merge(DEFAULTS, y), overrides)


# --------------------------------------------------------------------------------------------------------------
# recording proxies: see what the components returned without touching their code


class Rec:
    """Forwards everything to `target`; calls to methods named in `hooks` are timed and their results handed to
    hook(result, args, kwargs, ms)."""

    def __init__(self, target, hooks: dict):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_hooks", hooks)

    def __getattr__(self, name):
        v = getattr(self._target, name)
        h = self._hooks.get(name)
        if h is None or not callable(v):
            return v

        def wrapped(*a, **k):
            t0 = time.perf_counter()
            r = v(*a, **k)
            h(r, a, k, (time.perf_counter() - t0) * 1000.0)
            return r
        return wrapped

    def __setattr__(self, name, value):
        setattr(self._target, name, value)


class Recorder:
    def __init__(self):
        self.now = 0.0
        self.truth_cam = {}             # image stamp -> true T_map_cam
        self.loc, self.align_fixes, self.looming, self.plans = [], [], [], []

    def _loc(self, out, a, k, ms):
        r = {"t": round(self.now, 3), "stamp": float(out.stamp), "fresh": bool(out.fresh), "loc_ok": bool(out.loc_ok),
             "inliers": int(out.n_inliers), "method": out.method, "reason": out.reason, "ms": round(ms, 1)}
        T = self.truth_cam.get(round(float(out.stamp), 4))
        if out.fresh and out.T_map_cam is not None and T is not None:
            r["pos_err_m"], r["rot_err_deg"] = pose_error(out.T_map_cam, T)
        self.loc.append(r)

    def _fix(self, res, a, k, ms):
        if isinstance(res, dict):
            self.align_fixes.append({"t": round(self.now, 3), "stamp": float(a[0]) if a else None,
                                     "accepted": bool(res.get("accepted")), "reason": res.get("reason", ""),
                                     "residual_m": res.get("residual_m"), "residual_deg": res.get("residual_deg")})

    def _loom(self, out, a, k, ms):
        self.looming.append({"t": round(self.now, 3), "stamp": float(out.stamp), "emergency": bool(out.emergency),
                             "min_ttc_s": float(out.min_ttc_s) if np.isfinite(out.min_ttc_s) else None,
                             "map_anchored": bool(out.map_anchored), "n_tracks": int(out.n_tracks),
                             "reason": out.reason, "ms": round(ms, 1),
                             "obstacles": [_ob_dict(o) for o in out.obstacles]})

    def _plan(self, res, a, k, ms):
        self.plans.append({"t": round(self.now, 3), "ok": bool(res.ok), "shape": res.shape, "reason": res.reason,
                           "cost": float(res.cost) if np.isfinite(res.cost) else None,
                           "n_wp": 0 if res.waypoints is None else int(len(res.waypoints)), "ms": round(ms, 1)})


def _ob_dict(o) -> dict:
    return {"id": int(o.id), "lo": [round(float(v), 3) for v in o.lo], "hi": [round(float(v), 3) for v in o.hi],
            "source": o.source, "top_seen": bool(o.top_seen), "first_seen": float(o.first_seen)}


class EventLog(list):
    """Mission log sink: a list that is also callable (Mission accepts either)."""

    def __call__(self, ev):
        self.append(ev)


# --------------------------------------------------------------------------------------------------------------
# the factory: the real flight software, wired for the sim


def build_mission(world: SimWorld, vehicle: SimVehicle, scenario: S.Scenario, sim_cfg: dict, log, recorder: Recorder):
    """Build navigation.mission.Mission from the real components, configured from the repo YAML files.

    -> (mission, components dict with the RAW (unproxied) objects: localizer, aligner, looming, static_map,
        planner, follower, safety, memory)
    What it passes: the sim world's landmark DB and map cloud (map side, cached), the Scout camera model
    (localization.config), the SIM camera mount T_BODY_CAM_SYNTH (to the localizer, looming and Mission alike),
    the fixed takeoff pose A (scout_mission.yaml takeoff.*) and the goal wall point B (goal.point_b)."""
    try:
        from navigation.build import build_flight_stack
    except ImportError as e:                                      # pragma: no cover - depends on the other pieces
        raise ComponentsMissing(f"flight software not importable: {e}") from e

    ms = world.map_side()
    T_bc = sr.T_BODY_CAM_SYNTH
    A = world.takeoff_pose()
    over = {"takeoff": {"position_map": [float(v) for v in A[:3, 3]], "yaw_deg": math.degrees(yaw_of(A))},
            "goal": {"point_b": [float(v) for v in world.wall_point(scenario.goal)]}}
    over = _merge(_merge(over, sim_cfg.get("mission")), scenario.mission)
    hooks = {"localizer": {"process": recorder._loc}, "aligner": {"add_fix": recorder._fix},
             "looming": {"process": recorder._loom}, "planner": {"plan": recorder._plan}}
    try:
        mission, comps = build_flight_stack(vehicle, ms.db, ms.map_xyz, world.cam, T_bc, over, log=log,
                                            wrap=lambda name, obj: Rec(obj, hooks[name]) if name in hooks else obj,
                                            map_spacing_m=float(world.cfg["map_spacing_m"]))
    except ImportError as e:                                      # pragma: no cover
        raise ComponentsMissing(f"flight software not importable: {e}") from e
    return mission, comps


# --------------------------------------------------------------------------------------------------------------
# helpers


def true_goal(world: SimWorld, scenario: S.Scenario, sim_cfg: dict, comps: dict) -> dict:
    """The goal pose from the TRUE wall geometry: B + standoff * inward normal, camera facing B."""
    B = world.wall_point(scenario.goal)
    n = world.wall_normal_true(B)
    nav = comps.get("nav_cfg") or {}
    standoff = sim_cfg.get("goal_standoff_m") or nav.get("standoff_m") or 0.8
    p = B + float(standoff) * n
    lo_z = float(nav.get("min_alt_m", 0.4))
    hi_z = float(world.size[2]) - float(nav.get("ceiling_margin_m", 0.5))
    p[2] = min(max(p[2], lo_z), hi_z)
    return {"position": p, "yaw": math.atan2(-n[1], -n[0]), "point_b": B, "standoff_m": float(standoff)}


def _place_late_obstacle(world: SimWorld, veh: SimVehicle, spec: dict, now: float):
    v = veh.v[:2]
    sp = float(np.linalg.norm(v))
    if sp < spec.get("min_speed_mps", 0.2):
        return None
    u = v / sp
    size = np.asarray(spec["size"], float)
    half_along = 0.5 * float(np.abs(u[0]) * size[0] + np.abs(u[1]) * size[1])
    c = veh.p[:2] + u * (float(spec["ahead_m"]) + half_along)
    lo = np.array([c[0] - size[0] / 2, c[1] - size[1] / 2, 0.0])
    hi = lo + size
    m = float(spec.get("clear_margin_m", 0.25))
    L, W, _ = world.size
    if lo[0] < m or lo[1] < m or hi[0] > L - m or hi[1] > W - m:
        return None
    for nm, blo, bhi in world.solid_boxes():
        if nm == spec["name"]:
            continue
        if M.box_gap(lo - m, hi + m, blo, bhi) <= 0:
            return None
    world.place_obstacle(spec["name"], lo, stamp=now)
    face = float(spec["ahead_m"]) - veh.hull_r
    return {"t": round(now, 3), "name": spec["name"], "lo": lo.round(3).tolist(), "hi": hi.round(3).tolist(),
            "vehicle_p": veh.p.round(3).tolist(), "speed_mps": round(sp, 3), "hull_to_face_m": round(face, 3)}


def _row(t, state, veh: SimVehicle, mission, comps, cl):
    Tt = veh.T_map_body_true()
    od = veh.odometry()
    sp = getattr(mission, "last_setpoint", None)
    r = {"t": round(t, 3), "state": state, "mode": veh.mode, "armed": int(veh.armed), "landed": int(veh.landed),
         "airborne": int(not veh.on_ground), "x": float(Tt[0, 3]), "y": float(Tt[1, 3]), "z": float(Tt[2, 3]),
         "yaw_deg": math.degrees(veh.yaw), "roll_deg": math.degrees(veh.roll), "pitch_deg": math.degrees(veh.pitch),
         "vx": float(veh.v[0]), "vy": float(veh.v[1]), "vz": float(veh.v[2]),
         "odom_x": float(od.T_odom_body[0, 3]), "odom_y": float(od.T_odom_body[1, 3]),
         "odom_z": float(od.T_odom_body[2, 3]), "odom_yaw_deg": math.degrees(yaw_of(od.T_odom_body)),
         "sp_x": None, "sp_y": None, "sp_z": None, "sp_yaw_deg": None,
         "clear_lateral_m": cl.get("lateral"), "clear_below_m": cl.get("below"),
         "align_xy_err_m": None, "align_yaw_err_deg": None, "pose_xy_err_m": None, "pose_z_err_m": None}
    if sp is not None:
        r.update(sp_x=float(sp.position[0]), sp_y=float(sp.position[1]), sp_z=float(sp.position[2]),
                 sp_yaw_deg=math.degrees(sp.yaw))
    al = comps.get("aligner")
    if al is not None:
        try:
            Tmo = al.T_map_odom()
            if Tmo is not None:
                Ttr = veh.T_map_odom_true()
                r["align_xy_err_m"] = float(np.linalg.norm(Tmo[:2, 3] - Ttr[:2, 3]))
                r["align_yaw_err_deg"] = abs(math.degrees(wrap(yaw_of(Tmo) - yaw_of(Ttr))))
            pm = al.pose_map(t)
            if pm is not None:
                r["pose_xy_err_m"] = float(np.linalg.norm(pm[0][:2, 3] - Tt[:2, 3]))
                r["pose_z_err_m"] = float(pm[0][2, 3] - Tt[2, 3])
        except Exception as e:  # noqa: BLE001 - metrics must never stop the flight
            r["align_err_exc"] = repr(e)
    return r


# --------------------------------------------------------------------------------------------------------------
# the run


def run_scenario(scenario, seed: int = 0, out_dir: str | None = None, sim_cfg: dict | None = None,
                 mission_factory=None, world: SimWorld | None = None, verbose: bool = False) -> dict:
    """Run one scenario closed-loop. -> metrics dict (with "passed" and "checks"). Raises ComponentsMissing when
    the flight software cannot be built."""
    t_wall0 = time.perf_counter()
    scen = S.get(scenario)
    cfg = sim_cfg if sim_cfg is not None and "loop" in sim_cfg else load_sim_config(overrides=sim_cfg)
    lc, cc = dict(cfg["loop"], **(scen.loop or {})), cfg["camera"]
    world = world or SimWorld(cfg["world"], obstacles=scen.world_obstacles())
    A = world.takeoff_pose()
    dx, dy, dyaw = scen.takeoff_offset
    T0 = A.copy()
    T0[:2, 3] += [dx, dy]
    T0[:3, :3] = T0[:3, :3] @ np.array([[math.cos(math.radians(dyaw)), -math.sin(math.radians(dyaw)), 0],
                                        [math.sin(math.radians(dyaw)), math.cos(math.radians(dyaw)), 0], [0, 0, 1]])
    veh = SimVehicle(world, T0, _merge(cfg["vehicle"], scen.vehicle), seed=seed)
    log = EventLog()
    rec = Recorder()
    factory = mission_factory or build_mission
    t_b0 = time.perf_counter()
    mission, comps = factory(world, veh, scen, cfg, log, rec)
    build_s = time.perf_counter() - t_b0
    goal = true_goal(world, scen, cfg, comps)

    rate = 1.0 / veh.dt
    cap_every = max(1, int(round(rate / cc["rate_hz"])))
    tick_every = max(1, int(round(rate / lc["tick_hz"])))
    lat_steps = int(round(cc["latency_s"] * rate))
    phase_steps = int(round(cc.get("phase_s", 0.0) * rate))
    img_rng = np.random.default_rng([int(seed), 4242])
    queue = deque()
    sim_events = []
    rows = []
    states = []
    arrived_t = None
    arr_samples = []
    arrival = None
    terminal_t = None
    late = scen.late_obstacle
    late_ev = reset_ev = None
    a_xy = A[:2, 3].copy()
    g_xy = goal["position"][:2]
    u_ag = (g_xy - a_xy) / max(1e-9, float(np.linalg.norm(g_xy - a_xy)))
    tm = {"render_s": 0.0, "mission_tick_s": 0.0, "mission_image_s": 0.0, "physics_s": 0.0, "frames": 0, "ticks": 0}
    ever_armed = False
    stop_reason = "timeout"
    max_steps = int(round(scen.timeout_s * rate))
    step = 0
    state = getattr(mission, "state", "?")
    while step <= max_steps:
        t = step * veh.dt
        rec.now = t
        state = getattr(mission, "state", "?")
        progress = float((veh.p[:2] - a_xy) @ u_ag)
        # --- scenario hooks ---------------------------------------------------------------------------------
        if late and late_ev is None and state == "NAV" and progress >= late["trigger_progress_m"]:
            late_ev = _place_late_obstacle(world, veh, late, t)
            if late_ev:
                sim_events.append(dict({"src": "sim", "event": "late_obstacle_appeared"}, **late_ev))
        if scen.reset_at_progress_m is not None and reset_ev is None and state == "NAV" \
                and progress >= scen.reset_at_progress_m:
            veh.cfg["estimator"]["reset_at_s"] = t
            reset_ev = {"t": round(t, 3), "progress_m": round(progress, 3)}
            sim_events.append(dict({"src": "sim", "event": "reset_triggered"}, **reset_ev))
        # --- camera capture (true pose) -> latency queue ---------------------------------------------------
        if step >= phase_steps and (step - phase_steps) % cap_every == 0:
            t0 = time.perf_counter()
            Tc = veh.T_map_body_true() @ sr.T_BODY_CAM_SYNTH
            img = world.render(Tc, cc["gain"], cc["noise_sigma"], img_rng)
            tm["render_s"] += time.perf_counter() - t0
            tm["frames"] += 1
            rec.truth_cam[round(t, 4)] = Tc
            queue.append((step + lat_steps, t, img))
            for k in [k for k in rec.truth_cam if k < t - 5.0]:
                del rec.truth_cam[k]
        while queue and queue[0][0] <= step:
            _, stamp, img = queue.popleft()
            t0 = time.perf_counter()
            mission.on_image(img, stamp)
            tm["mission_image_s"] += time.perf_counter() - t0
        # --- mission tick --------------------------------------------------------------------------------------
        if step % tick_every == 0:
            t0 = time.perf_counter()
            mission.tick(t)
            tm["mission_tick_s"] += time.perf_counter() - t0
            tm["ticks"] += 1
            state = getattr(mission, "state", "?")
            if not states or states[-1] != state:
                states.append(state)
            cl = getattr(veh, "last_clearance", {}) or {}
            row = _row(t, state, veh, mission, comps, cl)
            rows.append(row)
            ever_armed |= veh.armed
            # arrival: hold ARRIVED for arrived_dwell_s, score the last arrival_window_s
            if state == "ARRIVED":
                if arrived_t is None:
                    arrived_t, arr_samples = t, []
                Tt = veh.T_map_body_true()
                arr_samples.append((t, float(np.linalg.norm(Tt[:3, 3] - goal["position"])),
                                    abs(math.degrees(wrap(yaw_of(Tt) - goal["yaw"]))), Tt[:3, 3].copy()))
                if arrival is None and t - arrived_t >= lc["arrived_dwell_s"] - 1e-9:
                    win = [s for s in arr_samples if s[0] >= t - lc["arrival_window_s"] - 1e-9]
                    arrival = {"t": round(t, 3), "pos_err_m": round(max(s[1] for s in win), 4),
                               "pos_err_mean_m": round(float(np.mean([s[1] for s in win])), 4),
                               "yaw_err_deg": round(max(s[2] for s in win), 3),
                               "position": np.mean([s[3] for s in win], axis=0).round(4).tolist(),
                               "goal_position": goal["position"].round(4).tolist(),
                               "goal_yaw_deg": round(math.degrees(goal["yaw"]), 2)}
                    mg = getattr(mission, "goal", None)
                    if mg is not None:
                        arrival["pos_err_vs_mission_goal_m"] = round(float(np.linalg.norm(
                            arrival["position"] - np.asarray(mg.position, float))), 4)
                        arrival["mission_goal_vs_true_goal_m"] = round(float(np.linalg.norm(
                            np.asarray(mg.position, float) - goal["position"])), 4)
                    if lc["stop_at"] == "arrived":
                        stop_reason = "arrived"
                        break
            else:
                arrived_t = None
            if state in TERMINAL:
                terminal_t = t if terminal_t is None else terminal_t
                if t - terminal_t >= lc["post_terminal_s"]:
                    stop_reason = "terminal"
                    break
        if veh.crashed:
            stop_reason = "crashed"
            rows.append(_row(t, state, veh, mission, comps, getattr(veh, "last_clearance", {}) or {}))
            break
        t0 = time.perf_counter()
        veh.step()
        tm["physics_s"] += time.perf_counter() - t0
        step += 1
    sim_s = step * veh.dt
    tm["sim_s"] = round(sim_s, 3)
    tm["wall_s"] = round(time.perf_counter() - t_wall0, 3)
    tm["build_s"] = round(build_s, 3)
    tm["mission_s"] = tm["mission_tick_s"] + tm["mission_image_s"]
    for k in ("render_s", "mission_tick_s", "mission_image_s", "physics_s", "mission_s"):
        tm[k] = round(tm[k], 3)

    final_state = getattr(mission, "state", "?")
    if veh.crashed:
        outcome = "crashed"
    elif arrival is not None:
        outcome = "arrived"
    elif final_state == "REFUSED":
        outcome = "refused"
    elif final_state == "ABORTED":
        outcome = "aborted"
    elif final_state == "DONE":
        outcome = "done"
    else:
        outcome = "timeout"
    summ = mission.summary() if hasattr(mission, "summary") else {}
    mem = comps.get("memory")
    obs_final = []
    if mem is not None and hasattr(mem, "active"):
        try:
            obs_final = [_ob_dict(o) for o in mem.active(sim_s)]
        except Exception:  # noqa: BLE001
            obs_final = []
    hover = None
    hv = [r for r in rows if r.get("state") == "HOVER"]
    if hv:
        A = world.takeoff_pose()
        h = (getattr(mission, "cfg", {}) or {}).get("hover_test") or {}
        z_t = float(A[2, 3]) + float(h.get("climb_m", 0.0))
        settled = [r for r in hv if r["t"] >= hv[0]["t"] + 1.0] or hv      # skip the first second (climb overshoot)
        hover = {"duration_s": round(hv[-1]["t"] - hv[0]["t"], 3), "target_z": round(z_t, 4),
                 "z_mean": round(float(np.mean([r["z"] for r in settled])), 4),
                 "z_err_max_m": round(float(max(abs(r["z"] - z_t) for r in settled)), 4),
                 "xy_dev_max_m": round(float(max(np.hypot(r["x"] - A[0, 3], r["y"] - A[1, 3]) for r in hv)), 4)}
    landing = None
    if final_state == "DONE":
        Tt = veh.T_map_body_true()
        A = world.takeoff_pose()
        lx, ly = (float(v) for v in Tt[:2, 3])
        pad_top = world.terrain_z(float(A[0, 3]), float(A[1, 3]))
        landing = {"position": Tt[:3, 3].round(4).tolist(),
                   "pad_xy_err_m": round(float(np.hypot(lx - A[0, 3], ly - A[1, 3])), 4),
                   "surface_z": round(float(world.terrain_z(lx, ly)), 4), "pad_top_z": round(float(pad_top), 4),
                   "on_pad": bool(abs(world.terrain_z(lx, ly) - pad_top) < 1e-6 and not veh.armed)}
    run = {"scenario": scen.name, "seed": seed, "outcome": outcome, "final_state": final_state, "landing": landing, "hover": hover,
           "reason": getattr(mission, "reason", ""), "states_visited": states, "stop_reason": stop_reason,
           "collisions": veh.collisions,
           "min_lateral_clearance_m": None if not math.isfinite(veh.min_lateral) else round(veh.min_lateral, 4),
           "min_lateral_what": veh.min_lateral_what,
           "min_below_clearance_m": None if not math.isfinite(veh.min_below_airborne) else round(veh.min_below_airborne, 4),
           "max_speed_xy_mps": round(veh.max_speed_xy, 3), "max_tilt_deg": round(veh.max_tilt_deg, 2),
           "ever_armed": ever_armed or any(e["event"] == "armed" for e in veh.events),
           "arrival": arrival, "looming": rec.looming, "obstacles_final": obs_final,
           "true_obstacles": [{"name": o.name, "lo": o.lo.tolist(), "hi": o.hi.tolist(), "appear_at": o.appear_at}
                              for o in world.obstacles if o.active],
           "plans": rec.plans, "plan_shapes": summ.get("plan_shapes"), "mission_emergencies": summ.get("emergencies"),
           "loc": rec.loc, "align_fixes": rec.align_fixes, "rows": rows,
           "takeoff_xy": a_xy.tolist(), "goal_xy": g_xy.tolist(), "late_obstacle_event": late_ev,
           "reset_event": reset_ev, "timing": tm}
    met = M.compute(run)
    met["stop_reason"] = stop_reason
    met["goal_true"] = {"position": goal["position"].round(4).tolist(), "yaw_deg": round(math.degrees(goal["yaw"]), 2),
                        "point_b": goal["point_b"].tolist(), "standoff_m": goal["standoff_m"]}
    met["passed"], met["checks"] = M.evaluate(met, scen.expect)
    if out_dir:
        _write(out_dir, scen, seed, met, rows, list(log), veh.events, sim_events)
    if verbose:
        print(json.dumps({k: met[k] for k in ("scenario", "seed", "outcome", "passed")}))
    return met


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return repr(o)


def _write(out_dir, scen, seed, met, rows, mission_events, vehicle_events, sim_events):
    runid = f"{scen.name}_s{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    d = os.path.join(out_dir, runid)
    k = 1
    while os.path.exists(d):
        d = os.path.join(out_dir, f"{runid}_{k}")
        k += 1
    os.makedirs(d)
    met["runid"] = os.path.basename(d)
    met["run_dir"] = d
    evs = [dict({"src": "mission"}, **e) for e in mission_events] + list(vehicle_events) + list(sim_events)
    evs.sort(key=lambda e: (e.get("t") is None, e.get("t") or 0.0))
    with open(os.path.join(d, "events.jsonl"), "w", encoding="utf-8") as f:
        for e in evs:
            f.write(json.dumps(e, default=_jsonable) + "\n")
    if rows:
        keys = list(rows[0].keys())
        for r in rows:
            for kk in r:
                if kk not in keys:
                    keys.append(kk)
        with open(os.path.join(d, "trajectory.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()})
    with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(met, f, indent=1, default=_jsonable)

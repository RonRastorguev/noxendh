"""The Scout's mission: take off from the fixed pad, fly to a point on a wall, avoid what the map does not know.

docs/SCOUT_NAV.md §7. One object owns the flight: it feeds camera frames to the localizer (PnP fixes -> map↔odom
alignment) and to the looming detector (obstacles -> memory -> replans), and at `tick()` (20 Hz) it runs the state
machine and streams one odom position setpoint to the vehicle every tick from ARMING on (PX4 offboard needs a
continuous stream; when the stream stops, PX4's own offboard-loss failsafe takes over, DH-05).

Modes (cfg `mode`):
    wall_goal   (default) take off, fly to the wall point B, avoid what looms, then hold / return_land / land.
    hover_test  take off, rise `hover_test.climb_m` (2 m) straight up from the pad, hold that spot `hover_test.hover_s`
                (10 s), land back on the pad: PREFLIGHT -> ARMING -> TAKEOFF -> HOVER -> LAND -> DONE. No goal, no
                looming (nothing moves sideways); the climb is capped below the map ceiling minus its margin.

States:
    PREFLIGHT  on the ground: odom present, the alignment initialised from the FIXED takeoff pose; if PnP fixes on
               the ground disagree with that pose, the Scout is not where it should be -> REFUSED.
    ARMING     stream hold setpoints, ask for offboard, arm (auto_arm) or wait for the pilot to arm.
    HOVER      (hover_test) hold the pad's x, y at the climb height for hover_s, then LAND.
    TAKEOFF    straight up above the pad to max(floor + takeoff_alt_m, pad top + takeoff_min_climb_m); the pad may be
               a raised box (Ron 2026-10-17), which the prior map contains; the column above it is assumed clear (DN-04).
    SETTLE     hover until localization has given `settle_fixes` fresh accepted fixes.
    NAV        plan (2.5-D, over / around) to the wall goal; follow; replan on new obstacles, after an emergency,
               periodically, or when far off the path.
    ARRIVED    hold the goal pose, camera on the wall point, for dwell_s; then hold / return_land / land.
    RETURN     plan back above the pad.
    LAND       hand the descent to the FC (land mode) and wait for it to disarm.
    HOLD       alignment invalid: hold the odom position (the FC's own estimate, no map needed); LAND in place
               after loc_lost_abort_s.
    DONE / REFUSED / ABORTED  terminal. ABORTED stops the setpoint stream (pilot or FC failsafe has the aircraft).

Every transition and every safety-relevant event goes to `log` (a list, or a callable taking a dict).
No ROS; Python 3.9.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from localization.geometry import inv_T, make_T, rot_z
from .types import OdomSample, Setpoint

DEFAULTS = {
    "takeoff": {"position_map": None, "yaw_deg": None},   # REQUIRED: the fixed pad pose (body origin on the ground)
    "goal": {"point_b": None},                             # REQUIRED (wall_goal) unless given to the constructor
    "mode": "wall_goal",              # "wall_goal" | "hover_test"
    "hover_test": {"climb_m": 2.0,    # rise this far above the takeoff pose ...
                   "hover_s": 10.0,   # ... hold it this long, then land on the pad (Ron 2026-10-17)
                   "ceiling_margin_m": None},  # None: the planner's ceiling_margin_m (else 0.5)
    "takeoff_alt_m": 1.0,             # height of the body above the floor after takeoff ...
    "takeoff_min_climb_m": 0.5,       # ... and at least this far above the pad top (the pad may be a raised box)
    "pad_max_rest_height_m": 0.25,    # the pad pose must sit this close above a mapped surface (the pad top)
    "takeoff_speed_mps": 0.3,
    "takeoff_decel_mps2": 0.25,       # the climb ramp slows into its target at this rate
    "takeoff_tol_m": 0.08,
    "preflight_max_offset_m": 0.30,   # PnP fix vs the fixed pad pose
    "preflight_max_offset_deg": 15.0,
    "preflight_fixes": 2,             # consistent fixes needed to accept or to refuse
    "preflight_require_fix": False,   # True: never take off without a PnP fix on the ground
    "preflight_timeout_s": 15.0,
    "stream_before_offboard_s": 1.0,  # PX4 wants setpoints before it accepts offboard
    "auto_arm": False,                # False: the pilot arms (RC); the mission waits
    "arm_timeout_s": 30.0,
    "settle_fixes": 3,
    "settle_timeout_s": 15.0,
    "replan_period_s": 1.0,
    "replan_deviation_m": 0.5,
    "no_path_land_s": 20.0,           # no plan this long -> land in place
    "dwell_s": 5.0,
    "on_arrival": "hold",             # "hold" | "return_land" | "land"
    "return_tol_m": 0.15,
    "loc_lost_abort_s": 6.0,          # alignment invalid this long -> land in place
    "max_mission_s": 300.0,
    "odom_buffer_s": 3.0,
    "ff_alpha": 0.3,                  # low-pass of the odom-setpoint velocity feed-forward (per 20 Hz tick)
    "ff_max_mps": 0.6,
}

TERMINAL = ("DONE", "REFUSED", "ABORTED")
AIRBORNE = ("TAKEOFF", "SETTLE", "NAV", "ARRIVED", "RETURN", "HOLD", "HOVER")
MODES = ("wall_goal", "hover_test")


def _yaw_of(T) -> float:
    return math.atan2(T[1, 0], T[0, 0])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _merge(base: dict, over: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


class Mission:
    def __init__(self, cfg, vehicle, localizer, aligner, looming, static_map, planner, follower, safety, memory,
                 cam=None, T_body_cam=None, log=None, goal_point=None, wall_goal_fn=None):
        self.cfg = _merge(DEFAULTS, cfg)
        c = self.cfg
        self.vehicle = vehicle
        self.localizer = localizer
        self.aligner = aligner
        self.looming = looming
        self.static_map = static_map
        self.planner = planner
        self.follower = follower
        self.safety = safety
        self.memory = memory
        self.cam = cam
        self.T_body_cam = np.eye(4) if T_body_cam is None else np.asarray(T_body_cam, float)
        self._log_sink = log
        self.events = []

        if c["mode"] not in MODES:
            raise ValueError(f"scout_mission: mode must be one of {MODES}, got {c['mode']!r}")
        tk = c["takeoff"]
        if tk.get("position_map") is None or tk.get("yaw_deg") is None:
            raise ValueError("scout_mission: takeoff.position_map and takeoff.yaw_deg are required (the fixed pad)")
        self.T_map_pad = make_T(rot_z(math.radians(float(tk["yaw_deg"]))), np.asarray(tk["position_map"], float))
        self.goal_point = np.asarray(goal_point if goal_point is not None else c["goal"]["point_b"], float) \
            if (goal_point is not None or c["goal"].get("point_b") is not None) else None
        self._wall_goal_fn = wall_goal_fn
        self.goal = None                          # GoalPose of the wall goal
        self.return_goal = None                   # GoalPose-like for RETURN

        self.state = "PREFLIGHT"
        self.reason = ""
        self._t0 = None
        self._state_t = None
        self._now = None
        self._odom = deque()
        self._last_odom_stamp = None
        self._reset_counter = None
        self._stream = False
        self._stream_t0 = None
        self._hold_odom = None                    # latched (p_odom, yaw_odom)
        self._emergency_t = None
        self._emergencies = 0
        self._pre_fixes = []                      # (stamp, T_map_body) on the ground
        self._fixes_since = 0
        self._plan = None
        self._plan_t = None
        self._mem_version = None
        self._need_replan = True
        self._no_path_since = None
        self._arrived_t = None
        self._offboard_seen = False
        self._land_requested = False
        self._align_invalid_since = None
        self._prev_state = None
        self.last_setpoint = None
        self.last_looming = None
        self.last_loc = None
        self.plan_shapes = []

        if hasattr(self.aligner, "set_terrain"):
            self.aligner.set_terrain(self._terrain)
        if hasattr(self.aligner, "set_height_ref"):
            self.aligner.set_height_ref(self.vehicle.capabilities().get("height_ref", "baro"))

    # ------------------------------------------------------------------------------------------------ logging
    def _log(self, kind: str, **kw):
        ev = {"t": None if self._now is None else round(float(self._now), 3), "state": self.state, "event": kind}
        ev.update(kw)
        self.events.append(ev)
        if callable(self._log_sink):
            self._log_sink(ev)
        elif isinstance(self._log_sink, list):
            self._log_sink.append(ev)

    def _to(self, state: str, reason: str = ""):
        if state == self.state:
            return
        self._log("transition", to=state, reason=reason)
        self._prev_state = self.state
        self.state, self.reason, self._state_t = state, reason, self._now
        if state in ("REFUSED", "ABORTED"):
            self._stream = False

    # ------------------------------------------------------------------------------------------------ odom
    def _add_odom(self, s: OdomSample):
        if s is None or (self._last_odom_stamp is not None and s.stamp <= self._last_odom_stamp):
            return
        if self._reset_counter is not None and s.reset_counter != self._reset_counter:
            self._log("odom_reset", counter=int(s.reset_counter))
            self._hold_odom = None              # a latched odom hold is meaningless across a frame reset
            self._odom.clear()
        self._reset_counter = s.reset_counter
        self._last_odom_stamp = s.stamp
        self._odom.append(s)
        while self._odom and self._odom[0].stamp < s.stamp - self.cfg["odom_buffer_s"]:
            self._odom.popleft()
        self.aligner.add_odom(s)

    def _odom_at(self, stamp):
        """Nearest-in-time odom sample interpolated in position / yaw (attitude from the nearer sample)."""
        if not self._odom:
            return None
        if stamp <= self._odom[0].stamp:
            return self._odom[0]
        if stamp >= self._odom[-1].stamp:
            return self._odom[-1]
        prev = self._odom[0]
        for s in self._odom:
            if s.stamp >= stamp:
                a = (stamp - prev.stamp) / max(1e-9, s.stamp - prev.stamp)
                near = s if a >= 0.5 else prev
                T = near.T_odom_body.copy()
                T[:3, 3] = (1 - a) * prev.T_odom_body[:3, 3] + a * s.T_odom_body[:3, 3]
                v = (1 - a) * np.asarray(prev.v_odom) + a * np.asarray(s.v_odom)
                return OdomSample(stamp, T, v, s.reset_counter, s.dist_bottom_m)
            prev = s
        return self._odom[-1]

    # ------------------------------------------------------------------------------------------------ geometry
    def _terrain(self, x, y, z_body_map):
        """Map z of the surface below (x, y): the static map plus remembered obstacles under the vehicle."""
        z = float(self.static_map.terrain_z(x, y, z_body_map))
        if self.memory is not None and self._now is not None:
            for ob in self.memory.active(self._now):
                lo, hi = ob.lo, ob.hi
                if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1] and hi[2] <= z_body_map and hi[2] > z:
                    z = float(hi[2])
        return z

    def _contains_mapped_surface(self, ob, step=0.1) -> bool:
        """True when the obstacle box contains occupied voxels of the prior map above the floor."""
        occ = getattr(self.static_map, "occupied", None)
        if occ is None:
            return False
        lo, hi = np.asarray(ob.lo, float), np.asarray(ob.hi, float)
        z0 = max(lo[2], self._floor_z() + 0.3)
        z1 = min(hi[2], float(getattr(self.static_map, "ceiling_z", hi[2])) - 0.2)
        if z1 < z0:
            return False
        g = [np.arange(a, b + 1e-9, step) if b > a else np.array([a]) for a, b in
             ((lo[0], hi[0]), (lo[1], hi[1]), (z0, z1))]
        P = np.stack(np.meshgrid(*g, indexing="ij"), axis=-1).reshape(-1, 3)
        return bool(np.any(occ(P)))

    def _floor_z(self):
        return float(getattr(self.static_map, "floor_z", 0.0))

    def _pad_top_z(self) -> float:
        """Map z of the surface the pad pose rests on (the pad box top, or the floor), from the prior map."""
        p = self.T_map_pad[:3, 3]
        return float(self.static_map.terrain_z(float(p[0]), float(p[1]), float(p[2]) + 0.05))

    def _takeoff_target_z(self) -> float:
        """Where TAKEOFF stops: the hover-test height, or the normal departure height above the pad."""
        if self.cfg["mode"] != "hover_test":
            return self._hover_z_above_pad()
        h = self.cfg["hover_test"]
        want = float(self.T_map_pad[2, 3]) + float(h["climb_m"])
        margin = h.get("ceiling_margin_m")
        if margin is None:
            margin = (getattr(self.planner, "cfg", None) or {}).get("ceiling_margin_m", 0.5)
        cap = float(getattr(self.static_map, "ceiling_z", float("inf"))) - float(margin)
        if want > cap and not getattr(self, "_hover_cap_logged", False):
            self._hover_cap_logged = True
            self._log("hover_height_capped", wanted_z=round(want, 3), cap_z=round(cap, 3),
                      note="climb_m would come within ceiling_margin_m of the mapped ceiling")
        return min(want, cap)

    def _hover_z_above_pad(self) -> float:
        c = self.cfg
        return max(self._floor_z() + float(c["takeoff_alt_m"]), self._pad_top_z() + float(c["takeoff_min_climb_m"]))

    def _pose_map(self):
        pm = self.aligner.pose_map(self._now)
        if pm is None:
            return None, None
        return pm[0], np.asarray(pm[1], float)

    # ------------------------------------------------------------------------------------------------ setpoints
    def _send_odom(self, p_odom, yaw_odom, v_ff=None):
        sp = Setpoint(self._now, np.asarray(p_odom, float), float(yaw_odom), v_ff)
        self.last_setpoint = sp
        self.vehicle.send_setpoint(sp)

    def _send_map(self, p_map, yaw_map):
        """Map setpoint -> odom, with a velocity feed-forward = the odom setpoint's own rate of change. The odom
        setpoint moves both because the follower moves it and because the alignment slides it to cancel the FC's
        drift; PX4's position loop is proportional, so without the feed-forward it trails a moving setpoint by
        speed / MPC_XY_P (the sim's strong-drift run sat 10-16 cm off the goal, DN-08)."""
        p_odom, yaw_odom = self.aligner.setpoint_to_odom(np.asarray(p_map, float), float(yaw_map))
        p_odom = np.asarray(p_odom, float)
        # The feed-forward follows the setpoint WITHOUT the height reference's changes (DN-63): under the range height
        # reference a terrain step re-references PX4's z (its position estimate steps, its velocity does not), so the
        # odom setpoint's step is not motion. Fed forward, it made the vehicle really dip by it (low_box: 0.26 m at
        # crate_c's 0.7 m edge -> a looming emergency at the crate top) or bump up leaving the pad.
        href = getattr(self.aligner, "height_reference_z", None)
        p_ff = p_odom + np.array([0.0, 0.0, float(href()) if href is not None else 0.0])
        v_ff = None
        prev = getattr(self, "_ff_prev", None)
        if prev is not None and self._hold_odom is None and 0.0 < self._now - prev[0] <= 0.25:
            v = (p_ff - prev[1]) / (self._now - prev[0])
            a = float(self.cfg["ff_alpha"])
            v_ff = v if prev[2] is None else a * v + (1 - a) * prev[2]
            n = float(np.linalg.norm(v_ff))
            if n > float(self.cfg["ff_max_mps"]):
                v_ff = v_ff * (float(self.cfg["ff_max_mps"]) / n)
        self._ff_prev = (self._now, p_ff, v_ff)
        self._hold_odom = None
        self._send_odom(p_odom, yaw_odom, v_ff)

    def _send_hold(self):
        """Hold where the FC thinks it is: a latched odom position (needs no map alignment)."""
        self._ff_prev = None
        if self._hold_odom is None:
            o = self._odom[-1] if self._odom else None
            if o is None:
                return
            self._hold_odom = (o.T_odom_body[:3, 3].copy(), _yaw_of(o.T_odom_body))
            self._log("hold_latched", p_odom=[round(float(v), 3) for v in self._hold_odom[0]])
        self._send_odom(*self._hold_odom)

    # ------------------------------------------------------------------------------------------------ camera
    def on_image(self, image, stamp: float) -> None:
        if self.state in TERMINAL:
            return
        odom = self._odom_at(stamp)
        prior = None
        if self.aligner.valid(stamp):
            Tb = self.aligner.body_pose_map_at(stamp)
            if Tb is not None:
                prior = Tb @ self.T_body_cam
        if prior is None and self.state in ("PREFLIGHT", "ARMING"):
            prior = self.T_map_pad @ self.T_body_cam
        out = self.localizer.process(image, stamp, prior)
        self.last_loc = out
        if out.fresh and out.T_map_body is not None:
            if self.state == "PREFLIGHT":
                self._pre_fixes.append((stamp, out.T_map_body))
            else:
                r = self.aligner.add_fix(stamp, out.T_map_body, {"n_inliers": out.n_inliers, "method": out.method})
                if isinstance(r, dict) and r.get("accepted"):
                    self._fixes_since += 1
                elif isinstance(r, dict):
                    self._log("fix_rejected", reason=r.get("reason", ""))

        if self.looming is None or odom is None or self.state not in ("NAV", "RETURN", "ARRIVED", "HOLD"):
            return
        T_map_body = self.aligner.body_pose_map_at(stamp) if self.aligner.valid(stamp) else None
        if self._odom and stamp < self._odom[0].stamp - 0.05:
            return                                # image older than the post-reset odom: no consistent attitude for it
        lo = self.looming.process(image, stamp, odom.T_odom_body, np.asarray(odom.v_odom, float), T_map_body,
                                  reset_counter=odom.reset_counter)
        self.last_looming = lo
        if lo.emergency:
            if self._emergency_t is None or stamp - self._emergency_t > 0.5:
                self._emergencies += 1
                self._log("emergency", min_ttc_s=round(float(lo.min_ttc_s), 2), reason=lo.reason)
            self._emergency_t = stamp
            self._need_replan = True
        obs = list(lo.obstacles or [])
        kept = []
        for ob in obs:
            if ob.source == "looming_raw" and self._contains_mapped_surface(ob):
                # the map-free emergency path does not know the map, so its box often just IS a mapped wall the
                # Scout was closing on. Stored, it could swallow the goal for good (sim round_trip: a raw box on
                # wall_x0 covered the pad hover point -> "goal inside obstacle" -> landed off the pad). The hover
                # itself already happened; only the memory entry is dropped (DN-62).
                self._log("obstacle_explained_by_map", center=[round(float(v), 2) for v in ob.center])
                continue
            kept.append(ob)
        if kept and self.memory is not None:
            self.memory.update(kept, stamp)
            for ob in kept:
                self._log("obstacle", source=ob.source, center=[round(float(v), 2) for v in ob.center],
                          half_size=[round(float(v), 2) for v in ob.half_size], top_seen=bool(ob.top_seen))

    # ------------------------------------------------------------------------------------------------ tick
    def tick(self, now: float) -> None:
        self._now = float(now)
        if self._t0 is None:
            self._t0 = self._state_t = self._now
            self._log("start")
        self._add_odom(self.vehicle.odometry())
        st = self.vehicle.status()
        if self.state in TERMINAL:
            return

        # the FC or the pilot took the aircraft: stop commanding
        if st.failsafe and self.state not in ("LAND",):
            self._to("ABORTED", "vehicle failsafe")
            return
        if self._offboard_seen and not st.offboard and self.state not in ("LAND",):
            self._to("ABORTED", "vehicle left offboard (pilot or FC)")
            return
        if st.offboard:
            self._offboard_seen = True
        if self._now - self._t0 > self.cfg["max_mission_s"] and self.state in AIRBORNE:
            self._to("LAND", "mission time limit")

        handler = getattr(self, "_s_" + self.state.lower())
        handler(st)
        if self._stream and self.state not in TERMINAL and self.state != "LAND":
            if self.last_setpoint is None or self.last_setpoint.stamp != self._now:
                self._send_hold()

    # ------------------------------------------------------------------------------------------------ states
    def _s_preflight(self, st):
        c = self.cfg
        if not self._odom:
            if self._now - self._state_t > c["preflight_timeout_s"]:
                self._to("REFUSED", "no odometry from the flight controller")
            return
        if st.armed:
            self._to("REFUSED", "armed before preflight finished")
            return
        if self.goal is None and c["mode"] == "wall_goal":
            try:
                self.goal = self._make_goal(self.goal_point)
            except Exception as e:  # noqa: BLE001 - every reason goes to the log
                self._to("REFUSED", f"goal: {e}")
                return
            self._log("goal", position=[round(float(v), 3) for v in self.goal.position],
                      yaw_deg=round(math.degrees(self.goal.yaw), 1), target=[float(v) for v in self.goal.target])
        if self.aligner.T_map_odom() is None:
            rest = float(self.T_map_pad[2, 3]) - self._pad_top_z()
            if not (-0.05 <= rest <= float(c["pad_max_rest_height_m"])):
                self._to("REFUSED", "takeoff pad pose is %.2f m above the mapped surface under it (expected 0..%.2f): "
                         "pad pose and map disagree" % (rest, float(c["pad_max_rest_height_m"])))
                return
            self.aligner.init_known_pose(self.T_map_pad, self._odom[-1].stamp)
            self._log("align_init_pad")
        n = int(c["preflight_fixes"])
        if len(self._pre_fixes) >= n:
            recent = self._pre_fixes[-n:]
            offs = [(float(np.linalg.norm(T[:3, 3] - self.T_map_pad[:3, 3])),
                     abs(math.degrees(_wrap(_yaw_of(T) - _yaw_of(self.T_map_pad))))) for _, T in recent]
            bad = [o for o in offs if o[0] > c["preflight_max_offset_m"] or o[1] > c["preflight_max_offset_deg"]]
            if len(bad) == n:
                self._to("REFUSED", "not on the takeoff pad: PnP says %.2f m / %.1f deg away"
                         % (offs[-1][0], offs[-1][1]))
                return
            if not bad:
                for stamp, T in recent:
                    self.aligner.add_fix(stamp, T, {"phase": "preflight"})
                self._log("preflight_fix_ok", offset_m=round(offs[-1][0], 3), offset_deg=round(offs[-1][1], 1))
                self._start_arming()
                return
        if self._now - self._state_t > c["preflight_timeout_s"]:
            if c["preflight_require_fix"]:
                self._to("REFUSED", "no consistent PnP fix on the pad")
            else:
                self._log("preflight_no_fix", note="taking off on the known pad pose alone")
                self._start_arming()

    def _start_arming(self):
        self._stream, self._stream_t0 = True, self._now
        self._to("ARMING")

    def _s_arming(self, st):
        c = self.cfg
        self._send_hold()
        if self._now - self._stream_t0 < c["stream_before_offboard_s"]:
            return
        if not st.offboard:
            self.vehicle.request_offboard()
        if st.offboard and not st.armed and c["auto_arm"]:
            self.vehicle.request_arm()
        if st.offboard and st.armed:
            self._takeoff_z0 = None
            self._to("TAKEOFF")
            return
        if self._now - self._state_t > c["arm_timeout_s"]:
            self._to("ABORTED", "not armed / not in offboard within arm_timeout_s")

    def _s_takeoff(self, st):
        c = self.cfg
        T, _ = self._pose_map()
        if T is None:
            self._send_hold()
            return
        z_target = self._takeoff_target_z()
        if getattr(self, "_takeoff_z0", None) is None:
            self._takeoff_z0 = float(T[2, 3])
            self._tk_z, self._tk_t = self._takeoff_z0, self._now
        # climb ramp that decelerates into the target (v <= sqrt(2 a d)): a ramp that stops at full speed overshot
        # by ~14 cm in the sim hover test, because PX4 follows the feed-forward until the setpoint stops
        dt = max(0.0, self._now - self._tk_t)
        self._tk_t = self._now
        rem = max(0.0, z_target - self._tk_z)
        v = min(float(c["takeoff_speed_mps"]), math.sqrt(2.0 * float(c["takeoff_decel_mps2"]) * rem))
        self._tk_z = min(z_target, self._tk_z + max(v, 0.02) * dt)
        if z_target - self._tk_z < 0.005:
            self._tk_z = z_target
        z_sp = self._tk_z
        p = self.T_map_pad[:3, 3].copy()
        p[2] = z_sp
        self._send_map(p, _yaw_of(self.T_map_pad))
        if z_sp >= z_target and abs(T[2, 3] - z_target) <= c["takeoff_tol_m"]:
            self._fixes_since = 0
            if c["mode"] == "hover_test":
                self._to("HOVER", "hover height reached")
            else:
                self._to("SETTLE", "takeoff altitude reached")

    def _s_hover(self, st):
        """hover_test: hold the pad's x, y at the climb height (map setpoint while the alignment is valid, else the
        FC's own odom position), then land on the pad."""
        c = self.cfg
        if self.aligner.valid(self._now):
            p = self.T_map_pad[:3, 3].copy()
            p[2] = self._takeoff_target_z()
            self._send_map(p, _yaw_of(self.T_map_pad))
        else:
            self._send_hold()
        if self._now - self._state_t >= float(c["hover_test"]["hover_s"]):
            self._to("LAND", "hover test: %.1f s hover done" % float(c["hover_test"]["hover_s"]))

    def _s_settle(self, st):
        c = self.cfg
        self._send_hold()
        if self._fixes_since >= int(c["settle_fixes"]) and self.aligner.valid(self._now):
            self._need_replan = True
            self._to("NAV")
        elif self._now - self._state_t > c["settle_timeout_s"]:
            self._to("LAND", "localization did not settle after takeoff")

    def _s_nav(self, st):
        self._fly_to(self.goal, arrive_state="ARRIVED")

    def _s_return(self, st):
        self._fly_to(self.return_goal, arrive_state="LAND", land_on_arrival=True)

    def _s_arrived(self, st):
        c = self.cfg
        if not self.aligner.valid(self._now):
            self._to("HOLD", "alignment invalid")
            return
        self._send_map(self.goal.position, self.goal.yaw)
        if self._now - self._state_t >= c["dwell_s"]:
            mode = c["on_arrival"]
            if mode == "land":
                self._to("LAND", "arrived; on_arrival=land")
            elif mode == "return_land":
                pad = self.T_map_pad[:3, 3].copy()
                pad[2] = self._hover_z_above_pad()
                self.return_goal = _SimpleGoal(pad, _yaw_of(self.T_map_pad))
                self._need_replan = True
                self._to("RETURN", "arrived; returning to the pad")
            elif self._now - self._state_t >= c["dwell_s"] and not getattr(self, "_dwell_logged", False):
                self._dwell_logged = True
                self._log("dwell_done", note="holding at the goal (on_arrival=hold)")

    def _s_hold(self, st):
        c = self.cfg
        self._send_hold()
        if self.aligner.valid(self._now):
            self._align_invalid_since = None
            self._need_replan = True
            self._to(self._prev_state if self._prev_state in ("NAV", "RETURN", "ARRIVED") else "NAV",
                     "alignment valid again")
            return
        if self._align_invalid_since is None:
            self._align_invalid_since = self._state_t
        if self._now - self._align_invalid_since > c["loc_lost_abort_s"]:
            self._to("LAND", "localization lost too long: landing in place")

    def _s_land(self, st):
        if not self._land_requested:
            self.vehicle.request_land()
            self._land_requested = True
            self._log("land_requested")
        if st.landed and not st.armed:
            self._to("DONE", "landed and disarmed")

    # ------------------------------------------------------------------------------------------------ flying
    def _make_goal(self, point_b):
        if point_b is None:
            raise ValueError("no goal point B")
        if self._wall_goal_fn is not None:
            return self._wall_goal_fn(self.static_map, point_b)
        from .goal import wall_goal
        return wall_goal(self.static_map, point_b, getattr(self.planner, "cfg", None))

    def _fly_to(self, goal, arrive_state, land_on_arrival=False):
        c = self.cfg
        if not self.aligner.valid(self._now):
            self._to("HOLD", "alignment invalid")
            self._send_hold()
            return
        T, v = self._pose_map()
        if T is None:
            self._send_hold()
            return
        emergency = self._emergency_t is not None and (self._now - self._emergency_t) < float(
            getattr(self.safety, "emergency_hover_s", 1.0))
        obstacles = self.memory.active(self._now) if self.memory is not None else []

        version = getattr(self.memory, "version", None)
        if version != self._mem_version:
            self._mem_version = version
            self._need_replan = True
        if self._plan_t is not None and self._now - self._plan_t >= c["replan_period_s"]:
            self._need_replan = True
        if self._plan is not None and self._plan.ok:
            d = _dist_to_path(T[:3, 3], self._plan.waypoints)
            if d > c["replan_deviation_m"]:
                self._need_replan = True
                self._log("off_path", dist_m=round(d, 2))

        if self._need_replan and not emergency:
            self._replan(T, goal, obstacles)

        if emergency or self._plan is None or not self._plan.ok:
            # no setpoint: the gate latches its hold (DN-55). Passing the current pose instead made the setpoint
            # follow the vehicle with the velocity feed-forward of its own motion: a plan failure kept the drift
            # going (sim low_box: 0.24 m/s east for 12 s, over crate_c, up into the ceiling)
            p_sp, yaw_sp = None, _yaw_of(T)
            done = False
            # and the follower's reference comes back onto the body: replanning from where it had led the vehicle
            # (0.45 m ahead, toward the obstacle that stopped it) flew the Scout into that box (sim late_obstacle)
            if hasattr(self.follower, "hold_at"):
                self.follower.hold_at(T)
        else:
            p_sp, yaw_sp, done, _info = self.follower.step(self._now, T, v, obstacles)
        p_f, yaw_f, action, why = self.safety.filter(self._now, p_sp, yaw_sp, T, True, emergency)
        action = str(action).lower()
        if action == "hold":
            self._send_hold()
        else:
            self._send_map(p_f, yaw_f)
            if action not in ("pass", "ok") and why:
                self._log("safety", action=action, reason=why)

        if self._plan is None or not self._plan.ok:
            if self._no_path_since is None:
                self._no_path_since = self._now
            elif self._now - self._no_path_since > c["no_path_land_s"]:
                self._to("LAND", "no path for no_path_land_s")
            return
        self._no_path_since = None
        if done and not emergency:
            if land_on_arrival:
                self._to("LAND", "above the pad")
            else:
                self._to(arrive_state, "goal reached")

    def _replan(self, T, goal, obstacles):
        self._need_replan = False
        self._plan_t = self._now
        start = T[:3, 3].copy()
        ref = getattr(self.follower, "reference", None)
        ref = ref() if callable(ref) else ref
        if ref is not None and float(np.linalg.norm(ref[0] - start)) <= 0.5:
            start = np.asarray(ref[0], float).copy()     # plan from the follower's reference: no setpoint jump
        res = self.planner.plan(start, goal, obstacles)
        changed = (self._plan is None or not self._plan.ok or not res.ok
                   or res.shape != self._plan.shape or len(res.waypoints) != len(self._plan.waypoints)
                   or float(np.max(np.abs(res.waypoints[-1] - self._plan.waypoints[-1]))) > 1e-6
                   or _path_moved(res.waypoints, self._plan.waypoints) > 0.15)
        self._plan = res
        if not res.ok:
            self._log("plan_failed", reason=res.reason)
            return
        if changed or not getattr(self.follower, "has_path", True):     # a hold dropped the follower's plan
            self.follower.set_path(res.waypoints, goal.yaw)
            self.plan_shapes.append(res.shape)
            self._log("plan", shape=res.shape, n_wp=int(len(res.waypoints)), cost=round(float(res.cost), 2),
                      n_obstacles=len(obstacles),
                      waypoints=[[round(float(v), 2) for v in w] for w in res.waypoints])

    # ------------------------------------------------------------------------------------------------ summary
    def summary(self) -> dict:
        return {"mode": self.cfg["mode"], "state": self.state, "reason": self.reason, "emergencies": self._emergencies,
                "plan_shapes": list(self.plan_shapes),
                "obstacles": 0 if self.memory is None else len(self.memory.active(self._now or 0.0)),
                "goal": None if self.goal is None else {"position": [float(v) for v in self.goal.position],
                                                        "yaw": float(self.goal.yaw)}}


class _SimpleGoal:
    """A hover pose (RETURN's target) with the GoalPose fields the planner and follower read."""

    def __init__(self, position, yaw):
        self.position = np.asarray(position, float)
        self.yaw = float(yaw)
        self.target = self.position.copy()
        self.normal = np.array([math.cos(yaw), math.sin(yaw), 0.0])


def _dist_to_path(p, wps) -> float:
    if wps is None or len(wps) == 0:
        return float("inf")
    if len(wps) == 1:
        return float(np.linalg.norm(p - wps[0]))
    best = float("inf")
    for a, b in zip(wps[:-1], wps[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 < 1e-12 else min(1.0, max(0.0, float((p - a) @ ab) / L2))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def _path_moved(new, old) -> float:
    """Largest distance from a new waypoint to the old path (0 when the plan is effectively unchanged)."""
    return max(_dist_to_path(w, old) for w in new)

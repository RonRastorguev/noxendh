"""`SimVehicle`: a PX4-like multicopter + flight-controller model behind navigation.vehicle.VehicleInterface.

What it models (all numbers config, see DEFAULTS / config/scout_sim.yaml; FC limits from docs/SCOUT_NAV.md §3):
- Position control the PX4 way: P on the ESTIMATED position (MPC_XY_P 0.95, MPC_Z_P 1.0) + optional velocity
  feed-forward -> velocity caps (1.5 xy, 1.0 up, 0.7 down) -> first-order velocity response (tau 0.25 s) ->
  acceleration limit (3 m/s2). Because the controller closes the loop on its own estimate, estimator drift moves the
  TRUE vehicle, exactly as on the real aircraft.
- Attitude: roll / pitch follow the horizontal acceleration (thrust vector = a + g), lagged by att_tau_s, so the
  forward camera pitches and rolls like a real multicopter; yaw follows the yaw setpoint at <= 60 deg/s.
- Ground: sits on whatever is below until armed AND commanded upward; landed detector; land mode (descend at
  0.6 m/s, disarm `disarm_after_land_s` after touchdown); auto-disarm if armed and never taken off
  (COM_DISARM_PRFLT-like).
- Offboard rules: request_offboard is accepted only after >= offboard_min_stream_s of continuously streamed setpoints;
  no setpoint for offboard_loss_s (COM_OF_LOSS_T 1 s) -> Position mode, holding the current estimate, failsafe flag
  set (COM_OBL_RC_ACT 0). request_arm only in offboard (or a simulated pilot arms, `pilot_arm_after_s`).
- The ESTIMATOR (what odometry() returns): an odom frame with its own origin and yaw at power-on (not the map), a
  Gauss-Markov x/y velocity bias + flow scale error, gyro yaw drift (constant rate + random walk), the RANGE height
  reference: odom z = height of the body above the surface directly below in the TRUE world (floor or a box top),
  following a terrain step with a first-order lag hgt_step_tau_s; optional local-frame reset (reset_counter + a known
  delta in status().info["last_reset"], as PX4 publishes it); dist_bottom_m from the rangefinder.
- Collision: the hull (vertical cylinder of airframe.hull_radius_m from the gear to +hull_half_height_m, or a sphere)
  against the TRUE world at every physics step; min clearance tracked continuously (DN-75).
"""
from __future__ import annotations

import math

import numpy as np

from localization.geometry import inv_T, make_T, rot_x, rot_y, rot_z
from navigation.types import OdomSample, Setpoint, VehicleStatus

G = 9.81

DEFAULTS = {
    "rate_hz": 100.0,
    "control": {
        "mpc_xy_p": 0.95, "mpc_z_p": 1.0,
        "xy_vel_max": 1.5, "z_vel_max_up": 1.0, "z_vel_max_dn": 0.7,
        "vel_tau_s": 0.25,                 # provisional: first-order closed-loop velocity response
        "acc_max_xy": 3.0, "acc_max_z": 3.0,
        "att_tau_s": 0.08,                 # provisional: tilt follows the acceleration with this lag
        "max_tilt_deg": 35.0,
        "yaw_p": 2.8, "yaw_rate_max_dps": 60.0,
        "land_speed_mps": 0.6,
        "liftoff_vz_min": 0.05,            # commanded climb rate that lifts the vehicle off the ground
    },
    "fc": {
        "offboard_min_stream_s": 1.0,      # PX4: setpoints must be streaming before offboard is accepted
        "offboard_stream_max_gap_s": 0.5,  # a gap longer than this restarts the stream count
        "offboard_loss_s": 1.0,            # COM_OF_LOSS_T
        "landed_hold_s": 0.3,
        "disarm_after_land_s": 2.0,        # COM_DISARM_LAND (PX4 default 2 s)
        "preflight_disarm_s": 10.0,        # COM_DISARM_PRFLT (PX4 default 10 s)
        "arm_requires_offboard": True,
        "pilot_arm_after_s": None,         # a simulated pilot arms this long after offboard engaged (None = never)
    },
    "estimator": {
        "init_offset_sigma_m": 0.5,        # odom origin vs the power-on position (the map is never the odom frame)
        "init_yaw": "random",              # "random" (uniform) | "heading" (odom x = power-on heading, PX4 no-mag)
        "bias_sigma_mps": 0.01,            # stationary sigma of the x/y velocity bias (Gauss-Markov), provisional
        "bias_tau_s": 30.0,
        "vel_scale_sigma": 0.02,           # flow scale-factor error, provisional
        "yaw_rate_bias_sigma_dps": 0.008,  # ~0.5 deg/min constant gyro yaw drift, provisional
        "yaw_rw_dps_sqrt_s": 0.005,
        "hgt_step_tau_s": 0.3,             # NOT KNOWN for PX4 (docs/SCOUT_NAV.md §3): lag following a terrain step
        "hgt_noise_m": 0.003,              # rangefinder noise on dist_bottom
        "range_max_m": 6.0,
        "reset_at_s": None,                # local-frame reset at this sim time (None = never)
        "reset_delta": [0.3, -0.2, 0.0, 5.0],   # dx, dy, dz (m), dyaw (deg) of the odom origin at the reset
    },
    "hull": {"shape": "cylinder", "end_on_collision": True, "below_track_min_speed_mps": 0.1},
    "battery_v": 16.4,
}


def _merge(base: dict, over: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(T) -> float:
    return math.atan2(T[1, 0], T[0, 0])


class SimVehicle:
    """Implements navigation.vehicle.VehicleInterface. Advance it with step(dt) (or step_to(t))."""

    def __init__(self, world, T_map_body0, cfg=None, seed=0, hw=None):
        self.world = world
        self.cfg = _merge(DEFAULTS, cfg)
        self.hw = hw or world.hw
        ss = np.random.SeedSequence([int(seed), 7177])
        self.rng, self.rng_meas = (np.random.default_rng(x) for x in ss.spawn(2))   # estimator drift | sensor noise
        c, e = self.cfg["control"], self.cfg["estimator"]
        self.dt = 1.0 / float(self.cfg["rate_hz"])
        self.hull_r = float(self.hw["airframe"]["hull_radius_m"])
        self.hull_above = float(self.hw["airframe"]["hull_half_height_m"])
        self.rest_h = float(self.hw["derived"]["fc_height_at_rest_m"])
        T0 = np.asarray(T_map_body0, float)
        self.t = 0.0
        self.p = T0[:3, 3].copy()
        self.p[2] = world.terrain_z(self.p[0], self.p[1], self.p[2]) + self.rest_h
        self.v = np.zeros(3)
        self.a_att = np.zeros(2)
        self.yaw = yaw_of(T0)
        self.roll = self.pitch = 0.0
        # FC state
        self.armed = False
        self.mode = "position"
        self.on_ground = True
        self.landed = True
        self._ground_since = 0.0
        self.has_flown = False
        self.failsafe = False
        self.crashed = False
        self.armed_at = None
        self._offboard_since = None
        self._hold = None                      # (p_odom (3,), yaw_odom)
        self._sp: Setpoint | None = None
        self._sp_t = None
        self._stream_start = None
        self.events: list = []
        # estimator
        self.psi0 = float(self.rng.uniform(-math.pi, math.pi)) if e["init_yaw"] == "random" else -self.yaw
        self.dpsi = 0.0
        self.yaw_rate_bias = math.radians(float(self.rng.normal(0, e["yaw_rate_bias_sigma_dps"])))
        self.bias = np.zeros(2)
        self.scale = 1.0 + float(self.rng.normal(0, e["vel_scale_sigma"]))
        self.p_odom_xy = self.rng.normal(0, e["init_offset_sigma_m"], 2)
        self.z_odom = self._height_true()
        self.vz_odom = 0.0
        self.reset_counter = 0
        self.last_reset = None
        self._reset_done = False
        # collision bookkeeping (continuous, at the physics rate)
        self.min_lateral = math.inf
        self.min_lateral_what = ""
        self.min_below_airborne = math.inf        # gear above the surface below, airborne, outside land mode
        self.collisions: list = []
        self._in_contact = False
        self.last_clearance = {}
        self.max_speed_xy = 0.0
        self.max_tilt_deg = 0.0
        self._event("power_on", odom_yaw_offset_deg=math.degrees(self.psi0), scale=self.scale)

    # ------------------------------------------------------------------------------------------------ helpers
    def _event(self, kind, **kw):
        ev = {"t": round(self.t, 4), "src": "vehicle", "event": kind}
        ev.update(kw)
        self.events.append(ev)

    def _height_true(self) -> float:
        return float(self.p[2] - self.world.terrain_z(self.p[0], self.p[1], self.p[2] - self.rest_h + 1e-6))

    def _yaw_odom(self) -> float:
        return wrap(self.yaw + self.psi0 + self.dpsi)

    def _R_e(self) -> np.ndarray:
        a = self.psi0 + self.dpsi
        c, s = math.cos(a), math.sin(a)
        return np.array([[c, -s], [s, c]])

    def _p_odom(self) -> np.ndarray:
        return np.array([self.p_odom_xy[0], self.p_odom_xy[1], self.z_odom])

    def _v_odom(self) -> np.ndarray:
        vxy = self._R_e() @ (self.scale * self.v[:2]) + (self.bias if not self.on_ground else 0.0)
        return np.array([vxy[0], vxy[1], self.vz_odom])

    def _stream_ok(self) -> bool:
        f = self.cfg["fc"]
        return (self._sp_t is not None and self.t - self._sp_t <= f["offboard_stream_max_gap_s"]
                and self.t - self._stream_start >= f["offboard_min_stream_s"] - 1e-9)

    def _set_hold(self):
        self._hold = (self._p_odom(), self._yaw_odom())

    # --------------------------------------------------------------------------------------- VehicleInterface
    def odometry(self) -> OdomSample:
        R = rot_z(self._yaw_odom()) @ rot_y(self.pitch) @ rot_x(self.roll)
        h = self._height_true()
        e = self.cfg["estimator"]
        db = h + float(self.rng_meas.normal(0, e["hgt_noise_m"])) if h <= e["range_max_m"] else None
        return OdomSample(stamp=self.t, T_odom_body=make_T(R, self._p_odom()), v_odom=self._v_odom(),
                          reset_counter=self.reset_counter, dist_bottom_m=db)

    def status(self) -> VehicleStatus:
        return VehicleStatus(stamp=self.t, armed=self.armed, offboard=self.mode == "offboard", landed=self.landed,
                             failsafe=self.failsafe, battery_v=float(self.cfg["battery_v"]),
                             info={"mode": self.mode, "last_reset": self.last_reset, "crashed": self.crashed,
                                   "stream_ok": self._stream_ok()})

    def send_setpoint(self, sp: Setpoint) -> None:
        f = self.cfg["fc"]
        if self._sp_t is None or self.t - self._sp_t > f["offboard_stream_max_gap_s"]:
            self._stream_start = self.t
        self._sp, self._sp_t = sp, self.t

    def request_offboard(self) -> None:
        if self.crashed:
            return self._event("offboard_refused", reason="crashed")
        if not self._stream_ok():
            return self._event("offboard_refused", reason="setpoint stream not established")
        if self.mode != "offboard":
            self.mode = "offboard"
            self.failsafe = False
            self._offboard_since = self.t
            self._event("offboard_engaged")

    def request_arm(self) -> None:
        if self.armed:
            return
        reason = None
        if self.crashed:
            reason = "crashed"
        elif not self.landed:
            reason = "not landed"
        elif self.cfg["fc"]["arm_requires_offboard"] and self.mode != "offboard":
            reason = "not in offboard"
        if reason:
            return self._event("arm_refused", reason=reason)
        self._arm("request")

    def _arm(self, by):
        self.armed = True
        self.armed_at = self.t
        self._event("armed", by=by)

    def _disarm(self, why):
        self.armed = False
        self._event("disarmed", reason=why)

    def request_land(self) -> None:
        if not self.armed:
            return self._event("land_ignored", reason="disarmed")
        if self.mode != "land":
            self.mode = "land"
            self._set_hold()
            self._event("land_mode")

    def capabilities(self) -> dict:
        c, e = self.cfg["control"], self.cfg["estimator"]
        return {"height_ref": "range", "max_xy_vel_mps": c["xy_vel_max"], "max_z_vel_up_mps": c["z_vel_max_up"],
                "max_z_vel_dn_mps": c["z_vel_max_dn"], "max_yaw_rate_dps": c["yaw_rate_max_dps"],
                "offboard_loss_s": self.cfg["fc"]["offboard_loss_s"], "range_max_m": e["range_max_m"],
                "reset_delta_published": True}

    # ------------------------------------------------------------------------------------------------ truth
    def T_map_body_true(self) -> np.ndarray:
        return make_T(rot_z(self.yaw) @ rot_y(self.pitch) @ rot_x(self.roll), self.p)

    def T_map_odom_true(self) -> np.ndarray:
        """The TRUE map<-odom transform right now (4-DoF: the estimator's attitude error is yaw only). Its z part is
        terrain + height-lag error under the range height reference, so compare x, y, yaw."""
        R = rot_z(self._yaw_odom()) @ rot_y(self.pitch) @ rot_x(self.roll)
        return self.T_map_body_true() @ inv_T(make_T(R, self._p_odom()))

    # ------------------------------------------------------------------------------------------------ physics
    def step_to(self, t_end: float) -> None:
        while self.t < t_end - 1e-9:
            self.step()

    def _vel_setpoint(self):
        """-> (v_sp in odom (3,), yaw_sp odom) from the active mode."""
        c = self.cfg["control"]
        p = self._p_odom()
        if self.mode == "offboard" and self._sp is not None:
            target, yaw_sp = np.asarray(self._sp.position, float), float(self._sp.yaw)
            ff = np.zeros(3) if self._sp.velocity_ff is None else np.asarray(self._sp.velocity_ff, float)
        else:
            if self._hold is None:
                self._set_hold()
            target, yaw_sp = self._hold
            ff = np.zeros(3)
        e = target - p
        v = np.array([c["mpc_xy_p"] * e[0], c["mpc_xy_p"] * e[1], c["mpc_z_p"] * e[2]]) + ff
        if self.mode == "land":
            v[2] = -c["land_speed_mps"]
        n = math.hypot(v[0], v[1])
        if n > c["xy_vel_max"]:
            v[:2] *= c["xy_vel_max"] / n
        v[2] = min(max(v[2], -c["z_vel_max_dn"]), c["z_vel_max_up"])
        return v, yaw_sp

    def step(self, dt: float | None = None) -> None:
        dt = self.dt if dt is None else float(dt)
        c, f, e = self.cfg["control"], self.cfg["fc"], self.cfg["estimator"]
        if self.crashed:
            self.t += dt
            return
        # --- FC mode logic --------------------------------------------------------------------------------
        if self.mode == "offboard" and (self._sp_t is None or self.t - self._sp_t > f["offboard_loss_s"]):
            self.mode = "position"
            self.failsafe = True
            self._set_hold()
            self._event("offboard_loss", setpoint_age_s=None if self._sp_t is None else round(self.t - self._sp_t, 3))
        if (not self.armed and f["pilot_arm_after_s"] is not None and self.mode == "offboard" and self.landed
                and self._offboard_since is not None and self.t - self._offboard_since >= f["pilot_arm_after_s"]):
            self._arm("pilot")
        if self.armed and self.landed:
            if self.has_flown and self.t - self._ground_since >= f["disarm_after_land_s"]:
                self._disarm("landed")
            elif not self.has_flown and self.t - self.armed_at >= f["preflight_disarm_s"]:
                self._disarm("preflight timeout (never took off)")
        # --- local-frame reset ------------------------------------------------------------------------------
        if e["reset_at_s"] is not None and not self._reset_done and self.t >= e["reset_at_s"]:
            dx, dy, dz, dyaw = e["reset_delta"]
            self.p_odom_xy = self.p_odom_xy + np.array([dx, dy])
            self.z_odom += dz
            self.dpsi += math.radians(dyaw)
            if self._hold is not None:
                self._hold = (self._hold[0] + np.array([dx, dy, dz]), wrap(self._hold[1] + math.radians(dyaw)))
            self.reset_counter += 1
            self._reset_done = True
            self.last_reset = {"stamp": self.t, "delta_xy": [dx, dy], "delta_z": dz, "delta_yaw_rad": math.radians(dyaw),
                               "reset_counter": self.reset_counter}
            self._event("local_frame_reset", **self.last_reset)
        # --- control -> true motion -------------------------------------------------------------------------
        v_sp, yaw_sp = self._vel_setpoint() if self.armed else (np.zeros(3), self._yaw_odom())
        if self.on_ground:
            if self.armed and v_sp[2] > c["liftoff_vz_min"]:
                self.on_ground = False
                self.landed = False
                if not self.has_flown:
                    self._event("liftoff")
                else:
                    self._event("liftoff", again=True)
                self.has_flown = True
            else:
                self.v[:] = 0.0
                self.a_att[:] = 0.0
                self.roll = self.pitch = 0.0
        p_prev_z = self.p[2]
        if not self.on_ground:
            if not self.armed:                    # disarmed in the air: it falls
                a = np.array([0.0, 0.0, -G])
            else:
                vt_xy = self._R_e().T @ (v_sp[:2] - self.bias) / self.scale
                vt = np.array([vt_xy[0], vt_xy[1], v_sp[2]])
                a = (vt - self.v) / c["vel_tau_s"]
                n = math.hypot(a[0], a[1])
                if n > c["acc_max_xy"]:
                    a[:2] *= c["acc_max_xy"] / n
                a[2] = min(max(a[2], -c["acc_max_z"]), c["acc_max_z"])
                yr = math.radians(c["yaw_rate_max_dps"])
                rate = min(max(c["yaw_p"] * wrap(yaw_sp - self._yaw_odom()), -yr), yr)
                self.yaw = wrap(self.yaw + rate * dt)
            self.v += a * dt
            self.p += self.v * dt
            # attitude from the horizontal acceleration
            self.a_att += (a[:2] - self.a_att) * min(1.0, dt / c["att_tau_s"])
            cy, sy = math.cos(self.yaw), math.sin(self.yaw)
            fwd = self.a_att[0] * cy + self.a_att[1] * sy
            left = -self.a_att[0] * sy + self.a_att[1] * cy
            mt = math.radians(c["max_tilt_deg"])
            self.pitch = min(max(math.atan2(fwd, G), -mt), mt)
            self.roll = min(max(-math.atan2(left, G), -mt), mt)
            # ground contact (floor or a box top)
            top = self.world.terrain_z(self.p[0], self.p[1], p_prev_z - self.rest_h + 1e-6)
            if self.p[2] - self.rest_h <= top:
                vz_td = float(self.v[2])
                self.p[2] = top + self.rest_h
                self.v[:] = 0.0
                self.a_att[:] = 0.0
                self.roll = self.pitch = 0.0
                self.on_ground = True
                self._ground_since = self.t + dt
                self._event("touchdown", vz=round(vz_td, 3), mode=self.mode, on=top)
                if self.mode != "land":
                    self._collision("touchdown outside land mode", -0.0, {"vz": vz_td, "surface_z": top})
            spd = math.hypot(self.v[0], self.v[1])
            self.max_speed_xy = max(self.max_speed_xy, spd)
            self.max_tilt_deg = max(self.max_tilt_deg, math.degrees(max(abs(self.roll), abs(self.pitch))))
        if self.on_ground and not self.landed and self.t + dt - self._ground_since >= f["landed_hold_s"]:
            self.landed = True
            self._event("landed")
        # --- estimator ----------------------------------------------------------------------------------------
        self.dpsi += self.yaw_rate_bias * dt + math.radians(e["yaw_rw_dps_sqrt_s"]) * math.sqrt(dt) * self.rng.normal()
        if not self.on_ground:
            tb = e["bias_tau_s"]
            self.bias += -self.bias * dt / tb + e["bias_sigma_mps"] * math.sqrt(2 * dt / tb) * self.rng.normal(size=2)
            self.p_odom_xy = self.p_odom_xy + (self._R_e() @ (self.scale * self.v[:2]) + self.bias) * dt
        h = self._height_true()
        z_prev = self.z_odom
        tau = e["hgt_step_tau_s"]
        if tau <= dt:
            self.z_odom = h
        else:
            self.z_odom += self.v[2] * dt + (h - self.z_odom) * dt / tau
        self.vz_odom = (self.z_odom - z_prev) / dt
        self.t += dt
        # --- collision check against the TRUE world -------------------------------------------------------------
        cl = self.world.hull_clearance(self.p, self.hull_r, self.rest_h, self.hull_above, self.cfg["hull"]["shape"])
        if not self.on_ground:
            if cl["lateral"] < self.min_lateral:
                self.min_lateral, self.min_lateral_what = cl["lateral"], cl["lateral_what"]
            # height above what is below, while moving horizontally outside land mode (takeoff climbs and land-mode
            # descents legitimately reach 0; flying over a box does not)
            if self.mode != "land" and math.hypot(self.v[0], self.v[1]) > self.cfg["hull"]["below_track_min_speed_mps"]:
                self.min_below_airborne = min(self.min_below_airborne, cl["below"])
        if cl["lateral"] < -1e-4:
            if not self._in_contact:
                self._collision("hull penetrates " + cl["lateral_what"], cl["lateral"], {})
            self._in_contact = True
        else:
            self._in_contact = False
        self.last_clearance = cl

    def _collision(self, what, depth, info):
        rec = {"t": round(self.t, 3), "what": what, "depth_m": float(depth), "p": [round(float(v), 3) for v in self.p]}
        rec.update(info)
        self.collisions.append(rec)
        self._event("collision", **rec)
        if self.cfg["hull"]["end_on_collision"]:
            self.crashed = True
            self.armed = False
            self.mode = "position"
            self._event("crashed")

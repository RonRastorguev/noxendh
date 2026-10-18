"""Safety gate: every map setpoint passes through it before the map->odom conversion (docs/SCOUT_NAV.md §6; DN-55).

    gate = SafetyGate(cfg=None, static_map)
    gate.filter(now, p_sp, yaw_sp, T_map_body, align_valid, emergency, band_check=True)
        -> (p_out (3,), yaw_out, action, reason)
    gate.emergency_active(now) -> bool, gate.reset()
    action: "PASS" (unchanged) | "CLAMP" (moved into the limits; reason lists which) | "HOLD"

HOLD (setpoint = the body position and yaw LATCHED when the hold began, so a drifting vehicle is pulled back
instead of the hold creeping with it) while any of:
  - p_sp is None (no plan)                                                    reason "no_plan"
  - align_valid is False (map pose unusable)                                  reason "align_invalid"
  - an emergency was reported within the last emergency_hover_s (flight_demo) reason "emergency"
The latch is released when all clear; the next hold latches afresh.
Otherwise, in order: flight band (floor + min_alt_m .. ceiling - ceiling_margin_m; skipped with band_check=False for
takeoff / landing), geofence (map bounds - geofence_margin_m; flight_demo value, scout_nav default while it is
null), max step from the current pose (max_step_m), setpoint rate (max_sp_rate_mps between ticks).
The gate is independent of the planner and follower: it reads only the static map, the pose and the flags.
"""
from __future__ import annotations

import numpy as np

from .follower import wrap, yaw_of
from .nav_config import shared_defaults

DEFAULTS = dict(shared_defaults())
DEFAULTS.update({
    "max_step_m": 0.6,
    "max_sp_rate_mps": 0.75,
})


class SafetyGate:
    def __init__(self, cfg: dict | None = None, static_map=None):
        if static_map is None:
            raise ValueError("SafetyGate needs the static map (geofence and flight band)")
        c = dict(DEFAULTS)
        c.update(cfg or {})
        self.cfg = c
        self.map = static_map
        self.z_lo, self.z_hi = static_map.band(c)
        self.fence_lo, self.fence_hi = static_map.geofence(c)
        if np.any(self.fence_hi <= self.fence_lo):
            raise ValueError("geofence is empty: map bounds smaller than twice geofence_margin_m")
        self.reset()

    def reset(self):
        self._last_emergency = None
        self._hold = None
        self._last_out = None
        self._last_t = None

    def emergency_active(self, now) -> bool:
        return (self._last_emergency is not None
                and now - self._last_emergency < float(self.cfg["emergency_hover_s"]))

    def filter(self, now, p_sp, yaw_sp, T_map_body, align_valid, emergency, band_check=True):
        T = np.asarray(T_map_body, dtype=np.float64)
        p_body, yaw_body = T[:3, 3].copy(), yaw_of(T)
        if emergency:
            self._last_emergency = float(now)
        why = []
        if p_sp is None:
            why.append("no_plan")
        if not align_valid:
            why.append("align_invalid")
        if self.emergency_active(now):
            why.append("emergency")
        if why:
            if self._hold is None:
                self._hold = (p_body, yaw_body)
            p, y = self._hold
            return self._out(now, p.copy(), y, "HOLD", ",".join(why))
        self._hold = None

        p = np.asarray(p_sp, dtype=np.float64).reshape(3).copy()
        y = float(wrap(yaw_sp))
        clamps = []
        if band_check:
            z = float(np.clip(p[2], self.z_lo, self.z_hi))
            if z != p[2]:
                clamps.append("band")
                p[2] = z
        xy = np.clip(p[:2], self.fence_lo, self.fence_hi)
        if np.any(xy != p[:2]):
            clamps.append("geofence")
            p[:2] = xy
        d = p - p_body
        n = float(np.linalg.norm(d))
        if n > float(self.cfg["max_step_m"]):
            p = p_body + d * (float(self.cfg["max_step_m"]) / n)
            clamps.append("max_step")
        if self._last_out is not None and self._last_t is not None:
            dt = now - self._last_t
            if 0 < dt <= 0.5:
                step = p - self._last_out
                lim = float(self.cfg["max_sp_rate_mps"]) * dt
                ns = float(np.linalg.norm(step))
                if ns > lim:
                    p = self._last_out + step * (lim / ns)
                    clamps.append("sp_rate")
        return self._out(now, p, y, "CLAMP" if clamps else "PASS", ",".join(clamps))

    def _out(self, now, p, y, action, reason):
        self._last_out, self._last_t = p.copy(), float(now)
        return p, float(y), action, reason

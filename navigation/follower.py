"""Setpoint follower: a reference point (the carrot) moving along the plan (docs/SCOUT_NAV.md §6; DN-52..54, 57, 58).

    f = Follower(cfg=None, static_map=None)
    f.set_path(waypoints (N,3) map, goal_yaw)             new plan; the reference continues from where it is (no jump)
    f.clear()                                             drop the plan: step() holds the current reference
    f.hold_at(T_map_body)                                 drop the plan, reference onto the body (holds, DN-59)
    f.has_path                                            a plan is installed or pending
    f.step(now, T_map_body, v_map, obstacles=()) -> (p_sp_map (3,), yaw_sp_map, done, info)
    f.reference -> (p_ref, yaw_ref) | None
    info: mode ("move" | "yaw_first" | "wait" | "final" | "hold" | "done"), s, remaining_m, v (path speed),
          v_ref (3,) feed-forward, v_target, limit (what bounds v_target), d_ahead_m (centre ray), d_corridor_m,
          heading_err_deg

The reference moves ON the polyline (no corner cutting, so the planner's clearances hold) at a path speed v that is
acceleration-limited (|dv/dt| <= max_accel_mps2) and bounded by
  - the per-segment caps: horizontal <= max_speed_mps (flight_demo), vertical <= max_speed_v_mps;
  - braking to every vertex ahead: stop vertices (heading change > corner_stop_deg, horizontal <-> vertical change,
    the goal) to 0, gentle vertices to the slower neighbouring cap: v <= sqrt(v_vertex^2 + 2 a d);
  - the TTC speed law (horizontal motion): v_h <= (hull-contact distance along the horizontal motion direction to the
    first mapped surface or known obstacle - stop_margin_m) / ttc_floor_s, so closing on a mapped wall (incl. the goal
    wall) never reaches the looming emergency TTC (flight_demo avoid.looming_ttc_s < ttc_floor_s). AND the pure
    TTC over the looming emergency's own corridor (DN-57): v_h <= (nearest hit of parallel rays from the body across
    +-(hull_radius_m + corridor_margin_m) - hull_radius_m) / ttc_floor_s. The centre ray alone missed an oblique
    wall that the corridor's edge reaches first (diagonal approach: 1.65-1.85 s true, 1.26 s estimated);
  - LOOK WHERE YOU GO: horizontal motion only while |body heading - look heading| <= max_heading_err_deg, else
    v_target = 0 and the yaw reference turns first (<= max_yaw_rate_dps). Pure vertical segments keep the yaw.
    The look heading is the segment's own, except for a SHORT HOP (horizontal, < short_hop_m, followed by a longer
    horizontal segment): it is flown facing that next segment, sideways, instead of turning the camera onto a hop
    that typically creeps toward a near wall (DN-58).
    In the last final_yaw_dist_m of path the yaw turns to the goal yaw and the heading gate is waived (DN-53), but
    while the Scout still moves the yaw stays within max_heading_err_deg of the motion (DN-53 amended);
  - the lead: v_target = 0 while the reference is more than max_lead_m ahead of the vehicle (it waits for it).
done = reference at the goal, body within arrive_tol_m and arrive_tol_deg of the goal pose, speed <= arrive_max_speed.
"""
from __future__ import annotations

import numpy as np

from .nav_config import shared_defaults

DEFAULTS = dict(shared_defaults())
DEFAULTS.update({
    "max_speed_v_mps": 0.3,
    "max_accel_mps2": 0.5,
    "max_yaw_rate_dps": 45.0,
    "max_heading_err_deg": 20.0,
    "corner_stop_deg": 10.0,
    "final_yaw_dist_m": 1.0,
    "ttc_floor_s": 2.0,
    "stop_margin_m": 0.3,
    "ttc_ray_max_m": 5.0,
    "ttc_rays_lateral": 9,
    "short_hop_m": 0.3,
    "max_lead_m": 0.4,
    "arrive_tol_m": 0.10,
    "arrive_tol_deg": 8.0,
    "arrive_max_speed_mps": 0.1,
    "anchor_merge_m": 0.5,
})

_HORIZ = 0.1          # a segment with |u_xy| below this is "vertical": no heading requirement


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(T) -> float:
    T = np.asarray(T)
    return float(np.arctan2(T[1, 0], T[0, 0]))


class Follower:
    def __init__(self, cfg: dict | None = None, static_map=None):
        c = dict(DEFAULTS)
        c.update(cfg or {})
        self.cfg = c
        self.map = static_map
        self.p_ref = None
        self.yaw_ref = None
        self.v = 0.0
        self._u_last = None
        self._t = None
        self._P = None
        self._pending = None

    @property
    def reference(self):
        return None if self.p_ref is None else (self.p_ref.copy(), float(self.yaw_ref))

    def clear(self):
        self._P = None
        self._pending = None
        self.v = 0.0

    def hold_at(self, T_map_body):
        """Drop the plan and put the reference ON the body (an emergency / no-plan hold, DN-59): the next plan starts
        where the vehicle is, not where the reference had led it (up to max_lead_m ahead, toward the obstacle)."""
        T = np.asarray(T_map_body, dtype=np.float64)
        self.clear()
        self.p_ref, self.yaw_ref = T[:3, 3].copy(), yaw_of(T)
        self._u_last = None

    @property
    def has_path(self) -> bool:
        return self._P is not None or self._pending is not None

    def set_path(self, waypoints, goal_yaw):
        P = np.asarray(waypoints, dtype=np.float64).reshape(-1, 3)
        if len(P) == 0:
            raise ValueError("empty path")
        self._pending = (P, float(goal_yaw))

    # ---------------------------------------------------------------------------------------------------------
    def _install(self, P, goal_yaw):
        anchor = self.p_ref
        d0 = float(np.linalg.norm(P[0] - anchor))
        if d0 <= float(self.cfg["anchor_merge_m"]):
            P = P.copy()
            P[0] = anchor
        else:
            P = np.vstack([anchor, P])
        keep = np.concatenate([[True], np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6])
        P = P[keep]
        if len(P) == 1:
            P = np.vstack([P, P])                                   # degenerate: hold at the goal
        D = np.diff(P, axis=0)
        lens = np.linalg.norm(D, axis=1)
        U = D / np.maximum(lens, 1e-12)[:, None]
        uh = np.hypot(U[:, 0], U[:, 1])
        uz = np.abs(U[:, 2])
        vh, vv = float(self.cfg["max_speed_mps"]), float(self.cfg["max_speed_v_mps"])
        cap = np.minimum(np.where(uh > 1e-9, vh / np.maximum(uh, 1e-9), np.inf),
                         np.where(uz > 1e-9, vv / np.maximum(uz, 1e-9), np.inf))
        cap = np.where(lens > 1e-6, cap, 0.0)
        S = np.concatenate([[0.0], np.cumsum(lens)])
        n = len(P)
        vlim = np.zeros(n)                                          # speed allowed AT each vertex
        stop = np.zeros(n, bool)
        stop[-1] = True
        lim = np.radians(float(self.cfg["corner_stop_deg"]))
        for i in range(1, n - 1):
            a, b = U[i - 1], U[i]
            ha, hb = uh[i - 1] >= _HORIZ, uh[i] >= _HORIZ
            if ha != hb:
                stp = True
            elif ha:
                stp = abs(wrap(np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0]))) > lim
            else:
                stp = float(np.arccos(np.clip(a @ b, -1, 1))) > lim
            stop[i] = stp
            vlim[i] = 0.0 if stp else min(cap[i - 1], cap[i])
        # look heading per segment: its own, or for a short horizontal hop the next longer horizontal segment's (DN-58)
        heads = np.arctan2(U[:, 1], U[:, 0])
        hlen = lens * uh
        look = heads.copy()
        hop = float(self.cfg["short_hop_m"])
        for i in range(len(U)):
            if uh[i] >= _HORIZ and hlen[i] < hop:
                for j in range(i + 1, len(U)):
                    if uh[j] < _HORIZ:
                        break
                    if hlen[j] >= hop:
                        look[i] = heads[j]
                        break
        self._look = look
        self._P, self._S, self._U, self._uh, self._cap = P, S, U, uh, cap
        self._vlim, self._stop = vlim, stop
        self._goal_yaw = goal_yaw
        self._s = 0.0
        if self._u_last is not None:
            self.v = max(0.0, self.v * float(self._u_last @ U[0]))
        else:
            self.v = 0.0

    def _point(self, s):
        i = int(np.clip(np.searchsorted(self._S, s, side="right") - 1, 0, len(self._U) - 1))
        return self._P[i] + self._U[i] * (s - self._S[i]), i

    # ---------------------------------------------------------------------------------------------------------
    def step(self, now, T_map_body, v_map, obstacles=()):
        c = self.cfg
        T = np.asarray(T_map_body, dtype=np.float64)
        p_body, yaw_body = T[:3, 3].copy(), yaw_of(T)
        v_body = np.asarray(v_map, dtype=np.float64).reshape(3) if v_map is not None else np.zeros(3)
        dt = 0.0 if self._t is None else float(np.clip(now - self._t, 0.0, 0.2))
        self._t = float(now)
        if self.p_ref is None:
            self.p_ref, self.yaw_ref, self.v = p_body, yaw_body, 0.0
        if self._pending is not None:
            self._install(*self._pending)
            self._pending = None
        info = {"mode": "hold", "s": 0.0, "remaining_m": 0.0, "v": 0.0, "v_ref": np.zeros(3), "v_target": 0.0,
                "limit": "", "d_ahead_m": float("inf"), "heading_err_deg": 0.0}
        if self._P is None:
            self.v = 0.0
            return self.p_ref.copy(), float(self.yaw_ref), False, info

        s = self._s
        _, idx = self._point(s)
        u, uh = self._U[idx], self._uh[idx]
        remaining = float(self._S[-1] - s)
        final = remaining <= float(c["final_yaw_dist_m"])
        horizontal = uh >= _HORIZ

        # yaw reference
        if final:
            yaw_t = self._goal_yaw
            moving = max(self.v, float(np.hypot(v_body[0], v_body[1]))) > float(c["arrive_max_speed_mps"])
            if horizontal and moving:
                # while still moving, the camera may lead toward the goal yaw only within the heading gate (DN-53
                # amended): turning it further away flew the last metre half-blind (return to the pad, goal yaw 180
                # deg from the motion: the wall ahead sat at the image border, depth ~2x short -> emergency)
                gate = np.radians(float(c["max_heading_err_deg"]))
                yaw_t = float(self._look[idx] + np.clip(wrap(self._goal_yaw - self._look[idx]), -gate, gate))
        elif horizontal:
            yaw_t = float(self._look[idx])
        else:
            yaw_t = self.yaw_ref
        rate = np.radians(float(c["max_yaw_rate_dps"]))
        self.yaw_ref = float(wrap(self.yaw_ref + np.clip(wrap(yaw_t - self.yaw_ref), -rate * dt, rate * dt)))

        # path-speed target
        a = float(c["max_accel_mps2"])
        vt, limit = float(self._cap[idx]), "cap"
        ahead = self._S > s + 1e-9
        if ahead.any():
            # discrete-time braking curve: decelerating by a*dt per tick from vb covers exactly the distance left
            h = 0.5 * a * (dt if dt > 0 else 0.05)
            vb = np.sqrt((self._vlim[ahead] + h) ** 2 + 2 * a * (self._S[ahead] - s)) - h
            if vb.min() < vt:
                vt, limit = float(vb.min()), "brake"
        if idx + 1 < len(self._cap) and self._S[idx + 1] - s < vt * max(dt, 1e-3) and self._cap[idx + 1] < vt:
            vt, limit = float(self._cap[idx + 1]), "cap"           # this tick crosses into a slower segment
        mode = "final" if final else "move"
        heading_err = 0.0
        if horizontal:
            dh = np.array([u[0], u[1], 0.0]) / uh
            r_h = float(c.get("hull_radius_m", 0.0))
            d_ahead = d_corr = float("inf")
            if self.map is not None:
                # centre ray + the looming emergency's corridor (DN-25): anything within hull radius + corridor margin
                # of the motion line counts at (along - hull radius). A horizontal fan at body height is enough: the
                # planner keeps mapped surfaces / obstacle tops > hull half-height + 0.2 / 0.3 m above and below,
                # beyond the corridor's vertical half-axis (hull half-height + margin); vertical rays add only
                # voxel-edge stalls
                a_h = r_h + float(c["corridor_margin_m"])
                n = int(c["ttc_rays_lateral"])
                lat = np.union1d(np.linspace(-a_h, a_h, n) if n > 1 else [], [0.0])
                side = np.array([-dh[1], dh[0], 0.0])
                dist = self.map.distance_along_rays(p_body + lat[:, None] * side, dh, float(c["ttc_ray_max_m"]),
                                                    obstacles)
                d_ahead, d_corr = float(dist[lat == 0.0][0]), float(dist.min())
            info["d_ahead_m"], info["d_corridor_m"] = d_ahead, d_corr
            # measured from the hull's leading edge, not the FC origin: looming's hull-contact time is what must stay
            # above the emergency threshold (looming builder's finding: from the origin, 2.0 s meant 1.5 s at 0.8 m).
            # Centre ray: with stop_margin_m (stops short of the surface ahead, e.g. the goal wall). Corridor: the
            # pure TTC (no margin), so its hull-contact time stays >= ttc_floor_s while a vertex the planner placed
            # near an oblique wall is still reachable (with the margin at the corridor edge it deadlocked, DN-57)
            floor = float(c["ttc_floor_s"])
            v_ttc = min(max(0.0, d_ahead - r_h - float(c["stop_margin_m"])), max(0.0, d_corr - r_h)) / floor
            if v_ttc / uh < vt:
                vt, limit = v_ttc / uh, "ttc"
            heading_err = float(wrap(self._look[idx] - yaw_body))
            if not final and abs(heading_err) > np.radians(float(c["max_heading_err_deg"])):
                vt, limit, mode = 0.0, "heading", "yaw_first"
        if np.linalg.norm(self.p_ref - p_body) > float(c["max_lead_m"]):
            vt, limit, mode = 0.0, "lead", "wait"

        # acceleration-limited path speed, never past the next stop vertex
        self.v = float(np.clip(vt, self.v - a * dt, self.v + a * dt)) if dt > 0 else min(self.v, vt)
        self.v = max(self.v, 0.0)
        adv = self.v * dt
        stops = np.nonzero(self._stop & ahead)[0]
        if stops.size:
            adv = min(adv, float(self._S[stops[0]] - s))
        s = min(s + adv, float(self._S[-1]))
        if stops.size and self._S[stops[0]] - s <= 1e-9:
            s = float(self._S[stops[0]])                       # arrived on a stop vertex: stopped (|dv| <= 2 a dt)
            self.v = 0.0
        self._s = s
        self.p_ref, idx2 = self._point(s)
        self._u_last = self._U[idx2]
        at_end = s >= self._S[-1] - 1e-9
        if at_end:
            self.p_ref = self._P[-1].copy()
            self.v = 0.0
        goal = self._P[-1]
        done = bool(at_end and np.linalg.norm(p_body - goal) <= float(c["arrive_tol_m"])
                    and abs(np.degrees(wrap(yaw_body - self._goal_yaw))) <= float(c["arrive_tol_deg"])
                    and np.linalg.norm(v_body) <= float(c["arrive_max_speed_mps"]))
        info.update({"mode": "done" if done else mode, "s": s, "remaining_m": float(self._S[-1] - s), "v": self.v,
                     "v_ref": self._U[idx2] * self.v, "v_target": vt, "limit": limit,
                     "heading_err_deg": float(np.degrees(heading_err))})
        return self.p_ref.copy(), float(self.yaw_ref), done, info

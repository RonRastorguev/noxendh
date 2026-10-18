"""Map <-> odom alignment for the Scout (docs/SCOUT_NAV.md §3-§4; decisions DN-10..DN-19 in
docs/decisions_scout_nav/map_align.md). numpy only, no ROS, Python 3.9.

The flight controller flies on its own estimate (optical flow + rangefinder + gyro, no vision fusion), so its odom
frame drifts slowly against the prior map in x, y and yaw. This module estimates the 4-DoF transform T_map_odom
(x, y, z, yaw; both frames gravity-aligned) from camera PnP fixes, and converts map setpoints into odom.

Model (DN-10, DN-14):
    x/y/yaw   p_map_xy = Rz(psi) p_odom_xy + t_xy,   yaw_map = yaw_odom + psi
    z         z_map = z_odom + o(t)
              height_ref "baro" / "ev":  o = beta                      (a plain, slowly drifting offset)
              height_ref "range":        o = T_px4(t) + beta
                  T_px4 = the map z of the surface PX4 is currently referencing its height to. In steady state it is
                  terrain(x, y under the vehicle NOW); after the vehicle crosses an edge PX4 follows the step with an
                  unknown lag. T_px4 is estimated (a) from the raw rangefinder `dist_bottom_m` when the odom sample
                  carries it: the lag residual L = (z_odom - dist_bottom) - m (m = its slow steady value) and
                  T_px4 = terrain_now - L, which needs no assumption about PX4's lag; or (b) without it, as a
                  first-order lag of terrain_now with the configured `hgt_model_tau_*` (an ASSUMPTION about PX4).
    beta, t_xy, psi are estimated from fixes; T_px4 is not (it is a physical model of what PX4 does).

Two copies of (psi, t_xy, beta) are kept (DN-12): the FILTER estimate (a per-axis Kalman filter with random-walk
process noise, updated by gated fixes) and the OUTPUT estimate, which slews towards the filter estimate at a bounded
rate (`slew_max_*`) every odom sample. Everything that commands or reports the vehicle (pose_map, setpoint_to_odom,
T_map_odom) uses the OUTPUT, so an alignment update can never make a setpoint jump by more than a few cm per tick.
body_pose_map_at (a measurement-geometry query, e.g. for the looming map comparison) uses the FILTER estimate.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .geometry import interpolate_pose, make_T, rot_z

DEFAULTS = {
    # buffers / time
    "odom_buffer_s": 3.0,             # history kept for fix interpolation (camera latency 50-200 ms + margin)
    "max_extrapolate_s": 0.10,        # a stamp this far past the newest odom sample is extrapolated with v_odom
    "max_odom_age_s": 0.5,            # valid() is false when the newest odom sample is older than this
    "default_height_ref": "range",    # the real Scout: EKF2_HGT_REF=2 (docs/SCOUT_NAV.md §3)
    # fix noise (1 sigma), process noise (random walk), init uncertainty -- all PROVISIONAL
    "fix_sigma_xy_m": 0.03,
    "fix_sigma_z_m": 0.03,
    "fix_sigma_yaw_deg": 1.0,
    "q_xy_time": 0.01,                # m / sqrt(s): odom x/y drift while hovering
    "q_xy_dist": 0.05,                # m / sqrt(m travelled): flow odom drift with distance
    "q_yaw_time_deg": 0.1,            # deg / sqrt(s): gyro-only yaw drift
    "q_z_time": 0.005,                # m / sqrt(s): height-offset drift
    "init_sigma_xy_m": 0.03,          # the fixed takeoff pose (placed by hand)
    "init_sigma_yaw_deg": 2.0,
    "init_sigma_z_m": 0.03,
    # gating / outliers / re-anchor (DN-11)
    "gate_sigma": 4.0,                # gate = max(gate_min, gate_sigma * sqrt(P + R))
    "gate_min_xy_m": 0.15,
    "gate_min_z_m": 0.15,
    "gate_min_yaw_deg": 5.0,
    "reanchor_fixes": 5,              # consecutive rejected but mutually consistent fixes -> re-anchor
    "reanchor_window_s": 2.0,
    "reanchor_min_span_s": 0.3,
    "reanchor_tol_m": 0.08,
    "reanchor_tol_z_m": 0.08,
    "reanchor_tol_deg": 3.0,
    "init_fixes": 2,                  # without init_known_pose: consistent fixes needed to initialise
    # output slew limits (DN-12): 0.3 m/s = 1.5 cm per 50 ms tick
    "slew_max_mps": 0.3,
    "slew_max_z_mps": 0.2,
    "slew_max_dps": 6.0,
    # validity (DN-15): invalid when no fix for max_fix_age_s AND the propagated drift exceeds the budget
    "max_fix_age_s": 3.0,
    "drift_budget_m": 0.08,
    "drift_budget_deg": 5.0,
    # range height reference (DN-14)
    "use_dist_bottom": True,
    "dist_bottom_min_m": 0.02,        # MTF-02P dead zone (scout_hw.yaml range_flow)
    "dist_bottom_max_m": 6.0,
    "hgt_model_tau_up_s": 0.5,        # ASSUMED PX4 lag when no dist_bottom: terrain rising under the vehicle
    "hgt_model_tau_down_s": 0.5,      # ... terrain falling. tau_up large + tau_down 0 = "always err upwards"
    "rng_deadband_m": 0.03,           # |L| below this is treated as 0 (rangefinder noise)
    "rng_offset_tau_s": 5.0,          # EMA of the steady (z_odom - dist_bottom)
    "rng_offset_track_band_m": 0.05,  # ... updated only while within this band (not during a transient)
    "rng_offset_relock_s": 10.0,      # ... unless outside it this long (PX4's offset really changed)
    "terrain_max_rate_mps": 20.0,     # rate limit on T_px4 when estimated from dist_bottom (a mis-snapped surface)
    "terrain_transient_band_m": 0.05, # |terrain_now - T_px4| above this = transient: fixes do not update beta
    "terrain_transient_hold_s": 0.5,
    "terrain_settle_max_s": 5.0,
    # surface under the beam (DN-14b)
    "terrain_ambiguity_radius_m": 0.25, # beam footprint + x/y error: surfaces this close count as candidates
    "surface_step_m": 0.08,           # map z may move this much per sample (min) and still be the same surface
    "surface_snap_tol_m": 0.10,       # first acquisition: snap to a map surface this close
    "max_vz_mps": 1.5,                # vertical speed bound: per-sample continuity tolerance 2 * this * dt
    "candidates_period_s": 0.1,      # lag model only: after a terrain step under the vehicle, PX4 may still be
                                      # settling for this long (unknown lag), so fixes do not update beta meanwhile
}

_HEIGHT_REFS = ("range", "baro", "ev")


def wrap(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def yaw_of(R) -> float:
    """Yaw (about map z) of a rotation, ZYX convention: yaw_of(Rz(p) R) == yaw_of(R) + p."""
    R = np.asarray(R)
    return float(math.atan2(R[1, 0], R[0, 0]))


class _Align:
    """(psi, t_xy, beta): p_map_xy = Rz(psi) p_odom_xy + t_xy; z_map = z_odom + T_px4 + beta."""
    __slots__ = ("psi", "t", "beta")

    def __init__(self, psi=0.0, t=(0.0, 0.0), beta=0.0):
        self.psi, self.t, self.beta = float(psi), np.array(t, dtype=np.float64).reshape(2), float(beta)

    def copy(self):
        return _Align(self.psi, self.t.copy(), self.beta)

    def R2(self):
        c, s = math.cos(self.psi), math.sin(self.psi)
        return np.array([[c, -s], [s, c]])

    def xy(self, p_odom):
        return self.R2() @ np.asarray(p_odom, dtype=np.float64)[:2] + self.t

    def set_xy_about(self, p_odom, p_map_xy):
        """Keep psi, choose t so that p_odom maps to p_map_xy (rotations happen about the body, DN-12)."""
        self.t = np.asarray(p_map_xy, dtype=np.float64)[:2] - self.R2() @ np.asarray(p_odom, dtype=np.float64)[:2]

    def T(self, That=0.0) -> np.ndarray:
        return make_T(rot_z(self.psi), (self.t[0], self.t[1], That + self.beta))

    def map_pose(self, T_odom_body, That=0.0) -> np.ndarray:
        return self.T(That) @ np.asarray(T_odom_body, dtype=np.float64)


class _Entry:
    __slots__ = ("stamp", "T", "v", "That", "transient", "rc", "dist")

    def __init__(self, stamp, T, v, That, transient, rc, dist=None):
        self.stamp, self.T, self.v, self.That, self.transient, self.rc = stamp, T, v, That, transient, rc
        self.dist = dist


class MapOdomAligner:
    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        self._terrain_fn = None
        self.height_ref = str(self.cfg["default_height_ref"])
        if self.height_ref not in _HEIGHT_REFS:
            raise ValueError(f"height_ref must be one of {_HEIGHT_REFS}, got {self.height_ref!r}")
        self._buf = deque()
        self._filt = None                  # _Align, Kalman estimate
        self._out = None                   # _Align, slewed output
        self._P = None                     # {"xy": m^2, "yaw": rad^2, "z": m^2}
        self._That = 0.0                   # current T_px4 estimate
        self._m = None                     # steady (z_odom - dist_bottom)
        self._m_out_since = None
        self._hgt_source = "none"
        self._last_transient_t = -1e18
        self._last_terr = None
        self._last_step_t = -1e18
        self._surf = None                  # map z of the surface under the rangefinder beam (dist_bottom mode)
        self._surf_mapped = False
        self._dist_prev = None
        self._cands, self._cands_t, self._ambiguous = None, -1e18, False
        self._last_surface_step = None
        self._last_accept_t = None
        self._pending_init = None
        self._cand = []                    # rejected (or pre-init) fixes: dicts with an implied alignment
        self._last_rc = None
        self._n = {"accepted": 0, "rejected": 0, "reanchors": 0, "resets": 0, "z_skipped": 0, "odom": 0,
                   "surface_steps": 0, "surface_switch_by_fixes": 0}
        self._last = {"residual_m": None, "residual_deg": None, "residual_z_m": None, "reason": "", "stamp": None}
        self._last_reset = None
        self._events = deque(maxlen=50)

    # ------------------------------------------------------------------------------------------------ setup
    def set_terrain(self, fn) -> None:
        """fn(x, y, z_body_map) -> map z of the surface below (static map + known obstacles). Kept continuous: the
        current height offset is preserved, so swapping the function does not move the vehicle by itself."""
        o = self._offset_now() if self._filt is not None and self._buf else None
        self._terrain_fn = fn
        if o is not None:
            self._repartition_z(o)

    def set_height_ref(self, kind: str) -> None:
        kind = str(kind)
        if kind not in _HEIGHT_REFS:
            raise ValueError(f"height_ref must be one of {_HEIGHT_REFS}, got {kind!r}")
        o = self._offset_now() if self._filt is not None and self._buf else None
        self.height_ref = kind
        if o is not None:
            self._repartition_z(o)
        elif kind != "range":
            self._That = 0.0

    def _repartition_z(self, o):
        """Keep the total z offset o = T_px4 + beta while T_px4 is redefined (new terrain fn / height ref)."""
        e = self._buf[-1]
        self._surf, self._dist_prev, self._cands = None, None, None
        That_new = self._terrain_at(e.T[:3, 3], o) if self.height_ref == "range" else 0.0
        d = That_new - self._That
        self._That = That_new
        for a in (self._filt, self._out):
            a.beta -= d
        for b in self._buf:
            b.That += d

    def _terrain_at(self, p_odom, o) -> float:
        if self._terrain_fn is None:
            return 0.0
        a = self._out if self._out is not None else _Align()
        x, y = a.xy(p_odom)
        return float(self._terrain_fn(float(x), float(y), float(p_odom[2] + o)))

    # ------------------------------------------------------------------------------------------------ odom
    def add_odom(self, s) -> None:
        stamp = float(s.stamp)
        T = np.asarray(s.T_odom_body, dtype=np.float64)
        v = np.asarray(s.v_odom, dtype=np.float64).reshape(3)
        rc = int(getattr(s, "reset_counter", 0) or 0)
        if self._buf and stamp <= self._buf[-1].stamp:
            return                                                      # duplicate / out of order
        self._n["odom"] += 1
        if self._buf and self._last_rc is not None and rc != self._last_rc:
            self._handle_reset(stamp, T, v, rc, s)
        self._last_rc = rc
        prev = self._buf[-1] if self._buf else None
        dt = 0.0 if prev is None else stamp - prev.stamp
        ds = 0.0 if prev is None else float(np.linalg.norm(T[:2, 3] - prev.T[:2, 3]))

        if self._pending_init is not None and self._filt is None:
            T_mb, _ = self._pending_init
            self._pending_init = None
            self._init_from(T_mb, T, stamp, getattr(s, "dist_bottom_m", None))

        transient = False
        if self._filt is not None:
            self._propagate_P(dt, ds)
            transient = self._update_height(dt, T, s)
            self._slew(dt, T)
        self._buf.append(_Entry(stamp, T, v, self._That, transient, rc, getattr(s, "dist_bottom_m", None)))
        while len(self._buf) > 2 and self._buf[0].stamp < stamp - self.cfg["odom_buffer_s"]:
            self._buf.popleft()

    def _propagate_P(self, dt, ds):
        c = self.cfg
        self._P["xy"] += c["q_xy_time"] ** 2 * dt + c["q_xy_dist"] ** 2 * ds
        self._P["yaw"] += math.radians(c["q_yaw_time_deg"]) ** 2 * dt
        self._P["z"] += c["q_z_time"] ** 2 * dt

    def _dist_ok(self, d) -> bool:
        if d is None or not self.cfg["use_dist_bottom"]:
            return False
        d = float(d)
        return math.isfinite(d) and self.cfg["dist_bottom_min_m"] <= d <= self.cfg["dist_bottom_max_m"]

    def _candidates(self, p_odom, o) -> np.ndarray:
        """Distinct terrain heights (map z, sorted) within `terrain_ambiguity_radius_m` of the body's map x/y: every
        surface the rangefinder footprint could be hitting given the beam width and our position error (DN-14b)."""
        if self._terrain_fn is None:
            return np.array([0.0])
        a = self._out if self._out is not None else _Align()
        x, y = a.xy(p_odom)
        zb = float(p_odom[2] + o)
        r = float(self.cfg["terrain_ambiguity_radius_m"])
        pts = [(0.0, 0.0)]
        for rr in (0.5 * r, r):
            pts += [(rr * math.cos(k * math.pi / 4), rr * math.sin(k * math.pi / 4)) for k in range(8)]
        h_all = [float(self._terrain_fn(float(x + dx), float(y + dy), zb)) for dx, dy in pts]
        h0 = round(h_all[0], 3)                           # the surface straight under the body
        h = np.unique(np.round(np.asarray(h_all), 3))
        # merge heights closer than the transient band into one surface. Its height is the one straight under the
        # body if that sample belongs to the cluster, else the cluster's TOP: near a box edge a voxel mixes side-face
        # points with the top and its mean reads low (sim hover_test: a 0.5 m pad read 0.355 instead of 0.40 at its
        # edge, and keeping the cluster's LOWEST value biased the map z by -5.5 cm for the whole hover).
        band = self.cfg["terrain_transient_band_m"]
        clusters = [[h[0]]]
        for v in h[1:]:
            if v - clusters[-1][-1] > band:
                clusters.append([v])
            else:
                clusters[-1].append(v)
        return np.asarray([h0 if min(cl) - 1e-9 <= h0 <= max(cl) + 1e-9 else max(cl) for cl in clusters])

    def _snap(self, cands, s_raw):
        i = int(np.argmin(np.abs(cands - s_raw)))
        return float(cands[i]) if abs(cands[i] - s_raw) <= self.cfg["surface_snap_tol_m"] else float(s_raw)

    def _refresh_candidates(self, T, o, stamp, force=False):
        if force or self._cands is None or stamp - self._cands_t >= self.cfg["candidates_period_s"]:
            self._cands = self._candidates(T[:3, 3], o)
            self._cands_t = stamp
            self._ambiguous = bool(len(self._cands) > 1)
        return self._cands

    def _update_height(self, dt, T, s) -> bool:
        """Advance the T_px4 estimate to this sample; returns whether z from fixes is unreliable now (a terrain
        transient, or the terrain under the vehicle is ambiguous)."""
        c = self.cfg
        stamp = float(s.stamp)
        if self.height_ref != "range":
            self._That, self._hgt_source, self._ambiguous = 0.0, "offset", False
            return False
        o_prev = self._That + self._out.beta
        cands = self._refresh_candidates(T, o_prev, stamp)
        d = getattr(s, "dist_bottom_m", None)
        if self._dist_ok(d):
            d = float(d)
            # The surface under the BEAM is chosen so that the body's map z stays continuous (DN-14b): the map
            # surface near the vehicle that best explains (previous map z - range) wins; if none does (a mixed
            # footprint reading, an unmapped object) the previous unmapped surface continues, else the map z is
            # carried and the surface is whatever the range implies. The map EDGE never decides it, so a voxelized /
            # dilated edge or our x/y error cannot desynchronise it, and a map-explained sample is absolute (no
            # accumulation across flicker).
            thr = max(c["surface_step_m"], 2.0 * c["max_vz_mps"] * max(dt, 0.0))
            z_prev = (self._surf + self._dist_prev) if (self._surf is not None and self._dist_prev is not None)                 else float(T[2, 3]) + o_prev
            i = int(np.argmin(np.abs(cands + d - z_prev)))
            old = self._surf
            if abs(cands[i] + d - z_prev) <= (thr if self._surf is not None else c["surface_snap_tol_m"]):
                self._surf, self._surf_mapped = float(cands[i]), True
            elif self._surf is not None and not self._surf_mapped and abs(self._surf + d - z_prev) <= thr:
                pass                                            # an unmapped surface continues
            else:
                self._surf, self._surf_mapped = z_prev - d, False
            if old is not None and abs(self._surf - old) > c["terrain_transient_band_m"]:
                self._n["surface_steps"] += 1
                self._last_surface_step = {"stamp": stamp, "from": old, "to": self._surf,
                                           "mapped": self._surf_mapped}
            self._dist_prev = d
            terr = self._surf
            raw = float(T[2, 3]) - d
            if self._m is None:
                self._m = raw
            dev = raw - self._m
            if abs(dev) <= c["rng_offset_track_band_m"]:
                self._m_out_since = None
                track = True
            else:
                if self._m_out_since is None:
                    self._m_out_since = stamp
                track = stamp - self._m_out_since > c["rng_offset_relock_s"]
            if track and dt > 0:
                self._m += (1.0 - math.exp(-dt / max(1e-6, c["rng_offset_tau_s"]))) * (raw - self._m)
            dev = raw - self._m
            # smooth deadband: ~0 inside the range noise, exact (not offset by the band) for a real lag residual
            db = max(1e-6, c["rng_deadband_m"])
            L = dev * (1.0 - math.exp(-(dev / db) ** 2))
            T_des = terr - L
            step = c["terrain_max_rate_mps"] * dt
            self._That += float(np.clip(T_des - self._That, -step, step)) if dt > 0 else T_des - self._That
            self._hgt_source = "dist_bottom"
            transient = abs(L) > c["rng_deadband_m"] or abs(terr - self._That) > c["terrain_transient_band_m"]
        else:
            self._surf, self._dist_prev = None, None           # re-acquired from continuity when range returns
            terr = self._terrain_at(T[:3, 3], o_prev)
            tau = c["hgt_model_tau_up_s"] if terr > self._That else c["hgt_model_tau_down_s"]
            if tau <= 0:
                a = 1.0
            elif dt <= 0:
                a = 0.0
            else:
                a = 1.0 - math.exp(-dt / tau)
            self._That += a * (terr - self._That)
            self._hgt_source = "lag_model"
            transient = (abs(terr - self._That) > c["terrain_transient_band_m"]
                         or stamp - self._last_step_t < c["terrain_settle_max_s"])
        if self._last_terr is not None and abs(terr - self._last_terr) > c["terrain_transient_band_m"]:
            self._last_step_t = stamp
        self._last_terr = terr
        if transient:
            self._last_transient_t = stamp
        return bool(transient or self._ambiguous or stamp - self._last_transient_t < c["terrain_transient_hold_s"])

    def _slew(self, dt, T):
        """Move the output alignment towards the filter estimate, rate-limited at the current body (DN-12)."""
        c, f, o = self.cfg, self._filt, self._out
        p = T[:3, 3]
        target_xy = f.xy(p)
        cur_xy = o.xy(p)
        dpsi = wrap(f.psi - o.psi)
        mpsi = math.radians(c["slew_max_dps"]) * dt
        o.psi = wrap(o.psi + float(np.clip(dpsi, -mpsi, mpsi)))
        dxy = target_xy - cur_xy
        n = float(np.linalg.norm(dxy))
        mxy = c["slew_max_mps"] * dt
        if n > mxy:
            dxy = dxy * (mxy / n)
        o.set_xy_about(p, cur_xy + dxy)
        mz = c["slew_max_z_mps"] * dt
        o.beta += float(np.clip(f.beta - o.beta, -mz, mz))

    def _handle_reset(self, stamp, T_new, v_new, rc, s):
        """The FC reset its local frame. Continuity of the physical pose gives the delta D (old odom -> new odom);
        both alignments and the buffer are re-expressed in the new frame, so nothing jumps in map (DN-16)."""
        prev = self._buf[-1]
        dt = stamp - prev.stamp
        p_pred = prev.T[:3, 3] + prev.v * dt
        yaw_rate = 0.0
        if len(self._buf) >= 2:
            pp = self._buf[-2]
            if prev.stamp > pp.stamp:
                yaw_rate = wrap(yaw_of(prev.T[:3, :3]) - yaw_of(pp.T[:3, :3])) / (prev.stamp - pp.stamp)
        yaw_pred = yaw_of(prev.T[:3, :3]) + yaw_rate * dt
        dpsi = wrap(yaw_of(T_new[:3, :3]) - yaw_pred)
        Rz2 = rot_z(dpsi)[:2, :2]
        dxy = T_new[:2, 3] - Rz2 @ p_pred[:2]
        dz = float(T_new[2, 3] - p_pred[2])
        D = make_T(rot_z(dpsi), (dxy[0], dxy[1], dz))
        for a in (self._filt, self._out):
            if a is None:
                continue
            # p_map = A(p_old) = A(D^-1 p_new): psi' = psi - dpsi, t' = t - Rz(psi') dxy, beta' = beta - dz
            a.psi = wrap(a.psi - dpsi)
            a.t = a.t - a.R2() @ dxy
            a.beta -= dz
        for b in self._buf:
            b.T = D @ b.T
            b.v = rot_z(dpsi) @ b.v
        for cnd in self._cand:
            cnd["psi"] = wrap(cnd["psi"] - dpsi)
            cnd["t"] = cnd["t"] - rot_z(cnd["psi"])[:2, :2] @ dxy
            cnd["beta"] -= dz
            cnd["p_odom"] = D[:3, :3] @ cnd["p_odom"] + D[:3, 3]
        z_reset = abs(dz) > 0.01
        if self.height_ref == "range" and z_reset and self._filt is not None:
            # A z reset under a range height reference most likely means PX4 re-referenced its height to the
            # surface now below it: T_px4 := terrain now, the total offset stays continuous (beta absorbs it).
            o_new = self._That + self._out.beta
            terr = self._surf if self._surf is not None else self._terrain_at(T_new[:3, 3], o_new)
            d = terr - self._That
            self._That = terr
            for a in (self._filt, self._out):
                a.beta -= d
            for b in self._buf:
                b.That += d
            dist = getattr(s, "dist_bottom_m", None)
            if self._dist_ok(dist):
                self._m = float(T_new[2, 3]) - float(dist)
                self._m_out_since = None
            elif self._m is not None:
                self._m += dz
        elif z_reset and self._m is not None:
            self._m += dz
        if self._P is not None:                                         # the continuity guess is not exact
            self._P["xy"] += 0.02 ** 2
            self._P["yaw"] += math.radians(1.0) ** 2
            self._P["z"] += 0.02 ** 2
        self._n["resets"] += 1
        self._last_reset = {"stamp": stamp, "reset_counter": rc, "dxy_m": [float(dxy[0]), float(dxy[1])],
                            "dz_m": dz, "dyaw_deg": math.degrees(dpsi)}
        self._events.append(dict(self._last_reset, event="odom_reset"))

    # ------------------------------------------------------------------------------------------------ buffer
    def _odom_at(self, stamp):
        """(T_odom_body, v_odom, T_px4, transient) at a stamp: interpolated, or extrapolated a little past the end."""
        if not self._buf:
            return None
        b = self._buf
        if stamp > b[-1].stamp:
            dt = stamp - b[-1].stamp
            if dt > self.cfg["max_extrapolate_s"]:
                return None
            T = b[-1].T.copy()
            T[:3, 3] = T[:3, 3] + b[-1].v * dt
            return T, b[-1].v.copy(), b[-1].That, b[-1].transient
        if stamp < b[0].stamp:
            return None
        stamps = [e.stamp for e in b]
        i = int(np.searchsorted(stamps, stamp, side="right"))
        if i >= len(b):
            e = b[-1]
            return e.T.copy(), e.v.copy(), e.That, e.transient
        e0, e1 = b[i - 1], b[i]
        T = interpolate_pose([e0.stamp, e1.stamp], [e0.T, e1.T], stamp)
        span = e1.stamp - e0.stamp
        a = 0.0 if span <= 0 else (stamp - e0.stamp) / span
        return T, (1 - a) * e0.v + a * e1.v, (1 - a) * e0.That + a * e1.That, (e0.transient or e1.transient)

    def _offset_now(self) -> float:
        return (self._That if self.height_ref == "range" else 0.0) + self._out.beta

    # ------------------------------------------------------------------------------------------------ init
    def init_known_pose(self, T_map_body, stamp) -> None:
        """Initialise at the FIXED takeoff pose, on the ground, before any PnP fix (DN-17). T_map_body z is the body
        origin's map height (floor + fc_height_at_rest). Applied at the odom sample at `stamp`, or at the next odom
        sample if there is no odom yet. A second call re-initialises."""
        T_map_body = np.asarray(T_map_body, dtype=np.float64)
        if not self._buf:
            self._pending_init = (T_map_body, float(stamp))
            self._filt = None
            return
        od = self._odom_at(float(stamp))
        e = min(self._buf, key=lambda q: abs(q.stamp - float(stamp)))
        T_ob = od[0] if od is not None else e.T
        self._init_from(T_map_body, T_ob, float(stamp), e.dist if self._dist_ok(e.dist) else None)

    def _init_from(self, T_map_body, T_odom_body, stamp, dist):
        c = self.cfg
        psi = wrap(yaw_of(T_map_body[:3, :3]) - yaw_of(T_odom_body[:3, :3]))
        a = _Align(psi=psi)
        a.set_xy_about(T_odom_body[:3, 3], T_map_body[:2, 3])
        self._out = a                                       # needed by _terrain_at
        if self.height_ref == "range":
            self._That = self._terrain_fn(float(T_map_body[0, 3]), float(T_map_body[1, 3]),
                                          float(T_map_body[2, 3])) if self._terrain_fn is not None else 0.0
            self._That = float(self._That)
        else:
            self._That = 0.0
        a.beta = float(T_map_body[2, 3] - T_odom_body[2, 3] - self._That)
        self._filt, self._out = a, a.copy()
        self._P = {"xy": c["init_sigma_xy_m"] ** 2, "yaw": math.radians(c["init_sigma_yaw_deg"]) ** 2,
                   "z": c["init_sigma_z_m"] ** 2}
        self._m = None if dist is None else float(T_odom_body[2, 3]) - float(dist)
        self._m_out_since = None
        self._surf, self._dist_prev, self._cands = None, None, None   # re-acquired at the next sample
        for b in self._buf:
            b.That = self._That
        self._last_accept_t = stamp
        self._cand = []
        self._events.append({"event": "init", "stamp": stamp, "psi_deg": math.degrees(psi)})

    # ------------------------------------------------------------------------------------------------ fixes
    def _implied(self, stamp, T_map_body, od):
        T_ob, _, That, _ = od
        psi = wrap(yaw_of(T_map_body[:3, :3]) - yaw_of(T_ob[:3, :3]))
        a = _Align(psi=psi)
        a.set_xy_about(T_ob[:3, 3], T_map_body[:2, 3])
        a.beta = float(T_map_body[2, 3] - T_ob[2, 3] - (That if self.height_ref == "range" else 0.0))
        return {"stamp": stamp, "psi": a.psi, "t": a.t, "beta": a.beta, "p_odom": T_ob[:3, 3].copy()}

    def _cand_consistent(self, cands, check_z):
        """Mutual consistency of implied alignments, compared where they matter: at the newest odom position."""
        c = self.cfg
        p = self._buf[-1].T[:3, 3]
        xs = np.array([_Align(k["psi"], k["t"]).xy(p) for k in cands])
        psis = np.array([k["psi"] for k in cands])
        mpsi = math.atan2(np.mean(np.sin(psis)), np.mean(np.cos(psis)))
        mxy = xs.mean(axis=0)
        ok = (np.all(np.linalg.norm(xs - mxy, axis=1) <= c["reanchor_tol_m"])
              and all(abs(wrap(q - mpsi)) <= math.radians(c["reanchor_tol_deg"]) for q in psis))
        betas = np.array([k["beta"] for k in cands])
        if check_z:
            ok = ok and bool(np.all(np.abs(betas - betas.mean()) <= c["reanchor_tol_z_m"]))
        return bool(ok), mpsi, mxy, float(np.median(betas))

    def _push_candidate(self, k):
        c = self.cfg
        self._cand = [q for q in self._cand if k["stamp"] - q["stamp"] <= c["reanchor_window_s"]]
        self._cand.append(k)
        while len(self._cand) > 1:
            ok, _, _, _ = self._cand_consistent(self._cand, check_z=not k.get("transient", False))
            if ok:
                break
            self._cand.pop(0)                         # drop the oldest until the run is consistent again

    def _adopt(self, mpsi, mxy, beta, z_ok, reason, stamp):
        """Set the filter estimate from a consistent run (init or re-anchor). The OUTPUT still slews (DN-11)."""
        c = self.cfg
        p = self._buf[-1].T[:3, 3]
        new = _Align(psi=mpsi)
        new.set_xy_about(p, mxy)
        new.beta = beta if z_ok or self._filt is None else self._filt.beta
        first = self._filt is None
        if z_ok and not first and self.height_ref == "range" and self._surf is not None:
            # The fixes say z is off by dz. If a map surface near the vehicle sits exactly dz from the one we think
            # the beam sees, the beam surface was misidentified: switch it rather than corrupt beta (DN-14b).
            dz = beta - self._filt.beta
            cands = self._refresh_candidates(self._buf[-1].T, self._That + self._out.beta, stamp, force=True)
            cnd = self._snap(cands, self._surf + dz)
            if abs(cnd - (self._surf + dz)) <= c["surface_snap_tol_m"] and                     abs(cnd - self._surf) > c["terrain_transient_band_m"] and                     any(abs(cnd - q) < 1e-9 for q in cands):
                self._events.append({"event": "surface_switch_by_fixes", "stamp": stamp, "from": self._surf,
                                     "to": cnd})
                self._surf = cnd
                self._n["surface_switch_by_fixes"] += 1
                new.beta = self._filt.beta
        self._filt = new
        if first:
            self._surf, self._dist_prev, self._cands = None, None, None
            # candidates carried T_px4 = 0 (nothing was modelled before init): re-split the offset at the body now
            self._out = new.copy()
            if self.height_ref == "range":
                o = new.beta
                That = self._terrain_at(p, o)
                for a in (self._filt, self._out):
                    a.beta = o - That
                self._That = That
                for b in self._buf:
                    b.That = That
                e = self._buf[-1]
                self._m = float(e.T[2, 3]) - float(e.dist) if self._dist_ok(e.dist) else None
            else:
                self._That = 0.0
        self._P = {"xy": c["fix_sigma_xy_m"] ** 2, "yaw": math.radians(c["fix_sigma_yaw_deg"]) ** 2,
                   "z": c["fix_sigma_z_m"] ** 2 if z_ok or first else self._P["z"]}
        self._cand = []
        self._last_accept_t = stamp
        self._events.append({"event": reason, "stamp": stamp, "psi_deg": math.degrees(mpsi)})

    def add_fix(self, stamp, T_map_body, info=None) -> dict:
        """A PnP fix: the body pose in map at the IMAGE stamp. -> {"accepted", "reason", "residual_m" (horizontal),
        "residual_deg", "residual_z_m", "gate_m", "gate_deg"}."""
        c = self.cfg
        info = info or {}
        stamp = float(stamp)
        T_map_body = np.asarray(T_map_body, dtype=np.float64)
        res = {"accepted": False, "reason": "", "residual_m": None, "residual_deg": None, "residual_z_m": None,
               "gate_m": None, "gate_deg": None}
        od = self._odom_at(stamp)
        if od is None:
            res["reason"] = "stamp_outside_odom_buffer" if self._buf else "no_odom"
            return self._finish_rejected(res, count=False)
        T_ob, _, That, transient = od
        k = self._implied(stamp, T_map_body, od)
        k["transient"] = bool(transient and self.height_ref == "range")

        if self._filt is None:                                  # not initialised: a consistent run initialises
            if self._pending_init is not None:
                res["reason"] = "init_pending_odom"
                return self._finish_rejected(res, count=False)
            self._push_candidate(k)
            n = int(c["init_fixes"])
            if len(self._cand) >= n:
                ok, mpsi, mxy, beta = self._cand_consistent(self._cand, check_z=True)
                if ok:
                    self._adopt(mpsi, mxy, beta, True, "init_from_fixes", stamp)
                    res.update(accepted=True, reason="init_from_fixes")
                    self._n["accepted"] += 1
                    self._last.update(reason=res["reason"], stamp=stamp)
                    return res
            res["reason"] = f"not_initialized (candidate {len(self._cand)}/{n})"
            return self._finish_rejected(res, count=False)

        f, P = self._filt, self._P
        R_xy = (float(info.get("sigma_m", c["fix_sigma_xy_m"]))) ** 2
        R_z = (float(info.get("sigma_z_m", c["fix_sigma_z_m"]))) ** 2
        R_yaw = math.radians(float(info.get("sigma_deg", c["fix_sigma_yaw_deg"]))) ** 2
        p_o = T_ob[:3, 3]
        pred_xy = f.xy(p_o)
        r_xy = T_map_body[:2, 3] - pred_xy
        r_psi = wrap(yaw_of(T_map_body[:3, :3]) - (yaw_of(T_ob[:3, :3]) + f.psi))
        z_pred = p_o[2] + (That if self.height_ref == "range" else 0.0) + f.beta
        r_z = float(T_map_body[2, 3] - z_pred)
        g_xy = max(c["gate_min_xy_m"], c["gate_sigma"] * math.sqrt(P["xy"] + R_xy))
        g_yaw = max(math.radians(c["gate_min_yaw_deg"]), c["gate_sigma"] * math.sqrt(P["yaw"] + R_yaw))
        g_z = max(c["gate_min_z_m"], c["gate_sigma"] * math.sqrt(P["z"] + R_z))
        n_xy = float(np.linalg.norm(r_xy))
        res.update(residual_m=n_xy, residual_deg=math.degrees(r_psi), residual_z_m=r_z, gate_m=g_xy,
                   gate_deg=math.degrees(g_yaw))
        self._last.update(residual_m=n_xy, residual_deg=math.degrees(r_psi), residual_z_m=r_z, stamp=stamp)
        z_skip = k["transient"]
        why = []
        if n_xy > g_xy:
            why.append(f"xy residual {n_xy:.3f} m > gate {g_xy:.3f}")
        if abs(r_psi) > g_yaw:
            why.append(f"yaw residual {math.degrees(r_psi):.1f} deg > gate {math.degrees(g_yaw):.1f}")
        if not z_skip and abs(r_z) > g_z:
            why.append(f"z residual {r_z:.3f} m > gate {g_z:.3f}")
        if why:
            self._push_candidate(k)
            if len(self._cand) >= int(c["reanchor_fixes"]) and \
                    self._cand[-1]["stamp"] - self._cand[0]["stamp"] >= c["reanchor_min_span_s"]:
                ok, mpsi, mxy, beta = self._cand_consistent(self._cand, check_z=not z_skip)
                if ok:
                    n = len(self._cand)
                    self._adopt(mpsi, mxy, beta, not z_skip, "reanchor", stamp)
                    self._n["reanchors"] += 1
                    self._n["accepted"] += 1
                    res.update(accepted=True, reason=f"reanchor ({n} consistent fixes; was: {'; '.join(why)})")
                    self._last["reason"] = res["reason"]
                    return res
            res["reason"] = "rejected: " + "; ".join(why)
            return self._finish_rejected(res)

        # Kalman update, per axis; the position correction is applied ABOUT THE BODY at the fix stamp.
        K_xy = P["xy"] / (P["xy"] + R_xy)
        K_yaw = P["yaw"] / (P["yaw"] + R_yaw)
        f.psi = wrap(f.psi + K_yaw * r_psi)
        f.set_xy_about(p_o, pred_xy + K_xy * r_xy)
        P["xy"] *= (1.0 - K_xy)
        P["yaw"] *= (1.0 - K_yaw)
        if z_skip:
            self._n["z_skipped"] += 1
        else:
            K_z = P["z"] / (P["z"] + R_z)
            f.beta += K_z * r_z
            P["z"] *= (1.0 - K_z)
        self._cand = []
        self._last_accept_t = stamp if self._last_accept_t is None else max(self._last_accept_t, stamp)
        self._n["accepted"] += 1
        res.update(accepted=True, reason="accepted" + (" (z not used: terrain transient / ambiguous)" if z_skip
                                                       else ""))
        self._last["reason"] = res["reason"]
        return res

    def _finish_rejected(self, res, count=True):
        if count:
            self._n["rejected"] += 1
        self._last["reason"] = res["reason"]
        return res

    # ------------------------------------------------------------------------------------------------ queries
    def initialized(self) -> bool:
        return self._filt is not None

    def _sigmas(self, now):
        extra = 0.0 if not self._buf or now is None else max(0.0, float(now) - self._buf[-1].stamp)
        c = self.cfg
        sxy = math.sqrt(self._P["xy"] + c["q_xy_time"] ** 2 * extra)
        syaw = math.degrees(math.sqrt(self._P["yaw"] + math.radians(c["q_yaw_time_deg"]) ** 2 * extra))
        return sxy, syaw, math.sqrt(self._P["z"] + c["q_z_time"] ** 2 * extra)

    def _validity(self, now):
        if self._filt is None:
            return False, "not_initialized"
        if not self._buf:
            return False, "no_odom"
        now = self._buf[-1].stamp if now is None else float(now)
        if now - self._buf[-1].stamp > self.cfg["max_odom_age_s"]:
            return False, "odom_stale"
        age = now - self._last_accept_t
        if age <= self.cfg["max_fix_age_s"]:
            return True, "ok"
        sxy, syaw, _ = self._sigmas(now)
        if sxy > self.cfg["drift_budget_m"] or syaw > self.cfg["drift_budget_deg"]:
            return False, f"no fix for {age:.1f} s and drift budget exceeded (sigma {sxy:.3f} m, {syaw:.2f} deg)"
        return True, "ok (odom-only, within drift budget)"

    def valid(self, now) -> bool:
        return self._validity(now)[0]

    def T_map_odom(self) -> np.ndarray | None:
        """The OUTPUT alignment as a 4x4. Its z is the total height offset NOW (T_px4 + beta for a range height
        reference: it changes when the terrain under the vehicle changes)."""
        if self._out is None or not self._buf:
            return None if self._out is None else self._out.T(0.0)
        return self._out.T(self._That if self.height_ref == "range" else 0.0)

    def pose_map(self, now=None) -> tuple | None:
        """(T_map_body, v_map) from the newest odom sample (extrapolated to `now` with v_odom if a little later)."""
        if self._out is None or not self._buf:
            return None
        e = self._buf[-1]
        T = e.T.copy()
        if now is not None:
            dt = float(now) - e.stamp
            if 0.0 < dt <= self.cfg["max_extrapolate_s"]:
                T[:3, 3] = T[:3, 3] + e.v * dt
        Tm = self._out.map_pose(T, self._That if self.height_ref == "range" else 0.0)
        v = rot_z(self._out.psi) @ e.v
        return Tm, v

    def body_pose_map_at(self, stamp) -> np.ndarray | None:
        """Body pose in map at a past stamp (interpolated odom, FILTER alignment, T_px4 as it was then)."""
        if self._filt is None:
            return None
        od = self._odom_at(float(stamp))
        if od is None:
            return None
        return self._filt.map_pose(od[0], od[2] if self.height_ref == "range" else 0.0)

    def height_reference_z(self) -> float:
        """Map z of the surface PX4's height refers to now (T_px4; 0 unless the range height reference). Its changes
        re-reference the FC's z (a terrain step under the rangefinder): they are not vehicle motion (DN-63)."""
        return float(self._That) if self.height_ref == "range" else 0.0

    def setpoint_to_odom(self, p_map, yaw_map) -> tuple:
        """Map setpoint -> (p_odom (3,), yaw_odom). z uses the height offset under the vehicle NOW (not under the
        setpoint): PX4 tracks height above what is below it now (DN-14)."""
        if self._out is None:
            raise RuntimeError("MapOdomAligner not initialised")
        a = self._out
        p_map = np.asarray(p_map, dtype=np.float64).reshape(3)
        xy = a.R2().T @ (p_map[:2] - a.t)
        z = p_map[2] - self._offset_now()
        return np.array([xy[0], xy[1], z]), wrap(float(yaw_map) - a.psi)

    def status(self, now) -> dict:
        ok, why = self._validity(now)
        d = {"initialized": self._filt is not None, "valid": ok, "valid_reason": why, "height_ref": self.height_ref,
             "height_source": self._hgt_source if self.height_ref == "range" else "offset",
             "n_fixes_accepted": self._n["accepted"], "n_fixes_rejected": self._n["rejected"],
             "n_reanchors": self._n["reanchors"], "n_resets": self._n["resets"], "n_z_skipped": self._n["z_skipped"],
             "n_candidates": len(self._cand), "last_reason": self._last["reason"],
             "last_residual_m": self._last["residual_m"], "last_residual_deg": self._last["residual_deg"],
             "last_residual_z_m": self._last["residual_z_m"], "last_reset": self._last_reset,
             "fix_age_s": None, "sigma_xy_m": None, "sigma_yaw_deg": None, "sigma_z_m": None,
             "slew_pending_m": None, "slew_pending_deg": None, "terrain_px4_z": None, "z_bias_m": None,
             "rng_offset_m": self._m, "beam_surface_z": self._surf, "terrain_ambiguous": self._ambiguous,
             "n_surface_steps": self._n["surface_steps"],
             "n_surface_switch_by_fixes": self._n["surface_switch_by_fixes"],
             "last_surface_step": self._last_surface_step}
        if self._filt is not None and self._buf:
            n = self._buf[-1].stamp if now is None else float(now)
            d["fix_age_s"] = n - self._last_accept_t
            d["sigma_xy_m"], d["sigma_yaw_deg"], d["sigma_z_m"] = self._sigmas(n)
            p = self._buf[-1].T[:3, 3]
            d["slew_pending_m"] = float(np.linalg.norm(self._filt.xy(p) - self._out.xy(p)))
            d["slew_pending_deg"] = math.degrees(wrap(self._filt.psi - self._out.psi))
            d["terrain_px4_z"] = self._That if self.height_ref == "range" else None
            d["z_bias_m"] = self._out.beta
        return d

    def events(self) -> list:
        """Recent init / re-anchor / reset events (JSON-able), oldest first."""
        return list(self._events)

"""The localization state machine: estimators -> accepted fixes -> `pose` + `loc_ok` (ARCHITECTURE §8.3).

Rules (all numbers are config, provisional until tuned on real Scout footage):
- Each frame, the estimators run in order (e.g. ORB PnP, then AprilTag PnP) until one returns a fix that passes its
  own checks (min inliers, reprojection) and the plausibility gate below.
- `loc_ok` is true only while the last accepted fix is at most `loc_lost_hold_s` old. "PnP inliers below threshold
  for > 1 s -> loc_ok = false" (§8.3) is exactly this: every frame below threshold is a frame without a fix.
- Between fixes the pose is HELD (the last accepted one, with its own stamp). Propagating it with the flight
  controller's estimate (§8.3) is the caller's job: pass the propagated pose as `T_prior`; it is used to choose
  keyframes, never output as a fix.
- Plausibility gate while tracking: a fix that would mean moving faster than `max_speed_mps` (plus slack) since the
  last accepted fix is rejected. If the last accepted fix itself was wrong, the gate cannot lock the system out
  forever: after `loc_lost_hold_s` without an accepted fix the state drops to "lost", where the gate is replaced by
- reacquisition: `reacquire_fixes` consecutive frames with mutually consistent fixes before `loc_ok` turns true
  again (one stray PnP solution must not move the Scout).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fix import PoseFix
from .geometry import inv_T, rot_angle_deg

DEFAULTS = {
    "loc_lost_hold_s": 1.0,      # §8.3; the shared value lives in config/flight_demo.yaml safety.loc_lost_hold_s
    "reacquire_fixes": 2,        # provisional
    "max_speed_mps": 2.0,        # provisional plausibility bound (the Scout is commanded <= 0.5 m/s, §8.5)
    "max_rot_rate_dps": 180.0,   # provisional
    "jump_slack_m": 0.15,        # provisional: allowed pose noise between two fixes
    "jump_slack_deg": 10.0,
}


@dataclass
class LocOutput:
    stamp: float
    loc_ok: bool
    fresh: bool                       # this frame produced an accepted fix
    T_map_cam: np.ndarray | None      # latest accepted camera pose (held between fixes); None before the first
    T_map_body: np.ndarray | None     # same, for the body frame (camera mount from config)
    fix_stamp: float | None           # stamp of T_map_cam
    n_inliers: int = 0                # this frame's best attempt
    n_matches: int = 0
    method: str = ""
    reason: str = ""                  # why this frame gave no accepted fix
    state: str = "lost"               # "tracking" | "reacquiring" | "lost"
    fix: PoseFix | None = field(default=None, repr=False)

    @property
    def age_s(self):
        return None if self.fix_stamp is None else self.stamp - self.fix_stamp


class Localizer:
    def __init__(self, estimators, cfg=None, T_body_cam=None):
        """`estimators`: objects with `.estimate(image, T_prior) -> PoseFix`, tried in order.
        `T_body_cam`: camera pose in the body frame (mount extrinsics); identity if None."""
        self.estimators = list(estimators)
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        self.T_cam_body = inv_T(T_body_cam) if T_body_cam is not None else np.eye(4)
        self.reset()

    def reset(self):
        self._acquired = False
        self._last_T = None
        self._last_stamp = None
        self._cand = []                   # consecutive (stamp, T) fixes while reacquiring

    def loc_ok_at(self, stamp: float) -> bool:
        """loc_ok at a given time without a new frame (a node calls this from a timer, so a dead camera stream
        also ends loc_ok). Stamps are in the image clock. "> 1 s" means lost; the epsilon absorbs stamp rounding."""
        return bool(self._acquired and self._last_stamp is not None
                    and (stamp - self._last_stamp) <= self.cfg["loc_lost_hold_s"] + 1e-6)

    def _plausible(self, T_a, t_a, T_b, t_b) -> tuple:
        c = self.cfg
        dt = abs(t_b - t_a)
        dp = float(np.linalg.norm(T_b[:3, 3] - T_a[:3, 3]))
        dr = rot_angle_deg(T_a[:3, :3], T_b[:3, :3])
        if dp > c["max_speed_mps"] * dt + c["jump_slack_m"]:
            return False, f"jump {dp:.2f} m in {dt:.2f} s"
        if dr > c["max_rot_rate_dps"] * dt + c["jump_slack_deg"]:
            return False, f"turn {dr:.1f} deg in {dt:.2f} s"
        return True, ""

    def process(self, image, stamp: float, T_prior=None) -> LocOutput:
        prior = T_prior
        if prior is None:                                  # a reacquisition candidate is the freshest guess
            prior = self._cand[-1][1] if self._cand else self._last_T
        best = None
        for est in self.estimators:
            f = est.estimate(image, prior)
            if f.ok and self._acquired and self._last_T is not None:
                ok, why = self._plausible(self._last_T, self._last_stamp, f.T_map_cam, stamp)
                if not ok:
                    f.ok, f.reason = False, "implausible: " + why
            if best is None or (f.ok and not best.ok) or (not best.ok and f.n_inliers > best.n_inliers):
                best = f
            if f.ok:
                break

        fresh = False
        reason = "" if best is None else best.reason
        if best is not None and best.ok:
            if self._acquired:
                fresh = True
            else:
                if self._cand:
                    ok, why = self._plausible(self._cand[-1][1], self._cand[-1][0], best.T_map_cam, stamp)
                    if not ok:
                        self._cand = []
                        reason = "reacquire: inconsistent with previous fix (" + why + ")"
                self._cand.append((stamp, best.T_map_cam))
                if len(self._cand) >= int(self.cfg["reacquire_fixes"]):
                    self._acquired, fresh, self._cand = True, True, []
                else:
                    reason = reason or f"reacquiring ({len(self._cand)}/{int(self.cfg['reacquire_fixes'])})"
            if fresh:
                self._last_T, self._last_stamp = best.T_map_cam, stamp
        else:
            self._cand = []

        loc_ok = self.loc_ok_at(stamp)
        if self._acquired and not loc_ok:
            self._acquired = False
        if fresh:
            state = "tracking"
        elif self._cand:
            state = "reacquiring"
        else:
            state = "tracking" if loc_ok else "lost"
        T = self._last_T
        return LocOutput(stamp=stamp, loc_ok=bool(loc_ok), fresh=fresh, T_map_cam=T,
                         T_map_body=None if T is None else T @ self.T_cam_body, fix_stamp=self._last_stamp,
                         n_inliers=0 if best is None else best.n_inliers,
                         n_matches=0 if best is None else best.n_matches,
                         method="" if best is None else best.method, reason="" if fresh else reason,
                         state=state, fix=best)

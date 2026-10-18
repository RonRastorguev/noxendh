"""Synthetic evaluation scenarios, shared by tests/test_scout_loc.py and tools/eval_scout_loc_synthetic.py.

Each scenario renders frames from `synthetic_room`, runs them through the real Localizer / estimators and returns
per-frame records plus a summary. Nothing here is used at run time on the Scout.
"""
from __future__ import annotations

import time

import numpy as np

from . import synthetic_room as sr
from .apriltag import TagPnP
from .geometry import look_at, make_T, pose_error, rot_x, rot_y, rot_z
from .landmarks import Keyframe, build_landmark_db
from .localizer import Localizer
from .orb_pnp import OrbPnP

TEST_GAIN = 0.85          # test flight is rendered 15 % darker than the mapping walk (exposure change)


def rebuild_db(ds: sr.Dataset, extrinsic_error=None, map_xyz=None, build_cfg=None):
    """Same keyframe images, different believed poses / map: calibration and map-noise sensitivity without
    re-rendering. Believed camera pose = true pose @ extrinsic_error."""
    E = np.eye(4) if extrinsic_error is None else np.asarray(extrinsic_error)
    kfs = [Keyframe(k.kf_id, k.image, T @ E, k.stamp) for k, T in zip(ds.keyframes, ds.keyframes_true)]
    cfg = dict(build_cfg or {})
    cfg.setdefault("map_spacing_m", 0.03)
    db, _ = build_landmark_db(ds.map_xyz if map_xyz is None else map_xyz, kfs, ds.cam, cfg)
    return db


def extrinsic_perturbation(rot_deg=1.0, trans_m=0.02):
    """A fixed, deliberately non-axis-aligned calibration error: rotation about (1, 1, 1), translation along (1, -1, 1)."""
    a = np.radians(rot_deg) / np.sqrt(3)
    return make_T(rot_x(a) @ rot_y(a) @ rot_z(a), np.array([1.0, -1.0, 1.0]) / np.sqrt(3) * trans_m)


def flight_frames(n=150):
    stamps, body = sr.scout_flight(n=n)
    return stamps, [T @ sr.T_BODY_CAM_SYNTH for T in body]


def run_sequence(loc: Localizer, renderer: sr.Renderer, stamps, poses, gain=TEST_GAIN, noise=2.0, seed=7,
                 room_override=None):
    recs = []
    rend = renderer if room_override is None else sr.Renderer(room_override, renderer.cam, renderer.ss)
    for k, (s, T) in enumerate(zip(stamps, poses)):
        img, _ = rend.render(T, gain=gain, noise_sigma=noise, rng=np.random.default_rng(seed * 7919 + k))
        t0 = time.perf_counter()
        out = loc.process(img, float(s))
        dt = time.perf_counter() - t0
        rec = {"stamp": float(s), "loc_ok": out.loc_ok, "fresh": out.fresh, "inliers": out.n_inliers,
               "matches": out.n_matches, "method": out.method, "reason": out.reason, "ms": dt * 1000.0,
               "state": out.state}
        if out.fresh:
            rec["pos_err_m"], rec["rot_err_deg"] = pose_error(out.T_map_cam, T)
        recs.append(rec)
    return recs


def summarize(recs) -> dict:
    fixes = [r for r in recs if r["fresh"]]
    pe = np.array([r["pos_err_m"] for r in fixes]) if fixes else np.array([np.nan])
    re = np.array([r["rot_err_deg"] for r in fixes]) if fixes else np.array([np.nan])
    inl = np.array([r["inliers"] for r in fixes]) if fixes else np.array([0])
    ms = np.array([r["ms"] for r in recs])
    return {"frames": len(recs), "fixes": len(fixes), "loc_ok_frames": sum(r["loc_ok"] for r in recs),
            "pos_med_m": float(np.median(pe)), "pos_p95_m": float(np.percentile(pe, 95)), "pos_max_m": float(np.max(pe)),
            "rot_med_deg": float(np.median(re)), "rot_p95_deg": float(np.percentile(re, 95)),
            "rot_max_deg": float(np.max(re)), "inl_med": int(np.median(inl)), "inl_min": int(np.min(inl)),
            "ms_med": float(np.median(ms)), "ms_max": float(np.max(ms))}


def tracking(ds: sr.Dataset, db=None, n=150, loc_cfg=None):
    loc = Localizer([OrbPnP(db or ds.db, ds.cam)], loc_cfg)
    stamps, poses = flight_frames(n)
    return run_sequence(loc, ds.renderer, stamps, poses)


def kidnapped(ds: sr.Dataset, every=5, n=150):
    """No prior at all: each frame alone, global keyframe retrieval -> PnP."""
    est = OrbPnP(ds.db, ds.cam)
    stamps, poses = flight_frames(n)
    recs = []
    for k in range(0, n, every):
        img, _ = ds.renderer.render(poses[k], gain=TEST_GAIN, noise_sigma=2.0, rng=np.random.default_rng(31 + k))
        t0 = time.perf_counter()
        fix = est.estimate(img, None)
        rec = {"stamp": float(stamps[k]), "fresh": fix.ok, "loc_ok": fix.ok, "inliers": fix.n_inliers,
               "matches": fix.n_matches, "method": fix.method, "reason": fix.reason,
               "ms": (time.perf_counter() - t0) * 1000.0, "state": ""}
        if fix.ok:
            rec["pos_err_m"], rec["rot_err_deg"] = pose_error(fix.T_map_cam, poses[k])
        recs.append(rec)
    return recs


# --- texture-less view ---------------------------------------------------------------------------------------

BLANK_P0 = np.array([5.5, 4.5, 1.25])          # looking into the room ...
BLANK_YAW0 = float(np.degrees(np.arctan2(2.0 - 4.5, 0.0 - 5.5)))
BLANK_P1 = np.array([5.5, 5.4, 1.25])          # ... then 0.6 m from the plain wall section (wall_yW, x 4.5-8)
BLANK_YAW1 = 90.0


def _yaw_view(p, yaw_deg):
    a = np.radians(yaw_deg)
    return look_at(p, p + np.array([np.cos(a), np.sin(a), 0.0]))


def blank_wall_sequence(rate_hz=10.0, before_s=1.0, turn_s=1.5, hold_s=2.5, back_s=1.5, after_s=1.0):
    """Stamps + true camera poses: look into the room, turn (~115 deg at ~75 deg/s) to face the plain wall up close,
    hold, turn back. Also returns the [start, end) of the hold (fully texture-less) in seconds."""
    y0 = BLANK_YAW0
    y1 = y0 + ((BLANK_YAW1 - y0 + 180.0) % 360.0 - 180.0)          # the short way round
    seg = [(BLANK_P0, y0)] * int(before_s * rate_hz)
    n = int(turn_s * rate_hz)
    seg += [(BLANK_P0 + (BLANK_P1 - BLANK_P0) * i / n, y0 + (y1 - y0) * i / n) for i in range(1, n + 1)]
    hold0 = len(seg) / rate_hz
    seg += [(BLANK_P1, y1)] * int(hold_s * rate_hz)
    hold1 = len(seg) / rate_hz
    n = int(back_s * rate_hz)
    seg += [(BLANK_P1 + (BLANK_P0 - BLANK_P1) * i / n, y1 + (y0 - y1) * i / n) for i in range(1, n + 1)]
    seg += [(BLANK_P0, y0)] * int(after_s * rate_hz)
    stamps = np.arange(len(seg)) / rate_hz
    return stamps, [_yaw_view(p, y) for p, y in seg], (hold0, hold1)


def blank_wall(ds: sr.Dataset, with_tag=False, loc_cfg=None):
    """ORB-only (default) or ORB + AprilTag fallback with a tag on the plain wall. -> (records, (hold0, hold1))."""
    stamps, poses, hold = blank_wall_sequence()
    est = [OrbPnP(ds.db, ds.cam)]
    room = None
    if with_tag:
        room = sr.make_room(0, density=1.0, tags=list(sr.DEFAULT_TAGS) + [sr.Tag(7, "wall_yW", 5.5, 1.75, 0.20)])
        est.append(TagPnP(ds.cam, room.tags))
    loc = Localizer(est, loc_cfg)
    return run_sequence(loc, ds.renderer, stamps, poses, room_override=room), hold


def tag_only(ds: sr.Dataset, n=150):
    """AprilTag PnP alone on the test flight (default tags: one 0.30 m tag per wall)."""
    est = TagPnP(ds.cam, ds.room.tags)
    stamps, poses = flight_frames(n)
    recs = []
    for k in range(n):
        img, _ = ds.renderer.render(poses[k], gain=TEST_GAIN, noise_sigma=2.0, rng=np.random.default_rng(57 + k))
        fix = est.estimate(img)
        rec = {"stamp": float(stamps[k]), "fresh": fix.ok, "loc_ok": fix.ok, "inliers": fix.n_inliers,
               "matches": fix.n_matches, "method": fix.method, "reason": fix.reason, "ms": 0.0, "state": "",
               "tags": fix.info.get("tags", [])}
        if fix.ok:
            rec["pos_err_m"], rec["rot_err_deg"] = pose_error(fix.T_map_cam, poses[k])
        recs.append(rec)
    return recs


def landmark_depth_error(ds: sr.Dataset, every=5) -> dict:
    """Landmark depth vs the renderer's true depth at the same keypoint (relative error), every `every`-th keyframe."""
    errs = []
    for row, T in list(enumerate(ds.keyframes_true))[::every]:
        sel = np.nonzero(ds.db.kf_index == row)[0]
        if len(sel) == 0:
            continue
        _, gt = ds.renderer.render(T)
        uv = np.rint(ds.db.uv[sel]).astype(int)
        g = gt[np.clip(uv[:, 1], 0, gt.shape[0] - 1), np.clip(uv[:, 0], 0, gt.shape[1] - 1)]
        errs.append(np.abs(ds.db.depth[sel] - g) / g)
    e = np.concatenate(errs)
    return {"n": int(len(e)), "rel_med": float(np.median(e)), "rel_p95": float(np.percentile(e, 95)),
            "frac_over_5pct": float(np.mean(e > 0.05))}

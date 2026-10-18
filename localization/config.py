"""YAML config -> camera model + Localizer. The only module here that reads files besides LandmarkDB.load."""
from __future__ import annotations

import os

import numpy as np

from .apriltag import TagPnP, load_tag_map
from .camera import CameraModel
from .geometry import R_FLU_OPTICAL, make_T, rot_x
from .landmarks import LandmarkDB
from .localizer import Localizer
from .orb_pnp import OrbPnP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_configs(loc_path=None, hw_path=None, flight_path=None) -> dict:
    """-> {"loc": scout_loc.yaml, "hw": scout_hw.yaml, "flight": flight_demo.yaml}, repo defaults when None."""
    cdir = os.path.join(ROOT, "config")
    return {"loc": _yaml(loc_path or os.path.join(cdir, "scout_loc.yaml")),
            "hw": _yaml(hw_path or os.path.join(cdir, "scout_hw.yaml")),
            "flight": _yaml(flight_path or os.path.join(cdir, "flight_demo.yaml")),
            "loc_dir": os.path.dirname(os.path.abspath(loc_path or os.path.join(cdir, "scout_loc.yaml")))}


def _resolve(path, base):
    return path if os.path.isabs(path) else os.path.join(base, path)


def camera_from_config(cfgs: dict) -> CameraModel:
    cam = cfgs["loc"].get("camera", {})
    if cam.get("intrinsics_file"):
        return CameraModel.from_dict(_yaml(_resolve(cam["intrinsics_file"], cfgs["loc_dir"])))
    fov = float(cfgs["hw"]["camera_front"]["fov_deg_nominal"])
    axis = cam.get("nominal_fov_axis", "diagonal")
    return CameraModel.from_fov(int(cam["nominal_width"]), int(cam["nominal_height"]), fov, axis,
                                source=f"NOMINAL from scout_hw.yaml: {fov} deg {axis}, pinhole, no distortion")


def body_from_camera(cfgs: dict) -> np.ndarray:
    m = cfgs["loc"].get("camera", {}).get("body_from_camera", {}) or {}
    if m.get("T_body_cam") is not None:
        return np.asarray(m["T_body_cam"], dtype=np.float64)
    if m.get("mount", "forward_optical") != "forward_optical":
        raise ValueError(f"unknown camera mount {m.get('mount')!r}")
    R = R_FLU_OPTICAL @ rot_x(-np.radians(float(m.get("pitch_down_deg", 0.0))))
    return make_T(R, m.get("translation_m", [0.0, 0.0, 0.0]))


def build_localizer(cfgs: dict, db: LandmarkDB | None = None, db_path: str | None = None, camera=None,
                    tags: dict | None = None) -> Localizer:
    loc = cfgs["loc"]
    cam = camera or camera_from_config(cfgs)
    if db is None and db_path:
        db = LandmarkDB.load(db_path)
    est = {}
    if db is not None:
        est["orb_pnp"] = OrbPnP(db, cam, loc.get("orb_pnp"))
    at = dict(loc.get("apriltag") or {})
    if tags is None and at.get("enabled") and at.get("tags_file"):
        tags = load_tag_map(_yaml(_resolve(at["tags_file"], cfgs["loc_dir"])).get("tags", {}))
    if tags:
        est["apriltag"] = TagPnP(cam, tags, {k: v for k, v in at.items() if k not in ("enabled", "tags_file")})
    lcfg = dict(loc.get("localizer") or {})
    order = lcfg.pop("estimators", ["orb_pnp", "apriltag"])
    lcfg["loc_lost_hold_s"] = float(cfgs["flight"].get("safety", {}).get("loc_lost_hold_s", 1.0))
    chosen = [est[n] for n in order if n in est]
    if not chosen:
        raise ValueError("no estimator available: give a landmark DB and/or surveyed tags")
    return Localizer(chosen, lcfg, T_body_cam=body_from_camera(cfgs))

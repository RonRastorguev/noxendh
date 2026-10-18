"""Build the Scout's whole flight stack from the repo config: the ONE factory the sim and the real node share.

    mission, comps = build_flight_stack(vehicle, db, map_xyz, cam, T_body_cam, mission_overrides, log=..., wrap=...)

`db` is the landmark DB (localization.landmarks.LandmarkDB), `map_xyz` the prior map cloud (N, 3) in map, `cam` the
camera model, `T_body_cam` the camera mount. `mission_overrides` is merged over config/scout_mission.yaml (the sim
sets its pad and goal there; the real node takes them from parameters). `wrap(name, obj)` may return a proxy of a
component (the sim records calls through it); it defaults to the identity.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def merge(base: dict, over: dict | None) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def load_mission_config(path=None, overrides=None) -> dict:
    import yaml
    p = path or os.path.join(ROOT, "config", "scout_mission.yaml")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return merge(cfg, overrides)


def build_flight_stack(vehicle, db, map_xyz, cam, T_body_cam, mission_overrides=None, log=None, wrap=None,
                       map_spacing_m=None, apriltags=None):
    from avoidance.config import load_config as load_avoid_config
    from avoidance.looming import LoomingDetector
    from localization.apriltag import TagPnP
    from localization.config import load_configs
    from localization.localizer import Localizer
    from localization.map_align import MapOdomAligner
    from localization.orb_pnp import OrbPnP
    from .follower import Follower
    from .mission import Mission
    from .nav_config import load_nav_config
    from .obstacle_memory import ObstacleMemory
    from .planner import Planner
    from .safety import SafetyGate
    from .static_map import StaticMap

    wrap = wrap or (lambda name, obj: obj)
    cfgs = load_configs()
    loc_cfg = cfgs["loc"]
    lcfg = dict(loc_cfg.get("localizer") or {})
    order = lcfg.pop("estimators", ["orb_pnp", "apriltag"])
    lcfg["loc_lost_hold_s"] = float(cfgs["flight"].get("safety", {}).get("loc_lost_hold_s", 1.0))
    est = {"orb_pnp": OrbPnP(db, cam, loc_cfg.get("orb_pnp"))}
    if apriltags:
        at = {k: v for k, v in (loc_cfg.get("apriltag") or {}).items() if k not in ("enabled", "tags_file")}
        est["apriltag"] = TagPnP(cam, apriltags, at)
    localizer = Localizer([est[n] for n in order if n in est], lcfg, T_body_cam=T_body_cam)
    aligner = MapOdomAligner(loc_cfg.get("map_align"))
    spacing = float(map_spacing_m if map_spacing_m is not None
                    else (loc_cfg.get("landmark_map") or {}).get("map_spacing_m", 0.03))
    looming = LoomingDetector(cam, load_avoid_config(), map_points=map_xyz, map_spacing_m=spacing,
                              T_body_cam=T_body_cam)
    nav = load_nav_config()
    static_map = StaticMap.from_points(map_xyz, nav)
    planner = Planner(static_map, nav)
    follower = Follower(nav, static_map)
    safety = SafetyGate(nav, static_map)
    memory = ObstacleMemory(nav)
    mcfg = load_mission_config(overrides=mission_overrides)
    mission = Mission(mcfg, vehicle, wrap("localizer", localizer), wrap("aligner", aligner),
                      wrap("looming", looming), static_map, wrap("planner", planner), follower, safety, memory,
                      cam=cam, T_body_cam=T_body_cam, log=log)
    comps = {"localizer": localizer, "aligner": aligner, "looming": looming, "static_map": static_map,
             "planner": planner, "follower": follower, "safety": safety, "memory": memory, "nav_cfg": nav,
             "mission_cfg": mcfg}
    return mission, comps

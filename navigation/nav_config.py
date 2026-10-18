"""Navigation config: one flat dict for every navigation module (docs/SCOUT_NAV.md §6, DN-56).

    load_nav_config(nav_path=None, flight_path=None, hw_path=None, avoid_path=None) -> dict
    shared_defaults() -> dict           # the shared keys with their in-code fallbacks (tests without YAML)

Sources (read, never copied):
    config/scout_nav.yaml      this piece's own provisional values (sections flattened; a key lives in one section)
    config/flight_demo.yaml    max_speed_mps, geofence_margin_m, static/dynamic_inflate_extra_m, emergency_hover_s,
                               looming_ttc_s (sanity: ttc_floor_s must exceed it)
    config/scout_hw.yaml       hull_radius_m, hull_half_height_m (airframe placeholders, NEEDS_RON R19)
    config/scout_avoid.yaml    corridor_margin_m (looming.corridor_margin_m: the emergency corridor the follower's
                               TTC law measures over, DN-57)
Every module also keeps a DEFAULTS dict so it runs with cfg=None.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# In-code fallbacks for values owned by other files. Kept equal to those files today; load_nav_config() always
# overwrites them from the files, so these only matter for cfg=None unit tests.
SHARED = {
    "max_speed_mps": 0.5,             # flight_demo.yaml safety.max_speed_mps (horizontal)
    "static_inflate_extra_m": 0.3,    # flight_demo.yaml nav.static_inflate_extra_m
    "dynamic_inflate_extra_m": 0.5,   # flight_demo.yaml nav.dynamic_inflate_extra_m
    "emergency_hover_s": 1.0,         # flight_demo.yaml nav.emergency_hover_s
    "looming_ttc_s": 1.5,             # flight_demo.yaml avoid.looming_ttc_s
    "geofence_margin_m": 0.4,         # flight_demo.yaml safety.geofence_margin_m (null today -> scout_nav default)
    "hull_radius_m": 0.20,            # scout_hw.yaml airframe.hull_radius_m (PLACEHOLDER)
    "hull_half_height_m": 0.06,       # scout_hw.yaml airframe.hull_half_height_m (PLACEHOLDER)
    "corridor_margin_m": 0.15,        # scout_avoid.yaml looming.corridor_margin_m (the emergency corridor, DN-57)
    "min_alt_m": 0.4,                 # scout_nav.yaml band.min_alt_m
    "ceiling_margin_m": 0.5,          # scout_nav.yaml band.ceiling_margin_m
}


def shared_defaults() -> dict:
    return dict(SHARED)


def _yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_nav_config(nav_path=None, flight_path=None, hw_path=None, avoid_path=None) -> dict:
    """-> flat dict for StaticMap / wall_goal / Planner / Follower / SafetyGate / ObstacleMemory."""
    cdir = os.path.join(ROOT, "config")
    nav = _yaml(nav_path or os.path.join(cdir, "scout_nav.yaml"))
    flight = _yaml(flight_path or os.path.join(cdir, "flight_demo.yaml"))
    hw = _yaml(hw_path or os.path.join(cdir, "scout_hw.yaml"))
    avoid = _yaml(avoid_path or os.path.join(cdir, "scout_avoid.yaml")).get("looming", {}) or {}

    cfg = shared_defaults()
    seen = {}
    for section, values in nav.items():
        if not isinstance(values, dict):
            raise ValueError(f"scout_nav.yaml: section {section!r} must be a mapping")
        for k, v in values.items():
            if k in seen:
                raise ValueError(f"scout_nav.yaml: key {k!r} in both {seen[k]!r} and {section!r}")
            seen[k] = section
            cfg[k] = v

    safety, fnav, avoid = flight.get("safety", {}) or {}, flight.get("nav", {}) or {}, flight.get("avoid", {}) or {}
    cfg["max_speed_mps"] = float(safety["max_speed_mps"])
    gm = safety.get("geofence_margin_m")
    cfg["geofence_margin_m"] = float(gm if gm is not None else cfg.get("geofence_margin_m_default", 0.4))
    cfg["geofence_margin_source"] = "flight_demo.yaml" if gm is not None else "scout_nav.yaml default (flight_demo null)"
    cfg["static_inflate_extra_m"] = float(fnav["static_inflate_extra_m"])
    cfg["dynamic_inflate_extra_m"] = float(fnav["dynamic_inflate_extra_m"])
    cfg["emergency_hover_s"] = float(fnav["emergency_hover_s"])
    if avoid.get("looming_ttc_s") is not None:
        cfg["looming_ttc_s"] = float(avoid["looming_ttc_s"])

    air = hw.get("airframe", {}) or {}
    if air.get("hull_radius_m") is None or air.get("hull_half_height_m") is None:
        raise ValueError("scout_hw.yaml airframe.hull_radius_m / hull_half_height_m missing")
    cfg["hull_radius_m"] = float(air["hull_radius_m"])
    cfg["hull_half_height_m"] = float(air["hull_half_height_m"])
    if avoid.get("corridor_margin_m") is not None:
        cfg["corridor_margin_m"] = float(avoid["corridor_margin_m"])

    if cfg.get("memory_s") is None:
        cfg["memory_s"] = float("inf")
    if float(cfg.get("ttc_floor_s", 2.0)) <= cfg["looming_ttc_s"]:
        raise ValueError(f"ttc_floor_s {cfg['ttc_floor_s']} must exceed looming_ttc_s {cfg['looming_ttc_s']}: "
                         "a mapped wall approached at the speed law would trigger the looming emergency")
    if float(cfg.get("stop_margin_m", 0.3)) >= float(cfg.get("standoff_m", 0.8)) - float(cfg.get("arrive_tol_m", 0.1)):
        raise ValueError("stop_margin_m must be < standoff_m - arrive_tol_m, else the goal can never be reached")
    return cfg

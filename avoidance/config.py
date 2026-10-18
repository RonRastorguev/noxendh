"""Looming config: DEFAULTS <- config/scout_avoid.yaml `looming:` <- the shared values read from their owners.

Shared numbers are read, never copied (DH-06, DN-31):
    ttc_emergency_s   <- config/flight_demo.yaml  avoid.looming_ttc_s        (1.5 s)
    hull_radius_m     <- config/scout_hw.yaml     airframe.hull_radius_m     (0.20, PROVISIONAL placeholder)
    hull_half_height_m <- config/scout_hw.yaml    airframe.hull_half_height_m (0.06, PROVISIONAL placeholder)
    top_clear_m       <- hull_radius_m + flight_demo.yaml nav.dynamic_inflate_extra_m (the planner's obstacle
                         inflation: the clear space above an obstacle the Scout needs to fly over it)
A key set explicitly in scout_avoid.yaml (not null) overrides the shared value.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _yaml(path):
    import yaml
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(avoid_path=None, hw_path=None, flight_path=None) -> dict:
    """-> a cfg dict for LoomingDetector (DEFAULTS merged with the YAML files; repo files when None)."""
    from .looming import DEFAULTS
    cdir = os.path.join(ROOT, "config")
    avoid = _yaml(avoid_path or os.path.join(cdir, "scout_avoid.yaml")).get("looming", {}) or {}
    hw = _yaml(hw_path or os.path.join(cdir, "scout_hw.yaml"))
    flight = _yaml(flight_path or os.path.join(cdir, "flight_demo.yaml"))
    cfg = dict(DEFAULTS)
    shared = {}
    ttc = (flight.get("avoid") or {}).get("looming_ttc_s")
    if ttc is not None:
        shared["ttc_emergency_s"] = float(ttc)
    r = (hw.get("airframe") or {}).get("hull_radius_m")
    if r is not None:
        shared["hull_radius_m"] = float(r)
    hh = (hw.get("airframe") or {}).get("hull_half_height_m")
    if hh is not None:
        shared["hull_half_height_m"] = float(hh)
    extra = (flight.get("nav") or {}).get("dynamic_inflate_extra_m")
    if extra is not None:
        shared["top_clear_m"] = float(shared.get("hull_radius_m", cfg["hull_radius_m"])) + float(extra)
    cfg.update(shared)
    cfg.update({k: v for k, v in avoid.items() if v is not None})
    return cfg

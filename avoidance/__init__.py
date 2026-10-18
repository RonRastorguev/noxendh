"""Scout obstacle avoidance from the forward camera: generic looming (docs/SCOUT_NAV.md §5, decisions
docs/decisions_scout_nav/looming.md, DN-20..39). ROS-free, Python 3.9."""
from __future__ import annotations

from .config import load_config
from .looming import DEFAULTS, LoomingDetector

__all__ = ["LoomingDetector", "DEFAULTS", "load_config"]

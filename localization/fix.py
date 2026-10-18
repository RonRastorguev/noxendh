"""The one result type every pose estimator returns (ORB PnP, AprilTag PnP, anything later)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PoseFix:
    ok: bool                          # the estimator's own checks passed (enough inliers, sane reprojection)
    T_map_cam: np.ndarray | None      # 4x4 camera pose in map when a pose was computed (also when not ok)
    n_inliers: int = 0
    n_matches: int = 0
    method: str = ""                  # "orb_local" | "orb_global" | "apriltag"
    reproj_rms_px: float = float("nan")
    reason: str = ""                  # why not ok
    info: dict = field(default_factory=dict)

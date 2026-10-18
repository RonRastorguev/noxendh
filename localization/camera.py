"""Camera model: pinhole intrinsics + OpenCV distortion coefficients.

Until the checkerboard calibration (§6.9) exists, a NOMINAL model is derived from the lens's field of view
(`from_fov`). For the Scout's DECXIN 2633V2 that is 102°, ASSUMED diagonal (config/scout_hw.yaml, DH-06): the
nominal model is pinhole with zero distortion, which a 102° lens certainly is not. It exists so the code paths and
synthetic tests run; real localization needs the calibrated model (docs/SCOUT_LOC.md, "Real data still needed").
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))   # OpenCV (k1, k2, p1, p2[, k3, ...])
    source: str = ""                                                  # where the numbers came from

    def __post_init__(self):
        self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)

    @classmethod
    def from_fov(cls, width: int, height: int, fov_deg: float, axis: str = "diagonal", source: str = ""):
        """Nominal pinhole model: square pixels, principal point at the image centre, no distortion."""
        half = {"diagonal": 0.5 * np.hypot(width, height), "horizontal": 0.5 * width,
                "vertical": 0.5 * height}[axis]
        f = half / np.tan(np.radians(fov_deg) / 2.0)
        return cls(int(width), int(height), f, f, (width - 1) / 2.0, (height - 1) / 2.0, np.zeros(5),
                   source or f"nominal: {fov_deg} deg {axis} FOV, pinhole, no distortion")

    @classmethod
    def from_dict(cls, d: dict):
        """{width, height, fx, fy, cx, cy, dist?, source?} or {width, height, K: 3x3, dist?}."""
        if "K" in d:
            K = np.asarray(d["K"], dtype=np.float64).reshape(3, 3)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        else:
            fx, fy, cx, cy = d["fx"], d["fy"], d["cx"], d["cy"]
        return cls(int(d["width"]), int(d["height"]), float(fx), float(fy), float(cx), float(cy),
                   np.asarray(d.get("dist", np.zeros(5)), dtype=np.float64), str(d.get("source", "")))

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": float(self.fx), "fy": float(self.fy),
                "cx": float(self.cx), "cy": float(self.cy), "dist": [float(v) for v in self.dist],
                "source": self.source}

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def has_distortion(self) -> bool:
        return bool(np.any(self.dist != 0))

    def fov_deg(self) -> dict:
        """Pinhole FOV (ignores distortion)."""
        hw, hh = self.width / 2.0, self.height / 2.0
        return {"horizontal": float(np.degrees(2 * np.arctan(hw / self.fx))),
                "vertical": float(np.degrees(2 * np.arctan(hh / self.fy))),
                "diagonal": float(np.degrees(2 * np.arctan(np.hypot(hw / self.fx, hh / self.fy))))}

    def scaled(self, s: float) -> "CameraModel":
        """Same lens at s times the resolution (distortion coefficients are resolution-independent)."""
        return CameraModel(int(round(self.width * s)), int(round(self.height * s)), self.fx * s, self.fy * s,
                           (self.cx + 0.5) * s - 0.5, (self.cy + 0.5) * s - 0.5, self.dist.copy(),
                           self.source + f" (scaled x{s})")

    def project(self, X_cam) -> np.ndarray:
        """(N, 3) camera-frame points (z > 0) -> (N, 2) pixels, distortion included."""
        X = np.asarray(X_cam, dtype=np.float64).reshape(-1, 3)
        if not self.has_distortion:
            return np.stack([self.fx * X[:, 0] / X[:, 2] + self.cx, self.fy * X[:, 1] / X[:, 2] + self.cy], 1)
        import cv2
        uv, _ = cv2.projectPoints(X.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.dist)
        return uv.reshape(-1, 2)

    def normalized(self, uv) -> np.ndarray:
        """(N, 2) pixels -> (N, 2) undistorted normalized coordinates (x/z, y/z)."""
        uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
        if not self.has_distortion:
            return np.stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy], 1)
        import cv2
        return cv2.undistortPoints(uv.reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)

    def normalized_bounds(self) -> tuple:
        """(xmin, xmax, ymin, ymax) of normalized coordinates seen by the image (border sampled, undistorted).
        Used to drop points outside the view BEFORE distorting them: a polynomial distortion model folds
        far-off-axis points back into the image."""
        n = 16
        xs = np.linspace(0, self.width - 1, n)
        ys = np.linspace(0, self.height - 1, n)
        border = np.concatenate([np.stack([xs, np.zeros(n)], 1), np.stack([xs, np.full(n, self.height - 1.0)], 1),
                                 np.stack([np.zeros(n), ys], 1), np.stack([np.full(n, self.width - 1.0), ys], 1)])
        nb = self.normalized(border)
        return float(nb[:, 0].min()), float(nb[:, 0].max()), float(nb[:, 1].min()), float(nb[:, 1].max())

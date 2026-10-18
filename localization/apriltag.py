"""AprilTag fallback (ARCHITECTURE §8.3): tags at surveyed poses in `map` -> tag-corner PnP -> camera pose.

Same `PoseFix` as ORB PnP, so the Localizer (and nav) cannot tell which one answered except by `method`.
Detection is OpenCV's ArUco module with an AprilTag dictionary (default 36h11). Tag frame = OpenCV's marker /
SOLVEPNP_IPPE_SQUARE convention: origin at the tag centre, x right, y up, z out of the tag; `size_m` is the
black square's edge. Surveyed poses are a config input (how the tags get surveyed: docs/SCOUT_LOC.md).
"""
from __future__ import annotations

import cv2
import numpy as np

from .camera import CameraModel
from .features import to_gray
from .fix import PoseFix
from .geometry import inv_T, make_T, transform

DEFAULTS = {
    "family": "DICT_APRILTAG_36h11",
    "min_tags": 1,
    "min_tag_px": 40.0,          # provisional: smallest tag edge (pixels) to trust (synthetic: <40 px -> 10-50 cm)
    "max_reproj_rms_px": 2.0,    # provisional
    "corner_refine": "subpix",    # "subpix" | "apriltag" | "none"; synthetic: subpix far more accurate (DL-09)
    # A single square has two pose solutions (IPPE); far away / face-on they fit almost equally well and the wrong
    # one is metres off. Without a prior pose to choose, a lone tag is accepted only when the second solution's
    # reprojection error is at least this many times the first's (provisional).
    "single_tag_ambiguity_ratio": 3.0,
    "single_tag_max_spread_m": 0.10,   # ... unless both solutions put the camera within this distance of each other
    "prior_max_dist_m": 1.0,      # with a prior: the chosen solution must be this close to it (provisional)
}


def make_detector(family: str, corner_refine: str = "apriltag"):
    """-> callable(gray) -> (corners, ids, rejected). OpenCV >= 4.7 has ArucoDetector; 4.6 (Ubuntu 24.04's
    python3-opencv, i.e. the ROS side) has the older function API. Same detector either way."""
    aruco = cv2.aruco
    d = aruco.getPredefinedDictionary(getattr(aruco, family))
    refine = {"apriltag": aruco.CORNER_REFINE_APRILTAG, "subpix": aruco.CORNER_REFINE_SUBPIX,
              "none": aruco.CORNER_REFINE_NONE}[corner_refine]
    if hasattr(aruco, "ArucoDetector"):
        params = aruco.DetectorParameters()
        params.cornerRefinementMethod = refine
        det = aruco.ArucoDetector(d, params)
        return det.detectMarkers
    params = aruco.DetectorParameters_create()
    params.cornerRefinementMethod = refine
    return lambda gray: aruco.detectMarkers(gray, d, parameters=params)


def marker_image(family: str, tag_id: int, side_px: int) -> np.ndarray:
    """The tag's black square (1-cell black border), side_px wide. OpenCV 4.7+ / 4.6 API names differ."""
    aruco = cv2.aruco
    d = aruco.getPredefinedDictionary(getattr(aruco, family))
    if hasattr(aruco, "generateImageMarker"):
        return aruco.generateImageMarker(d, tag_id, side_px, borderBits=1)
    return aruco.drawMarker(d, tag_id, side_px, borderBits=1)


class TagPnP:
    def __init__(self, camera: CameraModel, tags: dict, cfg=None):
        """`tags`: {tag_id: {"T_map_tag": 4x4, "size_m": float}}."""
        self.cam = camera
        self.cfg = dict(DEFAULTS)
        self.cfg.update(cfg or {})
        self.tags = {int(k): {"T_map_tag": np.asarray(v["T_map_tag"], dtype=np.float64),
                              "size_m": float(v["size_m"])} for k, v in tags.items()}
        self._detect = make_detector(self.cfg["family"], self.cfg["corner_refine"])

    @staticmethod
    def tag_corners(size_m) -> np.ndarray:
        h = size_m / 2.0
        return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])   # TL, TR, BR, BL

    def detect(self, image) -> list:
        """-> [(tag_id, corners (4,2) px)] for surveyed tags big enough to trust."""
        corners, ids, _ = self._detect(to_gray(image))
        out = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.reshape(-1)):
            c = c.reshape(4, 2).astype(np.float64)
            edge = min(np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4))
            if int(i) in self.tags and edge >= self.cfg["min_tag_px"]:
                out.append((int(i), c))
        return out

    def estimate(self, image, T_prior=None) -> PoseFix:
        dets = self.detect(image)
        if len(dets) < int(self.cfg["min_tags"]):
            return PoseFix(False, None, 0, 0, "apriltag", reason=f"{len(dets)} surveyed tags seen")
        obj = np.concatenate([transform(self.tags[i]["T_map_tag"], self.tag_corners(self.tags[i]["size_m"]))
                              for i, _ in dets])
        img = np.concatenate([c for _, c in dets])
        K, dist = self.cam.K, self.cam.dist
        info = {"tags": [i for i, _ in dets]}
        if len(dets) == 1:
            # IPPE on the tag's own frame (well-posed for a single square, 2 solutions), then move to map
            i, c = dets[0]
            n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(self.tag_corners(self.tags[i]["size_m"]), c, K, dist,
                                                            flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not n_sol:
                return PoseFix(False, None, 0, 4, "apriltag", reason="IPPE failed")
            T_tag = inv_T(self.tags[i]["T_map_tag"])
            sols = [make_T(cv2.Rodrigues(rv)[0], tv.reshape(3)) @ T_tag for rv, tv in zip(rvecs, tvecs)]
            errs = np.asarray(errs, dtype=np.float64).reshape(-1)
            pick = 0
            ratio = float(errs[1] / max(errs[0], 1e-9)) if len(sols) > 1 else float("inf")
            info["ippe_ratio"] = ratio
            spread = float(np.linalg.norm(inv_T(sols[0])[:3, 3] - inv_T(sols[1])[:3, 3])) if len(sols) > 1 else 0.0
            info["ippe_spread_m"] = spread
            if len(sols) > 1 and T_prior is not None:
                d = [np.linalg.norm(inv_T(Tc)[:3, 3] - np.asarray(T_prior)[:3, 3]) for Tc in sols]
                pick = int(np.argmin(d))
                if d[pick] > self.cfg["prior_max_dist_m"]:
                    return PoseFix(False, inv_T(sols[pick]), 4, 4, "apriltag", reason="single tag: no solution "
                                   f"near the prior ({d[pick]:.2f} m)", info=info)
            elif ratio < self.cfg["single_tag_ambiguity_ratio"] and spread > self.cfg["single_tag_max_spread_m"]:
                return PoseFix(False, inv_T(sols[0]), 4, 4, "apriltag",
                               reason=f"single tag ambiguous (IPPE error ratio {ratio:.1f}, solutions {spread:.2f} m "
                                      "apart)", info=info)
            rvec, _ = cv2.Rodrigues(sols[pick][:3, :3])
            tvec = sols[pick][:3, 3].reshape(3, 1)
        else:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                return PoseFix(False, None, 0, len(obj), "apriltag", reason="SQPnP failed")
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        R, _ = cv2.Rodrigues(rvec)
        T = inv_T(make_T(R, tvec.reshape(3)))
        fix = PoseFix(True, T, len(obj), len(obj), "apriltag", rms, info=info)
        if rms > self.cfg["max_reproj_rms_px"]:
            fix.ok, fix.reason = False, f"tag reprojection rms {rms:.2f} px > {self.cfg['max_reproj_rms_px']}"
        return fix


def load_tag_map(d: dict) -> dict:
    """Config form -> {id: {"T_map_tag", "size_m"}}. Each tag: {size_m, T_map_tag: 4x4} or
    {size_m, position: [x,y,z], quat_xyzw: [x,y,z,w]}."""
    from .geometry import quat_xyzw_to_R
    out = {}
    for k, v in (d or {}).items():
        if "T_map_tag" in v:
            T = np.asarray(v["T_map_tag"], dtype=np.float64)
        else:
            T = make_T(quat_xyzw_to_R(v["quat_xyzw"]), v["position"])
        out[int(k)] = {"T_map_tag": T, "size_m": float(v["size_m"])}
    return out

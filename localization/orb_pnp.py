"""Per-frame pose from ORB matches to the landmark database + `cv2.solvePnPRansac` (ARCHITECTURE §8.3).

Candidate landmarks:
- local: the keyframes nearest a prior pose (the last fix, or a prior the caller supplies, e.g. propagated with the
  flight controller's odometry), scored by distance plus viewing-direction difference;
- global (no prior, or the local attempt failed): every query descriptor votes for the keyframe of its nearest
  landmark; the top-voted keyframes become the candidates. Fine for a room-sized map; a bag-of-words index is the
  upgrade for big maps.

Matching uses a SPATIAL ratio test: the same physical point is a landmark in several keyframes with nearly equal
descriptors, so Lowe's ratio against the plain second-best would reject exactly the best-seen points. The second-best
is instead the best match that lies more than `same_point_radius_m` away from the first.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import features
from .camera import CameraModel
from .fix import PoseFix
from .geometry import inv_T, make_T
from .landmarks import LandmarkDB

DEFAULTS = {
    "orb": {"n_features": 1500},
    "n_local_keyframes": 8,
    "view_angle_weight_m_per_rad": 1.0,   # keyframe score = distance + w * angle between optical axes
    "max_view_angle_deg": 75.0,           # keyframes looking further away than this are never local candidates
    "n_global_keyframes": 10,
    "global_vote_per_kf": 200,            # landmarks per keyframe that take part in the global vote (speed)
    "global_on_local_fail": True,
    "knn": 4,
    "ratio": 0.8,                         # provisional (Lowe's 0.7-0.8)
    "same_point_radius_m": 0.05,
    "max_hamming": 64,
    "ransac_reproj_px": 4.0,              # provisional, at the camera's native resolution
    "ransac_iterations": 300,
    "ransac_confidence": 0.999,
    "min_inliers": 25,                    # provisional: the "PnP inliers below threshold" of §8.3
    "min_inlier_ratio": 0.0,              # 0 = off
    "max_reproj_rms_px": 3.0,             # after LM refinement, on inliers
}


def merged_cfg(cfg) -> dict:
    out = dict(DEFAULTS)
    out["orb"] = dict(DEFAULTS["orb"])
    for k, v in (cfg or {}).items():
        if k == "orb":
            out["orb"].update(v or {})
        else:
            out[k] = v
    return out


class OrbPnP:
    def __init__(self, db: LandmarkDB, camera: CameraModel, cfg=None):
        self.db = db
        self.cam = camera
        self.cfg = merged_cfg(cfg)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._kf_pos = db.kf_poses[:, :3, 3]
        self._kf_axis = db.kf_poses[:, :3, 2]
        # a fixed, spread-out subset of every keyframe's landmarks for the global vote
        rng = np.random.default_rng(0)
        per = int(self.cfg["global_vote_per_kf"])
        parts = []
        for r in range(db.n_keyframes):
            idx = db.landmarks_of([r])
            parts.append(idx if len(idx) <= per else np.sort(rng.choice(idx, per, replace=False)))
        self._vote_idx = np.concatenate(parts) if parts else np.zeros(0, np.int64)

    # --- candidates -----------------------------------------------------------------------------------
    def local_keyframes(self, T_prior) -> np.ndarray:
        c = self.cfg
        d = np.linalg.norm(self._kf_pos - T_prior[:3, 3], axis=1)
        cosang = np.clip(self._kf_axis @ T_prior[:3, 2], -1.0, 1.0)
        ang = np.arccos(cosang)
        score = d + c["view_angle_weight_m_per_rad"] * ang
        score[ang > np.radians(c["max_view_angle_deg"])] = np.inf
        order = np.argsort(score)
        order = order[np.isfinite(score[order])]
        return order[: int(c["n_local_keyframes"])]

    def global_keyframes(self, des) -> np.ndarray:
        if len(des) == 0 or len(self._vote_idx) == 0:
            return np.zeros(0, np.int64)
        m = self._matcher.match(des, self.db.descriptors[self._vote_idx])
        good = [self._vote_idx[x.trainIdx] for x in m if x.distance <= self.cfg["max_hamming"]]
        votes = np.bincount(self.db.kf_index[good], minlength=self.db.n_keyframes) if good else np.zeros(1)
        order = np.argsort(-votes)
        order = order[votes[order] > 0]
        return order[: int(self.cfg["n_global_keyframes"])]

    # --- matching -------------------------------------------------------------------------------------
    def match(self, des, lm_idx) -> tuple:
        """-> (query indices, landmark indices) after the spatial ratio test, one query per landmark."""
        c = self.cfg
        if len(des) == 0 or len(lm_idx) < 2:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        train = self.db.descriptors[lm_idx]
        pts = self.db.points[lm_idx]
        knn = self._matcher.knnMatch(des, train, k=min(int(c["knn"]), len(lm_idx)))
        r2 = c["same_point_radius_m"] ** 2
        q_out, l_out, d_out = [], [], []
        for cand in knn:
            if not cand or cand[0].distance > c["max_hamming"]:
                continue
            m0 = cand[0]
            p0 = pts[m0.trainIdx]
            second = None
            for m in cand[1:]:
                dp = pts[m.trainIdx] - p0
                if float(dp @ dp) > r2:
                    second = m
                    break
            if second is not None and m0.distance >= c["ratio"] * second.distance:
                continue
            q_out.append(m0.queryIdx)
            l_out.append(lm_idx[m0.trainIdx])
            d_out.append(m0.distance)
        if not q_out:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        q, l, d = np.array(q_out), np.array(l_out), np.array(d_out)
        order = np.lexsort((d, l))                      # keep the best query per landmark
        q, l = q[order], l[order]
        first = np.ones(len(l), bool)
        first[1:] = l[1:] != l[:-1]
        return q[first], l[first]

    # --- PnP --------------------------------------------------------------------------------------------
    def solve(self, uv, X, method) -> PoseFix:
        c = self.cfg
        n = len(uv)
        if n < 6:
            return PoseFix(False, None, 0, n, method, reason=f"too few matches ({n})")
        K, dist = self.cam.K, self.cam.dist
        obj = np.ascontiguousarray(X, dtype=np.float64).reshape(-1, 1, 3)
        img = np.ascontiguousarray(uv, dtype=np.float64).reshape(-1, 1, 2)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, dist, iterationsCount=int(c["ransac_iterations"]),
                                                 reprojectionError=float(c["ransac_reproj_px"]),
                                                 confidence=float(c["ransac_confidence"]),
                                                 flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or len(inl) < 4:
            return PoseFix(False, None, 0 if inl is None else len(inl), n, method, reason="ransac failed")
        inl = inl.reshape(-1)
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, dist, rvec, tvec)
        # re-score every match with the refined pose
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
        R, _ = cv2.Rodrigues(rvec)
        zc = (obj.reshape(-1, 3) @ R.T + tvec.reshape(1, 3))[:, 2]
        inl = np.nonzero((err < c["ransac_reproj_px"]) & (zc > 0))[0]
        T_cam_map = make_T(R, tvec.reshape(3))
        T = inv_T(T_cam_map)
        rms = float(np.sqrt(np.mean(err[inl] ** 2))) if len(inl) else float("inf")
        fix = PoseFix(True, T, int(len(inl)), n, method, rms, info={"inlier_idx": inl})
        if len(inl) < c["min_inliers"]:
            fix.ok, fix.reason = False, f"inliers {len(inl)} < {c['min_inliers']}"
        elif c["min_inlier_ratio"] and len(inl) < c["min_inlier_ratio"] * n:
            fix.ok, fix.reason = False, f"inlier ratio {len(inl) / n:.2f} < {c['min_inlier_ratio']}"
        elif rms > c["max_reproj_rms_px"]:
            fix.ok, fix.reason = False, f"reprojection rms {rms:.2f} px > {c['max_reproj_rms_px']}"
        return fix

    def _attempt(self, kps_uv, des, kf_rows, method) -> PoseFix:
        lm = self.db.landmarks_of(kf_rows)
        q, l = self.match(des, lm)
        fix = self.solve(kps_uv[q], self.db.points[l].astype(np.float64), method)
        fix.info["keyframes"] = [int(self.db.kf_ids[r]) for r in kf_rows]
        return fix

    def estimate(self, image, T_prior=None) -> PoseFix:
        kps, des = features.detect(image, self.cfg["orb"])
        if len(kps) < 6:
            return PoseFix(False, None, 0, 0, "orb_local" if T_prior is not None else "orb_global",
                           reason=f"too few keypoints ({len(kps)})", info={"keypoints": len(kps)})
        uv = np.array([k.pt for k in kps], dtype=np.float64)
        fix = None
        if T_prior is not None:
            rows = self.local_keyframes(np.asarray(T_prior, dtype=np.float64))
            if len(rows):
                fix = self._attempt(uv, des, rows, "orb_local")
                if fix.ok or not self.cfg["global_on_local_fail"]:
                    fix.info["keypoints"] = len(kps)
                    return fix
        rows = self.global_keyframes(des)
        gfix = self._attempt(uv, des, rows, "orb_global") if len(rows) else \
            PoseFix(False, None, 0, 0, "orb_global", reason="no keyframe votes")
        if fix is not None and not gfix.ok and fix.n_inliers > gfix.n_inliers:
            gfix = fix                                     # report the more informative failure
        gfix.info["keypoints"] = len(kps)
        return gfix

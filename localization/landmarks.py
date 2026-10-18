"""Visual landmark database anchored in the lidar map (ARCHITECTURE §8.2).

For each keyframe (camera image + camera pose in `map`): ORB keypoints, then each keypoint gets the depth of the
map cloud rendered into that keyframe with a z-buffer (no triangulation), and is back-projected into `map`.
Keypoints without a trustworthy depth (map hole, depth edge, too far) are dropped.

On disk: one `.npz` (numpy only, loads anywhere):
    points (N,3) float32 map frame | descriptors (N,32) uint8 | kf_index (N,) int32 (row in the keyframe arrays)
    uv (N,2) float32 keypoint pixel in its keyframe | depth (N,) float32 camera z at the keypoint
    kf_ids (K,) int64 | kf_poses (K,4,4) float64 T_map_cam | kf_stamps (K,) float64
    camera_json, meta_json: JSON strings (the mapping camera model; build config + provenance)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from . import features
from .camera import CameraModel
from .depth_render import depth_at, render_depth
from .geometry import transform

FORMAT_VERSION = 1

DEFAULTS = {
    "orb": {"n_features": 2000},     # merged over features.DEFAULTS
    "map_spacing_m": 0.03,           # typical point spacing of the map cloud (after any voxel filter); sets splat size
    "splat_scale": 1.0,
    "max_splat_px": 8,
    "depth_window_px": 3,            # 7x7 window (synthetic: fewest wrong depths, DL-05)
    "max_rel_depth_spread": 0.05,    # provisional: keypoints on depth edges are dropped
    "min_valid_frac": 0.6,
    "min_depth_m": 0.2,
    "max_depth_m": 8.0,              # provisional: far landmarks carry more depth/pose error
}


def merged_cfg(cfg) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


@dataclass
class Keyframe:
    kf_id: int
    image: np.ndarray            # BGR or gray, mapping camera resolution
    T_map_cam: np.ndarray        # 4x4
    stamp: float = 0.0


@dataclass
class LandmarkDB:
    points: np.ndarray
    descriptors: np.ndarray
    kf_index: np.ndarray
    uv: np.ndarray
    depth: np.ndarray
    kf_ids: np.ndarray
    kf_poses: np.ndarray
    kf_stamps: np.ndarray
    camera: CameraModel
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        order = np.argsort(self.kf_index, kind="stable")
        if np.any(order != np.arange(len(order))):
            for name in ("points", "descriptors", "kf_index", "uv", "depth"):
                setattr(self, name, getattr(self, name)[order])
        k = len(self.kf_ids)
        starts = np.searchsorted(self.kf_index, np.arange(k), side="left")
        ends = np.searchsorted(self.kf_index, np.arange(k), side="right")
        self._slices = list(zip(starts.tolist(), ends.tolist()))

    def __len__(self):
        return len(self.points)

    @property
    def n_keyframes(self) -> int:
        return len(self.kf_ids)

    def landmarks_of(self, kf_rows) -> np.ndarray:
        """Landmark indices of the given keyframe rows."""
        parts = [np.arange(*self._slices[int(r)]) for r in kf_rows]
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    def counts(self) -> np.ndarray:
        return np.array([e - s for s, e in self._slices])

    def save(self, path: str) -> None:
        np.savez_compressed(
            path, format_version=np.int64(FORMAT_VERSION), points=self.points.astype(np.float32),
            descriptors=self.descriptors.astype(np.uint8), kf_index=self.kf_index.astype(np.int32),
            uv=self.uv.astype(np.float32), depth=self.depth.astype(np.float32),
            kf_ids=self.kf_ids.astype(np.int64), kf_poses=self.kf_poses.astype(np.float64),
            kf_stamps=self.kf_stamps.astype(np.float64), camera_json=np.array(json.dumps(self.camera.to_dict())),
            meta_json=np.array(json.dumps(self.meta, default=str)))

    @classmethod
    def load(cls, path: str) -> "LandmarkDB":
        with np.load(path, allow_pickle=False) as z:
            ver = int(z["format_version"])
            if ver != FORMAT_VERSION:
                raise ValueError(f"{path}: landmark format {ver}, this code reads {FORMAT_VERSION}")
            return cls(points=z["points"], descriptors=z["descriptors"], kf_index=z["kf_index"].astype(np.int64),
                       uv=z["uv"], depth=z["depth"], kf_ids=z["kf_ids"], kf_poses=z["kf_poses"],
                       kf_stamps=z["kf_stamps"], camera=CameraModel.from_dict(json.loads(str(z["camera_json"]))),
                       meta=json.loads(str(z["meta_json"])))


def landmarks_for_keyframe(map_xyz, kf: Keyframe, cam: CameraModel, cfg) -> tuple:
    """-> (points_map (n,3), descriptors (n,32), uv (n,2), depth (n,), stats dict) for one keyframe."""
    c = merged_cfg(cfg)
    kps, des = features.detect(kf.image, c["orb"])
    stats = {"kf_id": kf.kf_id, "keypoints": len(kps)}
    if len(kps) == 0:
        stats.update(landmarks=0, no_depth=0)
        return np.zeros((0, 3)), np.zeros((0, 32), np.uint8), np.zeros((0, 2)), np.zeros(0), stats
    depth_img = render_depth(map_xyz, kf.T_map_cam, cam, c["map_spacing_m"], c["splat_scale"], c["max_splat_px"],
                             max_z=c["max_depth_m"] * 1.5)
    uv = np.array([k.pt for k in kps], dtype=np.float64)
    d = depth_at(depth_img, uv, c["depth_window_px"], c["max_rel_depth_spread"], c["min_valid_frac"])
    ok = np.isfinite(d) & (d >= c["min_depth_m"]) & (d <= c["max_depth_m"])
    xy = cam.normalized(uv[ok])
    X_cam = np.column_stack([xy * d[ok, None], d[ok]])
    pts = transform(kf.T_map_cam, X_cam)
    stats.update(landmarks=int(ok.sum()), no_depth=int((~ok).sum()))
    return pts, des[ok], uv[ok], d[ok], stats


def build_landmark_db(map_xyz, keyframes, cam: CameraModel, cfg=None, meta=None) -> tuple:
    """-> (LandmarkDB, per-keyframe stats list). `keyframes`: iterable of Keyframe (camera poses in map)."""
    c = merged_cfg(cfg)
    map_xyz = np.asarray(map_xyz, dtype=np.float64).reshape(-1, 3)
    P, D, KI, UV, Z, ids, poses, stamps, all_stats = [], [], [], [], [], [], [], [], []
    for row, kf in enumerate(keyframes):
        pts, des, uv, d, st = landmarks_for_keyframe(map_xyz, kf, cam, c)
        P.append(pts)
        D.append(des)
        UV.append(uv)
        Z.append(d)
        KI.append(np.full(len(pts), row, np.int64))
        ids.append(int(kf.kf_id))
        poses.append(np.asarray(kf.T_map_cam, dtype=np.float64))
        stamps.append(float(kf.stamp))
        all_stats.append(st)
    m = dict(meta or {})
    m.update({"build_cfg": c, "n_map_points": int(len(map_xyz)), "format_version": FORMAT_VERSION})
    db = LandmarkDB(points=np.concatenate(P).astype(np.float32) if P else np.zeros((0, 3), np.float32),
                    descriptors=np.concatenate(D) if D else np.zeros((0, 32), np.uint8),
                    kf_index=np.concatenate(KI) if KI else np.zeros(0, np.int64),
                    uv=np.concatenate(UV).astype(np.float32) if UV else np.zeros((0, 2), np.float32),
                    depth=np.concatenate(Z).astype(np.float32) if Z else np.zeros(0, np.float32),
                    kf_ids=np.asarray(ids, np.int64), kf_poses=np.asarray(poses).reshape(-1, 4, 4),
                    kf_stamps=np.asarray(stamps, np.float64), camera=cam, meta=m)
    return db, all_stats


def select_keyframes(poses, min_dist_m=0.3, min_angle_deg=15.0) -> list:
    """Indices of frames to keep as keyframes along a mapping walk: a new keyframe whenever the camera moved
    min_dist_m or turned min_angle_deg since the last kept one (both provisional)."""
    from .geometry import rot_angle_deg
    keep = []
    for i, T in enumerate(poses):
        if not keep:
            keep.append(i)
            continue
        L = np.asarray(poses[keep[-1]])
        T = np.asarray(T)
        if (np.linalg.norm(T[:3, 3] - L[:3, 3]) >= min_dist_m
                or rot_angle_deg(T[:3, :3], L[:3, :3]) >= min_angle_deg):
            keep.append(i)
    return keep

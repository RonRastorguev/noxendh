"""Build the visual landmark database from the prior lidar map (ARCHITECTURE §8.2).

    py -3.9 tools/build_landmark_map.py --map map.pcd --keyframes keyframes.json --out data/landmarks.npz
        [--intrinsics cam.yaml] [--extrinsics lidar_cam.yaml] [--voxel 0.03] [--config config/scout_loc.yaml]

    # or from a trajectory + timestamped frames (keyframes are then selected by motion):
    py -3.9 tools/build_landmark_map.py --map map.pcd --trajectory traj_tum.txt --frames frames.csv
        --extrinsics lidar_cam.yaml --out data/landmarks.npz [--time-offset 0.0]

Inputs
- `--map`: the prior map cloud, PCD, frame `map` (perception/io.py reads ascii/binary PCD).
- `--keyframes`: JSON {"pose_kind": "T_map_lidar" | "T_map_cam", "keyframes": [{"id", "image", "stamp"?, "pose": 4x4}]}.
  Image paths are relative to the JSON file. With T_map_lidar poses, `--extrinsics` is required:
  camera pose = lidar pose x extrinsics (§8.2).
- `--trajectory`: TUM text "stamp x y z qx qy qz qw" per line (lidar poses in map); `--frames`: CSV "stamp,image".
  Each frame's pose is interpolated at stamp + time_offset; frames outside the trajectory are skipped. How the real
  Point-LIO output becomes TUM text is not settled yet (docs/SCOUT_LOC.md).
- `--extrinsics`: YAML with `T_lidar_cam` (camera pose in the lidar frame) or `T_cam_lidar` (the inverse; what a
  solvePnP of lidar points in the image returns, §6.9).
- `--intrinsics`: camera YAML {width, height, fx, fy, cx, cy, dist}; default = the config's camera (nominal model).

Prints a JSON summary; exit 0 = database written, 1 = nothing usable (no landmarks), 2 = bad input.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from localization import config as loc_config  # noqa: E402
from localization.camera import CameraModel  # noqa: E402
from localization.geometry import inv_T, interpolate_pose, make_T, quat_xyzw_to_R  # noqa: E402
from localization.landmarks import Keyframe, build_landmark_db, select_keyframes  # noqa: E402


def _yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_extrinsics(path) -> np.ndarray:
    """-> T_lidar_cam."""
    d = _yaml(path)
    if "T_lidar_cam" in d:
        return np.asarray(d["T_lidar_cam"], dtype=np.float64)
    if "T_cam_lidar" in d:
        return inv_T(np.asarray(d["T_cam_lidar"], dtype=np.float64))
    raise ValueError(f"{path}: needs T_lidar_cam or T_cam_lidar (4x4)")


def read_tum(path):
    stamps, poses = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = [float(x) for x in line.replace(",", " ").split()]
            stamps.append(v[0])
            poses.append(make_T(quat_xyzw_to_R(v[4:8]), v[1:4]))
    order = np.argsort(stamps)
    return np.asarray(stamps)[order], [poses[i] for i in order]


def read_frames_csv(path):
    out = []
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#") or row[0].strip().lower() == "stamp":
                continue
            img = row[1].strip()
            out.append((float(row[0]), img if os.path.isabs(img) else os.path.join(base, img)))
    return out


def keyframes_from_json(path, T_lidar_cam):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    kind = d.get("pose_kind", "T_map_cam")
    if kind == "T_map_lidar" and T_lidar_cam is None:
        raise ValueError("keyframe poses are T_map_lidar: --extrinsics is required")
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for k in d["keyframes"]:
        T = np.asarray(k["pose"], dtype=np.float64)
        if kind == "T_map_lidar":
            T = T @ T_lidar_cam
        img = k["image"] if os.path.isabs(k["image"]) else os.path.join(base, k["image"])
        out.append((int(k["id"]), float(k.get("stamp", 0.0)), img, T))
    return out


def keyframes_from_trajectory(traj_path, frames_path, T_lidar_cam, time_offset, min_dist, min_angle):
    stamps, poses = read_tum(traj_path)
    cand = []
    for s, img in read_frames_csv(frames_path):
        T = interpolate_pose(stamps, poses, s + time_offset)
        if T is not None:
            cand.append((s, img, T @ T_lidar_cam))
    keep = select_keyframes([c[2] for c in cand], min_dist, min_angle)
    return [(j, cand[i][0], cand[i][1], cand[i][2]) for j, i in enumerate(keep)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--map", required=True)
    ap.add_argument("--keyframes")
    ap.add_argument("--trajectory")
    ap.add_argument("--frames")
    ap.add_argument("--time-offset", type=float, default=0.0, help="added to frame stamps before pose lookup (s)")
    ap.add_argument("--extrinsics")
    ap.add_argument("--intrinsics")
    ap.add_argument("--voxel", type=float, default=None, help="voxel-filter the map first (m); sets map spacing")
    ap.add_argument("--config", default=None, help="scout_loc.yaml (default: config/scout_loc.yaml)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import cv2
    from perception.io import read_pcd

    try:
        cfgs = loc_config.load_configs(a.config)
        bcfg = dict(cfgs["loc"].get("landmark_map") or {})
        cam = CameraModel.from_dict(_yaml(a.intrinsics)) if a.intrinsics else loc_config.camera_from_config(cfgs)
        T_lc = load_extrinsics(a.extrinsics) if a.extrinsics else None
        xyz, inten = read_pcd(a.map)
        if a.voxel:
            from perception.preprocess import voxel_downsample
            xyz, inten = voxel_downsample(xyz, inten, a.voxel)
            bcfg["map_spacing_m"] = a.voxel
        if a.keyframes:
            kf_list = keyframes_from_json(a.keyframes, T_lc)
        elif a.trajectory and a.frames:
            if T_lc is None:
                raise ValueError("--trajectory poses are lidar poses: --extrinsics is required")
            kf_list = keyframes_from_trajectory(a.trajectory, a.frames, T_lc, a.time_offset,
                                                float(bcfg.get("keyframe_min_dist_m", 0.3)),
                                                float(bcfg.get("keyframe_min_angle_deg", 15.0)))
        else:
            raise ValueError("give --keyframes, or --trajectory and --frames")
    except (OSError, ValueError, KeyError) as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 2

    def gen():
        for kf_id, stamp, img_path, T in kf_list:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise OSError(f"cannot read {img_path}")
            if (img.shape[1], img.shape[0]) != (cam.width, cam.height):
                raise ValueError(f"{img_path}: {img.shape[1]}x{img.shape[0]}, camera model is {cam.width}x{cam.height}")
            yield Keyframe(kf_id, img, T, stamp)

    for k in ("keyframe_min_dist_m", "keyframe_min_angle_deg"):
        bcfg.pop(k, None)
    try:
        db, stats = build_landmark_db(xyz, gen(), cam, bcfg, meta={
            "map": os.path.abspath(a.map), "keyframes": a.keyframes, "trajectory": a.trajectory,
            "frames": a.frames, "time_offset_s": a.time_offset, "camera_source": cam.source,
            "extrinsics": a.extrinsics})
    except (OSError, ValueError) as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 2
    counts = db.counts()
    summary = {"out": a.out, "map_points": int(len(xyz)), "keyframes": int(db.n_keyframes), "landmarks": int(len(db)),
               "landmarks_per_kf": {"min": int(counts.min()) if len(counts) else 0,
                                    "median": float(np.median(counts)) if len(counts) else 0,
                                    "max": int(counts.max()) if len(counts) else 0},
               "keypoints_without_depth": int(sum(s["no_depth"] for s in stats)), "camera": cam.source}
    if len(db) == 0:
        print(json.dumps(summary, indent=1))
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    db.save(a.out)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

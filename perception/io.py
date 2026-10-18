"""Scan folders on disk (ARCHITECTURE §5): `cloud.pcd` + `image.png` + `meta.json`.

No ROS, no Open3D: a scan loads with numpy alone (DH-08). OpenCV is imported only to read or write
the image. A folder is written under a temporary name and renamed into place, so a reader never
sees a half-written scan.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

CLOUD_FILE = "cloud.pcd"
IMAGE_FILE = "image.png"
META_FILE = "meta.json"

IDENTITY_4X4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


@dataclass
class Scan:
    """One accumulated capture. `xyz` is (N, 3) float32 in `meta["frame"]`; `intensity` is (N,) float32."""
    scan_id: str
    xyz: np.ndarray
    intensity: np.ndarray
    image: np.ndarray | None = None          # HxWx3 uint8, BGR (OpenCV order)
    meta: dict = field(default_factory=dict)

    @property
    def sensor_pose(self) -> np.ndarray:
        return np.asarray(self.meta.get("sensor_pose", IDENTITY_4X4), dtype=np.float64)


def new_scan_id(now: datetime | None = None) -> str:
    """Sortable and unique: UTC time to the second plus 4 random hex chars."""
    now = now or datetime.now(timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ") + "_" + secrets.token_hex(2)


def make_meta(scan_id: str, *, accumulation_s: float, is_baseline: bool, frame: str = "sensor",
              sensor_pose=None, roi=None, timestamp: str | None = None, **extra) -> dict:
    """The spec's meta keys (§5) plus `frame`; anything in `extra` is added as-is (DH-08)."""
    meta = {
        "scan_id": scan_id,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "accumulation_s": float(accumulation_s),
        "sensor_pose": [[float(v) for v in row] for row in (sensor_pose if sensor_pose is not None else IDENTITY_4X4)],
        "roi": roi,
        "is_baseline": bool(is_baseline),
        "frame": frame,
    }
    meta.update(extra)
    return meta


# --- PCD -------------------------------------------------------------------------------------------

def write_pcd(path: str, xyz: np.ndarray, intensity: np.ndarray) -> None:
    """Binary PCD v0.7 with float32 fields x y z intensity."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    intensity = np.asarray(intensity, dtype=np.float32).reshape(-1)
    if len(intensity) != len(xyz):
        raise ValueError(f"xyz has {len(xyz)} points but intensity has {len(intensity)}")
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4")])
    rec["x"], rec["y"], rec["z"], rec["intensity"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], intensity
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {len(rec)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(rec)}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


_PCD_TYPES = {("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4",
              ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4"}


def read_pcd(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read an ascii or binary PCD with at least x y z. Returns (xyz float32 (N,3), intensity float32 (N,));
    intensity is zeros when the file has no intensity field. binary_compressed is not supported."""
    with open(path, "rb") as f:
        header = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: PCD header ended before DATA")
            text = line.decode("ascii", "replace").strip()
            if not text or text.startswith("#"):
                continue
            key, _, value = text.partition(" ")
            header[key.upper()] = value.split()
            if key.upper() == "DATA":
                break
        body = f.read()
    names = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(c) for c in header.get("COUNT", ["1"] * len(names))]
    if any(c != 1 for c in counts):
        raise ValueError(f"{path}: COUNT != 1 is not supported")
    n = int(header["POINTS"][0])
    dtype = np.dtype([(nm, _PCD_TYPES[(t, s)]) for nm, t, s in zip(names, types, sizes)])
    mode = header["DATA"][0].lower()
    if mode == "binary":
        rec = np.frombuffer(body, dtype=dtype, count=n)
    elif mode == "ascii":
        rows = np.loadtxt(body.decode("ascii").splitlines(), ndmin=2) if n else np.zeros((0, len(names)))
        rec = np.empty(n, dtype=dtype)
        for i, nm in enumerate(names):
            rec[nm] = rows[:, i]
    else:
        raise ValueError(f"{path}: DATA {mode} is not supported")
    for need in ("x", "y", "z"):
        if need not in names:
            raise ValueError(f"{path}: no '{need}' field")
    xyz = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float32)
    intensity = rec["intensity"].astype(np.float32) if "intensity" in names else np.zeros(n, np.float32)
    return xyz, intensity


# --- scan folders ----------------------------------------------------------------------------------

def save_scan(scans_dir: str, scan: Scan) -> str:
    """Write `scans_dir/<scan_id>/` atomically and return its path. Refuses to overwrite a scan."""
    final = os.path.join(scans_dir, scan.scan_id)
    if os.path.exists(final):
        raise FileExistsError(final)
    tmp = os.path.join(scans_dir, f".tmp_{scan.scan_id}")
    os.makedirs(tmp)
    try:
        write_pcd(os.path.join(tmp, CLOUD_FILE), scan.xyz, scan.intensity)
        meta = dict(scan.meta)
        meta["scan_id"] = scan.scan_id
        meta["n_points"] = int(len(scan.xyz))
        meta["has_image"] = scan.image is not None
        if scan.image is not None:
            import cv2
            if not cv2.imwrite(os.path.join(tmp, IMAGE_FILE), scan.image):
                raise OSError("cv2.imwrite failed for the scan image")
        with open(os.path.join(tmp, META_FILE), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, sort_keys=True)
        os.replace(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final


def load_scan(path: str, load_image: bool = True) -> Scan:
    with open(os.path.join(path, META_FILE), encoding="utf-8") as f:
        meta = json.load(f)
    xyz, intensity = read_pcd(os.path.join(path, CLOUD_FILE))
    image = None
    image_path = os.path.join(path, IMAGE_FILE)
    if load_image and os.path.exists(image_path):
        import cv2
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"cannot read {image_path}")
    return Scan(scan_id=meta.get("scan_id", os.path.basename(path.rstrip("/\\"))),
                xyz=xyz, intensity=intensity, image=image, meta=meta)


def list_scans(scans_dir: str) -> list[str]:
    """Scan ids in `scans_dir`, oldest first (ids sort by time). Skips in-progress `.tmp_` folders."""
    if not os.path.isdir(scans_dir):
        return []
    return sorted(d for d in os.listdir(scans_dir)
                  if not d.startswith(".") and os.path.isfile(os.path.join(scans_dir, d, META_FILE)))

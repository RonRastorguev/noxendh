"""The table frame: where the diorama is, found from a lidar scan, so every scan is expressed in it and cropped to it.

A fixed rig still has to know where its subject is. Instead of typing a sensor pose, `fit_table` finds it:

- "up" comes from the L2's own IMU. At rest the accelerometer reads +1 g along up, in the cloud's axes (the L2's
  cloud shares its IMU's axes, verified in earlier bench work). So any mount works (dome up, dome
  down, tilted) without typing an angle.
- The tabletop is the highest large horizontal surface below the sensor (objects on it are small; the floor is lower).
  A least-squares plane fit then refines "up" (the IMU only seeds it).
- The rectangle is the outline of that surface (cv2.minAreaRect), grown to at least the diorama's configured size,
  away from the lidar (the far end is the one the shelves hide).

Table frame (the same convention as perception/synthetic.py): origin at the tabletop centre, z up with the tabletop
at z = 0, y along the long side pointing AWAY from the lidar (so the far end, with the shelves, is +y), x = y × z.
With the lidar beside the long side (level with the middle, the recommended rig), y's sign is chosen so the lidar
sits on the -x side.

Scans keep this as `meta.frame = "table"` and `meta.sensor_pose` = the sensor's pose in the table frame, which is
what the pipeline already reads. For a moving sensor the same slot is filled by the map pose: "stationary" is a
pose that never changes, not a code path (DH-23).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import yaml

FILE_DEFAULT = "config/table_frame.yaml"


@dataclass
class TableFit:
    ok: bool
    reason: str = ""
    T_table_sensor: np.ndarray | None = None      # 4x4: p_table = R p_sensor + t (= the sensor's pose in the table frame)
    size_m: tuple | None = None                   # (width x, length y) used for the ROI
    measured_size_m: tuple | None = None          # outline of the visible tabletop
    roi: dict | None = None
    quality: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def up_from_accel(accel) -> np.ndarray:
    a = np.asarray(accel, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n < 1e-6:
        raise ValueError(f"no usable accelerometer vector: {accel}")
    return a / n


def _level_rotation(up: np.ndarray) -> np.ndarray:
    """A rotation R with R @ up = +z (heading arbitrary)."""
    z = up / np.linalg.norm(up)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def transform(xyz: np.ndarray, T) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    return (np.asarray(xyz, dtype=np.float64) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)


def fit_table(xyz: np.ndarray, up, *, expected_size_m=(0.6, 1.0), max_range_m: float = 3.0, cell_m: float = 0.02,
              plane_tol_m: float = 0.015, min_area_frac: float = 0.4, max_tilt_deg: float = 5.0,
              roi_margin_m: float = 0.03, roi_height_m: float = 0.45, min_clearance_m: float = 0.05) -> TableFit:
    """Find the tabletop in a sensor-frame cloud (sensor at the origin). See the module docstring."""
    import cv2

    W_exp, L_exp = float(expected_size_m[0]), float(expected_size_m[1])
    min_area = min_area_frac * W_exp * L_exp
    p = np.asarray(xyz, dtype=np.float64)
    p = p[np.linalg.norm(p, axis=1) <= max_range_m]
    R0 = _level_rotation(up_from_accel(up))
    q = p @ R0.T                                              # heights relative to the sensor, sensor at 0
    below = q[q[:, 2] < -min_clearance_m]
    if len(below) < 500:
        return TableFit(False, f"almost nothing below the lidar within {max_range_m:g} m ({len(below)} points): "
                               "it cannot see the table from this mount (a dome-up L2 sees only ~6° below level)")

    # candidate horizontal surfaces = peaks of the height histogram, tried from the highest down
    bins = np.arange(below[:, 2].min(), -min_clearance_m + 0.005, 0.005)
    hist, edges = np.histogram(below[:, 2], bins=bins)
    smooth = np.convolve(hist, np.ones(3) / 3, mode="same")
    floor_count = max(200, int(0.01 * len(below)))
    peaks = [i for i in range(1, len(smooth) - 1)
             if smooth[i] >= floor_count and smooth[i] >= smooth[i - 1] and smooth[i] > smooth[i + 1]]
    tried = []
    for i in sorted(peaks, key=lambda i: -edges[i]):
        h = 0.5 * (edges[i] + edges[i + 1])
        band = below[np.abs(below[:, 2] - h) < plane_tol_m]
        if len(band) < 200:
            continue
        lo = band[:, :2].min(axis=0)
        ij = np.floor((band[:, :2] - lo) / cell_m).astype(int)
        grid = np.zeros(ij.max(axis=0) + 1, np.uint8)
        grid[ij[:, 0], ij[:, 1]] = 1
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
        if n_cc < 2:
            continue
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = float(stats[best, cv2.CC_STAT_AREA]) * cell_m ** 2
        tried.append((round(-h, 3), round(area, 3)))
        if area < min_area:
            continue
        pts = band[labels[ij[:, 0], ij[:, 1]] == best]
        break
    else:
        seen = ", ".join(f"{a:.2f} m² at {d:.2f} m below" for d, a in tried[:4]) or "none"
        return TableFit(False, f"no horizontal surface of at least {min_area:.2f} m² below the lidar "
                               f"(largest per height: {seen})")

    # refine "up" with the surface itself; the IMU only seeded it
    centroid = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    n = vt[2] if vt[2][2] > 0 else -vt[2]
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
    if tilt > max_tilt_deg:
        return TableFit(False, f"the surface found is {tilt:.1f}° off level by the lidar's IMU (limit {max_tilt_deg:g}°)")
    Ln = _level_rotation(n)                                   # IMU-level frame -> surface-level frame
    R1 = Ln @ R0                                              # sensor -> levelled (z = table normal)
    pl = pts @ Ln.T                                           # the surface points, levelled
    h = float(np.median(pl[:, 2]))
    rms = float(np.sqrt(np.mean((pl[:, 2] - h) ** 2)))

    (cx, cy), (w, l), ang = cv2.minAreaRect(pl[:, :2].astype(np.float32))
    a = np.radians(ang)
    u = np.array([np.cos(a), np.sin(a)])                      # side of length w
    v = np.array([-np.sin(a), np.cos(a)])                     # side of length l
    if w > l:
        u, v, w, l = v, u, l, w
    c = np.array([cx, cy])                                    # sensor is at (0, 0) in this frame
    y = v if float(v @ c) >= 0 else -v                        # long axis, pointing away from the lidar
    warnings = []
    if abs(float(y @ c)) < 0.10:                              # level with the middle: put the lidar on the -x side
        if float(np.array([y[1], -y[0]]) @ c) < 0:
            y = -y
    x = np.array([y[1], -y[0]])                               # x = y × z (right-handed, z up)
    W_meas, L_meas = float(w), float(l)
    # Beside the long side (the recommended spot) or over the centre, "away from the lidar" along the length is
    # undefined; that only matters when the measured length is short and the rectangle must be grown one way.
    if abs(float(y @ c)) < 0.10 and L_meas < L_exp - 0.02:
        warnings.append(f"only {L_meas:.2f} m of the table's length was seen and the lidar is level with its middle: "
                        "the crop was extended in a guessed direction. Check the crop covers both ends.")
    W, L = max(W_meas, W_exp), max(L_meas, L_exp)
    near_edge = c - y * (L_meas / 2)                          # the best-observed edge stays where it was measured
    c = near_edge + y * (L / 2)
    if W_meas > 1.25 * W_exp or L_meas > 1.25 * L_exp:
        # a bigger table, or the floor: still a level frame, which is all the crop needs (Ron 2026-10-17: "it should
        # allow a floor size scan but i can just crop it to be what i want"). The dashboard's Crop picks the area.
        warnings.append(f"the surface found is {W_meas:.1f} × {L_meas:.1f} m, {-h:.2f} m below the lidar (a larger "
                        f"table or the floor), not the {W_exp:g} × {L_exp:g} m diorama: use Crop to pick the area")

    Rz = np.array([[x[0], x[1], 0.0], [y[0], y[1], 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ R1
    t = -Rz @ np.array([c[0], c[1], h])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    sensor_in_table = T[:3, 3]
    roi = {"min": [round(-W / 2 - roi_margin_m, 4), round(-L / 2 - roi_margin_m, 4), -0.02],
           "max": [round(W / 2 + roi_margin_m, 4), round(L / 2 + roi_margin_m, 4), round(roi_height_m, 4)]}
    quality = {"plane_rms_m": round(rms, 4), "n_surface_points": int(len(pl)),
               "visible_area_m2": round(float(len(np.unique(np.floor(pl[:, :2] / cell_m).astype(int), axis=0))) * cell_m ** 2, 3),
               "imu_vs_surface_tilt_deg": round(tilt, 2),
               "lidar_in_table_m": [round(float(v_), 3) for v_ in sensor_in_table],
               "lidar_height_above_table_m": round(float(sensor_in_table[2]), 3),
               "surfaces_tried": tried}
    return TableFit(True, "", T, (round(W, 3), round(L, 3)), (round(W_meas, 3), round(L_meas, 3)), roi, quality, warnings)


# --- persistence ------------------------------------------------------------------------------------------------

def save(path: str, fit: TableFit, *, scan_id: str, accel_mean=None) -> dict:
    doc = {
        "calib_id": f"table-{scan_id}",
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "from_scan": scan_id,
        "T_table_sensor": [[round(float(v), 6) for v in row] for row in fit.T_table_sensor],
        "size_m": list(fit.size_m),
        "measured_size_m": list(fit.measured_size_m),
        "roi": fit.roi,
        "imu_accel_mean": None if accel_mean is None else [round(float(v), 4) for v in accel_mean],
        "quality": fit.quality,
        "warnings": fit.warnings,
    }
    doc = json.loads(json.dumps(doc, default=float))           # numpy scalars -> plain floats for safe_dump
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Written by tools/calibrate_table.py (or the dashboard's Calibrate button). Do not hand-edit:\n"
                "# re-run the calibration. Scans captured with it are in the table frame (perception/table_frame.py).\n")
        yaml.safe_dump(doc, f, sort_keys=False)
    return doc


def load(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


CROP_FILE_DEFAULT = "config/crop.yaml"


def _path(cfg: dict, key: str, default: str, root: str) -> str:
    rel = cfg.get(key, default)
    return rel if os.path.isabs(rel) else os.path.join(root, rel)


def frame_id(applied_cfg: dict) -> str:
    """The frame a crop box is written in: the table calibration's id, or "sensor" before any calibration."""
    return applied_cfg.get("table_calib_id") or "sensor"


def save_crop(cfg: dict, root: str, roi: dict, *, frame: str, from_scan: str | None = None) -> dict:
    """Axis-aligned {min, max} or rotated {center, size, yaw_deg, z} (preprocess.roi_mask, DH-33)."""
    if roi.get("center") is not None:
        cx, cy = [float(v) for v in roi["center"]]
        w, l = [float(v) for v in roi["size"]]
        zlo, zhi = [float(v) for v in roi["z"]]
        yaw = float(roi.get("yaw_deg", 0.0))
        if not (w > 0 and l > 0 and zlo < zhi):
            raise ValueError(f"crop size must be positive and bottom below top: size {w} x {l}, z {zlo}..{zhi}")
        box = {"center": [round(cx, 4), round(cy, 4)], "size": [round(w, 4), round(l, 4)],
               "yaw_deg": round(((yaw + 180) % 360) - 180, 2), "z": [round(zlo, 4), round(zhi, 4)]}
    else:
        lo, hi = [float(v) for v in roi["min"]], [float(v) for v in roi["max"]]
        if not all(a < b for a, b in zip(lo, hi)):
            raise ValueError(f"crop min must be below max on every axis: {lo} / {hi}")
        box = {"min": [round(v, 4) for v in lo], "max": [round(v, 4) for v in hi]}
    doc = {"frame_id": frame, "set_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "from_scan": from_scan,
           "roi": box}
    path = _path(cfg, "crop_file", CROP_FILE_DEFAULT, root)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Written by the dashboard's Crop tool: the box every scan is cropped to, in the frame named by\n"
                "# frame_id (a table calibration, or 'sensor'). A crop for another frame is ignored. Reset = delete.\n")
        yaml.safe_dump(doc, f, sort_keys=False)
    return doc


def load_crop(cfg: dict, root: str) -> dict | None:
    return load(_path(cfg, "crop_file", CROP_FILE_DEFAULT, root))


def apply(cfg: dict, root: str) -> dict:
    """cfg with the calibrated table frame applied (frame, sensor_pose, roi, table_calib_id), or cfg unchanged
    (table_calib_id None) when there is no calibration yet; then Ron's hand crop (config/crop.yaml) replaces the
    ROI when it was drawn in that same frame (`crop_source`: "hand" / "table" / "config"). Read on every capture,
    so a new calibration or crop takes effect without restarting anything."""
    out = dict(cfg)
    doc = load(_path(cfg, "table_frame_file", FILE_DEFAULT, root))
    if doc:
        out.update(frame="table", sensor_pose=doc["T_table_sensor"], roi=doc["roi"], table_calib_id=doc["calib_id"])
        out["crop_source"] = "table"
    else:
        out["table_calib_id"] = None
        out["crop_source"] = "config"
    crop = load_crop(cfg, root)
    if crop and crop.get("frame_id") == frame_id(out):
        out["roi"] = crop["roi"]
        out["crop_source"] = "hand"
    return out


def calibrate_scan(scan, cfg: dict, root: str) -> tuple[TableFit, dict | None]:
    """Fit the table in a SENSOR-frame scan (its meta carries the IMU's mean accelerometer vector) and, on success,
    write the calibration file. Shared by tools/calibrate_table.py and the dashboard's Calibrate button."""
    if scan.meta.get("frame", "sensor") != "sensor":
        return TableFit(False, f"calibration needs a scan in the sensor frame, this one is in '{scan.meta.get('frame')}'"), None
    tfc = cfg.get("table_fit", {}) or {}
    accel = (scan.meta.get("lidar") or {}).get("imu_accel_mean") or tfc.get("up_sensor")
    if accel is None:
        return TableFit(False, "this scan has no IMU reading (and the config has no table_fit.up_sensor)"), None
    fit = fit_table(scan.xyz, accel, expected_size_m=tfc.get("expected_size_m", (0.6, 1.0)),
                    max_range_m=float(tfc.get("max_range_m", 3.0)), roi_margin_m=float(tfc.get("roi_margin_m", 0.03)),
                    roi_height_m=float(tfc.get("roi_height_m", 0.45)))
    if not fit.ok:
        return fit, None
    rel = cfg.get("table_frame_file", FILE_DEFAULT)
    return fit, save(rel if os.path.isabs(rel) else os.path.join(root, rel), fit, scan_id=scan.scan_id, accel_mean=accel)


def _as_oriented(roi: dict) -> dict:
    if roi.get("center") is not None:
        return dict(roi)
    lo, hi = roi["min"], roi["max"]
    return {"center": [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2], "size": [hi[0] - lo[0], hi[1] - lo[1]],
            "yaw_deg": 0.0, "z": [lo[2], hi[2]]}


def lidar_unmoved(scan_a, scan_b, *, max_tilt_deg: float = 0.5, near_m: float = 0.05, min_near_frac: float = 0.5,
                  max_range_m: float = 3.0, n: int = 20000) -> tuple[bool, str]:
    """Did the lidar stay put between two SENSOR-frame scans? Same gravity direction (IMU) AND the same scene in
    the same place: at least half of one scan's points within 5 cm of the other's (within 3 m). Measured on Ron's
    rig 2026-10-18: two scans 26 min apart, rig untouched, 68 %; after the rig was remounted, 20 %. People walking
    through keep an untouched lidar well below 100 %. A small slide (a few cm) can still pass, so a carried crop is
    reported to Ron to check, never applied silently."""
    from scipy.spatial import cKDTree
    a = (scan_a.meta.get("lidar") or {}).get("imu_accel_mean")
    b = (scan_b.meta.get("lidar") or {}).get("imu_accel_mean")
    if a is None or b is None:
        return False, "no IMU reading to compare"
    a, b = np.asarray(a, float), np.asarray(b, float)
    tilt = float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))
    if tilt > max_tilt_deg:
        return False, f"the lidar's tilt changed by {tilt:.1f}°"
    pa = scan_a.xyz[np.linalg.norm(scan_a.xyz, axis=1) < max_range_m]
    pb = scan_b.xyz[np.linalg.norm(scan_b.xyz, axis=1) < max_range_m]
    if len(pa) < 100 or len(pb) < 100:
        return False, "too few points to compare"
    pick = np.random.default_rng(0).choice(len(pb), min(n, len(pb)), replace=False)
    d, _ = cKDTree(pa).query(pb[pick], distance_upper_bound=near_m)
    frac = float(np.isfinite(d).mean())
    if frac < min_near_frac:
        return False, f"the scene moved relative to the lidar (only {frac:.0%} of it lines up)"
    return True, f"tilt {tilt:.2f}°, {frac:.0%} of the scene lines up"


def carry_crop(crop_doc: dict, old_table: dict, new_table: dict, *, max_tilt_deg: float = 2.0) -> dict | None:
    """The hand crop drawn in `old_table`'s frame, re-expressed in `new_table`'s frame, for a lidar that did NOT
    move (check with lidar_unmoved first): p_new = S_new · S_old⁻¹ · p_old. The box turns about the vertical with
    the frames; None if the two frames' verticals differ by more than max_tilt_deg (a box can only turn about z)."""
    S_old, S_new = np.asarray(old_table["T_table_sensor"], float), np.asarray(new_table["T_table_sensor"], float)
    M = S_new @ np.linalg.inv(S_old)
    if np.degrees(np.arccos(np.clip(M[2, 2], -1, 1))) > max_tilt_deg:
        return None
    box = _as_oriented(crop_doc["roi"])
    zmid = (box["z"][0] + box["z"][1]) / 2
    c = M @ np.array([box["center"][0], box["center"][1], zmid, 1.0])
    yaw = box.get("yaw_deg", 0.0) + np.degrees(np.arctan2(M[1, 0], M[0, 0]))
    half = (box["z"][1] - box["z"][0]) / 2
    return {"center": [float(c[0]), float(c[1])], "size": [float(v) for v in box["size"]],
            "yaw_deg": float(((yaw + 180) % 360) - 180), "z": [float(c[2] - half), float(c[2] + half)]}

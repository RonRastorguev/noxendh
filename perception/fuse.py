"""Build ChangeEvents (ARCHITECTURE §5) from geometric + visual changes, with before/after image crops (§6.7).

A camera calibration ({K, dist, T_cam_from_sensor, image_size}; tools/calibrate_extrinsics.py writes it on Saturday,
perception.synthetic.camera_model() gives one for synthetic scans) lets 3D boxes be projected into the image for crops
and image regions be placed in the map. Without one, geometric events still exist but get the whole frame as their
crop and `crop_located: false`; appearance events keep their pixel box and have no map position (`bbox_map: null`).
"""
from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone

import numpy as np


def new_event_id() -> str:
    return "evt_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + secrets.token_hex(3)


def project_box(cam: dict | None, lo, hi):
    """Pixel rectangle (x0, y0, x1, y1) covering a 3D box's 8 corners, or None (no calibration / behind the camera)."""
    if not cam:
        return None
    import cv2
    K, T = np.asarray(cam["K"], float), np.asarray(cam["T_cam_from_sensor"], float)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    c = corners @ T[:3, :3].T + T[:3, 3]
    if np.any(c[:, 2] <= 0.01):
        return None
    uv, _ = cv2.projectPoints(c, np.zeros(3), np.zeros(3), K, np.asarray(cam.get("dist", [0, 0, 0, 0, 0]), float))
    uv = uv.reshape(-1, 2)
    w, h = cam["image_size"]
    x0, y0 = np.clip(uv.min(axis=0), 0, [w - 1, h - 1])
    x1, y1 = np.clip(uv.max(axis=0), 0, [w - 1, h - 1])
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return int(x0), int(y0), int(np.ceil(x1)), int(np.ceil(y1))


def backproject_region(cam: dict | None, rect, xyz: np.ndarray):
    """Map-frame box of the lidar points that project inside an image rectangle (x0, y0, x1, y1), or None (§6.6)."""
    if not cam or xyz is None or len(xyz) == 0:
        return None
    K, T = np.asarray(cam["K"], float), np.asarray(cam["T_cam_from_sensor"], float)
    c = xyz @ T[:3, :3].T + T[:3, 3]
    front = c[:, 2] > 0.01
    uv = (c[front] @ K.T)
    uv = uv[:, :2] / uv[:, 2:3]
    x0, y0, x1, y1 = rect
    inside = (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    pts = xyz[front][inside]
    if len(pts) < 5:
        return None
    # the nearest surface in that window = what the camera actually sees there (§6.6: "depth of the nearest lidar points")
    zc = c[front][inside][:, 2]
    near = pts[zc <= np.percentile(zc, 30)]
    lo, hi = near.min(axis=0), near.max(axis=0)
    return {"center": [round(float(v), 4) for v in (lo + hi) / 2], "size": [round(float(v), 4) for v in hi - lo]}


def save_crop(img, rect, path: str, pad: int = 16, min_size: int = 64):
    """Padded crop to PNG (whole frame when rect is None). Returns the path or None."""
    if img is None:
        return None
    import cv2
    h, w = img.shape[:2]
    if rect is None:
        crop = img
    else:
        x0, y0, x1, y1 = rect
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w, half_h = max((x1 - x0) / 2 + pad, min_size / 2), max((y1 - y0) / 2 + pad, min_size / 2)
        a0, b0 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
        a1, b1 = int(min(w, cx + half_w)), int(min(h, cy + half_h))
        crop = img[b0:b1, a0:a1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, crop)
    return path


def _bbox_dict(lo, hi) -> dict:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return {"center": [round(float(v), 4) for v in (lo + hi) / 2], "size": [round(float(v), 4) for v in hi - lo]}


def build_events(*, scan_id: str, baseline_id: str, association, visual, before_img, after_img, cam: dict | None,
                 crops_dir: str, cloud_for_visual: np.ndarray | None = None) -> list:
    """ChangeEvent dicts (§5), classification left empty (classify/ fills it), crops written under crops_dir."""
    now = datetime.now(timezone.utc).isoformat()
    events = []

    def base_event(kind):
        return {"id": new_event_id(), "scan_id": scan_id, "baseline_id": baseline_id, "kind": kind, "bbox_map": None,
                "moved_from": None, "crop_before": None, "crop_after": None, "crop_located": False, "geo_score": None,
                "classification": None, "feedback": "none", "created_at": now}

    def crops(ev, rect_before, rect_after):
        ev["crop_before"] = save_crop(before_img, rect_before, os.path.join(crops_dir, ev["id"] + "_before.png"))
        ev["crop_after"] = save_crop(after_img, rect_after, os.path.join(crops_dir, ev["id"] + "_after.png"))
        ev["crop_located"] = rect_before is not None or rect_after is not None

    for m in association.moved:
        ev = base_event("moved")
        ev["bbox_map"] = _bbox_dict(m.added.bbox_min, m.added.bbox_max)
        ev["moved_from"] = {"center": _bbox_dict(m.removed.bbox_min, m.removed.bbox_max)["center"]}
        ev["geo_score"] = round(float(np.linalg.norm(m.displacement[:2])) * 100, 1)            # cm moved
        ev["association_score"] = round(m.score, 3)
        crops(ev, project_box(cam, m.removed.bbox_min, m.removed.bbox_max), project_box(cam, m.added.bbox_min, m.added.bbox_max))
        events.append(ev)
    for kind, clusters in (("added", association.added), ("removed", association.removed)):
        for c in clusters:
            ev = base_event(kind)
            ev["bbox_map"] = _bbox_dict(c.bbox_min, c.bbox_max)
            ev["geo_score"] = round(float(max(c.size)) * 100, 1)                                 # cm, largest extent
            rect = project_box(cam, c.bbox_min, c.bbox_max)
            crops(ev, rect, rect)
            events.append(ev)
    if visual is not None:
        for r in visual.regions:
            ev = base_event("appearance")
            x, y, w, h = r.bbox
            rect = (x, y, x + w, y + h)
            ev["bbox_px"] = [int(v) for v in r.bbox]
            ev["bbox_map"] = backproject_region(cam, rect, cloud_for_visual)
            ev["geo_score"] = round(r.mean_delta_e, 1)                                           # mean Delta E
            crops(ev, rect, rect)
            events.append(ev)
    return events

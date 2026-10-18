"""Camera calibration from four clicked table corners (DH-24).

The lidar calibration (perception/table_frame.py) already knows the tabletop's rectangle in 3D. Its four corners,
clicked in the camera image, fix the camera's pose (and, with a guess from the lens's field of view, its focal
length): a single-view planar calibration. The result is stored relative to the LIDAR, not the table
(T_cam_from_lidar), so it survives a table recalibration as long as the camera and lidar share a bracket.

Clicks may come in any order. They are put in cyclic order around their centroid, and every corner assignment
(4 rotations x 2 directions) is tried. The physically right one puts the camera above the table and close to the
lidar (same bracket), which is what breaks the rectangle's symmetry.

Not modelled: lens distortion (the 102° lens has some barrel distortion; a checkerboard can add it later, the
`dist` field is already read by the projection code).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import yaml

FILE_DEFAULT = "config/camera_calibration.yaml"


def table_corners(table_doc: dict) -> np.ndarray:
    """The MEASURED tabletop's corners in the table frame (z = 0), cyclic: near-left, near-right, far-right,
    far-left (left/right as seen from the near end; +x is right). The table frame's rectangle may have been grown
    to the configured diorama size away from the lidar; the near edge is where it was measured."""
    W, L = table_doc["size_m"]
    Wm, Lm = table_doc["measured_size_m"]
    y0 = -L / 2
    return np.array([[-Wm / 2, y0, 0], [Wm / 2, y0, 0], [Wm / 2, y0 + Lm, 0], [-Wm / 2, y0 + Lm, 0]], float)


def intrinsics_from_fov(image_size, fov_diag_deg: float) -> np.ndarray:
    w, h = image_size
    f = (np.hypot(w, h) / 2) / np.tan(np.radians(fov_diag_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def _cyclic(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]


def _reproj(obj, img, K, dist, rvec, tvec) -> float:
    import cv2
    uv, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    return float(np.sqrt(np.mean(np.sum((uv.reshape(-1, 2) - img) ** 2, axis=1))))


def solve_from_corners(clicks_px, corners_frame, image_size, fov_diag_deg: float, sensor_pose,
                       *, max_cam_lidar_m: float = 0.6, fit_focal: bool = True) -> dict:
    """Camera pose (and focal) from 4 clicked corners. `corners_frame`: the 4 corners in the scan frame, cyclic.
    `sensor_pose`: the lidar's pose in that frame. Returns {ok, reason | K, dist, T_cam_from_frame, ...}."""
    import cv2

    img_all = _cyclic(np.asarray(clicks_px, float).reshape(4, 2))
    obj = np.asarray(corners_frame, float).reshape(4, 3)
    S = np.asarray(sensor_pose, float)
    lidar = S[:3, 3]
    K0 = intrinsics_from_fov(image_size, fov_diag_deg)
    z5 = np.zeros(5)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    cands = []
    for rev in (False, True):
        seq = img_all[::-1] if rev else img_all
        for shift in range(4):
            img = np.roll(seq, shift, axis=0)
            options = []
            ok, rvec, tvec = cv2.solvePnP(obj, img, K0, z5, flags=cv2.SOLVEPNP_IPPE)
            if ok:
                options.append(("fov", K0, rvec, tvec))
            if fit_focal:
                try:
                    _, K1, _, rv, tv = cv2.calibrateCamera([obj.astype(np.float32)], [img.astype(np.float32)],
                                                           tuple(int(v) for v in image_size), K0.copy(), z5.copy(),
                                                           flags=flags)
                    if 0.75 <= K1[0, 0] / K0[0, 0] <= 1.33:
                        options.append(("fitted", K1, rv[0], tv[0]))
                except cv2.error:
                    pass
            for f_src, K, rvec, tvec in options:
                R, _ = cv2.Rodrigues(rvec)
                cam = (-R.T @ tvec).ravel()
                err = _reproj(obj, img, K, z5, rvec, tvec)
                cands.append({"err": err, "K": K, "R": R, "t": np.asarray(tvec).ravel(), "cam": cam, "f_src": f_src,
                              "d_lidar": float(np.linalg.norm(cam - lidar)), "above": bool(cam[2] > 0.05)})
    good = [c for c in cands if c["above"] and c["d_lidar"] <= max_cam_lidar_m]
    if not good:
        near = min(cands, key=lambda c: c["d_lidar"]) if cands else None
        return {"ok": False, "reason": "no corner assignment puts the camera above the table within "
                f"{max_cam_lidar_m:g} m of the lidar" + (f" (closest: {near['d_lidar']:.2f} m)" if near else "")
                + ". Click the four corners of the TABLETOP, and keep the camera on the lidar's bracket."}
    best = min(good, key=lambda c: c["err"])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = best["R"], best["t"]
    warnings = []
    if best["err"] > 8:
        warnings.append(f"the clicks fit loosely ({best['err']:.1f} px): re-click the corners more precisely")
    return {"ok": True, "K": best["K"], "dist": z5, "T_cam_from_frame": T, "reproj_px": round(best["err"], 2),
            "focal_px": round(float(best["K"][0, 0]), 1), "focal_from": best["f_src"],
            "camera_in_frame_m": [round(float(v), 3) for v in best["cam"]],
            "camera_to_lidar_m": round(best["d_lidar"], 3), "warnings": warnings}


def save(path: str, res: dict, *, image_size, sensor_pose, scan_id: str, table_calib_id: str | None,
         fov_diag_deg: float) -> dict:
    T_cam_lidar = res["T_cam_from_frame"] @ np.asarray(sensor_pose, float)
    doc = {
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "from_scan": scan_id, "table_calib_id": table_calib_id,
        "image_size": [int(v) for v in image_size],
        "K": res["K"].tolist(), "dist": list(res["dist"]),
        "T_cam_from_lidar": T_cam_lidar.tolist(),
        "fov_diag_deg_assumed": fov_diag_deg, "focal_px": res["focal_px"], "focal_from": res["focal_from"],
        "reproj_px": res["reproj_px"], "camera_to_lidar_m": res["camera_to_lidar_m"], "warnings": res["warnings"],
    }
    doc = json.loads(json.dumps(doc, default=float))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Written by the dashboard's Calibrate camera (four clicked table corners). Re-click to redo it.\n"
                "# T_cam_from_lidar: lidar (sensor) frame -> camera; valid while both stay on one bracket.\n")
        yaml.safe_dump(doc, f, sort_keys=False)
    return doc


def load(path: str) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def camera_for_frame(doc: dict | None, sensor_pose) -> dict | None:
    """The fuse.py camera dict for scans in a frame where the lidar sits at `sensor_pose`."""
    if not doc:
        return None
    S = np.asarray(sensor_pose if sensor_pose is not None else np.eye(4), float)
    T = np.asarray(doc["T_cam_from_lidar"], float) @ np.linalg.inv(S)
    return {"K": doc["K"], "dist": doc.get("dist", [0, 0, 0, 0, 0]), "T_cam_from_sensor": T.tolist(),
            "image_size": doc["image_size"]}


MIN_PAIRS = 6
FIT_GAIN = 0.9       # a fitted focal must cut the pair error below this fraction of the listed-focal fit (DH-26 sweep)


def _fit_pairs(img, obj, image_size, fov_diag_deg, lidar, max_cam_lidar_m):
    """Best physically valid camera fit to the pairs, or (None, reason)."""
    import cv2

    K0 = intrinsics_from_fov(image_size, fov_diag_deg)
    z5 = np.zeros(5)
    cands = []
    for method in (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K0, z5, flags=method)
        except cv2.error:
            continue
        if not ok:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj, img, K0, z5, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        cands.append(("fov", K0, rvec, tvec))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_USE_EXTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT
                 | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
                 | cv2.CALIB_FIX_K3)
        try:
            _, K1, _, rv, tv = cv2.calibrateCamera([obj.astype(np.float32)], [img.astype(np.float32)],
                                                   tuple(int(v) for v in image_size), K0.copy(), z5.copy(),
                                                   rvecs=[rvec.copy()], tvecs=[tvec.copy()], flags=flags)
            if 0.75 <= K1[0, 0] / K0[0, 0] <= 1.33:
                cands.append(("fitted", K1, rv[0], tv[0]))
        except cv2.error:
            pass
    if not cands:
        return None, "the point pairs could not be solved: spread them out across the picture"
    scored = []
    for f_src, K, rvec, tvec in cands:
        R, _ = cv2.Rodrigues(rvec)
        t = np.asarray(tvec).ravel()
        cam = -R.T @ t
        uv, _ = cv2.projectPoints(obj, rvec, tvec, K, z5)
        per = np.linalg.norm(uv.reshape(-1, 2) - img, axis=1)
        scored.append({"f_src": f_src, "K": K, "R": R, "t": t, "cam": cam, "per": per,
                       "err": float(np.sqrt(np.mean(per ** 2))), "front": bool(np.all((obj @ R.T + t)[:, 2] > 0.02)),
                       "d_lidar": float(np.linalg.norm(cam - lidar))})
    good = [c for c in scored if c["front"] and c["d_lidar"] <= max_cam_lidar_m]
    if not good:
        near = min(scored, key=lambda c: c["d_lidar"])
        return None, (f"the pairs put the camera {near['d_lidar']:.2f} m from the lidar (limit {max_cam_lidar_m:g} m on "
                      "one bracket): a pair is probably mismatched. Check each image click and lidar click are the same spot.")
    # focal and distance trade off on a shallow scene: a fitted focal only wins when it clearly explains the
    # clicks better (25 % lower error) than the lens's listed value; a small gain is noise being fitted
    fov = [c for c in good if c["f_src"] == "fov"]
    fit = [c for c in good if c["f_src"] == "fitted"]
    best_fov = min(fov, key=lambda c: c["err"]) if fov else None
    best_fit = min(fit, key=lambda c: c["err"]) if fit else None
    if best_fov is None or (best_fit is not None and best_fit["err"] < FIT_GAIN * best_fov["err"]):
        return best_fit, ""
    return best_fov, ""


def solve_from_pairs(pixels, points, image_size, fov_diag_deg: float, sensor_pose,
                     *, max_cam_lidar_m: float = 0.6, max_pair_err_px: float = 25.0) -> dict:
    """Camera pose and focal from clicked point pairs: a pixel in the camera frame and the same spot picked in the
    lidar cloud (x, y, z in the scan frame). Works in any frame (no table needed), so it covers a floor-size
    calibration too (DH-26). At least MIN_PAIRS: with four, 1 cm of picking error moved the camera ~12 cm.
    One clearly mismatched pair (dropping it more than halves the error) is left out and named."""
    img = np.asarray(pixels, float).reshape(-1, 2)
    obj = np.asarray(points, float).reshape(-1, 3)
    n = min(len(img), len(obj))
    if len(img) != len(obj) or n < MIN_PAIRS:
        return {"ok": False, "reason": f"need at least {MIN_PAIRS} point pairs, got {n}"}
    lidar = np.asarray(sensor_pose, float)[:3, 3]
    best, reason = _fit_pairs(img, obj, image_size, fov_diag_deg, lidar, max_cam_lidar_m)
    used = np.arange(n)
    warnings = []
    if n > MIN_PAIRS and (best is None or best["per"].max() > max_pair_err_px):
        trials = []
        for k in range(n):
            keep = np.delete(np.arange(n), k)
            b, _ = _fit_pairs(img[keep], obj[keep], image_size, fov_diag_deg, lidar, max_cam_lidar_m)
            if b is not None:
                trials.append((b["err"], k, b, keep))
        if trials:
            err, k, b, keep = min(trials, key=lambda x: x[0])
            if best is None or err < 0.5 * best["err"]:
                best, used = b, keep
                warnings.append(f"pair {k + 1} did not match the others and was left out (redo it if you like)")
    if best is None:
        return {"ok": False, "reason": reason}
    worst = int(np.argmax(best["per"]))
    if best["per"][worst] > max_pair_err_px:
        warnings.append(f"pair {int(used[worst]) + 1} is {best['per'][worst]:.0f} px off: redo it for a better fit")
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = best["R"], best["t"]
    per = np.full(n, np.nan)
    per[used] = best["per"]
    return {"ok": True, "K": best["K"], "dist": np.zeros(5), "T_cam_from_frame": T, "reproj_px": round(best["err"], 2),
            "focal_px": round(float(best["K"][0, 0]), 1), "focal_from": best["f_src"],
            "camera_in_frame_m": [round(float(v), 3) for v in best["cam"]],
            "camera_to_lidar_m": round(best["d_lidar"], 3),
            "pair_err_px": [None if np.isnan(v) else round(float(v), 1) for v in per], "warnings": warnings}

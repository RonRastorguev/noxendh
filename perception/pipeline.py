"""run_scan_comparison(baseline, current) -> ChangeEvents (ARCHITECTURE §4, §6): the whole Demo A comparison.

align (§6.3) -> geometric diff (§6.4) -> move association (§6.5) -> visual diff (§6.6) -> events + crops (§6.7)
-> classification (§6.8, optional here: the server passes its classifier chain).
A badly aligned scan is never diffed: the result carries `error` and no events (§6.3). `progress(stage, detail)` is
called at each stage so the UI never shows a frozen screen (§7.1).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from perception import associate as assoc_mod
from perception import fuse
from perception.geo_diff import geo_diff
from perception.io import Scan
from perception.preprocess import crop_roi, for_registration, preprocess, roi_bounds, roi_mask
from perception.register import Registration, apply, register
from perception.visual_diff import bbox_mask, visual_diff


@dataclass
class Comparison:
    events: list = field(default_factory=list)
    registration: dict | None = None
    stats: dict = field(default_factory=dict)
    error: str | None = None
    display: dict | None = None          # points for the 3D view: xyz, intensity, label (event index or -1), ghost


def display_cloud(xyz: np.ndarray, intensity: np.ndarray, cfg: dict, max_points: int = 80_000, seed: int = 0) -> dict:
    """ROI-cropped cloud for the dashboard's 3D view (unlabelled: a baseline, or a scan without events)."""
    xyz, intensity = crop_roi(xyz, intensity, cfg.get("roi"))
    if len(xyz) > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(len(xyz), max_points, replace=False))
        xyz, intensity = xyz[idx], intensity[idx]
    return {"xyz": xyz.astype(np.float32), "intensity": intensity.astype(np.float32),
            "label": np.full(len(xyz), -1, np.int16), "ghost": np.zeros(len(xyz), np.uint8)}


def _origin(scan: Scan) -> np.ndarray:
    return scan.sensor_pose[:3, 3]


def run_scan_comparison(baseline: Scan, current: Scan, cfg: dict, *, crops_dir: str, camera: dict | None = None,
                        classifier=None, progress=None) -> Comparison:
    say = progress or (lambda stage, detail=None: None)
    t0 = time.time()
    out = Comparison()

    say("aligning")
    b_reg = for_registration(baseline.xyz, baseline.intensity, cfg, seed=1)
    c_reg = for_registration(current.xyz, current.intensity, cfg, seed=2)
    reg = register(c_reg[0], b_reg[0], cfg)
    if not reg.ok and (cfg.get("registration") or {}).get("fixed_rig_known_pose"):
        # A fixed rig already knows its pose. When ICP cannot confirm the alignment (a flat, featureless crop lets it
        # slide: Ron's floor crop "found" 78 mm / 3° between two scans of an untouched lidar) and the lidar is
        # verifiably unmoved (same IMU tilt, the scene lines up), keep the known pose instead of refusing (DH-32).
        # The registration record says so; a moved lidar still fails here.
        from perception.table_frame import lidar_unmoved
        same_frame = baseline.meta.get("table_calib_id") == current.meta.get("table_calib_id")
        still, why = lidar_unmoved(baseline, current) if same_frame else (False, "different calibrations")
        if still:
            reg = Registration(T=np.eye(4), fitness=reg.fitness, rmse=reg.rmse, iterations=reg.iterations, ok=True,
                               reason=f"fixed rig: kept the known pose; ICP was not confirmed ({reg.reason}); lidar "
                                      f"unmoved ({why})", structure_fitness=reg.structure_fitness)
    out.registration = reg.as_dict()
    if not reg.ok:
        out.error = f"scan not aligned with the baseline ({reg.reason}); re-scan or re-baseline"
        out.stats["seconds"] = round(time.time() - t0, 2)
        return out

    say("comparing")
    b_xyz, b_int = preprocess(baseline.xyz, baseline.intensity, cfg)
    c_xyz, c_int = preprocess(current.xyz, current.intensity, cfg)
    c_xyz = apply(reg.T, c_xyz)
    cur_origin = (reg.T[:3, :3] @ _origin(current)) + reg.T[:3, 3]
    diff = geo_diff(b_xyz, b_int, c_xyz, c_int, cfg, base_origin=_origin(baseline), cur_origin=cur_origin)
    association = assoc_mod.associate(diff.removed, diff.added, cfg)

    visual = None
    if baseline.image is not None and current.image is not None:
        say("comparing colours")
        boxes = []
        for c in diff.added + diff.removed:                       # already explained by geometry (§6.6)
            r = fuse.project_box(camera, c.bbox_min, c.bbox_max)
            if r:
                boxes.append((r[0], r[1], r[2] - r[0], r[3] - r[1]))
        mask = bbox_mask(baseline.image.shape, boxes)
        bounds = roi_bounds(cfg.get("roi"))
        if camera and bounds is not None:                         # only the projected ROI counts (§6.6)
            r = fuse.project_box(camera, bounds[0], bounds[1])
            if r:
                roi_px = np.zeros_like(mask)
                roi_px[r[1]:r[3], r[0]:r[2]] = 255
                mask = np.minimum(mask, roi_px)
        visual = visual_diff(baseline.image, current.image, cfg, mask=mask)

    events = fuse.build_events(scan_id=current.scan_id, baseline_id=baseline.scan_id, association=association,
                               visual=visual, before_img=baseline.image, after_img=current.image, cam=camera,
                               crops_dir=crops_dir, cloud_for_visual=crop_roi(b_xyz, b_int, cfg.get("roi"))[0])
    # 3D view: tonight's (aligned, voxelised) cloud with each changed point labelled by its event index, plus the
    # points that DISAPPEARED (removed / moved-from) taken from the baseline as "ghost" points. Event order = fuse's:
    # moved, added, removed, appearance.
    label = np.full(len(c_xyz), -1, np.int16)
    ghosts_xyz, ghosts_int, ghosts_lab = [], [], []
    k = 0
    for m in association.moved:
        label[m.added.idx] = k
        ghosts_xyz.append(b_xyz[m.removed.idx]); ghosts_int.append(b_int[m.removed.idx]); ghosts_lab.append(np.full(len(m.removed.idx), k, np.int16))
        k += 1
    for c in association.added:
        label[c.idx] = k
        k += 1
    for c in association.removed:
        ghosts_xyz.append(b_xyz[c.idx]); ghosts_int.append(b_int[c.idx]); ghosts_lab.append(np.full(len(c.idx), k, np.int16))
        k += 1
    keep = roi_mask(c_xyz, cfg.get("roi"))
    if keep is None:
        keep = np.ones(len(c_xyz), bool)
    gx = np.concatenate(ghosts_xyz) if ghosts_xyz else np.zeros((0, 3), np.float32)
    out.display = {"xyz": np.concatenate([c_xyz[keep], gx]).astype(np.float32),
                   "intensity": np.concatenate([c_int[keep], np.concatenate(ghosts_int) if ghosts_int else np.zeros(0)]).astype(np.float32),
                   "label": np.concatenate([label[keep], np.concatenate(ghosts_lab) if ghosts_lab else np.zeros(0, np.int16)]).astype(np.int16),
                   "ghost": np.concatenate([np.zeros(int(keep.sum()), np.uint8), np.ones(len(gx), np.uint8)])}

    if classifier is not None and events:
        classifier.classify_events(events, progress=lambda i, n: say("classifying", f"{i}/{n}"))
    out.events = events
    out.stats = {"geo": diff.stats, "n_moved": len(association.moved), "n_added": len(association.added),
                 "n_removed": len(association.removed), "n_appearance": len(visual.regions) if visual else 0,
                 "visual": visual.stats if visual else None, "seconds": round(time.time() - t0, 2)}
    say("done")
    return out

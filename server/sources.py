"""Where a new scan comes from. `live`: the real L2 + table camera (capture.scan_capture). `synthetic`: a ray-cast
scan + rendered frame of a diorama scenario, written as an ordinary scan folder, so everything downstream (pipeline,
server, dashboard) runs unchanged without hardware (DH-18)."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from perception import io as scan_io  # noqa: E402
from perception import synthetic as syn  # noqa: E402

SYNTH_ORIGIN = (0.0, 0.0, 0.7)
SYNTH_ROI = syn.ROI


def synthetic_camera() -> dict:
    return syn.camera_model(origin=SYNTH_ORIGIN)


def capture_synthetic(scans_dir: str, scenario: str, *, is_baseline: bool, seed: int, progress=None) -> str:
    say = progress or (lambda stage, frac=None: None)
    say("accumulating", 0.0)
    scene, rust = syn.scenario(scenario)
    xyz, inten = syn.scan(scene, origin=SYNTH_ORIGIN, seed=seed)
    img = syn.render_image(scene, synthetic_camera(), rust_on=rust, seed=seed,
                           gain=1.0 if is_baseline else float(np.random.default_rng(seed).uniform(0.9, 1.05)))
    sid = scan_io.new_scan_id()
    pose = np.eye(4)
    pose[:3, 3] = SYNTH_ORIGIN
    meta = scan_io.make_meta(sid, accumulation_s=4.0, is_baseline=is_baseline, frame="sensor", sensor_pose=pose.tolist(),
                             roi=SYNTH_ROI, source=f"synthetic:{scenario}", camera="synthetic")
    os.makedirs(scans_dir, exist_ok=True)
    say("written", 1.0)
    return scan_io.save_scan(scans_dir, scan_io.Scan(sid, xyz, inten, img, meta))


def capture_live(cfg: dict, scans_dir: str, *, is_baseline: bool, progress=None) -> str:
    from capture.scan_capture import capture_scan
    return capture_scan(cfg, is_baseline=is_baseline, use_camera=bool(cfg.get("camera", {}).get("enabled", True)),
                        progress=progress, scans_dir=scans_dir)


def replay_scan(src_path: str, scans_dir: str, *, is_baseline: bool, progress=None) -> str:
    """An existing scan folder re-run as a new capture (new id, marked `source: replay:<id>`, `replay_of`), so the
    whole chain after capture (compare, classify, alert) can be rehearsed with the lidar off (DH-32)."""
    from datetime import datetime, timezone
    say = progress or (lambda stage, frac=None: None)
    say("accumulating", 1.0)
    src = scan_io.load_scan(src_path)
    sid = scan_io.new_scan_id()
    meta = dict(src.meta, scan_id=sid, is_baseline=is_baseline, timestamp=datetime.now(timezone.utc).isoformat(),
                source=f"replay:{src.scan_id}", replay_of=src.scan_id)
    os.makedirs(scans_dir, exist_ok=True)
    say("written", 1.0)
    return scan_io.save_scan(scans_dir, scan_io.Scan(sid, src.xyz, src.intensity, src.image, meta))

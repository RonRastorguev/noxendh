"""Capture one scan folder: accumulate L2 points, grab the camera frame at the end, write it (§6.1).

Plain Python, no ROS (DH-03). The server calls `capture_scan()` directly in a worker thread; the
CLI below is for bring-up and for recording the scenario scans (§11).

    python -m capture.scan_capture --config config/table_demo.yaml [--baseline] [--no-camera]

A capture with `--no-camera` is written with `has_image: false` and `camera: "not_run"` in its
meta: it never gets a placeholder image.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from drivers.unitree_l2.receiver import L2Receiver  # noqa: E402
from perception import io as scan_io  # noqa: E402
from perception import table_frame  # noqa: E402

from capture.camera import CameraGrabber  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str) -> str:
    """Config paths are relative to the repo root."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def capture_scan(cfg: dict, *, is_baseline: bool = False, use_camera: bool = True,
                 receiver: L2Receiver | None = None, camera: CameraGrabber | None = None,
                 progress=None, scans_dir: str | None = None, source: str = "live") -> str:
    """Capture and write one scan; returns the scan folder path.

    `receiver` / `camera` default to the real devices from `cfg`; tests pass a receiver bound to
    loopback and fed from a recorded pcap. `progress(stage, fraction)` is optional."""
    lc, cc = cfg["lidar"], cfg.get("camera", {})
    receiver = receiver or L2Receiver(lc["local_ip"], lc["local_port"], lc["range_min_m"], lc["range_max_m"])
    say = progress or (lambda stage, frac: None)
    started = datetime.now(timezone.utc)

    if use_camera and camera is None:
        camera = CameraGrabber(cc.get("device", 0), cc.get("width"), cc.get("height"), cc.get("warmup_frames", 15))
    image = None
    try:
        if use_camera:
            say("camera_open", 0.0)
            camera.start()                      # warms up while the lidar accumulates
        say("accumulating", 0.0)
        acc = receiver.accumulate(float(cfg["accumulation_s"]), float(lc.get("first_data_timeout_s", 3.0)),
                                  progress=lambda f: say("accumulating", f))
        if use_camera:
            image = camera.latest()             # the frame at the END of accumulation (§6.1)
    finally:
        if use_camera and camera is not None:
            camera.stop()

    scan_id = scan_io.new_scan_id(started)
    xyz = acc.xyz
    if cfg.get("frame") == "table" and cfg.get("sensor_pose") is not None:
        xyz = table_frame.transform(xyz, cfg["sensor_pose"])   # stored in the calibrated table frame (DH-23)
    meta = scan_io.make_meta(
        scan_id, accumulation_s=cfg["accumulation_s"], is_baseline=is_baseline,
        frame=cfg.get("frame", "sensor"), sensor_pose=cfg.get("sensor_pose"), roi=cfg.get("roi"),
        timestamp=started.isoformat(), source=source, camera="ok" if image is not None else "not_run",
        lidar=acc.stats(), table_calib_id=cfg.get("table_calib_id"))
    scan = scan_io.Scan(scan_id=scan_id, xyz=xyz, intensity=acc.intensity, image=image, meta=meta)
    out_dir = scans_dir or resolve(cfg.get("scans_dir", "data/scans"))
    os.makedirs(out_dir, exist_ok=True)
    path = scan_io.save_scan(out_dir, scan)
    say("written", 1.0)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "table_demo.yaml"))
    ap.add_argument("--baseline", action="store_true", help="mark this scan as a baseline")
    ap.add_argument("--no-camera", action="store_true", help="lidar only (meta says camera: not_run)")
    ap.add_argument("--scans-dir", help="override config scans_dir")
    a = ap.parse_args(argv)
    cfg = table_frame.apply(load_config(a.config), ROOT)
    last = {"stage": None}

    def progress(stage, frac):
        if stage != last["stage"]:
            print(f"[capture] {stage}", file=sys.stderr)
            last["stage"] = stage

    path = capture_scan(cfg, is_baseline=a.baseline, use_camera=not a.no_camera, progress=progress,
                        scans_dir=a.scans_dir)
    with open(os.path.join(path, scan_io.META_FILE), encoding="utf-8") as f:
        meta = json.load(f)
    print(json.dumps({"scan_dir": path, "n_points": meta["n_points"], "camera": meta["camera"],
                      "lidar": meta["lidar"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

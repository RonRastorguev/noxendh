"""Report what is in a scan folder (Milestone 0 gate: "a scan folder with a sane cloud + image").

    python tools/check_scan.py data/scans/<scan_id>

Prints JSON and exits 0 = structurally sound, 1 = a structural defect (missing file, empty or
non-finite cloud, unreadable or missing image), 2 = could not evaluate. The numbers (point count,
extent, ranges) are REPORTED for a human to judge "sane"; none of them is a threshold.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from perception import io as scan_io  # noqa: E402


def check(path: str) -> tuple[int, dict]:
    try:
        scan = scan_io.load_scan(path)
    except (OSError, ValueError, KeyError) as e:
        return 2, {"scan_dir": path, "error": f"{type(e).__name__}: {e}"}
    xyz, inten = scan.xyz, scan.intensity
    defects = []
    if len(xyz) == 0:
        defects.append("cloud is empty")
    if not np.all(np.isfinite(xyz)):
        defects.append("cloud has non-finite points")
    if scan.image is None:
        defects.append(f"no image (meta camera: {scan.meta.get('camera')})")
    elif scan.image.size == 0:
        defects.append("image is empty")
    rng = np.linalg.norm(xyz, axis=1) if len(xyz) else np.zeros(0)
    q = (lambda v: [round(float(x), 3) for x in np.percentile(v, [0, 1, 50, 99, 100])]) if len(xyz) else (lambda v: None)
    report = {
        "scan_dir": path,
        "scan_id": scan.scan_id,
        "is_baseline": scan.meta.get("is_baseline"),
        "n_points": int(len(xyz)),
        "extent_min_m": [round(float(v), 3) for v in xyz.min(axis=0)] if len(xyz) else None,
        "extent_max_m": [round(float(v), 3) for v in xyz.max(axis=0)] if len(xyz) else None,
        "range_m_p0_p1_p50_p99_p100": q(rng),
        "intensity_p0_p1_p50_p99_p100": q(inten),
        "image_shape": list(scan.image.shape) if scan.image is not None else None,
        "image_mean_bgr": [round(float(v), 1) for v in scan.image.reshape(-1, 3).mean(axis=0)] if scan.image is not None else None,
        "lidar": scan.meta.get("lidar"),
        "defects": defects,
        "gating": False,
    }
    return (1 if defects else 0), report


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    rc, report = check(args[0])
    print(json.dumps(report, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())

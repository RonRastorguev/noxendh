"""Find the diorama's table from a lidar scan and write config/table_frame.yaml (DH-23).

    py -3.9 tools/calibrate_table.py                 # capture 10 s from the L2 (lidar only), then fit
    py -3.9 tools/calibrate_table.py --scan DIR      # fit an existing SENSOR-frame scan folder

Every capture after this is stored in the table frame and cropped to the diorama rectangle; take a new baseline.
Exit 0 = table found and written, 1 = no table found (reason printed), 2 = could not run (no lidar data, bad scan).
The dashboard's "Calibrate table" button does the same thing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from capture.scan_capture import capture_scan, load_config  # noqa: E402
from drivers.unitree_l2.receiver import NoLidarData  # noqa: E402
from perception import io as scan_io  # noqa: E402
from perception import table_frame  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "config", "table_demo.yaml"))
    ap.add_argument("--scan", help="fit this scan folder instead of capturing")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    path = a.scan
    if not path:
        raw = dict(cfg, frame="sensor", sensor_pose=None, table_calib_id=None)
        try:
            path = capture_scan(raw, use_camera=False, scans_dir=os.path.join(ROOT, "data", "calibration"),
                                progress=lambda stage, frac=None: None)
        except NoLidarData as e:
            print(json.dumps({"ok": False, "reason": str(e)}, indent=1))
            return 2
    try:
        scan = scan_io.load_scan(path, load_image=False)
    except Exception as e:
        print(json.dumps({"ok": False, "reason": f"cannot read scan {path}: {e}"}, indent=1))
        return 2
    fit, doc = table_frame.calibrate_scan(scan, cfg, ROOT)
    out = {"ok": fit.ok, "scan": path}
    if fit.ok:
        out.update(written=cfg.get("table_frame_file", table_frame.FILE_DEFAULT), size_m=doc["size_m"],
                   measured_size_m=doc["measured_size_m"], roi=doc["roi"], quality=doc["quality"],
                   warnings=doc["warnings"])
    else:
        out["reason"] = fit.reason
    print(json.dumps(out, indent=1))
    return 0 if fit.ok else 1


if __name__ == "__main__":
    sys.exit(main())

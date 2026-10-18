"""Capture path on REAL L2 data: the bench pcaps replayed over UDP loopback into capture_scan().

Proves the lidar half of the Milestone 0 gate offline (the camera half needs the table camera).
Runs under pytest or plain `python tests/test_capture_replay.py`.
"""
from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "drivers", "unitree_l2", "bench"))
import l2_pcap  # noqa: E402

from capture.scan_capture import capture_scan  # noqa: E402
from drivers.unitree_l2.receiver import L2Receiver, NoLidarData, parse_datagrams  # noqa: E402
from perception import io as scan_io  # noqa: E402
from tools.check_scan import check  # noqa: E402

FIX = os.path.join(ROOT, "drivers", "unitree_l2", "test_fixtures")
WIDE = os.path.join(FIX, "l2_real_wide_5s.pcap")
COALESCED = os.path.join(FIX, "l2_real_imu_coalesced_3s.pcap")
CFG = {"accumulation_s": 2.0, "frame": "sensor", "sensor_pose": None, "roi": None,
       "lidar": {"local_ip": "127.0.0.1", "local_port": 0, "range_min_m": 0.05, "range_max_m": 30.0,
                 "first_data_timeout_s": 3.0}}


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _replay_soon(pcap, port, delay_s=0.3):
    t = threading.Thread(target=lambda: (time.sleep(delay_s), l2_pcap.replay(pcap, f"127.0.0.1:{port}")), daemon=True)
    t.start()
    return t


def test_parse_coalesced_fixture_counts_match_provenance():
    # test_fixtures/README.md: 753 IMU + 651 point frames, 34 datagrams carry both
    xyz, inten, n_pt, n_imu, drops = parse_datagrams([p for _, p in l2_pcap.read_pcap(COALESCED)], 0.05, 30.0)
    assert (n_pt, n_imu) == (651, 753), (n_pt, n_imu, drops)
    assert len(xyz) > 0 and np.all(np.isfinite(xyz)) and inten.min() >= 0 and inten.max() <= 1


def test_capture_scan_from_replayed_real_l2():
    port = _free_port()
    rx = L2Receiver("127.0.0.1", port, 0.05, 30.0)
    _replay_soon(WIDE, port)
    with tempfile.TemporaryDirectory() as d:
        path = capture_scan(CFG, use_camera=False, receiver=rx, scans_dir=d, source="pcap:l2_real_wide_5s")
        rc, rep = check(path)
        scan = scan_io.load_scan(path)
    lid = scan.meta["lidar"]
    print(f"    replay: {lid['n_point_packets']} point packets, {scan.meta['n_points']} points in "
          f"{lid['duration_s']} s, range p50 {rep['range_m_p0_p1_p50_p99_p100'][2]} m")
    # ~216 point packets/s for 2 s; the window starts at the first datagram, so expect close to 2 s of packets
    assert 350 <= lid["n_point_packets"] <= 450, lid
    assert 1.8 <= lid["duration_s"] <= 2.05, lid
    assert scan.meta["n_points"] > 10000 and scan.meta["camera"] == "not_run" and scan.meta["has_image"] is False
    assert sum(lid["drops"].values()) == 0, lid["drops"]
    # the only structural defect check_scan may report on a lidar-only capture is the missing image
    assert rc == 1 and len(rep["defects"]) == 1 and rep["defects"][0].startswith("no image"), rep["defects"]


def test_no_data_fails_loudly_and_writes_nothing():
    port = _free_port()
    cfg = dict(CFG, lidar=dict(CFG["lidar"], first_data_timeout_s=0.5))
    with tempfile.TemporaryDirectory() as d:
        try:
            capture_scan(cfg, use_camera=False, receiver=L2Receiver("127.0.0.1", port), scans_dir=d)
            raise AssertionError("expected NoLidarData")
        except NoLidarData:
            pass
        assert os.listdir(d) == []


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"  [ok] {name}")
    print(f"{len(tests)}/{len(tests)} passed")

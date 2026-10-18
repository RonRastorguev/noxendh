"""perception/io.py: PCD and scan-folder round trips. Runs under pytest or plain `python tests/test_io.py`."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from perception import io as scan_io  # noqa: E402


def _cloud(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)).astype(np.float32), rng.random(n).astype(np.float32)


def test_pcd_binary_round_trip_is_exact():
    xyz, inten = _cloud()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.pcd")
        scan_io.write_pcd(p, xyz, inten)
        xyz2, inten2 = scan_io.read_pcd(p)
    assert xyz2.dtype == np.float32 and xyz2.shape == xyz.shape
    assert np.array_equal(xyz, xyz2) and np.array_equal(inten, inten2)


def test_pcd_empty_cloud_round_trips():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.pcd")
        scan_io.write_pcd(p, np.zeros((0, 3)), np.zeros(0))
        xyz, inten = scan_io.read_pcd(p)
    assert xyz.shape == (0, 3) and inten.shape == (0,)


def test_pcd_reads_ascii_without_intensity():
    text = ("VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 2\nHEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2\nDATA ascii\n1 2 3\n4 5 6\n")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.pcd")
        with open(p, "w") as f:
            f.write(text)
        xyz, inten = scan_io.read_pcd(p)
    assert np.allclose(xyz, [[1, 2, 3], [4, 5, 6]]) and np.array_equal(inten, [0, 0])


def test_pcd_rejects_mismatched_lengths():
    try:
        scan_io.write_pcd(os.devnull, np.zeros((3, 3)), np.zeros(2))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_scan_folder_round_trip_with_image():
    xyz, inten = _cloud(500)
    img = np.zeros((48, 64, 3), np.uint8)
    img[10:20, 10:30] = (0, 128, 255)
    sid = scan_io.new_scan_id()
    meta = scan_io.make_meta(sid, accumulation_s=4.0, is_baseline=True, roi={"min": [0, 0, 0], "max": [1, 1, 1]},
                             source="test")
    with tempfile.TemporaryDirectory() as d:
        path = scan_io.save_scan(d, scan_io.Scan(sid, xyz, inten, img, meta))
        assert sorted(os.listdir(path)) == ["cloud.pcd", "image.png", "meta.json"]
        s = scan_io.load_scan(path)
        assert scan_io.list_scans(d) == [sid]
    assert s.scan_id == sid and np.array_equal(s.xyz, xyz) and np.array_equal(s.image, img)
    assert s.meta["is_baseline"] is True and s.meta["n_points"] == 500 and s.meta["has_image"] is True
    assert s.meta["source"] == "test" and s.meta["frame"] == "sensor"
    assert np.array_equal(s.sensor_pose, np.eye(4))


def test_scan_without_image_says_so():
    xyz, inten = _cloud(10)
    sid = scan_io.new_scan_id()
    with tempfile.TemporaryDirectory() as d:
        path = scan_io.save_scan(d, scan_io.Scan(sid, xyz, inten, None,
                                                 scan_io.make_meta(sid, accumulation_s=1, is_baseline=False)))
        assert not os.path.exists(os.path.join(path, "image.png"))
        assert scan_io.load_scan(path).meta["has_image"] is False


def test_save_refuses_overwrite_and_leaves_no_tmp_on_failure():
    xyz, inten = _cloud(10)
    sid = scan_io.new_scan_id()
    meta = scan_io.make_meta(sid, accumulation_s=1, is_baseline=False)
    with tempfile.TemporaryDirectory() as d:
        scan_io.save_scan(d, scan_io.Scan(sid, xyz, inten, None, meta))
        try:
            scan_io.save_scan(d, scan_io.Scan(sid, xyz, inten, None, meta))
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass
        bad = scan_io.Scan(scan_io.new_scan_id() + "x", xyz, inten[:5], None, meta)   # length mismatch -> write fails
        try:
            scan_io.save_scan(d, bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
        assert [n for n in os.listdir(d) if n.startswith(".tmp_")] == []
        assert scan_io.list_scans(d) == [sid]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"  [ok] {name}")
    print(f"{len(tests)}/{len(tests)} passed")

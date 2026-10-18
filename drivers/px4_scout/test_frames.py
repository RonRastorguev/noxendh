"""PX4 frame conversions (drivers/px4_scout/frames.py). Run: py -3.9 drivers/px4_scout/test_frames.py"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from drivers.px4_scout import frames as F  # noqa: E402


def q_yaw_ned(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def test_position_axes():
    T = F.ned_to_odom_pose([1.0, 2.0, -3.0], [1, 0, 0, 0])   # 1 m north, 2 m east, 3 m up
    assert np.allclose(T[:3, 3], [2.0, 1.0, 3.0])


def test_heading_north_is_odom_plus_y():
    T = F.ned_to_odom_pose([0, 0, 0], q_yaw_ned(0.0))          # nose north
    assert np.allclose(T[:3, 0], [0, 1, 0], atol=1e-12)        # body forward = +y (north) in ENU
    assert np.allclose(T[:3, 2], [0, 0, 1], atol=1e-12)        # body up = up
    T = F.ned_to_odom_pose([0, 0, 0], q_yaw_ned(math.pi / 2))  # nose east
    assert np.allclose(T[:3, 0], [1, 0, 0], atol=1e-12)


def test_yaw_round_trip_and_consistency():
    for y in np.linspace(-3.1, 3.1, 25):
        T = F.ned_to_odom_pose([0, 0, 0], q_yaw_ned(y))
        yaw_odom = math.atan2(T[1, 0], T[0, 0])
        assert abs(math.remainder(F.yaw_ned_to_odom(y) - yaw_odom, 2 * math.pi)) < 1e-9
        assert abs(math.remainder(F.yaw_odom_to_ned(F.yaw_ned_to_odom(y)) - y, 2 * math.pi)) < 1e-9


def test_rotation_is_proper():
    q = [0.9, 0.1, -0.2, 0.3]
    T = F.ned_to_odom_pose([0, 0, 0], q)
    assert abs(np.linalg.det(T[:3, :3]) - 1.0) < 1e-9


def test_setpoint_fields():
    f = F.trajectory_setpoint_fields([2.0, 1.0, 1.5], 0.0)     # odom: 2 m x(east), 1 m y(north), 1.5 m up, facing east
    assert np.allclose(f["position"], [1.0, 2.0, -1.5])
    assert abs(f["yaw"] - math.pi / 2) < 1e-12
    assert all(math.isnan(v) for v in f["velocity"])


def test_commands_and_status():
    assert F.command_fields("arm")["command"] == 400 and F.command_fields("arm")["param1"] == 1.0
    assert F.command_fields("offboard")["param2"] == 6.0
    assert F.command_fields("land")["command"] == 21
    s = F.status_from_fields(1.0, 2, 14, False, False)
    assert s["armed"] and s["offboard"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    print("ALL PASS" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)

"""Px4ScoutVehicle (drivers/px4_scout/px4_scout_vehicle.py) on a fake node + fake px4_msgs, no ROS needed.

Run: py -3.9 drivers/px4_scout/test_px4_scout_vehicle.py
The fakes carry only the fields and constants the adapter touches (values from px4_msgs, PX4 v1.17). The real
messages were exercised in PX4 SITL (bench/run_sitl_check.sh, docs/SCOUT_NAV.md §10).
"""
from __future__ import annotations

import math
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _install_fakes():
    class Msg:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class VehicleOdometry(Msg):
        POSE_FRAME_NED, POSE_FRAME_FRD = 1, 2
        VELOCITY_FRAME_NED, VELOCITY_FRAME_FRD = 1, 2

    class DistanceSensor(Msg):
        ROTATION_DOWNWARD_FACING, ROTATION_FORWARD_FACING = 25, 0

    names = ["OffboardControlMode", "TrajectorySetpoint", "VehicleCommand", "VehicleLandDetected", "VehicleStatus"]
    msg = types.ModuleType("px4_msgs.msg")
    for n in names:
        setattr(msg, n, type(n, (Msg,), {}))
    msg.VehicleOdometry, msg.DistanceSensor = VehicleOdometry, DistanceSensor
    pkg = types.ModuleType("px4_msgs")
    pkg.msg = msg
    sys.modules.setdefault("px4_msgs", pkg)
    sys.modules.setdefault("px4_msgs.msg", msg)
    if "rclpy.qos" not in sys.modules:
        qos = types.ModuleType("rclpy.qos")
        qos.QoSProfile = lambda **kw: kw
        qos.HistoryPolicy = types.SimpleNamespace(KEEP_LAST=1)
        qos.ReliabilityPolicy = types.SimpleNamespace(BEST_EFFORT=2)
        rclpy = types.ModuleType("rclpy")
        rclpy.qos = qos
        sys.modules.setdefault("rclpy", rclpy)
        sys.modules["rclpy.qos"] = qos
    return msg


M = _install_fakes()
from drivers.px4_scout import frames as F  # noqa: E402
from drivers.px4_scout.px4_scout_vehicle import Px4ScoutVehicle  # noqa: E402
from navigation.types import Setpoint  # noqa: E402

# what PX4 v1.17 SITL actually advertised (result.json "topics" of the SITL check)
SITL_TOPICS = ["/fmu/out/vehicle_odometry", "/fmu/out/distance_sensor", "/fmu/out/vehicle_status_v1",
               "/fmu/out/vehicle_land_detected", "/fmu/in/offboard_control_mode", "/fmu/in/trajectory_setpoint",
               "/fmu/in/vehicle_command"]


class FakeNode:
    def __init__(self, topics):
        self.topics = list(topics)
        self.subs = {}                           # topic -> callback
        self.pubs = {}                           # topic -> [messages]
        self.log = []
        self.t_ns = 1_000_000_000

    def get_topic_names_and_types(self):
        return [(t, ["px4_msgs/msg/X"]) for t in self.topics]

    def create_subscription(self, _type, topic, cb, _qos):
        self.subs[topic] = cb

    def create_publisher(self, _type, topic, _qos):
        out = self.pubs.setdefault(topic, [])
        return types.SimpleNamespace(publish=out.append)

    def get_clock(self):
        return types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=self.t_ns))

    def get_logger(self):
        return types.SimpleNamespace(info=self.log.append, warning=self.log.append, error=self.log.append)

    def advance(self, s):
        self.t_ns += int(s * 1e9)


def make(topics=SITL_TOPICS, **kw):
    import drivers.px4_scout.px4_scout_vehicle as mod
    mod.time.sleep = lambda s: None              # find_topic's retry wait (module-level `time`)
    node = FakeNode(topics)
    return node, Px4ScoutVehicle(node, {"height_ref": "range"}, **kw)


def status_msg(arming=1, nav=4, failsafe=False):
    return M.VehicleStatus(arming_state=arming, nav_state=nav, failsafe=failsafe)


def odom_msg(pos=(0.0, 0.0, -1.0), yaw_ned=0.0, vel=(0.0, 0.0, 0.0), frame=1, reset=0):
    q = [math.cos(yaw_ned / 2), 0.0, 0.0, math.sin(yaw_ned / 2)]
    return M.VehicleOdometry(pose_frame=frame, velocity_frame=frame, position=list(pos), q=q, velocity=list(vel),
                             reset_counter=reset)


def range_msg(d, orientation=25, quality=-1, dmin=0.02, dmax=6.0):
    return M.DistanceSensor(current_distance=d, orientation=orientation, signal_quality=quality,
                            min_distance=dmin, max_distance=dmax)


def test_versioned_topic_names_found():
    node, _ = make()
    assert "/fmu/out/vehicle_status_v1" in node.subs          # PX4 1.17 names it _v1
    assert set(node.pubs) == {"/fmu/in/offboard_control_mode", "/fmu/in/trajectory_setpoint",
                              "/fmu/in/vehicle_command"}


def test_no_failsafe_before_first_status_within_grace():
    """The SITL bug: status() before PX4's first vehicle_status said failsafe -> the mission ABORTED on tick 1."""
    node, v = make()
    s = v.status()
    assert not s.failsafe and not s.armed and s.landed and not s.offboard
    assert s.info["status_received"] is False and "stale_status" not in s.info
    node.advance(0.9)
    assert not v.status().failsafe


def test_failsafe_if_status_never_arrives():
    node, v = make()
    node.advance(1.1)
    s = v.status()
    assert s.failsafe and s.info["stale_status"]


def test_grace_starts_after_construction_waits():
    """Construction can wait for topics (8 s in SITL); the grace must start when it ends."""
    node = FakeNode([t for t in SITL_TOPICS if "vehicle_odometry" not in t])
    import drivers.px4_scout.px4_scout_vehicle as mod
    calls = {"n": 0}

    def slow_sleep(_s):                          # the odometry topic appears after 8 "seconds" of waiting
        calls["n"] += 1
        node.advance(1.0)
        if calls["n"] == 8:
            node.topics.append("/fmu/out/vehicle_odometry")
    mod.time.sleep = slow_sleep
    v = Px4ScoutVehicle(node, {})
    assert not v.status().failsafe


def test_status_flags_and_stale_after_messages():
    node, v = make()
    node.subs["/fmu/out/vehicle_status_v1"](status_msg(arming=F.ARMING_STATE_ARMED, nav=F.NAVIGATION_STATE_OFFBOARD))
    node.subs["/fmu/out/vehicle_land_detected"](M.VehicleLandDetected(landed=False))
    s = v.status()
    assert s.armed and s.offboard and not s.landed and not s.failsafe and s.info["status_received"]
    node.advance(1.2)
    assert v.status().failsafe                   # silence after messages: still a failsafe
    node.subs["/fmu/out/vehicle_status_v1"](status_msg(nav=F.NAVIGATION_STATE_AUTO_LAND, failsafe=True))
    s = v.status()
    assert s.failsafe and not s.offboard and s.info["nav_state"] == 18


def test_odometry_ned_to_odom():
    node, v = make()
    node.subs["/fmu/out/vehicle_odometry"](odom_msg(pos=(2.0, 1.0, -1.5), yaw_ned=math.pi / 2, vel=(0.0, 0.5, -0.1)))
    o = v.odometry()
    assert np.allclose(o.T_odom_body[:3, 3], [1.0, 2.0, 1.5])  # x = east, y = north, z = up
    assert abs(math.atan2(o.T_odom_body[1, 0], o.T_odom_body[0, 0])) < 1e-9   # nose east = odom yaw 0
    assert np.allclose(o.v_odom, [0.5, 0.0, 0.1])
    assert abs(o.stamp - (1.0 - 0.03)) < 1e-9    # DN-05: receive time minus odom_latency_s


def test_odometry_refuses_non_ned():
    node, v = make()
    node.subs["/fmu/out/vehicle_odometry"](odom_msg(frame=2))
    assert v.odometry() is None and node.log


def test_range_to_dist_bottom():
    node, v = make(range_below_body_m=0.05)
    rng, odo = node.subs["/fmu/out/distance_sensor"], node.subs["/fmu/out/vehicle_odometry"]
    rng(range_msg(1.0))
    odo(odom_msg())
    assert abs(v.odometry().dist_bottom_m - 1.05) < 1e-9
    rng(range_msg(1.0, orientation=0))           # a forward sensor is ignored (the last downward value stays)
    odo(odom_msg())
    assert abs(v.odometry().dist_bottom_m - 1.05) < 1e-9
    rng(range_msg(0.01))                         # below min (SIH reads ~0 on the ground) -> None
    odo(odom_msg())
    assert v.odometry().dist_bottom_m is None
    rng(range_msg(655.35))                       # PX4's "out of range" (UINT16_MAX / 100) -> None
    odo(odom_msg())
    assert v.odometry().dist_bottom_m is None
    rng(range_msg(1.0))
    node.advance(0.3)                            # stale range (> 0.2 s) -> None
    odo(odom_msg())
    assert v.odometry().dist_bottom_m is None


def test_range_tilt_correction():
    node, v = make()
    node.subs["/fmu/out/distance_sensor"](range_msg(1.0))
    a = math.radians(20.0)                       # 20 deg roll
    q = [math.cos(a / 2), math.sin(a / 2), 0.0, 0.0]
    node.subs["/fmu/out/vehicle_odometry"](M.VehicleOdometry(pose_frame=1, velocity_frame=1, position=[0, 0, -1],
                                                             q=q, velocity=[0, 0, 0], reset_counter=0))
    assert abs(v.odometry().dist_bottom_m - math.cos(a)) < 1e-9


def test_no_distance_sensor_topic_fallback():
    node, v = make(topics=[t for t in SITL_TOPICS if "distance_sensor" not in t])
    assert "/fmu/out/distance_sensor" not in node.subs and any("distance_sensor" in str(m) for m in node.log)
    node.subs["/fmu/out/vehicle_odometry"](odom_msg())
    assert v.odometry().dist_bottom_m is None


def test_setpoint_and_commands_published_in_ned():
    node, v = make()
    v.send_setpoint(Setpoint(0.0, np.array([1.0, 2.0, 1.0]), 0.0, np.array([0.5, 0.0, 0.0])))
    ocm = node.pubs["/fmu/in/offboard_control_mode"][-1]
    tsp = node.pubs["/fmu/in/trajectory_setpoint"][-1]
    assert ocm.position and not ocm.velocity and ocm.timestamp == 1_000_000
    assert np.allclose(tsp.position, [2.0, 1.0, -1.0]) and abs(tsp.yaw - math.pi / 2) < 1e-12
    assert np.allclose(tsp.velocity, [0.0, 0.5, 0.0])
    v.request_offboard()
    v.request_arm()
    v.request_land()
    cmds = node.pubs["/fmu/in/vehicle_command"]
    assert [c.command for c in cmds] == [176, 400, 21] and all(c.from_external for c in cmds)


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

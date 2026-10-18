"""`VehicleInterface` (navigation/vehicle.py) over PX4 uXRCE-DDS, for the real Scout.
Run end to end against PX4 v1.17 SITL (SIH) on 2026-10-17 (`bench/run_sitl_check.sh`, docs/SCOUT_NAV.md §10);
NOT yet run against the real FC (no Radxa <-> FC link: R9).

Wiring (DH-05 / DH-07): the FC's TELEM2 -> the Radxa's uXRCE-DDS agent -> DDS over Wi-Fi -> this node on the laptop.
All PX4 frame math is in `frames.py` (unit-tested); this file only moves message fields.

Clock (DN-05): every stamp handed to the flight software is the LAPTOP's receive time minus a fixed configured latency
(`odom_latency_s`), and the camera node does the same for frames, so odometry and images share one clock without
trusting the FC's or the Radxa's clocks. The error is the link jitter (tens of ms over Wi-Fi ≈ 1–2 cm at 0.5 m/s);
measure it before trusting anything faster.

Topics are found by base name with an optional `_vN` suffix (PX4 1.17 SITL publishes /fmu/out/vehicle_status_v1;
never hard-code).
"""
from __future__ import annotations

import re
import time

import numpy as np

from navigation.types import OdomSample, VehicleStatus
from . import frames as F


def find_topic(node, base, tries=60):
    pat = re.compile(rf"^{re.escape(base)}(_v\d+)?$")
    for _ in range(tries):
        for name, _types in node.get_topic_names_and_types():
            if pat.match(name):
                return name
        time.sleep(1.0)
    raise RuntimeError(f"PX4 topic not found: {base} (is the uXRCE-DDS agent on the Radxa running?)")


class Px4ScoutVehicle:
    def __init__(self, node, capabilities: dict, odom_latency_s: float = 0.03, now_fn=None,
                 range_below_body_m: float = 0.0, range_max_m: float = 6.0):
        """`range_below_body_m`: how far the downward rangefinder sits below the FC origin (scout_hw.yaml
        range_flow.pos_from_fc_frd_m[2]); its reading is tilt-corrected and shifted to the FC origin, so
        OdomSample.dist_bottom_m = body-origin height above the surface under the beam (what map_align expects)."""
        from px4_msgs.msg import (DistanceSensor, OffboardControlMode, TrajectorySetpoint, VehicleCommand,
                                  VehicleLandDetected, VehicleOdometry, VehicleStatus as PxStatus)
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        self.node = node
        self._caps = dict(capabilities)
        self._lat = float(odom_latency_s)
        self._now = now_fn or (lambda: node.get_clock().now().nanoseconds * 1e-9)
        self._M = {"ocm": OffboardControlMode, "tsp": TrajectorySetpoint, "cmd": VehicleCommand}
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)
        self._odom = None
        self._range = None                       # (receive time, raw distance m)
        self._range_below = float(range_below_body_m)
        self._range_max = float(range_max_m)
        self._st = {"arming_state": 0, "nav_state": 0, "failsafe": False, "landed": True}
        self._st_t = None
        node.create_subscription(VehicleOdometry, find_topic(node, "/fmu/out/vehicle_odometry"), self._on_odom, be)
        try:                                     # not in every PX4 dds_topics.yaml: optional (map_align then models
            t_rng = find_topic(node, "/fmu/out/distance_sensor", tries=5)        # the height-step lag instead)
            node.create_subscription(DistanceSensor, t_rng, self._on_range, be)
        except RuntimeError:
            node.get_logger().warning("no /fmu/out/distance_sensor: dist_bottom_m stays None (lag model in map_align)")
        node.create_subscription(PxStatus, find_topic(node, "/fmu/out/vehicle_status"), self._on_status, be)
        node.create_subscription(VehicleLandDetected, find_topic(node, "/fmu/out/vehicle_land_detected"),
                                 self._on_land, be)
        self._pub_ocm = node.create_publisher(OffboardControlMode, find_topic(node, "/fmu/in/offboard_control_mode"), be)
        self._pub_tsp = node.create_publisher(TrajectorySetpoint, find_topic(node, "/fmu/in/trajectory_setpoint"), be)
        self._pub_cmd = node.create_publisher(VehicleCommand, find_topic(node, "/fmu/in/vehicle_command"), be)
        self._t_created = self._now()            # the 1 s status timeout runs from here until the first message
        #                                          (set AFTER the topic waits: construction took 8 s in SITL)

    # --- FC -> us
    def _on_odom(self, msg):
        if msg.pose_frame != msg.POSE_FRAME_NED or msg.velocity_frame != msg.VELOCITY_FRAME_NED:
            self.node.get_logger().error("vehicle_odometry is not in NED: refusing it (frames.py assumes NED)")
            return
        T = F.ned_to_odom_pose(msg.position, msg.q)
        v = F.ned_to_odom_vec(msg.velocity)
        if not np.all(np.isfinite(T[:3, 3])):
            return
        self._odom = OdomSample(self._now() - self._lat, T, np.nan_to_num(v), int(msg.reset_counter),
                                self._dist_bottom(T))

    def _on_range(self, msg):
        if int(msg.orientation) != int(msg.ROTATION_DOWNWARD_FACING):
            return
        d = float(msg.current_distance)
        ok = (msg.min_distance <= d <= min(self._range_max, msg.max_distance)) and int(msg.signal_quality) != 0
        self._range = (self._now(), d if ok else None)

    def _dist_bottom(self, T_odom_body):
        if self._range is None or self._range[1] is None or self._now() - self._range[0] > 0.2:
            return None
        cos_tilt = float(T_odom_body[2, 2])      # body z axis vs vertical
        if cos_tilt < 0.85:                      # > ~32 deg tilt: the beam is not "below" any more
            return None
        return self._range[1] * cos_tilt + self._range_below

    def _on_status(self, msg):
        self._st.update(arming_state=int(msg.arming_state), nav_state=int(msg.nav_state), failsafe=bool(msg.failsafe))
        self._st_t = self._now()

    def _on_land(self, msg):
        self._st["landed"] = bool(msg.landed)

    # --- VehicleInterface
    def odometry(self):
        return self._odom

    def status(self):
        s = F.status_from_fields(self._now(), self._st["arming_state"], self._st["nav_state"], self._st["failsafe"],
                                 self._st["landed"])
        # No status for 1 s: treat as a failsafe (the mission stops commanding). Before the FIRST message the clock
        # runs from construction: PX4 SITL took ~0.45 s to deliver one after the topic was found, and a failsafe on
        # the mission's first 20 Hz tick would ABORT it in PREFLIGHT (mission.py aborts on a failsafe in every state
        # but LAND; found by the SITL check 2026-10-17). Until a message arrives the fields are the safe defaults
        # (disarmed, landed, not offboard), so nothing can arm on them.
        s["info"]["status_received"] = self._st_t is not None
        if self._now() - (self._t_created if self._st_t is None else self._st_t) > 1.0:
            s["failsafe"] = True
            s["info"]["stale_status"] = True
        return VehicleStatus(**s)

    def _us(self):
        return int(self.node.get_clock().now().nanoseconds // 1000)

    def send_setpoint(self, sp):
        ocm = self._M["ocm"]()
        for k, v in F.offboard_mode_fields().items():
            setattr(ocm, k, v)
        ocm.timestamp = self._us()
        self._pub_ocm.publish(ocm)
        tsp = self._M["tsp"]()
        for k, v in F.trajectory_setpoint_fields(sp.position, sp.yaw, sp.velocity_ff).items():
            setattr(tsp, k, v)
        tsp.timestamp = self._us()
        self._pub_tsp.publish(tsp)

    def _command(self, kind):
        cmd = self._M["cmd"]()
        for k, v in F.command_fields(kind).items():
            setattr(cmd, k, v)
        cmd.timestamp = self._us()
        self._pub_cmd.publish(cmd)
        self.node.get_logger().info(f"vehicle command: {kind}")

    def request_arm(self):
        self._command("arm")

    def request_offboard(self):
        self._command("offboard")

    def request_land(self):
        self._command("land")

    def capabilities(self):
        return dict(self._caps)

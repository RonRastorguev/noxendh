"""ROS 2 node that flies the Scout: takeoff from the fixed pad -> a point on a wall, looming avoidance (SCOUT_NAV.md).

Run end to end against PX4 v1.17 SITL (SIH) with a synthetic camera on 2026-10-18 (drivers/px4_scout/bench/
run_sitl_check.sh mission; docs/SCOUT_NAV.md §10). NOT run with a real FC link (NEEDS_RON R9), a real camera stream or a
Point-LIO map. All logic is in the repo's ROS-free packages; the stack is built by
`navigation.build.build_flight_stack`, exactly as the closed-loop sim builds it.

Subscribes : <image_topic>  sensor_msgs/Image (or CompressedImage with compressed:=true), the Scout camera
             PX4 /fmu/out/* via drivers/px4_scout/px4_scout_vehicle.py (uXRCE-DDS agent on the Radxa)
Publishes  : PX4 /fmu/in/* (offboard setpoints, commands), scout_mission/state (std_msgs/String, JSON)

Parameters : repo_root, landmarks (.npz), map_pcd (prior map, PCD in map frame), takeoff ([x, y, z, yaw_deg],
             the fixed pad; overrides scout_mission.yaml), goal ([x, y, z], a wall point B; not needed in
             hover_test), mode (wall_goal | hover_test: climb hover_climb_m above the pad, hold hover_s, land on the
             pad; hover_climb_m / hover_s <= 0 = scout_mission.yaml's hover_test values), on_arrival
             (hold | return_land | land), auto_arm (false: the pilot arms on the RC), image_topic, compressed,
             image_latency_s / odom_latency_s (DN-05: stamps = laptop receive time minus these), apriltags (YAML,
             optional), height_ref (the FC's EKF2 height reference: range = the real Scout, EKF2_HGT_REF=2;
             baro for PX4 SITL, which cannot arm on range in SIH), keepalive_gap_s, stats_period_s,
             gil_switch_interval_s (sys.setswitchinterval, default 0.5 ms; 0 = Python's 5 ms: in SITL that let the
             frame thread starve the tick, mission.tick up to 1.9 s and 2-4 Hz, frames 400 ms instead of 150 ms).

    ros2 run nx_scout_nav scout_mission --ros-args -p repo_root:=/mnt/c/Users/Ron/Desktop/noxen-dh26 \\
        -p landmarks:=/path/landmarks.npz -p map_pcd:=/path/map.pcd -p goal:="[8.0, 3.6, 1.2]"

Threading: camera frames and the 20 Hz tick run in separate callback groups on a MultiThreadedExecutor; one lock
serialises them. The offboard setpoint stream does not depend on either: the mission talks to the vehicle through
`node_logic.StreamKeeper`, whose own watchdog thread re-sends the last setpoint when nothing was sent for
`keepalive_gap_s` (0.1 s). In PX4 SITL (WSL, loaded host) a frame held the lock 0.4-1.1 s and a planning tick ran
past 1 s, both beyond PX4's COM_OF_LOSS_T = 1 s, and PX4 went to RTL. Only the newest frame is processed (queue
depth 1); it is stamped from its DDS reception time (`node_logic.frame_stamp`), not from when its callback ran.
Every `stats_period_s` the node logs a `node_stats` event (frame processing / lock wait, reception-to-callback lag,
tick durations, setpoint gaps and keepalives).
"""
from __future__ import annotations

import json
import os
import sys
import threading

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String


class ScoutMissionNode(Node):
    def __init__(self):
        super().__init__("scout_mission")
        p = self.declare_parameter
        repo = p("repo_root", os.environ.get("NOXEN_DH26_ROOT", "")).value
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        gil = float(p("gil_switch_interval_s", 0.0005).value)  # 0 = Python's default (5 ms); see the docstring
        if gil > 0:
            sys.setswitchinterval(gil)
        import yaml
        from drivers.px4_scout.px4_scout_vehicle import Px4ScoutVehicle
        from localization import config as loc_config
        from localization.apriltag import load_tag_map
        from localization.landmarks import LandmarkDB
        from navigation.build import build_flight_stack
        from perception.io import read_pcd
        from nx_scout_loc import convert                      # image decoding, shared with nx_scout_loc
        from nx_scout_nav import node_logic

        self._convert = convert
        landmarks, map_pcd = p("landmarks", "").value, p("map_pcd", "").value
        if not landmarks or not map_pcd:
            raise RuntimeError("parameters 'landmarks' (.npz) and 'map_pcd' (prior map) are required")
        cfgs = loc_config.load_configs()
        hw = cfgs["hw"]
        cam = loc_config.camera_from_config(cfgs)
        T_body_cam = loc_config.body_from_camera(cfgs)
        xyz, _ = read_pcd(map_pcd)
        db = LandmarkDB.load(landmarks)
        tags = None
        if p("apriltags", "").value:
            with open(p("apriltags", "").value, encoding="utf-8") as f:
                tags = load_tag_map((yaml.safe_load(f) or {}).get("tags", {}))

        over = node_logic.mission_overrides(
            on_arrival=p("on_arrival", "hold").value, auto_arm=p("auto_arm", False).value,
            takeoff=p("takeoff", [0.0]).value, goal=p("goal", [0.0]).value, mode=p("mode", "wall_goal").value,
            hover_climb_m=p("hover_climb_m", -1.0).value, hover_s=p("hover_s", -1.0).value)

        rf = hw.get("range_flow", {})
        caps = {"height_ref": str(p("height_ref", "range").value), "max_xy_vel_mps": 1.5, "max_z_vel_up_mps": 1.0,
                "max_z_vel_dn_mps": 0.7, "max_yaw_rate_dps": 60.0,
                "source": "Scout FC dump after S5 (EKF2_HGT_REF=2, MPC_* limits)"}
        self.vehicle = Px4ScoutVehicle(self, caps, odom_latency_s=float(p("odom_latency_s", 0.03).value),
                                       range_below_body_m=float(rf.get("pos_from_fc_frd_m", [0, 0, 0])[2]),
                                       range_max_m=float(rf.get("range_max_m", 6.0)))
        self.pub_state = self.create_publisher(String, "scout_mission/state", 10)
        self.keeper = node_logic.StreamKeeper(self.vehicle, max_gap_s=float(p("keepalive_gap_s", 0.1).value))
        self.mission, self.comps = build_flight_stack(self.keeper, db, xyz, cam, T_body_cam, over,
                                                      log=self._on_event, apriltags=tags)
        self.keeper.is_streaming = lambda: node_logic.mission_streaming(self.mission)
        self.keeper.start(self._now)
        self._lock = threading.Lock()
        self._logic = node_logic
        self._img_lat = float(p("image_latency_s", 0.12).value)
        self._started = False
        self._st_proc, self._st_wait, self._st_lag, self._st_tick = (
            node_logic.DurationStats(), node_logic.DurationStats(), node_logic.DurationStats(),
            node_logic.DurationStats())
        self._n_frames = {"received": 0, "callback_stamped": 0}
        self._stats_period = float(p("stats_period_s", 10.0).value)
        self._stats_t = None
        g_img, g_tick = MutuallyExclusiveCallbackGroup(), MutuallyExclusiveCallbackGroup()
        q = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        compressed = bool(p("compressed", False).value)
        topic = p("image_topic", "scout/camera/image_raw").value
        self.create_subscription(CompressedImage if compressed else Image, topic,
                                 self._on_compressed if compressed else self._on_image, q, callback_group=g_img)
        self.create_timer(0.05, self._tick, callback_group=g_tick)
        self.get_logger().info(f"scout_mission ready: map {len(xyz)} pts, {len(db.points)} landmarks, camera {topic}")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_event(self, ev):
        self.get_logger().info(json.dumps(ev))
        self.pub_state.publish(String(data=json.dumps(ev)))

    def _on_image(self, msg, info):          # 2 required args: rclpy then passes MessageInfo
        stamp = self._stamp(info)
        try:
            img = self._convert.image_to_bgr(msg.encoding, msg.height, msg.width, msg.step, msg.data)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        self._frame(img, stamp)

    def _on_compressed(self, msg, info):
        stamp = self._stamp(info)
        try:
            img = self._convert.compressed_to_bgr(msg.data)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        self._frame(img, stamp)

    def _stamp(self, info):
        """DN-05: one clock (the laptop's) for everything; the frame's reception time, not the callback's."""
        now = self._now()
        rx = None
        if info is not None:
            try:
                rx = info["received_timestamp"]
            except (KeyError, TypeError):
                rx = None
        stamp, src = self._logic.frame_stamp(rx, now, self._img_lat)
        self._n_frames["received"] += 1
        if src != "received":
            self._n_frames["callback_stamped"] += 1
        else:
            self._st_lag.add(now - (stamp + self._img_lat))
        return stamp

    def _frame(self, img, stamp):
        if not self._started:
            return
        t0 = self._now()
        with self._lock:
            t1 = self._now()
            self.mission.on_image(img, stamp)
        self._st_wait.add(t1 - t0)
        self._st_proc.add(self._now() - t1)

    def _tick(self):
        now = self._now()
        if not self._started:
            st = self.vehicle.status()
            if self.vehicle.odometry() is None or st.info.get("stale_status") \
                    or not st.info.get("status_received", True):
                self.get_logger().info("waiting for PX4 odometry + status", throttle_duration_sec=5.0)
                return
            self._started = True
            self._stats_t = now
        with self._lock:
            t0 = self._now()
            self.mission.tick(t0)
            self._st_tick.add(self._now() - t0)
        if self._stats_period > 0 and now - self._stats_t >= self._stats_period:
            self._stats_t = now
            self._on_event({"t": round(now, 3), "state": self.mission.state, "event": "node_stats",
                            "frame_process": self._st_proc.summary(), "frame_lock_wait": self._st_wait.summary(),
                            "frame_rx_to_callback": self._st_lag.summary(), "tick": self._st_tick.summary(),
                            "frames": dict(self._n_frames), "setpoints": self.keeper.gap_summary()})


def main(args=None):
    rclpy.init(args=args)
    node = ScoutMissionNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.keeper.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

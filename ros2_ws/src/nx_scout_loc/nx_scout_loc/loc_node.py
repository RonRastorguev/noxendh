"""ROS 2 adapter for the Scout's camera relocalization (ARCHITECTURE §8.3). All logic is in `localization/`.

Subscribes : <image_topic>            sensor_msgs/Image (or CompressedImage with compressed:=true), sensor-data QoS
Publishes  : scout_pose               geometry_msgs/PoseStamped, frame `map`, body frame, on every accepted fix
             scout_camera_pose        geometry_msgs/PoseStamped, frame `map`, optical camera frame (for nx_scout_avoid)
             loc_ok                   std_msgs/Bool, every frame and from a timer (a dead stream also ends loc_ok)

Parameters: repo_root (default: $NOXEN_DH26_ROOT), landmarks (.npz from tools/build_landmark_map.py),
loc_config / hw_config / flight_config (default: the repo's config/), image_topic, compressed, map_frame,
watchdog_hz. Not run yet: there is no Scout camera stream (docs/SCOUT_LOC.md).

    ros2 run nx_scout_loc scout_loc --ros-args -p repo_root:=/path/to/noxen-dh26 -p landmarks:=/path/landmarks.npz
"""
from __future__ import annotations

import os
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool


class ScoutLocNode(Node):
    def __init__(self):
        super().__init__("scout_loc")
        p = self.declare_parameter
        repo = p("repo_root", os.environ.get("NOXEN_DH26_ROOT", "")).value
        if repo and repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            from localization import config as loc_config
            from . import convert
        except ImportError as e:
            raise RuntimeError("cannot import the localization core: set repo_root or NOXEN_DH26_ROOT") from e
        self._convert = convert
        landmarks = p("landmarks", "").value
        if not landmarks:
            raise RuntimeError("parameter 'landmarks' (landmark DB .npz) is required")
        cfgs = loc_config.load_configs(p("loc_config", "").value or None, p("hw_config", "").value or None,
                                       p("flight_config", "").value or None)
        self.loc = loc_config.build_localizer(cfgs, db_path=landmarks)
        self.frame = p("map_frame", "map").value
        compressed = bool(p("compressed", False).value)
        topic = p("image_topic", "scout/camera/image_raw").value
        self.pub_pose = self.create_publisher(PoseStamped, "scout_pose", 10)
        self.pub_cam = self.create_publisher(PoseStamped, "scout_camera_pose", 10)
        self.pub_ok = self.create_publisher(Bool, "loc_ok", 10)
        self.create_subscription(CompressedImage if compressed else Image, topic,
                                 self.on_compressed if compressed else self.on_image, qos_profile_sensor_data)
        self._last_img_stamp = None             # image clock
        self._last_img_wall = None              # monotonic receipt time of that image
        self.create_timer(1.0 / float(p("watchdog_hz", 10.0).value), self.on_timer)
        self.get_logger().info(f"scout_loc: {len(self.loc.estimators)} estimator(s), listening on {topic}")

    def on_image(self, msg: Image):
        try:
            img = self._convert.image_to_bgr(msg.encoding, msg.height, msg.width, msg.step, msg.data)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        self._process(img, msg.header.stamp)

    def on_compressed(self, msg: CompressedImage):
        try:
            img = self._convert.compressed_to_bgr(msg.data)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        self._process(img, msg.header.stamp)

    def _process(self, img, stamp_msg):
        stamp = self._convert.stamp_to_sec(stamp_msg.sec, stamp_msg.nanosec)
        out = self.loc.process(img, stamp)
        self._last_img_stamp, self._last_img_wall = stamp, time.monotonic()
        if out.fresh:
            self.pub_pose.publish(self._pose_msg(out.T_map_body, stamp_msg))
            self.pub_cam.publish(self._pose_msg(out.T_map_cam, stamp_msg))
        self.pub_ok.publish(Bool(data=bool(out.loc_ok)))

    def on_timer(self):
        """Between frames: the image clock advanced by the wall time since the last frame."""
        if self._last_img_stamp is None:
            self.pub_ok.publish(Bool(data=False))
            return
        now = self._last_img_stamp + (time.monotonic() - self._last_img_wall)
        if not self.loc.loc_ok_at(now):
            self.pub_ok.publish(Bool(data=False))

    def _pose_msg(self, T, stamp_msg) -> PoseStamped:
        (x, y, z), (qx, qy, qz, qw) = self._convert.pose_fields(T)
        m = PoseStamped()
        m.header.stamp = stamp_msg
        m.header.frame_id = self.frame
        m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = qx, qy, qz, qw
        return m


def main(args=None):
    rclpy.init(args=args)
    node = ScoutLocNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""SITL ONLY: a synthetic Scout camera for the `scout_mission` node flying PX4 SIH (docs/SCOUT_NAV.md §10).

PX4 SIH has no world to look at, so this node renders one: the closed-loop sim's synthetic room (flightsim/world.py)
WITHOUT the raised pad box (SIH's ground is flat, so the map must be too), at the aircraft's TRUE pose from SIH's
ground truth (MAVLink HIL_STATE_QUATERNION, which PX4 fills from vehicle_*_groundtruth; requested at 50 Hz on the
offboard MAVLink link, UDP 14540). The camera mount is the one the node uses (scout_loc.yaml body_from_camera), so
the node's PnP fixes measure the true pose and the map<->odom alignment has to cancel SIH's EKF wander
(0.3-0.7 m), as it must on the aircraft.

Map frame: SIH's first ground-truth sample (on the ground) is the takeoff pad pose A of the exported assets (yaw only;
SIH's own heading, whatever it is, becomes map yaw A).

    sitl_camera_sim.py --export DIR       # map.pcd + landmarks.npz + assets.json (pad pose, goal) for the node; exits
    sitl_camera_sim.py --assets DIR --out-dir ART [--rate-hz 10] [--latency-s 0.12] [--pose-source truth|odom]

Images: sensor_msgs/Image bgr8 on scout/camera/image_raw (BEST_EFFORT), published `latency_s` after capture (the node
stamps a frame "receive time - image_latency_s", so the two numbers must match for the stamp to be the capture
time). Writes ART/camera_truth.jsonl (true map pose of every frame) and ART/camera_sim.json (rates, render times).
`--pose-source odom` renders at PX4's EKF pose instead (no truth needed; the alignment then sees no wander).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)

from drivers.px4_scout import frames as F  # noqa: E402

WORLD_CFG = {"pad_box": None}             # SIH's ground is flat: take off from the floor (flightsim DN-88 option)
RENDER_SUPERSAMPLE = 1                    # flightsim renders at 2: 211 / 344 ms p50 / p95 here next to PX4 + the node
#                                           (first run). The map cache key does not include it: same landmark DB.
GOAL_NAME = "B1"
R_EARTH = 6371000.0


def _yaw_only(T):
    yaw = math.atan2(T[1, 0], T[0, 0])
    out = np.eye(4)
    out[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    out[:3, 3] = T[:3, 3]
    return out


def make_world(supersample=None):
    from flightsim.world import SimWorld
    return SimWorld(dict(WORLD_CFG, **({} if supersample is None else {"supersample": int(supersample)})))


def export_assets(out_dir: str) -> dict:
    from perception.io import write_pcd
    os.makedirs(out_dir, exist_ok=True)
    world = make_world()
    t0 = time.time()
    ms = world.map_side(verbose=True)
    write_pcd(os.path.join(out_dir, "map.pcd"), ms.map_xyz, np.zeros(len(ms.map_xyz)))
    ms.db.save(os.path.join(out_dir, "landmarks.npz"))
    A = world.takeoff_pose()
    a = {"takeoff": [float(A[0, 3]), float(A[1, 3]), float(A[2, 3]), math.degrees(math.atan2(A[1, 0], A[0, 0]))],
         "goal": [float(v) for v in world.wall_point(GOAL_NAME)], "goal_name": GOAL_NAME, "world_cfg": WORLD_CFG,
         "map_key": ms.meta.get("key"), "n_map_points": int(len(ms.map_xyz)), "n_landmarks": int(len(ms.db)),
         "map_side_s": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "assets.json"), "w", encoding="utf-8") as f:
        json.dump(a, f, indent=1)
    return a


def truth_to_odom(msg, origin) -> np.ndarray:
    """HIL_STATE_QUATERNION (lat/lon degE7, alt mm, q FRD->NED) -> T_odom_body (flat earth about `origin`)."""
    lat0, lon0, alt0 = origin
    lat, lon = math.radians(msg.lat * 1e-7), math.radians(msg.lon * 1e-7)
    n = (lat - lat0) * R_EARTH
    e = (lon - lon0) * R_EARTH * math.cos(lat0)
    d = -(msg.alt - alt0) * 1e-3
    return F.ned_to_odom_pose([n, e, d], list(msg.attitude_quaternion))


class TruthMav(threading.Thread):
    """Latest SIH ground truth over MAVLink (HIL_STATE_QUATERNION requested at 50 Hz)."""

    def __init__(self, url):
        super().__init__(daemon=True)
        from pymavlink import mavutil
        self.mu = mavutil
        self.conn = mavutil.mavlink_connection(url, source_system=250)
        self.latest = None                  # (receive monotonic, T_odom_body)
        self.origin = None
        self.n = 0
        self._last_req = 0.0

    def _request(self):
        self.conn.mav.command_long_send(1, 1, self.mu.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                        float(self.mu.mavlink.MAVLINK_MSG_ID_HIL_STATE_QUATERNION), 20000.0,
                                        0, 0, 0, 0, 0)

    def run(self):
        while True:
            now = time.monotonic()
            if (self.latest is None or now - self.latest[0] > 1.0) and now - self._last_req > 1.0:
                self._request()
                self._last_req = now
            m = self.conn.recv_match(type="HIL_STATE_QUATERNION", blocking=True, timeout=0.5)
            if m is None:
                continue
            if self.origin is None:
                self.origin = (math.radians(m.lat * 1e-7), math.radians(m.lon * 1e-7), float(m.alt))
            self.latest = (time.monotonic(), truth_to_odom(m, self.origin))
            self.n += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--export", help="write the node's assets to this dir and exit")
    ap.add_argument("--assets")
    ap.add_argument("--out-dir")
    ap.add_argument("--rate-hz", type=float, default=10.0)
    ap.add_argument("--latency-s", type=float, default=0.25)   # = the node's image_latency_s (run_sitl_check.sh)
    ap.add_argument("--gain", type=float, default=0.85)          # flightsim/loop.py camera defaults
    ap.add_argument("--noise-sigma", type=float, default=2.0)
    ap.add_argument("--pose-source", choices=("truth", "odom"), default="truth")
    ap.add_argument("--mavlink", default="udpin:0.0.0.0:14540")
    ap.add_argument("--topic", default="scout/camera/image_raw")
    a = ap.parse_args()
    if a.export:
        print(json.dumps(export_assets(a.export)))
        return 0
    if not (a.assets and a.out_dir):
        ap.error("--assets and --out-dir are required (or --export)")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    from localization.config import body_from_camera, load_configs

    with open(os.path.join(a.assets, "assets.json"), encoding="utf-8") as f:
        assets = json.load(f)
    tk = assets["takeoff"]
    yaw = math.radians(tk[3])
    T_map_pad = np.eye(4)
    T_map_pad[:3, :3] = [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    T_map_pad[:3, 3] = tk[:3]
    T_body_cam = body_from_camera(load_configs())
    world = make_world(RENDER_SUPERSAMPLE)
    print(f"[camera_sim] world ready ({len(world.room.surfaces)} surfaces), camera {world.cam.width}x{world.cam.height},"
          f" pose source {a.pose_source}", flush=True)

    rclpy.init()
    node = Node("scout_sitl_camera")
    q = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=2)
    pub = node.create_publisher(Image, a.topic, q)
    src = {"latest": None}
    if a.pose_source == "truth":
        tm = TruthMav(a.mavlink)
        tm.start()
        get_pose = lambda: tm.latest                      # noqa: E731
    else:
        from px4_msgs.msg import VehicleOdometry
        from drivers.px4_scout.px4_scout_vehicle import find_topic
        be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5)

        def on_odom(m):
            src["latest"] = (time.monotonic(), F.ned_to_odom_pose(m.position, m.q))
        node.create_subscription(VehicleOdometry, find_topic(node, "/fmu/out/vehicle_odometry"), on_odom, be)
        get_pose = lambda: src["latest"]                  # noqa: E731

    pending = queue.Queue()
    stats = {"frames": 0, "published": 0, "late_dropped": 0, "render_s": [], "pose_age_s": [], "anchor": None}
    flog = open(os.path.join(a.out_dir, "camera_truth.jsonl"), "w", encoding="utf-8")
    anchor = {"T": None}
    rng = np.random.default_rng(4242)
    stop = threading.Event()

    def render_loop():
        period = 1.0 / a.rate_hz
        t_next = time.monotonic()
        while not stop.is_set():
            t_next += period
            lp = get_pose()
            t_cap = time.monotonic()
            if lp is not None:
                if anchor["T"] is None:                       # first sample: the aircraft rests on pad A
                    anchor["T"] = T_map_pad @ np.linalg.inv(_yaw_only(lp[1]))
                    stats["anchor"] = {"T_odom_body0": np.round(lp[1], 4).tolist()}
                T_mb = anchor["T"] @ lp[1]
                t0 = time.perf_counter()
                img = world.render(T_mb @ T_body_cam, a.gain, a.noise_sigma, rng)
                stats["render_s"].append(time.perf_counter() - t0)
                stats["pose_age_s"].append(t_cap - lp[0])
                stats["frames"] += 1
                if time.monotonic() > t_cap + a.latency_s - 0.005:
                    stats["late_dropped"] += 1           # a real camera's latency is constant: drop, never publish late
                else:
                    msg = Image()
                    msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
                    msg.encoding, msg.step = "bgr8", int(img.shape[1] * 3)
                    msg.data = img.tobytes()
                    msg.header.frame_id = "scout_camera"
                    pending.put((t_cap + a.latency_s, msg))
                    flog.write(json.dumps({"t_mono": round(t_cap, 4),
                                           "ros_t": round(node.get_clock().now().nanoseconds * 1e-9, 4),
                                           "p_map": np.round(T_mb[:3, 3], 4).tolist(),
                                           "yaw_map_deg": round(math.degrees(math.atan2(T_mb[1, 0], T_mb[0, 0])), 2)})
                               + "\n")
            dt = t_next - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            else:
                t_next = time.monotonic()

    def publish_loop():
        # its own thread, sleeping until each frame is due: a 200 Hz ROS timer here starved the render thread of
        # the GIL (2nd SITL run: 393 of 547 frames dropped late)
        while not stop.is_set():
            try:
                due, msg = pending.get(timeout=0.2)
            except queue.Empty:
                continue
            dt = due - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            msg.header.stamp = node.get_clock().now().to_msg()
            pub.publish(msg)
            stats["published"] += 1

    th = threading.Thread(target=render_loop, daemon=True)
    th_pub = threading.Thread(target=publish_loop, daemon=True)
    th.start()
    th_pub.start()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):  # noqa: BLE001 - SIGTERM from the runner ends the spin
        pass
    finally:
        stop.set()
        th.join(timeout=2.0)
        th_pub.join(timeout=2.0)
        flog.close()
        rs, pa = np.asarray(stats["render_s"] or [0.0]), np.asarray(stats["pose_age_s"] or [0.0])
        out = {"frames": stats["frames"], "published": stats["published"], "late_dropped": stats["late_dropped"],
               "rate_hz": a.rate_hz, "supersample": RENDER_SUPERSAMPLE,
               "latency_s": a.latency_s, "pose_source": a.pose_source,
               "truth_msgs": getattr(locals().get("tm"), "n", None),
               "render_ms_p50_p95": [round(float(np.percentile(rs, 50)) * 1e3, 1),
                                     round(float(np.percentile(rs, 95)) * 1e3, 1)],
               "pose_age_ms_p95": round(float(np.percentile(pa, 95)) * 1e3, 1), "anchor": stats["anchor"]}
        with open(os.path.join(a.out_dir, "camera_sim.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(f"[camera_sim] {json.dumps(out)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

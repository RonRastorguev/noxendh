"""Data types shared by the Scout's flight software (ARCHITECTURE §8.3–8.5; design: docs/SCOUT_NAV.md).

Frames (all z up, metres, radians; poses are 4x4 `T_a_b` mapping frame-b points into frame a, as in localization/):
    map    the prior lidar map (Point-LIO). Planning, goals, obstacles and the geofence live here.
    odom   the flight controller's own local frame, re-expressed z-up / x-y-z right-handed by the vehicle adapter
           (PX4 NED -> this happens ONLY in the adapter). Gravity-aligned, drifts slowly in x, y, yaw; with a
           rangefinder height reference its z is height above whatever is below (docs/SCOUT_NAV.md §3).
    body   forward-left-up at the flight controller (REP-103 base_link).
    cam    OpenCV optical (x right, y down, z forward); `T_body_cam` from config.

Everything here is plain data: numpy + dataclasses, no ROS, Python 3.9.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OdomSample:
    """One flight-controller state estimate, already in the z-up odom frame."""
    stamp: float                         # seconds, the same clock as camera image stamps
    T_odom_body: np.ndarray              # 4x4
    v_odom: np.ndarray                   # (3,) body-origin velocity in odom, m/s
    reset_counter: int = 0               # increments whenever the FC resets its local frame (xy, z or yaw)
    dist_bottom_m: float | None = None   # rangefinder-derived height of the body origin above the surface below


@dataclass
class VehicleStatus:
    stamp: float
    armed: bool
    offboard: bool                       # the FC is in offboard mode (it follows our setpoints)
    landed: bool                         # the FC's own landed detector
    failsafe: bool                       # the FC reports an active failsafe
    battery_v: float | None = None
    info: dict = field(default_factory=dict)


@dataclass
class Setpoint:
    """A position setpoint in ODOM (the vehicle adapter converts it to the FC's frame). Stream it at >= 10 Hz."""
    stamp: float
    position: np.ndarray                 # (3,) odom
    yaw: float                           # odom yaw of body x
    velocity_ff: np.ndarray | None = None  # optional (3,) odom feed-forward velocity


@dataclass
class Obstacle:
    """An obstacle in MAP that the prior map does not contain, as an axis-aligned box."""
    id: int
    center: np.ndarray                   # (3,) map
    half_size: np.ndarray                # (3,) half extents; a dimension the camera could not see is extended
    top_seen: bool                       # the camera saw background ABOVE it (so flying over is an option)
    first_seen: float
    last_seen: float
    source: str                          # "looming_map" (map-anchored) | "looming_raw" (emergency, map-free)
    confidence: float = 0.0              # 0..1, detector's own measure
    info: dict = field(default_factory=dict)

    @property
    def lo(self) -> np.ndarray:
        return np.asarray(self.center) - np.asarray(self.half_size)

    @property
    def hi(self) -> np.ndarray:
        return np.asarray(self.center) + np.asarray(self.half_size)


@dataclass
class LoomingOutput:
    stamp: float
    emergency: bool                      # something in the flight corridor will be hit within ttc_emergency_s
    min_ttc_s: float                     # smallest corridor time-to-contact this frame (inf = none)
    obstacles: list                      # [Obstacle] detected THIS frame (memory lives in navigation)
    n_tracks: int = 0
    map_anchored: bool = False           # the map comparison ran this frame (needs a valid map pose)
    reason: str = ""
    debug: dict = field(default_factory=dict)


@dataclass
class GoalPose:
    """Where to hover in map to look at a wall point: position + yaw (camera faces the point)."""
    position: np.ndarray                 # (3,) map
    yaw: float                           # map yaw of body x
    target: np.ndarray                   # (3,) the wall point B itself
    normal: np.ndarray                   # (3,) unit wall normal at B, pointing into the room (towards the Scout)


@dataclass
class PlanResult:
    ok: bool
    waypoints: np.ndarray | None         # (N, 3) map, start first, goal position last
    reason: str = ""
    cost: float = float("inf")
    shape: str = ""                      # "direct" | "around" | "over" | "over+around"
    info: dict = field(default_factory=dict)

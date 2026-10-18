"""PX4 <-> Scout flight-software frames and message fields, ROS-free (docs/SCOUT_NAV.md §2).

The ONLY place PX4's NED / FRD conventions exist in this repo; navigation/ and avoidance/ see the z-up odom frame of
navigation/types.py. Same math as our reference PX4 odometry bridge:

    NED world -> ENU world : swap x / y, negate z          (R_EN, a proper rotation)
    FRD body  -> FLU body  : rotate pi about body x         (R_XPI)
    R(FLU -> ENU) = R_EN @ R(FRD -> NED) @ R_XPI
    yaw_enu = pi/2 - yaw_ned

Plain numbers in, plain numbers out, so it is unit-tested on Windows without px4_msgs. The ROS node
(`px4_scout_vehicle.py`) only copies message fields through these functions.

Constants are PX4's (px4_msgs VehicleCommand / VehicleStatus, PX4 1.18): copied here so the conversion layer does not
need px4_msgs to be importable.
"""
from __future__ import annotations

import math

import numpy as np

R_EN = np.array([[0.0, 1.0, 0.0],
                 [1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0]])
R_XPI = np.array([[1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0],
                  [0.0, 0.0, -1.0]])

# px4_msgs/VehicleCommand
VEHICLE_CMD_NAV_LAND = 21
VEHICLE_CMD_DO_SET_MODE = 176
VEHICLE_CMD_COMPONENT_ARM_DISARM = 400
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6.0
# px4_msgs/VehicleStatus
ARMING_STATE_ARMED = 2
NAVIGATION_STATE_OFFBOARD = 14
NAVIGATION_STATE_AUTO_LAND = 18


def quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def ned_to_odom_pose(position_ned, q_wxyz_frd_to_ned) -> np.ndarray:
    """PX4 VehicleOdometry position + q -> T_odom_body (4x4, z-up odom, FLU body)."""
    T = np.eye(4)
    T[:3, :3] = R_EN @ quat_wxyz_to_R(q_wxyz_frd_to_ned) @ R_XPI
    p = np.asarray(position_ned, float)
    T[:3, 3] = R_EN @ p
    return T


def ned_to_odom_vec(v_ned) -> np.ndarray:
    return R_EN @ np.asarray(v_ned, float)


def odom_to_ned_vec(v_odom) -> np.ndarray:
    return R_EN @ np.asarray(v_odom, float)          # R_EN is its own inverse


def yaw_odom_to_ned(yaw_odom: float) -> float:
    return math.atan2(math.sin(math.pi / 2 - yaw_odom), math.cos(math.pi / 2 - yaw_odom))


def yaw_ned_to_odom(yaw_ned: float) -> float:
    return yaw_odom_to_ned(yaw_ned)                  # the same reflection both ways


def trajectory_setpoint_fields(p_odom, yaw_odom, v_ff_odom=None) -> dict:
    """-> the TrajectorySetpoint fields (NED; NaN = not controlled) for a position setpoint."""
    nan3 = [float("nan")] * 3
    return {"position": [float(v) for v in odom_to_ned_vec(p_odom)],
            "velocity": nan3 if v_ff_odom is None else [float(v) for v in odom_to_ned_vec(v_ff_odom)],
            "acceleration": nan3, "jerk": nan3,
            "yaw": float(yaw_odom_to_ned(yaw_odom)), "yawspeed": float("nan")}


def offboard_mode_fields() -> dict:
    return {"position": True, "velocity": False, "acceleration": False, "attitude": False, "body_rate": False}


def command_fields(kind: str) -> dict:
    """-> VehicleCommand fields for "arm" | "offboard" | "land" (target 1/1, from external)."""
    base = {"target_system": 1, "target_component": 1, "source_system": 1, "source_component": 1,
            "from_external": True, "param1": 0.0, "param2": 0.0}
    if kind == "arm":
        base.update(command=VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
    elif kind == "offboard":
        base.update(command=VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=PX4_CUSTOM_MAIN_MODE_OFFBOARD)
    elif kind == "land":
        base.update(command=VEHICLE_CMD_NAV_LAND)
    else:
        raise ValueError(f"unknown command {kind!r}")
    return base


def status_from_fields(stamp, arming_state, nav_state, failsafe, landed, battery_v=None) -> dict:
    return {"stamp": float(stamp), "armed": int(arming_state) == ARMING_STATE_ARMED,
            "offboard": int(nav_state) == NAVIGATION_STATE_OFFBOARD, "landed": bool(landed),
            "failsafe": bool(failsafe), "battery_v": battery_v,
            "info": {"nav_state": int(nav_state), "arming_state": int(arming_state)}}

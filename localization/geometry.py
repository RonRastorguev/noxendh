"""Rigid-transform helpers. `T_a_b` (4x4) maps points from frame b into frame a."""
from __future__ import annotations

import numpy as np


def make_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inv_T(T) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def transform(T, pts) -> np.ndarray:
    """Apply T to (N, 3) points."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    return pts @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def rot_angle_deg(Ra, Rb) -> float:
    """Angle of the rotation Ra^T Rb, degrees."""
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def pose_error(T_est, T_gt) -> tuple:
    """(position error m, rotation error deg) between two poses."""
    return (float(np.linalg.norm(np.asarray(T_est)[:3, 3] - np.asarray(T_gt)[:3, 3])),
            rot_angle_deg(np.asarray(T_est)[:3, :3], np.asarray(T_gt)[:3, :3]))


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Optical camera (x right, y down, z forward) expressed in a forward-left-up body frame (REP-103 base_link).
R_FLU_OPTICAL = np.array([[0.0, 0.0, 1.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0]])


def body_pose(position, yaw, pitch=0.0, roll=0.0) -> np.ndarray:
    """T_map_body for a forward-left-up body, z-up map. Positive pitch = nose down (rotation about body y = left)."""
    return make_T(rot_z(yaw) @ rot_y(pitch) @ rot_x(roll), position)


def look_at(position, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """T_map_cam for an optical camera at `position` looking at `target` with no roll about the view axis."""
    position = np.asarray(position, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - position
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_T(np.stack([x, y, z], axis=1), position)


def R_to_quat_xyzw(R) -> np.ndarray:
    """Rotation matrix -> unit quaternion (x, y, z, w), w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return -q if q[3] < 0 else q


def quat_xyzw_to_R(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def interpolate_pose(stamps, poses, t) -> np.ndarray | None:
    """Pose at time t from a time-sorted trajectory (linear position, slerp rotation). None outside its span."""
    stamps = np.asarray(stamps, dtype=np.float64)
    if len(stamps) == 0 or t < stamps[0] or t > stamps[-1]:
        return None
    i = int(np.searchsorted(stamps, t, side="right"))
    if i >= len(stamps):
        return np.array(poses[-1], dtype=np.float64)
    i0, i1 = i - 1, i
    span = stamps[i1] - stamps[i0]
    a = 0.0 if span <= 0 else (t - stamps[i0]) / span
    T0, T1 = np.asarray(poses[i0]), np.asarray(poses[i1])
    q = _slerp(R_to_quat_xyzw(T0[:3, :3]), R_to_quat_xyzw(T1[:3, :3]), a)
    return make_T(quat_xyzw_to_R(q), (1 - a) * T0[:3, 3] + a * T1[:3, 3])

"""Message <-> numpy conversion for the node, with NO ROS imports (so it is unit-tested in plain Python).

Works on the fields of sensor_msgs/Image (encoding, height, width, step, data) and sensor_msgs/CompressedImage
(data), and turns a 4x4 pose into the position + quaternion fields of geometry_msgs/Pose.
"""
from __future__ import annotations

import numpy as np

_CHANNELS = {"mono8": 1, "bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}


def image_to_bgr(encoding: str, height: int, width: int, step: int, data) -> np.ndarray:
    """sensor_msgs/Image fields -> HxWx3 uint8 BGR (or HxW for mono8). Raises ValueError for other encodings."""
    import cv2
    enc = encoding.lower()
    if enc not in _CHANNELS:
        raise ValueError(f"unsupported image encoding {encoding!r} (supported: {sorted(_CHANNELS)})")
    ch = _CHANNELS[enc]
    buf = np.frombuffer(bytes(data), dtype=np.uint8)
    if step < width * ch or len(buf) < step * height:
        raise ValueError(f"image buffer too small: step {step}, {len(buf)} bytes for {width}x{height}x{ch}")
    rows = buf[: step * height].reshape(height, step)[:, : width * ch]
    img = rows.reshape(height, width, ch) if ch > 1 else rows.reshape(height, width)
    if enc == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return np.ascontiguousarray(img)


def compressed_to_bgr(data) -> np.ndarray:
    import cv2
    img = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode compressed image")
    return img


def pose_fields(T) -> tuple:
    """4x4 pose -> ((x, y, z), (qx, qy, qz, qw))."""
    from localization.geometry import R_to_quat_xyzw
    T = np.asarray(T, dtype=np.float64)
    q = R_to_quat_xyzw(T[:3, :3])
    return tuple(float(v) for v in T[:3, 3]), tuple(float(v) for v in q)


def stamp_to_sec(sec: int, nanosec: int) -> float:
    return float(sec) + float(nanosec) * 1e-9

"""Accumulate points from a Unitree L2 over UDP, without ROS (DH-03).

The vendor-specific part (wire format, calibration math, coalesced datagrams) is all in
`l2_protocol.py`, reused verbatim (approved, DH-02). This module only opens a socket,
collects raw datagrams for a fixed window, then parses them. It never sends anything to the sensor.

For a FIXED sensor (the table demo) the per-point time inside the window does not matter: the
accumulated cloud is the union of every point received. A moving sensor needs the ROS driver's
timebase and de-skewing instead (DH-09); that is not this module's job.
"""
from __future__ import annotations

import socket
import time
from dataclasses import dataclass, field

import numpy as np

from . import l2_protocol as proto

_RCVBUF_BYTES = 8 * 1024 * 1024     # ~470 datagrams/s x ~1.1 kB: several seconds of headroom


class NoLidarData(RuntimeError):
    """No datagram arrived within the first-data timeout (L2 off, wrong IP/port, firewall)."""


@dataclass
class Accumulation:
    xyz: np.ndarray                 # (N, 3) float32, sensor frame (the L2's own axes)
    intensity: np.ndarray           # (N,) float32 in [0, 1] (raw u8 / 255; not reflectivity-calibrated)
    duration_s: float               # wall time from the first datagram to the last one kept
    first_data_wait_s: float        # socket open -> first datagram
    n_datagrams: int = 0
    n_point_packets: int = 0
    n_imu_packets: int = 0
    drops: dict = field(default_factory=dict)
    imu_accel_mean: list | None = None  # mean accelerometer vector, native units, the L2's axes: "up" when at rest

    def stats(self) -> dict:
        return {"n_points": int(len(self.xyz)), "duration_s": round(self.duration_s, 3),
                "first_data_wait_s": round(self.first_data_wait_s, 3), "n_datagrams": self.n_datagrams,
                "n_point_packets": self.n_point_packets, "n_imu_packets": self.n_imu_packets,
                "drops": dict(self.drops), "imu_accel_mean": self.imu_accel_mean}


def parse_datagrams(datagrams, range_min_m: float, range_max_m: float, imu_accel_out: list | None = None):
    """Raw UDP payloads -> (xyz, intensity, n_point_packets, n_imu_packets, drops). Pure, testable.
    `imu_accel_out`, if given, collects each IMU packet's linear_acceleration (native units)."""
    stats = proto.DropStats()
    xs, ints = [], []
    n_pt = n_imu = 0
    for data in datagrams:
        for frame in proto.split_datagram(data, stats):
            parsed = proto.parse_packet(frame, stats)
            if parsed is None:
                continue
            ptype, rec = parsed
            if ptype == proto.PT_POINT:
                n_pt += 1
                got = proto.point_packet_xyzit(rec, range_min_m, range_max_m)
                if got is not None:
                    x, y, z, intensity, _t = got
                    xs.append(np.stack([x, y, z], axis=1))
                    ints.append(intensity)
            elif ptype == proto.PT_IMU:
                n_imu += 1
                if imu_accel_out is not None:
                    imu_accel_out.append(np.asarray(rec["linear_acceleration"], dtype=np.float64))
    if xs:
        xyz = np.concatenate(xs).astype(np.float32)
        intensity = np.concatenate(ints).astype(np.float32)
        keep = np.all(np.isfinite(xyz), axis=1) & np.isfinite(intensity)
        if not keep.all():
            stats.drop("nonfinite_point", int((~keep).sum()))
        xyz, intensity = xyz[keep], intensity[keep]
    else:
        xyz, intensity = np.zeros((0, 3), np.float32), np.zeros(0, np.float32)
    return xyz, intensity, n_pt, n_imu, dict(stats.counts)


class L2Receiver:
    def __init__(self, local_ip: str = "0.0.0.0", local_port: int = 6201,
                 range_min_m: float = 0.05, range_max_m: float = 30.0):
        self.local_ip, self.local_port = local_ip, int(local_port)
        self.range_min_m, self.range_max_m = float(range_min_m), float(range_max_m)

    def accumulate(self, duration_s: float, first_data_timeout_s: float = 3.0, progress=None) -> Accumulation:
        """Open a fresh socket, wait for the first datagram, collect for `duration_s`, then parse.

        A fresh socket per capture means nothing buffered from before the request can leak into the
        scan. Raises NoLidarData if nothing arrives within `first_data_timeout_s`. `progress`, if
        given, is called with the fraction of the window elapsed (0..1) about every 0.25 s."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RCVBUF_BYTES)
            sock.bind((self.local_ip, self.local_port))
            t_open = time.monotonic()
            sock.settimeout(first_data_timeout_s)
            try:
                first, _ = sock.recvfrom(65536)
            except socket.timeout:
                raise NoLidarData(f"no L2 datagram on {self.local_ip}:{self.local_port} within "
                                  f"{first_data_timeout_s} s (sensor powered? laptop Ethernet 192.168.1.2? "
                                  f"firewall allows UDP {self.local_port}?)") from None
            t_first = time.monotonic()
            datagrams = [first]
            t_last = t_first
            deadline = t_first + duration_s
            next_tick = t_first + 0.25
            sock.settimeout(0.2)
            while True:
                now = time.monotonic()
                if now >= deadline:
                    break
                if progress is not None and now >= next_tick:
                    progress(min(1.0, (now - t_first) / duration_s))
                    next_tick = now + 0.25
                try:
                    data, _ = sock.recvfrom(65536)
                except socket.timeout:
                    continue
                datagrams.append(data)
                t_last = time.monotonic()
        finally:
            sock.close()
        accel = []
        xyz, intensity, n_pt, n_imu, drops = parse_datagrams(datagrams, self.range_min_m, self.range_max_m, accel)
        accel_mean = [round(float(v), 5) for v in np.mean(accel, axis=0)] if accel else None
        return Accumulation(xyz=xyz, intensity=intensity, duration_s=t_last - t_first,
                            first_data_wait_s=t_first - t_open, n_datagrams=len(datagrams),
                            n_point_packets=n_pt, n_imu_packets=n_imu, drops=drops, imu_accel_mean=accel_mean)

"""Unitree L2 wire protocol — pure functions, no ROS, no rclpy import.

Everything device-specific for candidate C (F2, protocol-level driver) lives
here so it is testable offline (`test_protocol.py`, the H7 malformed-input
test) and reusable by both `driver.py` (real UDP socket) and
`conformance_feed.py` (synthetic UDP packet generator).

Wire layout: the vendor protocol notes. They are is explicit that
struct sizes are UNVERIFIED (the web-fetch summariser's own byte count
disagreed with the struct it printed) and the CRC32 polynomial is not
published. This module therefore:
  * derives byte offsets from the struct layout using NATURAL ALIGNMENT
    (every field is a 4-byte type or an array of one, so there is no padding
    to guess at) and asserts the resulting sizes at import time — a
    contradiction between our reading and the numbers below fails LOUDLY at
    import, not silently at runtime;
  * NEVER trusts a fixed offset table without also checking `packet_size`
    against what THIS packet_type is supposed to be — a mismatch is dropped
    and counted (H7), never guessed at;
  * verifies crc32: zlib CRC-32 over the data bytes (identified 2026-09-30 on
    the real unit; it matched every captured packet). A mismatch is dropped
    and counted. (Old text, retained: "does NOT verify crc32 (the polynomial
    is unpublished): the field is read and ignored".)
"""
import struct
import zlib
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Frame constants (unitree_lidar_protocol.h, verbatim from the vendor notes).
FRAME_HEADER_MAGIC = bytes([0x55, 0xAA, 0x05, 0x0A])
FRAME_TAIL_MAGIC = bytes([0x00, 0xFF])
PT_POINT = 102
PT_2D_POINT = 103          # not produced by this candidate's feed; unsupported by the driver
PT_IMU = 104
PT_VERSION = 105           # unsupported: dropped + counted, not a fault
PT_TIME_STAMP = 106        # unsupported: dropped + counted, not a fault
KNOWN_UNSUPPORTED_TYPES = (PT_2D_POINT, PT_VERSION, PT_TIME_STAMP)

HEADER_LEN = 12             # header[4] + packet_type(u4) + packet_size(u4)
TAIL_LEN = 12                # crc32(u4) + msg_type_check(u4) + reserve[2] + tail[2]
_HEADER_STRUCT = struct.Struct("<4sII")
_TAIL_STRUCT = struct.Struct("<II2s2s")

MAX_POINTS_PER_PACKET = 300

# LidarPointData, natural alignment (no #pragma pack in the header; every
# member here is a 4-byte scalar/array so there is nothing for a compiler to
# pad): DataInfo(16) + LidarInsideState(36) + LidarCalibParam(32) +
# 8 more floats(32) + point_num(4) + ranges[300](600) + intensities[300](300)
# = 1020 B, per the vendor-notes arithmetic (source of the 1020/1044 figures).
POINT_DATA_DT = np.dtype([
    ("info_seq", "<u4"), ("info_payload_size", "<u4"),
    ("info_sec", "<u4"), ("info_nsec", "<u4"),
    ("sys_rotation_period", "<u4"), ("com_rotation_period", "<u4"),
    ("dirty_index", "<f4"), ("packet_lost_up", "<f4"), ("packet_lost_down", "<f4"),
    ("apd_temperature", "<f4"), ("apd_voltage", "<f4"), ("laser_voltage", "<f4"),
    ("imu_temperature", "<f4"),
    ("a_axis_dist", "<f4"), ("b_axis_dist", "<f4"),
    ("theta_angle_bias", "<f4"), ("alpha_angle_bias", "<f4"),
    ("beta_angle", "<f4"), ("xi_angle", "<f4"),
    ("range_bias", "<f4"), ("range_scale", "<f4"),
    ("com_horizontal_angle_start", "<f4"), ("com_horizontal_angle_step", "<f4"),
    ("scan_period", "<f4"), ("range_min", "<f4"), ("range_max", "<f4"),
    ("angle_min", "<f4"), ("angle_increment", "<f4"), ("time_increment", "<f4"),
    ("point_num", "<u4"),
    ("ranges", "<u2", (MAX_POINTS_PER_PACKET,)),
    ("intensities", "u1", (MAX_POINTS_PER_PACKET,)),
])
POINT_DATA_LEN = 1020
assert POINT_DATA_DT.itemsize == POINT_DATA_LEN, (
    f"POINT_DATA_DT.itemsize={POINT_DATA_DT.itemsize} != {POINT_DATA_LEN}: "
    "the vendor-notes arithmetic and this layout disagree — fix before trusting a byte of it")
POINT_PACKET_LEN = HEADER_LEN + POINT_DATA_LEN + TAIL_LEN     # 1044

# LidarImuData: DataInfo(16) + quaternion[4](16) + angular_velocity[3](12) +
# linear_acceleration[3](12) = 56 B.
IMU_DATA_DT = np.dtype([
    ("info_seq", "<u4"), ("info_payload_size", "<u4"),
    ("info_sec", "<u4"), ("info_nsec", "<u4"),
    ("quaternion", "<f4", (4,)),
    ("angular_velocity", "<f4", (3,)),
    ("linear_acceleration", "<f4", (3,)),
])
IMU_DATA_LEN = 56
assert IMU_DATA_DT.itemsize == IMU_DATA_LEN, (
    f"IMU_DATA_DT.itemsize={IMU_DATA_DT.itemsize} != {IMU_DATA_LEN}")
IMU_PACKET_LEN = HEADER_LEN + IMU_DATA_LEN + TAIL_LEN          # 80

_EXPECTED_DATA_LEN = {PT_POINT: POINT_DATA_LEN, PT_IMU: IMU_DATA_LEN}
_DTYPE = {PT_POINT: POINT_DATA_DT, PT_IMU: IMU_DATA_DT}

G_TO_MPS2 = 9.80665
DEG_TO_RAD = 3.141592653589793 / 180.0

# Native UDP endpoint defaults, shared by driver.py and conformance_feed.py so
# there is exactly one place a port number is spelled. `real` driver_params
# (config/hw/*.yaml) may override any of these; the F2/C conformance run
# overrides them via the manifest's `conformance.driver_params` to loopback
# ports 6131 (device/feed) / 6231 (driver) per the F2 builder logistics table.
NATIVE_DEFAULTS = {
    "local_ip": "127.0.0.1", "local_port": 6231,       # where THIS DRIVER listens
    "lidar_ip": "127.0.0.1", "lidar_port": 6131,        # the device's own port (feed binds here)
    "cloud_scan_num": 18,
    "imu_accel_unit": "mps2", "imu_gyro_unit": "rad_s",
    "seq_modulus": None,        # None = 32-bit counter; the real L2 wraps DataInfo.seq at 1024 (2026-09-30 pcap)
}


class DropStats:
    """Counts + a throttled human-readable reason per drop cause (H7: rejected
    loudly, not guessed). No ROS dependency: `sink` is any callable(str)."""

    def __init__(self, sink=None):
        self.counts = {}
        self.events = {}        # counted NON-loss events (note()); never part of total()
        self.sink = sink or (lambda msg: None)

    def drop(self, reason, detail=None):
        n = self.counts.get(reason, 0) + 1
        self.counts[reason] = n
        # log the first few occurrences of each reason, then quiet down
        if n <= 3 or n % 500 == 0:
            self.sink(f"[unitree_l2] DROP {reason} (n={n}): {detail}")

    def total(self):
        return sum(self.counts.values())

    def note(self, event, detail=None):
        """A counted, throttled, logged EVENT that is not a loss (e.g. a coalesced datagram): kept apart from
        `counts`, whose total() means frames dropped."""
        n = self.events.get(event, 0) + 1
        self.events[event] = n
        if n <= 3 or n % 500 == 0:
            self.sink(f"[unitree_l2] NOTE {event} (n={n}): {detail}")


def split_datagram(buf, stats=None):
    """One UDP datagram -> the list of frames it carries.

    Measured 2026-10-06 on the real L2 with its IMU on (bench log, window 2): an IMU frame
    (80 B) and a point frame (1 044 B) can arrive back to back in ONE 1 124-byte datagram (187 of 8 331 in 20 s),
    and config replies come with their ACK the same way. parse_packet() checks one frame against the datagram
    length, so such a datagram used to be dropped whole, losing both frames.

    Never salvages part of a datagram: the frames are returned only if EVERY one starts with the frame magic,
    declares HEADER_LEN + TAIL_LEN <= packet_size, and the sizes tile the datagram to the last byte. Otherwise
    the datagram comes back whole ([buf]) and parse_packet() rejects it exactly as before. Each returned frame
    still goes through every parse_packet() check (magic, size == length, tail, type, data length, CRC) on its
    own. A coalesced datagram is counted with stats.note("coalesced_datagram") (not a drop)."""
    n, off, frames = len(buf), 0, []
    while off + HEADER_LEN + TAIL_LEN <= n:
        magic, _ptype, size = _HEADER_STRUCT.unpack_from(buf, off)
        if magic != FRAME_HEADER_MAGIC or size < HEADER_LEN + TAIL_LEN or off + size > n:
            return [buf]
        frames.append(buf[off:off + size])
        off += size
    if off != n or len(frames) < 2:
        return [buf]
    if stats is not None:
        stats.note("coalesced_datagram", {"frames": len(frames), "len": n})
    return frames


def parse_packet(buf, stats):
    """UDP datagram -> (packet_type, structured record) or None (dropped;
    `stats` already updated). Never raises, never guesses a size.

    Trusts `header.packet_size` only after checking it against the ACTUAL
    datagram length AND the size this packet_type is supposed to have.
    Everything else — bad magic, wrong tail, truncated payload, an unknown
    packet_type — is counted and dropped, never parsed as if it were valid.
    """
    n = len(buf)
    if n < HEADER_LEN + TAIL_LEN:
        stats.drop("short_packet", {"n": n})
        return None
    magic, packet_type, packet_size = _HEADER_STRUCT.unpack_from(buf, 0)
    if magic != FRAME_HEADER_MAGIC:
        stats.drop("bad_header_magic", {"got": magic.hex()})
        return None
    if packet_size != n:
        # header.packet_size is the total packet length (vendor notes): if it
        # does not match what we actually received, we do NOT trust it as an
        # offset table — drop whole.
        stats.drop("packet_size_mismatch_actual", {"declared": packet_size, "actual": n})
        return None
    crc32, msg_type_check, reserve, tail = _TAIL_STRUCT.unpack_from(buf, n - TAIL_LEN)
    del msg_type_check, reserve
    if tail != FRAME_TAIL_MAGIC:
        stats.drop("bad_tail_magic", {"got": tail.hex()})
        return None
    if packet_type not in _EXPECTED_DATA_LEN:
        if packet_type in KNOWN_UNSUPPORTED_TYPES:
            stats.drop("unsupported_packet_type_known", {"packet_type": packet_type})
        else:
            stats.drop("unsupported_packet_type_unknown", {"packet_type": packet_type})
        return None
    data_len = n - HEADER_LEN - TAIL_LEN
    if data_len != _EXPECTED_DATA_LEN[packet_type]:
        # packet_size claimed to match `n`, but `n` itself is not the size we
        # expect for this packet_type: a genuinely different (undocumented)
        # variant, or corruption that happened to keep packet_size consistent
        # with the truncated length. Either way: do not guess the layout.
        stats.drop("unexpected_data_len_for_type",
                   {"packet_type": packet_type, "data_len": data_len,
                    "expected": _EXPECTED_DATA_LEN[packet_type]})
        return None
    # CRC (identified 2026-09-30 on the real unit and on the vendor library's own
    # commands: standard zlib CRC-32 over the DATA bytes only, header and tail
    # excluded; it matched on every captured packet). A corrupted datagram is
    # dropped, never parsed.
    if zlib.crc32(bytes(buf[HEADER_LEN:n - TAIL_LEN])) & 0xFFFFFFFF != crc32:
        stats.drop("crc_mismatch", {"packet_type": packet_type})
        return None
    rec = np.frombuffer(buf, dtype=_DTYPE[packet_type], count=1, offset=HEADER_LEN)[0]
    return packet_type, rec


def point_packet_xyzit(rec, range_min_m, range_max_m):
    """One LidarPointData record -> (x, y, z, intensity[0,1], t_within_packet_s)
    for its valid points, vectorised (no per-point python loop). Published
    calibration math (vendor protocol notes):

        range_m   = range_scale * (ranges_raw + range_bias)
                    (NO /1000: vendor parseFromPacketToPointCloud, unilidar_sdk2
                    0e3c51f; the real unit sends range_scale 0.000982, bias -301.6,
                    i.e. range_scale itself carries raw->metres. The 2026-09-17
                    notes' '/1000' was a misreading that zeroed every real point.)
        alpha[j]  = angle_min + alpha_angle_bias + j * angle_increment
        theta[j]  = com_horizontal_angle_start + theta_angle_bias
                    + j * com_horizontal_angle_step
        A = (-cos(beta)*sin(xi) + sin(beta)*cos(xi)*sin(alpha)) * r + b_axis_dist
        B =   cos(alpha)*cos(xi) * r
        C = ( sin(beta)*sin(xi) + cos(beta)*cos(xi)*sin(alpha)) * r
        x = cos(theta)*A - sin(theta)*B
        y = sin(theta)*A + cos(theta)*B
        z = C + a_axis_dist

    `ranges[j] == 0` is the vendor's "invalid sample" marker (dropped, not a
    protocol error). Points outside [range_min_m, range_max_m] are dropped
    here too (LidarSource contract normalisation)."""
    n = int(rec["point_num"])
    n = 0 if n < 0 else min(n, MAX_POINTS_PER_PACKET)
    if n == 0:
        return None
    ranges_mm = rec["ranges"][:n].astype(np.float64)
    inten_raw = rec["intensities"][:n].astype(np.float64)
    range_bias = float(rec["range_bias"])
    range_scale = float(rec["range_scale"])
    r = range_scale * (ranges_mm + range_bias)       # vendor-exact, see docstring

    j = np.arange(n, dtype=np.float64)
    alpha = float(rec["angle_min"]) + float(rec["alpha_angle_bias"]) + j * float(rec["angle_increment"])
    theta = (float(rec["com_horizontal_angle_start"]) + float(rec["theta_angle_bias"])
             + j * float(rec["com_horizontal_angle_step"]))
    t = j * float(rec["time_increment"])

    beta, xi = float(rec["beta_angle"]), float(rec["xi_angle"])
    a_axis, b_axis = float(rec["a_axis_dist"]), float(rec["b_axis_dist"])
    cb, sb, cx, sx = np.cos(beta), np.sin(beta), np.cos(xi), np.sin(xi)
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    A = (-cb * sx + sb * cx * sa) * r + b_axis
    B = ca * cx * r
    C = (sb * sx + cb * cx * sa) * r
    x = ct * A - st * B
    y = st * A + ct * B
    z = C + a_axis

    intensity = inten_raw / 255.0
    range_mag = np.sqrt(x * x + y * y + z * z)
    valid = ((ranges_mm > 0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
              & (range_mag >= range_min_m) & (range_mag <= range_max_m))
    if not np.any(valid):
        return None
    return x[valid], y[valid], z[valid], intensity[valid], t[valid]


def imu_native_to_si(rec, accel_unit, gyro_unit):
    """LidarImuData record, native units (driver_params imu_accel_unit /
    imu_gyro_unit) -> (accel_mps2 (3,), gyro_rads (3,), quaternion_xyzw (4,)).
    Native units are UNVERIFIED (vendor protocol notes): the driver
    never guesses them, it takes them as a declared parameter."""
    accel = np.asarray(rec["linear_acceleration"], dtype=np.float64)
    gyro = np.asarray(rec["angular_velocity"], dtype=np.float64)
    quat = np.asarray(rec["quaternion"], dtype=np.float64)
    if accel_unit == "g":
        accel = accel * G_TO_MPS2
    elif accel_unit != "mps2":
        raise ValueError(f"unknown imu_accel_unit '{accel_unit}' (want g|mps2)")
    if gyro_unit == "deg_s":
        gyro = gyro * DEG_TO_RAD
    elif gyro_unit != "rad_s":
        raise ValueError(f"unknown imu_gyro_unit '{gyro_unit}' (want rad_s|deg_s)")
    return accel, gyro, quat


# ---------------------------------------------------------------------------
# Device -> node timebase (lidar_source.md "Timestamp semantics", imu_source.md
# "ONE offset to both streams"). Pure, no ROS/socket import: `driver.py` feeds
# these plain-float functions/classes from its own recv-thread timestamps and
# publishes whatever they return; `test_l2_protocol.py` drives them directly.
#
# Model: every point packet has a DEVICE time (the vendor DataInfo.stamp if
# usable, else reconstructed from `seq` + the packet's own declared span —
# `L2TimeSync.point_packet_device_time` below). `L2TimeSync` maintains ONE
# running estimate of `offset = recv_time - device_time` (seconds, node
# clock) shared by the LiDAR and IMU streams, and `node_time(device_time) =
# device_time + offset`. The offset is estimated by a MIN-filter over a
# sliding window (the smallest observed recv-minus-device sample in the
# window is the least-delayed one, i.e. the closest to the true
# propagation+processing delay — standard one-way clock-offset practice) and
# only allowed to move by a bounded slew rate — the larger of a per-second
# (real time) budget and a per-sample (packet span) budget, see
# `_update_offset` — so a monotonic device-time input always maps to a
# strictly-increasing published stamp, INCLUDING immediately after a burst of
# backlogged packets drains in a near-zero real-time window. A real-time
# stall or a device-time jump forces a hard re-anchor (counted); an
# untrustworthy `seq` delta alone does NOT (`VirtualDeviceClock`: `seq` is
# not guaranteed globally unique by this driver's contract, only `t` is —
# only a `seq` anomaly that ALSO coincides with a real-time stall is logged
# as a probable device reboot, see `_update_offset`).
DEFAULT_OFFSET_WINDOW_S = 2.0        # sliding window for the min-filter
DEFAULT_MAX_SLEW_S_PER_S = 0.05      # bounded offset correction rate, per second of REAL time
_PER_SAMPLE_SLEW_FRACTION = 0.8      # bounded offset correction rate, per SAMPLE (see _update_offset)
DEFAULT_STALL_RESET_S = 1.0          # a recv-to-recv gap this large forces a re-anchor
_SEQ_RESET_GAP = 1_000_000           # a seq delta bigger than this is not ordinary packet loss
# A `seq` delta THIS big is only trustworthy as a genuine 32-bit counter
# wraparound if `last_seq` was already within this margin of 0xFFFFFFFF —
# otherwise it is NOT a real wraparound (see `VirtualDeviceClock.advance`).
_SEQ_NEAR_WRAP_MARGIN = 10_000
# Out-of-order / duplicated point packets (reported by the lio-l2 session 2026-10-07 against d19e7c0: after a
# ~25 ms receive stall the driver logged `cloud_seq_gap_lost_packets n_lost 1024` -- exactly the signature of two
# adjacent packets SWAPPED (a, a+2, a+1, a+3 -> 1 + 1022 + 1) -- and stamped clouds up to 4.7 s = ~1023 x 4.6 ms in
# the future until the disjointness floor caught up). UDP order is not guaranteed (a loaded host, CPU migration
# on loopback, a NIC queue). A point packet up to SEQ_MAX_LATE_PACKETS BEHIND the last accepted one, arriving
# within SEQ_LATE_WINDOW_S of it, is LATE: dropped and counted, never read as ~1023 lost packets. A larger backward
# step is not judged late (the conformance feed's own counter wraps 1023 -> 0 against a 32-bit modulus; a device
# reboot restarts it), and the existing logic handles it.
SEQ_MAX_LATE_PACKETS = 64            # ~0.3 s of L2 point packets (4.6 ms each): far beyond any plausible reorder
SEQ_LATE_WINDOW_S = 1.0              # a backward step after this much real time is a resync, not a late packet


# The DEVICE-STAMP clock (2026-10-07, after the lio-l2 IMU-timebase request). Measured on the real L2 (window-2
# sequence, 225-265 s): DataInfo.stamp runs at ~0.5027x real time but is LINEAR: a straight line fits it to packet
# ARRIVAL with p99 0.28 ms residual over 5 s windows (0.62 ms over 40 s), while the CADENCE clock (one `scan_period`
# per packet) runs 0.8 % slow and wanders p99 8 ms / max 12 ms off arrival over 5 s (a smooth drift, autocorrelation
# 0.998 at lag 1: the period field understates the real interval, it is not emission jitter). So once a rate is
# established, point packets AND IMU frames take their device time from the stamp: device_s = base_vt +
# (stamp - base_stamp) / rate. ONE map for both streams (imu_source.md "ONE offset to both"), monotone in the stamp,
# no IMU sample dropped. The cadence clock stays the fallback (unusable stamps, a reboot) and the clock of the
# per-point `t` INSIDE a cloud (driver.flush_cloud; 0.7 ms over one 84 ms cloud).
DEVCLOCK_MIN_BASELINE_S = 5.0        # receive-time baseline before a rate is trusted
DEVCLOCK_WINDOW_S = 20.0             # rate fitted over the last this-many seconds of receive time
DEVCLOCK_EDGE_S = 1.0                # the least-delayed sample in the first / last second anchors the rate
DEVCLOCK_REFIT_S = 0.5               # re-estimate the rate this often (receive time)


class DeviceStampClock:
    """Device time from DataInfo.stamp, at real rate. add(stamp_s, recv_s, device_s_now) feeds one point packet with
    a usable stamp (device_s_now = the time the fallback clock gives this packet, so the switch is continuous) and
    returns True once the clock is ACTIVE. map(stamp_s) -> device time. The rate (stamp seconds per receive second)
    is the slope between the least-delayed sample of the window's first DEVCLOCK_EDGE_S and that of its last
    (one-way delay is one-sided, so the minimum is the honest anchor; a mean would follow stalls). Every refit
    re-bases the line at the current stamp, so the map stays continuous and monotone. A stamp going BACKWARDS
    (device reboot) resets the clock (the caller falls back to the cadence clock until a new baseline exists)."""

    def __init__(self, min_baseline_s=None, window_s=None):
        self.min_baseline_s = DEVCLOCK_MIN_BASELINE_S if min_baseline_s is None else float(min_baseline_s)
        self.window_s = DEVCLOCK_WINDOW_S if window_s is None else float(window_s)
        self.n_resets = 0
        self.reset()

    def reset(self):
        self.samples = deque()            # (recv_s, stamp_s)
        self.rate = None
        self.base_stamp = self.base_vt = None
        self.active = False
        self._last_fit_recv = None

    def _estimate(self, r0):
        first_end = self.samples[0][0] + DEVCLOCK_EDGE_S
        last_start = self.samples[-1][0] - DEVCLOCK_EDGE_S
        a = min((s for s in self.samples if s[0] <= first_end), key=lambda s: s[0] - s[1] / r0)
        b = min((s for s in self.samples if s[0] >= last_start), key=lambda s: s[0] - s[1] / r0)
        if b[0] - a[0] <= 0 or b[1] - a[1] <= 0:
            return None
        return (b[1] - a[1]) / (b[0] - a[0])

    def add(self, stamp_s, recv_s, device_s_now):
        if self.samples and stamp_s < self.samples[-1][1]:
            self.n_resets += 1
            self.reset()                  # device reboot: start a new baseline
        self.samples.append((recv_s, stamp_s))
        cutoff = recv_s - self.window_s
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        span = self.samples[-1][0] - self.samples[0][0]
        if span < self.min_baseline_s:
            return self.active
        if self._last_fit_recv is not None and recv_s - self._last_fit_recv < DEVCLOCK_REFIT_S:
            return self.active
        self._last_fit_recv = recv_s
        r0 = self.rate or (self.samples[-1][1] - self.samples[0][1]) / span
        r = self._estimate(r0) if r0 > 0 else None
        if r is None or r <= 0:
            return self.active
        if not self.active:
            self.base_stamp, self.base_vt, self.rate, self.active = stamp_s, device_s_now, r, True
        else:
            self.base_vt, self.base_stamp, self.rate = self.map(stamp_s), stamp_s, r
        return self.active

    def map(self, stamp_s):
        return self.base_vt + (stamp_s - self.base_stamp) / self.rate


class SeqOrderGuard:
    """accept(seq, recv_s) -> None (accept) or a drop reason: "duplicate" (seq equal to the last accepted one) or
    "late" (1..SEQ_MAX_LATE_PACKETS behind it, within SEQ_LATE_WINDOW_S). Only ACCEPTED packets move the guard."""

    def __init__(self, seq_modulus=None, max_late=SEQ_MAX_LATE_PACKETS, late_window_s=SEQ_LATE_WINDOW_S):
        self.m = int(seq_modulus) if seq_modulus else (1 << 32)
        self.max_late, self.late_window_s = int(max_late), float(late_window_s)
        self.last_seq = None
        self.last_recv_s = None

    def accept(self, seq, recv_s):
        seq = int(seq) % self.m
        if self.last_seq is not None:
            back = (self.last_seq - seq) % self.m
            recent = self.last_recv_s is not None and recv_s - self.last_recv_s < self.late_window_s
            if back == 0 and recent:
                return "duplicate"
            if 0 < back <= self.max_late and recent:
                return "late"
        self.last_seq, self.last_recv_s = seq, recv_s
        return None


def packet_span_s(rec):
    """This point packet's own declared duration (`point_num *
    time_increment`, both read from the packet itself) — the SAME quantity
    `point_packet_xyzit` uses for intra-cloud `t`, reused here as the device
    clock's tick size (both the virtual-clock reconstruction below and the
    driver's cumulative per-point offset) so the stamp basis and the
    per-point offsets stay mutually consistent."""
    n = int(rec["point_num"])
    n = 0 if n < 0 else min(n, MAX_POINTS_PER_PACKET)
    return n * float(rec["time_increment"])


def packet_period_s(rec):
    """The device's point-packet CADENCE: the packet's own `scan_period` (one
    vertical mirror revolution), when it is plausible, else its point span.

    Measured on the real L2 (2026-09-30, 42 801 packets, 200 s pcap):
    `scan_period` 4.635 ms == the capture inter-arrival 4.636 ms, while the
    point span (300 x `time_increment` 8 us) is only 2.474 ms = 192/360 of it
    (wide mode measures over 192 deg of each 360 deg mirror turn). Using the
    span as the tick ran the driver's device time at 0.53x real (stamp age up
    to 1 s, re-anchoring every second). Plausible = span <= scan_period <=
    4 x span: a 180/192 deg arc gives 1.9-2.0x; anything else (0 in old
    fixtures, a per-CLOUD figure) falls back to the span."""
    span = packet_span_s(rec)
    sp = float(rec["scan_period"])
    if span > 0 and sp == sp and span <= sp <= 4.0 * span:
        return sp
    return span


def _device_stamp_usable(sec, nsec, prev_device_time_s):
    """A DataInfo.stamp is trusted only if it is non-zero (the vendor notes'
    only stated guarantee is about `seq`, not `stamp`; an all-zero stamp is
    this candidate's and every synthetic fixture's "not populated" value) AND
    monotonic non-decreasing versus the last ACCEPTED device time — nothing
    in the vendor notes promises monotonicity either, so a backward jump is
    treated as unusable (falls back to the seq/span reconstruction) rather
    than trusted and used to detect a "device time jump" re-anchor; a
    forward jump big enough to matter is still caught, one layer up, by
    `L2TimeSync._update_offset`'s own jump detector."""
    if sec == 0 and nsec == 0:
        return False
    t = float(sec) + float(nsec) * 1e-9
    if prev_device_time_s is not None and t < prev_device_time_s - 1e-9:
        return False
    return True


class VirtualDeviceClock:
    """Reconstructs a monotone DEVICE time axis from `seq` (DataInfo.seq,
    "consecutively increasing" per the vendor notes) plus each packet's own
    declared span, for use when the packet's own DataInfo.stamp is not
    usable. A `seq` gap of `k` missing packets advances virtual time by
    `k * (the last known span)` so a lost packet never shifts the timing of
    packets that DID arrive — only the (never-observed) missing slice is
    skipped, not collapsed out of the timeline.

    `t` never resets or goes backwards: a `seq` delta this class cannot
    trust — either a genuine 32-bit counter wraparound (only plausible if
    `last_seq` was actually near the top of the range) or something else
    (a device reboot restarting `seq` from 0; a fixture that legitimately
    reuses a small `seq` range across many clouds, as this candidate's own
    conformance feed does — `seq` is NOT globally unique per this driver's
    contract, `t` is) — is NOT counted as loss (there is no reliable way to
    say how many packets that represents) and does NOT reset `t`; it simply
    advances by exactly one span, as if this packet immediately followed the
    last one. `last_advance_was_unreliable_seq` records that this happened,
    for `L2TimeSync` to fold into its OWN (real-receive-time-based) judgment
    of whether a re-anchor is warranted — `seq` alone is never trusted to
    force one; a genuine device reboot is expected to ALSO produce a real
    receive-time gap, which `L2TimeSync._update_offset` already detects."""

    def __init__(self, seq_modulus=None):
        self.seq_modulus = int(seq_modulus) if seq_modulus else None
        self.t = 0.0
        self.last_seq = None
        self.last_span = None
        self.n_lost_packets = 0
        self.last_advance_was_unreliable_seq = False

    def advance(self, seq, span_s):
        """seq: this packet's DataInfo.seq (any int, wrapped to u32). Always
        returns the new (monotonically increasing) virtual device time."""
        seq = int(seq) & 0xFFFFFFFF
        self.last_advance_was_unreliable_seq = False
        if self.seq_modulus and self.last_seq is not None:
            # A counter of known width (the real L2: 10 bits, wraps 1023 -> 0):
            # the gap is exact modulo its width, so a wrap is never "unreliable".
            gap = (seq - self.last_seq) % self.seq_modulus or 1
            if gap > self.seq_modulus - SEQ_MAX_LATE_PACKETS:
                # defence in depth (the driver's SeqOrderGuard drops these first): a SMALL BACKWARD step is a
                # late / reordered packet, never ~1023 lost packets (lio-l2 report 2026-10-07: +4.7 s stamps).
                # It belongs in the PAST: the clock does not advance and the reference seq does not move, so the
                # next in-order packet continues from the last accepted one.
                self.last_advance_was_unreliable_seq = True
                return self.t
            if gap > 1:
                self.n_lost_packets += gap - 1
                self.t += (gap - 1) * (self.last_span if self.last_span is not None else span_s)
            self.t += span_s
            self.last_seq, self.last_span = seq, span_s
            return self.t
        if self.last_seq is not None:
            raw_gap = (seq - self.last_seq) & 0xFFFFFFFF
            if raw_gap == 0:
                raw_gap = 1                       # duplicate/replayed seq: treat as one slot
            if raw_gap > _SEQ_RESET_GAP:
                near_wrap = self.last_seq >= (0xFFFFFFFF - _SEQ_NEAR_WRAP_MARGIN)
                if not near_wrap:
                    # `last_seq` was nowhere near the top of the 32-bit
                    # range, so this is NOT a genuine wraparound — do not
                    # fabricate a multi-billion-packet "loss" count.
                    self.last_advance_was_unreliable_seq = True
                    self.t += span_s
                    self.last_seq = seq
                    self.last_span = span_s
                    return self.t
                # else: plausible genuine wraparound near the boundary —
                # `raw_gap` is trustworthy, fall through to the normal path.
            missing = raw_gap - 1
            if missing > 0:
                self.n_lost_packets += missing
                self.t += missing * (self.last_span if self.last_span is not None else span_s)
        self.t += span_s
        self.last_seq = seq
        self.last_span = span_s
        return self.t


class L2TimeSync:
    """Owns the FULL device -> node timebase for one unitree_l2 driver
    process: reconstructs device time for LiDAR point packets, maps IMU
    packets through the SAME offset, and re-anchors on a real-time stall or a
    device-clock jump (a `seq` anomaly alone never forces one — see the
    module docstring; a device reboot is still caught, because it produces a
    real-time stall too, and is then logged with reason `device_reboot`).
    One instance per driver process; `driver.py` feeds it plain
    (rec, recv_time_s) pairs and stamps outgoing messages with what it
    returns.

    IMU: this candidate's DataInfo.stamp is never populated by the real
    device (unverified, vendor protocol notes) and the IMU frame
    carries no `point_num`/`time_increment` to reconstruct a virtual clock
    from, so when its device stamp is unusable the IMU is stamped at its own
    RAW receive time. That is deliberately "receive time minus the current
    offset's implied latency, then plus that same offset back" reduced to
    its algebraic identity (`recv - offset + offset == recv`): it goes
    through the same `node_time()` shape as the LiDAR path (offset is still
    the single source of truth whenever a real device stamp IS available —
    see the `imu_node_time` branch below) and, now that the LiDAR clock
    itself is kept within a bounded, re-anchored distance of real receive
    time (see the module docstring — never a runaway extrapolation), the two
    streams' published stamps both approximate the same node clock and their
    relative timing survives, which is the property imu_source.md requires.
    """

    def __init__(self, window_s=DEFAULT_OFFSET_WINDOW_S,
                 max_slew_s_per_s=DEFAULT_MAX_SLEW_S_PER_S,
                 stall_reset_s=DEFAULT_STALL_RESET_S, seq_modulus=None, devclock_min_baseline_s=None):
        self.window_s = window_s
        self.max_slew_s_per_s = max_slew_s_per_s
        self.stall_reset_s = stall_reset_s
        self._offset = None                 # recv_s - device_s, node-clock seconds
        # Monotonic-increasing deque of (recv_s, raw_offset_s): a standard
        # sliding-window-minimum structure, O(1) amortised per sample rather
        # than re-scanning the whole window (matters at 216 Hz over a
        # long-running session, and for the multi-minute unit tests below):
        # front = current window minimum; a new sample first evicts every
        # BACK entry whose raw_offset it dominates (>=), since those can
        # never again be the minimum once a smaller, more recent sample
        # exists; expired FRONT entries (older than `window_s`) are evicted
        # lazily, on the next sample.
        self._samples = deque()
        self._last_recv_s = None
        self._last_device_s = None
        self._vclock = VirtualDeviceClock(seq_modulus)
        # The device-stamp clock (DeviceStampClock, above): once its rate is established, point packets AND IMU
        # frames take their device time from DataInfo.stamp through ONE linear map. Old text, retained: "DataInfo.stamp
        # -> virtual-clock map (for IMU packets only) ... NEVER used as the point packets' clock ... vt = vt_ref + (dev
        # - dev_ref) / rate" -- that map re-anchored on EVERY point packet and carried the cadence clock's drift into
        # the IMU stamps (0.4-0.6 % of IMU frames came out non-increasing, 2026-10-07).
        self._dclock = DeviceStampClock(min_baseline_s=devclock_min_baseline_s)
        self._cad_shift = 0.0               # keeps the cadence fallback continuous with the last device-stamp time
        self._last_device_s = None
        self._last_from_dclock = False
        self._jitter = deque()              # (recv_s, recv - map(stamp)) of recent point packets, for the report
        self._recovery_deadline_s = None    # recv_s past which downward slew is blocked again
        self.n_reanchors = 0
        self.n_lost_packets = 0             # persists across re-anchors (unlike the vclock's own counter)
        self.reanchor_log = []              # [{"recv_s", "device_s", "reason"}], for driver logging

    def _reanchor(self, recv_s, device_s, reason):
        prev_offset = self._offset
        self._offset = recv_s - device_s
        self._samples = deque([(recv_s, self._offset)])
        # Recovery window: a FIXED real-time deadline (not "until offset
        # exactly matches target", which real, continuously-fluctuating
        # jitter may never satisfy bit-for-bit — see test (c)'s docstring
        # for the analytic derivation of the worst-case settle time), sized
        # to the jump THIS re-anchor just introduced. The very first ("init")
        # re-anchor has no prior offset to jump from, so its window is ~0:
        # ordinary operation starts in the safe (downward-blocked) mode
        # immediately, which is what keeps `clouds_disjoint_in_time` intact
        # for a run with no real stall.
        jump_s = abs(self._offset - prev_offset) if prev_offset is not None else 0.0
        self._recovery_deadline_s = recv_s + 1.5 * jump_s / _PER_SAMPLE_SLEW_FRACTION
        self.n_reanchors += 1
        self.reanchor_log.append({"recv_s": recv_s, "device_s": device_s, "reason": reason})

    def _update_offset(self, recv_s, device_s, span_hint_s=0.0, seq_unreliable=False):
        if self._offset is None:
            self._reanchor(recv_s, device_s, "init")
        else:
            gap_s = recv_s - self._last_recv_s if self._last_recv_s is not None else 0.0
            went_backwards = (self._last_device_s is not None
                               and device_s < self._last_device_s - 1e-9)
            raw_offset = recv_s - device_s
            jumped = abs(raw_offset - self._offset) > self.stall_reset_s
            stalled = gap_s > self.stall_reset_s
            if stalled or went_backwards or jumped:
                # `seq_unreliable` (VirtualDeviceClock: a seq delta too big
                # to trust as ordinary loss, and not a plausible 32-bit
                # wraparound either) is NEVER on its own grounds for a
                # re-anchor — this candidate's own conformance feed reuses a
                # small `seq` range across every cloud, which must NOT
                # re-anchor the offset on every cloud boundary (that broke
                # `clouds_disjoint_in_time`: see git history). It only
                # sharpens the LOGGED reason when a re-anchor is ALREADY
                # warranted on real-time grounds (`stalled`/`jumped`/
                # `went_backwards`) — e.g. a genuine device reboot, which
                # produces both a real gap and a seq restart.
                reason = ("device_reboot" if (stalled and seq_unreliable)
                          else "device_time_backwards" if went_backwards
                          else "device_time_jump" if jumped else "stall")
                self._reanchor(recv_s, device_s, reason)
            else:
                while self._samples and self._samples[-1][1] >= raw_offset:
                    self._samples.pop()
                self._samples.append((recv_s, raw_offset))
                cutoff = recv_s - self.window_s
                while self._samples and self._samples[0][0] < cutoff:
                    self._samples.popleft()
                target = self._samples[0][1]
                # Never publish a stamp in the future of the node clock by
                # more than one packet span: fold that bound into `target`
                # itself, as a TIGHTER ceiling, rather than a separate
                # unconditional post-hoc override — `future_ceiling` is
                # ALWAYS >= `target` in ordinary operation (raw_offset itself
                # already satisfies device_s + raw_offset == recv_s <=
                # recv_s + span_hint_s, and target <= raw_offset by
                # definition of the window minimum), so this only binds when
                # `self._offset` has drifted implausibly far above what the
                # CURRENT window would justify (exactly the "frozen too high"
                # case below) — never during ordinary recovering-branch
                # tracking (where it would otherwise force a disruptive
                # re-anchor instead of a smooth, bounded correction).
                future_ceiling = (recv_s + max(span_hint_s, 0.0)) - device_s
                effective_target = min(target, future_ceiling)
                # `clouds_disjoint_in_time` (lidar_source.md) has only a
                # ONE-`time_increment` margin (the per-point `t` construction
                # in `driver.py.flush_cloud`) against consecutive cloud
                # stamps getting CLOSER together than a full cloud span — and
                # `stamp[i+1] - stamp[i]` shrinks exactly when the offset
                # DECREASES between the two clouds' first packets (it never
                # shrinks when the offset increases). So: an UPWARD
                # correction (effective_target > offset) is always safe and
                # uses the ordinary bounded real-time slew; a DOWNWARD
                # correction is applied at that same fast (per-sample-
                # floored) rate whenever it is needed to honour the
                # future-stamp invariant (`future_ceiling < target`, however
                # rare) OR inside the post-re-anchor recovery window (the
                # case that NEEDS a fast correction for its own sake: a
                # stall's backlog draining in a near-zero-real-time burst,
                # `test_l2_protocol.py` test (c)); otherwise, further
                # downward nudges from ordinary jitter are deliberately
                # IGNORED rather than risking the disjointness margin for a
                # sub-millisecond latency-estimate improvement nobody needs
                # (`max_latency_s` is 0.5 s; this margin is ~15 us) — EITHER
                # way, the correction is always bounded by the same `step`,
                # never an instant, unconditional jump.
                if effective_target > self._offset:
                    step = self.max_slew_s_per_s * max(gap_s, 0.0)
                    self._offset = min(effective_target, self._offset + step)
                elif effective_target < self._offset and (
                        future_ceiling < target or recv_s <= self._recovery_deadline_s):
                    step = max(self.max_slew_s_per_s * max(gap_s, 0.0),
                               _PER_SAMPLE_SLEW_FRACTION * span_hint_s)
                    self._offset = max(effective_target, self._offset - step)
        self._last_recv_s = recv_s
        self._last_device_s = device_s

    @property
    def device_stamp_rate(self):
        """Stamp seconds per real second: the device-stamp clock's fitted rate once active (~0.5027 on the real L2),
        else a raw estimate over the samples seen so far (None before 1 s)."""
        if self._dclock.active:
            return self._dclock.rate
        sm = self._dclock.samples
        if len(sm) >= 2 and sm[-1][0] - sm[0][0] >= 1.0:
            return (sm[-1][1] - sm[0][1]) / (sm[-1][0] - sm[0][0])
        return None

    def point_packet_device_time(self, rec, recv_s):
        """-> device_time_s for this LiDAR point packet; updates the shared offset as a side effect.

        Device time = the DEVICE-STAMP clock once established (DeviceStampClock: one linear map of DataInfo.stamp,
        shared with the IMU), else the CADENCE clock (`VirtualDeviceClock`: seq + `packet_period_s`), kept
        continuous across every switch. Old text, retained: "The virtual clock ticks by the packet CADENCE ...
        DataInfo.stamp is not trusted as a clock (real L2: 0.5x rate) and only feeds the IMU map." -- the 0.5x rate
        is a scale, not a defect: the stamp is the LINEAR clock, the cadence one drifts (2026-10-07)."""
        seq = int(rec["info_seq"])
        span = packet_span_s(rec)
        period = packet_period_s(rec)
        raw_cad = self._vclock.advance(seq, period)
        seq_unreliable = self._vclock.last_advance_was_unreliable_seq
        self.n_lost_packets = self._vclock.n_lost_packets
        if self._last_from_dclock and self._last_device_s is not None:
            # while the device-stamp clock runs, the cadence clock SHADOWS it (re-based on every packet), so a switch
            # to the fallback (zero stamps, a reboot) on any packet continues from the last device-stamp time
            self._cad_shift = self._last_device_s + period - raw_cad
        cad = raw_cad + self._cad_shift
        sec, nsec = int(rec["info_sec"]), int(rec["info_nsec"])
        device_s, from_dclock = cad, False
        if sec or nsec:
            stamp = float(sec) + float(nsec) * 1e-9
            if self._dclock.add(stamp, recv_s, cad):
                device_s, from_dclock = self._dclock.map(stamp), True
                self._jitter.append((recv_s, recv_s - device_s))
                while self._jitter and self._jitter[0][0] < recv_s - 10.0:
                    self._jitter.popleft()
        self._last_device_s, self._last_from_dclock = device_s, from_dclock
        # CAUSALITY (certifier finding, 2026-09-17): a packet cannot arrive before
        # its LAST sample was taken, so the receive instant is matched to the END
        # of the packet's point span (not of its cadence slot), not its start.
        # The residual bias is the unknowable one-way transport latency, always late-side.
        self._update_offset(recv_s, device_s + span, span_hint_s=span, seq_unreliable=seq_unreliable)
        return device_s

    def imu_node_time(self, rec, recv_s):
        """-> node_time_s for this IMU packet: through the SAME device-stamp map as the point packets once it is
        established (one oscillator, so LiDAR/IMU relative timing is preserved and the stamp is monotone in the
        device stamp: no IMU sample is dropped for order), otherwise at its own raw receive time."""
        sec, nsec = int(rec["info_sec"]), int(rec["info_nsec"])
        if self._offset is not None and self._dclock.active and (sec or nsec):
            return self.node_time(self._dclock.map(float(sec) + float(nsec) * 1e-9))
        return recv_s

    def timebase_report(self):
        """For the driver's periodic log: which clock is in use, the fitted rate, resets, and the arrival jitter
        against the device-stamp map over the last 10 s (recv - map(stamp) minus its window minimum: p50/p99/max,
        ms). That jitter is the bound on how far a stamp can sit from a perfectly linear device clock."""
        out = {"clock": "device_stamp" if self._dclock.active else "cadence", "rate": self._dclock.rate,
               "devclock_resets": self._dclock.n_resets, "reanchors": self.n_reanchors}
        if len(self._jitter) >= 10:
            v = np.array([j for _t, j in self._jitter])
            v = (v - v.min()) * 1e3
            out["arrival_jitter_ms"] = {"p50": round(float(np.percentile(v, 50)), 3),
                                        "p99": round(float(np.percentile(v, 99)), 3), "max": round(float(v.max()), 3)}
        return out

    def node_time(self, device_s):
        if self._offset is None:
            raise RuntimeError("L2TimeSync.node_time called before any point packet observed")
        return device_s + self._offset


def pack_point_packet(seq, point_num, ranges_mm, intensities_u8, calib, angles, time_increment,
                       recv_stamp_sec_nsec=(0, 0), scan_period=0.0834, range_min=0.05, range_max=30.0):
    """Inverse of `point_packet_xyzit`'s framing (not its math — see
    conformance_feed.py for the geometry inversion): assemble ONE valid
    LidarPointDataPacket on the wire. `calib` = (a_axis, b_axis, theta_bias,
    alpha_bias, beta, xi, range_bias, range_scale). `angles` = (angle_min,
    angle_increment, h_start, h_step)."""
    n = len(ranges_mm)
    assert n <= MAX_POINTS_PER_PACKET
    ranges_full = np.zeros(MAX_POINTS_PER_PACKET, dtype="<u2")
    inten_full = np.zeros(MAX_POINTS_PER_PACKET, dtype="u1")
    ranges_full[:n] = ranges_mm
    inten_full[:n] = intensities_u8
    a_axis, b_axis, theta_bias, alpha_bias, beta, xi, range_bias, range_scale = calib
    angle_min, angle_increment, h_start, h_step = angles
    rec = np.zeros(1, dtype=POINT_DATA_DT)[0]
    rec["info_seq"] = seq & 0xFFFFFFFF
    rec["info_payload_size"] = POINT_DATA_LEN
    rec["info_sec"], rec["info_nsec"] = recv_stamp_sec_nsec
    rec["sys_rotation_period"] = 0
    rec["com_rotation_period"] = 0
    rec["dirty_index"] = 0.0
    rec["packet_lost_up"] = 0.0
    rec["packet_lost_down"] = 0.0
    rec["apd_temperature"] = 25.0
    rec["apd_voltage"] = 0.0
    rec["laser_voltage"] = 0.0
    rec["imu_temperature"] = 25.0
    rec["a_axis_dist"] = a_axis
    rec["b_axis_dist"] = b_axis
    rec["theta_angle_bias"] = theta_bias
    rec["alpha_angle_bias"] = alpha_bias
    rec["beta_angle"] = beta
    rec["xi_angle"] = xi
    rec["range_bias"] = range_bias
    rec["range_scale"] = range_scale
    rec["com_horizontal_angle_start"] = h_start
    rec["com_horizontal_angle_step"] = h_step
    rec["scan_period"] = scan_period
    rec["range_min"] = range_min
    rec["range_max"] = range_max
    rec["angle_min"] = angle_min
    rec["angle_increment"] = angle_increment
    rec["time_increment"] = time_increment
    rec["point_num"] = n
    rec["ranges"] = ranges_full
    rec["intensities"] = inten_full
    return _pack(PT_POINT, rec.tobytes())


def pack_imu_packet(seq, accel_native, gyro_native, quat_xyzw, recv_stamp_sec_nsec=(0, 0)):
    rec = np.zeros(1, dtype=IMU_DATA_DT)[0]
    rec["info_seq"] = seq & 0xFFFFFFFF
    rec["info_payload_size"] = IMU_DATA_LEN
    rec["info_sec"], rec["info_nsec"] = recv_stamp_sec_nsec
    rec["quaternion"] = quat_xyzw
    rec["angular_velocity"] = gyro_native
    rec["linear_acceleration"] = accel_native
    return _pack(PT_IMU, rec.tobytes())


def _pack(packet_type, data_bytes):
    packet_size = HEADER_LEN + len(data_bytes) + TAIL_LEN
    header = _HEADER_STRUCT.pack(FRAME_HEADER_MAGIC, packet_type, packet_size)
    # crc32 = zlib CRC-32 of the data bytes (the real unit's rule, see parse_packet).
    # msg_type_check: the real unit sends 210 in data packets and the vendor
    # library unrelated values in commands; nothing checks it, packet_type is a placeholder.
    tail = _TAIL_STRUCT.pack(zlib.crc32(data_bytes) & 0xFFFFFFFF, packet_type, b"\x00\x00", FRAME_TAIL_MAGIC)
    return header + data_bytes + tail

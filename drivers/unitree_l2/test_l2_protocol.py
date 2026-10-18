#!/usr/bin/env python3
"""Self-test for l2_protocol.py — pure functions, NO ROS, no socket, no
rclpy. Run: python3 drivers/unitree_l2/test_l2_protocol.py

Part 1 is the F2/C hard-fail check H7: malformed input (a truncated packet, a
wrong `packet_size`, a bad header/tail magic, an unsupported packet_type, a
wrong-length data section) must be counted and DROPPED, never guessed at or
turned into a published point — and the driver must never raise on it either
(a crash on garbage input is worse than a silent drop: it takes the whole
role down instead of just the one bad packet).

Part 2 (`test_timesync_*`) exercises `L2TimeSync`/`VirtualDeviceClock`, the
device -> node clock model driving `driver.py`'s published stamps
(the lidar contract's "Timestamp semantics",
the IMU contract's "ONE offset to both streams"): packet loss,
device clock drift, a real host stall + backlog burst (the reproduced
`stamp_in_node_timebase` conformance failure under host load), the LiDAR/IMU
shared-clock property, and a device `seq` reset. All synthetic, pure Python —
no socket, no rclpy, no real time.sleep: every "recv time" / "device time" is
a plain float computed directly from the scenario's own ground truth.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import l2_protocol as proto  # noqa: E402

n_checks = 0


def check(desc, cond):
    global n_checks
    n_checks += 1
    print(f"  [{'ok' if cond else 'FAIL'}] {desc}")
    assert cond, desc


class Sink:
    def __init__(self):
        self.lines = []

    def __call__(self, msg):
        self.lines.append(msg)


def make_good_point_packet(point_num=300, range_min=0.05, range_max=30.0):
    j = np.arange(point_num, dtype=np.float64)
    ranges_mm = np.clip(1000 + j, 1, 65535).astype(np.uint16)          # 1.0..1.3 m-ish
    inten = np.clip(j % 256, 0, 255).astype(np.uint8)
    calib = (0.0, 0.0, 0.01, -0.02, 0.05, -0.03, 2.0, 1.01e-3)          # non-trivial; range_scale raw->m like the real unit
    angles = (-0.7, 0.001, 0.2, 0.0002)
    return proto.pack_point_packet(
        seq=7, point_num=point_num, ranges_mm=ranges_mm, intensities_u8=inten,
        calib=calib, angles=angles, time_increment=1.5e-5,
        range_min=range_min, range_max=range_max)


def make_good_imu_packet():
    return proto.pack_imu_packet(
        seq=3, accel_native=np.array([0.0, 0.0, 1.0]),          # g
        gyro_native=np.array([0.0, 0.0, 5.729577951308232]),    # deg/s == 0.1 rad/s
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]))


def main():
    # ---- well-formed round trip -------------------------------------------
    good = make_good_point_packet()
    check("well-formed point packet has the expected length",
          len(good) == proto.POINT_PACKET_LEN)
    stats = proto.DropStats()
    parsed = proto.parse_packet(good, stats)
    check("well-formed point packet parses", parsed is not None)
    check("well-formed point packet drops nothing", stats.total() == 0)
    ptype, rec = parsed
    check("parsed packet_type is PT_POINT", ptype == proto.PT_POINT)
    xyzit = proto.point_packet_xyzit(rec, 0.05, 30.0)
    check("well-formed point packet yields points", xyzit is not None)
    x, y, z, intensity, t = xyzit
    check("all coordinates finite", np.all(np.isfinite(x)) and np.all(np.isfinite(y))
          and np.all(np.isfinite(z)))
    check("intensity in [0, 1]", float(intensity.min()) >= 0.0 and float(intensity.max()) <= 1.0)
    check("t starts at 0 for the first sample", abs(float(t[0])) < 1e-9)
    check("t is non-decreasing (time-ordered within a packet)", np.all(np.diff(t) >= 0))

    good_imu = make_good_imu_packet()
    check("well-formed IMU packet has the expected length", len(good_imu) == proto.IMU_PACKET_LEN)
    stats2 = proto.DropStats()
    parsed_imu = proto.parse_packet(good_imu, stats2)
    check("well-formed IMU packet parses", parsed_imu is not None)
    ptype2, rec2 = parsed_imu
    check("parsed packet_type is PT_IMU", ptype2 == proto.PT_IMU)
    accel, gyro, quat = proto.imu_native_to_si(rec2, "g", "deg_s")
    check("accel g->m/s^2 matches canonical fixture", abs(float(accel[2]) - 9.80665) < 1e-3)
    check("gyro deg/s->rad/s matches canonical fixture", abs(float(gyro[2]) - 0.1) < 1e-3)
    check("orientation quaternion passes through unchanged", np.allclose(quat, [0, 0, 0, 1]))

    # ---- H7: malformed input is REJECTED LOUDLY, never guessed -------------
    sink = Sink()
    stats3 = proto.DropStats(sink=sink)

    truncated = good[:20]
    check("truncated packet: parse returns None", proto.parse_packet(truncated, stats3) is None)

    very_short = good[:8]
    check("very short packet (< header+tail): parse returns None",
          proto.parse_packet(very_short, stats3) is None)

    bad_header = bytearray(good)
    bad_header[0] = 0x00
    check("bad header magic: parse returns None", proto.parse_packet(bytes(bad_header), stats3) is None)

    bad_tail = bytearray(good)
    bad_tail[-1] = 0x12
    check("bad tail magic: parse returns None", proto.parse_packet(bytes(bad_tail), stats3) is None)

    bad_size = bytearray(good)
    bad_size[8:12] = (999999).to_bytes(4, "little")
    check("packet_size lying about the actual length: parse returns None",
          proto.parse_packet(bytes(bad_size), stats3) is None)

    # packet_size IS consistent with the truncated length (an attacker/bug
    # that also patches packet_size), so the check must be on the TYPE's
    # expected data length, not just packet_size == len(buf).
    fake_consistent = good[:200]
    fixed = bytearray(fake_consistent)
    fixed[8:12] = (len(fake_consistent)).to_bytes(4, "little")
    fixed[-2:] = proto.FRAME_TAIL_MAGIC       # keep the tail magic valid at the new (wrong) offset
    check("packet_size matches actual length but not this type's expected size: parse returns None",
          proto.parse_packet(bytes(fixed), stats3) is None)

    unsupported = bytearray(good)
    unsupported[4:8] = (105).to_bytes(4, "little")     # LIDAR_VERSION_PACKET_TYPE
    check("unsupported (but structurally valid) packet_type: parse returns None",
          proto.parse_packet(bytes(unsupported), stats3) is None)

    bad_imu_len = good_imu[:-4]      # truncate the IMU packet by 4 bytes, patch packet_size to match
    bad_imu_len = bytearray(bad_imu_len)
    bad_imu_len[8:12] = (len(bad_imu_len)).to_bytes(4, "little")
    bad_imu_len[-2:] = proto.FRAME_TAIL_MAGIC
    check("IMU packet with wrong data length: parse returns None",
          proto.parse_packet(bytes(bad_imu_len), stats3) is None)

    check("every malformed case was counted (H7: count + drop, never silent)",
          stats3.total() == 8)
    check("the drop sink was actually told something (loud, not silent)", len(sink.lines) >= 8)

    # ---- fuzz: random-length garbage never raises ---------------------------
    rng = np.random.default_rng(0)
    stats4 = proto.DropStats()
    n_fuzz = 200
    for _ in range(n_fuzz):
        n = int(rng.integers(0, 2000))
        garbage = rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
        result = proto.parse_packet(garbage, stats4)         # must never raise
        if result is not None:
            # vanishingly unlikely with random bytes, but if the header magic
            # and packet_size happened to line up by chance, the payload must
            # still be a real, correctly-sized record — not a crash either way.
            proto.point_packet_xyzit(result[1], 0.0, 1e9) if result[0] == proto.PT_POINT else None
    check(f"{n_fuzz} random-length garbage packets: zero exceptions raised", True)
    check("fuzz run dropped everything it should have (no false accepts crashed anything)",
          stats4.total() <= n_fuzz)

    test_timesync_packet_loss(0.01, "1% packet loss")
    test_timesync_packet_loss(0.10, "10% packet loss")
    test_timesync_clock_drift()
    test_timesync_host_stall_recovery()
    test_timesync_imu_lidar_shared_clock_usable_stamp()
    test_timesync_imu_lidar_shared_clock_unusable_stamp()
    test_timesync_seq_reset_without_real_gap_does_not_reanchor()
    test_timesync_device_reboot_seq_reset_plus_real_gap_reanchors()
    test_real_pcap_timebase()
    test_split_datagram_coalesced()
    test_real_pcap_coalesced_imu_point()
    test_seq_order_guard()
    test_vclock_small_backward_step_is_not_loss()
    test_real_pcap_planted_swap()
    test_devclock_synthetic_drifting_cadence()
    test_devclock_reboot_falls_back_continuously()
    test_devclock_real_fixture_imu_monotone_one_map()

    print(f"\n{n_checks}/{n_checks} checks passed")


# =============================================================================
# Part 2: L2TimeSync / VirtualDeviceClock (device -> node timebase)
# =============================================================================

_PERIOD_S = 1.0 / 216.0          # the manifest's declared point-packet rate
_POINT_NUM = 300


def _point_rec(seq, point_num=_POINT_NUM, time_increment=_PERIOD_S / _POINT_NUM,
                info_sec=0, info_nsec=0):
    """A minimal parsed LidarPointData record built directly (bypassing the
    wire pack/parse round trip — already covered by Part 1's H7 tests) with
    only the fields `L2TimeSync` reads."""
    rec = np.zeros(1, dtype=proto.POINT_DATA_DT)[0]
    rec["info_seq"] = int(seq) & 0xFFFFFFFF
    rec["info_sec"] = int(info_sec)
    rec["info_nsec"] = int(info_nsec)
    rec["point_num"] = point_num
    rec["time_increment"] = time_increment
    return rec


def _imu_rec(seq, info_sec=0, info_nsec=0):
    rec = np.zeros(1, dtype=proto.IMU_DATA_DT)[0]
    rec["info_seq"] = int(seq) & 0xFFFFFFFF
    rec["info_sec"] = int(info_sec)
    rec["info_nsec"] = int(info_nsec)
    return rec


def _sim_point_stream(n_packets, period_s, rng, drop_prob=0.0, drift_ppm=0.0,
                       jitter_max_s=0.0, true_offset_s=1_000_000.0, start_seq=0):
    """Synthetic LiDAR point-packet stream: yields `(recv_s, rec,
    true_capture_s)` for every packet that "arrives" (a `drop_prob` fraction
    never does — real UDP loss, the receiver only ever sees a `seq` jump, no
    gap marker). `true_capture_s` is the GROUND TRUTH host-clock instant this
    packet's data was actually captured by the device (device time scaled by
    `1 + drift_ppm * 1e-6` plus the constant `true_offset_s`) — what a
    perfect clock sync would reproduce exactly, independent of any transport
    delay this particular packet happens to suffer. `recv_s` additionally
    carries up to `jitter_max_s` of one-sided queueing/transport jitter, as
    any real receive timestamp would."""
    time_increment = period_s / _POINT_NUM
    device_t = 0.0
    for i in range(n_packets):
        seq = start_seq + i
        this_device_t = device_t
        device_t += period_s
        if drop_prob and rng.random() < drop_prob:
            continue
        true_capture_s = true_offset_s + this_device_t * (1.0 + drift_ppm * 1e-6)
        jitter = rng.uniform(0.0, jitter_max_s) if jitter_max_s else 0.0
        recv_s = true_capture_s + jitter
        rec = _point_rec(seq, _POINT_NUM, time_increment)
        yield recv_s, rec, true_capture_s


def test_timesync_packet_loss(drop_prob, label):
    rng = np.random.default_rng(1000 + int(round(drop_prob * 10000)))
    n_packets = int(600 * 216)          # 10 simulated minutes at the declared 216 Hz
    jitter_max_s = 1e-3                  # 1 ms of one-sided receive jitter
    true_offset_s = 1_000_000.0
    warmup_s = 5.0                       # let the min-filter converge before judging error
    sync = proto.L2TimeSync()
    last_node_s = None
    monotonic_ok = True
    max_err_s = 0.0
    n_seen = 0
    for recv_s, rec, true_capture_s in _sim_point_stream(
            n_packets, _PERIOD_S, rng, drop_prob=drop_prob,
            jitter_max_s=jitter_max_s, true_offset_s=true_offset_s):
        device_s = sync.point_packet_device_time(rec, recv_s)
        node_s = sync.node_time(device_s)
        if last_node_s is not None and node_s <= last_node_s:
            monotonic_ok = False
        last_node_s = node_s
        n_seen += 1
        if recv_s - true_offset_s > warmup_s:
            max_err_s = max(max_err_s, abs(node_s - true_capture_s))
    check(f"{label}: {n_seen}/{n_packets} packets delivered", 0 < n_seen < n_packets)
    check(f"{label}: published stamps strictly increase across the whole stream",
          monotonic_ok)
    # Bound: a lost packet never shifts LATER packets' device time
    # (VirtualDeviceClock bridges the gap using the last known span), so once
    # the min-filter has converged (after `warmup_s`) the only remaining
    # error sources are (1) the ~one packet span of granularity in when a
    # "floor" (near-zero-jitter) sample last entered the window, and (2) the
    # injected jitter ceiling itself. Neither depends on the loss rate.
    bound_s = _PERIOD_S + jitter_max_s + 1e-3
    check(f"{label}: steady-state stamp error <= {bound_s * 1e3:.2f} ms "
          f"(observed {max_err_s * 1e3:.4f} ms)", max_err_s <= bound_s)


def test_timesync_clock_drift():
    rng = np.random.default_rng(99)
    n_packets = int(1200 * 216)          # 20 simulated minutes
    jitter_max_s = 1e-3
    drift_ppm = 300.0
    true_offset_s = 1_000_000.0
    warmup_s = 5.0
    sync = proto.L2TimeSync()
    last_node_s = None
    monotonic_ok = True
    max_err_s = 0.0
    for recv_s, rec, true_capture_s in _sim_point_stream(
            n_packets, _PERIOD_S, rng, drift_ppm=drift_ppm,
            jitter_max_s=jitter_max_s, true_offset_s=true_offset_s):
        device_s = sync.point_packet_device_time(rec, recv_s)
        node_s = sync.node_time(device_s)
        if last_node_s is not None and node_s <= last_node_s:
            monotonic_ok = False
        last_node_s = node_s
        if recv_s - true_offset_s > warmup_s:
            max_err_s = max(max_err_s, abs(node_s - true_capture_s))
    check("300 ppm device clock, 20 simulated minutes: "
          "published stamps strictly increase throughout", monotonic_ok)
    # Bound: the real-time slew budget (0.05 s of offset correction per
    # second of elapsed real time) is ~165x the drift rate (300 ppm =
    # 0.0003 s/s), so the offset tracks the slowly-growing true offset with
    # only the min-filter's own window lag (<= window_s * drift_ppm*1e-6,
    # here 2 s * 3e-4 = 0.6 ms) rather than falling behind — total drift over
    # the whole 20 min run is 1200 s * 300e-6 = 0.36 s, and this bound is
    # nowhere near that, which is the point: the slew tracks continuously,
    # it does not accumulate the drift and correct in one lump.
    bound_s = _PERIOD_S + jitter_max_s + 5e-3
    check(f"300 ppm device clock, 20 simulated minutes: stamp error bounded "
          f"(observed {max_err_s * 1e3:.4f} ms, bound {bound_s * 1e3:.2f} ms)",
          max_err_s <= bound_s)


def test_timesync_host_stall_recovery():
    """Reproduces the real conformance failure: a host stall delays a whole
    backlog of packets, which then drain in a near-instantaneous burst. The
    naive fix (reanchor to the very first post-stall sample) bakes the
    ENTIRE stall duration into the offset as if it were the true one-way
    latency; `_update_offset`'s per-sample slew floor (see its docstring) is
    what lets that inflated offset actually drain back down using the
    burst's own decreasing raw-offset samples, rather than being stuck for
    ~40 s of subsequent real-time-gated slewing (or forever, as in the
    original bug)."""
    rng = np.random.default_rng(7)
    true_offset_s = 1_000_000.0
    jitter_max_s = 1e-3
    sync = proto.L2TimeSync()

    last_node_s = [None]
    monotonic_ok = [True]
    seq_box = [0]
    device_t_box = [0.0]

    def feed(recv_s):
        this_t = device_t_box[0]
        device_t_box[0] += _PERIOD_S
        rec = _point_rec(seq_box[0])
        seq_box[0] += 1
        device_s = sync.point_packet_device_time(rec, recv_s)
        node_s = sync.node_time(device_s)
        if last_node_s[0] is not None and node_s <= last_node_s[0]:
            monotonic_ok[0] = False
        last_node_s[0] = node_s
        return node_s, this_t

    # Phase 1: 3 s of ordinary delivery so the offset is well converged.
    n_warmup = int(3.0 / _PERIOD_S)
    for _ in range(n_warmup):
        this_t = device_t_box[0]
        jitter = rng.uniform(0.0, jitter_max_s)
        feed(true_offset_s + this_t + jitter)
    n_reanchors_before = sync.n_reanchors

    # Phase 2: the DEVICE keeps ticking through a 2 s host stall (nothing
    # delivered), then the backlog arrives in one burst — each burst
    # packet's OWN recv_s only microseconds apart (a real kernel backlog
    # drain), device_s keeps its normal per-packet spacing throughout.
    stall_s = 2.0
    n_stalled = int(stall_s / _PERIOD_S)
    device_t_before_stall = device_t_box[0]
    burst_recv_s0 = true_offset_s + device_t_before_stall + stall_s + 1e-3
    burst_errs = []
    for k in range(n_stalled):
        this_recv_s = burst_recv_s0 + k * 1e-6
        node_s, this_t = feed(this_recv_s)
        true_capture_s = true_offset_s + this_t
        burst_errs.append(abs(node_s - true_capture_s))

    check("2 s host stall + burst: a re-anchor was triggered by the stall",
          sync.n_reanchors == n_reanchors_before + 1)
    check("2 s host stall + burst: published stamps strictly increase "
          "across the whole stall + burst", monotonic_ok[0])

    # Phase 3: normal delivery resumes. Worst-case analytic settle time (if
    # NONE of the correction happened during the burst's own near-zero-
    # real-time span, only via the per-sample floor during ordinary-cadence
    # packets afterward) is stall_s / _PER_SAMPLE_SLEW_FRACTION; allow 2x
    # that as margin, then judge convergence only over the FINAL second (a
    # clean "already settled" window) — this is the property the task asks
    # for: recovery, not permanent lag (the reproduced failure was ~1.65 s
    # of age that NEVER recovered).
    settle_s = 2.0 * stall_s / proto._PER_SAMPLE_SLEW_FRACTION
    n_settle = int(settle_s / _PERIOD_S)
    n_tail = int(1.0 / _PERIOD_S)
    tail_errs = []
    for i in range(n_settle):
        this_t = device_t_box[0]
        jitter = rng.uniform(0.0, jitter_max_s)
        node_s, _ = feed(true_offset_s + this_t + jitter)
        if i >= n_settle - n_tail:
            tail_errs.append(abs(node_s - (true_offset_s + this_t)))

    bound_s = _PERIOD_S + jitter_max_s + 1e-3
    check(f"2 s host stall + burst: error right after the burst is large "
          f"(observed {max(burst_errs) * 1e3:.1f} ms) — the failure mode this "
          "reproduces", max(burst_errs) > 100 * bound_s)
    check(f"2 s host stall + burst: recovers to the steady-state bound "
          f"<= {bound_s * 1e3:.2f} ms within {settle_s:.1f} s "
          f"(observed {max(tail_errs) * 1e3:.4f} ms over the final 1 s)",
          max(tail_errs) <= bound_s)


def test_timesync_imu_lidar_shared_clock_usable_stamp():
    """LiDAR and IMU packets generated from ONE device clock (zero jitter, so
    the ground truth is exact): with a USABLE IMU DataInfo.stamp, both
    streams go through the identical `device_time + offset` path, so their
    node-time difference matches their device-time difference to floating-
    point precision."""
    true_offset_s = 500_000.0
    sync = proto.L2TimeSync()
    device_t = 0.0
    lidar_node = lidar_device = None
    for seq in range(20):
        this_t = device_t
        device_t += _PERIOD_S
        recv_s = true_offset_s + this_t
        rec = _point_rec(seq)
        device_s = sync.point_packet_device_time(rec, recv_s)
        lidar_node = sync.node_time(device_s)
        lidar_device = device_s

    imu_device_t = lidar_device + 0.0123
    imu_sec = int(imu_device_t)
    imu_nsec = int(round((imu_device_t - imu_sec) * 1e9))
    imu_rec = _imu_rec(0, info_sec=imu_sec, info_nsec=imu_nsec)
    # Ground-truth recv time for the IMU sample: since this scenario is ONE
    # noise-free device clock, its true capture instant is exactly 0.0123 s
    # later than `lidar_device`'s — expressed relative to `lidar_node` (the
    # already-established-correct node time for `lidar_device`) rather than
    # re-deriving it from `true_offset_s` directly, which would silently
    # assume `lidar_device` is on a zero-based "seconds since the first
    # packet" axis — it is not: `VirtualDeviceClock` reports a packet's
    # device time as of the END of that packet's own span, one packet ahead
    # of that convention. `node_time()` already absorbs that internal
    # bookkeeping choice; re-deriving ground truth from it directly would
    # silently reintroduce it as a spurious one-packet-span test error.
    imu_recv_s = lidar_node + 0.0123
    imu_node = sync.imu_node_time(imu_rec, imu_recv_s)

    node_diff = lidar_node - imu_node
    device_diff = lidar_device - imu_device_t
    check("IMU usable device stamp: LiDAR/IMU node-time difference matches "
          f"device-time difference within 1e-6 s (node_diff={node_diff:.9f}, "
          f"device_diff={device_diff:.9f})", abs(node_diff - device_diff) < 1e-6)


def test_timesync_imu_lidar_shared_clock_unusable_stamp():
    """Same property, but with the IMU's DataInfo.stamp UNUSABLE (all zero —
    this candidate's real and synthetic-conformance reality today): the IMU
    is stamped at its own raw receive time (module docstring). With zero
    jitter, receive time IS the true capture instant, so the property still
    holds to floating-point precision."""
    true_offset_s = 700_000.0
    sync = proto.L2TimeSync()
    device_t = 0.0
    lidar_node = lidar_device = None
    for seq in range(20):
        this_t = device_t
        device_t += _PERIOD_S
        recv_s = true_offset_s + this_t
        rec = _point_rec(seq)
        device_s = sync.point_packet_device_time(rec, recv_s)
        lidar_node = sync.node_time(device_s)
        lidar_device = device_s

    imu_device_t = lidar_device + 0.0071
    # See the "usable device stamp" test above for why the ground truth is
    # expressed relative to `lidar_node`, not `true_offset_s + imu_device_t`
    # directly.
    imu_recv_s = lidar_node + 0.0071     # zero jitter: recv == true capture
    imu_rec = _imu_rec(0, info_sec=0, info_nsec=0)  # unusable (all-zero) device stamp
    imu_node = sync.imu_node_time(imu_rec, imu_recv_s)

    check("IMU unusable device stamp: node time is the documented raw-"
          "receive passthrough", imu_node == imu_recv_s)
    node_diff = lidar_node - imu_node
    device_diff = lidar_device - imu_device_t
    check("IMU unusable device stamp: LiDAR/IMU node-time difference still "
          f"matches device-time difference within 1e-6 s (node_diff={node_diff:.9f}, "
          f"device_diff={device_diff:.9f})", abs(node_diff - device_diff) < 1e-6)


def test_timesync_seq_reset_without_real_gap_does_not_reanchor():
    """A `seq` restart on its own — no accompanying real-time gap — must NOT
    force a re-anchor: this candidate's own conformance feed legitimately
    reuses a small `seq` range (0..cloud_scan_num-1) across every cloud for
    the whole session (it replays one canonical cloud's packets forever), so
    treating that as a device reboot re-anchored `clouds_disjoint_in_time`'s
    offset on almost every cloud boundary — the real regression this test
    guards against. `t` (virtual device time) must still just keep advancing
    smoothly, one span at a time, across the `seq` restart."""
    true_offset_s = 42.0
    sync = proto.L2TimeSync()
    device_t = 0.0
    last_device_s = None
    for seq in range(20):
        this_t = device_t
        device_t += _PERIOD_S
        rec = _point_rec(seq)
        last_device_s = sync.point_packet_device_time(rec, true_offset_s + this_t)
    n_before = sync.n_reanchors

    # `seq` restarts at 0 (like every cloud boundary in the real conformance
    # feed), with the NEXT packet arriving at ordinary cadence (no real gap).
    recv_s = true_offset_s + device_t
    rec = _point_rec(0)
    device_s = sync.point_packet_device_time(rec, recv_s)

    check("seq restart with NO real-time gap: no re-anchor",
          sync.n_reanchors == n_before)
    check("seq restart with NO real-time gap: virtual device time keeps "
          f"advancing smoothly by one span (last={last_device_s:.6f}, "
          f"new={device_s:.6f})", abs(device_s - (last_device_s + _PERIOD_S)) < 1e-9)

    last_node = sync.node_time(device_s)
    device_t2 = device_s
    monotonic_ok = True
    for i in range(1, 20):
        this_t = device_t2
        device_t2 += _PERIOD_S
        recv_s2 = recv_s + i * _PERIOD_S
        rec2 = _point_rec(i)
        d2 = sync.point_packet_device_time(rec2, recv_s2)
        n2 = sync.node_time(d2)
        if n2 <= last_node:
            monotonic_ok = False
        last_node = n2
    check("seq restart with NO real-time gap: published stamps keep "
          "strictly increasing afterward", monotonic_ok)


def test_timesync_device_reboot_seq_reset_plus_real_gap_reanchors():
    """A GENUINE device reboot plausibly restarts `seq` from 0 AND produces
    real downtime (a real-time gap) — that combination DOES force a
    re-anchor, logged with reason `device_reboot` so a human reading
    `reanchor_log` can tell it apart from an ordinary host stall."""
    true_offset_s = 42.0
    sync = proto.L2TimeSync()
    device_t = 0.0
    for seq in range(50):
        this_t = device_t
        device_t += _PERIOD_S
        rec = _point_rec(seq)
        sync.point_packet_device_time(rec, true_offset_s + this_t)
    n_before = sync.n_reanchors

    # Device reboot: `seq` drops back to 0 AND a real ~1.5 s gap (the device
    # was actually down) separates it from the last pre-reboot packet.
    recv_s = true_offset_s + device_t + 1.5
    rec = _point_rec(0)
    device_s = sync.point_packet_device_time(rec, recv_s)
    node_s = sync.node_time(device_s)

    check("device reboot (seq reset + real gap): a re-anchor was triggered",
          sync.n_reanchors == n_before + 1)
    check("device reboot (seq reset + real gap): the re-anchor reason is "
          "recorded as device_reboot",
          sync.reanchor_log[-1]["reason"] == "device_reboot")
    check("device reboot (seq reset + real gap): recovers to a sane node "
          f"time immediately (not stuck on the pre-reboot offset): "
          f"node_s - recv_s = {node_s - recv_s:.6f} s", abs(node_s - recv_s) < 1.0)

    last_node = node_s
    device_t2 = device_s
    monotonic_ok = True
    for i in range(1, 20):
        this_t = device_t2
        device_t2 += _PERIOD_S
        recv_s2 = recv_s + i * _PERIOD_S
        rec2 = _point_rec(i)
        d2 = sync.point_packet_device_time(rec2, recv_s2)
        n2 = sync.node_time(d2)
        if n2 <= last_node:
            monotonic_ok = False
        last_node = n2
    check("device reboot (seq reset + real gap): published stamps keep "
          "strictly increasing after recovery (no permanent lockup)", monotonic_ok)



def test_real_pcap_timebase():
    """REAL L2 data (5 s slice of the 2026-09-30 bench capture, test_fixtures/
    l2_real_wide_5s.pcap, 1 068 point packets, receive time = the capture's own
    kernel timestamps). The live driver published clouds 44.5 ms apart for 84 ms
    of data and re-anchored about once a second (stamp age up to 1.035 s),
    because its device clock ticked by the 2.47 ms point span and then trusted
    DataInfo.stamp, which runs at 0.5x real on this unit. Every property below
    failed on that code."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "bench"))
    import l2_pcap
    pkts = []
    for t, buf in l2_pcap.read_pcap(os.path.join(here, "test_fixtures", "l2_real_wide_5s.pcap")):
        r = proto.parse_packet(buf, proto.DropStats())
        if r is not None and r[0] == proto.PT_POINT:
            pkts.append((t, r[1]))
    check(f"real pcap: point packets parsed, every CRC valid ({len(pkts)})", len(pkts) > 1000)
    raw = next(l2_pcap.read_pcap(os.path.join(here, "test_fixtures", "l2_real_wide_5s.pcap")))[1]
    bad = bytearray(raw)
    bad[proto.HEADER_LEN + 200] ^= 0x01                   # one flipped bit inside the ranges
    st = proto.DropStats()
    check("real packet with one flipped payload bit is dropped as crc_mismatch",
          proto.parse_packet(bytes(bad), st) is None and st.counts.get("crc_mismatch") == 1)
    period = proto.packet_period_s(pkts[0][1])
    span = proto.packet_span_s(pkts[0][1])
    check(f"real pcap: packet cadence = scan_period ({period*1e3:.3f} ms), not the "
          f"point span ({span*1e3:.3f} ms)", abs(period - 0.004635) < 5e-5 and span < 0.6 * period)
    sync = proto.L2TimeSync(seq_modulus=1024)
    stamps, ages = [], []
    for k in range(0, len(pkts) - 18, 18):              # one cloud = 18 packets (cloud_scan_num)
        first = None
        for t, rec in pkts[k:k + 18]:
            d = sync.point_packet_device_time(rec, t)
            if first is None:
                first = sync.node_time(d)
        stamps.append(first)
        ages.append(pkts[k + 17][0] - first)             # age when the cloud's last packet arrived
    dst = np.diff(stamps)
    drecv = np.diff([pkts[k][0] for k in range(0, len(pkts) - 18, 18)])
    ratio = float(np.median(dst) / np.median(drecv))
    check(f"real pcap: stamp cadence / receive cadence = {ratio:.4f} (the old code: 0.53)",
          abs(ratio - 1.0) <= 0.02)
    check(f"real pcap: max cloud stamp age {max(ages)*1e3:.1f} ms < 0.5 s max_latency_s "
          f"(the old code: 1 035 ms)", max(ages) < 0.5)
    check(f"real pcap: no re-anchor after init ({sync.n_reanchors} total)", sync.n_reanchors == 1)
    check(f"real pcap: the 10-bit seq wraps are not losses (lost={sync.n_lost_packets})",
          sync.n_lost_packets == 0)
    rate = sync.device_stamp_rate
    check(f"real pcap: DataInfo.stamp rate vs real time measured {rate:.4f} (this unit: ~0.50; "
          "reported, and never used as the point clock)", rate is not None and 0.45 < rate < 0.55)


def test_split_datagram_coalesced():
    """The real L2 (IMU on) sends an IMU frame and a point frame back to back in ONE datagram (2026-10-06,
    window 2: 187 of 8 331 datagrams in 20 s, 1 124 B = 80 + 1 044). split_datagram returns the frames only
    when they tile the datagram exactly; anything else comes back whole and is rejected as before."""
    imu, pt = make_good_imu_packet(), make_good_point_packet()
    for name, dgram, want in (("IMU+point", imu + pt, (proto.PT_IMU, proto.PT_POINT)),
                              ("point+IMU", pt + imu, (proto.PT_POINT, proto.PT_IMU))):
        sink = Sink()
        st = proto.DropStats(sink=sink)
        frames = proto.split_datagram(dgram, st)
        parsed = [proto.parse_packet(f, st) for f in frames]
        check(f"{name}: 2 frames, both parse, types {want}",
              len(frames) == 2 and all(r is not None for r in parsed) and tuple(r[0] for r in parsed) == want)
        check(f"{name}: nothing dropped, 1 coalesced datagram counted apart from drops (and logged)",
              st.total() == 0 and st.events.get("coalesced_datagram") == 1
              and any("coalesced_datagram" in line for line in sink.lines))
    st = proto.DropStats()
    check("a single frame comes back as itself, not counted as coalesced",
          proto.split_datagram(pt, st) == [pt] and st.events == {})

    # a valid frame followed by a CUT-OFF frame: never salvaged, rejected whole exactly as before
    cut = pt + imu[:50]
    st = proto.DropStats()
    frames = proto.split_datagram(cut, st)
    check("valid frame + cut-off frame: datagram returned whole", frames == [cut])
    check("... and parse drops it as 1 packet_size_mismatch_actual, nothing coalesced",
          proto.parse_packet(frames[0], st) is None and st.counts == {"packet_size_mismatch_actual": 1}
          and st.events == {})

    # tiled frames, one with a bad CRC: the good one parses, the bad one is dropped as crc_mismatch
    bad_imu = bytearray(imu)
    bad_imu[proto.HEADER_LEN + 20] ^= 0x01
    st = proto.DropStats()
    parsed = [proto.parse_packet(f, st) for f in proto.split_datagram(bytes(bad_imu) + pt, st)]
    check("tiled IMU(bad CRC)+point: 1 parsed (point) + 1 crc_mismatch",
          [r[0] for r in parsed if r is not None] == [proto.PT_POINT] and st.counts == {"crc_mismatch": 1})

    # garbage after a valid frame: whole
    for junk in (b"\x01\x02\x03", bytes(40), proto.FRAME_HEADER_MAGIC + bytes(20)):
        st = proto.DropStats()
        d = pt + junk
        frames = proto.split_datagram(d, st)
        check(f"{len(junk)} B of garbage after a valid frame: returned whole, dropped by parse",
              frames == [d] and proto.parse_packet(frames[0], st) is None and st.total() == 1 and st.events == {})
    # a second frame whose declared size runs past the datagram: whole
    lie = bytearray(imu)
    lie[8:12] = (len(imu) + 4).to_bytes(4, "little")
    st = proto.DropStats()
    check("second frame declaring a size past the datagram end: returned whole",
          proto.split_datagram(pt + bytes(lie), st) == [pt + bytes(lie)] and st.events == {})
    rng = np.random.default_rng(1)
    for _ in range(200):
        g = rng.integers(0, 256, size=int(rng.integers(0, 2500)), dtype=np.uint8).tobytes()
        for f in proto.split_datagram(g, proto.DropStats()):       # must never raise
            proto.parse_packet(f, proto.DropStats())
    check("200 random garbage datagrams through split_datagram + parse: zero exceptions", True)


def test_real_pcap_coalesced_imu_point():
    """REAL L2 data with its IMU on: test_fixtures/l2_real_imu_coalesced_3s.pcap = seconds 5-8 of
    window2/imu_check_20s.pcap (2026-10-06, provenance in test_fixtures/README.md). It contains datagrams
    carrying IMU + point frames together; before split_datagram the driver dropped each one as
    packet_size_mismatch_actual (both frames lost)."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "bench"))
    import l2_pcap
    path = os.path.join(here, "test_fixtures", "l2_real_imu_coalesced_3s.pcap")
    dgrams = [b for _t, b in l2_pcap.read_pcap(path)]
    old = proto.DropStats()
    for b in dgrams:
        proto.parse_packet(b, old)
    n_old = old.counts.get("packet_size_mismatch_actual", 0)
    check(f"real IMU pcap: the fixture plants the defect ({n_old} datagrams the old path drops whole)", n_old > 0)

    # independent split: walk each datagram by the KNOWN per-type length, never its size field
    known = {proto.PT_IMU: proto.IMU_PACKET_LEN, proto.PT_POINT: proto.POINT_PACKET_LEN}
    ind = {proto.PT_IMU: 0, proto.PT_POINT: 0}
    n_multi = 0
    for b in dgrams:
        off, k = 0, 0
        while off < len(b):
            ptype = int.from_bytes(b[off + 4:off + 8], "little")
            off += known[ptype]
            ind[ptype] += 1
            k += 1
        assert off == len(b), "independent split did not tile a datagram"
        n_multi += k > 1

    st = proto.DropStats()
    got = {proto.PT_IMU: 0, proto.PT_POINT: 0}
    for b in dgrams:
        for f in proto.split_datagram(b, st):
            r = proto.parse_packet(f, st)
            if r is not None:
                got[r[0]] += 1
    coal = st.events.get("coalesced_datagram", 0)
    check(f"real IMU pcap: {coal} coalesced datagrams counted (independent count {n_multi}), > 0",
          coal > 0 and coal == n_multi == n_old)
    check(f"real IMU pcap: 0 packet_size_mismatch_actual and no drops at all ({st.counts})", st.total() == 0)
    check(f"real IMU pcap: parsed IMU {got[proto.PT_IMU]} / point {got[proto.PT_POINT]} frames == the "
          f"independent byte split ({ind[proto.PT_IMU]} / {ind[proto.PT_POINT]})", got == ind)



def test_seq_order_guard():
    """SeqOrderGuard (lio-l2 report 2026-10-07): late / duplicated point packets are dropped, never read as loss."""
    g = proto.SeqOrderGuard(1024)
    check("guard: in-order packets accepted", all(g.accept(s, 0.001 * s) is None for s in range(1000, 1024)))
    check("guard: the 10-bit wrap 1023 -> 0 is accepted", g.accept(0, 0.0241) is None and g.accept(1, 0.0242) is None)
    check("guard: a duplicate of the last seq -> 'duplicate'", g.accept(1, 0.0243) == "duplicate")
    g.accept(3, 0.0250)
    check("guard: a packet behind the last accepted one (2 after 3) -> 'late'", g.accept(2, 0.0251) == "late")
    check("guard: the next in-order packet is accepted after a late one", g.accept(4, 0.0252) is None)
    check("guard: a backward step of 65 (> max_late) is NOT judged late", g.accept((4 - 65) % 1024, 0.0253) is None)
    g2 = proto.SeqOrderGuard(1024)
    g2.accept(500, 0.0)
    check("guard: a small backward step after > 1 s of silence is a resync (accepted)", g2.accept(499, 1.5) is None)
    g3 = proto.SeqOrderGuard(None)          # the conformance feed: 32-bit modulus, its counter wraps 1023 -> 0
    g3.accept(1023, 0.0)
    check("guard (32-bit modulus): the feed's own 1023 -> 0 wrap is accepted", g3.accept(0, 0.005) is None)


def test_vclock_small_backward_step_is_not_loss():
    vc = proto.VirtualDeviceClock(1024)
    span = 0.004635
    ts = [vc.advance(s, span) for s in (10, 12, 11, 13)]
    check(f"vclock: seqs 10,12,11,13 count 1 lost packet (the real 10->12 gap), not ~1023 (got {vc.n_lost_packets})",
          vc.n_lost_packets == 1)
    check(f"vclock: ... and time advances by a few spans, not ~4.7 s (total {ts[-1] - ts[0]:.4f} s)",
          ts[-1] - ts[0] < 10 * span)


def test_real_pcap_planted_swap():
    """REAL L2 data (test_fixtures/l2_real_imu_coalesced_3s.pcap) with two adjacent POINT packets swapped (the
    lio-l2 n_lost=1024 signature), run through the driver's own timing path (guard -> L2TimeSync). Stamps must stay
    within the real-time bound; with the defence disabled the swap must read as ~1024 lost packets again."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "bench"))
    import l2_pcap
    pts = []
    for t, b in l2_pcap.read_pcap(os.path.join(here, "test_fixtures", "l2_real_imu_coalesced_3s.pcap")):
        for f in proto.split_datagram(b, None):
            r = proto.parse_packet(f, proto.DropStats())
            if r is not None and r[0] == proto.PT_POINT:
                pts.append((t, r[1]))
    k = len(pts) // 2
    pts[k], pts[k + 1] = (pts[k][0], pts[k + 1][1]), (pts[k + 1][0], pts[k][1])   # swap the RECORDS, keep each slot's receive time

    def run(guarded, max_late):
        old = proto.SEQ_MAX_LATE_PACKETS
        proto.SEQ_MAX_LATE_PACKETS = max_late
        try:
            sync, guard = proto.L2TimeSync(seq_modulus=1024), proto.SeqOrderGuard(1024)
            worst, n_late = 0.0, 0
            for t, rec in pts:
                if guarded and guard.accept(int(rec["info_seq"]), t) is not None:
                    n_late += 1
                    continue
                d = sync.point_packet_device_time(rec, t)
                worst = max(worst, sync.node_time(d) - t)
            return worst, n_late, sync.n_lost_packets
        finally:
            proto.SEQ_MAX_LATE_PACKETS = old

    w, n_late, lost = run(True, proto.SEQ_MAX_LATE_PACKETS)
    check(f"real fixture + planted swap, guarded: 1 late packet dropped (got {n_late}), stamps never > 0.1 s ahead of "
          f"receive (worst {w * 1e3:.1f} ms), lost {lost}", n_late == 1 and w < 0.1 and lost < 5)
    w2, _, lost2 = run(False, proto.SEQ_MAX_LATE_PACKETS)
    check(f"... unguarded but with the vclock defence: still sane (worst {w2 * 1e3:.1f} ms, lost {lost2})",
          w2 < 0.1 and lost2 < 5)
    w3, _, lost3 = run(False, 0)
    check(f"NEGATIVE: defence disabled -> the swap reads as {lost3} lost (the reported n_lost 1024 signature). The STAMP "
          "consequence lives in driver.py flush_cloud (per-point t); test_seq_reorder.py proves it on the real driver",
          lost3 >= 1000)


def _stamp_rec(maker, seq, stamp_s, **kw):
    sec = int(stamp_s)
    return maker(seq, info_sec=sec, info_nsec=int(round((stamp_s - sec) * 1e9)), **kw)


def test_devclock_synthetic_drifting_cadence():
    """A device like the real L2 (measured 2026-10-07): DataInfo.stamp at 0.5027x real time and LINEAR; the packets'
    true interval 0.8 % LONGER than their `scan_period` field, and wandering (a slow sinusoid). Truth = the true
    capture instant. The device-stamp clock must keep stamps within 1 ms of truth; the cadence path (the same stream
    with stamps zeroed) drifts away (it is what the driver did before)."""
    rate, period_field = 0.5027, _PERIOD_S
    true_t, stamp0, off = 0.0, 1000.0, 2_000_000.0
    rng = np.random.default_rng(3)
    out = {}
    for mode in ("devclock", "cadence"):
        sync = proto.L2TimeSync(seq_modulus=1024, devclock_min_baseline_s=1.0)
        errs, t = [], 0.0
        for i in range(216 * 20):
            true_interval = period_field * 1.008 * (1.0 + 0.002 * np.sin(2 * np.pi * t / 7.0))
            t += true_interval
            recv = off + t + _PERIOD_S / 2 + rng.uniform(0.0, 0.0003)
            st = stamp0 + t * rate if mode == "devclock" else 0.0
            rec = _stamp_rec(_point_rec, i % 1024, st) if st else _point_rec(i % 1024)
            d = sync.point_packet_device_time(rec, recv)
            if i > 216 * 8:
                errs.append(sync.node_time(d) - (off + t))
        e = np.array(errs)
        out[mode] = float(np.ptp(e)) * 1e3
    check(f"devclock: stamp error spread vs truth over 12 s <= 1 ms (got {out['devclock']:.3f} ms)", out["devclock"] <= 1.0)
    check(f"cadence path on the same device drifts more (spread {out['cadence']:.2f} ms > 3 ms): the clock the driver "
          "used before", out["cadence"] > 3.0)


def test_devclock_reboot_falls_back_continuously():
    sync = proto.L2TimeSync(seq_modulus=1024, devclock_min_baseline_s=1.0)
    out, t, last = [], 0.0, None
    ok = True
    for i in range(216 * 12):
        t += _PERIOD_S
        stamp = (500.0 + t * 0.5) if i < 216 * 6 else (1.0 + (t - 6.0) * 0.5)   # the stamp restarts at 6 s (reboot)
        d = sync.point_packet_device_time(_stamp_rec(_point_rec, i % 1024, stamp), 3e6 + t)
        if last is not None and d <= last:
            ok = False
        last = d
        out.append(sync._dclock.active)
    check("devclock reboot: point device times stay strictly increasing through the stamp restart", ok)
    check(f"devclock reboot: the clock reset ({sync._dclock.n_resets}) and is active again by the end",
          sync._dclock.n_resets == 1 and out[-1] and not out[216 * 6 + 5])


def test_devclock_real_fixture_imu_monotone_one_map():
    """REAL L2 data (test_fixtures/l2_real_imu_coalesced_3s.pcap, IMU on): with the device-stamp clock (baseline
    shortened to 1 s for a 3 s fixture) every IMU stamp is strictly increasing and kept, and a point packet's own
    stamp maps through the IDENTICAL function as an IMU stamp. NEGATIVE: the old per-packet map (re-anchored on
    every point packet), emulated here, produces non-increasing IMU stamps on the same data."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "bench"))
    import l2_pcap
    sync = proto.L2TimeSync(seq_modulus=1024, devclock_min_baseline_s=1.0)
    old = proto.L2TimeSync(seq_modulus=1024, devclock_min_baseline_s=1e9)   # never activates: the cadence clock, as before
    guard = proto.SeqOrderGuard(1024)
    imu_new, imu_old, same_map = [], [], []
    map_first = map_last = None
    for t, b in l2_pcap.read_pcap(os.path.join(here, "test_fixtures", "l2_real_imu_coalesced_3s.pcap")):
        for f in proto.split_datagram(b, None):
            r = proto.parse_packet(f, proto.DropStats())
            if r is None:
                continue
            rec = r[1]
            stamp = float(rec["info_sec"]) + float(rec["info_nsec"]) * 1e-9
            if r[0] == proto.PT_POINT and guard.accept(int(rec["info_seq"]), t) is None:
                d = sync.point_packet_device_time(rec, t)
                if sync._dclock.active:
                    same_map.append(abs(d - sync._dclock.map(stamp)))
                vt_old = old.point_packet_device_time(rec, t)   # the old map anchored on every point packet's CADENCE time
                map_first = map_first or (stamp, vt_old)
                map_last = (stamp, vt_old)
            elif r[0] == proto.PT_IMU and sync._dclock.active:
                imu_new.append(sync.imu_node_time(rec, t))
                ddev, dvt = map_last[0] - map_first[0], map_last[1] - map_first[1]
                old_rate = ddev / dvt if dvt > 0 else 0.5
                imu_old.append(old.node_time(map_last[1] + (stamp - map_last[0]) / old_rate))
    dn, do = np.diff(imu_new), np.diff(imu_old)
    check(f"real fixture: device-stamp clock active, {len(imu_new)} IMU stamps, NONE non-increasing (got {int((dn <= 0).sum())})",
          len(imu_new) > 300 and int((dn <= 0).sum()) == 0)
    check(f"real fixture: point packets map through the identical function as IMU (max diff {max(same_map):.2e} s)",
          same_map and max(same_map) < 1e-9)
    check(f"NEGATIVE: the old per-packet map on the same data gives non-increasing IMU stamps ({int((do <= 0).sum())})",
          int((do <= 0).sum()) > 0)
    check(f"IMU stamp spacing jitter (std): device-stamp map {dn.std() * 1e3:.3f} ms vs old {do.std() * 1e3:.3f} ms",
          dn.std() < do.std())


if __name__ == "__main__":
    main()

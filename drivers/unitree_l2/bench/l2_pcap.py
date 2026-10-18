#!/usr/bin/env python3
"""l2_pcap — offline tools for raw L2 captures (tcpdump -w, classic libpcap, Ethernet/IPv4/UDP).

  analyze <pcap>              timing facts: device stamp vs capture time, seq, scan_period,
                              head revolution period from the azimuth (theta) progression.
  replay <pcap> --to IP:PORT  re-send the UDP payloads with their original spacing (for the
                              driver / estimator / vendor SDK offline; never sends to the sensor
                              unless you point it there — default 127.0.0.1:6231).

No ROS. Pure python + numpy + l2_protocol.
"""
import argparse, json, os, socket, struct, sys, time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import l2_protocol as P  # noqa: E402


def read_pcap(path, port=6201):
    """Yield (capture_time_s, udp_payload) for UDP packets to `port`."""
    with open(path, "rb") as f:
        gh = f.read(24)
        magic = struct.unpack("<I", gh[:4])[0]
        if magic == 0xA1B2C3D4:
            endian, nano = "<", False
        elif magic == 0xA1B23C4D:
            endian, nano = "<", True
        else:
            raise ValueError(f"not a little-endian classic pcap: magic {magic:#x}")
        linktype = struct.unpack(endian + "I", gh[20:24])[0]
        while True:
            h = f.read(16)
            if len(h) < 16:
                return
            ts_s, ts_frac, incl, _orig = struct.unpack(endian + "IIII", h)
            data = f.read(incl)
            t = ts_s + ts_frac * (1e-9 if nano else 1e-6)
            off = 14 if linktype == 1 else (16 if linktype == 113 else None)   # EN10MB / Linux SLL
            if off is None:
                raise ValueError(f"unsupported linktype {linktype}")
            if linktype == 1:
                ethertype = struct.unpack(">H", data[12:14])[0]
                if ethertype == 0x8100:
                    off, ethertype = 18, struct.unpack(">H", data[16:18])[0]
                if ethertype != 0x0800:
                    continue
            ip = data[off:]
            if ip[9] != 17:
                continue
            ihl = (ip[0] & 0x0F) * 4
            udp = ip[ihl:]
            dport = struct.unpack(">H", udp[2:4])[0]
            if dport != port:
                continue
            ulen = struct.unpack(">H", udp[4:6])[0]
            yield t, bytes(udp[8:ulen])


def _frames(path, stats):
    """(capture time, frame) for every frame of every datagram: the L2 coalesces an IMU frame and a point
    frame into one datagram (2026-10-06); P.split_datagram splits it (or returns it whole) and counts it in
    stats.events. Frames from one datagram share its capture time."""
    for t, buf in read_pcap(path):
        for frame in P.split_datagram(buf, stats):
            yield t, frame


def pct(v, qs=(0, 1, 50, 99, 100)):
    v = np.asarray(v, float)
    return None if v.size == 0 else {f"p{q}": round(float(np.percentile(v, q)), 6) for q in qs}


def analyze(path):
    stats = P.DropStats()
    cap, dev, seq, sp, tinc, npts, theta0, thstep, rot, comrot = ([] for _ in range(10))
    types = Counter()
    for t, frame in _frames(path, stats):
        r = P.parse_packet(frame, stats)
        if r is None:
            continue
        ptype, rec = r
        types[ptype] += 1
        if ptype != P.PT_POINT:
            continue
        cap.append(t)
        dev.append(float(rec["info_sec"]) + float(rec["info_nsec"]) * 1e-9)
        seq.append(int(rec["info_seq"]))
        sp.append(float(rec["scan_period"])); tinc.append(float(rec["time_increment"]))
        npts.append(int(rec["point_num"]))
        theta0.append(float(rec["com_horizontal_angle_start"])); thstep.append(float(rec["com_horizontal_angle_step"]))
        rot.append(float(rec["sys_rotation_period"])); comrot.append(float(rec["com_rotation_period"]))
    cap, dev = np.array(cap), np.array(dev)
    dcap, ddev = np.diff(cap), np.diff(dev)
    # device clock rate vs capture clock over the whole file (least squares)
    A = np.vstack([cap - cap[0], np.ones_like(cap)]).T
    slope = float(np.linalg.lstsq(A, dev - dev[0], rcond=None)[0][0]) if len(cap) > 2 else None
    # head revolution: unwrap the azimuth at each packet's start
    th = np.unwrap(np.array(theta0))
    revs = (th[-1] - th[0]) / (2 * np.pi)
    dur = cap[-1] - cap[0]
    # per-revolution durations: times where the unwrapped azimuth crosses multiples of 2*pi
    k = np.floor((th - th[0]) / (2 * np.pi))
    cross = cap[1:][np.diff(k) != 0]
    span = np.array(npts) * np.array(tinc)
    return {
        "file": os.path.basename(path), "packet_types": dict(types), "drops": stats.counts,
        "coalesced_datagrams": stats.events.get("coalesced_datagram", 0),
        "n_point_packets": len(cap), "duration_s": round(float(dur), 3),
        "capture_interarrival_ms": pct(dcap * 1e3),
        "device_stamp_step_ms": pct(ddev * 1e3),
        "device_stamp_first_s": round(float(dev[0]), 6),
        "device_vs_capture_rate": slope,
        "device_minus_capture_drift_ms_over_file": round(float(((dev[-1] - dev[0]) - dur) * 1e3), 3),
        "scan_period_ms": pct(np.array(sp) * 1e3), "packet_point_span_ms": pct(span * 1e3),
        "seq_diff_counts": dict(Counter(int(x) for x in np.diff(seq)).most_common(5)),
        "azimuth": {"revolutions": round(float(revs), 3), "rev_period_ms_mean": round(float(dur / revs * 1e3), 2) if revs else None,
                    "rev_period_ms_each": pct(np.diff(cross) * 1e3), "direction": "ccw(+)" if revs > 0 else "cw(-)",
                    "theta_step_per_point": pct(thstep, (0, 50, 100))},
        "sys_rotation_period": pct(rot, (0, 50, 100)), "com_rotation_period": pct(comrot, (0, 50, 100)),
    }


def imu(path):
    """IMU facts from a capture (IMU verification): rate, native units, gravity axis at rest,
    gyro bias, quaternion norm, and IMU device stamps on the point packets' device clock."""
    st = P.DropStats()
    it, ia, ig, iq, idev = [], [], [], [], []
    pdev, pcap = [], []
    for t, frame in _frames(path, st):
        r = P.parse_packet(frame, st)
        if r is None:
            continue
        ptype, rec = r
        dev = float(rec["info_sec"]) + float(rec["info_nsec"]) * 1e-9
        if ptype == P.PT_IMU:
            it.append(t); idev.append(dev)
            ia.append(np.asarray(rec["linear_acceleration"], float)); ig.append(np.asarray(rec["angular_velocity"], float))
            iq.append(np.asarray(rec["quaternion"], float))
        elif ptype == P.PT_POINT:
            pcap.append(t); pdev.append(dev)
    if not it:
        return {"n_imu": 0, "drops": st.counts, "coalesced_datagrams": st.events.get("coalesced_datagram", 0),
                "n_point_packets": len(pcap)}
    it, ia, ig, iq, idev = map(np.array, (it, ia, ig, iq, idev))
    norm = np.linalg.norm(ia, axis=1)
    gyro_norm = np.linalg.norm(ig, axis=1)
    rest = gyro_norm <= max(np.percentile(gyro_norm, 30) * 1.5, 1e-9)   # quietest samples = at rest (<=: a noiseless gyro reads exactly 0)
    g_mean = ia[rest].mean(0)
    # IMU device stamps vs point device stamps: same oscillator? fit both against capture time
    rate_imu = float(np.polyfit(it - it[0], idev - idev[0], 1)[0])
    rate_pt = float(np.polyfit(np.array(pcap) - pcap[0], np.array(pdev) - pdev[0], 1)[0]) if len(pcap) > 2 else None
    return {"n_imu": int(len(it)), "rate_hz": round((len(it) - 1) / (it[-1] - it[0]), 3),
            "interarrival_ms": pct(np.diff(it) * 1e3), "drops": st.counts,
            "coalesced_datagrams": st.events.get("coalesced_datagram", 0),
            "accel_norm_native": pct(norm, (1, 50, 99)),
            "units_guess": "g" if 0.8 < np.median(norm) < 1.2 else "m/s^2" if 8.0 < np.median(norm) < 11.6 else "unknown",
            "accel_mean_at_rest_native": g_mean.round(4).tolist(),
            "gravity_axis_at_rest": ["x", "y", "z"][int(np.argmax(np.abs(g_mean)))] + ("+" if g_mean[np.argmax(np.abs(g_mean))] > 0 else "-"),
            "gyro_bias_at_rest_native": ig[rest].mean(0).round(5).tolist(),
            "gyro_std_at_rest_native": ig[rest].std(0).round(5).tolist(),
            "gyro_peak_native": {"x": pct(ig[:, 0], (0, 100)), "y": pct(ig[:, 1], (0, 100)), "z": pct(ig[:, 2], (0, 100))},
            "quat_norm": pct(np.linalg.norm(iq, axis=1), (0, 50, 100)),
            "device_stamp_rate_imu": round(rate_imu, 5), "device_stamp_rate_points": None if rate_pt is None else round(rate_pt, 5),
            "note": "tilt/rotation responses: see gyro_peak_native signs against the scripted motions (FLU: nose-up = -pitch about +y, left-side-up = +roll about +x... see L2_WINDOW_2.md)"}


def replay(path, to, speed=1.0, loops=1):
    ip, port = to.split(":")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    n = 0
    for _ in range(loops):
        pkts = list(read_pcap(path))
        t0c, t0w = pkts[0][0], time.monotonic()
        for tc, payload in pkts:
            due = t0w + (tc - t0c) / speed
            dt = due - time.monotonic()
            if dt > 0:
                time.sleep(dt)
            s.sendto(payload, (ip, int(port)))
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("analyze"); a1.add_argument("pcap"); a1.add_argument("--out")
    a3 = sub.add_parser("imu"); a3.add_argument("pcap"); a3.add_argument("--out")
    a2 = sub.add_parser("replay"); a2.add_argument("pcap"); a2.add_argument("--to", default="127.0.0.1:6231")
    a2.add_argument("--speed", type=float, default=1.0); a2.add_argument("--loops", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "analyze":
        rep = analyze(a.pcap)
        txt = json.dumps(rep, indent=1)
        print(txt)
        if a.out:
            open(a.out, "w").write(txt)
    elif a.cmd == "imu":
        txt = json.dumps(imu(a.pcap), indent=1)
        print(txt)
        if a.out:
            open(a.out, "w").write(txt)
    else:
        print(json.dumps({"sent": replay(a.pcap, a.to, a.speed, a.loops)}))


if __name__ == "__main__":
    main()

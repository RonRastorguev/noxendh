"""Scout flight log analysis (offline, reads a ULog file; touches no hardware).

Separates the two usual causes of a hovering multicopter "rocking in a circle":
  * ATTITUDE oscillation (controller tuning): roll/pitch rate vs its setpoint, dominant frequency (typically several Hz);
  * POSITION orbit (flow position hold circling its hold point): x/y deviation going round, period (typically seconds),
    radius, and whether x and y are ~90 deg out of phase (a circle) or in phase (a line).
Also: modes over time, flow fusion, height from the rangefinder, vibration, battery under load.

    py -3.9 analyze_flight.py <log.ulg>
"""
import sys

import numpy as np
from pyulog import ULog

NAV = {0: "Manual", 1: "Altitude", 2: "Position", 3: "Mission", 4: "Hold", 5: "Return", 10: "Acro", 14: "Offboard",
       15: "Stabilized", 17: "Takeoff", 18: "Land", 12: "Descend"}


def topic(ulog, name, inst=0):
    for d in ulog.data_list:
        if d.name == name and d.multi_id == inst:
            return d.data
    return None


def dom_freq(t, x, fmin=0.2, fmax=30.0):
    """Dominant frequency (Hz) and its share of the band's power, for an evenly resampled signal."""
    if len(t) < 64:
        return None, 0.0
    fs = 1.0 / np.median(np.diff(t))
    tt = np.arange(t[0], t[-1], 1.0 / fs)
    xx = np.interp(tt, t, x)
    xx = xx - xx.mean()
    spec = np.abs(np.fft.rfft(xx * np.hanning(len(xx)))) ** 2
    f = np.fft.rfftfreq(len(xx), 1.0 / fs)
    band = (f >= fmin) & (f <= min(fmax, fs / 2))
    if not band.any() or spec[band].sum() == 0:
        return None, 0.0
    i = np.argmax(spec[band])
    return float(f[band][i]), float(spec[band][i] / spec[band].sum())


def main(path):
    ulog = ULog(path)
    t0 = ulog.start_timestamp
    sec = lambda ts: (np.asarray(ts, dtype=np.int64) - t0) * 1e-6
    print(f"log {path}: {(ulog.last_timestamp - t0) * 1e-6:.0f} s")

    vs = topic(ulog, "vehicle_status")
    segs = []
    if vs is not None:
        ts, nav, arm = sec(vs["timestamp"]), vs["nav_state"], vs["arming_state"]
        start = 0
        for i in range(1, len(ts) + 1):
            if i == len(ts) or nav[i] != nav[start] or arm[i] != arm[start]:
                segs.append((ts[start], ts[min(i, len(ts) - 1)], int(nav[start]), int(arm[start]) == 2))
                start = i
        print("modes:")
        for a, b, n, armed in segs:
            if b - a >= 0.5:
                print(f"  {a:6.1f}-{b:6.1f} s  {NAV.get(n, n):<11} {'ARMED' if armed else 'disarmed'}")

    lp = topic(ulog, "vehicle_local_position")
    land = topic(ulog, "vehicle_land_detected")
    air = None
    if land is not None:
        lt, landed = sec(land["timestamp"]), land["landed"]
        air_idx = np.where(landed == 0)[0]
        if len(air_idx):
            air = (lt[air_idx[0]], lt[min(air_idx[-1] + 1, len(lt) - 1)])
            print(f"airborne (land detector): {air[0]:.1f}-{air[1]:.1f} s ({air[1] - air[0]:.0f} s)")

    flags = topic(ulog, "estimator_status_flags")
    if flags is not None and air is not None:
        ft = sec(flags["timestamp"])
        m = (ft > air[0] + 1) & (ft < air[1] - 1)
        if m.any():
            print(f"flow fused while airborne: {100 * flags['cs_opt_flow'][m].mean():.0f} % of samples; "
                  f"range height {100 * flags['cs_rng_hgt'][m].mean():.0f} %")

    def window(name):
        out = []
        for a, b, n, armed in segs:
            if armed and NAV.get(n) == name and b - a > 4:
                if air is not None:
                    a, b = max(a, air[0] + 1.5), min(b, air[1] - 1.0)
                if b - a > 4:
                    out.append((a, b))
        return out

    att = topic(ulog, "vehicle_angular_velocity")
    rsp = topic(ulog, "vehicle_rates_setpoint")
    for mode in ("Position", "Stabilized", "Altitude"):
        wins = window(mode)
        if not wins:
            continue
        print(f"\n== {mode}: {sum(b - a for a, b in wins):.0f} s airborne")
        if att is not None:
            at = sec(att["timestamp"])
            for axis, k, sk in (("roll", "xyz[0]", "roll"), ("pitch", "xyz[1]", "pitch")):
                for a, b in wins:
                    m = (at >= a) & (at <= b)
                    if m.sum() < 100:
                        continue
                    rate = att[k][m]
                    f, share = dom_freq(at[m], rate, 1.0, 40.0)
                    line = f"  {axis} rate: rms {np.degrees(rate.std()):.1f} deg/s, dominant {f:.1f} Hz ({100 * share:.0f} % of 1-40 Hz power)"
                    if rsp is not None:
                        st = sec(rsp["timestamp"])
                        sp = np.interp(at[m], st, rsp[sk])
                        line += f", tracking error rms {np.degrees((rate - sp).std()):.1f} deg/s"
                    print(line)
        vatt = topic(ulog, "vehicle_attitude")
        if vatt is not None:
            vt = sec(vatt["timestamp"])
            q = np.stack([vatt[f"q[{i}]"] for i in range(4)], 1)
            roll = np.degrees(np.arctan2(2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)))
            pitch = np.degrees(np.arcsin(np.clip(2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]), -1, 1)))
            for a, b in wins:
                m = (vt >= a) & (vt <= b)
                if m.sum() > 50:
                    fr, sr = dom_freq(vt[m], roll[m], 0.1, 10)
                    print(f"  tilt: roll {roll[m].mean():+.1f} ± {roll[m].std():.1f} deg, pitch {pitch[m].mean():+.1f} ± {pitch[m].std():.1f} deg; "
                          f"slow tilt rhythm {fr:.2f} Hz ({100 * sr:.0f} %)")
        if lp is not None:
            pt = sec(lp["timestamp"])
            for a, b in wins:
                m = (pt >= a) & (pt <= b)
                if m.sum() < 50:
                    continue
                x, y = lp["x"][m] - lp["x"][m].mean(), lp["y"][m] - lp["y"][m].mean()
                fx, sx = dom_freq(pt[m], x, 0.05, 3)
                fy, sy = dom_freq(pt[m], y, 0.05, 3)
                c = np.corrcoef(x, y)[0, 1]
                # phase between x and y at the dominant frequency: ~ +-90 deg = circle, ~0/180 = line
                if fx:
                    tt = pt[m]
                    w = 2 * np.pi * fx
                    X = np.sum(x * np.exp(-1j * w * tt)); Y = np.sum(y * np.exp(-1j * w * tt))
                    ph = np.degrees(np.angle(Y / X)) if abs(X) > 0 else float("nan")
                else:
                    ph = float("nan")
                r = np.hypot(x, y)
                print(f"  position: radius {r.mean() * 100:.0f} cm mean / {r.max() * 100:.0f} cm max; x {fx and round(fx, 2)} Hz, "
                      f"y {fy and round(fy, 2)} Hz; x-y phase {ph:+.0f} deg (±90 = circling), corr {c:+.2f}; "
                      f"speed rms {np.hypot(lp['vx'][m], lp['vy'][m]).std() * 100:.0f} cm/s")
                print(f"  height (dist_bottom): {lp['dist_bottom'][m].mean():.2f} m ± {lp['dist_bottom'][m].std() * 100:.0f} cm; "
                      f"eph {lp['eph'][m].mean():.2f} m")

    vib = topic(ulog, "vehicle_imu_status")
    if vib is not None and air is not None:
        t = sec(vib["timestamp"])
        m = (t > air[0] + 1) & (t < air[1] - 1)
        if m.any():
            print(f"\nvibration (accel_vibration_metric) airborne: {vib['accel_vibration_metric'][m].mean():.1f} m/s^2 "
                  f"(PX4 flags > ~30 as high); gyro {vib['gyro_vibration_metric'][m].mean():.3f} rad/s")
    bat = topic(ulog, "battery_status")
    if bat is not None:
        bt = sec(bat["timestamp"])
        v = bat["voltage_v"]
        if air is not None:
            m = (bt > air[0]) & (bt < air[1])
            if m.any():
                print(f"battery under load: {v[m].max():.2f} -> {v[m].min():.2f} V, current {bat['current_a'][m].mean():.1f} A mean")
    print("\nPX4 params of note:", {k: ulog.initial_parameters.get(k) for k in
          ("SENS_FLOW_ROT", "SDLOG_PROFILE", "MC_ROLLRATE_P", "MC_ROLLRATE_D", "MC_PITCHRATE_P", "MC_PITCHRATE_D",
           "MC_ROLL_P", "MC_PITCH_P", "MPC_XY_VEL_P_ACC", "MPC_XY_P", "IMU_GYRO_CUTOFF", "IMU_DGYRO_CUTOFF")})


if __name__ == "__main__":
    main(sys.argv[1])

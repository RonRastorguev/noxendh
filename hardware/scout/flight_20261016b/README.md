# Scout flights after S4 (2026-10-16): `sess010/log100` (10 s) and `log101` (63 s airborne)

Pulled read-only over MAVLink FTP (tx audit clean: FTP reads + heartbeats only). Gains: rate P 0.12, rate D 0.002,
attitude P 6.5 (S4); ch 5 low = Altitude, high = Position.

## The 1.5 Hz rocking: fixed

Hands-off Position windows, amplitude of the attitude oscillation at 1.5 Hz (actual / commanded):

| | pitch | roll |
|---|---|---|
| before S4 (sess009/log103, 19.5–24.1 s) | **6.3° / 3.2°** | 1.6° / 0.8° |
| before S4 (68.9–72.9 s) | **5.5° / 3.6°** | 2.4° / 1.5° |
| after S4 (sess010/log101, 42.1–48.0 s) | **0.2° / 0.2°** | 0.1° / 0.1° |
| after S4 (51.0–66.8 s) | **0.3° / 0.2°** | 0.4° / 0.3° |

About 20× smaller, and the actual now follows the command (ratio ~1) instead of doubling it. Pitch tilt in Position ±1–3°
(was ±5–6°). Small residual at ~1.2 Hz in one window (1.2–1.3° actual vs 0.5° commanded): optional further step later.

## Position hold

- 42–48 s hands-off: within **10 cm** mean (15 cm max), 5 cm/s rms, height ±3 cm.
- 51–67 s hands-off "3 m drift" is an artifact: at **57.0 s PX4 switched primary estimator 1 → 0** (instance 1's combined test ratio
  reached 1.11 > 1). On flow only, each estimator's x/y origin drifts independently, so the switch is a 3.6 m step in the
  estimate; PX4 shifted the hold point with it and the Scout held within ~0.2–0.3 m of the new point. sess009/log103 had two
  switches at takeoff (5.4 s, 6.7 s).
- **Design note for the autonomy (laptop setpoints / EV):** every estimator switch resets the local frame. Either the nav code
  handles PX4's reset counters/deltas, or the FC runs one estimator (`EKF2_MULTI_IMU` 1, on IMU 0 — IMU 1 is the one with
  thousands of data gaps on the bench and more accel clipping: 19 vs 1 here, 66 vs 26 in log103). A decision for Ron later;
  EV fusion (camera pose) will pin both instances to the map frame anyway.

## log100 (10 s, Altitude only)

Took off with the stick at +0.5…+0.88 (climb), reached 0.75 m at 11 s, then the throttle went to full down (−1.00): height
read ~0 at 12 s, 0.28 m at 14 s (a bounce), roll −6° at 15 s, then landed. Looks like an aborted hop; ask Ron.

## Battery

log101: 14.85 → **13.61 V** under load (9.9 A mean), at the 13.6 V land line. log100: 15.03 → 14.40 V at 4.2 A.

## Ron: "when I flip it onto high it still drifts upward a bit" (and after touching it)

- The throttle stick sat at **+0.34…+0.39** (−1…+1) the whole flight, outside the ±0.10 dead zone (`MAN_DEADZONE`): PX4 read
  a slow climb command (+0.12…+0.15 m/s) and held no height setpoint. Position/Altitude hold height only with the stick centred.
- Underneath: with **estimator instance 1 (IMU 1)** in charge the estimated vertical speed ran **+0.19 m/s** above the true
  rate (9 five-second chunks, +0.12…+0.26) for 49 s; it agreed again after the 57 s switch to instance 0. On sess009/log103
  (instance 0 throughout, 107 s) the bias was +0.04 m/s mean. So the climb came out slower than commanded: a gentle drift.
- The barometer is alive (20 Hz, ±0.35 m in prop wash; an earlier "0.00" column was a lookup bug).
- A hand under the Scout looks like the floor rising to the rangefinder: it climbs to keep the distance (range-height hold).
- Fix proposed: `hardware/scout/PARAM_CHANGESET_S5.json` (single estimator, IMU 1 low priority), dry-run clean.

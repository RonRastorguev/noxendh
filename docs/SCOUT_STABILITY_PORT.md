# Porting our reference stability set to the Scout (plan, 2026-10-15)

> **Status 2026-10-15:** sections A and B applied as `hardware/scout/PARAM_CHANGESET_S1.json` and `S2.json` on Ron's go
> (`hardware/scout/apply_20261015/`). Offboard switch = ch 6 high. EV fusion (A, last row) and the software below are next.
> **2026-10-16:** the readiness checks found the flow sensor mounted 90° from its setting; S3 (`SENS_FLOW_ROT` 4 → 2) fixed it,
> verified by rocking and displacement tests (`hardware/scout/READINESS_20261016.md`).
> Flight-tested the same day: S4 (inner-loop gains, Altitude on ch 5 low) removed a 1.5 Hz pitch resonance; S5 (one estimator
> on IMU 0) removed a +0.19 m/s vertical-speed bias from IMU 1 and the estimator-switch frame jumps. Position hold returns to
> its spot in 1.3–1.8 s after a 45–60 cm push (`hardware/scout/flight_20261016c/`). **Manual stability: done.**

DH-01 allows reusing our existing stability software. The Scout's FC already runs the reference airframe's
exact PX4 build (`hardware/scout/fc_baseline_20261015/`), so the port is mostly **parameters + process**,
not firmware. This compares the Scout's baseline with the reference airframe's live parameter set: 71 parameters differ once calibration,
counters and GPS-only values are filtered out.

## Process carried over unchanged

- No parameter write without Ron's "go" on a change-set file's sha256; the write tool reads every
  value back and the FC is re-dumped and diffed afterwards (the bench's `fc_apply_changeset.py`).
- Read-only access only through the logging bridge; every session's transmit log is audited.
- Never by an agent: firmware flash, calibration, arming, motor output, actuator tests.
- Battery in = ESCs live: props off on the bench, voltage watched, never unattended.

## A. Port (estimation + failsafe: the "stability" part)

| Parameter | Reference | Scout now | For the Scout | Why |
|---|---|---|---|---|
| `COM_LOW_BAT_ACT` | 2 (Land) | 0 (warn only) | 2 | no GPS, so Return is meaningless; land, don't fall |
| `SYS_HAS_NUM_DIST` | 1 | 0 | 1 | refuse to arm without the rangefinder (the floor-stop sensor) |
| `SDLOG_PROFILE` | 3 | 1 | 3 | estimator replay data in every log: how the reference airframe's EKF2 issues were found |
| `EKF2_OF_POS_X/Y/Z` | 0.0225 / 0 / 0.074 | 0 / 0 / 0.02 | 0 / 0.0076 / 0.0127 | Ron's measurement (`config/scout_hw.yaml`) |
| `EKF2_RNG_POS_X/Y/Z` | same | same | same as flow | same module |
| `EKF2_MIN_RNG` | 0.15 | 0.01 | the rangefinder's at-rest reading | the reference airframe's "on the ground" lesson: never assume 0. Needs the at-rest reading (bench showed 8 cm vs 1.7 in measured: Q2) |
| EV fusion (`EKF2_EV_CTRL`, `EV_NOISE_MD`, `EVP/EVA_NOISE`, `DELAY_MAX`) | tuned for its EV | off | port when the laptop's camera pose (§8.3) is fed in | stage 2, after PnP works |

## B. Scale for the Scout (judgment calls: Ron decides)

| Parameter group | Reference | Scout now | Proposal |
|---|---|---|---|
| Speed limits `MPC_XY_VEL_MAX`, `MPC_VEL_MANUAL`, `MPC_XY_CRUISE` | 1.5 m/s | 12 / 10 / 5 m/s (racer defaults) | indoor caps ~1 m/s manual; autonomous ≤ 0.5 m/s is the spec's (§8.5) |
| Vertical `MPC_Z_*`, `MPC_TKO_SPEED`, `MPC_LAND_SPEED` | 0.6–1.0 m/s | 1.5–3 m/s | reference-like values |
| Offboard loss `COM_OF_LOSS_T` / `COM_OBL_RC_ACT` | — | 1 s / 0 | the DH-05 backstop: exact action chosen from the PX4 1.18 source for a no-GPS, flow-only vehicle |
| `RC_MAP_OFFB_SW` | 6 | 0 | needs the transmitter's free switch (Q4) |

## C. Do NOT port (airframe-specific)

Rate/attitude gains (`MC_*`), gyro filters (`IMU_*_CUTOFF`), `THR_MDL_FAC`, board and flow rotation,
battery capacity, `SYS_AUTOSTART`, serial-port assignments, GPS. The Scout is a different, smaller
airframe; its own values were set by whoever configured it.

## Software (beyond parameters)

- Laptop side: the vehicle adapter (offboard setpoints), the planner's safety gate (speed cap,
  geofence, hover when `loc_ok` is false). Ported from our reference PX4 adapter's patterns.
- Onboard (Radxa): no safety logic (DH-05); the uXRCE-DDS agent on the FC's TELEM2 and the camera stream.

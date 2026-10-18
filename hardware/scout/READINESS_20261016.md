# Scout readiness check: Position-mode hover on flow (2026-10-16 ~00:20, read-only)

FC on laptop USB (battery removed mid-check at 14.17 V, below the 14.4 V line). Read-only only: the
logging bridge, whitelisted console commands (`listener`, `sensors status`, `commander status`,
`param show`). This is a report of what the checks show. **Whether to fly is Ron's call.**

## Passes

| Check | Result |
|---|---|
| Firmware / params | PX4 1.18.0 RC `fca3df86`; S1 + S2 live (18 params, verified) |
| IMUs, baro | 2 accel + 2 gyro + baro all `state: OK`, validator failsafe NO |
| Rangefinder | `dist_bottom` 0.044 m valid, `cs_rng_hgt` true, kinematically consistent |
| Height estimate | `z_valid`, `epv` 0.023 m (range primary, baro also fused) |
| Failsafes | RC loss → Land after 0.5 s (`NAV_RCL_ACT` 3, `COM_RC_LOSS_T` 0.5); critical battery → Land (`COM_LOW_BAT_ACT` 2, S1); kill switch ch 8 |
| Modes | ch 5 selects Stabilized / Altitude / Position (`COM_FLTMODE1/4/6` = 8/1/2); offboard ch 6 high (S1) |
| Arming blockers | none active: disarmed, not in failsafe, Remote ID not required (`COM_ARM_ODID` 0) |

## Findings that decide HOW to fly

1. **Flow is not fused on the ground** (`cs_opt_flow` false): flow is ignored below `SENS_FLOW_MINHGT` 0.08 m and the
   lens sits at 0.044 m. With nothing aiding it, the horizontal estimate has drifted to x −15.6 m, y 38.0 m with
   `eph` 14 m (29 position resets) while the Scout sat still. PX4 still reports `xy_valid` true, so **Position mode
   would engage on the ground on a meaningless position.**
   → Take off in **Altitude** mode, climb past ~0.3 m, and switch to **Position** only once flow is fused. The
   hand-lift test below checks that it is.
2. **No heading reference** (`cs_yaw_align` false, `heading_good_for_control` false: the compass is off). Expected
   for a flow-only setup: position hold works in the local frame, but yaw drifts slowly.
3. **The selected IMU (#1) shows 19 250 accel / 18 442 gyro data gaps** over ~2.3 h of uptime; IMU #0 shows 1. Not a
   blocker on its own; worth asking Spark whether that is known on this board.

## Not ready yet

| Item | Why |
|---|---|
| **Battery** | 14.17 V resting (~24 % on PX4's 3.5–4.2 V/cell scale), below the 14.4 V line. Fly on a **charged** pack. `COM_ARM_BAT_MIN` is −1: PX4 will NOT refuse to arm on a low pack. `BAT1_CAPACITY` 1500 mAh: confirm it matches the pack. |
| **RC** | `manual_control_signal_lost` true: transmitter off or unbound. Mode switch (ch 5), kill switch (ch 8) and offboard switch (ch 6) are unverified until it is on. |
| **USB** | `CBRK_USB_CHK` is set, so PX4 WILL arm with USB plugged in. Unplug the laptop cable before flight. |
| **Tuning** | Nobody has confirmed these `MC_*` gains have flown a Position hold (NEEDS_RON R11). |
| **Logging profile** | FC uptime 2.3 h: not power-cycled since S1, so `SDLOG_PROFILE` 3 is not active yet. Logs still record when armed (profile 1). |

## Cannot be checked from here

Motor order and direction, prop direction, ESC behaviour, vibration in flight, real hover throttle, flow quality
at height over the actual floor texture and lighting.

## Hand-lift flow test 1 (2026-10-16 ~07:33Z, USB only, no battery)

`readiness_20261016/handtest.txt`. Lifted by hand to "about waist height" (rangefinder 0.66–0.72 m), held, moved,
set down. Flow quality 36–43 on the floor, 75–95 held up. Estimated velocity within ±0.08 m/s held still
(±0.04 for 66–78 s), swinging up to ±0.28 m/s while moved and back to ~0 when stopped. `POS_HORIZ_REL` on
throughout. Horizontal accuracy grew 0.14 → 0.44 m over a minute aloft (velocity-only aiding: expected).
**Shows flow is fused off the ground. Does NOT show direction** (forward→forward, right→right): that needs the
direction test (`tools/scout_fc/flow_direction_test.py`) before any Position-mode flight. Also open: waist height
vs 0.66 m on the rangefinder; check against a tape measure.

## Direction test 1 (2026-10-16 ~07:39Z): INCONCLUSIVE

`readiness_20261016/dirtest.txt`. Ron: held "about a metre" high (rangefinder 0.83–0.89 m), moves about half a metre
forward and left, the right move shorter. Velocity responded to every move and settled between them, but most
moves showed on BOTH axes at once (e.g. 34–37 s back −0.13 with right +0.15; 58–61 s fwd +0.17 with right +0.17),
and the lift came at ~23 s, so segments cannot be matched to moves with confidence. Either the hand moved
diagonally or the flow axes are rotated: not settled. Bridge sent 0 bytes. Next: `flow_displacement_test.py`
(finds the pauses itself, reports distance per move in the Scout's frame; also checks scale) along tape marks,
at a tape-measured height.

## Displacement test (2026-10-16 ~07:50Z): FAIL — flow directions rotated ~+105…120°

`readiness_20261016/displacement.csv` (raw 30 Hz: range, x, y, vx, vy, yaw). Ron, hand-held at 38 in (0.97 m) by tape:
forward 39 in (0.99 m), back; right 6 in (0.15 m), back; left 21 in (0.53 m), back. The automatic pause detection
was too strict for a hand-held drone (3 pauses); the path was reconstructed by integrating body-frame velocity
(per-sample yaw):

| Move (true) | Measured fwd / right (m) | Measured dir | Offset |
|---|---|---|---|
| forward 0.99 (0°) | −0.45 / +0.74 | 121° | +121° |
| back (180°) | +0.42 / −1.03 | 292° | +112° |
| right 0.15 (90°) | −0.42 / +0.21 | 153° | +63° (small, noisy) |
| back (270°) | +0.35 / +0.23 | 33° | +123° |
| left 0.53 (270°) | +0.77 / +0.19 | 14° | +104° |
| back (90°) | −0.81 / −0.22 | 195° | +105° |

- Every move is rotated the same way by about +105…120°: **Position mode would push the wrong way. No Position flight.**
- Magnitude roughly right (0.87 / 1.11 m for the 0.99 m legs).
- **Rangefinder reads 0.81 m at a tape-measured 0.97 m** (16 % low, far outside the MTF-02P's ±2 cm): re-measure
  the reference point; a low height also scales flow velocity low.
- Candidate causes: (1) `SENS_FLOW_ROT` (4 = yaw 180) wrong: the data sits between 90° and 135° options;
  (2) the FC's body frame is not the camera direction (`SENS_BOARD_ROT` 8 missing a yaw), which would affect every
  mode. Next: the tilt test (`tools/scout_fc/attitude_tilt_test.py`) separates them.

## Tilt test (2026-10-16 ~08:06Z): PASS — PX4's body frame matches the airframe

`readiness_20261016/tilttest.txt`. Nose (camera side) down → pitch −26…−29° with roll ~+2°; right side down →
roll +32…+33° with pitch ~+2°. Correct FRD signs on the correct axes: `SENS_BOARD_ROT` 8 is right, forward = the
camera. So the displacement test's rotation is the **flow sensor's own orientation** (`SENS_FLOW_ROT`, now 4 =
yaw 180). The displacement data puts the error near +110°, between the 45° and 90° settings, too noisy to choose.
Next: `tools/scout_fc/flow_gyro_rotation.py` measures the misalignment from rotation alone (flow vs gyro in the same
OPTICAL_FLOW_RAD message while rocking in place). (A first tilt attempt recorded nothing: the bridge died with a
Windows socket "insufficient buffer space" error while the kernel non-paged pool stood at 2.5 GB and `TiWorker`
held 3 GB; a restart worked.)

## Rocking test (2026-10-16 ~08:15Z): flow is mounted 90° from what PX4 is told

`readiness_20261016/flow_gyro.csv`, `rocktest.txt`. Rocked in place at 0.52 m (flow quality 76), 215 samples with
rotation > 0.3 rad/s. PX4's OPTICAL_FLOW_RAD carries body-frame flow and the gyro together; aligned, pure rotation
gives flow = gyro on each axis. Measured: same-axis ratios +0.07 (x) / −0.03 (y), cross-axis flow_y = +1.16·gyro_x and
flow_x = −1.03·gyro_y: an exact +90° yaw rotation (not a mirror), misalignment +88.4°, magnitude ratio 1.11.
Consistent in sign with the displacement test's rougher ~+110°.
**Fix: `SENS_FLOW_ROT` 4 (yaw 180) → 2 (yaw 90)** (`hardware/scout/PARAM_CHANGESET_S3.json`), verified afterwards by the same
rocking test (same-axis ≈ +1, misalignment ≈ 0) and a displacement re-run before any Position flight.

## S3 applied + verification rocking test (2026-10-16): orientation FIXED

S3 (`SENS_FLOW_ROT` 4 → 2) written on Ron's go (`hardware/scout/apply_20261016_S3/`, tx log: one PARAM_SET, `SENS_FLOW_ROT=2`).
Verification (`readiness_20261016/flow_gyro_verify.csv`): misalignment **+88.4° → +7.3°** (last 5 s window −1.4°), same-axis
flow/gyro pitch +1.16 and roll +0.63 (were −0.03 / +0.07), magnitude 1.09. Positive same-axis ratios rule out the
value-6 convention. Only Ron's second attempt (35–50 s) fell inside the window: 72 rotating samples, roll under-sampled.
Battery: none connected at the time (FC on USB, reads 65.535 V). Remaining before a Position hover: the displacement
re-run, a charged pack, the transmitter on (modes/kill/offboard switches unverified), USB unplugged for flight.

## Displacement re-run after S3 (2026-10-16 ~21:55Z): PASS

`readiness_20261016/displacement2.csv`, `disptest2.txt`. Same moves as before (Ron: 38 in up; forward ~39 in, right ~6 in,
left ~25 in, each back to start). Integrated Scout-frame path: forward **+1.06 m** (true ~0.99), back −1.00; right **+0.34**
(Ron's estimate 0.15), back −0.35; left **−0.72** (true ~0.64), back +0.77. Cross-axis leakage ≤ ~7°; ended within 0.2 m of
the start. Directions correct on every move (were rotated ~110° before S3). Bridge sent 0 bytes.

**Flow orientation: DONE.** Still open before a Position-mode hover (Ron decides whether to fly):
- a charged pack (last reading 14.17 V; `COM_ARM_BAT_MIN` −1 means PX4 will not refuse a low pack);
- transmitter on, then check the mode switch (ch 5: Stabilized / Altitude / Position), kill switch (ch 8), offboard (ch 6);
- USB unplugged for flight (`CBRK_USB_CHK` lets PX4 arm with it in);
- take off in **Altitude**, switch to **Position** above ~0.3 m (flow is ignored below 0.08 m);
- the gains are unconfirmed for a Position hold (R11, ask Spark); `SDLOG_PROFILE` 3 active after a power-cycle;
- rangefinder reads 0.77–0.89 m at a tape-measured 0.97 m (~15–20 % low): check where the 38 in was measured from.

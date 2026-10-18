# Scout flight after S5 (2026-10-16): `sess011/log100` (64 s airborne, 29 s Position)

Pulled read-only (tx audit clean). Config: S1–S5 (flow rot 2, gains S4, one estimator on IMU 0).

- **Vertical-speed bias: fixed.** Estimated minus actual climb rate **+0.028 m/s** mean over 11 five-second chunks
  (−0.025…+0.058); was **+0.186** with IMU 1's estimator in charge.
- **One estimator** (one `estimator_status_flags` instance, no selector). One xy reset, at 72.7 s **after landing**, size 0.00 m:
  no frame jumps in flight.
- **Throttle centred this time:** stick median +0.02 in Position (dead zone ±0.10); height setpoint engaged 83 % of the time.
  Height in Position 0.93 m ± 5 cm.
- **Position hold:** 18 cm mean radius over the Position time, which includes four pushes by hand (sticks centred):

| push | peak speed | pushed from hold point | back within 15 cm | hold point moved |
|---|---|---|---|---|
| 49.1 s | 0.58 m/s | 60 cm | (a second push followed) | 0 cm |
| 52.2 s | 0.89 m/s | 60 cm | 1.8 s | 0 cm |
| 55.3 s | 0.46 m/s | 46 cm | 1.3 s | 0 cm |
| 59.4 s | 1.52 m/s | 44 cm | 1.3 s | 0 cm |

  Returning to the original spot is Position mode working as designed (the hold point stays fixed until a stick input).
- Attitude: no dominant 1.5 Hz peak in Position (≤ 8 % of band power). Vibration 0.7 m/s².
- Battery: 15.24 → 13.82 V under load, 9.9 A mean.

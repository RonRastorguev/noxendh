# Scout FC: S5 applied (2026-10-16)

- **Go:** Ron, in chat: "go S5 cfda56bc…d777". File hash matched; unchanged since its commit.
- **Board:** UID `0x3533511634343432` (the Scout's) confirmed by `scout_apply.sh`. FC on USB only, disarmed.
- **Writes (echoed exactly; post-dump changed exactly these 4):** `EKF2_MULTI_IMU` 3 → 0, `SENS_IMU_MODE` 0 → 1,
  `CAL_ACC1_PRIO` 50 → 25, `CAL_GYRO1_PRIO` 50 → 25. Transmit log: 4 PARAM_SET, all S5 names and values, no reboot sent.
- **Pending:** `EKF2_MULTI_IMU` and `SENS_IMU_MODE` take effect only after Ron power-cycles the FC (battery AND USB out).
  Then verify read-only: one estimator instance, primary IMU = accel 6946834 / gyro 6684690.

## Verified after Ron's power-cycle (2026-10-16, FC uptime ~50 s)

Read-only (`s5verify.txt`; tx log audit clean: heartbeats + whitelisted console lines):
- `EKF2_MULTI_IMU` 0, `SENS_IMU_MODE` 1, `CAL_ACC1_PRIO` 25, `CAL_GYRO1_PRIO` 25 (all live after the reboot); `SDLOG_PROFILE` 3.
- **One estimator:** `estimator_status_flags` has a single instance (was 2); `estimator_selector_status` "never published".
- **IMU 0 primary:** selected gyro 6684690 (0), selected accel 6946834 (0); IMU 1 (3604506) state OK at priority 25 = backup.
- On the ground: height valid (0.043 m), eph 1.3 cm.
- Still to confirm in flight: estimated vertical speed matches the height slope (no +0.19 m/s bias) and no frame jumps.

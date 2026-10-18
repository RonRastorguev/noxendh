# Scout FC: S4 applied (2026-10-16)

- **Go:** Ron, in chat: "go S4 6cc2936f…77ba". File hash matched and the file was unchanged since its commit.
- **Board:** `scout_apply.sh` confirmed UID `0x3533511634343432` (the Scout's). FC on USB only (no battery), disarmed.
- **Writes (all echoed exactly; post-dump changed exactly these 7):** `MC_ROLLRATE_P` and `MC_PITCHRATE_P` 0.08 → 0.12,
  `MC_ROLLRATE_D` and `MC_PITCHRATE_D` 0.001 → 0.002, `MC_ROLL_P` and `MC_PITCH_P` 8.0 → 6.5, `COM_FLTMODE1` 8 → 1.
  None needs a reboot.
- **Transmit log:** 7 PARAM_SET, every name and value in S4, all to target 1/1; no reboot sent.
- **Now:** ch 5 low = **Altitude**, high = Position (Stabilized is no longer on the switch; the kill switch stays on ch 8).
- **Next:** charged pack, take off in Altitude (low), climb past ~0.3 m, flip to Position, hover 30–60 s hands-off; then the
  log is read for the 1.5 Hz pitch oscillation. A high-pitched buzz or hot motors = rate gains too high: land.

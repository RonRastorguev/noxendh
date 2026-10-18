# Scout FC: S3 applied (2026-10-16 ~08:25Z)

- **Go:** Ron, in chat: "go S3 b038e6d8…e5b0". File hash matched before the apply.
- **Board:** `scout_apply.sh` confirmed UID `0x3533511634343432` (the Scout's). FC on USB only (no battery), disarmed.
- **Write:** `SENS_FLOW_ROT` 4 → 2, echoed 2; post-dump changed exactly that parameter. No reboot needed (firmware metadata).
- Windows had been restarted just before (non-paged pool 2.5 GB → 0.73 GB); the bridge accepted both connections.
- **Pending:** verification by the rocking test (same-axis flow/gyro ≈ +1, misalignment ≈ 0), then a displacement re-run,
  before any Position-mode flight. The transmit log and audit are added after the session closes.

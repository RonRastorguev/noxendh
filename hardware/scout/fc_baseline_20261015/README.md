# Scout flight controller: read-only baseline (2026-10-15)

Taken over USB (laptop COM4) with the read-only bench tool (`fc_readonly.py`) through the
logging COM bridge. Only whitelisted read-only messages were sent; the transmit log `tx_scoutprobe.bin`
audits clean (`audit-tx`: `non_whitelisted: []`, no reboots). Powered from USB only (battery 4.47 V
reading = no pack): ESCs not live.

| Fact | Value |
|---|---|
| PX4 | 1.18.0 RC, git `fca3df865af36124a28c9d607e850f111dbaaea9` (the build we target), built Sep 10 2026 |
| Board target | `MICOAIR_H743_V2` |
| UID | `0x3533511634343432` |
| USB | 1b8c:0036 (other MicoAir FCs share it: check the UID, never the COM port) |
| Airframe | `SYS_AUTOSTART` 4050, already configured by someone (Spark?): flow on, range height, mag off, uXRCE-DDS on TELEM2 @ 921600, kill switch on RC ch 8, board mounted rolled 180° |
| Rangefinder (MTF-02P) | streaming, read 8 cm on the bench (`DISTANCE_SENSOR`, 48 Hz on comp 88 + 10 Hz on comp 1) |
| Flow | `OPTICAL_FLOW_RAD` streaming |

Files: `scout_baseline_20261015.params` (+ `.json`, all 999 parameters), `probe2.json` (heartbeat,
`AUTOPILOT_VERSION`, message inventory), `tx_scoutprobe.bin` (every byte sent).

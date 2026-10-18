# S6 applied 2026-10-18 (Ron's go on sha256 882e9eab66be5724ed56025b6bc2b749589943bc6c1b77c4edd1296b62fc9761)

RadioMaster Pocket layout over ExpressLRS (`hardware/scout/PARAM_CHANGESET_S6.json`, radio side
`hardware/scout/RADIO_POCKET_S6.md`). Via `tools/scout_fc/scout_apply.sh` (UID check = Scout) → the bench's
`fc_apply_changeset.py`:

| Parameter | Before → after |
|---|---|
| RC_MAP_KILL_SW | 8 → 5 |
| RC_MAP_FLTMODE | 5 → 6 |
| RC_MAP_OFFB_SW | 6 → 7 |
| COM_FLTMODE1 | 1 → 2 (Position) |
| COM_FLTMODE6 | 2 → 11 (Land) |

Every write echoed back; pre/post dump diff = exactly these 5 (`UNEXPECTED changes: []`); no reboot needed.
Transmit log `tx_scout_S6.bin`: the 5 PARAM_SET are exactly these names and values (decoded), plus read-only traffic
(HEARTBEAT, PARAM_REQUEST_*, REQUEST_MESSAGE, whitelisted shell); no reboot frames.

**Not yet verified:** the switches end to end. The FC saw no RC signal since boot (battery out; the receiver is likely
battery-powered). Next: battery in (props off), Pocket on, move each switch, read `listener manual_control_switches`.

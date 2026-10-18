# Scout FC: S1 + S2 applied (2026-10-15, ~22:51–22:53 local)

- **Go:** Ron, in chat: "go S1 aa3c6176…df18 and S2 bff26d4c…3d3e". Both file hashes matched before the apply.
- **Board:** `scout_apply.sh` confirmed UID `0x3533511634343432` (the Scout's) before each set.
- **Bench state:** FC on laptop USB + Scout battery in (14.78 → 14.72 V, 4S), disarmed. Ron asked to keep the props off.
- **S1** (9 rows) and **S2** (9 rows): every write echoed back exactly, and each set's post-dump changed exactly its own rows.
  - PX4 itself lowered `MPC_VEL_MANUAL` and `MPC_XY_CRUISE` from the written 3.0 to 1.5 right after S2: it keeps
    them ≤ `MPC_XY_VEL_MAX` (1.5). End values = the reference set's. (The 3.0 in S2 was the firmware metadata's documented
    minimum; PX4's own constraint wins.)
- **Whole-session check:** `scout_after_S1S2.params` vs the morning baseline differs in exactly the 18 approved
  parameters (no others, not even gyro auto-cal).
- **Transmit log** (`tx_scout_apply_S1S2.bin`): `audit-tx` lists the 18 `PARAM_SET` as non-whitelisted (it flags every
  write by design); decoding them shows 18 distinct names, every name and value in S1/S2, all to target 1/1. No reboot sent.
- **Pending:** `SDLOG_PROFILE` (S1 row 3) takes effect after a power-cycle of the FC (battery AND USB out). Then
  re-check it with a read-only dump.

Folders: `apply_PARAM_CHANGESET_S1_055138/`, `apply_PARAM_CHANGESET_S2_055202/` (the tool's pre/post dumps and log).

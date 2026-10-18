# RadioMaster Pocket → Scout switch layout (goes with PARAM_CHANGESET_S6)

Link: ExpressLRS (the FC's receiver input is CRSF). ELRS sends **CH5 as an on/off channel in every packet** (most reliable,
but only 2 positions); CH6+ keep 3 positions in ELRS's default *Hybrid* switch mode. Hence the channel numbers below.

| Switch | Position | Does | Channel |
|---|---|---|---|
| **SD** (2-pos) | away from you (up) | normal | **CH5** |
| | toward you (down) | **KILL: motors stop instantly** (it falls; last resort) | |
| **SB** (3-pos) | up | **Position** (holds its spot; the normal mode) | **CH6** |
| | middle | **Altitude** (holds height only; drifts) | |
| | down | **Land** (descends where it is, disarms) | |
| **SA** (2-pos) | up | **Pilot** flies | **CH7** |
| | down | **Software** flies (offboard; the mission runs) | |
| **SC** (3-pos) | (later) | mission select, read by the laptop | **CH8** |
| Sticks | Mode 2 | roll CH1, pitch CH2, throttle CH3, yaw CH4 (unchanged) | |

Arming is unchanged: **throttle down + yaw right** (stick gesture). Before every flight: SD up, SB up, SA up.

## On the Pocket (EdgeTX), with the Scout's model selected

1. Press **MDL**, page to **MIXES**.
2. Delete whatever is on **CH5–CH8** (long-press a line → Delete).
3. Add one line per channel, weight 100, offset 0:
   - **CH5**: Source **SD**
   - **CH6**: Source **SB**
   - **CH7**: Source **SA**
   - **CH8**: Source **SC**
4. Check: **SYS → ExpressLRS** (Lua) → *Other Devices / Switch Mode* = **Hybrid** (the default; *Wide* also works).
5. Check on the radio's **Channel monitor**: SD up = −100, down = +100; SB up / mid / down = −100 / 0 / +100;
   SA up = −100, down = +100. If one is reversed, edit that mix line → **Weight −100**.

## Order of operations (props OFF, battery out or disarmed)

1. Change the radio mixes (above).
2. Apply S6 on the FC (Ron's go on the sha256).
3. Verify, read-only: move every switch and watch `listener manual_control_switches` on the FC (kill / mode / offboard
   reported as expected). Only then fly.

Until both are changed, the old CH5 switch (flight mode) would act as the new **kill** channel and vice versa.

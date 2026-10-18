# Wiring: Scout Radxa Zero 3W ↔ flight controller (TELEM2)

The FC is already configured for a companion on **TELEM2: uXRCE-DDS at 921600 baud** (`UXRCE_DDS_CFG` 102,
`SER_TEL2_BAUD` 921600). Only three wires are needed. **No 5 V wire:** the Radxa is already powered by its UBEC.

## Which FC pads are TELEM2

The FC runs PX4's `MICOAIR_H743_V2` target (`hardware/scout/fc_baseline_20261015/`). In that target (PX4 source,
`boards/micoair/h743-v2`), TELEM2 = `/dev/ttyS3`. NuttX numbers the enabled ports in peripheral order
(USART1, USART2, USART3, UART4, UART5, USART6, UART7, UART8 → ttyS0…ttyS7), so **TELEM2 = UART4 = the pads marked
R4 / T4** (or RX4 / TX4).

Careful: PX4's separate `h743-aio` target maps TELEM2 to USART2. The Scout's board is listed as "743 AIO" but runs the
V2 target, and the pad *labels* follow the UART number, so look for **R4 / T4** whichever board it physically is.

| PX4 port (V2 target) | UART | Pads |
|---|---|---|
| TELEM1 (MAVLink, `MAV_0_CONFIG` 101) | USART1 | R1 / T1 |
| GPS2 | USART2 | R2 / T2 |
| GPS1 | USART3 | R3 / T3 |
| **TELEM2 (uXRCE-DDS, 921600)** | **UART4** | **R4 / T4** |
| TELEM3 | UART5 | R5 / T5 |
| RC | USART6 | R6 |
| TELEM4 (MAVLink, `MAV_2_CONFIG` 104) | UART8 | R8 / T8 |

## Which Radxa pins

Radxa Zero 3W 40-pin header ([Radxa hardware interface page](https://docs.radxa.com/en/zero/zero3/hardware-design/hardware-interface),
[product brief](https://dl2.radxa.com/zero3/docs/hw/3w/radxa_zero_3w_product_brief.pdf)):

- **UART4 (M1): pin 16 = RX, pin 18 = TX.** Use this one.
- Not pins 8 / 10 (UART2): that is the Linux debug console (`ttyFIQ0`, `console=both`).
- Not pins 3 / 5 (UART3): extra I2C pull-ups on them, per Radxa.
- Ground: pin 6 (or 9, 14, 20, 25, 30, 34, 39).

## The three wires (TX goes to RX)

| FC pad | Radxa pin |
|---|---|
| **T4** (FC transmits) | **16** (UART4 RX) |
| **R4** (FC receives) | **18** (UART4 TX) |
| **GND** | **6** (GND) |

Both sides are 3.3 V logic (STM32H7 / RK3566). Double-check pins 16 / 18 against the Radxa page before soldering.

## After wiring (Claude does this, with the Radxa on Wi-Fi)

1. Enable the UART4-M1 overlay on the Radxa (Armbian `armbianEnv.txt` overlays) and reboot it.
2. Listen on the new serial port at 921600 and confirm the FC's uXRCE-DDS client traffic arrives, i.e. the wires go to the
   right pads and TX/RX are not swapped (swapped = silence; then swap the two signal wires).
3. Then the uXRCE-DDS link to the laptop.

## As built (2026-10-18)

The wires are on **Radxa pins 3 / 5 (UART3-M0)**, not 16 / 18: FC **T4 → Radxa pin 3 (UART3 RX)**, FC **R4 ← Radxa pin 5
(UART3 TX)**, GND. Found by watching every free header pin for signal edges (only pin 3 showed the FC's 1 Hz uXRCE-DDS
pings). On the Radxa: `/boot/armbianEnv.txt` `overlays=rk3568-uart3-m0` → `/dev/ttyS3` at 921600. Pins 3 / 5 are free here
(I2C3 uses its M1 pins on this board). FC → Radxa verified (ping frames received); Radxa → FC needs the uXRCE-DDS agent.
Details: `hardware/scout/access_20261018/README.md`. `config/scout_hw.yaml` `companion.fc_link` holds the facts.

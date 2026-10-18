# Noxen dock firmware (SKR Mini E3 V3, Marlin + 2 servos)

The board shipped with BTT's stock Ender-3 Marlin, which has **no servo
support** (`M280` → `Unknown command`). This build is upstream **Marlin
2.1.2.9** with Marlin's own example config
`Configurations release-2.1.2.9 / config/examples/Creality/Ender-3/BigTreeTech SKR Mini E3 3.0`.
It changes only what `noxen_dock_servos.diff` shows:

| Change | Why |
|---|---|
| `NUM_SERVOS 2` | servo 0 = **PA1** (5-pin PROBE header), servo 1 = **PA8** (3-pin NEOPIXEL header) |
| `SERVO1_PIN PA8` | the NEOPIXEL header is otherwise unused |
| `SERVO_DELAY { 300, 300 }` | one entry per servo (Marlin requires it) |
| `SERVO_DETACH_GCODE` | adds `M282 P<n>` to stop driving a servo |
| `DEFAULT_AXIS_STEPS_PER_UNIT` 8.8889 on all axes | factory default = degrees (200 × 16 / 360), so an EEPROM reset or `M502` lands in degrees |
| `DEFAULT_MAX_FEEDRATE` 360, `DEFAULT_MAX_ACCELERATION` / `DEFAULT_*_ACCELERATION` 720 | °/s and °/s², the same bench placeholders as the stock board was set to by hand |
| `CUSTOM_MACHINE_NAME "Noxen dock"` | `M115` shows `MACHINE_TYPE:Noxen dock`, so you can tell this build is flashed |
| `default_envs = STM32G0B1RE_btt` | the SKR Mini E3 V3's chip |

Everything else (directions, 580/650 mA, 16 microsteps, 115200 baud over
USB) matches the stock config.

## Flash it (microSD)

1. MicroSD card, **32 GB or smaller, FAT32**. Other files on it can stay.
2. Copy `firmware.bin` to the card's **root folder**. Keep the name exactly
   `firmware.bin`.
3. **Power off** the 24 V supply. Put the card in the board's microSD slot.
4. Power on. The board flashes itself in about 10 s (the status LED blinks),
   then renames the file to `FIRMWARE.CUR`. That rename is the proof it took.
5. Plug in USB and check: `M115` shows `MACHINE_TYPE:Noxen dock`.
6. **Check the settings.** A new firmware version discards the old EEPROM
   and starts from this build's defaults, which are already in degrees. Send
   `M502` then `M500` once to store them, and check that `M503` shows
   `M92 X8.89 Y8.89 Z8.89 E8.89`. Then re-check each motor's direction with
   `jog.py motorN 90` (all clockwise), and each servo with
   `jog.py servo1 90` / `jog.py servo2 90`.
7. Take the card out (or leave it). The board only reflashes when it finds a
   file named `firmware.bin`.

**Back to stock:** flash BTT's prebuilt `firmware.bin` for the SKR Mini E3 V3
the same way.

## Rebuild it

```bash
# Marlin 2.1.2.9 source + Configurations branch release-2.1.2.9 (GitHub, MarlinFirmware)
cp "Configurations/config/examples/Creality/Ender-3/BigTreeTech SKR Mini E3 3.0"/*.h Marlin/Marlin/
cd Marlin && patch -p1 < noxen_dock_servos.diff   # or apply the table above by hand
pip install platformio && pio run -e STM32G0B1RE_btt
# -> .pio/build/STM32G0B1RE_btt/firmware.bin
```

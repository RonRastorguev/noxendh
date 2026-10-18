# dock_motors: battery hot-swap steppers

Three NEMA 17 steppers on a BigTreeTech **SKR Mini E3 V3** (TMC2209 drivers,
stock BTT Marlin), driven over USB from **native Windows**, not WSL: WSL can't
see the COM port without usbipd passthrough.

## Setup (Windows, PowerShell)

```powershell
cd C:\Users\Ron\Desktop\noxen-dh26\dock_motors
py -3.9 -m venv .venv
.venv\Scripts\python -m pip install pyserial
.venv\Scripts\python -m serial.tools.list_ports -v   # board = VID:PID 0483:5740
.venv\Scripts\python jog.py --list
.venv\Scripts\python jog.py motor1 90                # +90 deg, clockwise
```

`board.py` auto-detects the port by USB ID (0483:5740, the V3's native USB, or
1A86:7523 for CH340 boards). It never sends G-code to any other serial device,
such as the PX4 flight controller (VID 3162). To pin a port, set `"port": "COM5"`
in `config.json`.

## What was found (2026-10-17)

| | |
|---|---|
| Port | COM5, USB 0483:5740 |
| Firmware | Marlin bugfix-2.0.x (Oct 11 2022), MACHINE_TYPE Ender-3, EEPROM on |
| Drivers | 4 × TMC2209 over UART, all answer (`M122`), **16 microsteps**, stealthChop on |
| Currents | `M906` X/Y/Z 580 mA, E 650 mA (RMS), BTT defaults, not changed |
| Steps/unit (stock) | X 80, Y 80, Z 400, E 93 (mm) |
| Steps/unit (now, saved) | **8.8889 on X/Y/Z/E: 1 unit = 1° of shaft** (200 × 16 / 360) |
| Speed limits (now, saved) | `M203` 360 °/s, `M201` / `M204` 720 °/s² on every axis. Conservative bench values I picked, not derived from the mechanism |

## Motor map

| Name | Header | Axis | `+` on the board | `invert` | `+` via board.py |
|---|---|---|---|---|---|
| motor1 | `XM` | X | clockwise | no | clockwise |
| motor2 | `YM` | Y | clockwise | no | clockwise |
| motor3 | `ZAM`/`ZBM` | Z | counter-clockwise | **yes** | clockwise |

Directions are as Ron saw them from the shaft end. Each motor was confirmed
with a +90° move through `jog.py` (800 microsteps each by Marlin's counter).
`EM` is free. Rename the motors (lift / slide / clamp …) in `config.json`.
`invert` stays with the **axis**, so re-check it if a cable moves header.

## API

```python
from board import Board
with Board() as b:              # connect(): opens, checks M115 says Marlin
    b.prepare()                 # M211 S0, M302 P1, G91: resend after every reset
    b.rotate("motor1", 90)      # degrees, + = CW; blocks until done (M400)
    b.rotate("motor3", -45, speed=10)   # deg/s
    b.set_current("motor2", 800)        # mA RMS (M906), b.save() to keep it
    b.set_idle_timeout(0)       # keep holding torque forever (M84 S0)
    b.enable(); b.disable()     # M17 / M84
```

## Servos (2 × MG995): needs the Noxen dock firmware

Stock BTT firmware has no servo support. `firmware/` holds a Marlin build
with 2 servos and the steps to flash it from a microSD card
(`firmware/README.md`). **Flashed 2026-10-17:** M115 reports `Marlin 2.1.2.9`,
`MACHINE_TYPE:Noxen dock`, and `M280 P0` / `M280 P1` answer. Degree settings
are stored (M500), and all three steppers turned +90° clockwise again on it.
Both servos swung 90 → 70 → 110 → 90 on command, as Ron saw.

| Name | Marlin index | Board pin | Header |
|---|---|---|---|
| servo1 | 0 | PA1 | 5-pin **PROBE** header, servo signal pin |
| servo2 | 1 | PA8 | 3-pin **NEOPIXEL** header, data pin |

Wiring: only the servo's **orange signal wire** goes to the board. Red/brown
come from the **UBEC** (input 7–34 V from the 24 V PSU, output jumper on 5/6 V;
the MG995 takes 4.8–7.2 V). The servo ground must be common with the SKR's
GND, or the signal has no reference. Here it already is, through the PSU: the
UBEC is non-isolated (input − = output −), and its input − and the SKR's power
− both sit on PSU −. So no separate ground wire is needed. With an isolated
supply, add a wire from servo − to an SKR GND pin. Never power the servos from
the board's 5 V: an MG995 stalls at ~2.5 A and would brown out the board.
PSU: 24 V, 3 A limit.

```python
b.servo("servo1", 90)        # ABSOLUTE angle 0..180 (M280), waits 0.5 s
b.servo_release("servo1")    # stop pulses, servo goes limp (M282)
```
```
python jog.py servo1 90
python jog.py servo1 --release
```
Marlin maps 0–180 onto 544–2400 µs pulses, wider than many MG995s travel.
Start at 90 and creep toward the ends. A servo buzzing at an end is pushing
against its own stop: narrow `servo_range_deg` in `config.json`.

`send(line)` sends exactly one line and waits for `ok`. "busy" keep-alives
extend the wait, and `Error:` / unknown commands raise `BoardError`.

## Swap wheel (`swap.py`, `mechanism.json`)

The battery swap from the **Nox Box Mechanism** artifact
(claude.ai/code/artifact/cfe4560d-d81c-44b6-9ca7-2e51a6b592ad). The wheel turns
on a horizontal axle. Head A points at φ and head B at φ + 90° (90° = the top,
under the drone seat). Each head slides along its bar (park 32 / bay 84.6 /
wait 81 / drone 89.6 mm), and each head's servo opens or closes two blades on
the cartridge tab. There are 9 fixed bays at 30°.

| Role | Default | Conversion |
|---|---|---|
| wheel | motor1 | 4 motor° per wheel° (20T on the fixed 80T sun) |
| head A / B | motor2 / motor3 | 40 mm per motor turn (20T GT2, **inferred**) |
| gripper A / B | servo1 / servo2 | open 66° / closed 114° (**provisional**) |

The **roles, directions, belt and gripper angles are not confirmed** on the
real mechanism. Everything lives in `mechanism.json`, with its source noted.

```
python swap.py zero --phi 0          # mechanism placed by hand: A at bay 0, heads parked
python swap.py status
python swap.py wheel 90 / head A bay / grip A closed
python swap.py swap --charged-bay 0 --return-bay 0 --step      # Enter before every step
python swap.py --dry-run swap --charged-bay 0 --return-bay 0   # no board, prints the moves
python test_swap.py                                            # offline checks
```

- **No endstops.** Position is dead-reckoned from the pose you declare with
  `zero` and saved in `state.json` after every finished move. A move that's
  interrupted (USB unplugged, Ctrl-C, error) leaves the state "unknown" until
  the next `zero`. Motors are kept energised (`M84 S0`), because a released
  motor can be pushed and lose its position.
- **Interlocks:** the wheel turns only with both heads parked. A head extends
  only when pointing at a bay (up to the bay depth) or the top (up to the
  drone). φ stays within −300…90°, exactly what the bays and the top need
  (cable wrap; narrow it if the cables have less slack).
- **The drone side isn't sensed** (seat switch, magnet, pogo pads). The swap
  stops at "drone seated" and "drone gone" and waits for Enter (`--no-wait`
  on the bench).
- **Bench run 2026-10-17:** the full swap (`--no-wait`) ran twice on the
  loose motors and servos, about 40 s each, with no errors, and Ron saw the
  expected pattern. It hasn't run in the assembled mechanism: confirm the
  roles, directions, belt and gripper angles there first, with single
  `wheel` / `head` / `grip` moves.
- The sequence is the artifact's, generalised to any charged and return bay,
  and run one move at a time. The artifact overlaps "A fetches" with "B rises
  under the seat"; that overlap is not used here.

## Gotchas

- **Never home.** There are no endstops. `send()` refuses G28/G29/G30/G33/G34.
- **No heater commands.** The thermistor inputs are open (they read −15 °C).
  M104/M109/M140/M190/M303 would trip MINTEMP and halt the board, so `send()`
  refuses them.
- **Unplug or plug motors only with the 12–24 V supply OFF.** Hot-plugging can
  kill a TMC2209.
- **`ZAM` and `ZBM` share one driver.** Two motors on them always move
  together, each at about half current. Use one Z header per motor.
- **One program per COM port.** Close Pronterface, Cura, Arduino IDE, etc.
- **`G1`'s ok means queued, not finished.** `rotate()` follows it with `M400`.
- **After M92, the speed caps are in °/s.** The stock Z cap (5 units/s) would
  become 5 °/s. Re-check `M203`/`M201` whenever M92 changes.
- **Session settings reset on reboot.** M211 S0, M302 P1 and G91 aren't
  persistent, so call `prepare()` after every connect (jog.py does). The board
  rebooted once while cables were being moved.
- **Motors cut current after 120 s idle** (Marlin default) and lose holding
  torque. Use `set_idle_timeout(0)` for a clamp that must hold.
- **Direction:** Marlin's direction invert is compile-time only, so it's done
  per motor with `invert` in `config.json`.
- **Resolution:** 3200 microsteps/rev = 0.1125° per step. Odd angles round to
  the nearest step.
- `signature.py` wiggles X 1×, Y 2×, Z 3×, E 4× so a motor can be identified
  by counting.

## Open

- The motors' rated current is unknown. 580 mA is BTT's default and may be
  well under what the motors can take (torque). Set it with `set_current` once
  the datasheet value is known.
- The speed/acceleration caps above are placeholders until the mechanism's
  loads are known.

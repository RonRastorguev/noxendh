"""Serial wrapper for the dock's BigTreeTech SKR Mini E3 (stock Marlin, TMC2209).

Rules this class enforces:
  - one G-code line at a time, each waits for "ok" before the next is sent;
  - no homing command is ever sent (there are no endstops): G28/G29 raise;
  - a COM port is auto-selected only when its USB ID is a known SKR ID, so it
    never sends G-code to some other serial device (e.g. the PX4 flight
    controller, VID 0x3162).

Units: after setup (README, step 7) one axis unit = one degree of motor-shaft
rotation, so rotate(name, 90) turns that shaft 90 degrees.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import serial
from serial.tools import list_ports

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"

# USB IDs of SKR Mini E3 boards: native USB CDC on the STM32 (V2 / V3), and
# the CH340 bridge on V1.x. Anything else is never auto-selected.
SKR_USB_IDS = {(0x0483, 0x5740), (0x1A86, 0x7523)}

AXES = ("X", "Y", "Z", "E")
# Homing/probing (no endstops) and heater commands (no thermistors: the open
# inputs read -15 C, so asking for heat trips MINTEMP and halts the board).
FORBIDDEN = re.compile(r"^\s*(G28|G29|G30|G33|G34|G425|M48|M104|M109|M140|M190|M303)\b", re.IGNORECASE)


class BoardError(RuntimeError):
    pass


def load_config(path: Path = CONFIG_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def find_port() -> str:
    """Return the single attached port whose USB ID is a known SKR ID."""
    hits = [p.device for p in list_ports.comports() if (p.vid, p.pid) in SKR_USB_IDS]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise BoardError("no SKR board found (is the USB cable a data cable and the board powered?); "
                         "set \"port\" in config.json to override")
    raise BoardError(f"several SKR-like ports {hits}; set \"port\" in config.json")


class Board:
    def __init__(self, port: str | None = None, baud: int | None = None,
                 names: dict | None = None, config: dict | None = None):
        cfg = load_config() if config is None else config
        self.port = port or cfg.get("port")
        self.baud = baud or cfg.get("baud", 115200)
        # motors: {"lift": {"axis": "X", "invert": false}, ...} (or just
        # {"lift": "X"}); axis letters always work as names too. invert flips
        # the sign so +degrees turns every motor the same way (README).
        self.names, self.inverted = {}, set()
        for k, v in (names or cfg.get("motors", {})).items():
            v = {"axis": v} if isinstance(v, str) else v
            self.names[k.lower()] = v["axis"].upper()
            if v.get("invert"):
                self.inverted.add(v["axis"].upper())
        self.default_speed = float(cfg.get("default_speed_deg_s", 30))
        # servos: {"latch": 0, ...} -> Marlin servo index (M280 P<index>);
        # needs the Noxen dock firmware (firmware/README.md), stock has none.
        self.servos = {k.lower(): int(v) for k, v in cfg.get("servos", {}).items()}
        self.servo_range = tuple(cfg.get("servo_range_deg", [0, 180]))
        self.ser: serial.Serial | None = None
        self.banner: list[str] = []
        self.firmware = ""
        self.verbose = False

    # ---- connection -------------------------------------------------------
    def connect(self, reset_wait: float = 2.0) -> "Board":
        port = self.port or find_port()
        self.ser = serial.Serial(port, self.baud, timeout=0.1, write_timeout=2)
        self.port = port
        # Opening the port resets some boards; collect the banner, then drain.
        deadline = time.time() + reset_wait
        while time.time() < deadline:
            line = self._readline()
            if line:
                self.banner.append(line)
        self.firmware = " ".join(self.send("M115"))
        if "FIRMWARE_NAME:Marlin" not in self.firmware:
            raise BoardError(f"not Marlin on {port}: {self.firmware!r}")
        return self

    def prepare(self) -> None:
        """Session setup: no soft limits, cold E moves allowed, relative mode.
        None of these are homing; they must be resent after every reset."""
        self.send("M211 S0")
        self.send("M302 P1")
        self.send("G91")

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # ---- protocol ---------------------------------------------------------
    def _readline(self) -> str:
        raw = self.ser.readline()
        line = raw.decode("ascii", errors="replace").strip()
        if line and self.verbose:
            print("<", line)
        return line

    def send(self, line: str, timeout: float = 10.0) -> list[str]:
        """Send one line and block until Marlin answers "ok". Returns the
        lines received before the ok. "busy" keep-alives extend the wait."""
        if self.ser is None:
            raise BoardError("not connected")
        line = line.split(";")[0].strip()
        if not line:
            return []
        if "\n" in line:
            raise BoardError("one G-code line at a time")
        if FORBIDDEN.match(line):
            raise BoardError(f"refused {line.split()[0]}: no homing/probing (no endstops) or heaters (no thermistors)")
        if self.verbose:
            print(">", line)
        self.ser.reset_input_buffer()
        self.ser.write((line + "\n").encode("ascii"))
        out: list[str] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            resp = self._readline()
            if not resp:
                continue
            if resp.startswith("ok"):
                if resp[2:].strip():  # e.g. M105 answers "ok T:..." on one line
                    out.append(resp[2:].strip())
                return out
            if "busy" in resp:
                deadline = time.time() + timeout
                continue
            if resp.startswith("Error") or "Unknown command" in resp:
                raise BoardError(f"{line!r} -> {resp}")
            out.append(resp)
        raise BoardError(f"no ok for {line!r} within {timeout:.0f} s (got {out})")

    # ---- motors -----------------------------------------------------------
    def axis(self, name: str) -> str:
        a = self.names.get(name.lower(), name.upper())
        if a not in AXES:
            raise BoardError(f"unknown motor {name!r}; known: {sorted(self.names) or list(AXES)}")
        return a

    def rotate(self, name: str, degrees: float, speed: float | None = None) -> None:
        """Turn one motor shaft by `degrees` (signed; + = clockwise seen from
        the shaft end) at `speed` deg/s and return when the move has finished
        (M400)."""
        a = self.axis(name)
        speed = float(speed or self.default_speed)
        if speed <= 0:
            raise BoardError("speed must be > 0")
        sign = -1 if a in self.inverted else 1
        self.send("G91")
        self.send(f"G1 {a}{sign * degrees:.3f} F{speed * 60:.1f}")
        # G1's ok means "queued"; M400's ok means "finished". Marlin caps the
        # feedrate at M203, so allow for a slower move than requested.
        self.send("M400", timeout=abs(degrees) / speed * 3 + 10)

    def enable(self, *names: str) -> None:
        self.send("M17 " + " ".join(self.axis(n) for n in names) if names else "M17")

    def disable(self, *names: str) -> None:
        self.send("M84 " + " ".join(self.axis(n) for n in names) if names else "M84")

    def set_current(self, name: str, milliamps: int) -> None:
        """TMC2209 RMS run current (M906). Not saved until save()."""
        self.send(f"M906 {self.axis(name)}{int(milliamps)}")

    def set_idle_timeout(self, seconds: int) -> None:
        """Seconds of inactivity before Marlin cuts motor current (M84 S);
        0 keeps the motors energised (holding torque) indefinitely."""
        self.send(f"M84 S{int(seconds)}")

    # ---- servos -----------------------------------------------------------
    def servo_index(self, name: str) -> int:
        if name.lower() in self.servos:
            return self.servos[name.lower()]
        if name.isdigit():
            return int(name)
        raise BoardError(f"unknown servo {name!r}; known: {sorted(self.servos) or ['0', '1']}")

    def servo(self, name: str, angle: float, settle: float = 0.5) -> None:
        """Command a hobby servo to `angle` degrees (M280) and wait `settle`
        seconds: Marlin answers ok at once, the servo reports nothing back."""
        lo, hi = self.servo_range
        if not lo <= angle <= hi:
            raise BoardError(f"servo angle {angle} outside {lo}..{hi} (servo_range_deg in config.json)")
        self._servo_cmd(f"M280 P{self.servo_index(name)} S{int(round(angle))}")
        time.sleep(settle)

    def _servo_cmd(self, line: str) -> None:
        try:
            self.send(line)
        except BoardError as e:
            if "Unknown command" in str(e):
                raise BoardError("this firmware has no servo support; flash "
                                 "firmware/firmware.bin (firmware/README.md)") from e
            raise

    def servo_release(self, name: str) -> None:
        """Stop sending pulses (M282): the servo goes limp and stops drawing
        holding current. The next servo() call re-attaches it."""
        self._servo_cmd(f"M282 P{self.servo_index(name)}")

    def save(self) -> None:
        self.send("M500")

    def settings(self) -> list[str]:
        return self.send("M503")

"""Identify which board header drives which motor.

Each axis wiggles a different number of times (X 1, Y 2, Z 3, E 4), a pause
between axes. One wiggle = a quarter turn out and back, so every motor ends
where it started. Count the wiggles on each motor: that count names its axis.

    python signature.py              # all axes
    python signature.py Z E          # only some
"""
import sys
import time

from board import Board

# A quarter turn is 800 microsteps (16x microstepping), whatever M92 says.
COUNT = {"X": 1, "Y": 2, "Z": 3, "E": 4}


def main() -> int:
    axes = [a.upper() for a in sys.argv[1:]] or list(COUNT)
    b = Board().connect()
    try:
        b.prepare()
        print("starting in 5 s", flush=True)
        time.sleep(5)
        m92 = next(l for l in b.settings() if "M92" in l)
        steps = {a: float(m92.split(a)[1].split()[0]) for a in COUNT}
        for a in axes:
            quarter = 800 / steps[a]          # 800 microsteps at 16x = 90 deg
            feed = min(600.0, quarter * 60)   # ~1 s per quarter turn, never above F600
            print(f"{time.strftime('%H:%M:%S')}  {a}: {COUNT[a]} wiggle(s)", flush=True)
            for _ in range(COUNT[a]):
                b.send(f"G1 {a}{quarter:.3f} F{feed:.0f}"); b.send("M400", timeout=30)
                b.send(f"G1 {a}{-quarter:.3f} F{feed:.0f}"); b.send("M400", timeout=30)
                time.sleep(0.6)
            time.sleep(4)
    finally:
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

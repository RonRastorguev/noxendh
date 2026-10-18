"""Offline checks of swap.py (no board): interlocks, unit conversion, and the
swap sequence's net motor motion.   python test_swap.py"""
import json
import tempfile
from pathlib import Path

from swap import MECH_PATH, SwapError, Wheel


class Recorder:
    def __init__(self):
        self.moves, self.servos = [], []

    def rotate(self, name, deg, speed=None):
        self.moves.append((name, deg, speed))

    def servo(self, name, angle, settle=0.5):
        self.servos.append((name, angle))


def wheel(rec):
    mech = json.loads(MECH_PATH.read_text(encoding="utf-8"))
    path = Path(tempfile.mkdtemp()) / "state.json"
    w = Wheel(rec, mech, path, log=lambda m: None)
    return w, mech, path


def refuses(fn, text):
    try:
        fn()
    except SwapError as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"expected refusal containing {text!r}")


def test_needs_zero():
    w, _, _ = wheel(Recorder())
    refuses(lambda: w.wheel(90), "position unknown")


def test_conversions():
    rec = Recorder()
    w, m, _ = wheel(rec)
    w.zero(0)
    w.wheel(90)                                    # 90 wheel deg x 4 = 360 motor deg
    assert rec.moves[-1][0] == m["roles"]["wheel"] and abs(rec.moves[-1][1] - 360) < 1e-9
    w.head("A", "drone")                           # 89.6 - 32 = 57.6 mm / 40 mm per rev = 518.4 deg
    assert abs(rec.moves[-1][1] - 57.6 / 40 * 360) < 1e-9


def test_interlocks():
    w, _, _ = wheel(Recorder())
    w.zero(0)
    w.head("A", "bay")                             # A points at bay 0
    refuses(lambda: w.wheel(90), "not parked")     # wheel locked while a head is out
    refuses(lambda: w.head("A", "drone"), "max")   # past the bay depth
    w.head("A", "park")
    w.wheel(15)                                    # between bays
    refuses(lambda: w.head("A", "bay"), "neither a bay nor the top")
    refuses(lambda: w.wheel(120), "outside the limits")


def test_interrupted_move_needs_rezero():
    class Dies(Recorder):
        def rotate(self, *a, **k):
            raise OSError("USB unplugged")
    w, mech, path = wheel(Dies())
    w.zero(0)
    try:
        w.wheel(90)
    except OSError:
        pass
    w2 = Wheel(Recorder(), mech, path, log=lambda m: None)   # a fresh process reads the state
    refuses(lambda: w2.wheel(0), "interrupted")


def test_swap_sequence():
    rec = Recorder()
    w, m, _ = wheel(rec)
    w.zero(0)
    w.swap(charged_bay=0, return_bay=0)
    # every head move out is matched by one back: net head motion zero
    for h in ("head_A", "head_B"):
        assert abs(sum(d for n, d, _ in rec.moves if n == m["roles"][h])) < 1e-6
    # wheel: 0 -> 0 (A at bay 0) -> 0 (B top) -> 90 (A top) -> -90 (B at bay 0)
    assert abs(sum(d for n, d, _ in rec.moves if n == m["roles"]["wheel"]) - (-90 * 4)) < 1e-6
    assert w.state["phi"] == -90 and w.parked("A") and w.parked("B")
    assert w.state["grip"] == {"A": "closed", "B": "closed"}


def test_swap_other_bays():
    w, _, _ = wheel(Recorder())
    w.zero(0)
    w.swap(charged_bay=7, return_bay=8)            # bay 7 = -210 deg, bay 8 = +30 deg
    assert w.state["phi"] == 30 - 90


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)

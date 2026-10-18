"""Nox Box swap wheel: the battery swap from the mechanism artifact, on the
SKR Mini E3 V3 (board.py).

Mechanism (artifact "Nox Box Mechanism"; numbers in mechanism.json):
  - the wheel turns about a horizontal axle; phi = where head A points
    (degrees, 90 = the top, under the drone seat). Head B points at phi + 90.
  - each head slides along its bar (extension in mm): park / bay / wait / drone.
  - each head's servo opens or closes two blades on the cartridge tab.
  - 9 fixed bays at bays_deg; a head reaches a bay when it points at it.

There are no endstops, so position is dead-reckoned from a pose you declare
with `zero` (mechanism placed there by hand) and saved in state.json after
every finished move. An interrupted move marks the state unknown until the
next `zero`.

Interlocks (refuse, never "try"):
  - the wheel turns only with BOTH heads at park;
  - a head goes past park only when it points at a bay (bay) or the top
    (wait / drone), and never beyond the drone extension;
  - phi stays inside phi_limits_deg (cable wrap).

    python swap.py status
    python swap.py zero --phi 0                  # mechanism is at phi 0, heads parked
    python swap.py wheel 90                      # turn so head A points up
    python swap.py head A bay                    # or park / wait / drone / a number in mm
    python swap.py grip A closed                 # or open
    python swap.py swap --charged-bay 0 --return-bay 0 --step
    python swap.py swap --charged-bay 0 --return-bay 0 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from board import Board, BoardError

HERE = Path(__file__).resolve().parent
MECH_PATH = HERE / "mechanism.json"
STATE_PATH = HERE / "state.json"
EPS = 0.05


class SwapError(RuntimeError):
    pass


def ang_eq(a: float, b: float) -> bool:
    return abs((a - b + 180) % 360 - 180) < EPS


class DryBoard:
    """Stands in for Board: prints what would be sent, moves nothing."""

    def __init__(self, log):
        self.log = log

    def connect(self):
        return self

    def prepare(self):
        pass

    def close(self):
        pass

    def set_idle_timeout(self, s):
        self.log(f"    [dry] M84 S{s}")

    def rotate(self, name, deg, speed=None):
        self.log(f"    [dry] {name}: {deg:+.1f} motor deg at {speed:.0f} deg/s")

    def servo(self, name, angle, settle=0.5):
        self.log(f"    [dry] {name} -> {angle:g} deg")


class Wheel:
    def __init__(self, board, mech: dict, state_path: Path = STATE_PATH, log=print):
        self.b, self.m, self.state_path, self.log = board, mech, state_path, log
        self.roles = mech["roles"]
        self.ext = mech["heads"]["ext_mm"]
        self.state = self._load()
        self._check_speeds()

    # ---- state -------------------------------------------------------------
    def _load(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"zeroed": False}

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    def zero(self, phi: float, ext_a: float | None = None, ext_b: float | None = None) -> None:
        """Declare the current physical pose (manual homing)."""
        lo, hi = self.m["wheel"]["phi_limits_deg"]
        if not lo <= phi <= hi:
            raise SwapError(f"phi {phi} outside the limits {lo}..{hi}")
        park = self.ext["park"]
        self.state = {"zeroed": True, "moving": None, "phi": phi,
                      "ext": {"A": park if ext_a is None else ext_a, "B": park if ext_b is None else ext_b},
                      "grip": {"A": None, "B": None}}
        self._save()
        self.log(f"zeroed: phi {phi}, A {self.state['ext']['A']} mm, B {self.state['ext']['B']} mm")

    def _ready(self) -> None:
        if not self.state.get("zeroed"):
            raise SwapError("position unknown: place the mechanism and run `swap.py zero`")
        if self.state.get("moving"):
            raise SwapError(f"a move was interrupted ({self.state['moving']}): position unknown, "
                            "check the mechanism by eye and run `swap.py zero` again")

    def _move(self, label: str, motor: str, motor_deg: float, speed_motor: float, apply) -> None:
        self.state["moving"] = label
        self._save()
        self.b.rotate(motor, motor_deg, speed_motor)
        apply()
        self.state["moving"] = None
        self._save()

    def _check_speeds(self) -> None:
        cap = self.m["motor_max_deg_s"]
        w = self.m["wheel"]["speed_deg_s"] * self.m["wheel"]["motor_deg_per_deg"]
        h = self.m["heads"]["speed_mm_s"] / self.m["heads"]["mm_per_motor_rev"] * 360
        for what, v in (("wheel", w), ("head", h)):
            if v > cap:
                raise SwapError(f"{what} speed needs {v:.0f} motor deg/s, above the board's {cap} (M203)")

    # ---- geometry ----------------------------------------------------------
    def head_dir(self, head: str, phi: float | None = None) -> float:
        phi = self.state["phi"] if phi is None else phi
        return phi + (0 if head == "A" else 90)

    def bay_of(self, head: str) -> int | None:
        d = self.head_dir(head)
        for i, a in enumerate(self.m["bays_deg"]):
            if ang_eq(d, a):
                return i
        return None

    def at_top(self, head: str) -> bool:
        return ang_eq(self.head_dir(head), self.m["wheel"]["top_deg"])

    def parked(self, head: str) -> bool:
        return abs(self.state["ext"][head] - self.ext["park"]) < EPS

    # ---- moves -------------------------------------------------------------
    def wheel(self, phi: float) -> None:
        self._ready()
        lo, hi = self.m["wheel"]["phi_limits_deg"]
        if not lo <= phi <= hi:
            raise SwapError(f"phi {phi} outside the limits {lo}..{hi} (cable wrap)")
        for h in "AB":
            if not self.parked(h):
                raise SwapError(f"head {h} is not parked ({self.state['ext'][h]} mm): the wheel may not turn")
        delta = phi - self.state["phi"]
        if abs(delta) < EPS:
            return
        w = self.m["wheel"]
        self.log(f"  wheel {self.state['phi']:g} -> {phi:g} deg")

        def apply():
            self.state["phi"] = phi
        self._move(f"wheel to {phi}", self.roles["wheel"], delta * w["motor_deg_per_deg"] * w["dir"],
                   w["speed_deg_s"] * w["motor_deg_per_deg"], apply)

    def head(self, head: str, target) -> None:
        self._ready()
        mm = float(self.ext[target]) if isinstance(target, str) else float(target)
        park, top = self.ext["park"], self.ext["drone"]
        if not park - EPS <= mm <= top + EPS:
            raise SwapError(f"head {head} target {mm} mm outside park..drone ({park}..{top})")
        if mm > park + EPS:
            bay = self.bay_of(head)
            if self.at_top(head):
                pass  # wait / drone, or anything up to the drone extension
            elif bay is not None:
                if mm > self.ext["bay"] + EPS:
                    raise SwapError(f"head {head} points at bay {bay}: max {self.ext['bay']} mm")
            else:
                raise SwapError(f"head {head} points at {self.head_dir(head):g} deg: neither a bay nor the top")
        delta = mm - self.state["ext"][head]
        if abs(delta) < EPS:
            return
        hd = self.m["heads"]
        self.log(f"  head {head} {self.state['ext'][head]:g} -> {mm:g} mm")

        def apply():
            self.state["ext"][head] = mm
        self._move(f"head {head} to {mm}", self.roles[f"head_{head}"],
                   delta / hd["mm_per_motor_rev"] * 360 * hd["dir"][head],
                   hd["speed_mm_s"] / hd["mm_per_motor_rev"] * 360, apply)

    def grip(self, head: str, how: str) -> None:
        g = self.m["grippers"]
        angle = (g["closed_deg"] if how == "closed" else g["open_deg"])[head]
        self.log(f"  grip {head} {how} ({angle} deg{', PROVISIONAL angle' if g.get('provisional') else ''})")
        self.b.servo(self.roles[f"grip_{head}"], angle, settle=g["settle_s"])
        if self.state.get("zeroed"):
            self.state["grip"][head] = how
            self._save()

    # ---- the swap ------------------------------------------------------------
    def swap(self, charged_bay: int, return_bay: int, pause=None, wait_for=None) -> None:
        """One swap, the artifact's sequence generalised to any two bays:
        A fetches the charged pack, B pulls the dead one out of the drone,
        A pushes the charged one in, B parks the dead one in return_bay.
        pause(step) runs before every step; wait_for(event) blocks until the
        drone side is ready (seated / gone), which nothing here can sense."""
        bays = self.m["bays_deg"]
        for i in (charged_bay, return_bay):
            if not 0 <= i < len(bays):
                raise SwapError(f"bay {i}: there are bays 0..{len(bays) - 1}")
        self._ready()
        pause = pause or (lambda s: None)
        wait_for = wait_for or (lambda e: None)
        top = self.m["wheel"]["top_deg"]
        cb, rb = bays[charged_bay], bays[return_bay]
        steps = [
            ("both heads to park", lambda: (self.head("A", "park"), self.head("B", "park"))),
            (f"turn A to bay {charged_bay} ({cb} deg)", lambda: self.wheel(cb)),
            ("A opens", lambda: self.grip("A", "open")),
            (f"A extends into bay {charged_bay}", lambda: self.head("A", "bay")),
            ("A grips the charged pack", lambda: self.grip("A", "closed")),
            ("A retracts with it", lambda: self.head("A", "park")),
            ("turn B to the top, under the seat", lambda: self.wheel(top - 90)),
            ("B opens", lambda: self.grip("B", "open")),
            ("B rises to wait under the seat", lambda: self.head("B", "wait")),
            ("drone seated (switch, magnet on, pogo pads)", lambda: wait_for("drone seated")),
            ("B reaches the dead pack", lambda: self.head("B", "drone")),
            ("B grips the dead pack", lambda: self.grip("B", "closed")),
            ("B retracts: bolts leave the drone pockets", lambda: self.head("B", "park")),
            ("turn A to the top", lambda: self.wheel(top)),
            ("A inserts the charged pack up into the drone", lambda: self.head("A", "drone")),
            ("A releases: latched, contacts make", lambda: self.grip("A", "open")),
            ("A retracts", lambda: self.head("A", "park")),
            ("A closes; magnet off, drone may leave", lambda: self.grip("A", "closed")),
            ("drone gone", lambda: wait_for("drone gone")),
            (f"turn B to bay {return_bay} ({rb} deg)", lambda: self.wheel(rb - 90)),
            (f"B extends into bay {return_bay}", lambda: self.head("B", "bay")),
            ("B releases into the bay latch; charging starts", lambda: self.grip("B", "open")),
            ("B retracts", lambda: self.head("B", "park")),
            ("B closes: ready for the next landing", lambda: self.grip("B", "closed")),
        ]
        for n, (name, fn) in enumerate(steps, 1):
            pause(f"step {n}/{len(steps)}: {name}")
            self.log(f"step {n}/{len(steps)}: {name}")
            fn()
        self.log("swap done")

    def status(self) -> str:
        s = self.state
        if not s.get("zeroed"):
            return "position unknown (run `swap.py zero`)"
        out = [f"phi {s['phi']:g} deg" + (f"   INTERRUPTED: {s['moving']}" if s.get("moving") else "")]
        for h in "AB":
            bay = self.bay_of(h)
            where = "top" if self.at_top(h) else (f"bay {bay}" if bay is not None else f"{self.head_dir(h):g} deg")
            out.append(f"head {h}: {s['ext'][h]:g} mm, points at {where}, grip {s['grip'][h]}")
        return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="no board: print the motor moves only")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    z = sub.add_parser("zero", help="declare the current pose (place the mechanism first)")
    z.add_argument("--phi", type=float, required=True)
    z.add_argument("--ext-a", type=float)
    z.add_argument("--ext-b", type=float)
    w = sub.add_parser("wheel"); w.add_argument("phi", type=float)
    h = sub.add_parser("head"); h.add_argument("head", choices="AB"); h.add_argument("target")
    g = sub.add_parser("grip"); g.add_argument("head", choices="AB"); g.add_argument("how", choices=["open", "closed"])
    s = sub.add_parser("swap")
    s.add_argument("--charged-bay", type=int, required=True)
    s.add_argument("--return-bay", type=int, required=True)
    s.add_argument("--step", action="store_true", help="press Enter before every step")
    s.add_argument("--no-wait", action="store_true", help="don't stop for 'drone seated' / 'drone gone' (bench)")
    args = ap.parse_args()

    mech = json.loads(MECH_PATH.read_text(encoding="utf-8"))
    if args.dry_run:
        board = DryBoard(print)
        state_path = HERE / "state.dry.json"
    else:
        board, state_path = Board(), STATE_PATH
    wh = Wheel(board, mech, state_path)
    if args.dry_run and args.cmd == "swap":
        wh.zero(0.0)   # a dry run always starts from the artifact's pose

    if args.cmd == "status":
        print(wh.status())
        return 0
    if args.cmd == "zero":
        wh.zero(args.phi, args.ext_a, args.ext_b)
        return 0
    try:
        board.connect()
        board.prepare()
        board.set_idle_timeout(0)   # hold torque: a released motor loses the dead-reckoned position
        if args.cmd == "wheel":
            wh.wheel(args.phi)
        elif args.cmd == "head":
            t = args.target
            wh.head(args.head, t if t in mech["heads"]["ext_mm"] else float(t))
        elif args.cmd == "grip":
            wh.grip(args.head, args.how)
        elif args.cmd == "swap":
            pause = (lambda m: input(f"{m}  [Enter] ")) if args.step else None
            wait = None if args.no_wait or args.dry_run else (lambda e: input(f"waiting for: {e}  [Enter when true] "))
            wh.swap(args.charged_bay, args.return_bay, pause, wait)
        print(wh.status())
        return 0
    except (SwapError, BoardError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        print(wh.status(), file=sys.stderr)
        return 1
    finally:
        board.close()


if __name__ == "__main__":
    sys.exit(main())

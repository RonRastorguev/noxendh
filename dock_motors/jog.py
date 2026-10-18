"""Rotate one dock motor by N degrees, or set a servo's angle, for testing.

    python jog.py lift 90            # +90 deg at the default speed
    python jog.py clamp -45 --speed 10
    python jog.py X 10 --port COM5   # axis letters work too
    python jog.py servo1 90          # servo to ABSOLUTE 90 deg (servos in config.json)
    python jog.py servo1 --release   # stop driving that servo (M282)
    python jog.py --list             # motor names -> axes, servo names -> index
    python jog.py --off              # de-energise all motors (M84)
"""
import argparse
import sys

from board import Board, BoardError


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("motor", nargs="?", help="motor name from config.json, or X/Y/Z/E")
    ap.add_argument("degrees", nargs="?", type=float, help="signed shaft rotation in degrees")
    ap.add_argument("--speed", type=float, help="deg/s (default from config.json)")
    ap.add_argument("--port", help="COM port (default: config.json, else auto-detect)")
    ap.add_argument("--list", action="store_true", help="print the motor-name -> axis map")
    ap.add_argument("--off", action="store_true", help="disable all motors and exit")
    ap.add_argument("--release", action="store_true", help="servo only: stop driving it (M282)")
    ap.add_argument("-v", "--verbose", action="store_true", help="echo the serial traffic")
    args = ap.parse_args()

    b = Board(port=args.port)
    if args.list:
        for name, axis in sorted(b.names.items()):
            print(f"{name:10s} {axis}{'  (inverted)' if axis in b.inverted else ''}")
        for name, index in sorted(b.servos.items()):
            print(f"{name:10s} servo {index}")
        return 0
    is_servo = args.motor is not None and args.motor.lower() in b.servos
    if is_servo and args.release:
        pass
    elif not args.off and (args.motor is None or args.degrees is None):
        ap.error("give a motor and a number of degrees, or --list / --off")

    b.verbose = args.verbose
    try:
        b.connect()
        if args.off:
            b.disable()
            print(f"{b.port}: motors off")
            return 0
        if is_servo:
            if args.release:
                b.servo_release(args.motor)
                print(f"{b.port}: {args.motor} released")
            else:
                b.servo(args.motor, args.degrees)
                print(f"{b.port}: {args.motor} (servo {b.servo_index(args.motor)}) -> {args.degrees:g} deg")
            return 0
        b.prepare()
        b.rotate(args.motor, args.degrees, args.speed)
        print(f"{b.port}: {args.motor} ({b.axis(args.motor)}) rotated {args.degrees:+g} deg")
        return 0
    except (BoardError, OSError) as e:  # serial.SerialException is an OSError
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        b.close()


if __name__ == "__main__":
    sys.exit(main())

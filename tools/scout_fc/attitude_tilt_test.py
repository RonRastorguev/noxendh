"""Attitude orientation (tilt) test for the Scout (passive: reads ATTITUDE through the logging bridge, sends nothing).

Tells whether PX4's body frame matches the airframe (camera = forward). Ron holds the Scout and tilts it:
    level -> NOSE (camera side) DOWN ~30 deg -> level -> RIGHT SIDE DOWN ~30 deg -> level
PX4 convention (FRD): nose down = NEGATIVE pitch; right side down = POSITIVE roll.
If nose-down shows up as roll, or with the wrong sign, the FC's mounting rotation (SENS_BOARD_ROT) is wrong,
and that affects every flight mode. If both read correctly, a rotated flow velocity is the flow sensor's own
rotation (SENS_FLOW_ROT).

    py -3.9 attitude_tilt_test.py [seconds]
"""
import math
import os
import sys
import time

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 45
m = mavutil.mavlink_connection("tcp:127.0.0.1:47650", source_system=255, source_component=190)
t0 = time.time()
next_print = t0 + 0.5
ext = {"roll+": 0.0, "roll-": 0.0, "pitch+": 0.0, "pitch-": 0.0}
print("   t    roll(deg)   pitch(deg)")
while time.time() - t0 < secs:
    msg = m.recv_match(type="ATTITUDE", blocking=True, timeout=0.3)
    if msg is None:
        continue
    r, p = math.degrees(msg.roll), math.degrees(msg.pitch)
    ext["roll+"] = max(ext["roll+"], r); ext["roll-"] = min(ext["roll-"], r)
    ext["pitch+"] = max(ext["pitch+"], p); ext["pitch-"] = min(ext["pitch-"], p)
    if time.time() >= next_print:
        next_print += 0.5
        print(f"{time.time() - t0:5.1f}   {r:+7.1f}     {p:+7.1f}", flush=True)
print("extremes (deg):", {k: round(v, 1) for k, v in ext.items()})
print("expected: nose down -> pitch about -30; right side down -> roll about +30")
print("sent: nothing")

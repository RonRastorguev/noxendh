"""Hand-lift flow test for the Scout (passive: reads MAVLink through the logging bridge, sends nothing).

Ron lifts the Scout by hand (USB only, no battery: motors cannot spin) to ~0.5 m, holds it, moves it slowly,
sets it down. Once a second this prints what the estimator does with the flow sensor:
  range   rangefinder distance (m)          q      flow quality (0..255)
  eph     horizontal position accuracy (m)  vx/vy  estimated velocity (m/s)
  POS_REL ESTIMATOR_STATUS says relative horizontal position is good
Pass looks like: above ~0.08 m the eph falls from metres to centimetres, POS_REL turns on, and vx/vy
follow the hand's motion (sign and size) then return to ~0 when held still.

    py -3.9 flow_hand_test.py [seconds]      (PYTHONPATH must hold pymavlink; bridge on tcp:127.0.0.1:47650)
"""
import os
import sys
import time

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil

POS_HORIZ_REL = 8      # ESTIMATOR_STATUS_FLAGS bit
secs = float(sys.argv[1]) if len(sys.argv) > 1 else 90
m = mavutil.mavlink_connection("tcp:127.0.0.1:47650", source_system=255, source_component=190)
last = {}
t0 = time.time()
next_print = t0 + 1
print("  t   range    q     eph     vx      vy    POS_REL")
while time.time() - t0 < secs:
    msg = m.recv_match(type=["DISTANCE_SENSOR", "OPTICAL_FLOW_RAD", "ESTIMATOR_STATUS", "LOCAL_POSITION_NED"],
                       blocking=True, timeout=0.3)
    if msg is not None:
        t = msg.get_type()
        if t == "DISTANCE_SENSOR" and msg.get_srcComponent() == 1:
            last["range"] = msg.current_distance / 100.0
        elif t == "OPTICAL_FLOW_RAD" and msg.get_srcComponent() == 1:
            last["q"] = msg.quality
        elif t == "ESTIMATOR_STATUS":
            last["eph"] = msg.pos_horiz_accuracy
            last["rel"] = bool(msg.flags & POS_HORIZ_REL)
        elif t == "LOCAL_POSITION_NED":
            last["vx"], last["vy"] = msg.vx, msg.vy
    if time.time() >= next_print:
        next_print += 1
        g = lambda k, f: (f % last[k]) if k in last else "  -  "
        print(f"{time.time() - t0:4.0f}  {g('range', '%5.2f')}  {g('q', '%3d')}  {g('eph', '%6.2f')}  "
              f"{g('vx', '%6.2f')}  {g('vy', '%6.2f')}  {'yes' if last.get('rel') else 'no'}", flush=True)
print("sent: nothing")

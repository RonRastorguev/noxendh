"""Flow DIRECTION test for the Scout (passive: reads MAVLink through the logging bridge, sends nothing).

The estimator's velocity (LOCAL_POSITION_NED vx, vy) is rotated into the Scout's own frame with its yaw
(ATTITUDE), so the output reads directly as FORWARD (towards the camera) and RIGHT. Ron moves the Scout by
hand, held level at a measured height: forward, stop, back, stop, right, stop, left, stop. A correct flow
setup shows +fwd when pushed towards the camera and +right when pushed to the Scout's right. A sign that is
wrong means Position mode would push the wrong way: do not fly Position until it is fixed.

Also prints the rangefinder, so it can be compared with a tape measure.

    py -3.9 flow_direction_test.py [seconds]   (PYTHONPATH must hold pymavlink; bridge on tcp:127.0.0.1:47650)
"""
import math
import os
import sys
import time

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 75
m = mavutil.mavlink_connection("tcp:127.0.0.1:47650", source_system=255, source_component=190)
last = {}
t0 = time.time()
next_print = t0 + 0.5
peak = {"fwd+": 0.0, "fwd-": 0.0, "right+": 0.0, "right-": 0.0}
print("   t   range   fwd(m/s)  right(m/s)   yaw(deg)")
while time.time() - t0 < secs:
    msg = m.recv_match(type=["DISTANCE_SENSOR", "LOCAL_POSITION_NED", "ATTITUDE"], blocking=True, timeout=0.2)
    if msg is not None:
        t = msg.get_type()
        if t == "DISTANCE_SENSOR" and msg.get_srcComponent() == 1:
            last["range"] = msg.current_distance / 100.0
        elif t == "ATTITUDE":
            last["yaw"] = msg.yaw
        elif t == "LOCAL_POSITION_NED" and "yaw" in last:
            c, s = math.cos(last["yaw"]), math.sin(last["yaw"])
            fwd = c * msg.vx + s * msg.vy          # NED -> body (yaw only; held level)
            right = -s * msg.vx + c * msg.vy
            last["fwd"], last["right"] = fwd, right
            peak["fwd+"] = max(peak["fwd+"], fwd); peak["fwd-"] = min(peak["fwd-"], fwd)
            peak["right+"] = max(peak["right+"], right); peak["right-"] = min(peak["right-"], right)
    if time.time() >= next_print and "fwd" in last:
        next_print += 0.5
        bar = lambda v: ("+" * min(10, int(abs(v) / 0.03))) if v > 0 else ("-" * min(10, int(abs(v) / 0.03)))
        print(f"{time.time() - t0:5.1f}  {last.get('range', float('nan')):5.2f}   {last['fwd']:+6.2f} {bar(last['fwd']):<10}"
              f"  {last['right']:+6.2f} {bar(last['right']):<10}  {math.degrees(last['yaw']):7.1f}", flush=True)
print("peaks:", {k: round(v, 2) for k, v in peak.items()})
print("sent: nothing")

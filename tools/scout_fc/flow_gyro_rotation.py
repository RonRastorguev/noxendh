"""Measure the flow sensor's yaw misalignment from ROTATION alone (passive: reads MAVLink, sends nothing).

Rocking the Scout in place over a textured floor makes the flow sensor see the floor sweep past. PX4's
OPTICAL_FLOW_RAD (from the autopilot, component 1) carries the rotated body-frame flow AND the gyro over the same
interval; PX4's convention is that a right-handed rotation about a body axis produces a POSITIVE flow on that axis.
So with a correctly set SENS_FLOW_ROT, pure rotation gives flow ~= gyro (angle 0). The least-squares angle between
the flow vector and the gyro vector, over the samples with real rotation, IS the misalignment, independent of
small hand translation.

Ron holds it ~0.5 m over a textured floor and rocks it: nose up/down a few times, then side to side a few times.

    py -3.9 flow_gyro_rotation.py [seconds] [csv_out]
Prints the misalignment delta (deg) and, for the current SENS_FLOW_ROT, the setting that would cancel it.
"""
import csv
import math
import os
import sys
import time

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 45
out = sys.argv[2] if len(sys.argv) > 2 else "flow_gyro.csv"
MIN_RATE = 0.3          # rad/s: only samples with real rotation
MIN_QUALITY = 30

m = mavutil.mavlink_connection("tcp:127.0.0.1:47650", source_system=255, source_component=190)
rows = []
t0 = time.time()
while time.time() - t0 < secs:
    msg = m.recv_match(type="OPTICAL_FLOW_RAD", blocking=True, timeout=0.3)
    if msg is None or msg.get_srcComponent() != 1 or msg.integration_time_us <= 0:
        continue
    dt = msg.integration_time_us * 1e-6
    rows.append((time.time() - t0, msg.integrated_x / dt, msg.integrated_y / dt,
                 msg.integrated_xgyro / dt, msg.integrated_ygyro / dt, msg.quality, msg.distance))
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "flow_x", "flow_y", "gyro_x", "gyro_y", "quality", "distance"])
    w.writerows(rows)

use = [r for r in rows if math.hypot(r[3], r[4]) > MIN_RATE and r[5] >= MIN_QUALITY]
print(f"{len(rows)} flow samples, {len(use)} with rotation > {MIN_RATE} rad/s and quality >= {MIN_QUALITY} (saved {out})")
if len(use) < 20:
    print("too few rotating samples: rock it more firmly, over a textured floor, ~0.5 m up")
    sys.exit(1)
s_cross = sum(gx * fy - gy * fx for _, fx, fy, gx, gy, _, _ in use)
s_dot = sum(gx * fx + gy * fy for _, fx, fy, gx, gy, _, _ in use)
delta = math.degrees(math.atan2(s_cross, s_dot))          # flow direction relative to gyro direction (z-down yaw sense)
gain = math.hypot(s_cross, s_dot) / sum(gx * gx + gy * gy for _, _, _, gx, gy, _, _ in use)
for axis, gi, fi in (("x (roll)", 3, 1), ("y (pitch)", 4, 2)):
    pts = [r for r in use if abs(r[gi]) > MIN_RATE]
    if pts:
        k = sum(r[gi] * r[fi] for r in pts) / sum(r[gi] ** 2 for r in pts)
        print(f"  {axis}: flow/gyro on the same axis = {k:+.2f} over {len(pts)} samples (aligned: about +1)")
print(f"misalignment delta = {delta:+.1f} deg (flow vs gyro), magnitude ratio {gain:.2f}")
print("sent: nothing")

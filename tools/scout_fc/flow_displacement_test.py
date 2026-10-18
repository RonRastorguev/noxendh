"""Flow DISPLACEMENT test for the Scout (passive: reads MAVLink through the logging bridge, sends nothing).

Robust to timing and hand wobble: it finds the still pauses itself (estimated speed < 0.04 m/s for >= 1.5 s
while lifted) and reports, between consecutive pauses, how far the estimator says the Scout moved in its OWN
frame (FORWARD = towards the camera, RIGHT = the Scout's right), using the mean yaw of the move.

Ron, along tape marks on the floor, keeping the camera pointing the same way:
    lift (hold 5 s) -> forward 1 m (pause 5 s) -> back 1 m (pause) -> right 1 m (pause) -> left 1 m (pause) -> down
Pass: the four moves read about +1 fwd, -1 fwd, +1 right, -1 right, with the other axis small. A sign the wrong
way round, or most of a move on the wrong axis, means the flow sensor's orientation is wrong: no Position-mode
flight until fixed. The size also checks the flow scale (a 1 m move reading 0.5 m or 2 m is a problem too).

    py -3.9 flow_displacement_test.py [seconds] [csv_out]
"""
import csv
import math
import os
import sys
import time

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 120
out = sys.argv[2] if len(sys.argv) > 2 else "displacement.csv"
STILL_V, STILL_S, LIFTED_M = 0.04, 1.5, 0.15

m = mavutil.mavlink_connection("tcp:127.0.0.1:47650", source_system=255, source_component=190)
rows, last = [], {}
t0 = time.time()
while time.time() - t0 < secs:
    msg = m.recv_match(type=["DISTANCE_SENSOR", "LOCAL_POSITION_NED", "ATTITUDE"], blocking=True, timeout=0.2)
    if msg is None:
        continue
    t = msg.get_type()
    if t == "DISTANCE_SENSOR" and msg.get_srcComponent() == 1:
        last["range"] = msg.current_distance / 100.0
    elif t == "ATTITUDE":
        last["yaw"] = msg.yaw
    elif t == "LOCAL_POSITION_NED" and "yaw" in last and "range" in last:
        rows.append((time.time() - t0, last["range"], msg.x, msg.y, msg.vx, msg.vy, last["yaw"]))

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "range", "x", "y", "vx", "vy", "yaw"])
    w.writerows(rows)

# still segments while lifted
segs, start = [], None
for i, (t, rng, x, y, vx, vy, yaw) in enumerate(rows):
    still = rng > LIFTED_M and math.hypot(vx, vy) < STILL_V
    if still and start is None:
        start = i
    if (not still or i == len(rows) - 1) and start is not None:
        end = i if not still else i + 1
        if rows[end - 1][0] - rows[start][0] >= STILL_S:
            segs.append((start, end))
        start = None

print(f"{len(rows)} samples, {len(segs)} still pauses while lifted (saved {out})")
for k, (s, e) in enumerate(segs):
    t_a, t_b = rows[s][0], rows[e - 1][0]
    rng = sum(r[1] for r in rows[s:e]) / (e - s)
    print(f"  pause {k}: {t_a:5.1f}-{t_b:5.1f} s at range {rng:.2f} m")
print("moves between pauses (Scout frame):")
for k in range(len(segs) - 1):
    a, b = segs[k], segs[k + 1]
    xa = sum(r[2] for r in rows[a[0]:a[1]]) / (a[1] - a[0]); ya = sum(r[3] for r in rows[a[0]:a[1]]) / (a[1] - a[0])
    xb = sum(r[2] for r in rows[b[0]:b[1]]) / (b[1] - b[0]); yb = sum(r[3] for r in rows[b[0]:b[1]]) / (b[1] - b[0])
    mid = rows[a[1]:b[0]] or rows[a[0]:b[1]]
    yaw = math.atan2(sum(math.sin(r[6]) for r in mid), sum(math.cos(r[6]) for r in mid))
    dx, dy = xb - xa, yb - ya
    fwd = math.cos(yaw) * dx + math.sin(yaw) * dy
    right = -math.sin(yaw) * dx + math.cos(yaw) * dy
    print(f"  move {k + 1} ({rows[a[1] - 1][0]:5.1f}->{rows[b[0]][0]:5.1f} s): forward {fwd:+.2f} m, right {right:+.2f} m")
print("sent: nothing")

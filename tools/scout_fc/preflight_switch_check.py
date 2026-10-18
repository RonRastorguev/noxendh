"""Pre-flight check on the bench (passive: reads MAVLink through the logging bridge, sends nothing).

Prints battery voltage, armed state, PX4's actual flight mode, and the raw RC channels Ron's switches drive
(ch 5 = mode: Stabilized / Altitude / Position, ch 6 = offboard, ch 8 = kill). Ron flips each switch while it runs;
the mode column shows what PX4 actually switched to. Prints a line only when something changes.

    py -3.9 preflight_switch_check.py [seconds]
"""
import os
import sys
import time

os.environ.setdefault("MAVLINK20", "1")
from pymavlink import mavutil

secs = float(sys.argv[1]) if len(sys.argv) > 1 else 45
MAIN = {1: "Manual", 2: "Altitude", 3: "Position", 4: "Auto", 5: "Acro", 6: "Offboard", 7: "Stabilized", 8: "Rattitude"}
AUTO = {2: "Takeoff", 3: "Hold", 4: "Mission", 5: "Return", 6: "Land", 9: "Precision land"}
m = mavutil.mavlink_connection("tcp:127.0.0.1:47650", source_system=255, source_component=190)
st, last_line = {}, None
t0 = time.time()
while time.time() - t0 < secs:
    msg = m.recv_match(type=["HEARTBEAT", "SYS_STATUS", "RC_CHANNELS"], blocking=True, timeout=0.3)
    if msg is None:
        continue
    t = msg.get_type()
    if t == "HEARTBEAT" and msg.get_srcComponent() == 1:
        main = (msg.custom_mode >> 16) & 0xFF
        sub = (msg.custom_mode >> 24) & 0xFF
        st["mode"] = AUTO.get(sub, f"Auto/{sub}") if main == 4 else MAIN.get(main, str(main))
        st["armed"] = "ARMED" if msg.base_mode & 0x80 else "disarmed"
    elif t == "SYS_STATUS":
        st["v"] = round(msg.voltage_battery / 1000.0, 1)
    elif t == "RC_CHANNELS":
        st["rc"] = "no RC" if msg.chancount == 0 or msg.chan5_raw == 0 else \
            f"ch5 {msg.chan5_raw} ch6 {msg.chan6_raw} ch8 {msg.chan8_raw} rssi {msg.rssi}"
    line = f"battery {st.get('v')} V | {st.get('armed')} | mode {st.get('mode')} | {st.get('rc')}"
    if line != last_line and "v" in st and "mode" in st and "rc" in st:
        print(f"{time.time() - t0:5.1f}s  {line}", flush=True)
        last_line = line
print("sent: nothing")

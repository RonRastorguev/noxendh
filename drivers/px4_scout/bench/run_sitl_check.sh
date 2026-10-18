#!/bin/bash
# run_sitl_check.sh -- PX4 SIH SITL (no Gazebo) + Micro XRCE-DDS agent + sitl_offboard_check.py, in WSL.
#
#   wsl.exe -d Ubuntu-24.04 bash /mnt/c/Users/Ron/Desktop/noxen-dh26/drivers/px4_scout/bench/run_sitl_check.sh [flight|fallback|mission]
#
# * Takes the laptop's simulator lock (flock -n on <artifacts>/sim.lock, then <artifacts>/heavy.lock).
#   Held -> exit 2 "sim_busy" at once: never waits, never kills.
# * Refuses (exit 2) if a px4 or MicroXRCEAgent it did not start is already running.
# * PX4 = ~/PX4-Autopilot/build/px4_sitl_default (airframe 10040 sihsim_quadx), fresh rootfs per run, its own ROS
#   domain (UXRCE_DDS_DOM_ID = ROS_DOMAIN_ID = 57) and SHM-only DDS (this laptop's WSL drops UDP DDS data).
#   flight mode flies SIH's default baro/GPS height (SCOUT_HGT=range: the Scout's range reference, does not arm in SIH);
#   fallback mode silences SIH's distance sensor (SIH_DISTSNSR_OVR 10000) to test the no-range path.
#   mission mode runs the REAL `scout_mission` node (ros2_ws/src/nx_scout_nav) with auto_arm:=true (SITL ONLY, DN-07)
#   on the flightsim room without the pad box (sitl_camera_sim.py --export), fed by sitl_camera_sim.py (images rendered
#   at SIH's ground-truth pose) and watched by sitl_mission_monitor.py (exit 0 = DONE). Same baro/GPS height as flight.
#   SCOUT_CAM_POSE=odom renders at PX4's EKF pose instead (hides SIH's estimator error: its EKF z is off up to 1.2 m
#   near the ground, which the alignment rightly rejects); SCOUT_GIL_S sets the node's gil_switch_interval_s.
# * flight: after PX4 exits, sitl_ulog_check.py checks PX4's own .ulg (received setpoints, SIH ground truth).
# * Teardown (trap): TERM the processes it started, KILL survivors (same PID + same start time only),
#   `ros2 daemon stop`, then removes only /dev/shm/fastrtps_* files no process holds open. DONE <rc> sentinel.
# Artifacts: $ART_ROOT/scout_sitl_<utc>_<mode>/ (WSL ext4, never /mnt/c).
# NOTE: no `set -u` (ROS setup scripts reference unset vars).
set -o pipefail
MODE="${1:-flight}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ART_ROOT="${ART_ROOT:-$HOME/noxen_artifacts}"     # WSL ext4; also where the shared sim lock lives
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
BUILD="$PX4_DIR/build/px4_sitl_default"
DOMAIN=57
export NOXEN_RUNID="scout_sitl_$(date -u +%Y%m%dT%H%M%SZ)_$MODE${SCOUT_CAM_POSE:+_$SCOUT_CAM_POSE}${SCOUT_MISSION_MODE:+_$SCOUT_MISSION_MODE}"
ART="$ART_ROOT/$NOXEN_RUNID"
mkdir -p "$ART"
case "$MODE" in flight|fallback|mission) ;; *) echo "usage: $0 [flight|fallback|mission]"; exit 2 ;; esac

# The laptop's one-simulator lock (shared with every other simulator user on this machine, same files): sim.lock is
# taken without waiting (held -> exit 2 sim_busy), heavy.lock (long CPU jobs) is waited for up to HEAVY_WAIT_S.
LOCK_DIR="${SIM_LOCK_DIR:-$ART_ROOT}"
exec 9>"$LOCK_DIR/sim.lock"
if ! flock -n 9; then
  printf '{"schema_version":1,"tool":"run_sitl_check","runid":"%s","result":"ERROR","exit":2,"reason":"sim_busy"}\n' "$NOXEN_RUNID" | tee "$ART/lock.json"
  exit 2
fi
exec 8>"$LOCK_DIR/heavy.lock"
if ! flock -w "${HEAVY_WAIT_S:-10}" 8; then
  printf '{"schema_version":1,"tool":"run_sitl_check","runid":"%s","result":"ERROR","exit":2,"reason":"heavy_busy"}\n' "$NOXEN_RUNID" | tee "$ART/lock.json"
  exit 2
fi

FOREIGN="$(pgrep -a -x px4; pgrep -a -f MicroXRCEAgent)"
if [ -n "$FOREIGN" ]; then
  printf '{"schema_version":1,"tool":"run_sitl_check","runid":"%s","result":"ERROR","exit":2,"reason":"foreign_sim_processes"}\n' "$NOXEN_RUNID" | tee "$ART/result_runner.json"
  echo "$FOREIGN"
  exit 2
fi

source /opt/ros/jazzy/setup.bash
source "$HOME/drone_ws/install/setup.bash"
source "$HOME/px4_ros_uxrce_dds_ws/install/local_setup.bash"
export ROS_DOMAIN_ID=$DOMAIN
export FASTDDS_BUILTIN_TRANSPORTS=SHM
export PYTHONUNBUFFERED=1

PIDS=()
pid_start() { awk '{print $22}' "/proc/$1/stat" 2>/dev/null; }
declare -A START
teardown() {
  local rc=$?
  trap - EXIT INT TERM
  echo "[runner] teardown (rc $rc)"
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null; done
  for _ in $(seq 1 50); do
    local alive=0
    for p in "${PIDS[@]}"; do [ "$(pid_start "$p")" = "${START[$p]}" ] && alive=1; done
    [ $alive = 0 ] && break
    sleep 0.1
  done
  for p in "${PIDS[@]}"; do
    if [ "$(pid_start "$p")" = "${START[$p]}" ]; then echo "[runner] KILL $p"; kill -KILL "$p" 2>/dev/null; fi
  done
  ros2 daemon stop > /dev/null 2>&1
  for f in /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*; do
    [ -e "$f" ] || continue
    fuser -s "$f" 2>/dev/null || rm -f "$f"
  done
  echo "DONE $rc" > "$ART/DONE"
  echo "[runner] DONE $rc  artifacts: $ART"
  exit $rc
}
trap teardown EXIT
trap 'exit 130' INT TERM

FASTRTPS_DEFAULT_PROFILES_FILE="$HERE/agent_shm_profile.xml" MicroXRCEAgent udp4 -p 8888 > "$ART/agent.log" 2>&1 &
PIDS+=($!); START[$!]="$(pid_start $!)"

mkdir -p "$ART/rootfs"
PX4_ENV=(PX4_SYS_AUTOSTART=10040 PX4_SIMULATOR=sihsim HEADLESS=1 PX4_PARAM_UXRCE_DDS_DOM_ID=$DOMAIN)
# flight: SIH's default height (baro + GPS). SCOUT_HGT=range tries the real Scout's reference (FC dump
# hardware/scout/apply_20261016_S5: EKF2_HGT_REF 2, EKF2_RNG_CTRL 2) but in SIH it never arms: SIH's rangefinder sits
# at the body origin, reads -0.0002 m on the ground (invalid) and PX4 1.17 keeps "Preflight Fail: height estimate not
# stable" for 120 s (2026-10-17). fallback: no range at all, so baro.
if [ "$MODE" = fallback ]; then
  SCOUT_PARAMS=(PX4_PARAM_SIH_DISTSNSR_OVR=10000)
  export NOXEN_SCOUT_HGT_REF=baro
elif [ "${SCOUT_HGT:-baro}" = range ]; then
  SCOUT_PARAMS=(PX4_PARAM_EKF2_HGT_REF=2 PX4_PARAM_EKF2_RNG_CTRL=2)
  export NOXEN_SCOUT_HGT_REF=range
else
  SCOUT_PARAMS=()
  export NOXEN_SCOUT_HGT_REF=baro
fi
PX4_ENV+=("${SCOUT_PARAMS[@]}")
export NOXEN_SCOUT_PX4_PARAMS="${SCOUT_PARAMS[*]}"
( cd "$ART/rootfs" && exec env "${PX4_ENV[@]}" "$BUILD/bin/px4" -d "$BUILD/etc" ) > "$ART/px4.log" 2>&1 &
PX4_PID=$!
PIDS+=($PX4_PID); START[$PX4_PID]="$(pid_start $PX4_PID)"
echo "[runner] px4 pid $PX4_PID, agent pid ${PIDS[0]}, domain $DOMAIN, artifacts $ART"

if [ "$MODE" = mission ]; then
  REPO="$(cd "$HERE/../../.." && pwd)"
  AS="$ART/assets"
  python3 "$HERE/sitl_camera_sim.py" --export "$AS" > "$ART/assets.log" 2>&1 || { echo "[runner] asset export failed"; exit 2; }
  TK="$(python3 -c "import json;print(json.load(open('$AS/assets.json'))['takeoff'])")"
  GOAL="$(python3 -c "import json;print(json.load(open('$AS/assets.json'))['goal'])")"
  # SCOUT_CAM_POSE=truth (default: SIH ground truth) or odom (PX4 EKF pose: hides SIH estimator error from the node)
  python3 "$HERE/sitl_camera_sim.py" --assets "$AS" --out-dir "$ART" --pose-source "${SCOUT_CAM_POSE:-truth}" \
    > "$ART/camera_sim.log" 2>&1 &
  PIDS+=($!); START[$!]="$(pid_start $!)"
  timeout -s TERM 480 python3 "$HERE/sitl_mission_monitor.py" --out-dir "$ART" --timeout-s 420 \
    --camera-pose "${SCOUT_CAM_POSE:-truth}" 2>&1 | tee "$ART/check.log" &
  MON_PID=$!
  sleep 2
  # SCOUT_MISSION_MODE=wall_goal (default: pad -> B1 -> back) or hover_test (no goal; SCOUT_HOVER_CLIMB_M,
  # SCOUT_HOVER_S, defaults 2.0 m / 10.0 s; decimals, rclpy parameters are type-strict)
  if [ "${SCOUT_MISSION_MODE:-wall_goal}" = hover_test ]; then
    MODE_ARGS=(-p mode:=hover_test -p hover_climb_m:="${SCOUT_HOVER_CLIMB_M:-2.0}" -p hover_s:="${SCOUT_HOVER_S:-10.0}")
  else
    MODE_ARGS=(-p mode:=wall_goal -p goal:="$GOAL" -p on_arrival:=return_land)
  fi
  # SITL ONLY: auto_arm true (never on the real aircraft, DN-07); height_ref = what SIH flies here (baro by default)
  PYTHONPATH="$REPO/ros2_ws/src/nx_scout_loc:$REPO/ros2_ws/src/nx_scout_nav:$REPO:$PYTHONPATH" python3 \
    "$REPO/ros2_ws/src/nx_scout_nav/nx_scout_nav/mission_node.py" --ros-args \
    -p repo_root:="$REPO" -p landmarks:="$AS/landmarks.npz" -p map_pcd:="$AS/map.pcd" \
    -p takeoff:="$TK" "${MODE_ARGS[@]}" -p auto_arm:=true \
    -p height_ref:="$NOXEN_SCOUT_HGT_REF" -p image_latency_s:=0.25 -p odom_latency_s:=0.005 \
    -p gil_switch_interval_s:="${SCOUT_GIL_S:-0.0005}" \
    > "$ART/mission_node.log" 2>&1 &
  PIDS+=($!); START[$!]="$(pid_start $!)"
  echo "[runner] mission node pid $!, mode ${SCOUT_MISSION_MODE:-wall_goal}, takeoff $TK, goal $GOAL"
  wait "$MON_PID"
  RC=$?
  case "$RC" in 0|1) ;; *) echo "[runner] monitor exited $RC (no terminal state / timeout): ERROR"; RC=2 ;; esac
  exit "$RC"
fi

NOXEN_SCOUT_SITL="px4 pid $PX4_PID rootfs $ART/rootfs" timeout -s TERM 300 \
  python3 "$HERE/sitl_offboard_check.py" --confirm-sitl --mode "$MODE" --out-dir "$ART" 2>&1 | tee "$ART/check.log"
RC=${PIPESTATUS[0]}
case "$RC" in 0|1) ;; *) echo "[runner] check exited $RC (timeout 124 / crash): ERROR"; RC=2 ;; esac
# PX4 must exit (and close its .ulg) before the log check; the trap's teardown handles the rest
kill -TERM "$PX4_PID" 2>/dev/null
for _ in $(seq 1 100); do [ "$(pid_start "$PX4_PID")" = "${START[$PX4_PID]}" ] || break; sleep 0.1; done
if [ "$MODE" = flight ] && [ "$RC" != 2 ]; then      # it merges into result.json and re-decides PASS / FAIL
  python3 "$HERE/sitl_ulog_check.py" "$ART" 2>&1 | tee -a "$ART/check.log"
  RC=${PIPESTATUS[0]}
fi
exit "$RC"

# Scout autonomous flight: takeoff → A to B (a wall point) → looming avoidance (design brief)

Ron, 2026-10-17: "locating itself given prior lidar and its camera view, along with obstacle avoidance via looming
(fly over the object or around it) … go point A to B, point B being some point on the wall … take off on its own;
takeoff location will be fixed." Stability (S1–S5) is done; camera relocalization exists (`docs/SCOUT_LOC.md`).
This brief is what every builder works from. Decisions: `docs/DECISIONS_SCOUT_NAV.md` (DN-nn).

**No hardware data exists.** Everything is built and tested ROS-free in Windows Python 3.9 against a closed-loop
synthetic world (`flightsim/`). Every threshold is PROVISIONAL. Nothing here arms, flies or writes to the real FC.

## 1. Pieces and owners

| Piece | Package / files | Consumes | Produces |
|---|---|---|---|
| Shared types, vehicle interface | `navigation/types.py`, `navigation/vehicle.py` (fixed; ask before changing) | — | — |
| A. Map↔odom alignment | `localization/map_align.py`, `tests/test_map_align.py` | odom samples, PnP fixes, known takeoff pose, terrain model | `T_map_odom`, body pose in map, map→odom setpoint conversion |
| B. Looming | `avoidance/` (`looming.py`, …), `tests/test_looming.py` | images, odom (attitude + velocity), map points, map pose (optional) | `LoomingOutput` (emergency flag, obstacles in map) |
| C. Navigation | `navigation/` (`static_map.py`, `goal.py`, `planner.py`, `obstacle_memory.py`, `follower.py`, `safety.py`), `tests/test_navigation.py` | map points, obstacles, pose in map | wall goal pose, 2.5-D plan (over / around), setpoints in map, safety gate |
| D. Mission | `navigation/mission.py`, `tests/test_mission.py` | all of the above + `VehicleInterface` | the state machine: preflight → takeoff → A→B → arrive (→ return → land) |
| E. Closed-loop sim | `flightsim/` (`world.py`, `vehicle_sim.py`, `loop.py`, `metrics.py`, `scenarios.py`), `tools/run_scout_sim.py`, `tests/test_flightsim.py` | everything | scenario runs → JSON metrics, exit 0/1/2 |

Config: one file per piece so builders never edit the same file — A adds a `map_align:` section to
`config/scout_loc.yaml`; B `config/scout_avoid.yaml`; C `config/scout_nav.yaml`; D `config/scout_mission.yaml`;
E `config/scout_sim.yaml`. Hardware facts only in `config/scout_hw.yaml` (DH-06), read never copied. Each module also
carries a `DEFAULTS` dict (like `localization/localizer.py`) so it runs without YAML in tests.

## 2. Frames and the vehicle interface

`navigation/types.py` defines the frames (map, odom, body FLU, cam optical) and the data types. The flight
controller is reached only through `navigation/vehicle.py` `VehicleInterface`: odometry in a z-up odom frame,
status, odom position setpoints streamed ≥ 10 Hz, arm / offboard / land requests, and `capabilities()`.
PX4/NED specifics live only in a vehicle adapter (later), never in these packages.

## 3. Facts about the real Scout that shape the design (from its FC dump, `hardware/scout/apply_20261016_S5/`)

- **Height reference = rangefinder** (`EKF2_HGT_REF=2`, `EKF2_RNG_CTRL=2`). PX4's local z is height above whatever
  is below. Flying over a 0.7 m box, the FC's z drops by ~0.7 m and an unchanged z setpoint makes it climb by the
  box's height. So: z in odom = z in map − terrain(x, y under the vehicle) (+ a slow bias), and the map→odom
  setpoint conversion must use the terrain under the vehicle NOW. Terrain comes from the static map plus known
  obstacles; the residual is corrected from PnP fixes. How fast PX4 follows a terrain step (fused gradually or
  reset) is NOT known: the sim models it as a first-order lag `hgt_step_tau_s` (provisional) and a test covers
  both a fast and a slow value. (Alternative = moving the FC to a baro height reference: a parameter change, Ron's
  call; not assumed.)
- x/y from optical flow + rangefinder: drift is slow but real; yaw from gyro only (no mag): slow drift. A PX4
  local-frame reset increments `reset_counter`.
- Limits on the FC: `MPC_XY_VEL_MAX` 1.5, `MPC_Z_VEL_MAX_UP` 1.0, `_DN` 0.7, `MPC_YAWRAUTO_MAX` 60°/s,
  `MPC_ACC_HOR` 3, `MPC_XY_P` 0.95, `MPC_Z_P` 1.0. The autonomous cap is ours: 0.5 m/s (`flight_demo.yaml`).
- Offboard loss: `COM_OF_LOSS_T` 1 s → `COM_OBL_RC_ACT` 0 (Position mode on flow). That is the backstop when the
  laptop, Wi-Fi or Radxa dies (DH-05).
- Camera: one forward DECXIN 2633V2, nominal 102° diagonal, mount not measured (synthetic mount
  `T_BODY_CAM_SYNTH`: 6 cm ahead, 5° down). **It sees only forward**, so the Scout must face where it moves.
- Hull radius: NOT measured. Provisional placeholder in `scout_hw.yaml` (`airframe.hull_radius_m`, NEEDS_RON).

## 4. A. Map↔odom alignment (`localization/map_align.py`)

The FC flies on its own estimate; the laptop does not feed it vision (EV fusion = a parameter change, stage 2).
We estimate `T_map_odom` — 4-DoF (x, y, z, yaw), both frames are gravity-aligned — and command setpoints in odom.

```python
class MapOdomAligner:
    def __init__(self, cfg: dict | None = None)
    def add_odom(self, s: OdomSample) -> None                    # every sample; keeps a short buffer (interpolation)
    def set_terrain(self, fn) -> None                            # fn(x, y, z_body_map) -> z of the surface below, map
    def set_height_ref(self, kind: str) -> None                  # "range" | "baro" | "ev" (from vehicle capabilities)
    def init_known_pose(self, T_map_body, stamp) -> None         # the fixed takeoff pose, on the ground
    def add_fix(self, stamp, T_map_body, info=None) -> dict      # PnP fix (image stamp) -> {"accepted", "reason", "residual_m", "residual_deg"}
    def valid(self, now) -> bool                                 # alignment usable (initialized, not stale)
    def T_map_odom(self) -> np.ndarray | None
    def pose_map(self, now=None) -> tuple | None                 # (T_map_body, v_map) from the latest odom
    def body_pose_map_at(self, stamp) -> np.ndarray | None       # interpolated (for the looming map comparison)
    def setpoint_to_odom(self, p_map, yaw_map) -> tuple          # (p_odom, yaw_odom), z via terrain under the vehicle now
    def status(self, now) -> dict
```

Requirements: interpolate odom at the fix's image stamp (camera latency 50–200 ms); residual gating with outlier
rejection (one bad PnP must not move the frame); smooth correction (rate-limited / filtered so setpoints never jump
more than a few cm per tick); yaw handled on the circle; z handled per §3 (range height ref: the z offset is
terrain + bias, never one constant); `reset_counter` change → re-anchor from the latest fix or the pre-reset
estimate (PX4 publishes the reset delta in reality; the sim may model a reset with a known delta); `valid` goes
false when no fix has been accepted for `max_fix_age_s` AND the drift budget is exceeded (odom-only propagation is
fine for a few seconds; the number is config). Tests: synthetic odom with drift + noisy fixes + outliers + latency +
a terrain step + a reset.

## 5. B. Looming (`avoidance/`)

"Looming" = an object's image grows as you approach it; time-to-contact TTC = scale / (d scale / dt), no depth
needed. Generic (any textured object), not the spec's person-only YOLO path (logged as a DN decision).

```python
class LoomingDetector:
    def __init__(self, cam: CameraModel, cfg: dict | None = None, map_points=None, map_spacing_m=0.03,
                 T_body_cam=None)
    def process(self, image, stamp, T_odom_body, v_odom, T_map_body=None) -> LoomingOutput
```

- **Ego-rotation removed with the FC's attitude** (odom, independent of PnP): tracked points from the previous
  frame(s) are de-rotated (`K R K⁻¹`) before measuring expansion, so a yaw or a pitch-on-acceleration does not loom.
- **Emergency (independent of localization and of the map):** features whose de-rotated local expansion gives
  TTC < `ttc_emergency_s` (1.5 s, `flight_demo.yaml avoid.looming_ttc_s`) AND that lie inside the flight corridor
  (the swept volume of the hull along the current velocity, using depth ≈ TTC × closing speed) → `emergency=True`.
  It uses only the image and the FC's own attitude/velocity, never the PnP pose or the map (a wrong pose must not
  be able to silence it). A raw obstacle (`source="looming_raw"`) is placed if a map pose is available.
- **Map-anchored detection (earlier, for replanning):** with a valid `T_map_body`, render the expected depth of the
  map (`localization/depth_render.py`), predict each tracked feature's expansion from the map depth and the known
  ego-motion, and flag features that loom significantly FASTER than the map predicts (they are nearer than the
  mapped surface behind them). Cluster → place in 3D at depth = TTC × closing speed → axis-aligned box in map;
  unseen depth extent = `assumed_depth_m`; `top_seen` = background features (map-consistent) were seen above the
  cluster; otherwise the box extends to the ceiling. Mapped walls — including the goal wall — must NOT be flagged.
- When the Scout is not moving (closing speed < `min_closing_speed_mps`) there is no looming: output nothing new.
- Tests (synthetic room renderer, an unmapped textured box approached at 0.3–0.5 m/s): detected at ≥ 1.5 m with a
  position error ≤ 0.3 m; mapped wall approached → no obstacle, no emergency (false-alarm test, the one that matters
  most); yaw turn in place → nothing; obstacle beside the corridor → no emergency; a late (close) obstacle →
  emergency within 0.3 s.

## 6. C. Navigation (`navigation/`)

- `static_map.py`: `StaticMap.from_points(points, cfg)` → occupancy voxels (default 0.1 m), map bounds, floor and
  ceiling heights, `terrain_z(x, y, z_below)` (top of the highest occupied voxel below a height),
  `distance_along(p, direction, max_d)` (ray to the first mapped surface).
- `goal.py`: `wall_goal(static_map, point_b, cfg) -> GoalPose`: the wall normal at B from the map points around it
  (PCA), oriented into free space; the hover point = B + standoff × normal (`standoff_m`, default 0.8), its z
  clamped into the flight band; yaw so the camera faces B. Errors with a reason if B is not on a mapped surface or the
  hover point is not free.
- `planner.py`: 2.5-D A* over (x, y, layer): flight layers every 0.2 m inside the band [`min_alt_m`, ceiling −
  `ceiling_margin_m`]; static occupancy inflated by hull radius + 0.3 m (`flight_demo.yaml nav.static_inflate_extra_m`),
  obstacles by hull radius + 0.5 m (`dynamic_inflate_extra_m`); horizontal moves 8-connected, vertical moves between
  layers cost `climb_cost_factor` × dz; so "over" vs "around" falls out of the cost. An obstacle with
  `top_seen=False` cannot be flown over. Line-of-sight shortcutting; result `PlanResult.shape` says direct / over /
  around. Must be fast enough to replan in < 0.3 s on a room-size map in Python.
- `obstacle_memory.py`: merges detections into persistent boxes. Looming cannot see a static obstacle while hovering,
  so memory lasts `memory_s` (default: the sortie), NOT the spec's 1.5 s decay (that was for continuous mono depth; DN).
- `follower.py`: carrot following at ≤ 0.5 m/s horizontal, ≤ 0.3 m/s vertical (config), acceleration-limited;
  **look where you go**: horizontal motion only within `max_heading_err_deg` of the camera's heading, else stop and
  yaw first (≤ 45°/s); final approach turns to the goal yaw. **TTC speed law:** horizontal speed ≤
  (distance to the first mapped surface or known obstacle ahead − stop margin) / `ttc_floor_s` (2.0 s > the 1.5 s
  emergency), so approaching a mapped wall (incl. the goal wall) can never trigger looming.
- `safety.py`: the gate every setpoint passes: geofence (map bounds − `geofence_margin_m`), flight band (floor +
  `min_alt_m` … ceiling − `ceiling_margin_m`), max step from the current pose, speed caps; HOLD (setpoint = current
  pose) when the alignment is invalid, there is no plan, or an emergency is active; emergency hover lasts
  `emergency_hover_s` (1.0 s) then a replan with the new obstacle.
- Tests: wall goal on synthetic walls; plans around a pillar, over a low wide box, around a tall box, refusal when
  boxed in; follower yaw-first behaviour and speed law; safety gate clamps.

## 7. D. Mission (`navigation/mission.py`)

```python
class Mission:
    def __init__(self, cfg, vehicle, localizer, aligner, looming, static_map, planner, follower, safety, memory,
                 cam, T_body_cam, log=None)
    def on_image(self, image, stamp) -> None     # localizer -> aligner.add_fix; looming -> memory
    def tick(self, now) -> None                  # 20 Hz: state machine, setpoint streaming
    state: str                                   # see below
```

Modes (`scout_mission.yaml mode`, node parameter `mode`): `wall_goal` (below) and `hover_test` (Ron 2026-10-17: take
off, rise `hover_test.climb_m` 2 m straight up from the pad, hold that spot `hover_test.hover_s` 10 s, land back on the
pad; `PREFLIGHT → ARMING → TAKEOFF → HOVER → LAND → DONE`, no goal, no looming, the climb capped below the map ceiling
minus `ceiling_margin_m`).

States: `PREFLIGHT` (odom valid; alignment initialized from the FIXED takeoff pose in `scout_mission.yaml`; if a PnP
fix on the ground disagrees with it by > tolerance → `REFUSED`, the Scout is not where it should be) → `ARMING`
(stream hold setpoints ≥ 1 s, request offboard, then arm if `auto_arm` else wait for the pilot to arm; timeout →
`ABORTED`) → `TAKEOFF` (vertical climb at ≤ `takeoff_speed_mps` to `takeoff_alt_m`, yaw held; the column above the
takeoff point is assumed clear like a surveyed dock zone, logged as a DN) → `SETTLE` (hover until loc is ok:
N fresh accepted fixes) → `NAV` (plan to the wall goal, follow, replan on new obstacles / deviation / 1 Hz) →
`ARRIVED` (hold facing B for `dwell_s`) → (`RETURN` → plan back above the takeoff point →) `LAND` (FC land mode) →
`DONE`. Any state: alignment invalid > `loc_lost_abort_s` → `HOLD` then `LAND` in place; emergency → hover, replan;
vehicle failsafe or leaving offboard → stop commanding (the pilot / FC has it). Every transition is logged
(JSON-able dict events).

## 8. E. Closed-loop sim (`flightsim/`)

- `world.py`: the map = `localization.synthetic_room` room with the default boxes (map cloud + landmark DB built from
  a mapping walk that covers the flight space, cached under `data/`); the WORLD = the same room + unmapped
  obstacles (append extra boxes AFTER the defaults so shared surfaces keep identical textures).
- `vehicle_sim.py`: `SimVehicle(VehicleInterface)`: PX4-like position control (P on position → velocity cap →
  first-order velocity lag → acceleration limit, tilt from acceleration so the camera pitches/rolls like a real
  multicopter), yaw rate limit; on the ground until armed + climbing; landed detection; offboard loss → hold; an
  estimator with x/y and yaw drift (random walk, seeded), the range height reference (odom z = height above the
  surface below, with the step lag `hgt_step_tau_s`), optional frame reset; camera images rendered at the TRUE
  camera pose with a delivery latency; collision check against the TRUE world (hull sphere vs all surfaces/boxes).
- `loop.py` + `scenarios.py` + `tools/run_scout_sim.py`: runs a scenario end to end through `Mission`, writes
  `data/sim_runs/<runid>/` (events.jsonl, trajectory.csv, metrics.json), prints JSON, exits 0 pass / 1 failed /
  2 could not run.
- Scenarios (A = fixed takeoff pose; B = a point on a wall; seeds vary noise / drift):
  1. `clear`: no unmapped obstacle → arrives; 0 emergencies, 0 obstacles created (false-alarm test).
  2. `low_box`: a wide low box (≈ 1.4 × 1.4 × 0.7 m) on the path, room beside it → over OR around, whichever costs
     less (in this room around is shorter); `low_barrier`: a 0.6 m barrier wall to wall → the only way is OVER.
  3. `tall_box`: a narrow tall box (≈ 0.5 × 0.5 × 1.9 m, person-sized) on the path → flies AROUND it.
  4. `late_obstacle`: an obstacle that appears ~1 m ahead mid-flight → emergency hover, replan, arrives.
  5. `drift`: strong odom drift + a slow height-step lag → still arrives (alignment works).
  6. `wrong_takeoff`: Scout placed 0.5 m / 20° off the fixed takeoff pose → `REFUSED`, never arms.
- Metrics: collision (any), min clearance hull↔world, arrival position / yaw error vs the goal pose, max speed,
  emergencies, obstacles created (true vs false), plan shapes used, alignment error (true vs estimated `T_map_odom`),
  localization fixes, wall time.
- Pass bars (provisional): no collision; arrival ≤ 0.15 m and ≤ 10°; scenario 1 zero obstacles and zero
  emergencies; `low_barrier` used `over` (and the true track crossed its footprint); 3 used `around` (and never
  entered its footprint); 4 had ≥ 1 emergency and still arrived; 6 refused.

## 9. Working rules for builders

- Python 3.9 compatible (`from __future__ import annotations`), numpy + OpenCV + SciPy only, no ROS imports.
- Read the existing `localization/` code and reuse it (geometry, camera, depth_render, synthetic_room); do not
  modify `localization/` files other than adding your own new ones, except where the brief says so.
- Do NOT commit; the main session reviews and commits. Do not edit files owned by another piece.
- Tests run as `py -3.9 tests/test_<piece>.py` (plain python, also pytest-compatible), and must pass before you report.
- Decisions: write them to `docs/decisions_scout_nav/<piece>.md` with ids in your range (A: DN-10..19,
  B: DN-20..39, C: DN-40..59, D: DN-60..69, E: DN-70..89): what, why, what it departs from.
- Report: files created, test output summary, open issues, any interface change you needed (ask; don't silently
  change `navigation/types.py`).

## 10. Status and results (2026-10-17)

Built: every piece above, plus the real-flight path: `navigation/build.py` (the ONE factory the sim and the real node
share), `drivers/px4_scout/` (PX4 frame math, unit-tested; the uXRCE-DDS `VehicleInterface`, run in PX4 SITL, below) and
`ros2_ws/src/nx_scout_nav` (`scout_mission` node; PX4 SITL run 2026-10-18, below). Decisions: `docs/DECISIONS_SCOUT_NAV.md`.

**Closed loop, 7 scenarios × seeds 0, 1, 2 = 21 runs: 21 pass** (`py -3.9 tools/run_scout_sim.py --scenario all
--seeds 0,1,2`, ~100 s per run on this laptop):

| Scenario | Result over 3 seeds |
|---|---|
| clear | arrives 1.8–2.4 cm / ≤ 0.16° from the goal; 0 obstacles, 0 emergencies (false-alarm test) |
| low_box | arrives ≤ 3 cm; around ×2, over ×1 (the planner's cost decides; both are valid there) |
| low_barrier | arrives ≤ 3 cm; flies OVER the wall-to-wall barrier every time |
| tall_box | arrives ≤ 2.2 cm; AROUND every time, never enters its footprint |
| late_obstacle | box appears 0.8 m ahead: emergency hover, escape, around, arrives ≤ 2.9 cm |
| drift | strong odom drift + a frame reset + a slow height-step lag: arrives 3.8–8.8 cm |
| wrong_takeoff | placed 0.5 m / 20° off the pad: REFUSED, never armed |

No collision in any run; minimum hull clearance 0.40 m. Unit tests: mission 7, navigation 20, alignment 11,
looming 9, flightsim 12, PX4 frames 6, localization 13 (unchanged).

**Known issues:**
- 3 of the 21 runs (none of them `clear`) made one false obstacle each, all against the mapped north wall (y ≈ 5.4–6.0)
  when the route around the box passed near it; mostly the map-free emergency path firing at a mapped wall on a turn.
  Cost: one needless 1 s hover. FIXED 2026-10-17 (night), see the last update below.
- Looming needs texture: a plain untextured object is invisible to it. The synthetic textures are richer than most
  real rooms.
- Map-anchored looming frames run 36–55 ms (target 40 ms) in Windows Python.
- Nothing has met real hardware or real images: lens distortion, blur, rolling shutter, lighting, real latency, PX4's
  own EKF2 behaviour over a terrain step and the real hull size are all unmodelled or placeholders.

**Next, in order:** (1) the PX4 adapter against PX4 SITL (offboard, arm, climb, land; frame conversions on real
firmware); (2) the Radxa ↔ FC link (R9) and the camera stream, then a props-off bench run of the node; (3) the real
prior map + landmark DB + camera calibration (`docs/SCOUT_LOC.md` list), then the pad pose in that map (R20);
(4) a tethered / pilot-supervised first takeoff, Ron's call.

### Update 2026-10-17 (evening): raised pad, round trip, and a late-obstacle regression

Ron: "assume the takeoff pad is like an elevated box; use the placeholder for the height and radius for now". The pad is
now a mapped 0.5 × 0.5 × 0.4 m box at A (provisional size, DN-88); the mission climbs to max(floor + 1.0 m, pad top +
0.5 m), refuses a pad pose that does not rest on a mapped surface, and a new `round_trip` scenario flies A → B → back
and lands on the pad box. Hull stays 0.20 m / 0.06 m (R19).

Suite on the raised-pad world, 8 scenarios × seeds 0, 1, 2: **21 of 24 pass**.
- `round_trip` 3/3: back on the pad top, touchdown 4.0–4.5 cm from the pad centre.
- clear, low_box, low_barrier (over ×3), tall_box (around ×3), drift, wrong_takeoff: all pass; arrival ≤ 7.2 cm.
- **`late_obstacle` 0/3 — REGRESSION, seed 2 collides with the late box.** Cause (traced frame by frame): the pad
  shifted the timing so the box now appears 0.8 m ahead at 0.505 m/s (1.6 s to contact) instead of 0.385 m/s; looming's
  emergency path never fires on the newly appeared box (0–4 corridor tracks, need 5; valid tracks collapse; the track
  history is wiped 0.3 s before contact). The earlier 3/3 pass was timing luck: a pop-in obstacle at full cruise speed
  is not reliably caught. Being fixed together with the wall false alarms in one owner of `avoidance/` (the
  "Stop Scout looming false alarms at mapped walls" session). Until then: treat 0.5 m/s past a possible pop-in as
  unproven. FIXED, see the last update below.

### Update 2026-10-17 (night): the PX4 adapter against real PX4 firmware (SITL)

`drivers/px4_scout/bench/run_sitl_check.sh [flight|fallback]` (WSL) runs PX4 v1.17 SITL with SIH (airframe 10040, no
Gazebo), the MicroXRCEAgent on UDP, SHM-only DDS, its own ROS domain (57), under the laptop's simulator lock. In flight mode,
`sitl_offboard_check.py` drives `Px4ScoutVehicle` through: stream setpoints, offboard, arm, climb to 1.0 m (odom),
hold 5 s, turn to face +x, +1 m in x, hold 5 s, land. `sitl_ulog_check.py` then reads PX4's own `.ulg` (the setpoints
the firmware received, plus SIH ground truth). SITL only: the bench arms (DN-07 still holds for the real aircraft).

**Final runs: flight PASS (25/25 gating checks, 10 reported), fallback PASS (4/4).** Artifacts are in WSL
`~/<artifacts>/scout_sitl_20261018T003743Z_flight` and `…T003845Z_fallback`. Four earlier flights failed. Each
failure traced to the bench or to SIH, and none to the adapter after the fix below.

Verified on the firmware:
- **Frames, incoming.** The adapter's pose equals `frames.py` applied to PX4's `VehicleOdometry`, to float precision
  (759/788 same-message samples, rotation error 0). In the climb, odom z is +0.89 and PX4's NED z is −0.89.
- **Yaw.** Odom yaw 0 reads as PX4 yaw +91° (nose east), and body x points along +x odom.
- **Frames, outgoing.** PX4's logged `trajectory_setpoint` holds exactly NED (y, x, −z) and yaw π/2 − yaw for every
  hold setpoint.
- **Ground truth.** Over the +1 m odom-x ramp, the aircraft truly moved 0.95 m east (direction −0.1°). Its true
  heading was east while the odom yaw setpoint was 0.
- **Commands.** Offboard (176), arm (400) and land (21) were each ACCEPTED in `vehicle_command_ack`. The status flags
  followed: offboard 0.9 s after the request, armed 0.04 s, AUTO_LAND 0.2 s; landed and auto-disarmed 8.9 s after the
  land request.
- **Topic discovery.** PX4 1.17 publishes the status as `/fmu/out/vehicle_status_v1`, and `find_topic`'s `_vN` match
  found it.
- **Range.** `dist_bottom_m` equals the raw range × cos(tilt) (p90 error 2.9 mm) on 100 % of the hold samples. With
  SIH's rangefinder silenced, the adapter still starts, and `dist_bottom_m` stays None for the whole run.

**Bug found and fixed (`px4_scout_vehicle.py`).** Before PX4's first `vehicle_status` arrived, `status()` reported
`failsafe=True` (treated as stale). In SITL that first message came 0.45 s after the topic was found. `mission.py`
aborts on a failsafe in every state but LAND, so the real node would have gone ABORTED in PREFLIGHT on its first
20 Hz tick. The 1 s status timeout now runs from the end of construction until the first message arrives. Until then
the fields keep the safe defaults (disarmed, landed, not offboard), and `info["status_received"]` says so. Tests:
`py -3.9 drivers/px4_scout/test_px4_scout_vehicle.py` (11 tests, fake `px4_msgs`); 3 of them fail on the old adapter.

**Findings (not adapter bugs):**
- **OFFBOARD after landing.** After the land-mode auto-disarm, PX4 sets `nav_state` back to OFFBOARD. So
  `status().offboard` is True while the aircraft is landed and disarmed. The mission handles it (LAND → DONE on landed
  and disarmed), but no code should read `offboard` as "flying".
- **SIH's estimator is poor on this host.** Its p95 error against truth is 0.19 / 0.25 m in xy, 0.13 m in z and
  0.3–0.4 m/s in velocity; the sim GPS carries 0.2 m white noise. The aircraft chases that error and wanders 0.3–0.7 m
  in a hold. Position-tracking numbers are therefore reported, not gated. The truth checks are sign/axis tests (45° on
  direction, 15° on heading): an axis swap would read 90° and a sign flip 180°.
- **The Scout's height reference could not be flown in SIH.** With `EKF2_HGT_REF=2` (range), SIH's rangefinder sits
  at the body origin and reads −0.0002 m on the ground. PX4 then reported "Preflight Fail: height estimate not
  stable" for 120 s. The SITL flights therefore ran on SIH's default baro/GPS height (`SCOUT_HGT=range` keeps the
  option). The real FC's range-height behaviour, including the terrain step in §3, is still untested.
- **Odometry rate.** Odometry arrives at 96 Hz through the agent on one host (p99 gap 13 ms). The Wi-Fi jitter of
  DN-05 is unmeasured.

Not covered: the `scout_mission` node itself (it needs a prior map, landmarks and a camera), the real Radxa ↔ FC link
(R9), and optical-flow xy (SIH has no flow sensor). Next-list item (1) above is done; (2) is now first.

### Update 2026-10-17 (late night): the north-wall false alarms and the late-obstacle regression fixed

**Suite on the raised-pad world, 8 scenarios × seeds 0, 1, 2: 24 of 24 pass, 0 false obstacles, 0 collisions**
(before: 21/24, false obstacles in 3–4 runs). late_obstacle 3/3 with one emergency each; clear and round_trip 0
emergencies; arrival 0.9–7.3 cm. Unit tests: navigation 22, mission 10, flightsim 12, looming 11.

How it was found: the looming inputs of failing closed-loop runs were recorded at the image stamps together with
the true depth, and every variant was replayed on the same frames (the sim is deterministic per seed and code).
Causes and fixes (DN-35..37 in `decisions_scout_nav/looming.md`, DN-57..59 in `navigation.md`):
- **Diagonal approach to the mapped wall.** The follower's TTC law cast one centre ray; the emergency judges
  contact over its ±0.35 m corridor, whose edge meets an oblique wall first (true 1.65–1.85 s at the allowed speed).
  The law now also bounds the pure TTC over that corridor (9 rays) (DN-57).
- **Short escape hops toward the wall** were flown facing the wall at 0.12–0.18 m/s; a hop < 0.3 m is now flown
  facing the next segment (DN-58).
- **Looming measured a feature's NEIGHBOURS' depth** (the anchor's depth cancels in a pair): at a pop-in box's
  silhouette the box read the background, so the emergency never fired. Each feature now takes the pairs that
  measure IT, in the emergency and the map path; pairs 16 within 120 px; aperture check 0.2 → 0.1 (DN-35). The
  tall box is now boxed from 3.35 m (was 1.99 m).
- **The odometry's motion direction is poor at low speed** (median 8°, up to 47° off in a slow creep; the drift
  scenario's bias made the corridor point at the wall). The direction now comes from the image's focus of expansion
  (median error 0.1–0.4°), the magnitude from the odometry's HORIZONTAL displacement (the rangefinder z steps at every
  terrain edge, e.g. leaving the pad) (DN-37). Still image + FC odometry only; no map, no PnP.
- **The map path's odometry scale** is measured on the mapped scene (median measured / predicted), so a flow-scale
  error is not read as excess everywhere (DN-36).
- **Mission, found on the way:** with no plan it fed the current pose to the gate, whose hold then never fired and
  the velocity feed-forward kept the drift (12 s at 0.24 m/s, over crate_c, into the ceiling); and after an emergency
  it replanned from the follower's stale reference 0.45 m ahead, toward the obstacle (late_obstacle seed 1 flew into
  the box). Now: no setpoint → the gate's hold, and every hold puts the reference back on the body (DN-59).

Still open: low_box seeds 0 and 2 each make one EMERGENCY with no obstacle (a needless 1 s hover; the map comparison
suppresses the box). The route crosses crate_c's north edge 0.28 m above its top while the rangefinder height step
dips the Scout ~0.1 m/s, so there is real closing motion toward the crate top. The wall-approach margin is thinner
than it should be on real data: on the recordings the lowest estimated contact time at a mapped wall is 1.5–2.0 s
for a true ≥ 2.0 s.

### Update 2026-10-18: the `scout_mission` node end to end in PX4 SITL

`drivers/px4_scout/bench/run_sitl_check.sh mission` (WSL; same lock, PX4 SIH, agent, domain 57 and teardown as the
adapter check) runs the REAL node (`ros2_ws/src/nx_scout_nav/nx_scout_nav/mission_node.py`) with `auto_arm:=true`
(SITL only, DN-07) and `on_arrival:=return_land`. Inputs: the flightsim room WITHOUT the pad box (SIH's ground is flat;
cached map side, 221k map points, 372k landmarks; pad A = (1.0, 4.0), goal B1 = (8.0, 3.6, 1.2)), exported by
`sitl_camera_sim.py --export`. The camera is `sitl_camera_sim.py`. It renders that room at 10 Hz and publishes each frame
a constant 0.25 s after capture (= the node's `image_latency_s`). The pose it renders at is SIH's ground truth
(MAVLink HIL_STATE_QUATERNION, `SCOUT_CAM_POSE=truth`), or PX4's EKF pose (`odom`), which hides SIH's estimator error.
`sitl_mission_monitor.py` records the node's events and exits 0 on DONE. Height: SIH's baro/GPS (`height_ref:=baro`,
a new node parameter; the real Scout stays `range`).
Code: the shared working tree at 01:41–02:13 UTC, not one commit: base c9befe8 (705e687, the late-night looming
fixes above, landed during the last run) plus other sessions' uncommitted edits then in `navigation/`, `avoidance/`,
`flightsim/` and `config/`. This commit adds only the node fix and the bench.

**Result: the node reaches DONE through the whole mission, PREFLIGHT → ARMING → TAKEOFF → SETTLE → NAV → ARRIVED →
RETURN → LAND ("above the pad") → DONE, with no PX4 failsafe (final run, truth camera, 215 s).** It took three node
fixes. Six runs, artifacts in WSL `~/<artifacts>/scout_sitl_20261018T*_mission*`:

| Run | Camera / node | Outcome |
|---|---|---|
| 014106Z | truth / as committed | **ABORTED** in NAV after 73 s: PX4 switched OFFBOARD → AUTO_RTL six times (failsafe), setpoint gaps up to 1.12 s |
| 014918Z | truth / + frame stamps, tick keepalive | DONE by **landing in place**: HOLD (every fix rejected) → LAND after `loc_lost_abort_s`. SIH's EKF z read +0.74 m while the aircraft still sat on the ground, and up to 1.23 m after landing (z gate 0.15 m). The node did the safe thing |
| 015531Z | odom / same | **ABORTED** 1.6 s into NAV: offboard loss again, a 1.10 s gap while a tick planned |
| 015939Z | odom / + StreamKeeper | DONE, full sequence; RETURN → LAND by the 300 s mission limit. Ticks ran at 2–4 Hz, NAV took 84 s |
| 020626Z | odom / + 0.5 ms GIL switch interval | DONE, full sequence, LAND "above the pad"; NAV 33 s |
| 021001Z | truth / final | **DONE, full sequence**, one 2.3 s HOLD in NAV (alignment invalid) and back; LAND "above the pad"; max setpoint gap 0.33 s |

**Node bugs found and fixed** (`nx_scout_nav/node_logic.py`, ROS-free; `py -3.9 tests/test_scout_node.py`, 7 tests):
- **The setpoint stream depended on the mission's timing.** The 20 Hz tick and the camera callback share one lock. A
  frame (localization + looming, 0.4–1.1 s here) or a planning tick (up to 1.9 s) stopped the stream past PX4's
  COM_OF_LOSS_T = 1 s (also 1 s on the real Scout's FC). PX4 went to RTL, and the mission aborted on the failsafe.
  Fix: the mission now talks to the FC through `StreamKeeper`, a vehicle wrapper whose own watchdog thread re-sends
  the last setpoint (position + yaw, no feed-forward) whenever nothing was sent for 0.1 s in a streaming state. The
  first fix, a keepalive inside the tick thread, covered frames but not slow ticks (run 015531Z). After it: every
  run's largest gap ≤ 0.34 s, none > 0.5 s, no failsafe.
- **Frames were stamped when their callback ran, not when they arrived.** With one frame in processing, the next
  waited in the reader up to a processing time. The fixes were late by that much, and the alignment rejected them
  (xy residual 0.18–0.35 m at 0.5 m/s). Fix: the stamp is now the DDS reception time (rclpy MessageInfo; the callback
  must REQUIRE the second argument, or rclpy silently passes none: a test checks it) minus `image_latency_s`.
- **GIL starvation.** With Python's 5 ms switch interval, the frame thread starved the tick. `mission.tick` took up to
  1.9 s, ran at 2–4 Hz, and frames took ~400 ms (170 ms offline on the same host). The new parameter
  `gil_switch_interval_s` (default 0.5 ms) cut frames to 140–200 ms and ticks to p50 2–15 ms, but the tick rate is
  still 4–10 Hz, not 20 (the keeper carries about half the stream). A mitigation, not a fix: vision and control share
  one interpreter.
- Smaller: the node waits for PX4's first `vehicle_status` before starting the mission. Before it arrives, the status
  is defaults, and a pilot-armed vehicle would read as disarmed. Every 10 s the node now logs a `node_stats` event
  (frame processing / lock wait, reception lag, tick durations, setpoint gaps, keepalives).

**Findings (not fixed here):**
- **SIH's estimator, not the node, limits the truth-camera runs.** EKF − truth: xy p95 0.47–0.61 m, z p95 0.13–1.16 m
  (the ground artefact above: the EKF climbs while SIH holds the aircraft on the ground), yaw p95 12–17°. The
  alignment rejects fixes it cannot reconcile (z gate 0.15 m, yaw 5°), and the mission holds or lands in place, as
  designed. Its 0.10 m arrival tolerance is reached only by luck in a 0.3–0.5 m hover wander. Run 015939Z timed out
  over the pad that way.
- **Yaw swings near the goal and over the pad (`navigation/`).** In the final run, PX4's logged yaw setpoint went
  −43 → −80 → −146 → +4° while the aircraft hovered at the goal. Over the pad it swung through ±180°. Each 1 Hz
  replan from a wandered hover point is a short leg in a new direction, and yaw-first turns the aircraft toward it.
  The 2-cm flightsim hovers never trigger this; a real EKF's hover error will, to the extent it wanders.
- 189 `safety` clamps (`max_step`) in the final run, from coarse ticks. 70 plans, including `over` with no unmapped
  obstacle.
- After the land-mode disarm, PX4 again set `nav_state` back to OFFBOARD (as in the adapter check). The mission's
  LAND → DONE is unaffected.

Not covered: the real FC link (R9), a real camera (distortion, blur, real latency), Point-LIO maps, range height,
obstacles (no unmapped box in the SITL room) and the laptop's real CPU load. Next-list item (2) is unchanged.

### Update 2026-10-17 (night): hover-test mode

New mode `hover_test` (DN-60). Sim scenario `hover_test` (raised pad, rise 2 m, 10 s, land), 3 seeds: **3/3 pass**:
hover mean 0.9 cm above the 2 m target, worst 5 cm; sideways drift ≤ 2 cm; back on the pad top 7–8 cm from its centre;
no obstacles, no emergencies. Found and fixed on the way:
- the alignment held map z 5.5 cm low over the pad (a box-edge voxel's low point mean won the candidate merge; DN-14c);
- the climb ramp stopped at full speed and overshot ~14 cm; it now decelerates into the target (DN-61);
- `round_trip` regression: a map-free emergency box on wall_x0 swallowed the pad hover point and the Scout landed
  off the pad; such boxes (they contain a mapped surface) are no longer remembered (DN-62). round_trip 3/3 again
  (touchdown 3.3–4.0 cm from the pad centre); late_obstacle 3/3. The needless emergency at the mapped wall on the
  return leg itself is with the looming owner (session "Scout: needless emergency hover over crate_c's edge").
- Note: emergency counts differ between repeated runs of the same seed under different CPU load: the sim is not
  bit-deterministic (timing-dependent), so bars are per-run thresholds, never equality.

Full suite on this code (9 scenarios × 3 seeds, before the DN-62 fix): 24/27, the 3 failures = round_trip, fixed above.

### Update 2026-10-18: hover_test mode in PX4 SITL

`SCOUT_MISSION_MODE=hover_test SCOUT_CAM_POSE=truth run_sitl_check.sh mission` (node `mode:=hover_test
hover_climb_m:=2.0 hover_s:=10.0`, Mission side 6e05345). Room without the pad box, so the climb is from the floor:
target z = 0.056 + 2.0 = 2.056 m (map ceiling 3.0 − 0.5 margin = 2.5, not capped). Artifact: WSL
`~/<artifacts>/scout_sitl_20261018T025420Z_mission_truth_hover_test`. `sitl_mission_monitor.py` now writes
`track_by_state` (SIH truth per state visit, and the touchdown).

**DONE: PREFLIGHT → ARMING → TAKEOFF (9.3 s) → HOVER (10.05 s) → LAND (12.1 s) → DONE**, no PX4 failsafe (only
OFFBOARD → AUTO_LAND on the land request), max setpoint gap 0.28 s.
- True height in HOVER: mean 2.00 m (−6 cm against the target), range 1.81–2.17 m.
- Sideways, true distance from the takeoff spot in HOVER: mean 0.43 m, max 0.51 m. It is PX4's tracking on SIH, not
  the alignment. The alignment followed SIH's EKF error: the setpoint moved −0.03 → −0.22 m east as the EKF error
  moved −0.03 → −0.25 m. But PX4's own EKF position sat ~0.3 m from the setpoint while it chased GPS noise.
- Touchdown 1.18 m from the takeoff spot. The slide happened in PX4's AUTO_LAND, after the mission handed over: the
  aircraft moved ~0.7 m even in PX4's own frame, plus SIH's EKF error. The real Scout lands on its own FC the same
  way, so whether it lands ON the pad depends on the FC's land-mode position hold (and on R20's pad).
- Localization: 5 fixes rejected, none of them disturbing HOVER (3 stamp_outside_odom_buffer, frames 300 ms here;
  2 on the ground after landing: SIH's EKF z read +0.29–0.38 m there). One EKF reset 1 s before HOVER, handled. No
  HOLD.

### Update 2026-10-18: the last needless emergencies (crate_c's edge, the return to the pad)

**Suite, 9 scenarios × seeds 0, 1, 2: 27/27 pass, 0 false obstacles, 0 collisions, and no emergency outside
late_obstacle** (late_obstacle 1 / 2 / 1). Min hull clearance 0.36 m; arrival ≤ 2.3 cm except drift (5.1–9.4 cm).
Unit tests: looming 11, navigation 23, mission 15, flightsim 12, map_align 12.
- **low_box seeds 0 and 2, the emergency over crate_c's edge (DN-63, `decisions_scout_nav/mission.md`).** The fix is
  in the mission's terrain handling, not the planner or looming. The dip was commanded. DN-08's velocity
  feed-forward differentiated the odom setpoint, which moves by the whole terrain step when the FC's range height
  re-references to the box top. So a 0.7 m edge came out as up to 0.6 m/s of commanded descent: a true 0.27 m dip,
  0.25 m above the crate. Leaving the pad, the same mechanism gave a +0.34 m/s bump. The feed-forward now leaves out
  the height reference (`MapOdomAligner.height_reference_z`). Same route, with and without: dip 0.273 → 0.007 m, pad
  bump +0.34 → +0.09 m/s, low_box 1 → 0 emergencies. Open on hardware: this assumes PX4's velocity estimate does not
  carry the terrain step (EKF2 height reset), the same unknown as §3.
- **round_trip (seed 2 here; seeds 0 and 1 in the main session's runs): the emergency at wall_x0 on the return (DN-53
  amended, `decisions_scout_nav/navigation.md`).** In the last metre the follower turned the yaw toward the pad's
  goal yaw, 180° from the motion, at 45°/s while still closing at ~0.45 m/s. The wall ahead then sat at the image
  border, where looming read its depth ~2× short (TTC 1.33 s estimated, ~2.3 s true). While moving, the
  final-approach yaw now stays within the 20° heading gate of the motion, and the turn finishes in place.
- **The sim IS deterministic per seed and code.** The same round_trip seed 2 run reproduced the same emergency (stamp
  39.2 s, TTC 1.33 s) three times under different CPU loads; wall time feeds only timing reports. Count differences
  between repeated runs came from code changing between them.

Still open: the wall-approach margin on real data (on the recordings the lowest estimated contact time at a mapped wall
is 1.5–2.0 s for a true ≥ 2.0 s). Old text, retained: "And obstacle memory: a merge keeps the max top, so one
ceiling-extended unseen-top detection outlives later seen-top detections of the same 0.6 m barrier. Its planner is
then boxed in (low_barrier seed 0 on a tree without DN-61/62, landed after `no_path_land_s`). Not in today's suite; a
follow-up." Fixed 2026-10-18 (below).

### Update 2026-10-18: obstacle memory, an unseen top no longer outlives the seen tops (DN-51 amended)

The box top in obstacle memory is now the max of the SEEN tops (never lowered) and the live UNSEEN tops. Each
top_seen=False detection is kept with its own footprint. A LATER seen-top detection retires it when it contains that
footprint's centre and covers at least half of its area (`unseen_top_cover_frac`, PROVISIONAL;
`decisions_scout_nav/navigation.md`). A seen top of a low box therefore never lowers a merged tall neighbour, and an
unseen top that arrives after the seen ones raises the box again.
- **The failing case, reproduced:** low_barrier seed 0 on 705e687 + the DN-63 feed-forward. Before: both barrier
  pieces got one ceiling-height unseen detection, the planner was boxed in from 10.3 s, and the Scout landed after
  `no_path_land_s`. After: the unseen top was retired at 6.9 s, and the Scout flew OVER and arrived at 19.4 s with 0
  emergencies.
- **Suite, 9 scenarios × seeds 0, 1, 2: still 27/27, 0 false obstacles, 0 collisions, emergencies only in
  late_obstacle (1 / 2 / 1).** Unit tests: navigation 25 (+2: the event sequence, which plans over afterwards and fails
  on the old merge; and a tall-neighbour negative test), mission green.
- Accepted limit: one wide unseen detection that is the only evidence of a tall neighbour, and lies mostly over the
  low box, is retired by the low box's seen top. The backstops are re-detection while closing (which raises the top
  again), the dynamic vertical margin and the looming emergency.

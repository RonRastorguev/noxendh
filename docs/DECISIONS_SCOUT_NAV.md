# Decisions log — Scout autonomous flight (takeoff → A to a wall point B, looming avoidance)

Judgment calls for `docs/SCOUT_NAV.md`. Same rules as `docs/DECISIONS.md`: what, why, what it departs from.
Ids `DN-nn`. DN-01..09 are the main session's; each piece keeps its own detailed log:

| Ids | Piece | File |
|---|---|---|
| DN-10..19 | map↔odom alignment (`localization/map_align.py`) | `docs/decisions_scout_nav/map_align.md` |
| DN-20..39 | looming (`avoidance/`) | `docs/decisions_scout_nav/looming.md` |
| DN-40..59 | navigation (`navigation/` planner, follower, safety, memory, goal) | `docs/decisions_scout_nav/navigation.md` |
| DN-60..69 | mission (`navigation/mission.py`) | DN-60..62 this file, below; DN-63.. `docs/decisions_scout_nav/mission.md` |
| DN-70..89 | closed-loop sim (`flightsim/`) | `docs/decisions_scout_nav/flightsim.md` |

Everything is tested on SYNTHETIC data in a pure-Python closed loop only; every number is provisional.

## DN-01 — The FC flies on its own estimate; the laptop aligns the map to it (no vision fusion into PX4 yet)

The Scout's PX4 keeps estimating from IMU + optical flow + rangefinder. The laptop estimates the 4-DoF transform
`T_map_odom` from PnP fixes (DN-10..19) and sends position setpoints in the FC's own frame. Why: no FC parameter change is
needed (EV fusion is `EKF2_EV_CTRL` etc. = a change set for Ron's go), a bad PnP fix can at worst slide a setpoint (rate
limited) instead of corrupting the FC's estimate, and if the laptop / link / Radxa dies the FC still holds on its own
estimate (DH-05). EV fusion stays the stage-2 option in `SCOUT_STABILITY_PORT.md` A.

## DN-02 — Looming is generic (any textured object), in two paths; the person-only YOLO path is not built

Departs from ARCHITECTURE §8.4 B (YOLO person bbox height). Ron asked for "obstacle avoidance via looming (fly over the
object or around it)", which needs any object, its extent and whether its top is visible. Two outputs (DN-20..34):
- **Emergency**: image expansion + the FC's attitude/velocity only, never the PnP pose or the map, so a wrong pose cannot
  silence it (a safety instrument must be independent of what it protects against).
- **Map-anchored obstacles**: features that loom faster than the prior map predicts → boxes in map with `top_seen`,
  which the planner uses to go over or around.
Monocular depth (§8.4 A) and YOLO remain possible extra obstacle sources behind the same `Obstacle` type.

## DN-03 — The rangefinder is the FC's height reference, so map z goes through "the surface under the beam"

From the Scout's dump: `EKF2_HGT_REF=2`, `EKF2_RNG_CTRL=2`. PX4's z is height above whatever is below; flying over a
0.7 m box drops it by 0.7 m. The alignment converts setpoints with the surface under the vehicle NOW, chosen consistently
with the rangefinder reading (DN-14, DN-14b), and the planner keeps extra vertical margin over obstacles. Moving the FC to
a baro height reference would remove this, but is a parameter change and a judgment for Ron; not assumed.

## DN-04 — The column above the fixed takeoff pad is assumed clear

The forward camera cannot see above the Scout, so the vertical climb from the pad to `takeoff_alt_m` (1.0 m) is not
checked by any sensor. Same reasoning as a certified dock clearance zone: the pad is a FIXED, surveyed spot, and
its siting is the guarantee. The mission refuses to start if PnP says the Scout is not on the pad (> 0.30 m / 15°).

**Amended 2026-10-17 (Ron: "assume the takeoff pad is like an elevated box"):** the pad is a raised box that the prior
map contains. The climb target is max(floor + `takeoff_alt_m` 1.0, pad top + `takeoff_min_climb_m` 0.5), with the pad
top read from the map under the pad pose; the mission refuses if the configured pad pose does not rest on a mapped
surface (0..0.25 m above it). Return-to-pad hovers at the same height, then PX4's land mode lands on the box top (with
the rangefinder height reference PX4 sees the box top as the ground). Sim: DN-88. The hull size stays the 0.20 m /
0.06 m placeholder for now (Ron, same day; R19).

## DN-05 — One clock: the laptop's receive time minus a configured latency

Odometry (FC → Radxa agent → Wi-Fi → laptop) and camera frames (Radxa → laptop) are both stamped with the laptop's
receive time minus a fixed latency (`odom_latency_s`, `image_latency_s`), so no FC↔Radxa↔laptop clock sync is needed.
The error is the link jitter (~1–2 cm at 0.5 m/s). A full FC clock corrector is the upgrade path;
measure the Scout's jitter before flying faster or trusting tighter numbers.

## DN-06 — A wall point B means "hover at a standoff in front of it, camera on it"

`wall_goal` (DN-4x): the wall normal at B from the map (PCA), hover at B + 0.8 m × normal, yaw facing B, z clamped into the
flight band. The follower's speed law keeps the hull-contact time to the goal wall above the looming threshold, so the
goal wall never trips an emergency.

## DN-07 — Pilot arms by default; the software takes off by itself

`auto_arm: false`: the mission streams setpoints, switches to offboard and waits for the pilot to arm on the RC, then
climbs, settles and flies by itself. `auto_arm: true` exists (the sim uses a simulated pilot). Why: the RC kill switch
(ch 8) and the offboard switch (ch 6) stay the pilot's; an agent never arms the real aircraft, and the first autonomous
takeoff is Ron's decision.

## DN-08 — Velocity feed-forward on every map setpoint

The odom setpoint moves when the follower moves it AND when the alignment slides it to cancel the FC's drift. PX4's
position loop is proportional (`MPC_XY_P` 0.95), so a setpoint moving at v trails by v / 0.95: in the strong-drift
scenario the Scout sat 16 cm off the goal while its own pose error was 5 cm. The mission now sends the odom setpoint's
rate of change (low-passed, ≤ 0.6 m/s) as `TrajectorySetpoint.velocity`.

Amended 2026-10-17 by DN-63 (`decisions_scout_nav/mission.md`): the feed-forward leaves out the height reference's changes (a terrain step under the
rangefinder re-references PX4's z; it is not motion, and fed forward it made the Scout dip 0.27 m at a crate edge).

## DN-09 — Closed-loop pure-Python sim instead of PX4 SITL + Gazebo for this stage

There is no Gazebo Scout model, and the laptop runs one simulator at a time; a Gazebo Scout would also need a
camera that renders textures PnP can use. `flightsim/` renders the textured synthetic room exactly, models PX4's position
control, drift, the rangefinder height reference, offboard rules and collisions, and runs the real flight software end
to end at ~2× slower than real time. What it cannot show: real PX4 behaviour (jerk limits, EKF2 step handling, failsafes),
real images (distortion, blur, rolling shutter, lighting), real latency. The PX4 adapter (`drivers/px4_scout/`) is the
next thing to check against PX4 SITL, then the real FC.

## DN-60 — Hover-test mode (Ron 2026-10-17)

"A new mode where it just takes off, hovers, and lands after hovering for 10 seconds, all in the same spot. maybe rising
2 meters." `mode: hover_test`: PREFLIGHT (same pad checks) → ARMING → TAKEOFF straight up to the takeoff pose + 2.0 m
→ HOVER at the pad's x, y for 10 s → LAND (PX4's land mode, back onto the pad) → DONE. No goal point, no looming (no
sideways motion), localization still corrects the hold. The target is capped at the mapped ceiling minus
`ceiling_margin_m` (logged `hover_height_capped`): in a 3 m room with the 0.4 m pad the 2 m rise fits (2.46 m < 2.5 m).
On the real Scout the 2 m column above the pad is unobserved by any sensor (the camera looks forward): the same
"siting is the guarantee" as DN-04, now 2 m tall.

## DN-61 — The climb ramp decelerates into its target

TAKEOFF used a constant 0.3 m/s ramp that stopped abruptly; with the velocity feed-forward (DN-08) PX4 kept climbing and
overshot ~14 cm at the hover height. The ramp now limits v ≤ sqrt(2 · `takeoff_decel_mps2` (0.25) · remaining).

## DN-62 — A map-free emergency obstacle that contains a mapped surface is not remembered

The emergency path is map-free by design (DN-02), so the box it places when it fires often IS the mapped wall the
Scout was closing on. In memory it inflates over the goal and blocks every plan (sim round_trip: the pad hover point
1 m from wall_x0 → "goal inside obstacle" for 20 s → landed in place, 0.58 m off the pad). The mission now drops a
`looming_raw` box whose interior (above the floor) contains occupied voxels of the prior map, and logs
`obstacle_explained_by_map`. The emergency HOVER is untouched; map-anchored (`looming_map`) obstacles are untouched. Cost:
a real unmapped object pressed against a wall is not remembered from an emergency alone (the emergency still stops
the Scout every time it closes on it).

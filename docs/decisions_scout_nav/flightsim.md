# Decisions — piece E, closed-loop flight sim (`flightsim/`)

Ids DN-70..DN-89 (docs/SCOUT_NAV.md §9). What, why, what it departs from. Every number here is PROVISIONAL: a model
of PX4 and the Scout, not a measurement. Nothing in `flightsim/` talks to hardware.

## DN-70 — One sim clock, fixed step order

Physics and the estimator integrate at 100 Hz (`vehicle.rate_hz`); `mission.tick(now)` runs at 20 Hz; the camera
captures at 10 Hz and delivers each image `latency_s` (0.1 s) later, stamped at capture. All stamps (odometry,
images, ticks) are the same sim clock. Per physics step: scenario hooks → camera capture (TRUE pose) → deliver due
images (`mission.on_image`) → `mission.tick` (every 5th step) → `vehicle.step()`. Steps are integer-indexed, so
tick and capture instants never drift. Nothing depends on wall time, so a seed fixes the run.

## DN-71 — A faster renderer beside `synthetic_room.Renderer` (not an edit of it)

`flightsim/render.py` `FastRenderer` draws the same image (pixel-identical on the poses tested; the test allows
≤ 2 grey levels at surface intersections) at ~60 ms instead of ~160 ms per 640×480 frame with 2× supersampling:
the depth test runs on 1/z (no division, no inf buffer) and a surface partly behind the camera is clipped against
the near plane before its pixel box is taken (the original composites the whole frame for every such surface — the
floor, ceiling and both side walls, always). It also takes an `active` mask (obstacles appear / disappear). The
landmark DB is still built with the original Renderer. `localization/` is unchanged.

## DN-72 — Takeoff pose A = (1.0, 4.0) facing +x; goal B1 = (8.0, 3.6, 1.2) on wall_xL

The brief suggested A ≈ (1.0, 3.0). From y = 3.0 the straight line to wall_xL runs through the mapped pillar
(3.8..4.2, 2.8..3.2), so every plan would be "around" and `clear` could not be told apart from `tall_box`. From
(1.0, 4.0) the line to the hover goal (~7.2, 3.6) passes 0.61 m from the pillar (0.41 m from the hull; the static
inflation is hull + 0.3 = 0.5 m) and south of crate_c: with no unmapped obstacle the direct path is free. A is on
the floor, body origin at `scout_hw.yaml derived.fc_height_at_rest_m` (0.0559 m). B1 is on textured wall, off tag 2
(8, 3, 1.4). B2 (8.0, 2.2, 1.2) and B3 (8.0, 4.0, 1.0) are exposed for other runs. Config: `scout_sim.yaml world`.

## DN-73 — The sim mapping walk adds three segments to the default four laps

`world.sim_mapping_walk`: the default `mapping_walk` (4 ellipse laps at 1.25 m) never comes closer than ~1.7 m to
wall_xL and never below 1.2 m. Added: an outer "look-out" lap close to the walls (radii 3.0 × 2.2 m, 1.35 m, yaw
−90° and 0°), a lawnmower sweep 1 m in front of wall_xL at 0.9 m and 1.5 m facing it (the goal views from 0.8 m),
and a **takeoff column**: the rig held at A (and two spots near it) from floor level (lidar 0.14 m → camera 6 cm,
the Scout's camera height on the pad) to 1.4 m, yaw −30/0/+30°. The column departs from a pure handheld walk: the
mapping operator puts the rig on the pad spot. Without it the ground fix (the PREFLIGHT check, the wrong-takeoff
refusal) relies on views from 1.2 m+. Result: 317 keyframes (default 152), 372 k landmarks, ~70 s to build; cached
under `data/sim_cache/<sha1 of the parameters>/` (landmarks.npz, map_xyz.npy, meta.json). Cost: a global
relocalization is ~440 ms (default DB ~280 ms); local tracking ~80 ms. Ground fixes on A and on the wrong-takeoff
spot: < 3 mm, global.

## DN-74 — World = the map room with obstacles appended; late obstacles are parked, not rebuilt

`make_room` draws every texture from one RNG in surface order, so boxes appended after `DEFAULT_BOXES` leave every
shared surface identical (`check_texture_identity` renders map room vs world room with its obstacles hidden: max
diff 0). A box's textures depend only on its size, so a late obstacle is generated with the world, parked outside
the room, masked out of the renderer and ignored by terrain / collision; when it appears its surfaces are translated
in place and switched on (`SimWorld.place_obstacle`). No texture regeneration or renderer rebuild mid-flight.

## DN-75 — Hull = vertical cylinder from the gear to +hull_half_height_m (not a sphere)

The brief says "hull sphere". A 0.20 m sphere around an FC 5.6 cm above the floor penetrates the floor at rest, so
every takeoff and landing would be a collision and the floor clearance would be meaningless. The cylinder (radius
`airframe.hull_radius_m`, from `fc_height_at_rest_m` below the FC to `hull_half_height_m` above) is the prop disk +
legs. `hull.shape: sphere` remains available. Clearance is split: **lateral** (walls, ceiling, box sides —
`min_clearance_m`) and **below** (floor / box tops under the hull — `min_height_above_surface_m`, counted only while
moving horizontally outside land mode, since takeoff and land-mode descents legitimately reach 0). Collision = any
lateral penetration, or a touchdown outside land mode (the mission lands only through the FC's land mode). Checked
at every physics step; `end_on_collision` stops the run. Both hull numbers are placeholders (NEEDS_RON R19).

## DN-76 — Vehicle control model

P on the ESTIMATED odom position (MPC_XY_P 0.95, MPC_Z_P 1.0) + the setpoint's velocity feed-forward → caps (1.5
xy, 1.0 up, 0.7 down) → first-order velocity response τ 0.25 s → acceleration limit 3 m/s² (xy and z). The true
velocity target is `R_e^T (v_sp − bias) / scale`: the FC tracks its own (biased, rotated, scaled) velocity
estimate, so estimator drift moves the TRUE vehicle as on the real aircraft. Roll/pitch = the tilt of the thrust
vector `a + g` (lag 0.08 s, ≤ 35°); yaw P 2.8 at ≤ 60°/s. Not modelled: rotor/thrust dynamics, PX4's trajectory
generator and jerk limits, wind, ground effect, battery sag, the land detector's thrust logic, sensor dropouts.

## DN-77 — Flight-controller rules

Offboard is accepted only after ≥ 1 s of continuously streamed setpoints (a gap > 0.5 s restarts the count); no
setpoint for 1 s (COM_OF_LOSS_T) → Position mode holding the current estimate, `failsafe=True` (COM_OBL_RC_ACT 0).
Arming only in offboard (`arm_requires_offboard`) or by the simulated pilot: `pilot_arm_after_s` 0.5 s after
offboard engages (scout_mission.yaml has `auto_arm: false`, so the sim pilot arms; with auto_arm true the mission
does). Liftoff when armed and the commanded climb > 0.05 m/s. Auto-disarm 2 s after landing (COM_DISARM_LAND) and
10 s after arming without takeoff (COM_DISARM_PRFLT): PX4 DEFAULTS, not read from the Scout's dump.

## DN-78 — Estimator drift model (invented, seeded)

Odom origin = power-on position + N(0, 0.5 m), odom yaw uniform random (`init_yaw: random`): no code may assume
odom ≈ map or odom x = the pad heading (`heading` mode = PX4 without a magnetometer). x/y: Gauss-Markov velocity
bias σ 0.01 m/s, τ 30 s, plus a flow scale error σ 2 %, integrated only while airborne. Yaw: a constant gyro rate
bias σ 0.008°/s (~0.5°/min) + random walk 0.005°/√s. The `drift` scenario: σ 0.05 m/s τ 20 s, scale 6 %, 0.05°/s
(3°/min), RW 0.02. No Scout flight log exists; these are guesses chosen to be noticeable, not data. The drift test
checks the GM variance against the analytic value. Ground-truth `T_map_odom` is 4-DoF (roll/pitch are exact).

## DN-79 — Range height reference

`z_odom' = vz + (h − z_odom) / hgt_step_tau_s`, with h = body height above the TRUE surface directly below (floor or
an active box top, point query at the body origin). Steady state z_odom = h; a terrain step is followed with the lag
(0.3 s default, 1.5 s in `drift`; PX4's real behaviour is unknown, §3). `dist_bottom_m` = h + noise (σ 3 mm),
immediate, None above 6 m. Ignored: tilt (range along the body axis), the 2° beam footprint, the 1.27 cm module
offset, the 2 cm dead zone.

## DN-80 — Local-frame reset with a published delta

A reset moves the odom origin by `reset_delta` (dx, dy, dz, dyaw; default 0.3, −0.2, 0, 5°), increments
`reset_counter` and publishes the delta in `status().info["last_reset"]` (PX4 publishes reset deltas). The FC's own
hold target moves with it; an unchanged offboard setpoint makes the true vehicle move by −delta (tested). The
`drift` scenario triggers it at 2 m of progress in NAV.

## DN-81 — Camera: true pose, the synthetic mount everywhere

Images are rendered at the TRUE camera pose = true body pose (with tilt) @ `synthetic_room.T_BODY_CAM_SYNTH` (6 cm
ahead, 5° down). The same mount goes to the Localizer, the LoomingDetector and the Mission (NOT scout_loc.yaml's
zero-offset placeholder: the sim's camera is that mount). Gain 0.85 and noise σ 2 like `synthetic_eval`. Pinhole,
no blur, no rolling shutter, no exposure change in flight.

## DN-82 — Scenario geometry

On the A→goal line between the pillar and crate_c: `low_box` 1.4 × 1.4 × 0.7 m centred (5.1, 3.74); `tall_box`
0.5 × 0.5 × 1.9 m centred (5.1, 3.735). `late_obstacle`: 0.6 × 0.6 × 2.0 m on the floor, its near face 1.0 m ahead
of the body origin along the current velocity, placed when NAV progress ≥ 3.2 m and speed ≥ 0.2 m/s, only if it
clears every solid box and wall by 0.25 m (else retried next step). `drift` includes the low box: the slow
height-step lag needs a terrain step and the clear path has none (so `drift` also depends on looming seeing the
box). `wrong_takeoff`: +0.5 m in y and +20° (inside the room, outside the 0.30 m / 15° preflight tolerance).

## DN-83 — Pass bars, plus a geometric check

Bars from the brief: no collision; arrival ≤ 0.15 m and ≤ 10°; clear: zero obstacles (memory at the end AND
per-frame detections) and zero emergencies; low_box: a plan shape containing "over"; tall_box: "around";
late_obstacle: ≥ 1 emergency and arrived; drift: arrived; wrong_takeoff: REFUSED and never armed. Added, gating: the
TRUE trajectory entered the low box's footprint ("over") / passed the tall box without entering its footprint
("around") — the direct measure, independent of the planner's own label. Arrival error = the worst sample of the
last 1 s of a 3 s ARRIVED dwell, against the goal from the TRUE wall geometry (B + standoff · inward normal,
standoff from the nav config; yaw facing B); also reported against the mission's own goal. Emergencies = looming
emergency episodes (frames > 0.5 s apart), the mission's counter alongside. Detected boxes are matched to true
unmapped obstacles within 0.3 m (box gap), else counted false.

## DN-84 — Observe the components through proxies, not edits

`loop.Rec` forwards everything to a component and times / records the results of `localizer.process`,
`aligner.add_fix`, `looming.process` and `planner.plan`. Each tick the loop compares `aligner.T_map_odom()` with the
true transform (x, y, yaw: its z part is terrain-dependent under the range reference) and `aligner.pose_map(now)`
with the true body pose (xy, z). PnP fixes are scored against the true camera pose at the image stamp.

## DN-85 — Exit codes

`tools/run_scout_sim.py`: 0 every run passed its bars; 1 some run failed its bars; 2 could not run — a piece is
not importable (`ComponentsMissing`) or the software under test raised (an exception is not a verdict; the run is
listed with its traceback).

## DN-86 — The Mission factory

`loop.build_mission(world, vehicle, scenario, sim_cfg, log, recorder) -> (mission, components)`: Localizer (ORB
PnP on the cached sim DB, scout_loc.yaml `orb_pnp` / `localizer`), MapOdomAligner (`scout_loc.yaml map_align`),
LoomingDetector (`avoidance.config.load_config()`, map cloud), StaticMap / Planner / Follower / SafetyGate /
ObstacleMemory (`navigation.nav_config.load_nav_config()`), Mission (`scout_mission.yaml` + `takeoff` = A,
`goal.point_b` = B, + `scout_sim.yaml mission:` + the scenario's overrides). All imports are lazy; a test or an
experiment passes its own factory with the same signature.

## DN-87 — Seeds and determinism

The room is fixed (`room_seed` 0, so the DB cache is shared). A scenario seed drives three independent streams
(SeedSequence): estimator drift, sensor noise (rangefinder) and image noise. Same seed → same run on this machine
(the loop never reads wall time); not checked bit-exact across machines or library versions.

## DN-88 — The takeoff pad is a raised, MAPPED box at A (Ron 2026-10-17)

Ron: "just assume the takeoff pad is like an elevated box". `world.pad_box` = [0.5, 0.5, 0.4] m (size x, y, height;
PROVISIONAL, no real pad measured), centred on A. It is part of the prior map (map cloud + landmark DB), inserted after
the default boxes and before any unmapped obstacle so every shared surface keeps its texture; the takeoff pose rests on
its top; the mapping walk's takeoff column starts on it. New scenario `round_trip` (A → B → back → land) is scored by
the TRUE touchdown: on the pad top, ≤ 0.15 m from the pad centre. `pad_box: null` restores the floor pad.

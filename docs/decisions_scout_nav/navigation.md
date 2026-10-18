# Scout navigation decisions (piece C, `navigation/`; ids DN-40..59)

Brief: `docs/SCOUT_NAV.md` §6. Every value is PROVISIONAL (synthetic room only, no flight data). Values live in
`config/scout_nav.yaml`; shared numbers are read from `flight_demo.yaml` / `scout_hw.yaml` by
`navigation/nav_config.py load_nav_config()`.

**DN-40 Static map = voxel occupancy with a point-count floor; the edge of the map is a surface.**
0.1 m voxels, occupied at >= 3 points (rejects stray lidar points). Floor / ceiling = lowest / highest voxel layer
holding >= 30 % of the fullest layer's occupied voxels, height = median point z there. Bounds = 0.1 / 99.9
percentiles of the points in occupied voxels. `occupied()` outside the grid is False (unknown), but
`distance_along()` treats leaving the grid as hitting a surface (conservative for the speed law). Departs from
ARCHITECTURE §8.5's "2D grid sliced at flight altitude": the slice is per planner layer (DN-44).

**DN-41 `terrain_z` uses the mean z of a voxel's points, not the voxel top.** A surface straddles two voxel layers;
the top of the upper voxel would overstate a crate's height by up to 0.1 m (and the range-height conversion of
§3 with it). Known obstacles with `top_seen` raise the terrain to their top; unseen-top obstacles do not (the
planner never flies over them).

**DN-42 Wall goal.** PCA over map points within 0.25 m of B; refused (ValueError + reason) when B is off the map
(< 20 points or nearest > 0.10 m), not planar (lambda_min/lambda_mid > 0.2), or not a wall (normal elevation > 30 deg).
The standoff (0.8 m) is applied along the HORIZONTAL normal; z is clamped into the band. Orientation = the sign
whose hover point is inside the geofence, whose sight line hover -> B is clear of mapped voxels, and whose hover
point has the planner's static clearance (cylinder: hull + 0.3 m horizontal, hull half-height + 0.2 m vertical);
both valid -> the one facing the map centre. Yaw faces B horizontally; the camera's 5 deg pitch is not compensated.
A B within one voxel of the ceiling / floor is refused (its sight-line start sits in that surface's voxels).

**DN-43 Planner search = Dijkstra on a scipy sparse graph, not a Python A*.** The (x, y, layer) graph of the
8 x 6 x 3 m room has ~59 k nodes / 138 k edges; the statically free edge list is built once, each replan masks
it with the obstacle cells and runs `scipy.sparse.csgraph.dijkstra` (C). Measured on this laptop (Windows,
Python 3.9): init ~10 ms, replan 18-27 ms with two obstacles (bar: < 0.3 s). A heuristic would not pay for
itself at this size.

**DN-44 Flight band and layers.** Band = [floor + `min_alt_m` 0.4, ceiling - `ceiling_margin_m` 0.5]; layers every
0.2 m from the band bottom. Vertical clearance: mapped voxels hull_half_height + 0.2 m (voxel spans, i.e. up to
one voxel conservative), obstacle boxes hull_half_height + 0.3 m (the extra 0.1 m covers the rangefinder
height step over a box, SCOUT_NAV §3). Horizontal inflation hull + 0.3 m (static) / + 0.5 m (dynamic) as in
flight_demo, plus one cell (static, voxel discretisation) / half a cell diagonal (dynamic). The geofence is part
of the planner's free space so the safety gate never has to clamp a plan.

**DN-45 Over vs around from cost.** Vertical edges cost `climb_cost_factor` (3.0) x dz; horizontal edges their
length. A 0.4 m up-and-down detour (0.2 m layer, both ways) costs 1.2 m of horizontal travel, so a low wide box is
flown over and a person-sized box around. Provisional: tune on battery / time once flight data exists.

**DN-46 `top_seen=False` boxes reach +inf** in the planner (never over) and the ceiling in the TTC ray, per the
looming spec.

**DN-47 Start inside inflation.** The vehicle is already there, so the start may hop straight to the nearest free
node (same or adjacent layer, <= `max_start_escape_m` 0.8 m) whose segment stays out of hull-only contact.
A goal inside inflation, outside the band or outside the grid is refused with a reason.

**DN-48 Shape is geometric.** `direct` = one straight segment; `over` = the plan climbs > dz/2 above both endpoints;
`around` = lateral deviation from the straight xy line > 0.25 m, or a dip under something; both = `over+around`.
`info.over_ids` / `info.around_ids` name the obstacles whose inflated footprint the straight line crosses, split by
whether the plan passes above them. Mapped structure (the pillar) counts too: going round it is `around`.

**DN-49 Shortcutting = greedy farthest-visible in 3D**, every sample between two layers must be free on both
bracketing layers (conservative). Each greedy step checks all later waypoints in one vectorized batch.

**DN-50 Obstacle memory lasts the sortie (`memory_s: null` = inf). DEPARTS from ARCHITECTURE §8.5's 1.5 s decay.**
Looming only sees an obstacle while closing on it; hovering in front of it (the emergency hover itself) it can
never be re-observed, so a decay would forget it exactly when the Scout replans around it. The decay was designed
for continuous mono depth. Cost: a false obstacle also persists for the sortie (it can only make the plan longer
or impossible, never unsafe). A finite `memory_s` is supported.

**DN-51 Merge = confidence-weighted mean of the x/y bounds; z bottom min, top max; `top_seen` once seen.**
A detection merges into every stored box within `merge_gap_m` 0.3 m (a detection bridging two boxes merges them;
the lowest id survives). Weighted averaging converges on the true footprint instead of growing with each noisy
detection (a union would grow by the detector's ~0.3 m error each time and close corridors). `looming_raw`
detections weigh 0.5x. `top_seen` turns True once any detection saw background above the box (allows "over"):
a mistaken top-seen could let the planner fly over a tall object; the dynamic vertical margin and the
independent looming emergency are the backstops. `version` bumps on add / merge / expiry / top_seen change / a
bound moving > 0.1 m; the mission replans when it differs.

**DN-52 Follower reference moves ON the polyline (a carrot with zero lateral lookahead), not toward a lookahead
point.** No corner cutting, so the planner's clearances hold for the setpoint. Vertices turning more than 10 deg
in heading (or switching between horizontal and vertical motion) are stop points: brake to 0 there, yaw, go.
Gentle vertices are crossed at the slower neighbouring segment cap. Speed along the path is acceleration-limited
(0.5 m/s^2) with a discrete-time braking curve, so the only acceleration above the limit is the one tick that lands
on a stop (<= 2 a). Segment caps: horizontal <= 0.5 m/s (flight_demo), vertical <= 0.3 m/s. The reference waits
when it leads the vehicle by > 0.4 m. A new path is anchored at the current reference (no setpoint jump) when its
first waypoint is within 0.5 m of it; the mission should plan from `follower.reference` or the body pose.

**DN-53 Look where you go, with a final-approach waiver.** Horizontal motion only while the BODY heading (measured,
not the yaw reference) is within 20 deg of the motion heading; otherwise the reference stops and yaws first at
<= 45 deg/s. Pure vertical segments keep the yaw. In the last `final_yaw_dist_m` (1.0 m) of path the yaw goes to
the goal yaw and the heading gate is waived, so the final approach may be partly sideways in that last metre
(the camera saw it during the approach; the TTC law and the safety gate still apply). Open risk: a long sideways
final segment would be flown blind for up to 1 m.

**DN-54 TTC speed law from the body along the horizontal motion direction:** v_h <= (distance to the first mapped
voxel, known obstacle box or map edge - `stop_margin_m` 0.3) / `ttc_floor_s` 2.0. Approaching a mapped wall (the
goal wall included) the apparent TTC therefore stays >= 2 s > the 1.5 s looming emergency (load_nav_config refuses a
config where ttc_floor_s <= looming_ttc_s, or stop_margin >= standoff - arrive tolerance, which would stall the
arrival). The ray is the body's centre line only; side clearance is the planner's job.

**DN-55 Safety gate: hold latch, band opt-out, provisional geofence margin.** HOLD = the body position + yaw at the
moment the hold began, latched until the condition clears (no creep with a drifting vehicle); conditions: no plan,
alignment invalid, an emergency within `emergency_hover_s` (1.0 s, flight_demo) of the last one. Otherwise: band
clamp (skippable with `band_check=False` for takeoff / landing, which leave the band by design), geofence clamp
(map bounds - `geofence_margin_m`; flight_demo has null, so scout_nav's 0.4 m default is used and
`cfg["geofence_margin_source"]` says so), max 0.6 m from the current pose, and a setpoint rate limit of 0.75 m/s
(1.5 x the speed cap) between ticks.

**DN-56 One flat navigation config.** `load_nav_config()` flattens `scout_nav.yaml` (a key may live in one section
only) and overwrites the shared keys from `flight_demo.yaml` (speed cap, inflation extras, emergency hover, geofence
margin, looming TTC) and `scout_hw.yaml` (hull radius / half-height placeholders). Modules keep `DEFAULTS` equal
to those files today so they run with cfg=None in tests; the test checks the loaded values against the files.


**Amendments by the main session (2026-10-17, closed-loop findings):**
- DN-47 amended: a start inside even the HARD keep-out may escape if the hop only leaves it (blocked samples form a
  prefix, never re-entered), and `max_start_escape_m` 0.8 → 1.2. Sim `late_obstacle`: the emergency stop left the
  Scout 0.15 m from the detected face and every replan failed ("start inside inflation"); test
  `test_plan_escape_after_emergency_stop`.
- DN-54 amended: the TTC speed law measures from the hull's leading edge (distance − `hull_radius_m`), as the looming
  builder found: from the body origin a 2.0 s floor meant a 1.5 s hull-contact time at 0.8 m.

**Closed-loop false alarms at the mapped north wall (2026-10-17, SCOUT_NAV.md §10):**

**DN-57 The TTC speed law also measures over the looming emergency's corridor (amends DN-54).** The emergency
(DN-25) counts anything within hull radius + `corridor_margin_m` (0.35 m) of the motion line at (along − hull
radius) / speed. One centre ray misses an oblique wall that the corridor's edge reaches first: on the route around
the low box the Scout approaches the north wall at 30–45° to a vertex 0.65 m from it, the centre ray allowed
~0.5 m/s, and the TRUE corridor contact time fell to 1.65–1.85 s (replay: estimated 1.26 s → emergency, a
`looming_raw` false obstacle, a needless hover). Now 9 parallel rays at body height across ±0.35 m
(`ttc_rays_lateral`, `corridor_margin_m` read from scout_avoid.yaml) give v_h ≤ min((centre − hull − stop_margin),
(corridor − hull)) / `ttc_floor_s`: the centre ray keeps the stop margin (the goal wall), the corridor is the pure
TTC. Putting the 0.3 m stop margin on the corridor edge too DEADLOCKED at the planner's vertex 0.65 m from the wall
(test caught it). No vertical rays: the planner keeps mapped surfaces and obstacle tops > hull half-height + 0.2 /
0.3 m above and below, beyond the corridor ellipse, and vertical rays only add voxel-edge stalls. A ray starting
inside an occupied voxel or box ignores that run (it is beside the hull, not ahead). Test
`test_follower_ttc_corridor_oblique_wall`: worst true corridor contact 1.65 s (centre ray) → 2.16 s, arrival 0.35 s
later.

**DN-58 A short hop is flown facing the next segment (amends DN-53).** Escape hops into the channel between the box
and the wall are 0.1–0.2 m long and point at the wall; look-where-you-go made the Scout yaw ~110° onto the hop, creep
toward the wall at 0.12–0.18 m/s (looming's worst regime: tiny baseline, absolute depth error divided by a low
speed; true contact 4–5 s, estimated 0.53–0.67 s → `looming_raw` false obstacles in low_box seeds 0 and 2), stop, and
yaw back. A horizontal segment shorter than `short_hop_m` (0.3 m) followed by a longer horizontal segment now takes
that segment's heading as its look heading (yaw target and heading gate), so it is flown sideways while the camera
already looks where the Scout goes next. Open risk, like DN-53's: ≤ 0.3 m flown blind sideways, inside the planner's
free space, at braking speed. Test `test_follower_short_hop_faces_next_segment` (a 0.5 m hop still yaws onto itself).

**DN-59 No plan → the mission passes NO setpoint, and the gate latches its hold (fixes the DN-55 interface).** With
no valid plan `Mission._fly_to` sent the CURRENT pose as the setpoint, so the gate's `no_plan` hold never fired and
`_send_map` added a velocity feed-forward equal to the setpoint's own rate, i.e. the vehicle's own motion: a plan
failure kept whatever velocity it had. Closed loop `low_box` seed 0 (after a false obstacle covered the goal): 12 s at
0.24 m/s east, over crate_c (the rangefinder height step then fed the same loop vertically), into the ceiling. Now
`p_sp = None` → HOLD at the latched pose (`_send_hold`, no feed-forward) until a plan exists; `no_path_land_s` still
lands after 20 s. The same hold also puts the follower's reference back ON the body (`Follower.hold_at`) and drops its
plan; the next valid plan is installed even when it equals the old one (`has_path`). Without that, the replan after an
emergency started from the follower's stale reference, which had led the vehicle by 0.45 m toward the obstacle that
stopped it (`_replan` plans from the reference when it is within 0.5 m of the body): closed-loop `late_obstacle` seed 1
flew its first leg straight into the late box's face.

**DN-53 amended (2026-10-17): the final-approach yaw stays inside the heading gate while the Scout moves.** DN-53's
open risk happened. On the return to the pad (round_trip), the goal yaw (the pad's, +x) is 180 deg from the last
segment (-x). In the last metre the yaw turned at 45 deg/s while the Scout was still closing at ~0.45 m/s on wall_x0,
1.1 m ahead of the hull. The camera ended 50 deg off the motion, so the wall it was closing on sat at the image's left
border (u 40-70 px). There the looming depth read ~2x short (0.46 m vs 0.97 m true), and the map-free emergency fired
(round_trip seed 2, est. TTC 1.33 s, true ~2.3 s; the main session saw it on seeds 0 and 1 too). Now, in the final
zone, the yaw may lead toward the goal yaw only within `max_heading_err_deg` of the motion heading. That holds while
the Scout moves (reference or body speed > `arrive_max_speed_mps`). The rest of the turn happens in place at the goal,
where no looming applies (closing < `min_closing_speed_mps`). Cost: up to ~4 s more at the goal for a 180 deg turn.
Looming is unchanged. Test `test_follower_final_yaw_stays_in_the_heading_gate_while_moving` (fails on the old
follower: 157 deg off the motion while moving).

**DN-51 amended (2026-10-18): an unseen top is kept per footprint, and a later seen top over it retires it.** DN-51
kept the box top as the max of every detection's top. One `top_seen=False` detection (the top out of view, so the box
reaches the ceiling, 3.02 m) merged into the 0.6 m barrier then outlived every later seen-top detection of it. The box
was top_seen True with its top at the ceiling, so it could not be flown over. Sim low_barrier seed 0 on 705e687 + the
DN-63 feed-forward: both barrier pieces got one such detection (t 6.75 s and 10.25 s), the planner was boxed in from
10.3 s, and the Scout landed after `no_path_land_s`. It does not show in the suite at 24e6ab2. Now the box top is the
max of (a) the SEEN tops, which are never lowered, and (b) the live UNSEEN tops. Each `top_seen=False` detection is
kept with its own footprint. A LATER `top_seen` detection retires it when its footprint contains the unseen
footprint's centre and covers at least `unseen_top_cover_frac` (0.5, PROVISIONAL) of its area. On the reproduction
each of the two later seen detections contains the unseen footprints' centres and covers 58–59 % of them (the unseen
boxes are wider mainly in depth, which looming assumes rather than measures), and the top drops 3.02 → 1.10 m.
Merges bridge distinct objects within `merge_gap_m`, so a seen top of a LOW box must not lower a merged TALL
neighbour. The neighbour's own unseen footprint lies mostly outside the low box's, so it stays. A wide unseen
detection that bridges both stays too when it is centred over the tall one. Order matters: an unseen top that comes
after the seen ones raises the box again, because it is new evidence. Accepted limit: one wide unseen detection that is
the ONLY evidence of a tall neighbour, and lies mostly over the low box, is retired by the low box's seen top. The
backstops are DN-51's: re-detection while closing raises the top again, the dynamic vertical margin, and the
independent looming emergency. Tests `test_obstacle_memory_unseen_top_retired_by_later_seen_top` (the event sequence;
fails on the old merge: still 3.02 m, boxed in) and `test_obstacle_memory_seen_low_top_keeps_tall_neighbour`.

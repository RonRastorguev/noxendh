# Decisions — Scout looming obstacle detection (piece B, `avoidance/`)

Ids DN-20..DN-39 (docs/SCOUT_NAV.md §9). What, why, what it departs from. Every number is PROVISIONAL: it was
chosen on the synthetic room (`localization/synthetic_room.py`), no Scout footage exists.

## DN-20 — Generic looming on any textured surface, not the spec's person-only YOLO path

ARCHITECTURE §8.4 B detects PEOPLE with YOLO and gets TTC from the bbox height (assumed 1.7 m person). We track
generic corners and measure local scale change, so any textured object looms (boxes, racking, pallets, people).
Why: the brief's obstacles are boxes; no GPU model on the critical path; no class prior to be wrong. What it gives
up: an untextured object (plain wall-coloured board, glass) produces no features and is invisible (see limits).
The §8.4 A mono-depth path is not built; the map-anchored looming path (DN-26) takes its role.

## DN-21 — Ego-rotation removed with the FC's attitude; odometry must be sampled at the image stamp

Old feature positions are rotated into the current camera orientation with the relative rotation from the FC
odometry (`T_odom_body @ T_body_cam`), never from PnP. The FC's short-term attitude is gyro-accurate; a wrong PnP
pose cannot touch this. The CALLER must pass odometry interpolated at the image stamp (the mission does,
`_odom_at(stamp)`); the latest sample would be 50–200 ms off and a yaw would loom.

## DN-22 — Scale change from neighbour pairs, projected on the tangent to the ray from the FOE

After de-rotation, a static point contracts radially about the FOE e: x_old − e = (x − e)·Z/(Z+τ). For a pair
(i, j), projecting the pair vector on the direction t ⊥ (x_i − e) gives exactly t·Δ_now / t·Δ_old − 1 = τ/Z_j for
ANY surface orientation (the slant-dependent radial term is projected out; the isotropic divergence would read the
floor and a side wall 1.5× too near) and any residual common-mode rotation cancels in the difference. Near e the
pair's own direction is used (the error term vanishes there). Per feature: weighted median over its neighbours
(weight = projected length). Depth Z = τ/w with τ from the odom displacement over the window, so a velocity scale
error of s % is a depth error of s %. Departs from the brief's "TTC = dt/(s−1) from neighbour distances" only in
the projection; it is the same looming quantity.

## DN-23 — Two measurements of the old position: direct (map path) and chained (emergency)

KLT chained frame-to-frame accumulates drift, and under yaw a 102° lens stretches the periphery by ~2·x·θ, which
a translation-only KLT tracks badly (synthetic: 3-frame error median 0.31 px, p95 1.5 px during a 30°/s yaw).
The map path (far range, small signal) therefore re-measures each old position by ONE KLT step from the current
image into the old image WARPED into the current orientation (warpPerspective K R K⁻¹; undistort+rotate remap for
a lens with distortion) and drops tracks where direct and chained disagree by > 1 px. That cut the
background false-excess p90 from ~0.35 to ~0.1 of the predicted expansion. At short range (expansion ≥ 0.06 per
window) the k-frame patch scale change (10–15 %) makes the direct step worse, so there, and in the emergency
path, the chained track is used.

## DN-24 — Static-point (epipolar) check and aperture check on every de-rotated track

A de-rotated static point moves exactly along the line through e. A track whose old position is off that line by
more than `epi_tol_px` (1 px) + what an FOE error of `foe_err_px` (25 px ≈ 4° of velocity direction) explains is
dropped from everything (as a feature and as a neighbour). It removes aperture slides along curved strokes and
T-junctions. Found on the tall box: a cluster of 10 features on one curved stroke read 0.4 m instead of 1.0 m and
raised an emergency at a true contact time of 2.2 s; with the check the TTC error over that approach is 0.88–1.1×.
A slide PARALLEL to the epipolar line is invisible to that check (it is motion along the unknown-depth direction)
and reads as a wrong depth: in the closed-loop `drift` seed 2, big letters on a grazing wall did exactly that and
produced a false obstacle that boxed the Scout in. So each feature must also have gradient energy (structure
tensor over a 9×9 patch) BOTH along and across its epipolar direction, each ≥ 20 % of the total
(`aperture_min_frac`). Cost: fewer features on line-art texture; the 0.9 m late-obstacle latency went 0.2 → 0.3 s.

## DN-25 — Emergency: corridor + hull-contact time from a cluster, image and odometry only

A feature is a corridor candidate when its 3D point (ray × Z, body frame) is ahead and within hull radius + 0.15 m
of the line along the odom velocity; a cluster counts only if ≥ 3 of its members are also inside the hull's real
swept cross-section, an ellipse hull radius + 0.15 m across and hull half-height + 0.15 m vertically (gravity from
the FC attitude; both from scout_hw.yaml, PROVISIONAL). Requiring ALL members inside the ellipse delayed the late
obstacle's emergency from 0.3 s to 0.4–1.0 s. The Scout is a flat disc: with a
0.35 m sphere, flying 0.5 m over the mapped crate_c (the planner's static inflation is hull + 0.3 m) raised an
emergency from a 45 % depth underestimate on grazing crate-top features (closed-loop `low_box`). The reported TTC is the HULL-CONTACT time ((along − hull radius)/speed), not the
camera's Z/v_z: "something will be hit within 1.5 s" is about the hull. Emergency needs a cluster of ≥ 5 candidate
features linked in the image (≤ 60 px) with consistent depth (±35 %), median contact time < 1.5 s. It runs with
`T_map_body=None` and never reads the map or PnP. With a map pose a `looming_raw` box is also emitted — unless the
map comparison ran and ≥ 50 % of the cluster is map-consistent (a mapped surface): navigation memory keeps boxes for
the sortie, and a raw box inside a mapped object only blocks the planner. The emergency FLAG is never suppressed.
Interface note for piece C: the follower's TTC speed law must measure the HULL's distance to the surface (or use a
stop margin ≥ hull radius); measured from the body origin, `ttc_floor_s` = 2.0 s gives a hull-contact time of
2 − 2r/d, i.e. exactly 1.5 s at d = 0.8 m. Synthetic: approaching the goal wall at a hull-contact TTC of 1.95 s
gave estimated min 1.67 s (estimate/true 0.84–1.1), 0.17 s above the emergency line — thin; real noise will be
worse. Limit: ≥ 3 features must lie within ±0.21 m of the hull's height on the object; an object untextured in that
band is flagged late (a mapped crate whose top is 0.2 m above the hull reported min TTC 4.8 s at a true ~2 s).

## DN-26 — Map-anchored: per-pair measured − predicted expansion, filtered, persistent, placed by own depth

The map depth predicts each feature's old position under the known odom ego-motion; the SAME pair estimator runs
on measured and predicted old positions, and the per-pair difference is aggregated (weighted median), so mapped
depth edges and slanted surfaces are predicted, not flagged. Per-track low-pass (α 0.5) of excess and prediction.
Flag: excess > max(0.012, 0.35 × predicted), TTC < 10 s, not on a mapped edge (DN-27); emit a cluster
(≥ 6 features, ≤ 50 px links) only when its tracks' median flag streak ≥ 3 frames AND its front face was placed
within max(0.3 m, 15 % of range) of the previous frame's placement (3D persistence: a cluster whose depth jumps
around, e.g. on the edge of the view, is held back). Features within 24 px of the image border are never flagged
(one-sided neighbourhoods). The map path needs ≥ 7 cm of forward motion per window (`map_min_tau_m`, ≈ 0.25 m/s):
at 0.15–0.2 m/s while turning near the goal the expansion of a 2.4 m wall is ~0.02, two noise sigmas, and the
closed-loop `drift` run produced a false obstacle there; the emergency path still runs at any closing speed
≥ 0.1 m/s. Clusters whose front faces come within the hull diameter of each other (3D) are merged: the Scout
cannot pass between them anyway, and close up one box otherwise splits into two (the second, placed beside the
true face, was a 0.9 m "position error" in the low-box test). Placement: the members' own
small-neighbourhood depths (emergency estimator) set the cluster depth at their 30th percentile — a cluster on an
UNMAPPED object also holds T-junctions on its silhouette whose depth mixes in the far background, and the object
is the near part (closed-loop `tall_box`: the median placed the 0.5 m box 1.3 m too deep; the 30th percentile put
the face within 0.07 m median). That depth must be consistent with the map path's own (wide, filtered) depth, or
nothing is emitted that frame; members far from it are dropped. The horizon was raised
from the brief's 5 s to 10 s: at 0.4 m/s, 5 s is only 2 m.

## DN-27 — Robust to map-pose error: nearest mapped surface nearby, and no flags on mapped depth edges

The rendered expected depth is min-filtered over `map_pose_tol_px` (2 render px = 8 image px ≈ 1.4°), so a few
cm / ~0.5–1° of pose error at a mapped edge predicts MORE expansion, never less. A feature whose window spans a
mapped depth edge (max/min > 1.25) can be a T-junction (near edge over far texture), which moves like no point; it
serves as a neighbour but is never flagged. Map-path pairs are only formed between features whose map depths are
within 1.5× (one mapped depth layer): passing 0.7 m beside the mapped pillar, far-wall features next to its
silhouette inherited excess through pairs with mis-tracked grazing pillar features (closed-loop `drift`: a stable
false obstacle for 1 s). Cost: an unmapped object right in front of a mapped edge loses its
edge features (its interior still flags). This removed the false obstacle on the top edge of the mapped crate_c
(seen in the unit long-flight test AND in the closed-loop `clear` run).

## DN-28 — No closing motion, no looming

Optical-axis speed < 0.10 m/s (hover, yaw in place, vertical climb): output nothing new and reset the per-track
filters. Looming needs translation along the view; a static obstacle seen while hovering is navigation memory's job.

## DN-29 — `top_seen` means "seen free above it, with room for the Scout"; the top is bounded by rays, not guessed

`top_seen=True` only when ≥ 3 background rays were seen above the cluster AND they show free space up to
top + `top_clear_m` (hull radius + nav.dynamic_inflate_extra_m = 0.7 m, the planner's own obstacle inflation) at
the obstacle's depth. The top bound is the 3rd-lowest such ray over the obstacle's depth span (one stray ray
cannot lower it). A background ray must (a) measure, with its own small-neighbourhood depth, beyond the obstacle
(> 1.5× the cluster depth and > cluster depth + assumed depth: it passed over it), (b) be ON THE MAP (map-consistent
excess, or its own depth within 15 % of its map depth — the obstacle's own top surface, seen from above, is far
but not mapped, and counting it lowered the low box's top to 0.61 m), and (c) lie above the 2nd-topmost feature in
that column that measures NEAR (≤ 1.2× the cluster depth: more obstacle). Cases:
- `seen` → top = bound; `seen_no_clearance` → top = bound, `top_seen=False`;
- `out_of_view` → ceiling (max z of the map): the cluster top is within 100 px of the image top, or ≥ 2 near
  features above it say the obstacle continues;
- `nothing_near_above` → highest member + 0.5 m, `top_seen=False`: the top region is in view, nothing near is
  measured there, but too few clear rays to bound it.
Closed loop (`low_barrier`, 0.6 m wall-to-wall): single noisy features had flipped a few frames to "ceiling" or a
1.5–1.8 m top, navigation memory keeps the worst, and the Scout was boxed in; with (a)–(c) as above every frame
bounded it at 0.77–1.0 m and the Scout flew over. The tall box (1.9 m) reads `top_seen=False` from honest geometry
at the ranges it is detected (camera 1.1 m, FOV top +31.5°: 0.7 m of free space above it is visible only beyond
~2.4 m); synthetic: detected from 2.0 m, `top_seen=False` on every detection, top never below 2.03 m. The low box:
top seen on 100 % of detections, top 0.65–1.09 m (true 0.7; at most 5 cm under, in 1 of 37 frames). Departs from the
brief's plain "background above the cluster" rule, which (a) would call the tall box's top seen from 3 m and (b) let
a single wrong ray lower it.

## DN-30 — Box geometry

Front face at the cluster depth (lateral extent = the members' bbox + 5 cm), extruded `assumed_depth_m` = 0.5 m
along the horizontal view direction (the back is never seen), floor (0.5 % z-percentile of the map) to top
(DN-29). Position error is therefore reported on the FRONT FACE (what the camera measures): a 1.4 m deep box's
full-centre error is dominated by the assumed depth (0.49 m median for the low box, 0.55 m for the tall one).

## DN-31 — Config: `config/scout_avoid.yaml`; shared values read from their owners

`ttc_emergency_s` from flight_demo.yaml `avoid.looming_ttc_s`, `hull_radius_m` from scout_hw.yaml, `top_clear_m`
derived from both (`avoidance.config.load_config`). `LoomingDetector(cfg=None)` uses `DEFAULTS`, which mirror them,
so tests run without YAML.

## DN-32 — Synthetic test model

Renderer at 2× supersampling (aliasing of a minified texture otherwise doubles KLT error; a real lens low-passes),
image noise σ 2. The detector gets a drifting odom frame (offset + 0.5°/min yaw drift), +5 % scale on odom
translation and velocity, a (0.02, −0.02) m/s velocity bias, σ 0.03° attitude noise, and a PnP pose with σ 3 cm /
0.5°. Not modelled: lens distortion (code path exists, untested), rolling shutter, motion blur, exposure changes,
moving objects, real latency jitter.

## DN-33 — Timing

640×480, Windows Python 3.9 on this laptop (numbers vary ±20 % with machine load): map-anchored frames 36–55 ms
mean (p95 45–74 ms), emergency-only (no map pose) 18 ms, no closing motion 13 ms. The 40 ms target is NOT met on
map-anchored frames; it is on emergency-only frames. Already done: the map is culled to the view cone before the
160×120 z-buffer render (~95 k → ~30 k points), corner detection runs only when ≥ 8 % of the track budget is free
or every 2nd frame. Per frame: KLT (chained + forward-backward + direct) ~8 ms, render ~7 ms, detection ~3 ms,
pairs / medians / clustering the rest. Next options: run the map path every 2nd frame (the emergency path every
frame), render the depth every 2nd frame and reuse it shifted by odometry, or fewer features (300).

## DN-34 — PX4 local-frame resets: drop the motion history

A reset (xy, z or yaw jump of the FC's odom) makes every odometry delta that spans it garbage: in the closed-loop
`drift` run a 0.36 m / 5° reset produced a false full-height obstacle. `process(..., reset_counter=None)` takes the
FC's counter (OdomSample.reset_counter); a change, an odom displacement that the velocity cannot explain
(> 8 cm per frame) or a frame gap > 0.5 s drops all old track positions, then 2 more frames are dropped as window
starts (odometry sampled at image stamps right around a reset can still be from the wrong side). A yaw-only reset
is caught only through `reset_counter`. Interface note for piece D: pass `reset_counter=odom.reset_counter` from
`Mission.on_image`; and `_odom_at(stamp)` returns the first post-reset sample for an image stamped before it (the
buffer is cleared on reset), so that frame pairs a pre-reset image with post-reset odometry.

## DN-35 — A feature's depth from the pairs that measure IT (own depth); longer pairs; looser aperture check

Found on the closed-loop `late_obstacle` regression (seed 2 flew into a box that popped in 0.8 m ahead at 0.5 m/s,
no emergency in 1.6 s) and on the north-wall false alarms (SCOUT_NAV.md §10). DN-22's pair, projected on the tangent
at its ANCHOR i, measures τ/Z_j — the NEIGHBOUR's depth (the anchor's own term cancels). The per-feature value was the
median over the anchor's own pairs, i.e. the median of its NEIGHBOURS' depths: at a silhouette (a near box beside far
background) a box feature read the background (replay of seed 2: box features 1.3–5× too far, so their contact times
stayed long; exact-geometry test: +567 % at the edge). Now each feature takes the weighted median over the pairs in
which it is the NEIGHBOUR (`own_depth: true`, `_wmedian_own`): exact for any anchor depth and surface orientation.
Because the anchor's depth cancels, pairs no longer need to be local for depth consistency, so they may be longer and
more numerous: `neighbours_k` 10 → 16, `max_pair_px` 80 → 120 (the map path already queried 16 within 120 px, so the
KD query is unchanged). And `aperture_min_frac` 0.2 → 0.1: at close range a pop-in box's magnified texture is
edge-like and the 0.2 check left 5–7 of its 30 tracks (needs a cluster of 5 in the corridor).

Replays of four recorded closed-loop runs (inputs at the image stamps, same frames for every variant):

| Variant | late box (seed 2): emergency | wall runs: lowest min TTC (drift 2 / low_box 2 / low_box 0) |
|---|---|---|
| before | never | 1.26 s (EMERGENCY) / 1.69 / 3.72 |
| own depth only | never | 1.32 (EMERGENCY) / 1.65 / 3.73 |
| own depth + 16 within 120 px + aperture 0.1 (chosen) | every frame from 0.45 s after it appeared (1.2 s before contact) | 1.59 / 1.90 / 3.59 |
| aperture 0.1 alone | from 0.45 s | 1.43 (EMERGENCY) / 1.49 (EMERGENCY) / 4.13 |

(The wall recordings were flown by the OLD follower, whose diagonal approaches reach 1.85 s TRUE corridor contact
time; the new follower keeps it ≥ 2 s, DN-57.) The emergency stays image + odometry only (DN-25); nothing here reads
the map or the PnP pose. The DN-24 aperture case (big letters on a grazing wall, `drift` seed 2) is re-checked by the
closed-loop suite, not by a unit test. The MAP path's per-feature excess, measured and predicted expansion use
the same own-measurement median (a mapped background feature next to an unmapped box no longer inherits the box's
excess, the effect DN-26 worked around with depth pruning): closed-loop `tall_box` seed 0 had flagged goal-wall
features (true 6.1 m, own depth 6.0–6.3 m) beside the box (3.0 m) and placed a mixed cluster at 4.1 m, on the goal;
with own excess the box is placed at 3.0–3.3 m from its first detection. Unit `tall_box`: first boxed at 3.35 m
instead of 1.99 m (61 detections vs 27), all `top_seen = False` as required; the unseen-top placeholder (highest
feature + 0.5 m) of a few far detections sits up to 6.4 cm under the true 1.9 m, which the planner never uses (DN-46),
so the test's ≥ 1.85 m bound now applies to claimed tops and unseen ones keep ≥ 1.80 m (`map_own`: false restores the
old map median). Tests: `test_own_depth_at_a_silhouette` (exact geometry),
`test_oblique_wall_approach_by_the_follower_no_false_alarm` (the real follower flies the diagonal route: 0 obstacles,
0 emergencies, min TTC 1.66 s).

## DN-36 — Map path: the odometry's scale is measured on the mapped scene

The map path predicts each pair's expansion from the ODOMETRY's displacement over the window. A flow-scale or
velocity-bias error (sim `drift`: σ 6 % scale, σ 0.05 m/s bias, so 15–20 % of the forward displacement at low speed)
then reads as excess expansion on EVERY mapped surface, and with noise on top crosses the 35 % margin: `drift` seed 1
boxed the mapped north wall from 1.9 m (a `looming_map` false obstacle) after the other fixes. Most features in a
frame are on the map, so the median measured / predicted expansion over map-consistent features (≥ 40 with a
predicted expansion ≥ 0.01, not on a mapped depth edge) IS that scale. The prediction is multiplied by it (clamped to
1 ± 0.3, `ego_scale_max`; 0 = off), and the placement depths (map-path and small-neighbourhood, for the consistency
check and the top) use the corrected displacement. On the recorded `drift` seed 2 the estimate tracks the true
displacement ratio (correlation 0.81; 1.157 vs 1.153 at its worst moment). A large unmapped obstacle filling most of
the view biases it up, so its map-path excess shrinks; the clamp keeps a real obstacle's 2× expansion visible, and the
emergency, which keeps the RAW odometry and never reads this (DN-25), still covers anything near. Debug:
`out.debug["ego_scale"]`. Side effect in `test_low_box_detected_early_and_placed` (whose odometry has a +5 % scale
error; the estimate reads 0.959 ≈ 1/1.05): the box is now detected at 2.83 m instead of 2.55 m, and those first
detections bound the top from fewer, longer rays (1.16–1.36 m for a 0.7 m box: over, the conservative side). The
test's ≤ 1.2 m top bound now applies within 2.6 m (all ≤ 1.09 m there); earlier ones must stay ≤ 1.5 m; never under
0.65 m, unchanged.

## DN-37 — The motion DIRECTION comes from the image (focus of expansion); the odometry gives the magnitude

After DN-35/36 and the follower fixes the remaining false emergencies all came from slow motion (0.13–0.25 m/s, just
after a stop vertex, often still yawing) near the mapped north wall. Measured on the trajectory log of `drift` seed 2
at the false emergency: the odometry's displacement over the window pointed 17–22° too far toward the wall and was
25 % short. The emergency built its corridor along the odometry velocity, so the wall fell inside it, and the window
displacement set the FOE for every static-point check and every pair (estimated contact 1.22 s, true ≈ 3.6 s). At low
speed a flow/velocity bias is a large share of the velocity: on the recordings the odometry's direction error was
median 8° / max 22° (drift seed 2) and up to 47° in a slow creep on a NOMINAL-drift run (low_box seed 2).

The direction of translation is observable from the image alone: after de-rotation every static point's flow lies
on a line through the FOE. `_image_foe` fits that point (lines through every track along its de-rotated flow;
weighted least squares, Huber-reweighted at 1 px; ≥ 30 tracks with ≥ 0.5 px of flow; ≥ 80 % of inliers expanding,
else the odometry's FOE is kept). On the recordings its error against the true camera translation was median
0.12–0.38°, max 0.84°. Per window the camera displacement becomes |odometry displacement| × the image direction, so
the FOE, the static-point checks, the pair expansions and the map path's prediction all use it; the corridor
direction is the BODY's displacement from it (camera displacement minus the lever arm of the rotation, from the FC
attitude). The magnitude stays the odometry's, from its HORIZONTAL displacement matched to the image direction's
horizontal part (|Δxy_odom| / |t_xy|; the projection m·t when the motion is near-vertical), and the emergency's speed
likewise (|v_xy| / |u_xy|): the Scout's FC z is height above the surface under the rangefinder (DN-03), so it steps at
every terrain edge. Leaving the raised pad, the odometry showed 0.3 m of "climb" in 0.3 s (|odom| / |true| up to 2.1);
the first version used |odom displacement| and placed `tall_box` seed 0's box 1.2 m too far. Depth (τ/w) and speed
carry the same scale error, so the TTC (Z / speed) is nearly scale-free except for the hull-radius term. Still image + FC odometry only: no map, no PnP
(DN-25). Debug: `out.debug["foe_odom_deg"]` (angle between the image and odometry directions).

# Decisions — piece A: map↔odom alignment (`localization/map_align.py`)

Ids DN-10..DN-19 (docs/SCOUT_NAV.md §9). What, why, what it departs from. All numbers are PROVISIONAL, chosen on
synthetic odom + fixes (`tests/test_map_align.py`); config in `config/scout_loc.yaml` `map_align:`.

## DN-10 — 4-DoF alignment; z is `T_px4(t) + beta`, not one offset

`p_map_xy = Rz(psi) p_odom_xy + t_xy`, `yaw_map = yaw_odom + psi`, `z_map = z_odom + o(t)`. With a baro / EV height
reference `o = beta` (a slowly drifting offset). With the range reference (the real Scout) `o = T_px4(t) + beta`, where
`T_px4` is the map z of the surface PX4 is currently referencing its height to (terrain under the vehicle in steady
state, lagging it after an edge) and `beta` is the slow residual (PX4's own z offset, map / terrain-model errors)
estimated from fixes. `T_map_odom()` returns the 4x4 with z = the total offset NOW, so its z changes when the terrain
under the vehicle changes; callers that need a rigid frame must not cache it across a terrain edge.

## DN-11 — Gating, outlier rejection and the consistent-run re-anchor

A fix is gated on its horizontal, yaw and z residuals against the filter estimate at the fix's image stamp:
`gate = max(gate_min, 4 sqrt(P + R))` (0.15 m / 5° / 0.15 m floors). P grows with time and distance flown while no
fix is accepted, so the gate widens by itself after a gap. A rejected fix never moves anything. Rejected fixes are kept
as candidates (their implied alignment); `reanchor_fixes` (5) consecutive ones that are mutually consistent
(±8 cm / ±3° / ±8 cm z at the current body position) over ≥ 0.3 s re-anchor the filter to their mean (P reset to
the fix noise). An accepted fix clears the run, so alternating bad fixes can never re-anchor.
Why: one bad PnP must not move the frame, but a genuinely changed alignment (flow glitch, accumulated drift past the
gate) must be followed; the Localizer's own `reacquire_fixes` logic is the same idea one level down.
Risk (open): a PnP that converges to the SAME wrong solution for 5 frames in a row (hovering in front of a repetitive
pattern) would re-anchor wrongly. The output is still slew-limited (DN-12), and `n_reanchors` / `events()` record it.

## DN-12 — Filter estimate vs slew-limited output; corrections rotate about the body

Two copies of (psi, t_xy, beta): the Kalman estimate, and the OUTPUT that every odom sample moves towards it by at
most 0.3 m/s (body position), 6°/s, 0.2 m/s (z). `pose_map`, `setpoint_to_odom` and `T_map_odom` use the output, so
the setpoint for a fixed map point moves ≤ 1.5 cm + 0.3° per 50 ms tick from alignment updates (tested: 2.2 cm /
0.36° per 60 ms through outliers and a 0.4 m / 6° re-anchor). `pose_map` and `setpoint_to_odom` must share one
alignment, or the follower would see its pose jump while its commands do not. `body_pose_map_at` (measurement
geometry, e.g. the looming map comparison) uses the filter estimate. Every correction (Kalman update, slew, re-anchor)
is applied ABOUT THE BODY: a yaw correction changes psi and re-solves t so the body's own map position moves only by
the position correction; otherwise a 1° correction would swing a body 5 m from the odom origin by 9 cm.

## DN-13 — Latency: interpolate odom at the image stamp

Odom is buffered for 3 s; a fix is compared with the odom pose interpolated (linear position, slerp attitude) at its
IMAGE stamp, extrapolated with `v_odom` ≤ 0.1 s past the newest sample. Older than the buffer → rejected
(`stamp_outside_odom_buffer`), frame untouched. Tested at 0.5 m/s + 30°/s with 200 ms latency: residuals ≤ 4.5 cm;
the same fix applied at its arrival stamp misfits by 10 cm / 6° and is rejected.

## DN-14 — Range height reference: conversion with the terrain under the vehicle NOW; PX4's step lag measured, not assumed

`setpoint_to_odom` converts z with `o` at the newest odom sample, i.e. the surface under the VEHICLE now (PX4 tracks
height above what is under it), never under the setpoint. What matters is not the true terrain but the surface PX4 is
CURRENTLY referenced to (`T_px4`), which lags a step by an unknown amount. Two ways to know it:

1. **`dist_bottom_m` present (preferred).** ASSUMPTION: it is the RAW rangefinder distance of the body origin above
   the surface (tilt-corrected), not PX4's own fused `dist_bottom`. Then `z_odom - dist_bottom = (terrain - T_px4) -
   const`, so the lag residual `L` is measured each sample (`const` = its slow steady value, an EMA frozen while
   |deviation| > 5 cm; 3 cm deadband for range noise) and `T_px4 = terrain_now - L`. No assumption about PX4's lag:
   tested closed-loop over a 0.7 m box at 0.4 m/s, true height stays within −0.6..+1.3 cm (PX4 lag 0.05 s) and
   −3.3..+2.6 cm (1.5 s). The instantaneous command still spikes (up to ~18 cm for ≤ 0.1 s) where the map edge and the
   range edge are crossed at slightly different times (alignment error ~2 cm); PX4's z loop (MPC_Z_P 1.0) averages that
   out. `T_px4` is rate-limited to 5 m/s against range flicker on an edge.
2. **No `dist_bottom` (or out of range): a lag model** — first-order with `hgt_model_tau_up/down_s` (0.5 s), an
   ASSUMPTION about PX4. Right when it matches (−0.3..+0.6 cm), wrong when it does not (PX4 1.5 s vs model 0.5 s:
   −19..+15 cm). Setting tau_up 3 s, tau_down 0 is the "conservative" mode: the vehicle can only err UPWARDS at an
   edge (tested: never below −0.3 cm, up to +45 cm high on the box), the safe direction when the ceiling is far.

During a terrain transient (and for 0.5 s after; with the lag model for 5 s after any step under the vehicle), fixes
update x/y/yaw but not `beta`, so PX4's settling cannot be absorbed into the bias. A FC z reset (reset_counter, z delta
> 1 cm) under the range reference is taken as PX4 re-referencing to the surface now below: `T_px4 := terrain now`,
with the total offset kept continuous. Departs from the brief's "z offset = terrain + bias" by the measured lag term.

## DN-14b — The surface under the vehicle is the one the RANGEFINDER sees, chosen by continuity (sim low_box bug)

Found by the closed-loop sim (`low_box` seed 0, 2026-10-17): near a MAPPED crate edge the 0.1 m voxel map said
"crate top" at the estimated x/y while the beam still hit the floor (and vice versa). `T_px4 = terrain(x̂, ŷ) − L`
was then off by the crate height (+0.49 / −0.58 m pose z error), correct PnP fixes were REJECTED by the z gate, and
two re-anchors pushed the error into beta. The same happens with any false obstacle in the mission's terrain memory.
Fix, when `dist_bottom` is available: the map no longer decides WHERE the edge is, only WHICH heights exist nearby.
- Candidates = distinct terrain heights within `terrain_ambiguity_radius_m` (0.25 m: beam footprint + x/y error) of
  the body (17 samples of the terrain fn, refreshed at 10 Hz).
- Each sample the beam surface is the candidate `c` that keeps the body's map z continuous:
  `|c + dist − z_prev| ≤ max(0.08, 2·1.5 m/s·dt)`. If none fits (a mixed-footprint reading, an unmapped object), an
  unmapped surface already being tracked continues; otherwise the map z is carried and the surface is whatever the
  range implies. A map-explained sample is absolute (`z = c + dist`), so flicker on an edge does not accumulate. The
  PX4 odom vertical velocity is NOT used to explain range changes (PX4's own range flicker corrupts it).
- `T_px4 = surface − L` as before. The surface and `L` flip together on a flicker, so `T_px4` stays smooth.
- While more than one candidate exists (near any edge), fixes do not gate or update z (x/y/yaw still do), so an
  ambiguous spot can never cause z rejections or a z re-anchor.
- If fixes do re-anchor z and a nearby map surface sits exactly that far from the tracked one, the beam surface is
  switched (misidentified) instead of changing beta.
- The lag residual deadband is now smooth (`L·(1 − exp(−(L/db)²))`, ≤ 1.3 cm worst case). The old soft threshold
  offset every slow PX4 transient by the full 3 cm band.
Does not rely on a point beam: the test's footprint (±6 cm) flickers between surfaces or returns a mix next to an edge.
Without `dist_bottom` the lag-model path still uses the map edge, but z stays unused while it is ambiguous.

## DN-15 — Validity: no fix for `max_fix_age_s` AND the drift budget exceeded

`valid(now)` = initialized, newest odom ≤ 0.5 s old, and NOT (no accepted fix for 3 s AND propagated drift σ > 8 cm or
5°). Drift σ grows with time (1 cm/√s) and distance flown (5 cm/√m): moving at 0.5 m/s it goes invalid ~4.5–6 s after
the last fix; hovering (or on the pad) it lasts minutes. This is the brief's §4 wording; the lead's shorter "age ≤
max_fix_age_s" is recovered with `drift_budget_m: 0`. Why: on the pad and in a hover the flow odom barely drifts, and
refusing to fly for want of a fix in the first 3 s would make PREFLIGHT fragile.

## DN-16 — Frame reset: the delta from continuity

`OdomSample` carries no reset delta, so on a `reset_counter` change the delta is computed from continuity: the previous
sample extrapolated with its velocity and yaw rate vs the first new sample (4-DoF). Both alignments, the odom buffer
and pending re-anchor candidates are re-expressed in the new frame, so the map pose does not jump (tested: 1.5 m /
0.25 m / 35° reset → 0.7 cm map-pose step, every fix across it accepted, including ones stamped before it). P is
inflated by 2 cm / 1°; the next fixes correct the residual. Interface gap: PX4 publishes the exact deltas
(`vehicle_local_position.delta_xy/delta_z/delta_heading`); an optional `reset_delta` on `OdomSample` would remove the
continuity guess.

## DN-17 — Initialization

`init_known_pose(T_map_body, stamp)` sets the alignment exactly from the fixed takeoff pose (σ 3 cm, 2°). Before any
odom it is applied at the first sample. With the range reference the body's map z (floor + fc_height_at_rest) and the
odom z (≈ fc_height_at_rest) give `beta ≈ 0`, `T_px4 = terrain at the pad`. Without a known pose, `init_fixes` (2)
consistent fixes initialize (same machinery as DN-11). The mission's REFUSED check stays the mission's own: the aligner
would re-anchor to a consistently different pose after 5 fixes, it does not judge the pad.

## DN-18 — Per-axis scalar Kalman filter, not a joint 4-state EKF

Each of xy (isotropic), yaw and z has its own variance with random-walk process noise; corrections are expressed at
the body (DN-12), which removes the yaw↔translation coupling a joint filter would otherwise have to model through the
lever arm. Simpler to read and tune; the gains adapt to fix rate and distance flown. Synthetic: 2 m / 4.5° of
accumulated drift over 90 s tracked to 1.0 cm median / 3.3 cm max, 0.32° max.

## DN-19 — How the tests judge "z stays right" and "smooth"

The terrain test runs a closed loop: a PX4-like z controller (P 1.0, +1.0 / −0.7 m/s) tracks the converted setpoint
on PX4's OWN estimate, and the vehicle's TRUE map z must stay within 5 cm of the commanded 1.2 m. The instantaneous
command error is printed but not gated (it contains the sub-0.1 s edge spikes PX4 never follows). Smoothness = the
worst change of the odom setpoint of a fixed map point over any 60 ms window, ≤ 3 cm / 0.5°.

## DN-14c — Merged terrain candidates take the surface under the body (main session, 2026-10-17)

`_candidates` merges heights closer than `terrain_transient_band_m`; it used to keep the cluster's LOWEST value. Near a
box edge a voxel mixes side-face points with the top and its point mean reads low (the sim's 0.5 m pad: 0.355 at the
edge vs 0.40 on top), so over the pad the map z sat 5.5 cm low for the whole hover test. A merged cluster now takes
the height straight under the body if that sample is in the cluster, else the cluster's top. Test
`test_candidates_keep_the_surface_under_the_body_not_a_low_edge_voxel`.

# Scout mission decisions (piece D, `navigation/mission.py`; ids DN-60..69)

Brief: `docs/SCOUT_NAV.md` §7. Every value is PROVISIONAL (synthetic room only, no flight data). Earlier mission
fixes were logged with the navigation piece (DN-59 in `navigation.md`); the main session's DN-08 (velocity
feed-forward) and DN-60..62 (hover-test mode, climb ramp, map-free boxes over a mapped surface) are in
`docs/DECISIONS_SCOUT_NAV.md`.

**DN-63 The velocity feed-forward leaves out the height reference's changes (amends DN-08).** The FC's z is the height
above the surface under the rangefinder (DN-03, DN-14), so when the Scout crosses a mapped terrain edge, the aligner
moves the odom setpoint by the step (`setpoint_to_odom`: z_odom = z_map − T_px4 − beta). That move re-references
PX4's z. Its position estimate steps (in the sim, a first-order lag `hgt_step_tau_s`), its velocity does not, and the
true target height does not change. DN-08 fed the odom setpoint's whole rate of change forward as a velocity, so the
step came out as a real vertical velocity command: up to the 0.6 m/s cap, low-passed.
- **Effect.** Crossing crate_c's 0.7 m edge at 0.54 m/s, the vehicle truly dipped 0.27 m (vz −0.37 m/s). Leaving the
  0.4 m pad it bumped up (vz +0.34 m/s). On closed-loop `low_box` seeds 0 and 2 (commit 705e687), the dip happened on
  the "around" route east along y ≈ 5.32 over crate_c's north edge. The map-free looming emergency then saw real
  closing motion toward the crate top (min TTC 0.58–0.61 s) and hovered for 1 s. The map comparison suppressed the
  raw box, so no obstacle was created. It was a needless hover caused by a real, commanded dip.
- **Fix.** `_send_map` differentiates the odom setpoint plus the aligner's current height reference
  (`MapOdomAligner.height_reference_z()` = T_px4 for the range height reference, 0 for baro / EV), i.e. the setpoint
  without the re-referencing. The P term still sees the full odom setpoint, so the vehicle still holds its map height
  across the edge. Horizontal feed-forward (drift cancellation, DN-08) and the beta slew are unchanged. An aligner
  without the method keeps the old behaviour.
- **Why not elsewhere.**
  - Planner: 0.44 m of hull clearance over the crate top is ample. Flying higher would hide a vertical command error
    that grows with the box height.
  - Looming: the emergency was right about the motion it saw. Grazing horizontal surfaces get no special case, and
    the emergency stays independent of the map and the PnP pose (DN-02, DN-25).
- **Measured.** Same seed, same route, HEAD with and without the change (scratch experiment, clear world, goal hover
  point above crate_c): edge dip 0.273 → 0.007 m, vz −0.37…+0.18 → −0.01…+0.04 m/s. Pad-edge vz at NAV start
  +0.34 → +0.09 m/s. `low_box` seeds 0 and 2 go from 1 emergency each to 0. Test
  `test_feed_forward_ignores_a_terrain_step` (fails on the old code: |ff_z| 0.59 m/s).
- **Assumption, open on hardware.** This is right if PX4's VELOCITY estimate does not carry the terrain step:
  EKF2 rejects a large range innovation and resets the height, which moves position only. If the real EKF2 instead
  fused the step gradually into vz, the vehicle would bump the other way at an edge, and the old feed-forward
  would have been closer. It is the same unknown as SCOUT_NAV §3 (how PX4 follows a terrain step). Measure it
  once on the real FC or SITL with `EKF2_HGT_REF=2`: log `vehicle_local_position.vz` against truth while crossing a
  box edge.

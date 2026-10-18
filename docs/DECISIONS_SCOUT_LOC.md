# Decisions log — Scout camera relocalization

Judgment calls made while building the Scout's camera relocalization (ARCHITECTURE §8.2–8.3). Same rules as
`docs/DECISIONS.md` (§14): what, why, what it departs from. Ids are `DL-nn` so this file can be merged into
`DECISIONS.md` later without collisions. Results and the list of missing real data: `docs/SCOUT_LOC.md`.

## DL-01 — The core is a new top-level package `localization/`, ROS-free

ARCHITECTURE §4 has only `ros2_ws/src/nx_scout_loc/` for this. All logic lives in `localization/` (numpy +
OpenCV, Python 3.9-compatible, no ROS imports); `nx_scout_loc` is a thin adapter that imports it.

- Not inside `perception/`: that package is the change-detection pipeline, and this is a separate product
  capability (any camera-carrying platform localizing in a lidar map).
- Not inside the ROS package: it must run and be tested in Windows Python on recorded files, like `perception/`.
- Sensor-agnostic in the DH-09 sense: the camera model, its mount and every threshold are config; nothing in the
  code says "Scout" or "DECXIN".

## DL-02 — Own config file; hardware facts are read, never copied

`config/scout_loc.yaml` holds the localization settings (a new file, so it merges cleanly with the main session).
The lens FOV is read from `config/scout_hw.yaml` (`camera_front.fov_deg_nominal`, DH-06) and the loc-lost hold time
from `config/flight_demo.yaml` (`safety.loc_lost_hold_s`, 1.0 s), so neither number exists twice.

The camera's mount on the Scout (camera pose in the body frame) is NOT measured. `scout_loc.yaml` carries a
placeholder (optical axis = body forward, zero offset, zero tilt), clearly marked; when Ron measures it, the numbers
belong in `scout_hw.yaml` (DH-06) and `scout_loc.yaml` should read them from there.

## DL-03 — Nominal camera model until calibration: pinhole from the 102° DIAGONAL FOV, no distortion

`CameraModel.from_fov(w, h, 102, "diagonal")`: f = (half diagonal in px) / tan(51°) = 323.9 px at 640×480
(horizontal 89.3°, vertical 73.1°). A 102° lens has strong barrel distortion, so this model is only for code paths
and synthetic tests; the code takes OpenCV distortion coefficients everywhere (projection, z-buffer, PnP), so the
calibrated model drops in through `camera.intrinsics_file`. Stream resolution is unknown; 640×480 is a placeholder.

## DL-04 — ORB with CLAHE and grid bucketing, the same extractor for map and query

`features.detect`: CLAHE (clip 2, 8×8 tiles) before ORB, because the mapping camera and the Scout camera see the
room at different exposures; then bucketing into an 8×6 grid so PnP gets points spread over the image instead of a
cluster on the busiest texture. Map side 2000 features per keyframe, query side 1500 (both provisional).

## DL-05 — Landmark depth from the z-buffer: holes are allowed, guesses are not

Each map point is splatted as a disc of radius round(0.5 · f · spacing / z) px (spacing = map point spacing,
0.03 m on the synthetic map), nearest depth wins. A keypoint gets the median depth of the 7×7 window around it, and
is DROPPED if under 60 % of the window has data or the window's depth spread exceeds 5 % (depth edge). A dropped
keypoint costs one landmark; a wrong depth makes a landmark that pulls PnP off.

Chosen on the synthetic dataset (152 keyframes; depth vs the renderer's exact depth at the keypoint):

| splat scale | window | landmarks | median rel. error | > 5 % off |
|---|---|---|---|---|
| 1.0 | 5×5 | 207 k | 0.25 % | 0.43 % |
| **1.0** | **7×7** | **185 k** | **0.21 %** | **0.10 %** |
| 1.5 | 7×7 | 248 k | 0.34 % | 0.39 % |
| 2.0 | 7×7 | 262 k | 0.51 % | 0.95 % |

Bigger splats close holes but bleed near surfaces over far ones at occlusion edges. Real Point-LIO maps are less
uniform than the synthetic jittered grid: `map_spacing_m` and these thresholds must be re-checked on the real map.

## DL-06 — Spatial ratio test

The same wall point is a landmark in several keyframes with almost the same descriptor, so Lowe's ratio test
against the plain second-best match would reject exactly the best-seen points. The second-best is taken as the best
match whose 3D point is more than 5 cm from the first's. Then one query per landmark (best Hamming distance).

## DL-07 — Keyframe retrieval: nearest to the last pose, global vote when lost (no bag of words yet)

§8.3 allows either. Local: the 8 keyframes closest to the prior pose, scored by distance + 1 m per radian of
optical-axis difference, never more than 75° apart. Global (no prior, or the local attempt failed): each query
descriptor votes for the keyframe of its nearest landmark among a fixed 200-landmark subset per keyframe; the top 10
keyframes are matched properly. The subset keeps a global attempt to a few hundred ms on this laptop for a
room-sized map (brute force over all 185 k landmarks took ~730 ms). A bag-of-words index is the upgrade if the map
grows to a building.

## DL-08 — loc_ok semantics, pose hold, reacquisition, plausibility

- `loc_ok` = an accepted fix exists and is at most `loc_lost_hold_s` (1.0 s) old: at exactly 1.0 s still true,
  after it false (§8.3 says "> 1 s"; a 1 µs epsilon absorbs stamp rounding).
- Between fixes the last pose is held. §8.3's "propagate with the FC estimate" is left to the caller: a propagated
  pose can be passed in as `T_prior` to choose keyframes; it is never published as a fix.
- From "lost", 2 consecutive mutually consistent fixes are needed before `loc_ok` turns true again (one stray PnP
  solution must not move the Scout). Startup counts as lost.
- While tracking, a fix that implies more than 2 m/s or 180°/s (+ 0.15 m / 10° slack) since the last accepted fix
  is rejected. This cannot lock the system onto a bad pose: after 1 s without an accepted fix it is "lost" and
  reacquires. These bounds are plausibility checks, NOT the flight limits (0.5 m/s, §8.5). First synthetic run
  found the gate rejecting a TRUE fix during a 290°/s test turn; the test turn was made realistic (75°/s) rather
  than loosening the gate.
- `scout_pose` is published only on accepted fixes (header stamp = the image's stamp), `loc_ok` on every frame and
  from a 10 Hz timer, so a dead camera stream also ends `loc_ok` (`Localizer.loc_ok_at`).

## DL-09 — AprilTag fallback: single-tag ambiguity and a minimum tag size

A lone square tag has two pose solutions (IPPE). Without a prior, a single-tag fix is accepted only if the second
solution's reprojection error is ≥ 3× the first's, OR both solutions put the camera within 10 cm of each other
(face-on and close, the two fit equally well but agree; at 3–4 m they were 2–3.5 m apart). With a prior, the
solution nearer the prior is used (and must be within 1 m). Tags are ignored below 40 px edge. On the synthetic
flight (0.30 m tags, one per wall, mostly 3–4 m away = 15–50 px) single-tag errors with OpenCV's AprilTag corner
refinement were 2–30 cm with one 2 m flip at 22 px; bigger tags or closer views are needed for tag-only flight.

Corner refinement: `CORNER_REFINE_SUBPIX`, not `CORNER_REFINE_APRILTAG`. On four synthetic views of a 0.30 m tag at
0.9–2 m, sub-pixel gave 0.3–3.8 cm / 0.1–1.0°, the AprilTag refinement 1.1–27 cm / 0.6–10° (its corners came out
about half a pixel biased). Re-check on real prints: synthetic edges are perfectly sharp. Multiple tags in view use SQPnP on all corners. Tag pose convention = OpenCV's
(`SOLVEPNP_IPPE_SQUARE`): centre origin, x right, y up, z out of the tag; size = black square edge.

## DL-10 — Synthetic room as the test bed

No hardware data exists. `localization/synthetic_room.py` renders an 8×6×3 m room exactly: each textured planar
surface through a homography (`cv2.warpPerspective`), per-pixel depth from the plane (1/z is affine in pixels), 2×
supersampling. Textures are random shapes and text on low-frequency colour noise (walls 10 shapes/m², floor 3,
ceiling 1.5, boxes 15); one 3.5 m wall section is plain paint (the texture-less case). The map cloud is a 3 cm
jittered grid on every surface + 1 cm Gaussian noise (as DH-12). The mapping walk is 4 laps of an ellipse at 1.25 m
(looking ahead / out / in / back); the test flight is a different ellipse at 0.6–1.4 m with yaw swings, ±6° pitch,
±4° roll, rendered 15 % darker, with σ 2 grey-level noise.

NOT modelled: lens distortion, rolling shutter, motion blur, auto-exposure, specular/glass, repetitive texture,
people moving through the scene, map–image time sync error. These are the real-world risks; the synthetic numbers
prove the logic, not the accuracy.

## DL-11 — Two published poses: body and camera

`scout_pose` is the body pose (forward-left-up, REP-103 `base_link` convention, via the camera mount from config);
`scout_camera_pose` is the optical camera pose, which `nx_scout_avoid` needs to render the expected depth (§8.4 A.3).
Until the mount is measured the two differ only by the fixed optical→body rotation.

## DL-12 — ROS wrapper without cv_bridge, importing the core by path

`nx_scout_loc` converts `sensor_msgs/Image` (mono8/bgr8/rgb8/bgra8/rgba8, any row stride) and `CompressedImage`
itself in a ROS-free module (`convert.py`, unit-tested in plain Python), so it needs no cv_bridge build. The core is
found through the `repo_root` parameter or `$NOXEN_DH26_ROOT`. Not run: there is no camera stream yet, and this
session ran nothing in WSL.

## DL-13 — Landmark database file and mapping inputs

One versioned `.npz` (numpy only): points, descriptors, keyframe index, keypoint pixel and depth, keyframe ids /
poses / stamps, the mapping camera model and build config as JSON. `tools/build_landmark_map.py` accepts either an
explicit keyframe list (JSON) or a TUM-format trajectory ("stamp x y z qx qy qz qw") + timestamped frames, with a
time offset and motion-based keyframe selection (0.3 m / 15°, provisional). How Point-LIO's trajectory becomes TUM
text is not settled (it depends on how the mapping run is recorded); the TUM format is the neutral interface.

## DL-14 — Works with OpenCV 4.6 (the ROS side) as well as 4.9 (Windows)

Ubuntu 24.04's `python3-opencv`, which the ROS node will use in WSL, is OpenCV 4.6.0; Windows Python has 4.9.0.
4.6 has no `cv2.aruco.ArucoDetector` / `generateImageMarker` (added in 4.7), so `localization/apriltag.py` picks the
old function API when the new one is missing (`make_detector`, `marker_image`). Everything else used (ORB,
BFMatcher, solvePnPRansac, solvePnPGeneric, SQPnP, solvePnPRefineLM) exists in both.

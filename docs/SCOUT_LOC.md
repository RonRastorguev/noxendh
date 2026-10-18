# Scout camera relocalization (Demo B, ARCHITECTURE §8.2–8.3)

Given the prior lidar map and one forward camera image, estimate the camera's 6-DoF pose in `map`.
Judgment calls: `docs/DECISIONS_SCOUT_LOC.md` (DL-01..). **Everything here is tested on SYNTHETIC data only**; no
hardware data exists yet, and every threshold is provisional.

## Pieces

| Piece | Where | What |
|---|---|---|
| Core (ROS-free) | `localization/` | camera model, ORB, z-buffer depth, landmark DB, ORB→PnP, AprilTag PnP, `loc_ok` state machine |
| Landmark map builder | `tools/build_landmark_map.py` | map PCD + keyframes (image + pose) + intrinsics (+ extrinsics) → `landmarks.npz` |
| ROS 2 node | `ros2_ws/src/nx_scout_loc/` | image in → `scout_pose`, `scout_camera_pose` (PoseStamped, frame `map`), `loc_ok` (Bool) |
| Config | `config/scout_loc.yaml` | all thresholds (provisional); FOV from `scout_hw.yaml`, hold time from `flight_demo.yaml` |
| Synthetic test bed | `localization/synthetic_room.py`, `synthetic_eval.py` | textured room, exact renderer, map cloud, mapping walk, test flight |
| Tests | `tests/test_scout_loc.py` | 13 tests, plain `python` or pytest, ~2.5 min; also pass in WSL (Python 3.12, OpenCV 4.6) |
| Results table | `tools/eval_scout_loc_synthetic.py` | prints the table below |

Per frame: ORB (CLAHE + grid bucketing) → candidate landmarks of the 8 keyframes nearest the last pose (global
keyframe vote when there is no pose or the local attempt fails) → spatial ratio test → `solvePnPRansac` (EPnP) →
LM refinement → accept if ≥ 25 inliers and reprojection RMS ≤ 3 px → plausibility gate → `loc_ok` state machine
(held for 1.0 s without a fix, 2 consistent fixes to reacquire). AprilTag PnP runs only if ORB gives no fix.

## Commands

```bash
py -3.9 tests/test_scout_loc.py
py -3.9 tools/eval_scout_loc_synthetic.py
py -3.9 tools/build_landmark_map.py --map map.pcd --keyframes keyframes.json --extrinsics lidar_cam.yaml --intrinsics cam.yaml --out data/landmarks.npz
```

ROS 2 (WSL, not run yet; needs `colcon build` of `ros2_ws`):

```bash
ros2 run nx_scout_loc scout_loc --ros-args -p repo_root:=/mnt/c/Users/Ron/Desktop/noxen-dh26 -p landmarks:=/path/to/landmarks.npz -p image_topic:=scout/camera/image_raw
```

## Synthetic results

`py -3.9 tools/eval_scout_loc_synthetic.py`, 2026-10-16, this laptop. Synthetic room 8×6×3 m (DL-10), camera =
nominal 102° diagonal pinhole at 640×480, map = 221 k points (3 cm grid, σ 1 cm noise), 152 keyframes from a
4-lap handheld walk → 184 k landmarks (depth error vs truth: median 0.21 %, 0.11 % of landmarks > 5 % off).
Test flight = a different path, 150 frames at 10 Hz, 0.6–1.4 m high, yaw swings ±40°, pitch ±6°, roll ±4°,
rendered 15 % darker with sensor noise.

| Case | Fixes | Position err cm (median / p95 / max) | Rotation err ° (median / p95 / max) | Inliers (median / min) | ms/frame |
|---|---|---|---|---|---|
| Tracking, held-out flight | 149/150 ¹ | 0.4 / 1.0 / 1.8 | 0.07 / 0.17 / 0.24 | 705 / 472 | 95 |
| Global relocalization, no prior (every 5th frame alone) | 30/30 | 0.5 / 1.5 / 2.0 | 0.07 / 0.22 / 0.37 | 821 / 632 | 282 |
| Map built with lidar↔camera extrinsics off by 0.5° + 1 cm | 149/150 | 1.8 / 3.3 / 5.0 | 0.48 / 0.65 / 0.75 | 692 / 439 | 83 |
| … off by 1° + 2 cm | 149/150 | 4.0 / 7.3 / 11.3 | 0.94 / 1.34 / 1.47 | 650 / 401 | 68 |
| … off by 2° + 5 cm | 147/150 | 9.0 / 17.7 / 23.0 | 2.00 / 2.79 / 3.43 | 460 / 228 | 74 |
| Map noise σ 1 cm → ~2.2 cm | 149/150 | 0.7 / 1.5 / 2.1 | 0.09 / 0.22 / 0.32 | 630 / 371 | 57 |
| AprilTag only (0.30 m tags, 1 per wall, ≥ 40 px) | 23/150 ² | 2.4 / 3.8 / 4.3 | 0.46 / 0.61 / 0.61 | 4 corners | — |

¹ The missing frame is the first one: reacquisition needs 2 consistent fixes by design (DL-08).
² Tags are mostly 3–4 m away (15–40 px) on this flight; smaller detections are refused (DL-09).

**Failure behaviour (texture-less view):** the camera turns from the room to 0.6 m in front of a plain painted wall,
holds 2.5 s, turns back. Last fix at 2.1 s (features vanish during the turn); **zero fixes while facing the wall**;
`loc_ok` stays true through 3.1 s (the held pose, ≤ 1.0 s old) and goes **false at 3.2 s** (1.1 s after the last
fix, the first 10 Hz frame past the 1 s limit); it comes back at 5.3 s, 0.3 s after the view returns (2 consistent
fixes). Tested in `test_loc_ok_goes_false_on_textureless_view_and_recovers`.
**Same wall with a 0.20 m AprilTag on it:** the tag fallback answers every frame of the hold, `loc_ok` never drops,
worst error 2.2 cm / 2.1°.

What this does and does not show: the pipeline logic (z-buffer depth, matching, PnP, gating, fallback) works and
is precise when its inputs are right. The real error budget is dominated by what the synthetic data gets for free:
lens distortion and calibration (DL-03), lidar↔camera extrinsics (row 3–5: error grows about linearly, ~4 cm per
1° + 2 cm), camera–lidar time sync, real texture (the synthetic walls are more cluttered than most real rooms),
lighting, motion blur and rolling shutter. Runtime: ~95 ms per tracked frame and ~280 ms per global attempt in
Windows Python on this laptop at 640×480, single-threaded.

## Real data still needed (in order)

1. **Camera intrinsics of the Scout's DECXIN 2633V2** at the resolution the Radxa will stream: checkerboard
   calibration (§6.9) → YAML `{width, height, fx, fy, cx, cy, dist}` → `camera.intrinsics_file`. Until then the
   model is the 102° *assumed diagonal* pinhole with NO distortion, which a 102° lens is not. Also needed: the
   stream resolution itself.
2. **The mapping camera on the handheld rig.** Best: the Scout's own camera, or the identical table 2633V2, at the
   same resolution and focus, so map and query descriptors are comparable. Its intrinsics too, if it is not the
   Scout's camera.
3. **Lidar↔camera extrinsics of the handheld rig** (camera pose in the L2 frame, or its inverse from a solvePnP of
   cloud points in the image, §6.9) → `--extrinsics`. Sensitivity (table above): 1° + 2 cm off → 4 cm median / 11 cm max
   pose error; 2° + 5 cm → 9 / 23 cm. This is the calibration that matters most.
4. **The Point-LIO prior map** of the flight space (`map.pcd`, frame `map`, z up) and its **trajectory** (lidar
   pose in `map` with timestamps), as TUM text `stamp x y z qx qy qz qw`, plus the map's point spacing after any
   voxel filter (sets `map_spacing_m`).
5. **Camera frames from the mapping walk, synchronized with the trajectory**: `frames.csv` (`stamp,image`), stamps in
   the same clock as the trajectory, or a measured time offset (`--time-offset`). An unknown camera–lidar time offset
   while walking acts like an extrinsics error proportional to walking speed.
6. **Scout camera footage carried by hand** through the mapped space (Milestone 5a: "carry it, don't fly"), with
   some ground truth: tape-marked spots with known map positions, to measure real accuracy and tune
   `min_inliers`, `ransac_reproj_px`, the ORB FAST threshold and the depth-edge thresholds.
7. **The camera mount on the Scout** (camera pose in the body frame): until measured, `scout_pose` = camera
   position with the fixed optical→body rotation (DL-02, DL-11).
8. **For the AprilTag fallback only:** printed tags (family 36h11, black-square edge measured), their poses in
   `map` (e.g. from clicking the tag corners in the Point-LIO cloud, or measured from mapped features), and a
   `tags_file`. At the stream resolution a tag must be ≥ 40 px to be used; 0.30 m tags are usable to ~2.4 m at
   640 px width with this lens, so bigger tags (or several per view) for tag-only flight.

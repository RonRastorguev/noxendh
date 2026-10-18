# Noxen @ hackathon '26 — Architecture

> Spec for Claude Code. Read fully before writing code.

> Repo note (2026-10-15): this is the spec as Ron gave it, unedited. Section 2 arrived with
> only item 5; items 1–4 were not in the text received. Where the build departs from this
> spec, the departure and its reason are in `docs/DECISIONS.md` (the spec text stays as given).

## 1. What we're building

Two demos sharing one codebase:

| Demo | Where | What it shows |
|---|---|---|
| **A. Table demo (live)** | Judging table | Fixed lidar + camera over a 1:10 diorama. Rescan → changes detected → classified → pins on dashboard. Routine changes (crate moved) not flagged; real issues (rust, drum in aisle) flagged. |
| **B. Flight demo (video)** | Recorded Sat night / Sun dawn | Scout localizes with its camera inside a prior lidar map and flies A→B. The same camera detects a person stepping into its path (monocular depth anchored to the prior map, plus looming), and the Scout replans around them. No depth sensor on the Scout. |


## 2. Hard constraints

5. **Dev machine.** Windows laptop + WSL2, GTX 1070. The L2 talks UDP over Ethernet; WSL2 networking must be in **mirrored mode** (or run the driver on Windows natively). Verify this works with Unitree's viewer **before** Saturday.

## 3. System overview

```
                    ┌──────────────────────── Laptop (ground station) ───────────────────────┐
 Unitree L2 ──UDP──▶│ lidar driver (ROS 2) ──▶ scan_capture ──▶ perception (pure Python)       │
 USB camera ───────▶│ camera_capture ─────────▶      │                 │                       │
                    │                                ▼                 ▼                       │
                    │                         data/scans/        classifier adapter ──▶ Perceptron API
                    │                                                  │                       │
                    │                       server (FastAPI + SQLite) ◀┘ ──▶ alerts (Photon, stretch)
                    │                                │  ▲                                      │
                    │                       web dashboard (React) ── "Send Scout" ─┐           │
                    │                                                              ▼           │
 Scout camera ─Wi-Fi▶ scout_loc (camera → map pose) ──▶ scout_nav (planner) ──▶ setpoints ─Wi-Fi▶ Scout FC
              │     │        ▲ prior map + visual landmark map    ▲ obstacles / emergency hover   │
              └────▶│ scout_avoid (map-anchored mono depth + looming) ┘                          │
                    └───────────────────────────────────────────────────────────────────────────┘
```

All heavy compute runs on the laptop, mirroring the product plan (compute in the dock).

## 4. Repo layout

```
noxen-dh26/
  ARCHITECTURE.md          this file
  README.md                setup + run commands
  docs/DECISIONS.md        log of judgment calls made during the build
  config/
    table_demo.yaml        ROI box, thresholds, calibration paths
    flight_demo.yaml
  perception/              pure Python, NO ROS imports — runs on recorded files
    io.py                  load/save scans (.pcd + .png + meta.json)
    preprocess.py          crop, downsample, outlier removal
    register.py            ICP to baseline
    geo_diff.py            added/removed points, clustering
    associate.py           removed↔added matching → "moved"
    visual_diff.py         camera color-change regions
    fuse.py                build ChangeEvents, project bboxes to image crops
    classify/
      base.py              Classifier interface
      perceptron.py        primary
      openai_fallback.py   fallback
      cached.py            replay last good results (demo safety net)
    pipeline.py            run_scan_comparison(baseline, current) -> [ChangeEvent]
  ros2_ws/src/
    nx_capture/            scan_capture + camera_capture nodes → write to data/scans/
    nx_scout_loc/          camera relocalization in map frame
    nx_scout_avoid/        obstacle detection from the Scout camera (mono depth + looming)
    nx_scout_nav/          planner + dynamic obstacle layer + setpoint publisher
  server/                  FastAPI app, SQLite, WebSocket push
  web/                     React (Vite) dashboard
  alerts/                  Photon integration (stretch)
  tools/
    calibrate_extrinsics.py   lidar↔camera
    make_topdown.py           baseline cloud → map PNG + pixel transform
    build_landmark_map.py     visual landmarks anchored in lidar frame
    replay.py                 run pipeline on recorded scans
  tests/
  data/                    gitignored
```

**Key design choice:** `perception/` has no ROS dependency. It's developed and tested against recorded scans, so it can be worked on without hardware and replayed deterministically.

## 5. Data contracts

### Scan on disk (`data/scans/<scan_id>/`)
- `cloud.pcd` — accumulated points in the sensor/map frame, float32 xyz + intensity
- `image.png` — camera frame captured at the end of accumulation
- `meta.json` — `{scan_id, timestamp, accumulation_s, sensor_pose (4x4), roi, is_baseline}`

### ChangeEvent (JSON, also the SQLite row)
```json
{
  "id": "evt_...",
  "scan_id": "...",
  "baseline_id": "...",
  "kind": "added | removed | moved | appearance",
  "bbox_map": {"center": [x,y,z], "size": [dx,dy,dz]},
  "moved_from": {"center": [x,y,z]},          // only for kind=moved
  "crop_before": "path", "crop_after": "path",
  "geo_score": 0.0,                            // e.g. mean point displacement, cm
  "classification": {
    "label": "pallet relocated",
    "routine": true,
    "severity": "none | low | medium | high",
    "rationale": "one sentence",
    "confidence": 0.0,
    "provider": "perceptron | openai | cached"
  },
  "feedback": "none | false_alarm | confirmed",
  "created_at": "ISO8601"
}
```
Events with `classification.routine == true` are shown as routine (teal) and never alert.

## 6. Demo A — perception pipeline

### 6.1 Capture
- The L2 driver publishes a point cloud on ROS 2 (Unitree's public `unilidar_sdk2` driver).
- `scan_capture` accumulates `accumulation_s` seconds of points (default 4 s; the L2's non-repetitive pattern needs time to fill in), then grabs one camera frame and writes a scan folder.
- Triggered by `POST /scans/trigger` on the server, which calls the node (a simple service or HTTP endpoint inside the node).

### 6.2 Preprocess (`preprocess.py`)
1. Crop to the ROI box from config (the diorama volume). **Critical:** this keeps judges' bodies out of the diff.
2. Voxel downsample (1 cm).
3. Statistical outlier removal.

### 6.3 Register (`register.py`)
- Point-to-plane ICP (Open3D) of current → baseline, initialized at identity (table) or at the estimated pose (flight).
- Reject the scan if fitness < threshold or RMSE > threshold. Never diff a badly aligned scan; return an error to the UI instead.
- With a fixed rig this converges at ~identity. Keep it anyway: it's the same code path flight needs.

### 6.4 Geometric diff (`geo_diff.py`)
- **Added:** current points whose nearest baseline neighbor is > τ (default 4 cm; tune from noise test).
- **Removed:** baseline points whose nearest current neighbor is > τ **and** that are visible from the sensor in the current scan. Visibility check: cast a ray from the sensor origin to the point; if the current scan has a return closer along that ray, the point is occluded, not removed. A coarse voxel-grid ray march is enough.
- Cluster added and removed sets separately (DBSCAN, eps ~2 cm). Drop clusters below a minimum point count or size (both in config).

### 6.5 Move association (`associate.py`)
- For each (removed cluster, added cluster) pair, compare bbox dimensions (tolerance ~15%), point-count ratio, and mean intensity.
- Greedy match by best score above threshold → a `moved` event. Unmatched clusters remain `added` / `removed`.
- Unit-test with synthetic clusters. This is the demo's signature moment and must be reliable.

### 6.6 Visual diff (`visual_diff.py`)
Catches appearance-only changes (rust, burns, stains) that geometry misses.
- Fixed camera: assume identity alignment, optionally refined with ECC on grayscale.
- Convert to Lab, compute per-pixel ΔE, blur, threshold, morphological open/close.
- Mask out pixels already explained by geometric events (projected bboxes) and anything outside the projected ROI.
- Remaining connected components above min area → `appearance` events, back-projected to the map via the depth of the nearest lidar points.
- Lighting at the venue will shift between baseline and rescan. Normalize brightness per image, and **[DECIDE]** retake the baseline right before each judge if drift is a problem.

### 6.7 Fuse (`fuse.py`)
- Project each 3D bbox into the image using lidar↔camera extrinsics plus intrinsics, and cut padded before/after crops.
- Build ChangeEvents.

### 6.8 Classify (`classify/`)
```python
class Classifier(Protocol):
    def classify(self, before: np.ndarray, after: np.ndarray,
                 kind: str, context: dict) -> Classification: ...
```
- **Primary: Perceptron** (Mk1 vision-language model). Fill in the API call from their docs at the event; don't guess the API shape now.
- Prompt asks for strict JSON: `label, routine, severity, rationale, confidence`. Give context: "industrial facility inspection; moved inventory is routine; corrosion, damage, spills, and blocked aisles are issues." Include `kind` (a move is a strong prior toward routine).
- Timeout 8 s → fall back to OpenAI → fall back to `cached` (the last good result for a similar event). The UI shows which provider answered; don't hide fallbacks.
- Run classifications in parallel per event.

### 6.9 Calibration (`tools/calibrate_extrinsics.py`)
- Camera intrinsics: OpenCV checkerboard.
- Lidar↔camera extrinsics: click ≥6 corresponding points (box corners) in the cloud and the image → `solvePnP`. Good enough for crops. Save to `config/`.
- Do this first on Saturday; everything downstream depends on it.

## 7. Server + dashboard

### 7.1 Server (FastAPI + SQLite)
| Endpoint | Purpose |
|---|---|
| `POST /baseline` | capture and set a new baseline |
| `POST /scans/trigger` | capture, run the pipeline, store events, push over WS |
| `GET /scans`, `GET /scans/{id}/events` | history |
| `GET /events/{id}` | detail incl. crop URLs |
| `POST /events/{id}/feedback` | `false_alarm` / `confirmed` → stored (the data-flywheel story) |
| `POST /scout/dispatch` | send event location as a nav goal (Demo B / stretch) |
| `WS /live` | pushes `scan_started`, `scan_progress`, `event_created`, `scout_pose` |
| `GET /map` | top-down PNG + pixel↔map transform |

The pipeline runs in a worker thread so the API stays responsive. Progress messages ("aligning", "comparing", "classifying 2/3") stream to the UI; judges should never stare at a frozen screen.

### 7.2 Dashboard (React + Vite)
- **Map panel:** top-down baseline image (`tools/make_topdown.py`, height-colored) with event pins. Coral = flagged, teal = routine, gray = dismissed. Moved events draw a dashed arrow from old to new spot.
- **Event detail:** before/after crops side by side, label, severity, rationale, provider badge, buttons **False alarm**, **Confirm issue**, **Send Scout**.
- **Scan history:** list with timestamps and flagged/routine counts.
- **Header:** "Rescan" button, live progress, connection status.
- Must look professional (20% of the score). Clean, dark-friendly, one accent color. The judge must understand it in 5 seconds.

## 8. Demo B — Scout localization + avoidance

### 8.1 Prior map
- Mount the L2 and a camera on a handheld rig. Walk the flight space and run **Point-LIO** (public, Unitree-supported) → `map.pcd` + trajectory.
- Simultaneously record camera frames with timestamps.

### 8.2 Visual landmark map (`tools/build_landmark_map.py`)
- For keyframes along the mapping trajectory, the camera pose = lidar pose × extrinsics.
- Extract ORB features. Get 3D positions by projecting map points into the image and taking depth at each keypoint (no triangulation needed).
- Save a landmark database: 3D point (map frame) + descriptor + keyframe id.

### 8.3 Scout localization (`nx_scout_loc`)
- Per Scout frame: ORB → match to landmarks (keyframe retrieval via bag-of-words, or just the nearest keyframes to the last pose) → `solvePnPRansac` → pose in map frame.
- Between successful PnP fixes, hold or propagate with the flight controller's own estimate (IMU + optical flow if fitted).
- Publish `scout_pose` plus a `loc_ok` flag. If PnP inliers fall below threshold for > 1 s, `loc_ok = false` and the Scout holds position.
- **Fallback (decide by Sat 8 PM if PnP isn't solid):** AprilTags at surveyed positions in the map frame, then tag-based PnP. Same output topic, so nav doesn't care.

### 8.4 Obstacle detection from the Scout camera (`nx_scout_avoid`)
One forward camera, no depth sensor. Two detectors, all public methods/weights. Runs on the laptop GPU (GTX 1070) on the streamed Scout frames.

**A. Map-anchored monocular depth (primary; any obstacle)**
1. Undistort the frame (calibrated intrinsics).
2. Run **Depth Anything V2 Small** (public weights) at reduced resolution. Its output is *relative* inverse depth, correct up to an unknown scale and shift. **Measure FPS on the 1070 at the event; target ≥ 8 Hz.** If too slow, drop resolution or skip frames. MiDaS small is the backup model.
3. **Render expected depth** from `map.pcd` at the current Scout pose (from `scout_loc`): project map points into the image with a z-buffer, then dilate to fill gaps. Sparse is fine.
4. **Fit scale + shift** per frame: robust (RANSAC / Huber) least squares between the model's inverse depth and rendered inverse depth, using pixels that have a map value. Most pixels are static background, so the fit ignores the obstacle as an outlier. Result: metric depth for every pixel.
5. **Obstacle pixels:** metric depth < rendered map depth − 0.5 m, within 4 m of the camera. Morphological open, connected components ≥ `min_area_px`.
6. Back-project obstacle pixels using their metric depth and the Scout pose → points in `map` → publish `[{x, y, radius, stamp}]`.
7. Skip the frame (publish nothing new) if `loc_ok` is false or the scale fit has too few inliers. A bad pose renders a bad expected depth, which creates fake obstacles.

Judge talking point: "the prior map gives the camera its scale." The single-camera depth model on its own doesn't know distances; anchored to the map, it does.

**B. Looming / time-to-contact (fast safety net; also the fallback)**
1. Person detection with **YOLO11n or YOLOv8n** (public weights) on each frame.
2. Distance ≈ `f_px × 1.7 m / bbox_height_px` (assumes an adult standing).
3. Time-to-contact = `h / (dh/dt)` from the bbox height across frames (smoothed over ~5 frames). This works for any tracked object, without depth.
4. If time-to-contact < 1.5 s **or** distance < 1.2 m → publish `emergency_hover`, independent of the planner.
5. If path A isn't passing its gate by Sun 12 AM, B alone drives avoidance: the detected person becomes a map obstacle using the distance estimate + Scout pose.

**Known limits (say these if judges ask):**
- Only sees what's in front of the camera. Fine for forward flight at ≤ 0.5 m/s; the full product adds the Overseer's lidar.
- Path A depends on good localization; it's gated on `loc_ok`.
- Monocular depth is noisy on blank walls and glass. The map anchoring is why it's usable at all.
- Latency (camera → laptop → model → setpoint) limits speed; measure it end to end.

**[DECIDE] with Spark before Saturday:** if the new Scout can carry a second camera 8–10 cm beside the first, use stereo (OpenCV `StereoSGBM`) instead of path A. Stereo gives metric depth directly and is more reliable. Keep B either way.

### 8.5 Avoidance + planning (`nx_scout_nav`)
- **Static layer:** 2D occupancy grid from `map.pcd`, sliced at flight altitude ± margin, inflated by Scout radius + 0.3 m.
- **Dynamic layer:** obstacles from `nx_scout_avoid`, inflated by Scout radius + 0.5 m, decaying ~1.5 s after last seen. Obstacles out of the camera's view are kept until they decay; the Scout must not turn into a spot it can't see that was just occupied.
- **Emergency hover:** `emergency_hover` from looming overrides everything for 1 s, then the planner replans.
- **Planner:** A* on the merged grid at 10 Hz, path shortcutting, then a constant-speed setpoint follower.
- **Safety gate:** max speed 0.5 m/s, geofence = map bounds minus margin. Hover if `loc_ok` is false or there's no path. Kill switch on Spark's RC.
- Output: position setpoints to the flight controller. **[DECIDE]** What flight controller and link does the new Scout use? If PX4: offboard mode via uXRCE-DDS over Wi-Fi. Confirm with Spark before Saturday.
- **[DECIDE]** Does the Scout carry its own compute, or stream camera to the laptop? Default assumption: stream to the laptop. Measure the latency; above ~150 ms, cap speed lower.

### 8.6 Frames
`map` (from Point-LIO) is the world frame for everything. Scout poses and obstacles are in `map`. Table demo uses `sensor` = `map` (fixed rig).

## 9. Alerts — Photon (stretch, only after Milestone 5)
- On a new flagged (non-routine) event: send an SMS/iMessage like "Noxen: possible corrosion on tank, NW corner. Reply LOOK to send the Scout."
- Inbound "LOOK" → `POST /scout/dispatch`.
- Built on Photon 2.0 (beta from Oct 15). Read their docs at the event.

## 10. Milestones (time-boxed)

| # | By | Deliverable | Gate (must pass to move on) |
|---|---|---|---|
| 0 | Sat 1:30 PM | Repo scaffold, driver streaming, scan capture writes folders | A scan folder with a sane cloud + image |
| 1 | Sat 3:30 PM | Calibration, preprocess, ICP, geometric diff on recorded scans | Drum added → exactly 1 `added` cluster, 0 false clusters, over 3 trials |
| 2 | Sat 5:30 PM | Move association + visual diff | Crate move → 1 `moved` event; rust swap → 1 `appearance` event; 3/3 trials |
| 3 | Sat 7:30 PM | Classifier adapter (Perceptron + fallbacks) + server endpoints | `POST /scans/trigger` returns classified events end-to-end |
| 4 | Sat 10:00 PM | Dashboard v1 (map, pins, detail, rescan) | Full table demo runs from the UI. **Devpost draft submitted by 10:30.** |
| 5a | Sun 12:00 AM | Prior map + landmark map + Scout loc + obstacle detection, tested on **handheld** Scout footage (carry it, don't fly) | Person at tape-marked spots → obstacle within ±0.3 m; empty path → no obstacles; looming fires as you walk toward the camera |
| 5b | Sun 2:00 AM | Nav + flight (AprilTag loc fallback if needed) | Scout holds pose, flies A→B, replans around a person |
| 6 | Sun 5:00 AM | Record flight video; dashboard polish | Usable clean takes saved |
| 7 | Sun 8:00 AM | Demo video cut, README, Devpost final | Submitted with a buffer before 10:00 |
| — | stretch | Photon alerts | Only if 0–6 are done |

If Milestone 5 is slipping at 1 AM, cut to the AprilTag fallback immediately. Do not trade sleepless hours on localization for a broken table demo.

## 11. Testing
- Record a baseline plus one scan per scenario (crate moved, rust swapped, drum added, no change) **early Saturday**. All perception development runs on these via `tools/replay.py`.
- The **no-change scan must produce zero events**. This is the false-alarm test, and it matters more than anything else.
- `tests/test_associate.py`: synthetic clusters for move matching.
- `tests/test_pipeline_replay.py`: expected event kinds per recorded scenario.

## 12. Config defaults (`config/table_demo.yaml`)
```yaml
roi: {min: [x,y,z], max: [x,y,z]}   # measured Saturday
accumulation_s: 4.0
voxel_m: 0.01
diff_threshold_m: 0.04              # tune from noise test
cluster_eps_m: 0.02
min_cluster_points: 50
min_cluster_size_m: 0.04
icp: {max_rmse_m: 0.01, min_fitness: 0.8}
visual: {delta_e_threshold: 18, min_area_px: 400}
classifier: {primary: perceptron, timeout_s: 8, fallbacks: [openai, cached]}
```

## 13. Risks

| Risk | Mitigation |
|---|---|
| Lidar noise causes false clusters | Noise test pre-event; tune τ and min cluster size; the no-change gate |
| Venue lighting shifts → visual false positives | Brightness normalization; re-baseline before each judge |
| Ballroom Wi-Fi too slow for classifier calls | Phone hotspot; cached fallback |
| WSL2 can't receive L2 UDP | Mirrored networking or run the driver on Windows; verify pre-event |
| Camera-in-lidar-map localization unreliable | AprilTag fallback, decided by Sat 8 PM |
| Depth model too slow on GTX 1070 | Lower resolution or skip frames; MiDaS small; looming-only fallback |
| Bad Scout pose → fake obstacles from map-anchored depth | Gate on `loc_ok` and scale-fit inliers; looming still runs |
| Camera only sees forward | Low speed, obstacle memory with decay, no sharp turns |
| Scout/laptop link latency | Low speed cap; hover on stale pose |
| 70/30 violation | New repo, public deps only, organizer sign-off |

## 14. Instructions for Claude Code
- Work milestone by milestone; don't start the next until the gate passes. Report gate results explicitly.
- Log every non-obvious choice (thresholds, library swaps, cut features) in `docs/DECISIONS.md`.
- Prefer simple, readable code over clever code. Keep `perception/` ROS-free.
- Never commit `.env`, `data/`, or API keys.
- Don't invent external API details (Perceptron, Photon). Leave a clearly marked stub and ask.

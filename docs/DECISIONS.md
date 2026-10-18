# Decisions log

Judgment calls made during the build (ARCHITECTURE.md §14). One entry per decision: what, why,
and what it departs from. Newest last. Ids are `DH-nn`.

## DH-01 — Reusing our existing flight-stability software on the Scout is allowed (Ron, 2026-10-15)

Ron checked with the organizers: our existing flight-stability / safety software
may be reused for the Scout because it is not the main part of the demo. It is brought in as a
port adapted to the Scout's hardware (DH-05, DH-06), not as a wholesale copy of an existing
stack. Everything the demo is judged on (perception, classification, dashboard, Scout
localization and avoidance) is written new in this repo.

## DH-02 — L2 driver: our own protocol parser (approved reuse), never Unitree's `unilidar_sdk2`

Departs from ARCHITECTURE §6.1 and §8.1, which name `unilidar_sdk2`.

- `drivers/unitree_l2/l2_protocol.py`, its test and the two real-sensor `.pcap` fixtures are
  reused verbatim from our earlier L2 work (organizer-approved, below). Pure Python +
  numpy, no vendor code. The tests (95 checks, incl. real-sensor captures) pass unchanged here.
- **Why not the vendor library:** its `initializeUDP` sets the sensor clock and its examples write
  the work mode. There is one L2 and it is shared with our other drone; running the vendor library could
  leave it configured differently from what its bench measurements assume.
- **Why not write a new parser:** this one already handles the real unit's quirks (DataInfo.stamp
  at 0.5x real, zlib CRC, IMU + point frames coalesced into one UDP datagram, 10-bit seq wrap).
- **Organizer sign-off: given (Ron, 2026-10-15).** The ported driver stays. (Was open as
  `NEEDS_RON.md` R1; the fallback, a fresh parser, is no longer needed.)
- Point-LIO (§8.1) gets its cloud + IMU from a ROS wrapper around the same parser, never from
  Unitree's launch files (which start the vendor library).

## DH-03 — Demo A capture runs natively on Windows, without ROS

Departs from ARCHITECTURE §3 / §4 / §6.1 (`nx_capture` ROS nodes under `ros2_ws/`).

- `capture/` is plain Python: a UDP receiver built on `l2_protocol` + an OpenCV camera grab, which
  writes the scan folder through `perception/io.py`. It runs in Windows Python.
- **Why:**
  1. The table camera is USB. WSL2 cannot see USB devices without `usbipd`, which this laptop
     does not have.
  2. It takes DDS and WSL networking off the live-demo path. On this host, default FastDDS
     discovers but drops data in about half of fresh pairings (measured on this laptop); a judging
     table is the wrong place to meet that.
  3. A fixed sensor needs no device-clock → host-clock sync; accumulation just collects points
     for `accumulation_s`.
- ROS 2 (WSL, Jazzy) stays for Demo B: Point-LIO, the Scout nodes, the PX4 link.
- `ros2_ws/src/nx_capture/` is therefore not created.

## DH-04 — Network: the laptop takes the L2's default host address

The L2 ships sending to `192.168.1.2:6201` from `192.168.1.62:6101` (its factory
defaults). The laptop's Ethernet adapter is set to static `192.168.1.2/24`, so the L2 needs
no reconfiguration at all. `config/table_demo.yaml` `lidar.local_ip` / `local_port` follow it.

## DH-05 — Scout compute split: laptop does the thinking, the Scout keeps itself safe

The Scout's onboard computer (Radxa Zero 3W, RK3566, 4× Cortex-A55; roughly a third to a quarter
of a Raspberry Pi 5's CPU) runs only:

- the link to the flight controller (uXRCE-DDS agent),
- ~~an onboard watchdog: stale setpoints, stale pose or a lost Wi-Fi link → hover, then land~~
  (moved to the flight controller, see the amendment below),
- the camera stream to the laptop.

Localization, depth, person detection and planning run on the laptop (§8). The watchdog is the
product rule for the Scout (a minimal onboard fallback; the Scout–laptop link
must not be a single point of failure). The laptop's looming `emergency_hover` (§8.4 B) is lost
with the link, so the onboard stale-setpoint timeout is its independent backstop.

**Amended the same day (2026-10-15): the backstop lives on the flight controller, not the Radxa.**
PX4's own offboard-loss failsafe (setpoints stop arriving for longer than a set time → hold, then
land) catches a dead laptop, a dropped Wi-Fi link AND a dead Radxa, because all three stop the
setpoints. A watchdog program on the Radxa would die with the Radxa, which is exactly the
classic companion-loss risk. If the laptop's pose stops, PX4's estimator falls back
to flow + rangefinder and holds. So the Radxa runs no safety logic; the exact PX4 parameters
(timeouts, actions) are chosen when the stability set is ported.

## DH-06 — Scout hardware facts (Ron, 2026-10-15)

Recorded in `config/scout_hw.yaml` (the only place these numbers are written):

- FC: **MicoAir 743 AIO**. Ron will flash it to PX4. Target PX4
  1.18.0 RC `fca3df86`, so the stability parameters and adapter port as-is.
- Companion: **Radxa Zero 3W** (per the budget sheet; board marked Rockchip / Radxa). RAM / eMMC
  variant unknown.
- Camera: **DECXIN 2633V2** (Ron confirmed 2026-10-15; first read
  as "3633V2"). The table demo's camera is a second, identical 2633V2, not mounted on the Scout.
  The 102° lens figure is assumed diagonal (provisional) until calibrated.
- Downward sensing: **MicoAir MTF-02P** (rangefinder + optical flow), the only downward range sensor.
- Flow / range module position, hand-measured ("about"):
  - lens **1.7 in (0.0432 m) above the floor** with the Scout at rest;
  - **0.5 in (0.0127 m) below the FC**, **0.3 in (0.0076 m) to the right**, and level with the FC
    along the forward axis (FC → camera).
  - So the FC rests about 2.2 in (0.0559 m) above the floor. "On the ground" is never z = 0.
  - In PX4's body frame (FRD) the module sits at x 0, y +0.0076, z +0.0127 m from the FC
    (`EKF2_OF_POS_*` and `EKF2_RNG_POS_*`).
- At rest the rangefinder reads ~4.3 cm, inside its range (2 cm dead zone), but the flow image at
  4 cm is likely poor: flow must not be relied on until the Scout is airborne. The flow camera
  needs > 60 lux, so the recording space must be lit.

## DH-07 — The Radxa runs Armbian Debian 13 minimal, without ROS

Ron's plan was to flash "whatever is on the Pi". A Raspberry Pi image will not boot on the Zero 3W
(different SoC and bootloader); it needs an RK3566 / Zero 3 image.

- First proposal (superseded the same day): an Ubuntu 24.04 image, to match the Pi's ROS 2 Jazzy.
  Armbian's `radxa-zero3` page (checked 2026-10-15) no longer offers 24.04, only Ubuntu 26.04 and
  Debian 13.
- **Decision:** Armbian **Debian 13 "Trixie" Minimal, vendor kernel 6.1** (Armbian's hand-picked
  stable image for the board), flashed with Armbian Imager, which presets user, Wi-Fi and SSH key.
- **Why no ROS onboard:** with DH-05's backstop on the flight controller itself, the Radxa only
  relays (FC serial ↔ laptop) and streams the camera. Both are plain Linux programs (the uXRCE-DDS
  agent is C++ with no ROS dependency; the camera stream is GStreamer / ffmpeg). Keeping ROS off a
  4× Cortex-A55 board also leaves CPU for the stream.
- Steps for Ron: `docs/SETUP.md` §2. Laptop → Scout SSH key: WSL `~/.ssh/id_ed25519_scout`
  (public half in `config/ssh/scout_laptop_key.pub`).

## DH-11 — How the Radxa actually got its OS (2026-10-15)

- Armbian Imager 2.x showed Ron no first-boot settings, and `wsl --mount` refuses a disk behind a
  USB SD reader. So the presets were written into the downloaded image (sha256 checked against
  Armbian's) inside WSL, Ron typed the secrets himself (`fill_secrets.sh`), and flashed it with
  Raspberry Pi Imager "Use custom" from the `\\wsl.localhost` path (`tools/scout_sdcard/`).
- **Incident: C: ran out of space** while decompressing (0 GB free; WSL's `ext4.vhdx` could not
  grow), so WSL's ext4 hit I/O errors and remounted read-only. Ron freed space; WSL restarted
  (journal recovery clean). The other session's `px4replay.sh` job did not survive the restart.
  Rule taken: images are flashed from WSL directly, never copied onto C:.
- **Incident: first login stalled on `tzselect`** (no `PRESET_TIMEZONE`; Armbian's docs say it
  detects one, it did not). Unstuck by TERM-ing `tzselect` on the Radxa (the script then set no
  timezone = UTC, ran `provisioning.sh` and deleted the preset file). `write_presets.sh` now sets
  `PRESET_TIMEZONE="Etc/UTC"`.
- The modified image (it held the plaintext Wi-Fi key and passwords) was deleted from WSL after
  first boot; the verified `.img.xz` is kept, so it can be rebuilt.
- Result: Armbian 26.8.1 / Debian 13 / kernel 6.1.115-vendor, 2 GB RAM, 29 GB root, on Wi-Fi
  `AlphaSigmaPhi` (2.4 GHz channel 1, 20 MHz: worth remembering for the camera stream's bandwidth).

## DH-08 — Scan format: own PCD reader/writer, meta.json is a superset

`perception/io.py` writes binary PCD (`x y z intensity`, float32) with its own small writer/reader,
so a recorded scan loads with numpy alone (no Open3D needed to read data, any OS). `meta.json`
carries the spec's keys plus `frame`, `n_points`, `source` and capture stats; readers ignore keys
they do not know.

## DH-09 — Perception is built for the moving product, configured for a fixed sensor

Every cloud carries its frame and `sensor_pose` (sensor → map, 4x4). "Stationary" is config
(`sensor_pose` = identity, `sensor` = `map`), never a code path, so the same pipeline later takes
registered clouds from a moving drone. De-skewing a moving sensor's points belongs upstream
(LIO), not in `perception/`.

## DH-10 — Windows Python 3.9 for now

The only Windows Python with numpy (1.26.4) and OpenCV (4.9.0) installed is 3.9. Code stays
3.9-compatible (`from __future__ import annotations`, no `match`). Moving to a newer Python is a
Windows-side install for Ron (`NEEDS_RON.md` R4), not a blocker.

## DH-12 — Synthetic ray-cast scans until the real noise test

`perception/synthetic.py` ray-casts an L2-like accumulated scan of a diorama (table plane + boxes) from a fixed sensor
0.7 m above it, with Gaussian range noise (σ 1 cm; the datasheet says ±2 cm). Because it is ray-cast, occlusion is real,
which the diff's visibility logic needs. Every threshold tuned on it stays provisional until the Saturday noise test
on the real L2 replaces the noise model (§11).

## DH-13 — The diff checks visibility BOTH ways

§6.4 asks for one check: a baseline point missing from the current scan only counts as REMOVED if the current scan
could have seen it (ray from the sensor not blocked). The mirror case is needed too: a current point missing from the
baseline only counts as ADDED if the baseline could have seen that spot. Otherwise every removed object also produces
an "added" patch of newly revealed table. Synthetic tests: crate removed → 1 removed, 0 added; drum added → 1 added,
0 removed; crate moved → 1 + 1; no change → 0 (3 seed pairs each).

## DH-14 — ICP: random subsample, up to 200 iterations, structure fitness

Found with the synthetic sweep (3 seed pairs × 18 offsets):
- **Not the voxel grid as ICP input:** two clouds snapped to the same 1 cm lattice have a false minimum where the
  lattices line up (2–3 cm knock → stuck 7.6 mm off). ICP gets a random 40 000-point subsample (`preprocess.for_registration`).
- **max_iter 30 → 200:** at 30, ICP stopped mid-slide on the plane-dominated scene (15 cm off → 1.5 cm left; 20 cm off →
  7–8 cm wrong) and 15 of 54 cases were wrong but ACCEPTED (RMSE 9.6–9.9 mm under the placeholder 10 mm). At 200 every
  case converges to ~1 mm out to 25 cm (120–160 iterations). Cost: 4–9 s per registration on this laptop — needs a
  coarse-to-fine speed-up before the live demo.
- **Structure fitness** (new): the match share counted only on points off the dominant plane, per face direction, worst
  direction wins. Overall fitness cannot see a slide along the table (a 25 cm offset still scored 0.82). Correct
  alignments score 1.00; the wrong ones in the sweep scored 0.64–0.88. Reject below 0.9 (provisional, `min_structure_fitness`).
  A 90° misrotation is refused.

## DH-15 — Move association compares TOP height and footprint, not the change cluster's extent

A change cluster's z extent is not the object's height: the bottom ~τ of any object sits within τ of the old table
points and is cut out of the diff (a 6 cm crate left a 2.9 cm sliver before and 2.3 cm after a move: 21 %, over the
spec's 15 % tolerance, so the move was missed). Clusters now carry `top_z` (95th percentile z) and `footprint`
(5th–95th percentile x/y, sorted, so a 90° turn still matches); association compares those (top within `dim_tol`,
footprint within 2×`dim_tol`), plus point-count ratio and mean intensity.
**Amended (Milestone 3 pipeline tests):** `top_z` is the cluster's MEDIAN z, not the 95th percentile: viewed straight down
vs at an angle the same crate's 95th-percentile top read 8.2 vs 7.1 cm (range noise lands in z differently), while the
median moved 0.18 cm on average (20 runs). And each size test allows max(relative tolerance, `dim_abs_tol_m` 2 cm,
≈ 2σ of the synthetic noise; provisional): 15 % of a 6 cm object is finer than the sensor. Result: the move found in
20/20 synthetic runs, score min 0.66 / median 0.81 (bar 0.6). Synthetic: crate moved → 1 moved, 3/3 trials;
crate moved + drum added → 1 moved + 1 added.

## DH-16 — Visual diff: ECC alignment + robust lighting normalisation before ΔE

`perception/visual_diff.py`: Euclidean ECC aligns the rescan to the baseline (a 5 px nudge otherwise produced 2 false
regions; recovered to within 0.02 px), then the rescan's Lab lightness median/spread and a/b medians are matched to the
baseline's (median/MAD, so the change itself barely moves them), then CIE76 ΔE, blur, threshold 18, open/close,
components ≥ 400 px (spec defaults). Synthetic frames: rust → 1 region at the patch; darker/warmer lighting alone → 0;
lighting + rust → 1; speck → 0; masked → 0. Back-projection of regions to the map waits for camera calibration (fuse.py).

## DH-17 — ICP speed: stop at 0.1 mm, coarse pass first

Registration took 4–9 s (too slow for a judge pressing Rescan). Two causes, two fixes:
- the stopping tolerance (1e-6) sat below the noise floor, so a fixed rig iterated 83 times for nothing → `tol` 1e-4
  (0.1 mm / 0.006° per step): 4.7 s → 0.6 s of ICP;
- long slides took ~110 full-cloud iterations → a coarse pass on 4 000 points with a 10 cm radius first, then the full
  cloud at 5 cm.
Measured on this laptop (40 k points, incl. normals + structure fitness): fixed rig 1.2 s, 2 cm + 1° knock 1.7 s,
15–25 cm off 2.6–2.7 s; all within 0.3–0.7 mm, all accepted.

## DH-18 — Milestone 3: synthetic capture source, a `rules` last resort, events stored whole

- **Synthetic source** (`source: synthetic` / `NOXEN_SOURCE=synthetic`): the server writes ray-cast scans + rendered frames
  of the §11 scenarios (`no_change`, `crate_moved`, `drum_added`, `rust_tank`, `everything`) as ordinary scan folders, with
  an exact synthetic camera calibration, so the pipeline / server / dashboard run end to end before the hardware is
  connected. Live mode uses `capture_scan` unchanged.
- **Classifier chain** = config order (Perceptron → OpenAI → cached) + a final **`rules`** provider (beyond the spec):
  it only restates what the geometry knows (moved = routine; added / removed / appearance = medium), confidence 0.3,
  provider "rules", so an event is never left unclassified when the Wi-Fi dies. Every attempt is stored on the event
  (`classification_attempts`): fallbacks are visible, never hidden (§6.8). **Perceptron is a stub** that raises
  NotConfigured until its API is filled in from their docs at the event. OpenAI needs a key AND a model name
  (no model is assumed).
- **Storage:** SQLite, the ChangeEvent JSON stored whole plus queryable columns; every feedback verdict also goes to
  `feedback_log` (the data flywheel). One scan at a time on a worker thread (`409` when busy); progress over `WS /live`.
- Measured: synthetic "everything" rescan on the real uvicorn server, 3.4 s end to end (2 flagged, 1 routine).

## DH-19 — Dashboard: one process, one URL; cividis map

- React 18.3.1 + Vite 5.4.10 in `web/` (exact pins). `npm run build` → `web/dist`, which the FastAPI server mounts at "/"
  after every API route: the demo is one process on one URL (no CORS, no second server to forget at the table).
  `npm run dev` proxies the API for UI work.
- What a judge sees first: "N issues · M routine" in large type, then the map with numbered pins (coral flagged / teal
  routine / grey dismissed, dashed arrow for moves), then the selected change: label, severity, provider badge with the
  fallback count, rationale, before/after crops, False alarm / Confirm issue / Send Scout. Rescan shows a live stage
  line ("Aligning to baseline", "Comparing geometry", "Classifying 2/3"). One accent colour (blue) for chrome.
- The top-down map uses cividis (colour-blind safe; the status colours stand out on it); small sampling gaps are
  inpainted, real lidar shadows stay visible in slate.
- Checked in the browser pane against the synthetic server: baseline, Rescan (stages appear, buttons lock), feedback
  (pin turns grey, counts update), phone width 375 px (one column, no horizontal scroll).

## DH-20 — Ron's diorama layout in the synthetic model; a dropped object must clear τ

Ron (2026-10-17): the diorama is a 0.6 × 1.0 m rectangle; two shelves along the top end; a tank that gets rust
(flagged); a crate that is moved (routine); something dropped in the aisle (flagged). `perception/synthetic.py`
models exactly that (long axis = map y, top = +y, an aisle down the middle, a pallet stack for realism) and the
scenarios are `no_change`, `crate_moved`, `rust_tank`, `dropped_in_aisle`, `everything`. The synthetic camera is
portrait (480 × 640, 64°) so the whole rectangle is in frame from 0.7 m, and boxes are drawn with their sides.
**Finding for building the real diorama:** a dropped object 4 cm tall (at 1:10) sits at the change threshold τ (4 cm,
the spec default for the L2's ±2 cm noise) and is invisible to the geometric diff; only the camera caught it, as a
"surface change". The dropped object should stand ≥ 5–6 cm tall at diorama scale (a fallen drum), or τ comes down after
the Saturday noise test. Tests updated to the layout; the lattice negative test (DH-14) is pinned to the old flat layout
(this layout's shelves give ICP enough structure not to stick), and a tall object near the edge now proves occlusion.

## DH-21 — Dashboard restyled to noxensys.com; the lidar shown in 3D

Ron (2026-10-17): match noxensys.com (minimal, black and orange) and show the lidar in 3D.
- **Theme = the site's own tokens**, read from its CSS: accent `#FF5A1F` (ink `#120602`), backgrounds `#0A0C0F` /
  `#0D1014`, surface `#111419`, lines `#1C2128` / `#2A3039`, text `#E9EBEE` / `#8B929C` / `#5A616B`, radius 4 px; Geist +
  Geist Mono (Google Fonts, as the site loads them); the site's Noxen mark and "NOXEN SYSTEMS" Michroma wordmark;
  section labels as small uppercase Geist Mono eyebrows, like the site's "Baseline / Tonight". Orange doubles as
  "issue"; routine = a muted teal (`#5EC2B0`), dismissed = the site's dim grey.
- **3D view** (three.js 0.169.0, pinned): tonight's cloud in a monochrome height gradient; points belonging to a change
  take its status colour; each change gets an outline box and a floating, clickable number; a moved object shows a
  dashed ghost box at its old place and a dashed link; Tonight / Baseline and 3D / Top toggles; orbit / zoom / pan;
  selecting a change glides the focus to it; a slow idle spin stops at the first interaction. It replaces the 2D map
  (`GET /map` stays). New endpoint `GET /scans/{id}/cloud` (float32 x, y, z, intensity, label, ghost); the pipeline saves
  `display.npz` per scan (labels = event index; "ghost" = baseline points that disappeared).
- Checked in the browser: 1366 × 860 and 375 px phone width (no horizontal scroll, wordmark on one line), selection,
  views, no console errors on a fresh load.

## DH-22 — iMessage alerts through Photon Spectrum (the §9 stretch)

Ron (2026-10-17): a flagged issue texts his iPhone with a before and after image and asks what to do (send the Scout,
real issue, false alarm); his reply is acted on and confirmed.
- **Photon's stable SDK, `spectrum-ts` 12.11** (project id is not `pho_prj_…`, so not the v2 beta API), cloud iMessage
  provider, Free plan. The SDK is TypeScript-only, so `alerts/` is a small Node process beside the Python server,
  not part of it: it reads `WS /live` (`event_created`, `event_updated`) and calls the existing REST endpoints. The
  server needed no change.
- **Who and what alerts:** one recipient (`NOXEN_ALERT_TO`), every event with `classification.routine != true`
  (§5: routine never alerts). One text (numbered flag, label, severity, rationale, map x/y) plus the two crops as PNG
  attachments; text only when the event has no crops.
- **Replies are plain text, not an iMessage poll.** Polls need iOS 26 on the phone and a separate answer path; "1 / 2
  / 3" works on any phone and in the SMS fallback. Words work too (`scout`, `issue`, `false alarm`), `#N` targets an
  older flag, `list` / `help` answer themselves. A bare reply answers the newest open flag. Only messages from the
  alert number are acted on.
- **Mapping:** 1 → `POST /scout/dispatch` (the flag stays open), 2 → feedback `confirmed`, 3 → feedback `false_alarm`.
  Feedback given on the dashboard closes the flag silently.
- **Dashboard switch (Ron, 2026-10-17): texts only when "Text alerts" is on.** Stored server-side (`POST /alerts`
  `{"text": bool}`, in `/status`), read when a scan STARTS and stamped on that scan's `event_created` as `text_alert`;
  the bridge texts only stamped events, so flipping the switch mid-scan never splits one scan's alerts. The bridge
  announces itself on `WS /live` ("bridge"); `/status.text_bridge` lets the dashboard show "bridge off" instead of
  failing silently. Checked in the browser on a synthetic server: off → rescan → nothing sent; on → rescan → flag
  texted with both crops. Test: `tests/test_server.py::test_text_alert_toggle_stamps_events`.
- **The Scout reply does not quote the dispatch status** ("queued: the Scout link is not connected yet"): the Scout
  does not fly in the live demo (shown on video), so the text says "Scout requested for flag #N"; the status is
  in the bridge log.
- **State** (flag numbers ↔ event ids, status) in `data/alerts_state.json`, so numbering survives a restart.
- **Free-plan limits, from Photon's docs and dashboard:** a shared line only messages users of the project (Ron
  added, +1 206…, assigned shared line <redacted>), and it cannot send the first message: the recipient texts
  their assigned number once (Ron's: <redacted>). Until he did, every send was refused with "Target not allowed
  for this project", even with him listed as a user; the verified account phone, the debug-bot handle check and a
  second user changed nothing. **Tested 2026-10-17 after Ron's first text:** his "hi" reached the bridge and the help
  reply reached his phone; a synthetic `rust_tank` scan (separate synthetic server on :8001; one trigger first went to Ron's live server by mistake, failed alignment, no events)
  sent flag #1 with both crops; Ron's reply "1" -> `POST /scout/dispatch` 202, dispatch row queued for that event. Each user gets their own pool number ("Texts on"), so the agent's number differs per
  recipient. 50 new conversations per line per day. A dedicated number is Business, $250 per line
  per month.

## DH-23 — The table frame: the diorama rectangle is found from a scan, every scan is stored in it and cropped to it

**Ron, 2026-10-17:** "can you make it so the lidar only captures like a rectangle of stuff (the diorama) right in
front of it".

**Decision.** The crop is not a typed box in sensor coordinates. `perception/table_frame.py` finds the table:
- **Up** is the L2's own IMU: the mean accelerometer vector over the capture, recorded in every scan's
  `meta.lidar.imu_accel_mean`. The L2's cloud shares its IMU's axes (verified in earlier bench work). Live
  reading on the bench: (0.21, 0.03, 10.22) with the L2 dome-up, so about 1.2° of tilt. Its magnitude reads as m/s²,
  although the driver manifest declares the unit as g; only the direction is used here.
- **The table** is the highest horizontal surface below the lidar of at least 0.4 × the configured diorama area
  (`table_fit.expected_size_m`, Ron: 0.6 × 1.0 m). A plane fit refines "up".
- **The rectangle** is `cv2.minAreaRect` of that surface, grown to the configured size away from the lidar. The far
  end is the one the shelves hide.
- **Table frame:** origin at the tabletop centre, z up, y along the length away from the lidar, so the shelves sit
  at +y, as in `synthetic.py`.
- **Calibration file:** `config/table_frame.yaml` holds the result. It is written by `tools/calibrate_table.py` or
  the dashboard's **Calibrate table** button (`POST /calibrate/table`).
- **After calibrating:** captures store points in the table frame (`meta.frame = "table"`, `meta.sensor_pose` =
  the lidar's pose in it, `meta.table_calib_id`), and the ROI is the calibrated rectangle plus 3 cm, from the
  tabletop to 0.45 m above it. A rescan against a baseline from a different calibration is refused ("set a new
  baseline").
- **Before any calibration:** behaviour is unchanged (sensor frame, the provisional 3 m box).

**Why.** It is the foundation-friendly choice Ron asked for: the pipeline only ever sees "points in a frame + the
sensor's pose in it". For the fixed rig, that pose comes from this fit. For a moving drone it comes from the map.
Stationary is a pose that never changes, not a code path. No angle or offset is typed by hand, and any mount works
(dome down, tilted, any heading).

**Evidence (synthetic, `tests/test_table_frame.py`, 7/7).**
- **Pose recovery:** the diorama sits on a finite table with a floor, seen from three mounts (dome down; dome down
  with a 37° heading and 12° tilt; tilted 130° with a −120° heading), with a 1° IMU error. The lidar pose is
  recovered to within 7.5–12 mm and 0.3–0.4°, and the table measures 0.61 × 1.01 m.
- **End to end:** a calibrated scan pair with the debris dropped gives exactly one "added" change, at the debris.
  The floor is cut out.
- **Refusal:** a dome-up L2 level with the table is refused with the reason (it sees ~6° below level at most).

**Not yet verified on the real L2.** It was dome-up during bring-up, and its link dropped before the live test.

**Physical consequence.** The L2 must look down on the diorama: dome down on an arm over the near end (opposite
the shelves), ~0.5–0.8 m above the tabletop. Mounted dome-up on the table,
it cannot see the tabletop at all.

## DH-24 — Coloured point cloud: camera calibration from four clicked table corners

**Ron, 2026-10-17:** "lets make the point map colored btw. since it has the camera".

**Decision.**
- **Colouring** (`perception/colorize.py`): every display point is projected into its own scan's camera frame and
  takes that pixel's colour. A coarse z-buffer (6 px blocks; visible = within 2 cm + 2 % of the nearest depth)
  leaves points the camera cannot see uncoloured, so an object's colour never bleeds onto the table behind it.
- **Storage:** colours are stored in `display.npz` (`rgb`, `seen`). The cloud endpoint gains a 7th column (packed
  `r*65536+g*256+b`, exact in float32; −1 = not seen).
- **3D view:** a **Colour / Height** toggle. Changed points keep their status colours.
- **Camera calibration** (`perception/camera_calib.py`, dashboard **Calibrate camera**): the lidar already knows the
  tabletop rectangle in 3D (DH-23), so Ron clicks its four corners in the latest baseline's camera frame, in any
  order. A single-view planar calibration gives the camera pose, and the focal length too, seeded from the 102°
  diagonal listing figure and accepted within ×0.75–1.33.
- **Which corner is which:** every corner assignment is tried. The camera must be above the table and within
  0.6 m of the lidar (same bracket), which breaks the rectangle's symmetry.
- **Calibration file:** `config/camera_calibration.yaml` stores `T_cam_from_lidar`. It is relative to the lidar,
  not the table, so a table recalibration keeps it valid while both stay on one bracket.
- **Uses of the calibration:** the per-job camera model (crops, appearance-change placement) now comes from it. On
  saving, every scan of the current table calibration is recoloured.

**Not modelled.** Lens distortion (the 102° lens has some barrel distortion). The `dist` field is already read
everywhere, so a checkerboard can add it later.

**Evidence (synthetic, `tests/test_camera_color.py`, 6/6).**
- **Pose recovery:** a tilted camera beside a dome-down lidar, 1.5 px click noise, corners clicked in random order.
  The pose is recovered to within 1.6 cm and 0.24°.
- **Focal fit:** a 92° FOV guess is corrected (f 662 against the true 648 px).
- **Tank:** the top takes the tank's colour, and the hidden table behind it does not.
- **Table recalibration:** the camera follows the lidar through a table recalibration.
- **Dashboard (synthetic server):** 92 % of the display points are coloured.

**Not yet on the real camera.** The rig has no mount yet.

## DH-25 — Hand crop in the dashboard; a floor-size calibration is allowed; the lidar shown as a green dot

**Ron, 2026-10-17:** "it should allow a floor size scan but i can just crop it to be what i want", and "i would
also like a green dot indicating where the actual lidar is".

**Decision.**
- **Floor-size calibration allowed.** A table fit larger than the diorama (a bigger table, or the floor when no
  table is in view) is accepted again, with a warning that points to Crop. Any level surface gives a valid frame.
  9dcc765's refusal is reverted.
- **The Crop tool.** The dashboard's **Crop** opens the newest scan in the current frame, UNCROPPED, seen from above
  (`GET /crop/preview`: points within 4 m of the lidar).
  - Drag anywhere to draw a box. The centre square moves it, an edge resizes it. Scroll zooms, right-drag pans.
  - The bottom and top heights fill in from the points inside the box (2nd percentile − 3 cm to 99.5th percentile
    + 5 cm) and are editable. Points inside the height band are bright, the rest dim.
- **Saving.** `POST /crop` writes `config/crop.yaml` (box + the `frame_id` it was drawn in: the table calibration,
  or "sensor"). `table_frame.apply` uses it as the ROI only in that same frame (`crop_source`: hand / table /
  config), so a recalibration never applies an old box in a new frame; the editor says so.
- **Reset.** `DELETE /crop` moves the file to `crop.yaml.reset` (kept, never silently lost).
- **On a crop change,** the baseline's 3D view is re-cropped. Rescans are cropped at comparison time anyway, so no
  new baseline is needed.
- **The lidar's position** (the scan's `sensor_pose` translation, in the points' frame) is sent as `X-Lidar`, on
  the cloud and the preview. It is drawn as a green dot with a halo in the 3D view and labelled LIDAR in the crop
  editor. Green is used only for this marker.

**Evidence.**
- `tests/test_table_frame.py` 10/10: the floor alone gives a level frame plus a Crop warning; the hand crop applies
  only in its own frame; an inverted box is refused.
- **Live, on Ron's floor-frame scan:** the editor showed the room from above with the green lidar dot. A drawn
  1.28 × 1.66 m box got heights −0.03…1.07 m.
- **API round trip:** save → `source: hand`; an inverted box → 400; reset → back to the table rectangle. Nothing
  was left in use.

## DH-26 — Camera calibration by clicked point pairs (works with a floor-size table fit)

**Why.** Ron's rig, 2026-10-17: the table fit found the floor (3.5 × 3.7 m), and the camera looks down at a can on
the floor, mounted upside down. DH-24's "click the four tabletop corners" has no corners to click there. So the scan
had no colour, and no calibration was possible.

**Decision.**
- **Point pairs** (`camera_calib.solve_from_pairs`, the dashboard's Calibrate camera, now offered for any live
  setup): Ron clicks a spot in the camera picture, then the same spot in the lidar's top-down view of THAT scan
  (`/crop/preview?scan_id=`). The lidar click snaps to the highest point within 8 px, so tops and corners of things
  pick well.
- **Pairs and outliers:** at least 6 pairs. One clearly mismatched pair (dropping it more than halves the error) is
  left out, and the dialog names it.
- **Focal length:** the lens's listed figure (102° diagonal) unless fitting the focal cuts the pair error below
  0.9× (`FIT_GAIN`, chosen by the sweep below). With a shallow scene, focal and distance trade off: a freely fitted
  focal can lower the click error while moving the camera ~12 cm.
- **Corner mode:** kept, and offered only when the fitted surface is the diorama (within 1.25× of the configured
  size). The colouring is unchanged (DH-24).

**Accuracy, measured as what colour depends on** (where the scene's points land in the image, fitted against true
camera; p90 over diorama points; synthetic, 1 cm picking error in 3D and 2 px in the image, 1280 × 960):

| Pairs | p90, median over 8 click sets | Worst |
|---|---|---|
| 6 | 21 px | 38 px |
| 8 | 15 px | 25 px |
| 12 | 15 px | 23 px |

That is about 2 cm on the scene at 8 or more pairs, and about 6 px with 12 pairs when the lens figure is wrong (a
90° guess, the focal fitted).

**Why the accuracy is measured this way.** Pose error is the wrong yardstick here: with spots ~0.2 m apart, the true
pose itself misses the noisy clicks by up to 19 px. `tests/test_camera_color.py` 10/10.

**Not yet done:** calibrated on the real camera (Ron's next step).

## DH-27 — Perceptron classifies each change from a closed list (the §6.8 primary, no longer a stub)

Ron (2026-10-17): send Perceptron the before and after and have it diagnose the change from a few choices (rust,
corrosion, leakage, object moved, new object, ...).
- **API, from docs.perceptron.inc:** `POST https://api.perceptron.inc/v1/chat/completions` (OpenAI chat-completions
  format), `Authorization: Bearer`. Models on Ron's account: `perceptron-mk1` and `perceptron-mk1.5`, same price
  ($0.15 / M input, $1.50 / M output tokens); Perceptron recommends Mk1.5, so it is the default
  (`classifier.perceptron.model`). Account: "Ron's Organization", $50 promo credit; key `noxen-dh26` created
  2026-10-17.
- **Key:** `PERCEPTRON_API_KEY` in the repo-root `.env` (gitignored). The served app loads that file at start-up
  (`server.app.load_dotenv`, an already-set environment variable wins); tests never do (`NOXEN_NO_AUTOAPP=1`), so they
  stay offline.
- **One request per change:** "BEFORE:" image, "AFTER:" image (PNG crops inline as data URLs), then the prompt (the
  existing facility context + the geometry hint for the kind + the lidar measurements + the category list). The reply
  is constrained by `response_format: json_schema` (category = enum of the configured ids, severity, rationale,
  confidence) and validated again here; off-schema or truncated = ClassifierError, the chain falls through.
- **Categories are config** (`config/table_demo.yaml` `classifier.perceptron.categories`): rust, corrosion, leak or
  spill, damage, new object, object removed, object moved, no real change, other. **Routine is the config's call, not
  the model's** (object moved and no real change are routine; routine forces severity none, an issue is never none).
- **Cloudflare:** the API answers `403 / error code 1010` to Python urllib's default User-Agent; the client sends its
  own (`noxen-dh26/1.0`).
- **Checked:** Mk1.5 on the 9 real crop pairs of live scan `20261017T210822Z_41e7`: 1.2–3.4 s each, sensible labels
  (new white box with vents, small object removed, two lighting changes called lighting). Synthetic `everything`
  through the server: moved → object moved (routine), added → new object, appearance → rust, all `provider:
  perceptron`. The synthetic "rust" is a flat brown disk; on it alone the model sometimes says "new object".
  Tests: `tests/test_classify.py` (request body, reply parsing, routine from config; no network).

## DH-28 — Perceptron switch on the dashboard

Ron (2026-10-17): a Perceptron toggle; when it is on, rescans run the model.
- Stored server-side (`POST /perceptron` `{"enabled": bool}`, `/status.perceptron` = enabled / configured (key
  present) / model), read when a scan STARTS, like the text switch (DH-22). **Default off**, so no credit is spent
  until Ron turns it on.
- Off = the same chain with Perceptron replaced by a stand-in that is never called; each event's
  `classification_attempts` then starts with `perceptron: not configured, "Perceptron is switched off on the
  dashboard"`, and openai / cached / rules answer as before. On with no key shows "Perceptron · no key" with a
  dashed border.
- Test: `tests/test_server.py::test_perceptron_toggle_decides_whether_perceptron_is_asked` (offline). Checked on the
  live server after its restart: the switch flips `/status.perceptron.enabled` both ways, key configured, model
  perceptron-mk1.5; left OFF.

## DH-33 — Rotated crop box (first numbered DH-27, a number the Perceptron note also uses)

**Ron, 2026-10-17:** "make it so when i crop i can rotate it bc rn the angles are off". The table frame's axes come
from the fitted surface (or the floor), so the diorama is generally not square to them.

**Decision.**
- **The ROI has a second form:** a box turned about the vertical, `{center: [x, y], size: [w, l], yaw_deg, z:
  [lo, hi]}` (yaw counter-clockwise from +x). The axis-aligned `{min, max}` form still works everywhere: the
  automatic table rectangle, older crop files, `config/table_demo.yaml`.
- **One function decides membership:** `preprocess.roi_mask`, used by `crop_roi` and the pipeline's display mask.
  The visual-diff image mask uses the box's axis-aligned bounds (`roi_bounds`), slightly larger than the box, which
  is harmless there.
- **Crop editor:** a round knob above the box rotates it, and an **Angle (°)** field takes exact values. A newly
  drawn box keeps the last angle, and edges resize along the box's own axes. The angle is saved normalised to
  −180…180°.

**Evidence.**
- `tests/test_table_frame.py` 11/11: a 0.6 × 1.0 m tabletop turned 30° is kept whole by the rotated box and loses
  corners with the same box unrotated; yaw 390° saves as 30°.
- **Live:** the knob turned Ron's crop to −49.3° in the editor (not saved). An API save of a rotated box applied as
  `source: hand` and re-cropped the baseline view, then Ron's exact crop was restored.

## DH-29 — Camera point pairs picked in 3D, with a live fit preview on the photo

**Why.** Ron's first point-pair calibration (2026-10-17) came out poorly: 64 px average miss, one pair 103 px off,
and the camera placed 51 cm from the lidar. "The color still does not work that well. maybe when i am selecting
the points to match in the lidar, make it 3d?" The top-down picker snaps to the HIGHEST point near the click, so a
spot seen from the side in the photo (a chair leg, the side of the can) got the top of whatever was above it.

**Decision.**
- **3D picker** (`LidarPicker3D.jsx`): the lidar pane is a three.js view that opens FROM the lidar's own position,
  looking at the crop's centre, with a ~72° vertical field of view like the DECXIN, so it looks like the photo.
  - Drag orbits, scroll zooms, right-drag pans.
  - A click (not a drag) picks the point you are LOOKING AT: the nearest point along the ray, among points within
    a small, distance-scaled radius of it.
  - A hover dot shows the pick before you click. The lidar dot is hidden while viewing from the lidar itself.
- **Live fit preview:** `POST /calibrate/camera/preview` solves without saving. From 6 complete pairs on, the
  dialog shows the fit:
  - the lidar points drawn on the photo with it (near = orange, far = blue);
  - a white line from each clicked pixel to where its 3D point lands;
  - each pair's miss in px, with pairs over 25 px in red;
  - × on a pair removes just that pair.
  Save stays the same call as before.
- **Dialogs are top-anchored, no longer vertically centred,** so a growing dialog does not move what is under the
  cursor between clicks.
- `topview.js` is removed (only the old top-down picker used it).

**Evidence.**
- **Plumbing:** 8 pairs synthesised from the saved calibration and the real scan → preview 0.0 px, camera 0.51 m,
  calibration file untouched.
- **Live:** the 3D view opened from the lidar and looked like the photo; a click placed pair 1's marker on the
  cloud (test pair removed, nothing saved).
- `test_server` 4/4, `test_camera_color` 10/10.

**Not yet:** Ron recalibrating with it.

## DH-30 — Auto-align the camera from the scan; looser colouring; the crop survives a table recalibration

**Ron, 2026-10-18:** "color is still cooked and the floor does not show up in the browser view points", then "can
you just calibrate it for me? you have the scan and the image".

**What was wrong (measured on Ron's scan 20261018T000057Z_44e9).**
- **Calibration:** Ron's 3D point-pair calibration was already fair (16 px pair rms, camera 10.5 cm from the lidar).
- **Colouring:** the occlusion test was too strict. Only 71 % of the points in the camera's view got a colour, so
  the view looked speckled. Half of the cloud within 3.5 m is outside the camera's field of view at all, and
  cannot be coloured.
- **The floor was missing for another reason.** Ron drew his crop at 23:47 and recalibrated the table at 23:48.
  The crop was tied to the old frame, so it was set aside silently, and the automatic band (−2…+45 cm around the
  fitted surface) cut the floor away.

**Decision.**
- **Auto-align** (`perception/camera_autocal.py`, `POST /calibrate/camera/auto`, the dialog's Auto-align button)
  refines the current calibration with no clicks:
  - lidar silhouette points (depth jumps seen from the camera) are matched to the photo's Canny edges through a
    clipped distance map;
  - Powell search over the camera pose (and focal), coarse to fine (clip 40 → 15 → 6 px), with silhouettes
    recomputed between passes;
  - refused if it wanders (> 15 cm, > 8°, focal outside ×0.8–1.25, camera > 0.6 m from the lidar).
- **On Ron's scan:** the mean edge distance went from 6.0 to 5.1 px (moved 8 mm, turned 2.0°, focal unchanged).
  The before/after overlays show the chair-back top, the can and the leg outline settling onto their edges. It was
  applied and saved: 2 scans recoloured. Ron's click calibration is backed up in the session scratchpad, not in
  the repo.
- **Colouring occlusion** loosened (4 px blocks, 4 cm + 4 %): 93 % of the points in view are coloured, against
  71 %. `test_camera_color` 10/10, the no-bleed test included.
- **The crop carries across a table recalibration** when the lidar did not move (`table_frame.lidar_unmoved`:
  same IMU tilt within 0.5° AND at least 50 % of the scan within 5 cm of the previous calibration scan's points).
  - Measured on Ron's rig: untouched 68 %, remounted 20 %.
  - The box is re-expressed through the lidar (`carry_crop`), and the dashboard says whether the crop was kept or
    must be redrawn.
- **Ron's crop:** carried from table-…232211 to table-…234757. The old frame was recovered from the `sensor_pose`
  of the scan it was drawn on. An ICP between the two calibration scans was inconclusive (structure fitness 0.19;
  it suggested up to 6.5 cm). The box has wide margins, and Ron was told to check it.

## DH-31 — The bottle: what was off, and the lidar edge-smear filter for the 3D view

**Ron, 2026-10-18:** "the bottle on the ground still looks wrong color i think we need to calibrate it a bit more".

**Measured on scan 20261018T000057Z_44e9.**
- **The bottle** is a clear bottle (~0.20 m tall, ~6 cm wide) with a dark cap.
- **Where its lidar points land:** projected into the photo, they cover the side facing the lidar (the lidar sits
  ~10 cm right of the camera), so they land on the bottle's right half. That is correct geometry, not an offset.
- **The fringe:** a band of points extends ~10 px past the bottle's right edge. These are mixed-pixel "veil"
  returns (one beam partly on the bottle, partly on the floor behind it) and took the floor's colour.
- **More calibration does not help:**
  - **Lens distortion** (k1, k2 in auto-align): the edge fit got worse (5.1 → 7.0 px) and the terms stayed ~0.
    Kept as an option, off by default.
  - **Crop-only alignment:** fixed the bottle horizontally but moved the camera 7 cm (under-constrained).
  - **Crop-weighted alignment** (×4 / ×8): the bottle offset is unchanged at weight 4, and weight 8 moved the
    camera 12 cm.
  - **The current calibration was kept.** The remaining misalignment at the bottle is ~5–10 px (~1 cm).

**Decision.**
- **Veil filter** (`preprocess.veil_mask`): the 3D view drops points whose surface lies within ~11° of edge-on to
  the lidar beam (|cos| < 0.2 between the beam and the local normal, k = 16).
  - **What it never drops:** points of a change or ghost points.
  - **Where it applies:** the 3D view only, every `_save_display`. Change detection still sees every point.
- **Measured effect:**
  - on the bottle, 25 % of its points are flagged, nearly all in the fringe;
  - synthetic test: 95 % of a smear strip, 0 % of the floor or a face-on box.
- **Limit:** the floor beyond ~3.9 m from a lidar 0.8 m up is grazing enough to be flagged. That is outside the
  crop on this rig.
- **Auto-align options added:** `refine(focus=…, focus_weight=…, fit_distortion=…)`. The endpoint keeps the
  defaults (whole picture, no lens terms) and passes any saved lens terms through.
- **Advice to Ron:** clear and shiny objects are hard for any lidar. For the demo, use matte, opaque props (the
  tank, crate and shelves are fine).

## DH-32 — End-to-end rehearsal with the lidar off (replay); a fixed rig keeps its known pose when ICP cannot confirm

**Ron, 2026-10-18:** "make sure everything is integrated: when i rescan, it flags differences, calls the perceptron
api and if it is an issue it calls photon". The L2 was unpowered, so: "can you just use an old scan".

**Decision.**
- **Replay** (`POST /scans/trigger {"replay": "<scan id>"}`, `sources.replay_scan`): an earlier scan folder is
  re-run as tonight's scan. It gets a new id and is marked `source: replay:<id>` and `replay_of`. Everything after
  capture runs for real: compare, classify, alert.
- **Fixed-rig fallback** (`registration.fixed_rig_known_pose`, on for the table rig): when ICP cannot confirm an
  alignment, the rig keeps its known pose (identity), but only if:
  - both scans are under the same table calibration, and
  - the lidar is verifiably unmoved (`table_frame.lidar_unmoved`: IMU tilt within 0.5°, at least 50 % of the scene
    within 5 cm).
  The registration record says so. A moved lidar is still refused.
- **Why it was needed:** the first replay was refused (structure fitness 0.689). Ron's crop is mostly bare floor
  plus a bottle, and people moved between the scans, so ICP slid ("found" 78 mm / 3°). Yet the lidar had not
  moved: tilt 0.11°, 74 % of the scene lines up, identical sensor poses.
- **Tests** (`tests/test_pipeline.py`): bare floor with an object that moves between scans → refused with the
  fallback off, kept and flagged with it on; a tilted (moved) lidar is still refused.

**The rehearsal** (scan 20261017T234829Z_03a6 against baseline 20261018T000057Z_44e9, Perceptron ON, texts ON):
- 18 changes. Perceptron answered all 18 (~1 s each, no fallbacks): 9 issues (new object / object removed, low
  severity) and 9 routine (moved / no real change).
- The alert bridge queued all 9 issues as flags #1–#9 at 01:44:45Z. Delivery is confirmed only by Ron's phone;
  the bridge logs send errors on its own console. **Ron confirmed all 9 texts arrived (2026-10-18).**
- Both switches were turned back OFF afterwards.

**Also:** my rotated-crop decision is renumbered DH-33. It was first numbered DH-27, the same number as the other
session's Perceptron note.

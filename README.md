# Noxen @ hackathon '26

Two demos on one codebase (see `ARCHITECTURE.md`):

- **A. Table demo (live):** a fixed Unitree L2 + camera over a 1:10 diorama. Rescan → changes
  detected → classified → pins on a dashboard. Routine changes stay quiet; real issues get flagged.
- **B. Flight demo (video):** the Scout localizes with its camera in a prior lidar map, flies A→B,
  and replans around a person it sees with the same camera.

Judgment calls are in `docs/DECISIONS.md`; things only Ron can do or answer are in `docs/NEEDS_RON.md`;
one-time setup steps (laptop Ethernet, Radxa SD card) are in `docs/SETUP.md`.

## Status

| Milestone | State |
|---|---|
| 0 — scaffold, driver streaming, scan capture | Code done. Lidar half proven on **recorded real L2 data** (replayed over UDP). Live L2 + table camera not yet run. |
| 1 — preprocess, ICP, geometric diff | Code done, 10/10 tests on SYNTHETIC ray-cast scans (DH-12..14). Gate is on recorded real scans (Saturday). |
| 2 — move association + visual diff | Code done: association 8/8, visual diff 7/7 on synthetic data (DH-15, DH-16). Gate on recorded scans. |
| 3 — classifier adapter + server | Code done: pipeline 6/6, classifier chain 6/6, server 2/2 (synthetic source). Perceptron = stub until the event (DH-18). |
| 4 — dashboard | Code done: React + Vite + three.js (`web/`), noxensys.com theme, 3D lidar view; served by the FastAPI server; checked in the browser on synthetic scans of Ron's diorama layout (DH-20, DH-21). Live hardware run pending. |
| Scout (Demo B) | Stability: S1–S5 flight-tested on the Scout FC (`hardware/scout/`). Camera relocalization: synthetic-tested, 13/13 (`docs/SCOUT_LOC.md`); needs the real prior map, camera calibration and the Radxa–FC link. |

## Where things run

- **Demo A runs in Windows Python** (DH-03): the USB camera is only visible there, and it keeps DDS
  and WSL networking off the live demo. Python 3.9 today (numpy, OpenCV, PyYAML, SciPy installed).
- **Demo B runs in WSL** (Ubuntu 24.04, ROS 2 Jazzy): Point-LIO, the Scout nodes, the PX4 link.

## Connecting the L2 (Windows)

The L2 sends to `192.168.1.2:6201` by default (DH-04), so the laptop takes that address and the
sensor is never reconfigured:

1. Plug the L2 into the laptop's Ethernet port.
2. Set that adapter to static IPv4 `192.168.1.2`, mask `255.255.255.0`, no gateway.
3. Allow inbound UDP 6201 for Python in Windows Firewall (Windows asks the first time Python binds).

Never run Unitree's `unilidar_sdk2` against the sensor: it sets the sensor clock on start-up (DH-02).

## Commands

Capture one scan (lidar + camera) into `data/scans/<scan_id>/`:

```bash
py -3.9 -m capture.scan_capture --config config/table_demo.yaml
```

Add `--baseline` to mark it as the baseline, `--no-camera` for lidar only (the scan then says
`camera: not_run`; it never gets a placeholder image).

Check a scan folder (Milestone 0 gate; prints the numbers, exits 0 / 1 / 2):

```bash
py -3.9 tools/check_scan.py data/scans/<scan_id>
```

Replay a recorded L2 capture into the capture path without the sensor: start the capture
command above first (it waits 3 s for data), then in a second terminal:

```bash
py -3.9 drivers/unitree_l2/bench/l2_pcap.py replay drivers/unitree_l2/test_fixtures/l2_real_wide_5s.pcap --to 127.0.0.1:6201
```

## Dashboard

```bash
cd web && npm install && npm run build
```

The server then serves it at http://localhost:8000. For UI work, `npm run dev` in `web/` (Vite proxies the API to :8000).

## Running the server

```bash
py -3.9 -m uvicorn server.app:app --port 8000
```

Without hardware, run the synthetic source (Ron's diorama; scenarios: no_change, crate_moved, rust_tank, dropped_in_aisle, everything):

```bash
NOXEN_SOURCE=synthetic py -3.9 -m uvicorn server.app:app --port 8000
```

Then `POST /baseline`, then `POST /scans/trigger` with `{"scenario": "everything"}`; watch `ws://localhost:8000/live`.
Classifier keys come from the environment (`PERCEPTRON_API_KEY`, `OPENAI_API_KEY` + `OPENAI_MODEL`), never from files in git;
the server also reads them from a gitignored `.env` at the repo root. Perceptron (Mk1.5) picks one category per change
from `classifier.perceptron.categories` in `config/table_demo.yaml` (DH-27). It only runs while the dashboard's **Perceptron**
switch is on (off by default, DH-28); otherwise the offline fallbacks classify.

## iMessage alerts (Photon)

`alerts/` texts every new flagged (non-routine) change to Ron's iPhone with its before/after crops, and acts on
the reply (DH-22). It runs beside the server, as a Node process (Photon's SDK, `spectrum-ts`, is TypeScript-only):

```bash
cd alerts && npm install && npm start
```

`alerts/.env` (gitignored; template `alerts/.env.example`) holds the Photon project id + secret and `NOXEN_ALERT_TO`.
Texts go out only for scans run while the dashboard's **Text alerts** switch is on (next to Set baseline; it
reads "bridge off" with a dashed border when the switch is on but the bridge isn't running). `npm start -- --hello`
sends one test message. Replies: **1** = send the Scout (`POST /scout/dispatch`), **2** = real
issue (feedback `confirmed`), **3** = false alarm (feedback `false_alarm`); add `#N` for an older flag, `list` for the
open ones. Free plan = Photon's shared line: the recipient must be a user of the project, and must text the line
once before it can message them (no cold outreach on the shared pool).

## Tests

No pytest needed; each file also runs under pytest.

```bash
py -3.9 tests/test_io.py
py -3.9 tests/test_capture_replay.py
py -3.9 tests/test_geo_diff.py
py -3.9 tests/test_associate.py
py -3.9 tests/test_visual_diff.py
py -3.9 tests/test_pipeline.py
py -3.9 tests/test_classify.py
py -3.9 tests/test_server.py
py -3.9 tests/test_scout_loc.py
py -3.9 drivers/unitree_l2/test_l2_protocol.py
cd alerts && npm test
```

## Layout

```
capture/            Demo A capture: L2 accumulation + camera grab → scan folder (no ROS, DH-03)
drivers/unitree_l2/ L2 wire protocol (approved reuse, DH-02) + UDP receiver + real-sensor fixtures
perception/         pure Python change-detection pipeline (no ROS); io.py = scan folders
config/             table_demo.yaml, flight_demo.yaml, scout_hw.yaml (Scout hardware facts)
tools/              check_scan.py; calibration / top-down / landmark tools later
tests/              offline tests
ros2_ws/src/        Scout nodes (Milestone 5)
server/ web/        Milestones 3–4
alerts/             Photon iMessage alerts + reply handling (Node, DH-22)
data/               scans and recordings (gitignored)
```

"""Demo A server (ARCHITECTURE §7.1): FastAPI + SQLite + WebSocket progress.

    py -3.9 -m uvicorn server.app:app --port 8000          (config: env NOXEN_CONFIG, default config/table_demo.yaml)

Scans run one at a time on a worker thread; the API stays responsive and every stage is pushed to `WS /live`
(scan_started, scan_progress, event_created, scan_done, scan_error, baseline_set) so judges never stare at a frozen
screen. `source: synthetic` in the config (or NOXEN_SOURCE=synthetic) runs the whole loop without hardware.
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import os
import sys
import threading
import time

import yaml
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from perception import io as scan_io  # noqa: E402
from perception import topdown  # noqa: E402
from perception.classify.chain import default_chain, with_provider_off  # noqa: E402
from perception.pipeline import display_cloud, run_scan_comparison  # noqa: E402
import numpy as np  # noqa: E402
from perception.preprocess import crop_roi, preprocess  # noqa: E402
from server import sources  # noqa: E402
from perception import table_frame  # noqa: E402
from perception import camera_calib  # noqa: E402
from perception.colorize import colorize  # noqa: E402
from perception.preprocess import veil_mask  # noqa: E402
from server.storage import Store  # noqa: E402

FEEDBACK = ("none", "false_alarm", "confirmed")


class Hub:
    """Fan-out of JSON messages to every connected dashboard; safe to call from the worker thread."""

    def __init__(self):
        self.clients: set = set()
        self.bridges: set = set()                      # sockets of the iMessage alert bridge (alerts/, DH-22)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.history: list = []

    def publish(self, msg: dict) -> None:
        msg = dict(msg, t=round(time.time(), 3))
        self.history = (self.history + [msg])[-200:]
        if self.loop is None:
            return
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(self._send(ws, msg), self.loop)

    async def _send(self, ws, msg):
        try:
            await ws.send_json(msg)
        except Exception:
            self.clients.discard(ws)


def _table_summary(doc: dict) -> dict:
    q = doc.get("quality") or {}
    return {"calib_id": doc.get("calib_id"), "fitted_at": doc.get("fitted_at"), "size_m": doc.get("size_m"),
            "measured_size_m": doc.get("measured_size_m"), "lidar_height_m": q.get("lidar_height_above_table_m"),
            "plane_rms_m": q.get("plane_rms_m"), "warnings": doc.get("warnings") or []}


def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("NOXEN_CONFIG", os.path.join(ROOT, "config", "table_demo.yaml"))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if os.environ.get("NOXEN_SOURCE"):
        cfg["source"] = os.environ["NOXEN_SOURCE"]
    return cfg


def create_app(cfg: dict | None = None, data_dir: str | None = None) -> FastAPI:
    cfg = dict(cfg or load_config())
    source = cfg.get("source", "live")
    data_dir = data_dir or os.path.join(ROOT, "data")
    scans_dir = os.path.join(data_dir, "scans_synthetic" if source == "synthetic" else "scans")
    os.makedirs(scans_dir, exist_ok=True)
    if source == "synthetic":
        cfg["roi"] = sources.SYNTH_ROI
        camera = sources.synthetic_camera()
    else:
        camera = None                                  # live: per job, from config/camera_calibration.yaml (DH-24)
    cam_file = os.path.join(ROOT, cfg.get("camera_calibration_file", camera_calib.FILE_DEFAULT))

    store = Store(os.path.join(data_dir, f"noxen_{source}.sqlite"))
    hub = Hub()
    chain = default_chain(cfg, os.path.join(data_dir, "classify_cache.json"))
    chain_perc_off = with_provider_off(chain, "perceptron", "Perceptron is switched off on the dashboard")   # DH-28
    perc = next((p for p in chain.providers if p.name == "perceptron"), None)
    worker = cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="scan_worker")
    busy = threading.Lock()
    seed = [int(time.time()) % 100000]

    app = FastAPI(title="Noxen table demo")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.store, app.state.hub, app.state.cfg = store, hub, cfg

    @app.on_event("startup")
    async def _startup():
        hub.loop = asyncio.get_running_loop()

    def prog(stage, frac=None):
        detail = round(float(frac), 2) if isinstance(frac, (int, float)) else frac
        msg = {"type": "scan_progress", "stage": stage, "detail": detail}
        if stage == "accumulating" and source != "synthetic" and isinstance(frac, (int, float)):
            msg["remaining_s"] = round(max(0.0, 1.0 - float(frac)) * float(cfg["accumulation_s"]), 1)
        hub.publish(msg)

    def calibrate_job():
        """Capture lidar-only in the SENSOR frame, find the table, write config/table_frame.yaml (DH-23)."""
        try:
            hub.publish({"type": "scan_started", "kind": "calibration"})
            raw = dict(cfg, frame="sensor", sensor_pose=None, table_calib_id=None)
            from capture.scan_capture import capture_scan
            path = capture_scan(raw, is_baseline=False, use_camera=False, progress=prog,
                                scans_dir=os.path.join(data_dir, "calibration"))
            prog("fitting table")
            old_table = table_frame.load(os.path.join(ROOT, cfg.get("table_frame_file", table_frame.FILE_DEFAULT)))
            new_scan = scan_io.load_scan(path, load_image=False)
            fit, doc = table_frame.calibrate_scan(new_scan, cfg, ROOT)
            if not fit.ok:
                hub.publish({"type": "scan_error", "error": f"table not found: {fit.reason}"})
                return
            summary = _table_summary(doc)
            summary["crop"] = _carry_crop(old_table, doc, new_scan)
            hub.publish({"type": "table_calibrated", "table": summary})
        except Exception as e:
            hub.publish({"type": "scan_error", "error": f"{type(e).__name__}: {e}"})
        finally:
            busy.release()

    def _carry_crop(old_table: dict | None, new_table: dict, new_scan) -> str:
        """After a table recalibration: keep Ron's hand crop if the lidar did not move (re-expressed in the new
        frame), else say it was set aside. Returns a sentence for the dashboard (DH-30)."""
        crop = table_frame.load_crop(cfg, ROOT)
        if not crop:
            return ""
        if not old_table or crop.get("frame_id") != old_table.get("calib_id"):
            return "Your hand crop belongs to an older calibration and is not in use: redraw it with Crop."
        old_scan_dir = os.path.join(data_dir, "calibration", old_table.get("from_scan", ""))
        if not os.path.isdir(old_scan_dir):
            return "Your hand crop could not be carried over (the previous calibration scan is missing): redraw it."
        still, why = table_frame.lidar_unmoved(scan_io.load_scan(old_scan_dir, load_image=False), new_scan)
        if not still:
            return f"The lidar moved ({why}), so your hand crop no longer fits: redraw it with Crop."
        box = table_frame.carry_crop(crop, old_table, new_table)
        if box is None:
            return "Your hand crop could not be carried over (the new frame is tilted against the old one): redraw it."
        table_frame.save_crop(cfg, ROOT, box, frame=new_table["calib_id"], from_scan=crop.get("from_scan"))
        return "Your hand crop was kept (the lidar did not move)."

    def job_camera(jcfg: dict) -> dict | None:
        """The camera model for scans in this job's frame: the synthetic camera, or the clicked-corner calibration
        composed with the CURRENT lidar pose (it is stored relative to the lidar, DH-24)."""
        if source == "synthetic":
            return camera
        return camera_calib.camera_for_frame(camera_calib.load(cam_file), jcfg.get("sensor_pose"))

    def job_cfg() -> dict:
        """The config for one live job, with the CURRENT table calibration applied (DH-23): a new calibration takes
        effect on the next capture, without a restart."""
        return cfg if source == "synthetic" else table_frame.apply(cfg, ROOT)

    def capture(jcfg: dict, is_baseline: bool, scenario: str | None, progress, replay: str | None = None) -> str:
        if replay:
            return sources.replay_scan(store.scan(replay)["path"], scans_dir, is_baseline=is_baseline, progress=progress)
        if source == "synthetic":
            seed[0] += 1
            return sources.capture_synthetic(scans_dir, scenario or ("baseline" if is_baseline else "no_change"),
                                             is_baseline=is_baseline, seed=seed[0], progress=progress)
        return sources.capture_live(jcfg, scans_dir, is_baseline=is_baseline, progress=progress)

    def job(is_baseline: bool, scenario: str | None, replay: str | None = None):
        kind = "baseline" if is_baseline else "scan"
        cfg = job_cfg()                                # shadows the app config for the rest of this job
        cam = job_camera(cfg)
        text_on = bool(store.get_state("text_alerts"))  # the text toggle as it was when this scan was started (DH-22)
        classifier = chain if store.get_state("perceptron_enabled") else chain_perc_off   # the Perceptron toggle (DH-28)
        try:
            hub.publish({"type": "scan_started", "kind": kind})

            path = capture(cfg, is_baseline, scenario, prog, replay)
            scan = scan_io.load_scan(path)
            baseline_id = store.get_state("baseline_id")
            store.add_scan(scan.scan_id, path=path, is_baseline=is_baseline, baseline_id=None if is_baseline else baseline_id,
                           source=scan.meta.get("source", source), n_points=len(scan.xyz))
            if is_baseline:
                import cv2
                img, meta = topdown.topdown(crop_roi(scan.xyz, scan.intensity, cfg.get("roi"))[0])
                cv2.imwrite(os.path.join(path, "topdown.png"), img)
                with open(os.path.join(path, "topdown.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f)
                _save_display(path, display_cloud(*preprocess(scan.xyz, scan.intensity, cfg), cfg), scan.image, cam,
                              scan.sensor_pose[:3, 3])
                store.set_state("baseline_id", scan.scan_id)
                store.finish_scan(scan.scan_id, status="baseline")
                hub.publish({"type": "baseline_set", "scan_id": scan.scan_id})
                return
            if not baseline_id:
                store.finish_scan(scan.scan_id, status="error", error="no baseline yet: capture one first")
                hub.publish({"type": "scan_error", "scan_id": scan.scan_id, "error": "no baseline yet: capture one first"})
                return
            base = scan_io.load_scan(store.scan(baseline_id)["path"])
            if base.meta.get("table_calib_id") != scan.meta.get("table_calib_id"):
                err = "the table was calibrated after this baseline: set a new baseline"
                store.finish_scan(scan.scan_id, status="error", error=err)
                hub.publish({"type": "scan_error", "scan_id": scan.scan_id, "error": err})
                return
            res = run_scan_comparison(base, scan, cfg, crops_dir=os.path.join(path, "crops"), camera=cam,
                                      classifier=classifier, progress=lambda stage, detail=None: hub.publish(
                                          {"type": "scan_progress", "stage": stage, "detail": detail}))
            if res.error:
                store.finish_scan(scan.scan_id, status="error", error=res.error, registration=res.registration, stats=res.stats)
                hub.publish({"type": "scan_error", "scan_id": scan.scan_id, "error": res.error})
                return
            if res.display is not None:
                _save_display(path, res.display, scan.image, cam, scan.sensor_pose[:3, 3])
            store.add_events(res.events)
            store.finish_scan(scan.scan_id, status="done", registration=res.registration, stats=res.stats)
            for ev in res.events:
                hub.publish({"type": "event_created", "event": ev, "text_alert": text_on})
            hub.publish({"type": "scan_done", "scan_id": scan.scan_id, "n_events": len(res.events),
                         "seconds": res.stats.get("seconds")})
        except Exception as e:                         # never leave the UI waiting: report and free the worker
            hub.publish({"type": "scan_error", "error": f"{type(e).__name__}: {e}"})
        finally:
            busy.release()

    def start(is_baseline: bool, scenario: str | None, replay: str | None = None):
        if replay and not store.scan(replay):
            raise HTTPException(404, f"no scan {replay} to replay")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "a scan is already running")
        worker.submit(job, is_baseline, scenario, replay)
        return {"accepted": True, "kind": "baseline" if is_baseline else "scan", "source": source, "scenario": scenario,
                "replay": replay}

    @app.get("/status")
    def status():
        doc = None if source == "synthetic" else table_frame.load(
            os.path.join(ROOT, cfg.get("table_frame_file", table_frame.FILE_DEFAULT)))
        return {"source": source, "baseline_id": store.get_state("baseline_id"), "busy": busy.locked(),
                "scenarios": list(sources.syn.SCENARIOS) if source == "synthetic" else None,
                "providers": [p.name for p in chain.providers],
                "calibrated": camera is not None or os.path.exists(cam_file),
                "table": _table_summary(doc) if doc else None, **alerts_state(), **perceptron_state()}

    def perceptron_state() -> dict:
        return {"perceptron": {"enabled": bool(store.get_state("perceptron_enabled")),
                               "configured": bool(perc is not None and getattr(perc, "api_key", None)),
                               "model": getattr(perc, "model", None)}}

    @app.post("/perceptron")
    def set_perceptron(body: dict = Body(...)):
        """The dashboard's Perceptron toggle (DH-28): while on, rescans classify with Perceptron; off = the fallbacks."""
        if not isinstance(body.get("enabled"), bool):
            raise HTTPException(400, "body must be {\"enabled\": true|false}")
        store.set_state("perceptron_enabled", body["enabled"])
        hub.publish({"type": "perceptron_changed", **perceptron_state()})
        return perceptron_state()

    def alerts_state() -> dict:
        return {"text_alerts": bool(store.get_state("text_alerts")), "text_bridge": len(hub.bridges) > 0}

    @app.get("/alerts")
    def alerts():
        return alerts_state()

    @app.post("/alerts")
    def set_alerts(body: dict = Body(...)):
        """The dashboard's text toggle (DH-22): while on, every rescan's flagged changes are texted by alerts/."""
        if not isinstance(body.get("text"), bool):
            raise HTTPException(400, "body must be {\"text\": true|false}")
        store.set_state("text_alerts", body["text"])
        hub.publish({"type": "alerts_changed", **alerts_state()})
        return alerts_state()

    @app.post("/calibrate/table", status_code=202)
    def calibrate_table():
        if source == "synthetic":
            raise HTTPException(400, "the synthetic source has a known table; nothing to calibrate")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "a scan is already running")
        worker.submit(calibrate_job)
        return {"accepted": True, "kind": "calibration"}

    # --- hand crop (Ron 2026-10-17: "it should allow a floor size scan but i can just crop it to be what i want") ---

    def _crop_scan(jc: dict):
        """The newest scan in the CURRENT frame (its full, uncropped cloud is what the crop is drawn on)."""
        for s_ in store.scans():
            sc = scan_io.load_scan(s_["path"], load_image=False)
            if sc.meta.get("table_calib_id") == jc.get("table_calib_id"):
                return sc
        return None

    def _rebuild_baseline_display():
        """Re-crop the baseline's 3D view with the new crop (rescans are cropped at comparison time anyway)."""
        bid = store.get_state("baseline_id")
        if not bid:
            return
        jc = job_cfg()
        base = scan_io.load_scan(store.scan(bid)["path"])
        if base.meta.get("table_calib_id") != jc.get("table_calib_id"):
            return
        _save_display(store.scan(bid)["path"], display_cloud(*preprocess(base.xyz, base.intensity, jc), jc),
                      base.image, job_camera(jc), base.sensor_pose[:3, 3])

    @app.get("/crop")
    def crop_state():
        if source == "synthetic":
            raise HTTPException(400, "the synthetic source has a fixed crop")
        jc = job_cfg()
        c = table_frame.load_crop(cfg, ROOT)
        return {"frame_id": table_frame.frame_id(jc), "roi": jc.get("roi"), "source": jc.get("crop_source"),
                "stale_crop": bool(c and c.get("frame_id") != table_frame.frame_id(jc))}

    @app.get("/crop/preview")
    def crop_preview(max_points: int = 80000, radius_m: float = 4.0, scan_id: str = ""):
        """float32 rows (x, y, z, intensity) of the newest scan in the current frame, UNCROPPED, within radius_m
        (horizontally) of the lidar: what the Crop tool draws on. X-Lidar = the lidar's position in that frame."""
        if source == "synthetic":
            raise HTTPException(400, "the synthetic source has a fixed crop")
        jc = job_cfg()
        if scan_id:
            row = store.scan(scan_id)
            if not row:
                raise HTTPException(404, "no such scan")
            sc = scan_io.load_scan(row["path"], load_image=False)
        else:
            sc = _crop_scan(jc)
        if sc is None:
            raise HTTPException(409, "take a scan first (Set baseline), then crop")
        xyz, inten = sc.xyz, sc.intensity
        lidar = sc.sensor_pose[:3, 3]
        keep = np.hypot(xyz[:, 0] - lidar[0], xyz[:, 1] - lidar[1]) <= radius_m
        xyz, inten = xyz[keep], inten[keep]
        if len(xyz) > max_points:
            idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
            xyz, inten = xyz[idx], inten[idx]
        rows = np.column_stack([xyz, inten]).astype("<f4")
        from fastapi.responses import Response
        return Response(rows.tobytes(), media_type="application/octet-stream",
                        headers={"X-Columns": "x,y,z,intensity", "X-Scan-Id": sc.scan_id, "Cache-Control": "no-cache",
                                 "X-Lidar": ",".join(f"{float(v):.4f}" for v in lidar)})

    @app.post("/crop")
    def set_crop(body: dict = Body(...)):
        if source == "synthetic":
            raise HTTPException(400, "the synthetic source has a fixed crop")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "a scan is running; crop when it is done")
        try:
            jc = job_cfg()
            try:
                box = {k: body[k] for k in ("min", "max", "center", "size", "yaw_deg", "z") if k in body}
                doc = table_frame.save_crop(cfg, ROOT, box,
                                            frame=table_frame.frame_id(jc), from_scan=body.get("scan_id"))
            except (KeyError, ValueError, TypeError) as e:
                raise HTTPException(400, f"bad crop box: {e}")
            _rebuild_baseline_display()
        finally:
            busy.release()
        hub.publish({"type": "crop_changed", "roi": doc["roi"], "source": "hand"})
        return {"roi": doc["roi"], "source": "hand"}

    @app.delete("/crop")
    def reset_crop():
        if source == "synthetic":
            raise HTTPException(400, "the synthetic source has a fixed crop")
        if not busy.acquire(blocking=False):
            raise HTTPException(409, "a scan is running; reset the crop when it is done")
        try:
            path = os.path.join(ROOT, cfg.get("crop_file", table_frame.CROP_FILE_DEFAULT))
            if os.path.exists(path):
                os.replace(path, path + ".reset")        # kept beside it, never silently lost
            _rebuild_baseline_display()
            jc = job_cfg()
        finally:
            busy.release()
        hub.publish({"type": "crop_changed", "roi": jc.get("roi"), "source": jc.get("crop_source")})
        return {"roi": jc.get("roi"), "source": jc.get("crop_source")}

    def _camera_target() -> dict:
        """The scan whose camera frame is clicked for the camera calibration: the newest one with an image, in the
        CURRENT frame (table calibration or sensor). `table` is set when a table calibration exists (corner mode)."""
        jc = job_cfg()
        tdoc = table_frame.load(os.path.join(ROOT, cfg.get("table_frame_file", table_frame.FILE_DEFAULT)))
        for s in store.scans():
            sc = scan_io.load_scan(s["path"], load_image=False)
            if sc.meta.get("table_calib_id") == jc.get("table_calib_id") and sc.meta.get("camera") == "ok":
                return {"scan_id": s["id"], "path": s["path"], "table": tdoc, "frame": table_frame.frame_id(jc)}
        raise HTTPException(409, "take a scan with the camera first (Set baseline), then calibrate the camera")

    @app.get("/calibrate/camera/target")
    def camera_target():
        if source == "synthetic":
            raise HTTPException(400, "the synthetic source has a known camera")
        t = _camera_target()
        sc = scan_io.load_scan(t["path"])
        h, w = sc.image.shape[:2]
        tdoc = t["table"]
        exp = (cfg.get("table_fit") or {}).get("expected_size_m", [0.6, 1.0])
        # corner mode only makes sense when the fitted surface is the diorama (its corners are visible); a larger
        # table or the floor has no clickable corners, so point pairs are the default there (DH-26)
        corners_ok = bool(tdoc) and all(m <= 1.25 * e for m, e in zip(tdoc["measured_size_m"], exp))
        return {"scan_id": t["scan_id"], "image_url": f"/files/{t['scan_id']}/{scan_io.IMAGE_FILE}",
                "image_size": [int(w), int(h)], "corners_ok": corners_ok, "min_pairs": camera_calib.MIN_PAIRS,
                "table_size_m": tdoc["measured_size_m"] if tdoc else None}

    @app.post("/calibrate/camera")
    def calibrate_camera(body: dict = Body(...)):
        t, scan, res, (w, h), fov = _solve_camera(body)
        if not res["ok"]:
            raise HTTPException(422, res["reason"])
        doc = camera_calib.save(cam_file, res, image_size=(w, h), sensor_pose=scan.sensor_pose, scan_id=scan.scan_id,
                                table_calib_id=scan.meta.get("table_calib_id"), fov_diag_deg=fov)
        return _after_camera_saved(scan, doc, res)

    @app.post("/calibrate/camera/auto")
    def calibrate_camera_auto():
        """Refine the CURRENT camera calibration with no clicks: line the lidar's object outlines up with the photo's
        edges on the newest scan (perception/camera_autocal.py, DH-30). Saves and recolours like a click calibration."""
        if source == "synthetic":
            raise HTTPException(400, "the synthetic source has a known camera")
        doc0 = camera_calib.load(cam_file)
        if not doc0:
            raise HTTPException(409, "calibrate the camera roughly first (point pairs), then auto-align refines it")
        t = _camera_target()
        scan = scan_io.load_scan(t["path"])
        cam0 = camera_calib.camera_for_frame(doc0, scan.sensor_pose)
        from perception import camera_autocal
        r = camera_autocal.refine(scan.xyz, scan.image, cam0["K"], cam0["T_cam_from_sensor"], scan.sensor_pose,
                                  dist0=cam0.get("dist"))
        if not r["ok"]:
            raise HTTPException(422, r["reason"])
        h, w = scan.image.shape[:2]
        res = {"K": r["K"], "dist": np.asarray(r["dist"], float), "T_cam_from_frame": r["T_cam_from_frame"], "reproj_px": r["cost_end"],
               "focal_px": r["focal_px"], "focal_from": "auto-align", "camera_to_lidar_m": r["camera_to_lidar_m"],
               "warnings": r["warnings"]}
        doc = camera_calib.save(cam_file, res, image_size=(w, h), sensor_pose=scan.sensor_pose, scan_id=scan.scan_id,
                                table_calib_id=scan.meta.get("table_calib_id"),
                                fov_diag_deg=float(cfg.get("camera", {}).get("fov_diag_deg", 102.0)))
        out = _after_camera_saved(scan, doc, res)
        out.update(auto={k: r[k] for k in ("cost_start", "cost_end", "moved_m", "turned_deg", "n_outline", "k1", "k2")})
        return out

    @app.post("/calibrate/camera/preview")
    def calibrate_camera_preview(body: dict = Body(...)):
        """Solve the clicks WITHOUT saving: the dialog draws the lidar points on the photo with this fit, and each
        pair's error, so a bad pair can be redone before saving (DH-29)."""
        t, scan, res, (w, h), fov = _solve_camera(body)
        if not res["ok"]:
            return {"ok": False, "reason": res["reason"]}
        return {"ok": True, "K": res["K"].tolist(), "T_cam_from_frame": res["T_cam_from_frame"].tolist(),
                "reproj_px": res["reproj_px"], "pair_err_px": res.get("pair_err_px"), "focal_px": res["focal_px"],
                "camera_to_lidar_m": res["camera_to_lidar_m"], "warnings": res["warnings"]}

    def _solve_camera(body: dict):
        if source == "synthetic":
            raise HTTPException(400, "the synthetic source has a known camera")
        t = _camera_target()
        if body.get("scan_id") and body["scan_id"] != t["scan_id"]:
            raise HTTPException(409, "a newer scan arrived; reopen the camera calibration")
        scan = scan_io.load_scan(t["path"])
        h, w = scan.image.shape[:2]
        fov = float(cfg.get("camera", {}).get("fov_diag_deg", 102.0))
        if body.get("pairs"):
            try:
                px = [p["px"] for p in body["pairs"]]
                xyz = [p["xyz"] for p in body["pairs"]]
            except (KeyError, TypeError):
                raise HTTPException(400, "pairs must be [{px: [u, v], xyz: [x, y, z]}, ...]")
            res = camera_calib.solve_from_pairs(px, xyz, (w, h), fov, scan.sensor_pose)
        else:
            corners = body.get("corners")
            if not corners or len(corners) != 4:
                raise HTTPException(400, "send exactly four [x, y] pixel corners, or point pairs")
            if not t["table"]:
                raise HTTPException(409, "corner mode needs a table calibration; use point pairs")
            res = camera_calib.solve_from_corners(corners, camera_calib.table_corners(t["table"]), (w, h), fov,
                                                  scan.sensor_pose)
        return t, scan, res, (w, h), fov

    def _after_camera_saved(scan, doc: dict, res: dict) -> dict:
        # recolour every scan of this table calibration so the 3D view shows colour at once
        n = 0
        for s in store.scans():
            dp = os.path.join(s["path"], "display.npz")
            if not os.path.exists(dp):
                continue
            sc = scan_io.load_scan(s["path"])
            if sc.meta.get("table_calib_id") != scan.meta.get("table_calib_id") or sc.image is None:
                continue
            d = dict(np.load(dp))
            _save_display(s["path"], d, sc.image, camera_calib.camera_for_frame(doc, sc.sensor_pose), sc.sensor_pose[:3, 3])
            n += 1
        summary = {"reproj_px": doc["reproj_px"], "focal_px": doc["focal_px"], "focal_from": doc["focal_from"],
                   "camera_to_lidar_m": doc["camera_to_lidar_m"], "warnings": doc["warnings"], "recoloured_scans": n,
                   "pair_err_px": res.get("pair_err_px")}
        hub.publish({"type": "camera_calibrated", "camera": summary})
        return summary

    @app.post("/baseline", status_code=202)
    def baseline(body: dict = Body(default={})):
        return start(True, body.get("scenario") if body else None)

    @app.post("/scans/trigger", status_code=202)
    def trigger(body: dict = Body(default={})):
        scenario = body.get("scenario") if body else None
        if source == "synthetic" and scenario and scenario not in sources.syn.SCENARIOS:
            raise HTTPException(400, f"unknown scenario; one of {sources.syn.SCENARIOS}")
        # {"replay": "<scan id>"}: re-run an earlier scan as tonight's, lidar off (rehearsal, DH-32)
        return start(False, scenario, (body or {}).get("replay"))

    @app.get("/scans")
    def scans():
        return store.scans()

    @app.get("/scans/{scan_id}/events")
    def scan_events(scan_id: str):
        if not store.scan(scan_id):
            raise HTTPException(404, "no such scan")
        return [_with_urls(e) for e in store.events_of(scan_id)]

    @app.get("/events/{event_id}")
    def event(event_id: str):
        ev = store.event(event_id)
        if not ev:
            raise HTTPException(404, "no such event")
        return _with_urls(ev)

    @app.post("/events/{event_id}/feedback")
    def feedback(event_id: str, body: dict = Body(...)):
        fb = body.get("feedback")
        if fb not in FEEDBACK:
            raise HTTPException(400, f"feedback must be one of {FEEDBACK}")
        ev = store.set_feedback(event_id, fb)
        if not ev:
            raise HTTPException(404, "no such event")
        hub.publish({"type": "event_updated", "event": ev})
        return _with_urls(ev)

    @app.post("/scout/dispatch", status_code=202)
    def dispatch(body: dict = Body(...)):
        ev = store.event(body.get("event_id", ""))
        if not ev:
            raise HTTPException(404, "no such event")
        store.add_dispatch(ev["id"], "queued: scout link not connected yet")
        return {"accepted": True, "event_id": ev["id"], "goal": ev.get("bbox_map"),
                "status": "queued: the Scout link is not connected yet (Demo B)"}

    @app.get("/scans/{scan_id}/cloud")
    def cloud(scan_id: str):
        """float32 rows of (x, y, z, intensity, label, ghost, rgb), little-endian: the dashboard's 3D view.
        label = index into the scan's event list (-1 = unchanged); ghost = 1 for points that disappeared (from the
        baseline); rgb = the camera's colour packed as r*65536 + g*256 + b (exact in float32), -1 = not seen (DH-24)."""
        s = store.scan(scan_id)
        p = s and os.path.join(s["path"], "display.npz")
        if not p or not os.path.exists(p):
            raise HTTPException(404, "no display cloud for this scan")
        d = np.load(p)
        pose = scan_io.load_scan(s["path"], load_image=False).sensor_pose
        if "rgb" in d.files:
            c = d["rgb"].astype(np.int64)
            packed = np.where(d["seen"], c[:, 0] * 65536 + c[:, 1] * 256 + c[:, 2], -1).astype(np.float32)
        else:
            packed = np.full(len(d["xyz"]), -1, np.float32)
        rows = np.column_stack([d["xyz"], d["intensity"], d["label"].astype(np.float32), d["ghost"].astype(np.float32),
                                packed]).astype("<f4")
        from fastapi.responses import Response
        return Response(rows.tobytes(), media_type="application/octet-stream",
                        headers={"X-Columns": "x,y,z,intensity,label,ghost,rgb", "X-Count": str(len(rows)), "Cache-Control": "no-cache",
                                 "X-Lidar": ",".join(f"{float(v):.4f}" for v in pose[:3, 3])})

    @app.get("/map")
    def map_info():
        bid = store.get_state("baseline_id")
        if not bid:
            raise HTTPException(404, "no baseline yet")
        p = store.scan(bid)["path"]
        with open(os.path.join(p, "topdown.json"), encoding="utf-8") as f:
            meta = json.load(f)
        return {"image_url": f"/files/{bid}/topdown.png", "baseline_id": bid, **meta}

    @app.get("/files/{scan_id}/{name:path}")
    def files(scan_id: str, name: str):
        s = store.scan(scan_id)
        if not s:
            raise HTTPException(404, "no such scan")
        base = os.path.realpath(s["path"])
        target = os.path.realpath(os.path.join(base, name))
        if not target.startswith(base + os.sep) or not os.path.isfile(target):
            raise HTTPException(404, "no such file")
        return FileResponse(target)

    @app.websocket("/live")
    async def live(ws: WebSocket):
        await ws.accept()
        hub.clients.add(ws)
        await ws.send_json({"type": "hello", "source": source, "baseline_id": store.get_state("baseline_id")})
        try:
            while True:
                msg = await ws.receive_text()         # keep-alive pings from the dashboard
                if msg == "bridge" and ws not in hub.bridges:   # the alert bridge says who it is (DH-22)
                    hub.bridges.add(ws)
                    hub.publish({"type": "alerts_changed", **alerts_state()})
        except WebSocketDisconnect:
            hub.clients.discard(ws)
            if ws in hub.bridges:
                hub.bridges.discard(ws)
                hub.publish({"type": "alerts_changed", **alerts_state()})

    # The built dashboard (web/dist, `npm run build`) is served at "/" after every API route, so the demo is one
    # process on one URL. In development, run `npm run dev` in web/ instead (Vite proxies the API to :8000).
    dist = os.path.join(ROOT, "web", "dist")
    if os.path.isdir(dist):
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")

    def _save_display(path: str, d: dict, image=None, cam: dict | None = None, origin=None) -> None:
        d = dict(d)
        if origin is not None and len(d["xyz"]):
            # drop lidar edge smear from the 3D view (never a point of a change, nor a ghost) (DH-31)
            veil = veil_mask(d["xyz"], origin) & (d["label"] < 0) & (d["ghost"] == 0)
            d = {k: v[~veil] for k, v in d.items()}
        d["rgb"], d["seen"] = colorize(d["xyz"], image, cam)      # camera colour per point (DH-24)
        np.savez_compressed(os.path.join(path, "display.npz"), **d)

    def _with_urls(ev: dict) -> dict:
        ev = dict(ev)
        for k in ("crop_before", "crop_after"):
            p = ev.get(k)
            if p:
                ev[k + "_url"] = f"/files/{ev['scan_id']}/crops/{os.path.basename(p)}"
        return ev

    return app


def load_dotenv(path: str = os.path.join(ROOT, ".env")) -> list:
    """Secrets (PERCEPTRON_API_KEY, ...) from the gitignored repo-root .env (DH-27). A variable already set in the
    environment wins. Only the served app calls this, never tests (they run with NOXEN_NO_AUTOAPP=1, no network)."""
    loaded = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                k, sep, v = line.strip().partition("=")
                if sep and k and not k.startswith("#") and k not in os.environ:
                    os.environ[k] = v.strip().strip('"').strip("'")
                    loaded.append(k)
    return loaded


if os.environ.get("NOXEN_NO_AUTOAPP") != "1":
    load_dotenv()
    app = create_app()
else:
    app = None

"""Server end to end (§7.1) with the synthetic source: baseline, rescan, events, crops, feedback, dispatch, map, busy,
and the WebSocket progress stream. No network (rules-only classification). Runs under pytest or plain python."""
from __future__ import annotations

import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["NOXEN_NO_AUTOAPP"] = "1"
for k in ("OPENAI_API_KEY", "OPENAI_MODEL", "PERCEPTRON_API_KEY"):
    os.environ.pop(k, None)

from fastapi.testclient import TestClient  # noqa: E402

from server.app import create_app, load_config  # noqa: E402


def wait_idle(client, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if not client.get("/status").json()["busy"]:
            return
        time.sleep(0.2)
    raise AssertionError("scan did not finish")


def make_client():
    cfg = load_config()
    cfg["source"] = "synthetic"
    return TestClient(create_app(cfg, data_dir=tempfile.mkdtemp(prefix="nx_srv_")))


def test_full_flow():
    with make_client() as client:
        st = client.get("/status").json()
        assert st["source"] == "synthetic" and st["baseline_id"] is None and st["providers"][-1] == "rules"
        assert client.get("/map").status_code == 404                                  # no baseline yet

        assert client.post("/baseline", json={}).status_code == 202
        wait_idle(client)
        st = client.get("/status").json()
        assert st["baseline_id"]
        m = client.get("/map").json()
        assert m["image_url"].endswith("topdown.png") and client.get(m["image_url"]).status_code == 200

        assert client.post("/scans/trigger", json={"scenario": "everything"}).status_code == 202
        wait_idle(client)
        scans = client.get("/scans").json()
        latest = scans[0]
        assert latest["status"] == "done", latest
        events = client.get(f"/scans/{latest['id']}/events").json()
        kinds = sorted(e["kind"] for e in events)
        assert kinds == ["added", "appearance", "moved"], kinds
        assert latest["n_routine"] == 1 and latest["n_flagged"] == 2, latest

        cl = client.get(f"/scans/{latest['id']}/cloud")
        assert cl.status_code == 200 and cl.headers["x-columns"] == "x,y,z,intensity,label,ghost,rgb"
        import numpy as np
        rows = np.frombuffer(cl.content, "<f4").reshape(-1, 7)
        assert (rows[:, 6] >= 0).mean() > 0.5                                      # the synthetic camera colours the points (DH-24)
        labels = set(rows[:, 4].astype(int).tolist())
        assert len(rows) > 5000 and {-1, 0, 1} <= labels, labels                  # unchanged + moved + added points
        assert rows[:, 5].max() == 1                                               # ghost points of the moved crate
        assert client.get(f"/scans/{st['baseline_id']}/cloud").status_code == 200

        ev = next(e for e in events if e["kind"] == "appearance")
        r = client.get(ev["crop_after_url"])
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert client.get(f"/events/{ev['id']}").json()["id"] == ev["id"]

        fb = client.post(f"/events/{ev['id']}/feedback", json={"feedback": "false_alarm"}).json()
        assert fb["feedback"] == "false_alarm"
        assert client.post(f"/events/{ev['id']}/feedback", json={"feedback": "maybe"}).status_code == 400
        assert client.get("/scans").json()[0]["n_flagged"] == 1                    # a false alarm stops counting

        d = client.post("/scout/dispatch", json={"event_id": ev["id"]})
        assert d.status_code == 202 and "not connected" in d.json()["status"]
        assert client.get("/files/" + latest["id"] + "/../../../etc/passwd").status_code == 404
        assert client.post("/scans/trigger", json={"scenario": "nonsense"}).status_code == 400


def test_busy_and_websocket_progress():
    with make_client() as client:
        with client.websocket_connect("/live") as ws:
            assert ws.receive_json()["type"] == "hello"
            assert client.post("/baseline", json={}).status_code == 202
            assert client.post("/scans/trigger", json={}).status_code == 409          # one scan at a time
            seen = []
            while True:
                msg = ws.receive_json()
                seen.append(msg["type"])
                if msg["type"] in ("baseline_set", "scan_error"):
                    break
            assert seen[0] == "scan_started" and seen[-1] == "baseline_set", seen

            assert client.post("/scans/trigger", json={"scenario": "crate_moved"}).status_code == 202
            stages, types = [], []
            while True:
                msg = ws.receive_json()
                types.append(msg["type"])
                if msg["type"] == "scan_progress":
                    stages.append(msg["stage"])
                if msg["type"] in ("scan_done", "scan_error"):
                    break
            assert types[-1] == "scan_done" and "event_created" in types, types
            for s in ("accumulating", "aligning", "comparing", "classifying"):
                assert s in stages, (s, stages)


def test_text_alert_toggle_stamps_events():
    """DH-22: the dashboard toggle is stored, each scan's event_created carries it as `text_alert`, and the alert
    bridge's presence ("bridge" on the socket) shows in /status."""
    def scan_events(ws, client, scenario):
        assert client.post("/scans/trigger", json={"scenario": scenario}).status_code == 202
        evs = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "event_created":
                evs.append(msg)
            if msg["type"] in ("scan_done", "scan_error"):
                return evs

    with make_client() as client:
        st = client.get("/status").json()
        assert st["text_alerts"] is False and st["text_bridge"] is False
        assert client.post("/alerts", json={"text": "yes"}).status_code == 400
        with client.websocket_connect("/live") as ws:
            assert ws.receive_json()["type"] == "hello"
            ws.send_text("bridge")
            msg = ws.receive_json()
            assert msg["type"] == "alerts_changed" and msg["text_bridge"] is True, msg
            assert client.get("/status").json()["text_bridge"] is True
            assert client.post("/baseline", json={}).status_code == 202
            while ws.receive_json()["type"] not in ("baseline_set", "scan_error"):
                pass
            off = scan_events(ws, client, "rust_tank")
            assert off and all(m["text_alert"] is False for m in off), off
            assert client.post("/alerts", json={"text": True}).json() == {"text_alerts": True, "text_bridge": True}
            on = scan_events(ws, client, "rust_tank")
            assert on and all(m["text_alert"] is True for m in on), on
        assert client.get("/status").json()["text_alerts"] is True               # persisted


def test_perceptron_toggle_decides_whether_perceptron_is_asked():
    """DH-28: off (the default) = Perceptron is never called, the attempt says it was switched off; on = it is asked
    (here without a key, so it reports that and the fallbacks answer). No network either way."""
    def first_attempt(client, scenario):
        assert client.post("/scans/trigger", json={"scenario": scenario}).status_code == 202
        wait_idle(client)
        evs = client.get(f"/scans/{client.get('/scans').json()[0]['id']}/events").json()
        assert evs
        return [e["classification_attempts"][0] for e in evs]

    with make_client() as client:
        st = client.get("/status").json()["perceptron"]
        assert st["enabled"] is False and st["configured"] is False and st["model"]
        assert client.post("/perceptron", json={"enabled": "on"}).status_code == 400
        assert client.post("/baseline", json={}).status_code == 202
        wait_idle(client)
        off = first_attempt(client, "rust_tank")
        assert all(a["provider"] == "perceptron" and "switched off" in a["detail"] for a in off), off
        assert client.post("/perceptron", json={"enabled": True}).json()["perceptron"]["enabled"] is True
        on = first_attempt(client, "rust_tank")
        assert all(a["provider"] == "perceptron" and "PERCEPTRON_API_KEY" in a["detail"] for a in on), on
        assert client.get("/status").json()["perceptron"]["enabled"] is True     # persisted


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        t0 = time.time()
        fn()
        print(f"  [ok] {name} ({time.time() - t0:.1f} s)")
    print(f"{len(tests)}/{len(tests)} passed")

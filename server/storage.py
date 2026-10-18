"""SQLite store for scans and ChangeEvents (ARCHITECTURE §5, §7.1). The event JSON is stored whole; a few columns are
pulled out for querying. One connection guarded by a lock: the API threads and the worker thread share it."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
  id TEXT PRIMARY KEY, created_at TEXT, is_baseline INTEGER, baseline_id TEXT, path TEXT, source TEXT,
  status TEXT, error TEXT, registration TEXT, stats TEXT, n_points INTEGER);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, scan_id TEXT, baseline_id TEXT, kind TEXT, routine INTEGER, feedback TEXT,
  created_at TEXT, body TEXT);
CREATE TABLE IF NOT EXISTS feedback_log (event_id TEXT, feedback TEXT, at TEXT);
CREATE TABLE IF NOT EXISTS dispatches (event_id TEXT, at TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


class Store:
    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.db.executescript(SCHEMA)
            self.db.commit()

    # -- state ------------------------------------------------------------------------------------------------------
    def get_state(self, key: str):
        with self.lock:
            r = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else None

    def set_state(self, key: str, value) -> None:
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO state VALUES (?,?)", (key, json.dumps(value)))
            self.db.commit()

    # -- scans ------------------------------------------------------------------------------------------------------
    def add_scan(self, scan_id: str, *, path: str, is_baseline: bool, baseline_id: str | None, source: str, n_points: int) -> None:
        with self.lock:
            self.db.execute("INSERT INTO scans (id, created_at, is_baseline, baseline_id, path, source, status, n_points) "
                            "VALUES (?,?,?,?,?,?,?,?)", (scan_id, datetime.now(timezone.utc).isoformat(), int(is_baseline),
                                                         baseline_id, path, source, "captured", n_points))
            self.db.commit()

    def finish_scan(self, scan_id: str, *, status: str, error: str | None = None, registration=None, stats=None) -> None:
        with self.lock:
            self.db.execute("UPDATE scans SET status=?, error=?, registration=?, stats=? WHERE id=?",
                            (status, error, json.dumps(registration), json.dumps(stats), scan_id))
            self.db.commit()

    def scan(self, scan_id: str):
        with self.lock:
            r = self.db.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
        return self._scan_row(r) if r else None

    def scans(self, limit: int = 50) -> list:
        with self.lock:
            rows = self.db.execute("SELECT s.*, "
                                   "(SELECT COUNT(*) FROM events e WHERE e.scan_id=s.id AND e.routine=0 AND e.feedback!='false_alarm') AS n_flagged, "
                                   "(SELECT COUNT(*) FROM events e WHERE e.scan_id=s.id AND e.routine=1) AS n_routine "
                                   "FROM scans s ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._scan_row(r) for r in rows]

    @staticmethod
    def _scan_row(r) -> dict:
        d = dict(r)
        for k in ("registration", "stats"):
            d[k] = json.loads(d[k]) if d.get(k) else None
        d["is_baseline"] = bool(d["is_baseline"])
        return d

    # -- events -----------------------------------------------------------------------------------------------------
    def add_events(self, events: list) -> None:
        with self.lock:
            for ev in events:
                c = ev.get("classification") or {}
                self.db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?)",
                                (ev["id"], ev["scan_id"], ev["baseline_id"], ev["kind"], int(bool(c.get("routine"))),
                                 ev.get("feedback", "none"), ev["created_at"], json.dumps(ev)))
            self.db.commit()

    def events_of(self, scan_id: str) -> list:
        with self.lock:
            rows = self.db.execute("SELECT body FROM events WHERE scan_id=? ORDER BY created_at", (scan_id,)).fetchall()
        return [json.loads(r["body"]) for r in rows]

    def event(self, event_id: str):
        with self.lock:
            r = self.db.execute("SELECT body FROM events WHERE id=?", (event_id,)).fetchone()
        return json.loads(r["body"]) if r else None

    def set_feedback(self, event_id: str, feedback: str):
        """Stored on the event AND appended to a log: every verdict is training data (the data flywheel)."""
        with self.lock:
            r = self.db.execute("SELECT body FROM events WHERE id=?", (event_id,)).fetchone()
            if not r:
                return None
            ev = json.loads(r["body"])
            ev["feedback"] = feedback
            now = datetime.now(timezone.utc).isoformat()
            self.db.execute("UPDATE events SET feedback=?, body=? WHERE id=?", (feedback, json.dumps(ev), event_id))
            self.db.execute("INSERT INTO feedback_log VALUES (?,?,?)", (event_id, feedback, now))
            self.db.commit()
        return ev

    def add_dispatch(self, event_id: str, status: str) -> None:
        with self.lock:
            self.db.execute("INSERT INTO dispatches VALUES (?,?,?)", (event_id, datetime.now(timezone.utc).isoformat(), status))
            self.db.commit()

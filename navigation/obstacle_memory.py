"""Obstacle memory: per-frame looming detections -> persistent, merged, stably-numbered boxes (docs/SCOUT_NAV.md §6;
DN-50, DN-51).

    mem = ObstacleMemory(cfg=None)
    mem.update(detections: list[Obstacle], now) -> bool        True when the memory changed (=> replan)
    mem.active(now) -> list[Obstacle]                          boxes remembered at `now` (copies; ids are memory ids)
    mem.version -> int                                         bumps on every change; the mission replans when it
                                                               differs from the version it planned with
    mem.clear()

Persistence: `memory_s` (default inf = the whole sortie). Looming sees an obstacle only while CLOSING on it, so a
hovering Scout can never re-observe it; a 1.5 s decay (ARCHITECTURE §8.5, built for continuous mono depth) would
forget an obstacle the moment the Scout stopped in front of it (DN-50).
Merge: a detection merges into every stored box it overlaps or comes within merge_gap_m of (stored boxes bridged by
one detection merge too; the lowest id survives). x/y bounds = confidence-weighted running mean of the bounds
(detection weight = max(confidence, 0.05), x raw_weight for "looming_raw"); z bottom = min; top_seen = True once
any detection saw background above it (else the box reaches the ceiling) (DN-51).
Top (DN-51 amended 2026-10-18): max of the SEEN tops (never lowered) and of the live UNSEEN tops. Each top_seen=False
detection is kept as an unseen top with its own footprint; a LATER top_seen detection retires it when its footprint
contains the unseen footprint's centre and covers at least unseen_top_cover_frac of its area. So one ceiling-extended
detection no longer outlives the seen tops of the same barrier, while a seen top of a low box never lowers a merged
tall neighbour (its unseen footprint lies mostly outside the low box's). A later unseen top raises the box again.
The version bumps when a box is added, removed (expiry) or merged, when top_seen changes, or when a bound moves by
more than change_tol_m.
"""
from __future__ import annotations

import copy

import numpy as np

from .types import Obstacle

DEFAULTS = {
    "memory_s": float("inf"),
    "merge_gap_m": 0.3,
    "change_tol_m": 0.1,
    "raw_weight": 0.5,
    "unseen_top_cover_frac": 0.5,
}


def _gap(lo_a, hi_a, lo_b, hi_b) -> float:
    """Euclidean gap between two boxes (0 when they overlap)."""
    d = np.maximum(0.0, np.maximum(lo_a - hi_b, lo_b - hi_a))
    return float(np.linalg.norm(d))


def _add_top(tr, lo, hi, top_seen):
    if top_seen:
        tr.seen_top = max(tr.seen_top, float(hi[2]))
    else:
        tr.unseen.append((np.asarray(lo[:2], float).copy(), np.asarray(hi[:2], float).copy(), float(hi[2])))


def _retire_unseen(tr, lo, hi, frac) -> None:
    """A top_seen detection over (lo, hi) retires every unseen top whose footprint centre it contains and whose
    footprint area it covers at least `frac` of (DN-51 amended)."""
    keep = []
    for ulo, uhi, top in tr.unseen:
        c = (ulo + uhi) / 2
        inside = bool(np.all(lo[:2] <= c) and np.all(c <= hi[:2]))
        ov = np.maximum(0.0, np.minimum(uhi, hi[:2]) - np.maximum(ulo, lo[:2]))
        area = float(np.prod(np.maximum(uhi - ulo, 1e-9)))
        if not (inside and float(np.prod(ov)) >= frac * area):
            keep.append((ulo, uhi, top))
    tr.unseen = keep


def _top(tr) -> float:
    return max([tr.seen_top] + [u[2] for u in tr.unseen])


class _Track:
    def __init__(self, tid, det: Obstacle, w):
        self.id = tid
        self.lo = np.asarray(det.lo, float).copy()
        self.hi = np.asarray(det.hi, float).copy()
        self.top_seen = bool(det.top_seen)
        self.seen_top = -np.inf                      # max top of the top_seen detections
        self.unseen = []                             # live unseen tops: (lo_xy (2,), hi_xy (2,), top z)
        _add_top(self, self.lo, self.hi, det.top_seen)
        self.w = w
        self.first_seen = float(det.first_seen)
        self.last_seen = float(det.last_seen)
        self.source = det.source
        self.confidence = float(det.confidence)
        self.n = 1
        self.detector_ids = [det.id]


class ObstacleMemory:
    def __init__(self, cfg: dict | None = None):
        c = dict(DEFAULTS)
        c.update(cfg or {})
        if c.get("memory_s") is None:
            c["memory_s"] = float("inf")
        self.cfg = c
        self.clear()

    def clear(self):
        self._tracks = []
        self._next_id = 1
        self.version = 0

    def _weight(self, det):
        w = max(float(det.confidence), 0.05)
        return w * (float(self.cfg["raw_weight"]) if det.source == "looming_raw" else 1.0)

    def _expire(self, now) -> bool:
        keep = [t for t in self._tracks if now - t.last_seen <= float(self.cfg["memory_s"])]
        changed = len(keep) != len(self._tracks)
        self._tracks = keep
        return changed

    def update(self, detections, now) -> bool:
        changed = self._expire(now)
        gap = float(self.cfg["merge_gap_m"])
        tol = float(self.cfg["change_tol_m"])
        for det in detections or ():
            lo, hi = np.asarray(det.lo, float), np.asarray(det.hi, float)
            hits = [t for t in self._tracks if _gap(lo, hi, t.lo, t.hi) <= gap]
            w = self._weight(det)
            if not hits:
                tr = _Track(self._next_id, det, w)
                self._next_id += 1
                self._tracks.append(tr)
                changed = True
                continue
            hits.sort(key=lambda t: t.id)
            tr = hits[0]
            for other in hits[1:]:                                   # one detection bridges two stored boxes
                self._absorb(tr, other)
                self._tracks.remove(other)
                changed = True
            old_lo, old_hi, old_top = tr.lo.copy(), tr.hi.copy(), tr.top_seen
            W = tr.w + w
            tr.lo[:2] = (tr.lo[:2] * tr.w + lo[:2] * w) / W
            tr.hi[:2] = (tr.hi[:2] * tr.w + hi[:2] * w) / W
            tr.lo[2] = min(tr.lo[2], lo[2])
            if det.top_seen:
                _retire_unseen(tr, lo, hi, float(self.cfg["unseen_top_cover_frac"]))
            _add_top(tr, lo, hi, det.top_seen)
            tr.hi[2] = _top(tr)
            tr.top_seen = tr.top_seen or bool(det.top_seen)
            tr.w = W
            tr.n += 1
            tr.last_seen = max(tr.last_seen, float(det.last_seen))
            tr.first_seen = min(tr.first_seen, float(det.first_seen))
            tr.confidence = max(tr.confidence, float(det.confidence))
            if det.source == "looming_map":
                tr.source = "looming_map"
            tr.detector_ids.append(det.id)
            if (tr.top_seen != old_top or np.max(np.abs(tr.lo - old_lo)) > tol
                    or np.max(np.abs(tr.hi - old_hi)) > tol):
                changed = True
        if changed:
            self.version += 1
        return changed

    def _absorb(self, a: _Track, b: _Track):
        W = a.w + b.w
        a.lo[:2] = (a.lo[:2] * a.w + b.lo[:2] * b.w) / W
        a.hi[:2] = (a.hi[:2] * a.w + b.hi[:2] * b.w) / W
        a.lo[2] = min(a.lo[2], b.lo[2])
        a.seen_top, a.unseen = max(a.seen_top, b.seen_top), a.unseen + b.unseen
        a.hi[2] = _top(a)
        a.top_seen = a.top_seen or b.top_seen
        a.w, a.n = W, a.n + b.n
        a.first_seen, a.last_seen = min(a.first_seen, b.first_seen), max(a.last_seen, b.last_seen)
        a.confidence = max(a.confidence, b.confidence)
        a.detector_ids += b.detector_ids

    def active(self, now) -> list:
        if self._expire(now):
            self.version += 1
        out = []
        for t in self._tracks:
            out.append(Obstacle(id=t.id, center=(t.lo + t.hi) / 2, half_size=(t.hi - t.lo) / 2, top_seen=t.top_seen,
                                first_seen=t.first_seen, last_seen=t.last_seen, source=t.source,
                                confidence=t.confidence,
                                info={"n_detections": t.n, "detector_ids": copy.copy(t.detector_ids)}))
        return out

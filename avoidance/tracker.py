"""Sparse feature tracks for looming: Shi-Tomasi corners + pyramidal KLT with a forward-backward check.

Each track keeps its pixel position in the last `history` frames (index 0 = the current frame, k = k frames ago,
NaN before the track was born), plus per-track attributes (`attrs`, arrays kept aligned with the tracks).
New corners are detected wherever no track is near, in the same frame whenever >= 8 % of the track budget is free
(a surface that suddenly appears kills the background tracks it covers and gets features at once), otherwise at
least every 2nd frame (detection is the costliest step, DN-33).
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class FeatureTracker:
    def __init__(self, history: int, max_features=400, quality=0.01, min_distance_px=10, block_size=7,
                 win_px=21, levels=3, fb_max_px=0.7, border_px=6, detect_every=2, detect_room_frac=0.08):
        self.H = int(history)
        self.max_features = int(max_features)
        self.quality = float(quality)
        self.min_distance = float(min_distance_px)
        self.block_size = int(block_size)
        self.lk = dict(winSize=(int(win_px), int(win_px)), maxLevel=int(levels),
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.fb_max = float(fb_max_px)
        self.border = float(border_px)
        self.detect_every = int(detect_every)
        self.detect_room = max(1, int(round(detect_room_frac * self.max_features)))
        self.reset()

    def reset(self):
        self._since_detect = 10 ** 6
        self.prev = None
        self.grays = deque(maxlen=self.H)        # newest first, aligned with hist[:, k]
        self.ids = np.zeros(0, np.int64)
        self.hist = np.full((0, self.H, 2), np.nan, np.float32)
        self.attrs = {}                          # name -> (array, default)
        self._next_id = 0

    def add_attr(self, name, default, dtype):
        self.attrs[name] = [np.full(len(self.ids), default, dtype), default]

    def attr(self, name):
        return self.attrs[name][0]

    def set_attr(self, name, values):
        self.attrs[name][0] = np.asarray(values, dtype=self.attrs[name][0].dtype)

    def _keep(self, m):
        self.ids = self.ids[m]
        self.hist = self.hist[m]
        for a in self.attrs.values():
            a[0] = a[0][m]

    def step(self, gray) -> dict:
        """Track into `gray`, drop failures, detect new corners. -> stats."""
        h, w = gray.shape[:2]
        n_in, n_lost = len(self.ids), 0
        self.hist = np.roll(self.hist, 1, axis=1)
        self.hist[:, 0] = np.nan
        if self.prev is not None and n_in:
            p0 = self.hist[:, 1].reshape(-1, 1, 2).astype(np.float32)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None, **self.lk)
            p0r, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, **self.lk)
            p1, p0r = p1.reshape(-1, 2), p0r.reshape(-1, 2)
            fb = np.linalg.norm(p0r - p0.reshape(-1, 2), axis=1)
            b = self.border
            good = ((st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < self.fb_max)
                    & (p1[:, 0] >= b) & (p1[:, 0] <= w - 1 - b) & (p1[:, 1] >= b) & (p1[:, 1] <= h - 1 - b))
            self.hist[:, 0] = p1
            n_lost = int((~good).sum())
            self._keep(good)
        elif n_in:
            self._keep(np.zeros(n_in, bool))
        room = self.max_features - len(self.ids)
        n_new = 0
        self._since_detect += 1
        # detection is the costliest step: run it when tracks were lost (something appeared / left the view) or
        # at least every `detect_every` frames
        if room > 0 and (room >= self.detect_room or self._since_detect >= self.detect_every):
            self._since_detect = 0
            mask = np.full((h, w), 255, np.uint8)
            b = int(np.ceil(self.border))
            mask[:b], mask[-b:], mask[:, :b], mask[:, -b:] = 0, 0, 0, 0
            r = int(round(self.min_distance))
            for x, y in self.hist[:, 0]:
                cv2.circle(mask, (int(round(x)), int(round(y))), r, 0, -1)
            pts = cv2.goodFeaturesToTrack(gray, room, self.quality, self.min_distance, mask=mask,
                                          blockSize=self.block_size)
            if pts is not None and len(pts):
                pts = pts.reshape(-1, 2).astype(np.float32)
                n_new = len(pts)
                hnew = np.full((n_new, self.H, 2), np.nan, np.float32)
                hnew[:, 0] = pts
                self.hist = np.concatenate([self.hist, hnew])
                self.ids = np.concatenate([self.ids, np.arange(self._next_id, self._next_id + n_new)])
                self._next_id += n_new
                for a in self.attrs.values():
                    a[0] = np.concatenate([a[0], np.full(n_new, a[1], a[0].dtype)])
        self.prev = gray
        self.grays.appendleft(gray)
        return {"tracked": n_in - n_lost, "lost": n_lost, "new": n_new, "total": len(self.ids)}

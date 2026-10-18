"""USB camera grab for the table demo (DH-03: runs in Windows Python, where the camera is visible).

The camera is opened when a capture starts and read continuously on a background thread, so
auto-exposure has settled and the frame kept is the newest one at the END of lidar accumulation
(ARCHITECTURE §6.1), not one buffered seconds earlier.
"""
from __future__ import annotations

import sys
import threading
import time

import numpy as np


class CameraError(RuntimeError):
    pass


class CameraGrabber:
    def __init__(self, device=0, width: int | None = None, height: int | None = None, warmup_frames: int = 15):
        self.device, self.width, self.height = device, width, height
        self.warmup_frames = int(warmup_frames)
        self._cap = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._n_frames = 0
        self._error: str | None = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self, open_timeout_s: float = 5.0) -> None:
        import cv2
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY   # DSHOW opens fast and honours size on Windows
        cap = cv2.VideoCapture(self.device, backend)
        if not cap.isOpened():
            raise CameraError(f"cannot open camera {self.device!r}")
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        self._thread = threading.Thread(target=self._loop, name="camera_grab", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + open_timeout_s
        while time.monotonic() < deadline:
            if self._error:
                raise CameraError(self._error)
            if self._n_frames > 0:
                return
            time.sleep(0.02)
        self.stop()
        raise CameraError(f"camera {self.device!r} opened but delivered no frame in {open_timeout_s} s")

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._error = f"camera {self.device!r} read failed"
                return
            with self._lock:
                self._frame = frame
                self._n_frames += 1

    def latest(self, timeout_s: float = 3.0) -> np.ndarray:
        """Newest frame, after at least `warmup_frames` have been read (auto-exposure settling)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._error:
                raise CameraError(self._error)
            with self._lock:
                if self._frame is not None and self._n_frames > self.warmup_frames:
                    return self._frame.copy()
            time.sleep(0.02)
        raise CameraError(f"camera {self.device!r}: fewer than {self.warmup_frames} warm-up frames in time "
                          f"(got {self._n_frames})")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        self._cap = self._thread = None

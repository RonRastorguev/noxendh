"""A faster drop-in for `localization.synthetic_room.Renderer.render` (image only), for the closed-loop sim.

Same algorithm and the same output image (to float rounding at surface intersections): each planar surface is
warped with its homography and composited with a per-pixel depth test. Two changes make it ~2x faster:
- the depth test runs on 1/z (affine in pixel coordinates, no division, no inf buffer);
- a surface that is partly behind the camera is clipped against the near plane before its pixel bounding box is
  taken. The original then uses the whole frame, and inside a room the floor, ceiling and both side walls are
  always partly behind the camera, so most of the frame was composited four or five times.
It can also skip surfaces (`active` mask): that is how an obstacle appears or disappears mid-flight without
rebuilding textures (flightsim/world.py).

The landmark DB is built with the original Renderer; `tests/test_flightsim.py` checks the two images agree.
"""
from __future__ import annotations

import cv2
import numpy as np

NEAR = 0.01


def _clip_near(Xc: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon (camera frame, (N, 3)) against z >= NEAR."""
    out = []
    n = len(Xc)
    for i in range(n):
        a, b = Xc[i], Xc[(i + 1) % n]
        ina, inb = a[2] >= NEAR, b[2] >= NEAR
        if ina:
            out.append(a)
        if ina != inb:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.asarray(out)


class FastRenderer:
    def __init__(self, room, cam, supersample: int = 2):
        if cam.has_distortion:
            raise ValueError("the synthetic renderer draws pinhole images only")
        self.room = room
        self.cam = cam
        self.ss = int(supersample)
        self.cam_ss = cam.scaled(self.ss) if self.ss != 1 else cam
        c = self.cam_ss
        self._dx = ((np.arange(c.width) - c.cx) / c.fx).astype(np.float32)
        self._dy = ((np.arange(c.height) - c.cy) / c.fy).astype(np.float32)
        self._tex = [np.dstack([np.clip(s.texture.astype(np.float32) * s.shade, 0, 255).astype(np.uint8),
                                np.full(s.texture.shape[:2], 255, np.uint8)]) for s in room.surfaces]

    def render(self, T_map_cam, gain: float = 1.0, noise_sigma: float = 0.0, rng=None, active=None,
               return_inv_depth: bool = False):
        """-> BGR uint8 image at the camera resolution (and the supersampled 1/z buffer if asked).
        `active`: optional sequence of bools, one per room surface; False = not drawn."""
        c = self.cam_ss
        Wd, Hd = c.width, c.height
        T = np.asarray(T_map_cam, dtype=np.float64)
        R, cpos = T[:3, :3], T[:3, 3]
        K = c.K
        img = np.zeros((Hd, Wd, 3), np.uint8)
        izbuf = np.full((Hd, Wd), np.float32(1e-6), np.float32)
        for k, (s, tex) in enumerate(zip(self.room.surfaces, self._tex)):
            if active is not None and not active[k]:
                continue
            n = s.normal
            den = float(n @ (s.origin - cpos))
            if den >= -1e-6:
                continue
            Xc = (s.corners() - cpos) @ R
            if np.all(Xc[:, 2] < NEAR):
                continue
            P = Xc if np.all(Xc[:, 2] >= NEAR) else _clip_near(Xc)
            uv = P[:, :2] / P[:, 2:3] * [c.fx, c.fy] + [c.cx, c.cy]
            x0, y0 = max(0, int(np.floor(uv[:, 0].min())) - 1), max(0, int(np.floor(uv[:, 1].min())) - 1)
            x1, y1 = min(Wd, int(np.ceil(uv[:, 0].max())) + 2), min(Hd, int(np.ceil(uv[:, 1].max())) + 2)
            if x0 >= x1 or y0 >= y1:
                continue
            o_t = s.origin + (s.right + s.down) * 0.5 / s.ppm
            H = K @ np.column_stack([R.T @ s.right / s.ppm, R.T @ s.down / s.ppm, R.T @ (o_t - cpos)])
            H = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ H
            warped = cv2.warpPerspective(tex, H, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            nc = R.T @ n
            inv_z = np.add.outer(np.float32(nc[1] / den) * self._dy[y0:y1],
                                 np.float32(nc[0] / den) * self._dx[x0:x1])
            inv_z += np.float32(nc[2] / den)
            zb = izbuf[y0:y1, x0:x1]
            upd = inv_z > zb
            upd &= warped[..., 3] > 127
            np.copyto(zb, inv_z, where=upd)
            np.copyto(img[y0:y1, x0:x1], warped[..., :3], where=upd[..., None])
        if self.ss != 1:
            img = cv2.resize(img, (self.cam.width, self.cam.height), interpolation=cv2.INTER_AREA)
        if gain != 1.0 or noise_sigma > 0:
            out = img.astype(np.float32) * np.float32(gain)
            if noise_sigma > 0:
                rng = rng or np.random.default_rng(0)
                out += rng.standard_normal(out.shape, dtype=np.float32) * np.float32(noise_sigma)
            img = np.clip(out, 0, 255).astype(np.uint8)
        return (img, izbuf) if return_inv_depth else img

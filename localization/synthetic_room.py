"""Synthetic textured room for developing and testing relocalization without hardware data.

What it provides (all deterministic from a seed):
- a room (default 8 x 6 x 3 m, z up) with textured walls, floor, ceiling, and a few textured boxes for 3D structure;
  one wall section is plain paint (the texture-less case), AprilTags can be pasted onto walls;
- `Renderer`: exact perspective rendering with occlusion. Each planar surface is drawn with a homography
  (cv2.warpPerspective) and composited with a per-pixel depth (1/z is affine in pixel coordinates for a plane);
  2x supersampling + area downsampling against aliasing;
- `map_cloud`: a lidar-map-like point cloud of the surfaces (jittered grid at a given spacing + Gaussian noise);
- trajectories: a handheld "mapping walk" (lidar poses; camera = lidar x extrinsics, §8.2) and a separate Scout
  test flight.

Limits (what this does NOT test): pinhole images only (no lens distortion, no rolling shutter, no motion blur),
uniform lighting with per-surface shading, no specular surfaces, a static scene, and a map whose noise is a model
(σ 1 cm, like DH-12). The numbers it produces prove the logic, not real-world accuracy.
"""
from __future__ import annotations

import string
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import CameraModel
from .geometry import R_FLU_OPTICAL, body_pose, make_T, rot_x

# --------------------------------------------------------------------------------------------------------------
# surfaces + textures


@dataclass
class Surface:
    name: str
    origin: np.ndarray         # top-left corner as seen from the front (map frame)
    right: np.ndarray          # unit vector, texture +x
    down: np.ndarray           # unit vector, texture +y
    width: float               # metres along `right`
    height: float              # metres along `down`
    texture: np.ndarray        # (h_px, w_px, 3) uint8 BGR
    ppm: float                 # texture pixels per metre
    shade: float = 1.0

    @property
    def normal(self) -> np.ndarray:
        return np.cross(self.down, self.right)

    def point(self, u_m, v_m) -> np.ndarray:
        return self.origin + self.right * u_m + self.down * v_m

    def corners(self) -> np.ndarray:
        return np.array([self.point(0, 0), self.point(self.width, 0), self.point(self.width, self.height),
                         self.point(0, self.height)])


def _frame_for(normal):
    n = np.asarray(normal, dtype=np.float64)
    if abs(n[2]) < 0.5:
        down = np.array([0.0, 0.0, -1.0])
    else:
        down = np.array([0.0, -1.0, 0.0]) if n[2] > 0 else np.array([0.0, 1.0, 0.0])
    return np.cross(n, down), down


def _rich_texture(w_px, h_px, ppm, rng, density_per_m2=25.0, base=None):
    base = rng.integers(110, 225, 3) if base is None else np.asarray(base)
    img = np.empty((h_px, w_px, 3), np.float32)
    img[:] = base
    low = rng.normal(0, 1, (max(2, h_px // 60), max(2, w_px // 60), 3)).astype(np.float32)
    img += cv2.resize(low, (w_px, h_px), interpolation=cv2.INTER_CUBIC) * 14
    n = int(density_per_m2 * (w_px / ppm) * (h_px / ppm))
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]
    letters = string.ascii_uppercase + string.digits
    for _ in range(n):
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        size = int(rng.uniform(0.04, 0.35) * ppm)
        x, y = int(rng.integers(0, w_px)), int(rng.integers(0, h_px))
        kind = rng.integers(0, 6)
        if kind == 0:
            cv2.rectangle(img, (x, y), (x + size, y + int(size * rng.uniform(0.3, 1.5))), color, -1)
        elif kind == 1:
            cv2.circle(img, (x, y), max(2, size // 2), color, -1, cv2.LINE_AA)
        elif kind == 2:
            p2 = (x + int(rng.normal(0, size)), y + int(rng.normal(0, size)))
            cv2.line(img, (x, y), p2, color, int(rng.integers(1, max(2, size // 10) + 1)), cv2.LINE_AA)
        elif kind == 3:
            pts = (np.array([x, y]) + rng.normal(0, size / 2, (int(rng.integers(3, 7)), 2))).astype(np.int32)
            cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
        elif kind == 4:
            txt = "".join(rng.choice(list(letters), int(rng.integers(2, 7))))
            scale = max(0.4, size / 30.0)
            cv2.putText(img, txt, (x, y), fonts[int(rng.integers(0, 4))], scale, color,
                        max(1, int(scale * 1.5)), cv2.LINE_AA)
        else:
            cv2.ellipse(img, (x, y), (max(2, size // 2), max(2, size // 4)), float(rng.uniform(0, 180)), 0, 360,
                        color, max(1, size // 15), cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return np.clip(img, 0, 255).astype(np.uint8)


def _plain_texture(w_px, h_px, rng, color=(214, 218, 220)):
    img = np.empty((h_px, w_px, 3), np.float32)
    img[:] = color
    low = rng.normal(0, 1, (max(2, h_px // 200), max(2, w_px // 200), 1)).astype(np.float32)
    img += cv2.resize(low, (w_px, h_px), interpolation=cv2.INTER_CUBIC)[..., None] * 3
    img += rng.normal(0, 0.8, img.shape).astype(np.float32)
    return np.clip(cv2.GaussianBlur(img, (0, 0), 1.5), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------------------------------------------
# room


@dataclass
class Tag:
    tag_id: int
    surface: str
    u_m: float                 # tag centre on the surface (metres along right / down from its origin)
    v_m: float
    size_m: float              # black-square edge length


@dataclass
class Room:
    size: tuple
    surfaces: list
    boxes: list                                  # (lo, hi) tuples
    blank: dict                                  # {"surface", "u0", "u1"} the texture-less wall section
    tags: dict = field(default_factory=dict)     # tag_id -> {"T_map_tag": 4x4, "size_m": float}
    tag_family: str = "DICT_APRILTAG_36h11"

    def surface(self, name) -> Surface:
        return next(s for s in self.surfaces if s.name == name)


DEFAULT_BOXES = [("crate_a", (1.0, 0.6, 0.0), (1.8, 1.4, 1.0)),
                 ("shelf_b", (5.5, 0.3, 0.0), (7.5, 0.9, 1.8)),
                 ("crate_c", (6.2, 4.6, 0.0), (7.0, 5.4, 0.7)),
                 ("pillar_d", (3.8, 2.8, 0.0), (4.2, 3.2, 3.0))]

DEFAULT_TAGS = [Tag(0, "wall_x0", 3.0, 1.6, 0.30), Tag(1, "wall_y0", 4.0, 1.5, 0.30),
                Tag(2, "wall_xL", 3.0, 1.6, 0.30), Tag(3, "wall_yW", 2.0, 1.5, 0.30)]


def make_room(seed=0, size=(8.0, 6.0, 3.0), boxes=None, ppm=200.0, blank_u=(4.5, 8.0), tags=None,
              density=1.0) -> Room:
    """`density`: texture clutter multiplier (1.0 = shapes per m2: walls 10, boxes 15, floor 3, ceiling 1.5).
    `blank_u`: the section of wall_yW (metres along its right = +x) painted plain. `tags`: list of Tag or None."""
    rng = np.random.default_rng(seed)
    L, W, H = size
    d = float(density)
    specs = [("wall_x0", (0, W / 2, H / 2), (1, 0, 0), W, H, 1.00, 10.0 * d),
             ("wall_xL", (L, W / 2, H / 2), (-1, 0, 0), W, H, 0.95, 10.0 * d),
             ("wall_y0", (L / 2, 0, H / 2), (0, 1, 0), L, H, 0.97, 10.0 * d),
             ("wall_yW", (L / 2, W, H / 2), (0, -1, 0), L, H, 0.92, 10.0 * d),
             ("floor", (L / 2, W / 2, 0), (0, 0, 1), L, W, 0.80, 3.0 * d),
             ("ceiling", (L / 2, W / 2, H), (0, 0, -1), L, W, 1.05, 1.5 * d)]
    box_list = [(nm, np.asarray(lo, float), np.asarray(hi, float)) for nm, lo, hi in (boxes or DEFAULT_BOXES)]
    for nm, lo, hi in box_list:
        c = (lo + hi) / 2
        ext = hi - lo
        specs += [(f"{nm}_px", (hi[0], c[1], c[2]), (1, 0, 0), ext[1], ext[2], 0.9, 15.0 * d),
                  (f"{nm}_nx", (lo[0], c[1], c[2]), (-1, 0, 0), ext[1], ext[2], 0.85, 15.0 * d),
                  (f"{nm}_py", (c[0], hi[1], c[2]), (0, 1, 0), ext[0], ext[2], 0.88, 15.0 * d),
                  (f"{nm}_ny", (c[0], lo[1], c[2]), (0, -1, 0), ext[0], ext[2], 0.93, 15.0 * d),
                  (f"{nm}_top", (c[0], c[1], hi[2]), (0, 0, 1), ext[0], ext[1], 1.0, 15.0 * d)]
    surfaces = []
    for name, center, normal, w, h, shade, dens in specs:
        right, down = _frame_for(normal)
        origin = np.asarray(center, float) - right * w / 2 - down * h / 2
        w_px, h_px = max(2, int(round(w * ppm))), max(2, int(round(h * ppm)))
        base = (150, 150, 145) if name == "floor" else ((228, 228, 225) if name == "ceiling" else None)
        tex = _rich_texture(w_px, h_px, ppm, rng, dens, base)
        if name == "wall_yW" and blank_u is not None:
            u0, u1 = int(blank_u[0] * ppm), int(min(blank_u[1], w) * ppm)
            tex[:, u0:u1] = _plain_texture(u1 - u0, h_px, rng)
        surfaces.append(Surface(name, origin, right, down, w, h, tex, ppm, shade))
    room = Room(size=tuple(size), surfaces=surfaces, boxes=[(lo, hi) for _, lo, hi in box_list],
                blank={"surface": "wall_yW", "u0": blank_u[0], "u1": blank_u[1]} if blank_u else {})
    for t in (DEFAULT_TAGS if tags is None else tags):
        add_tag(room, t)
    return room


def add_tag(room: Room, tag: Tag) -> None:
    """Paste an AprilTag (with a white quiet zone) into a surface texture and record its pose in map.
    Tag frame (OpenCV / IPPE_SQUARE convention): origin at the centre, x right, y up, z out of the wall."""
    s = room.surface(tag.surface)
    from .apriltag import marker_image
    cells = 8                                                # 36h11: 6x6 data + 1 black border cell each side
    side_px = int(round(tag.size_m * s.ppm))
    marker = marker_image(room.tag_family, tag.tag_id, cells * 10)
    marker = cv2.resize(marker, (side_px, side_px), interpolation=cv2.INTER_NEAREST)
    quiet = int(round(tag.size_m / cells * s.ppm * 1.5))
    x0 = int(round((tag.u_m - tag.size_m / 2) * s.ppm))
    y0 = int(round((tag.v_m - tag.size_m / 2) * s.ppm))
    s.texture[y0 - quiet:y0 + side_px + quiet, x0 - quiet:x0 + side_px + quiet] = 255
    s.texture[y0:y0 + side_px, x0:x0 + side_px] = marker[..., None]
    # the exact centre of the pasted square, in metres (texel i covers [i, i+1) / ppm)
    uc, vc = (x0 + side_px / 2) / s.ppm, (y0 + side_px / 2) / s.ppm
    R = np.stack([s.right, -s.down, s.normal], axis=1)
    room.tags[tag.tag_id] = {"T_map_tag": make_T(R, s.point(uc, vc)), "size_m": side_px / s.ppm}


# --------------------------------------------------------------------------------------------------------------
# rendering


class Renderer:
    def __init__(self, room: Room, cam: CameraModel, supersample: int = 2):
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

    def render(self, T_map_cam, gain: float = 1.0, noise_sigma: float = 0.0, rng=None, blur_sigma: float = 0.0):
        """-> (BGR uint8 image at the camera resolution, depth (camera z, float32, inf = nothing))."""
        c = self.cam_ss
        Wd, Hd = c.width, c.height
        T = np.asarray(T_map_cam, dtype=np.float64)
        R, cpos = T[:3, :3], T[:3, 3]
        K = c.K
        img = np.zeros((Hd, Wd, 3), np.uint8)
        zbuf = np.full((Hd, Wd), np.inf, np.float32)
        for s, tex in zip(self.room.surfaces, self._tex):
            n = s.normal
            den = float(n @ (s.origin - cpos))
            if den >= -1e-6:                                   # camera behind (or in) the surface: back face
                continue
            Xc = (s.corners() - cpos) @ R
            if np.all(Xc[:, 2] < 0.01):
                continue
            x0, y0, x1, y1 = 0, 0, Wd, Hd
            if np.all(Xc[:, 2] > 0.01):
                uv = Xc[:, :2] / Xc[:, 2:3] * [c.fx, c.fy] + [c.cx, c.cy]
                x0, y0 = max(0, int(np.floor(uv[:, 0].min())) - 1), max(0, int(np.floor(uv[:, 1].min())) - 1)
                x1, y1 = min(Wd, int(np.ceil(uv[:, 0].max())) + 2), min(Hd, int(np.ceil(uv[:, 1].max())) + 2)
                if x0 >= x1 or y0 >= y1:
                    continue
            o_t = s.origin + (s.right + s.down) * 0.5 / s.ppm        # texel centres sit at integer coords
            H = K @ np.column_stack([R.T @ s.right / s.ppm, R.T @ s.down / s.ppm, R.T @ (o_t - cpos)])
            H = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ H
            warped = cv2.warpPerspective(tex, H, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            nc = R.T @ n
            inv_z = (np.float32(nc[0] / den) * self._dx[None, x0:x1] + np.float32(nc[1] / den) * self._dy[y0:y1, None]
                     + np.float32(nc[2] / den))
            z = np.full(inv_z.shape, np.inf, np.float32)
            np.divide(np.float32(1.0), inv_z, out=z, where=inv_z > 1e-6)
            zb = zbuf[y0:y1, x0:x1]
            upd = (warped[..., 3] > 127) & (z < zb)
            np.copyto(zb, z, where=upd)
            np.copyto(img[y0:y1, x0:x1], warped[..., :3], where=upd[..., None])
        if self.ss != 1:
            img = cv2.resize(img, (self.cam.width, self.cam.height), interpolation=cv2.INTER_AREA)
            o = self.ss // 2
            zbuf = zbuf[o::self.ss, o::self.ss][: self.cam.height, : self.cam.width]
        if blur_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
        if gain == 1.0 and noise_sigma <= 0:
            return img, zbuf
        out = img.astype(np.float32) * np.float32(gain)
        if noise_sigma > 0:
            rng = rng or np.random.default_rng(0)
            out += rng.standard_normal(out.shape, dtype=np.float32) * np.float32(noise_sigma)
        return np.clip(out, 0, 255).astype(np.uint8), zbuf


# --------------------------------------------------------------------------------------------------------------
# map cloud


def map_cloud(room: Room, spacing_m=0.03, noise_sigma_m=0.01, seed=1) -> np.ndarray:
    """Lidar-map-like cloud of every surface: jittered grid at `spacing_m` + isotropic Gaussian noise.
    Points inside a box (floor under a crate) are dropped; the map never sees them."""
    rng = np.random.default_rng(seed)
    out = []
    for s in room.surfaces:
        nu, nv = max(1, int(s.width / spacing_m)), max(1, int(s.height / spacing_m))
        uu, vv = np.meshgrid((np.arange(nu) + 0.5) * s.width / nu, (np.arange(nv) + 0.5) * s.height / nv)
        uu = uu.reshape(-1) + rng.uniform(-0.3, 0.3, uu.size) * spacing_m
        vv = vv.reshape(-1) + rng.uniform(-0.3, 0.3, vv.size) * spacing_m
        out.append(s.origin + np.outer(uu, s.right) + np.outer(vv, s.down))
    pts = np.concatenate(out)
    inside = np.zeros(len(pts), bool)
    m = 1e-3
    for lo, hi in room.boxes:
        inside |= (np.all((pts[:, :2] > lo[:2] + m) & (pts[:, :2] < hi[:2] - m), axis=1)
                   & (pts[:, 2] > lo[2] - m) & (pts[:, 2] < hi[2] - m))
    pts = pts[~inside]
    pts += rng.normal(0, noise_sigma_m, pts.shape)
    return pts.astype(np.float32)


# --------------------------------------------------------------------------------------------------------------
# trajectories

# Synthetic handheld rig: camera 5 cm ahead of and 8 cm below the lidar, looking along the lidar's +x.
T_LIDAR_CAM_SYNTH = make_T(R_FLU_OPTICAL, (0.05, 0.0, -0.08))
# Synthetic Scout: forward camera 6 cm ahead of the body origin, pitched 5 deg down.
T_BODY_CAM_SYNTH = make_T(R_FLU_OPTICAL @ rot_x(np.radians(-5.0)), (0.06, 0.0, 0.0))


def _ellipse(theta, center, radii):
    x = center[0] + radii[0] * np.cos(theta)
    y = center[1] + radii[1] * np.sin(theta)
    hx, hy = -radii[0] * np.sin(theta), radii[1] * np.cos(theta)       # CCW heading
    return x, y, np.arctan2(hy, hx)


def mapping_walk(rate_hz=10.0, speed_mps=0.6, center=(4.0, 3.0), radii=(2.3, 1.6), height=1.25,
                 yaw_offsets_deg=(0.0, -90.0, 90.0, 180.0)) -> tuple:
    """Handheld mapping walk: one lap of an ellipse per yaw offset (look ahead, out, in, back), lidar at `height`.
    -> (stamps (N,), list of T_map_lidar)."""
    per = 2 * np.pi * np.sqrt((radii[0] ** 2 + radii[1] ** 2) / 2)
    n_lap = int(per / speed_mps * rate_hz)
    stamps, poses = [], []
    t = 0.0
    for lap, off in enumerate(yaw_offsets_deg):
        for i in range(n_lap):
            th = 2 * np.pi * i / n_lap
            x, y, head = _ellipse(th, center, radii)
            z = height + 0.05 * np.sin(5 * th + lap)
            pitch = np.radians(4.0 * np.sin(3 * th + lap))
            poses.append(body_pose((x, y, z), head + np.radians(off), pitch, np.radians(2 * np.sin(2 * th))))
            stamps.append(t)
            t += 1.0 / rate_hz
    return np.asarray(stamps), poses


def scout_flight(rate_hz=10.0, n=150, center=(4.1, 2.9), radii=(1.8, 1.25), phase=0.6) -> tuple:
    """Held-out Scout trajectory (different path, heights 0.6-1.4 m, yaw swings, pitch/roll like a small drone).
    -> (stamps, list of T_map_body)."""
    stamps, poses = [], []
    for i in range(n):
        th = phase + 2 * np.pi * i / n
        x, y, head = _ellipse(th, center, radii)
        z = 1.0 + 0.4 * np.sin(2 * th)
        yaw = head + np.radians(40.0) * np.sin(3 * th)
        poses.append(body_pose((x, y, z), yaw, np.radians(6.0 * np.sin(4 * th)), np.radians(4.0 * np.cos(5 * th))))
        stamps.append(i / rate_hz)
    return np.asarray(stamps), poses


# --------------------------------------------------------------------------------------------------------------
# a complete synthetic dataset: room + map cloud + mapping keyframes -> landmark DB, plus a renderer for test frames


@dataclass
class Dataset:
    room: Room
    cam: CameraModel
    renderer: Renderer
    map_xyz: np.ndarray
    keyframes: list            # landmarks.Keyframe (poses as the map builder believed them)
    keyframes_true: list       # true camera poses of those keyframes
    db: object                 # landmarks.LandmarkDB
    kf_stats: list


def build_dataset(seed=0, cam: CameraModel | None = None, density=1.0, map_spacing_m=0.03, map_noise_m=0.01,
                  kf_min_dist_m=0.3, kf_min_angle_deg=15.0, extrinsic_error=None, build_cfg=None,
                  image_noise=2.0, tags=None) -> Dataset:
    """`extrinsic_error`: optional 4x4 applied to the lidar->camera extrinsics the map builder BELIEVES
    (images are rendered with the true ones), to measure sensitivity to calibration error."""
    from .landmarks import Keyframe, build_landmark_db, select_keyframes
    cam = cam or CameraModel.from_fov(640, 480, 102.0, "diagonal")
    room = make_room(seed, density=density, tags=tags)
    renderer = Renderer(room, cam)
    xyz = map_cloud(room, map_spacing_m, map_noise_m, seed + 1)
    _, lidar_poses = mapping_walk()
    T_lc_true = T_LIDAR_CAM_SYNTH
    T_lc_believed = T_lc_true if extrinsic_error is None else T_lc_true @ np.asarray(extrinsic_error)
    true_poses = [T @ T_lc_true for T in lidar_poses]
    keep = select_keyframes(true_poses, kf_min_dist_m, kf_min_angle_deg)
    kfs, kfs_true = [], []
    for j, i in enumerate(keep):
        img, _ = renderer.render(true_poses[i], noise_sigma=image_noise, rng=np.random.default_rng(seed * 100003 + i))
        kfs.append(Keyframe(j, img, lidar_poses[i] @ T_lc_believed, i / 10.0))
        kfs_true.append(true_poses[i])
    cfg = dict(build_cfg or {})
    cfg.setdefault("map_spacing_m", map_spacing_m)
    db, stats = build_landmark_db(xyz, kfs, cam, cfg, meta={"source": f"synthetic_room seed={seed}"})
    return Dataset(room, cam, renderer, xyz, kfs, kfs_true, db, stats)

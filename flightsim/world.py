"""The closed-loop sim's two sides of the same synthetic room (docs/SCOUT_NAV.md §8).

- MAP side: `localization.synthetic_room` room with the DEFAULT boxes only -> map cloud + landmark DB, built from a
  handheld mapping walk (`sim_mapping_walk`) and cached under data/sim_cache/<key>/. This is everything the flight
  software is allowed to know.
- WORLD side: the same room + unmapped obstacles, appended AFTER the default boxes so every shared surface keeps an
  identical texture (make_room draws textures from one RNG in surface order; `check_texture_identity` proves it).
  The world is what the camera sees, what the rangefinder measures and what the hull can hit.

Obstacles can be toggled and moved mid-flight (the late-obstacle scenario): a box's textures depend on its size
only, so it is generated once at a placeholder position and its surfaces are translated in place when it appears.

Fixed points (DN-72): takeoff pose A on the floor (body origin at scout_hw.yaml derived.fc_height_at_rest_m) and
wall points B on wall_xL.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from localization import synthetic_room as sr
from localization.camera import CameraModel
from localization.geometry import body_pose, rot_angle_deg

from .render import FastRenderer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_VERSION = 1

# hidden obstacles are parked here (outside the room, below the floor) until they are activated
_PARK = np.array([50.0, 50.0, -20.0])

DEFAULTS = {
    "room_seed": 0,
    "density": 1.0,
    "ppm": 200.0,
    "map_spacing_m": 0.03,
    "map_noise_m": 0.01,
    "keyframe_image_noise": 2.0,
    "kf_min_dist_m": 0.3,
    "kf_min_angle_deg": 15.0,
    "supersample": 2,
    # takeoff pose A: x, y, yaw_deg (z = top of whatever is there + fc_height_at_rest_m)
    "takeoff_xy_yaw": [1.0, 4.0, 0.0],
    # the takeoff pad: an ELEVATED BOX centred on A, part of the prior map (Ron 2026-10-17: "assume the takeoff pad
    # is like an elevated box"); [size_x, size_y, height] m, PROVISIONAL; null = take off from the floor (DN-88)
    "pad_box": [0.5, 0.5, 0.4],
    # wall points B (on wall_xL, x = 8.0)
    "wall_points": {"B1": [8.0, 3.6, 1.2], "B2": [8.0, 2.2, 1.2], "B3": [8.0, 4.0, 1.0]},
    # mapping walk (handheld lidar + camera rig); see sim_mapping_walk
    "walk": {
        "base_laps": True,                                    # synthetic_room.mapping_walk default 4 laps
        "outer_lap": {"center": [4.0, 3.0], "radii": [3.0, 2.2], "height": 1.35, "yaw_offsets_deg": [-90.0, 0.0]},
        "wall_sweep": {"x": 7.0, "y_range": [1.5, 4.6], "heights": [0.9, 1.5], "step_m": 0.15},
        "takeoff_column": {"heights": [0.14, 0.3, 0.6, 1.0, 1.4], "yaw_offsets_deg": [-30.0, 0.0, 30.0],
                           "offsets_xy": [[0.0, 0.0], [0.0, 0.5], [0.4, -0.3]]},
    },
    "cache_dir": os.path.join(ROOT, "data", "sim_cache"),
}


def merged(cfg) -> dict:
    out = json.loads(json.dumps(DEFAULTS))
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def _yaml(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_hw(path=None) -> dict:
    return _yaml(path or os.path.join(ROOT, "config", "scout_hw.yaml"))


def default_camera(loc_cfg_path=None) -> CameraModel:
    """The Scout's camera model exactly as the flight software builds it (localization.config)."""
    from localization.config import camera_from_config, load_configs
    return camera_from_config(load_configs(loc_path=loc_cfg_path))


# --------------------------------------------------------------------------------------------------------------
# obstacles


@dataclass
class SimObstacle:
    name: str
    lo: np.ndarray
    hi: np.ndarray
    active: bool = True
    appear_at: float | None = None        # stamp it became visible (None = from the start)
    info: dict = field(default_factory=dict)

    @property
    def center(self):
        return (self.lo + self.hi) / 2

    @property
    def size(self):
        return self.hi - self.lo


def _box_surface_names(name):
    return [f"{name}_{s}" for s in ("px", "nx", "py", "ny", "top")]


# --------------------------------------------------------------------------------------------------------------
# mapping walk


def map_boxes(c: dict) -> list:
    """The MAPPED boxes: synthetic_room's defaults + the takeoff pad box (if any). Unmapped obstacles come after these,
    so every mapped surface keeps the same texture in the map room and the world room."""
    boxes = list(sr.DEFAULT_BOXES)
    pb = c.get("pad_box")
    if pb:
        x, y, _ = c["takeoff_xy_yaw"]
        sx, sy, h = (float(v) for v in pb)
        boxes.append(("pad", (x - sx / 2, y - sy / 2, 0.0), (x + sx / 2, y + sy / 2, h)))
    return boxes


def pad_top_at(c: dict, x, y) -> float:
    """Height of the pad top at (x, y) if (x, y) is on the pad, else 0 (the floor)."""
    pb = c.get("pad_box")
    if not pb:
        return 0.0
    x0, y0, _ = c["takeoff_xy_yaw"]
    if abs(x - x0) <= float(pb[0]) / 2 and abs(y - y0) <= float(pb[1]) / 2:
        return float(pb[2])
    return 0.0


def sim_mapping_walk(walk_cfg: dict, takeoff_xy_yaw, rate_hz=10.0, pad_top=lambda x, y: 0.0) -> list:
    """Handheld mapping walk covering the flight space -> list of T_map_lidar (lidar frame = FLU).

    base laps (synthetic_room default) + an outer lap close to the walls + a lawnmower sweep 1 m in front of wall_xL
    (where the B points are) facing it + a "takeoff column" (the rig held at the takeoff spot from floor level up, so
    the Scout can localize while still on the ground)."""
    poses = []
    if walk_cfg.get("base_laps", True):
        poses += sr.mapping_walk(rate_hz=rate_hz)[1]
    ol = walk_cfg.get("outer_lap")
    if ol:
        _, p = sr.mapping_walk(rate_hz=rate_hz, center=tuple(ol["center"]), radii=tuple(ol["radii"]),
                               height=float(ol["height"]), yaw_offsets_deg=tuple(ol["yaw_offsets_deg"]))
        poses += p
    ws = walk_cfg.get("wall_sweep")
    if ws:
        ys = np.arange(ws["y_range"][0], ws["y_range"][1] + 1e-9, ws["step_m"])
        for j, h in enumerate(ws["heights"]):
            row = ys if j % 2 == 0 else ys[::-1]
            for k, y in enumerate(row):
                yaw = np.radians(8.0 * np.sin(0.9 * k + j))
                poses.append(body_pose((ws["x"] - 0.05 * np.cos(0.7 * k), y, h), yaw, np.radians(3.0 * np.sin(k))))
    tc = walk_cfg.get("takeoff_column")
    if tc:
        x0, y0, yaw0 = takeoff_xy_yaw
        for ox, oy in tc["offsets_xy"]:
            for h in tc["heights"]:
                for dy in tc["yaw_offsets_deg"]:
                    z = h + pad_top(x0 + ox, y0 + oy)              # the column starts on the pad top, not the floor
                    poses.append(body_pose((x0 + ox, y0 + oy, z), np.radians(yaw0 + dy)))
    return poses


# --------------------------------------------------------------------------------------------------------------
# the map side (what the flight software knows), cached


@dataclass
class MapSide:
    room: sr.Room
    map_xyz: np.ndarray
    db: object                   # localization.landmarks.LandmarkDB
    n_keyframes: int
    meta: dict


def _cache_key(c: dict, cam: CameraModel, lm_cfg: dict) -> str:
    d = {k: c[k] for k in ("room_seed", "density", "ppm", "map_spacing_m", "map_noise_m", "keyframe_image_noise",
                           "kf_min_dist_m", "kf_min_angle_deg", "takeoff_xy_yaw", "walk")}
    d.update(cam=cam.to_dict(), lm=lm_cfg, version=CACHE_VERSION, boxes=[list(map(list, b[1:])) for b in map_boxes(c)])
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_map_side(cfg=None, cam: CameraModel | None = None, lm_cfg: dict | None = None, use_cache=True,
                   verbose=False) -> MapSide:
    """Map cloud + landmark DB of the DEFAULT room (no unmapped obstacles), from the sim mapping walk."""
    from localization.landmarks import Keyframe, LandmarkDB, build_landmark_db, select_keyframes
    c = merged(cfg)
    cam = cam or default_camera()
    if lm_cfg is None:
        lm_cfg = dict(_yaml(os.path.join(ROOT, "config", "scout_loc.yaml")).get("landmark_map") or {})
    room = sr.make_room(c["room_seed"], boxes=map_boxes(c), density=c["density"], ppm=c["ppm"])
    key = _cache_key(c, cam, lm_cfg)
    cdir = os.path.join(c["cache_dir"], key)
    if use_cache and os.path.exists(os.path.join(cdir, "meta.json")):
        db = LandmarkDB.load(os.path.join(cdir, "landmarks.npz"))
        xyz = np.load(os.path.join(cdir, "map_xyz.npy"))
        with open(os.path.join(cdir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        return MapSide(room, xyz, db, db.n_keyframes, meta)
    t0 = time.time()
    xyz = sr.map_cloud(room, c["map_spacing_m"], c["map_noise_m"], c["room_seed"] + 1)
    lidar = sim_mapping_walk(c["walk"], c["takeoff_xy_yaw"], pad_top=lambda x, y: pad_top_at(c, x, y))
    cams = [T @ sr.T_LIDAR_CAM_SYNTH for T in lidar]
    keep = select_keyframes(cams, c["kf_min_dist_m"], c["kf_min_angle_deg"])
    rend = sr.Renderer(room, cam, c["supersample"])
    kfs = []
    for j, i in enumerate(keep):
        img, _ = rend.render(cams[i], noise_sigma=c["keyframe_image_noise"],
                             rng=np.random.default_rng(c["room_seed"] * 100003 + i))
        kfs.append(Keyframe(j, img, cams[i], i / 10.0))
    bcfg = dict(lm_cfg)
    bcfg.setdefault("map_spacing_m", c["map_spacing_m"])
    db, _ = build_landmark_db(xyz, kfs, cam, bcfg, meta={"source": f"flightsim room_seed={c['room_seed']}"})
    meta = {"key": key, "n_walk_poses": len(lidar), "n_keyframes": len(kfs), "n_landmarks": int(len(db)),
            "n_map_points": int(len(xyz)), "build_s": round(time.time() - t0, 1), "cfg": {k: c[k] for k in c
                                                                                      if k != "cache_dir"}}
    if use_cache:
        os.makedirs(cdir, exist_ok=True)
        db.save(os.path.join(cdir, "landmarks.npz"))
        np.save(os.path.join(cdir, "map_xyz.npy"), xyz)
        with open(os.path.join(cdir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1, default=str)
    if verbose:
        print(f"[flightsim] built map side {key}: {meta}")
    return MapSide(room, xyz, db, len(kfs), meta)


# --------------------------------------------------------------------------------------------------------------
# the world side (the truth)


class SimWorld:
    """The TRUE world: default room + unmapped obstacles. Renders camera images, answers terrain and clearance."""

    def __init__(self, cfg=None, obstacles=(), cam: CameraModel | None = None, hw: dict | None = None):
        """`obstacles`: iterable of (name, lo, hi) or (name, lo, hi, active); inactive ones are hidden (and parked
        outside the room) until `activate`."""
        self.cfg = merged(cfg)
        self.hw = hw or load_hw()
        self.cam = cam or default_camera()
        self.obstacles: list = []
        boxes = map_boxes(self.cfg)
        for ob in obstacles:
            name, lo, hi = ob[0], np.asarray(ob[1], float), np.asarray(ob[2], float)
            active = bool(ob[3]) if len(ob) > 3 else True
            o = SimObstacle(name, lo.copy(), hi.copy(), active)
            if not active:                                   # textures depend on size only; park it
                o.info["home_lo"] = lo.copy()
            self.obstacles.append(o)
            boxes.append((name, tuple(lo), tuple(hi)))
        c = self.cfg
        self.room = sr.make_room(c["room_seed"], boxes=boxes, density=c["density"], ppm=c["ppm"])
        self.size = np.asarray(self.room.size, float)
        self.default_boxes = [(nm, np.asarray(lo, float), np.asarray(hi, float)) for nm, lo, hi in map_boxes(c)]
        self._surf_index = {s.name: k for k, s in enumerate(self.room.surfaces)}
        self.renderer = FastRenderer(self.room, self.cam, c["supersample"])
        for o in self.obstacles:
            if not o.active:
                self._translate(o, _PARK - o.lo)
        self._update_active()
        self._map_side = None

    # --- obstacles -------------------------------------------------------------------------------------------
    def obstacle(self, name) -> SimObstacle:
        return next(o for o in self.obstacles if o.name == name)

    def _translate(self, o: SimObstacle, delta):
        delta = np.asarray(delta, float)
        for sn in _box_surface_names(o.name):
            s = self.room.surfaces[self._surf_index[sn]]
            s.origin = s.origin + delta
        o.lo, o.hi = o.lo + delta, o.hi + delta

    def place_obstacle(self, name, lo, stamp=None):
        """Move obstacle `name` so its min corner is at `lo` and make it visible/solid from now on."""
        o = self.obstacle(name)
        self._translate(o, np.asarray(lo, float) - o.lo)
        o.active = True
        o.appear_at = stamp
        self._update_active()

    def _update_active(self):
        act = np.ones(len(self.room.surfaces), bool)
        for o in self.obstacles:
            if not o.active:
                for sn in _box_surface_names(o.name):
                    act[self._surf_index[sn]] = False
        self._active = act
        self._solid = [(nm, lo, hi) for nm, lo, hi in self.default_boxes] + \
                      [(o.name, o.lo, o.hi) for o in self.obstacles if o.active]
        self._solid_lo = np.array([b[1] for b in self._solid])
        self._solid_hi = np.array([b[2] for b in self._solid])

    def solid_boxes(self) -> list:
        """[(name, lo, hi)] of everything solid right now (default boxes + active obstacles)."""
        return list(self._solid)

    def active_mask(self, include_obstacles=True) -> np.ndarray:
        if include_obstacles:
            return self._active.copy()
        m = self._active.copy()
        for o in self.obstacles:
            for sn in _box_surface_names(o.name):
                m[self._surf_index[sn]] = False
        return m

    # --- truth queries ----------------------------------------------------------------------------------------
    def terrain_z(self, x, y, z_below=np.inf) -> float:
        """Top of the highest solid surface under (x, y) that is below z_below (floor = 0)."""
        lo, hi = self._solid_lo, self._solid_hi
        m = (x >= lo[:, 0]) & (x <= hi[:, 0]) & (y >= lo[:, 1]) & (y <= hi[:, 1]) & (hi[:, 2] <= z_below + 1e-9)
        return float(hi[m, 2].max()) if np.any(m) else 0.0

    def hull_clearance(self, p, radius, below, above, shape="cylinder") -> dict:
        """Signed clearance of the hull at body position p to every surface.
        cylinder: radius `radius`, from p.z - below (gear) to p.z + above. sphere: radius `radius` about p.
        -> {"lateral": min over walls / ceiling / box sides and undersides (m, < 0 = penetration), "lateral_what",
            "below": clearance to the horizontal surfaces under the hull (floor, box tops it is above), "below_what"}"""
        L, W, H = self.size
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        r = float(radius)
        if shape == "sphere":
            zb, zt = z - r, z + r
        else:
            zb, zt = z - below, z + above
        lat = {"wall_x0": x - r, "wall_xL": L - x - r, "wall_y0": y - r, "wall_yW": W - y - r, "ceiling": H - zt}
        blw = {"floor": zb}
        lo, hi = self._solid_lo, self._solid_hi
        dxv = np.maximum(np.maximum(lo[:, 0] - x, x - hi[:, 0]), 0.0)
        dyv = np.maximum(np.maximum(lo[:, 1] - y, y - hi[:, 1]), 0.0)
        dxy = np.hypot(dxv, dyv)
        for k, (nm, blo, bhi) in enumerate(self._solid):
            d = float(dxy[k])
            gap_above = zb - bhi[2]           # hull above the box top
            gap_below = blo[2] - zt           # hull under the box (never, boxes stand on the floor)
            gz = max(gap_above, gap_below)
            if shape == "sphere":
                # exact sphere-to-box distance
                dz = max(blo[2] - z, z - bhi[2], 0.0)
                inside = d == 0 and dz == 0
                dist = (np.hypot(d, dz) - r) if not inside else -r
                if z - bhi[2] >= 0 and d == 0:
                    blw[nm + "_top"] = z - bhi[2] - r
                else:
                    lat[nm] = float(dist)
                continue
            if gz >= 0 and d <= r:
                if gap_above >= 0:
                    blw[nm + "_top"] = gap_above
                else:
                    lat[nm] = gap_below
            elif gz >= 0:
                lat[nm] = float(np.hypot(d - r, gz))
            elif d > r:
                lat[nm] = d - r
            else:
                lat[nm] = -min(r - d, -gz)
        lw = min(lat, key=lat.get)
        bw = min(blw, key=blw.get)
        return {"lateral": float(lat[lw]), "lateral_what": lw, "below": float(blw[bw]), "below_what": bw}

    def render(self, T_map_cam, gain=1.0, noise_sigma=0.0, rng=None, include_obstacles=True):
        act = self._active if include_obstacles else self.active_mask(False)
        return self.renderer.render(T_map_cam, gain, noise_sigma, rng, active=act)

    # --- fixed points -----------------------------------------------------------------------------------------
    def rest_height(self) -> float:
        return float(self.hw["derived"]["fc_height_at_rest_m"])

    def takeoff_pose(self) -> np.ndarray:
        """Fixed takeoff pose A (T_map_body), resting on the pad box (or the floor without one)."""
        x, y, yaw = self.cfg["takeoff_xy_yaw"]
        return body_pose((x, y, self.terrain_z(x, y) + self.rest_height()), np.radians(yaw))

    def wall_point(self, name="B1") -> np.ndarray:
        return np.asarray(self.cfg["wall_points"][name], float)

    def wall_normal_true(self, point) -> np.ndarray:
        """True inward normal of the room wall the point lies on (sim truth, for arrival scoring)."""
        L, W, _ = self.size
        x, y, _ = point
        d = {(1, 0, 0): x, (-1, 0, 0): L - x, (0, 1, 0): y, (0, -1, 0): W - y}
        return np.asarray(min(d, key=d.get), float)

    # --- map side --------------------------------------------------------------------------------------------
    def map_side(self, use_cache=True, verbose=False) -> MapSide:
        if self._map_side is None:
            self._map_side = build_map_side(self.cfg, self.cam, use_cache=use_cache, verbose=verbose)
        return self._map_side


def check_texture_identity(world: SimWorld, poses=None) -> dict:
    """Render views of the MAP room (mapped boxes only: defaults + pad) and of the WORLD room with its obstacles hidden: the images
    must be identical (shared surfaces share textures). Uses the original synthetic_room Renderer for the map room."""
    map_room = sr.make_room(world.cfg["room_seed"], boxes=map_boxes(world.cfg), density=world.cfg["density"],
                            ppm=world.cfg["ppm"])
    r_map = FastRenderer(map_room, world.cam, world.cfg["supersample"])
    if poses is None:
        A = world.takeoff_pose()
        poses = [A @ sr.T_BODY_CAM_SYNTH,
                 body_pose((4.0, 4.5, 1.3), np.radians(-20.0)) @ sr.T_BODY_CAM_SYNTH,
                 body_pose((7.2, 3.6, 1.2), np.pi) @ sr.T_BODY_CAM_SYNTH]
    worst = 0
    for T in poses:
        a = r_map.render(T)
        b = world.render(T, include_obstacles=False)
        worst = max(worst, int(np.abs(a.astype(int) - b.astype(int)).max()))
    # also the map room through the ORIGINAL renderer (what the landmark DB was built with)
    a0, _ = sr.Renderer(map_room, world.cam, world.cfg["supersample"]).render(poses[0])
    worst_orig = int(np.abs(a0.astype(int) - world.render(poses[0], include_obstacles=False).astype(int)).max())
    return {"identical": worst == 0 and worst_orig == 0, "max_abs_diff": worst, "max_abs_diff_vs_original": worst_orig}


def pose_yaw(T) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def pose_err(T_a, T_b) -> tuple:
    return float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3])), rot_angle_deg(T_a[:3, :3], T_b[:3, :3])

"""Scout navigation (piece C, docs/SCOUT_NAV.md §6) on the SYNTHETIC 8 x 6 x 3 m room (localization/synthetic_room.py:
crates, a shelf and a full-height pillar at x 3.8..4.2, y 2.8..3.2). No hardware data exists; every bound below is a
regression bound on synthetic geometry, not a flight claim. Runs as `py -3.9 tests/test_navigation.py` or pytest.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from localization import synthetic_room as sr  # noqa: E402
from localization.geometry import body_pose  # noqa: E402
from navigation.follower import Follower, wrap  # noqa: E402
from navigation.goal import wall_goal  # noqa: E402
from navigation.nav_config import load_nav_config  # noqa: E402
from navigation.obstacle_memory import ObstacleMemory  # noqa: E402
from navigation.planner import Planner  # noqa: E402
from navigation.safety import SafetyGate  # noqa: E402
from navigation.static_map import StaticMap  # noqa: E402
from navigation.types import GoalPose, Obstacle  # noqa: E402

PILLAR = (np.array([3.8, 2.8, 0.0]), np.array([4.2, 3.2, 3.0]))
_CACHE = {}


def world():
    """(static_map, planner, cfg) on the default synthetic room, built once."""
    if not _CACHE:
        room = sr.make_room(0, ppm=20, tags=[])          # textures are irrelevant here; low ppm = fast
        cfg = load_nav_config()
        sm = StaticMap.from_points(sr.map_cloud(room, spacing_m=0.03, noise_sigma_m=0.01, seed=1), cfg)
        _CACHE.update(sm=sm, planner=Planner(sm, cfg), cfg=cfg, room=room)
    return _CACHE["sm"], _CACHE["planner"], _CACHE["cfg"]


def box(oid, lo, hi, top_seen=True, t=0.0, conf=0.8, source="looming_map"):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Obstacle(oid, (lo + hi) / 2, (hi - lo) / 2, top_seen, t, t, source, conf)


def goal_at(p, yaw=0.0):
    p = np.asarray(p, float)
    return GoalPose(p, yaw, p + [np.cos(yaw), np.sin(yaw), 0.0], -np.array([np.cos(yaw), np.sin(yaw), 0.0]))


def densify(P, step=0.02):
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
        out.append(a + np.linspace(0, 1, n)[1:, None] * (b - a))
    return np.vstack(out)


def box_dist_xy(pts, lo, hi):
    dx = np.maximum(np.maximum(lo[0] - pts[:, 0], pts[:, 0] - hi[0]), 0)
    dy = np.maximum(np.maximum(lo[1] - pts[:, 1], pts[:, 1] - hi[1]), 0)
    return np.hypot(dx, dy)


def assert_steps(sp, cfg, dt=0.05):
    """setpoints never jump: per-tick horizontal / vertical motion within the speed caps."""
    d = np.diff(np.asarray(sp), axis=0)
    assert np.hypot(d[:, 0], d[:, 1]).max() <= cfg["max_speed_mps"] * dt + 1e-6
    assert np.abs(d[:, 2]).max() <= cfg["max_speed_v_mps"] * dt + 1e-6


def fly(follower, p0, yaw0, t_max=90.0, dt=0.05, tau=0.3, yaw_rate_dps=60.0, obstacles=(), on_step=None):
    """Kinematic vehicle: first-order position lag `tau`, yaw lag with a rate limit. -> list of step records."""
    p, yaw, v = np.asarray(p0, float).copy(), float(yaw0), np.zeros(3)
    rec = []
    for k in range(int(t_max / dt)):
        t = k * dt
        if on_step is not None:
            on_step(t, p, follower)
        sp, ysp, done, info = follower.step(t, body_pose(p, yaw), v, obstacles)
        rec.append({"t": t, "p": p.copy(), "yaw": yaw, "sp": sp.copy(), "ysp": ysp, "done": done, "info": info})
        if done:
            break
        p_new = p + (sp - p) * (dt / tau)
        v = (p_new - p) / dt
        p = p_new
        dy = wrap(ysp - yaw) * min(1.0, dt / 0.2)
        yaw = float(wrap(yaw + np.clip(dy, -np.radians(yaw_rate_dps) * dt, np.radians(yaw_rate_dps) * dt)))
    return rec


# ------------------------------------------------------------------------------------------------------------------
# config + static map


def test_config_reads_shared_values():
    import yaml
    cfg = load_nav_config()
    hw = yaml.safe_load(open(os.path.join(ROOT, "config", "scout_hw.yaml"), encoding="utf-8"))
    fd = yaml.safe_load(open(os.path.join(ROOT, "config", "flight_demo.yaml"), encoding="utf-8"))
    assert cfg["hull_radius_m"] == hw["airframe"]["hull_radius_m"]
    assert cfg["hull_half_height_m"] == hw["airframe"]["hull_half_height_m"]
    assert cfg["max_speed_mps"] == fd["safety"]["max_speed_mps"]
    assert cfg["static_inflate_extra_m"] == fd["nav"]["static_inflate_extra_m"]
    assert cfg["dynamic_inflate_extra_m"] == fd["nav"]["dynamic_inflate_extra_m"]
    assert cfg["emergency_hover_s"] == fd["nav"]["emergency_hover_s"]
    if fd["safety"]["geofence_margin_m"] is None:
        assert cfg["geofence_margin_m"] == cfg["geofence_margin_m_default"]
    assert cfg["memory_s"] == float("inf")
    assert cfg["ttc_floor_s"] > cfg["looming_ttc_s"]


def test_static_map():
    sm, _, _ = world()
    assert abs(sm.floor_z) < 0.02 and abs(sm.ceiling_z - 3.0) < 0.02, (sm.floor_z, sm.ceiling_z)
    assert np.all(np.abs(sm.bounds_lo[:2]) < 0.05) and np.all(np.abs(sm.bounds_hi[:2] - [8, 6]) < 0.05)
    assert abs(sm.terrain_z(1.4, 1.0, 2.0) - 1.0) < 0.03                       # crate_a top
    assert abs(sm.terrain_z(1.4, 1.0, 0.5) - sm.floor_z) < 1e-9                # below the crate top: floor
    assert abs(sm.terrain_z(2.5, 4.0, 2.0) - 0.0) < 0.03                       # open floor
    ob = box(1, (2.2, 3.8, 0), (2.8, 4.2, 0.7))
    assert abs(sm.terrain_z(2.5, 4.0, 2.0, [ob]) - 0.7) < 1e-9                  # known obstacle top
    d = sm.distance_along([2.0, 3.0, 1.0], [1, 0, 0], 5.0)
    assert abs(d - 1.8) <= 0.15, d                                             # pillar face at x = 3.8
    d2 = sm.distance_along([2.0, 4.0, 0.5], [1, 0, 0], 5.0, [ob])
    assert abs(d2 - 0.2) < 1e-6, d2
    assert sm.occupied([4.0, 2.8, 1.5]) and not sm.occupied([2.0, 3.0, 1.0])
    # parallel rays (the follower's corridor, DN-57): the centre ray equals distance_along; a ray that starts inside
    # an occupied voxel (beside the hull) or a box ignores that run / box and finds what lies beyond it
    starts = np.array([[2.0, 3.0, 1.0], [2.0, 3.15, 1.0], [3.82, 3.0, 1.0], [2.5, 4.0, 0.5]])
    assert sm.occupied(starts[2])                                              # in the pillar's near-face voxel
    dr = sm.distance_along_rays(starts, [1, 0, 0], 5.0, [ob])
    assert abs(dr[0] - d) < 1e-9 and abs(dr[1] - 1.8) <= 0.15, dr
    assert 0.2 <= dr[2] <= 0.5 and abs(dr[3] - 5.0) < 1e-9, dr                # the far face / nothing past the box
    lo, hi = sm.band(world()[2])
    assert abs(lo - 0.4) < 0.03 and abs(hi - 2.5) < 0.03


# ------------------------------------------------------------------------------------------------------------------
# wall goal


def test_wall_goal_on_each_wall():
    sm, planner, cfg = world()
    cases = {"wall_x0": ((0.0, 4.5, 1.2), (1, 0)), "wall_xL": ((8.0, 4.5, 1.2), (-1, 0)),
             "wall_y0": ((3.0, 0.0, 1.2), (0, 1)), "wall_yW": ((2.0, 6.0, 1.2), (0, -1))}
    free = planner.free_mask([])
    for name, (b, inward) in cases.items():
        g = wall_goal(sm, b, cfg)
        n = np.array([*inward, 0.0])
        assert g.normal @ n > 0.99, (name, g.normal)
        assert np.linalg.norm(g.position - (np.asarray(b) + 0.8 * n)) < 0.03, (name, g.position)
        to_b = np.arctan2(b[1] - g.position[1], b[0] - g.position[0])
        assert abs(np.degrees(wrap(g.yaw - to_b))) < 1.0, name
        assert abs(np.degrees(wrap(g.yaw - np.arctan2(-n[1], -n[0])))) < 3.0, name
        i, j, _ = planner._cell(g.position)
        assert free[planner._layer(g.position[2]), i, j], f"{name}: hover point not free for the planner"
        r = planner.plan([1.0, 3.0, 1.0], g, [])
        assert r.ok, (name, r.reason)


def test_wall_goal_errors_and_clamp():
    sm, _, cfg = world()
    for b, word in [((4.0, 1.8, 1.5), "not on a mapped surface"),     # mid-air
                    ((3.0, 4.5, 0.0), "not a wall"),                  # the floor
                    ((6.5, 0.0, 1.2), "no free hover point")]:        # wall behind the shelf
        try:
            wall_goal(sm, b, cfg)
        except ValueError as e:
            assert word in str(e), (b, str(e))
        else:
            raise AssertionError(f"wall_goal accepted {b}")
    g = wall_goal(sm, (0.0, 4.5, 2.75), cfg)                        # high B: hover z clamped to the band top
    assert abs(g.position[2] - sm.band(cfg)[1]) < 1e-9


# ------------------------------------------------------------------------------------------------------------------
# planner


def test_plan_direct_and_around_pillar():
    sm, planner, cfg = world()
    r = planner.plan([1.5, 4.5, 1.0], goal_at([5.5, 4.5, 1.0]), [])
    assert r.ok and r.shape == "direct" and len(r.waypoints) == 2, (r.shape, r.reason)
    r = planner.plan([2.0, 3.0, 1.0], goal_at([6.0, 3.0, 1.0]), [])
    assert r.ok and r.shape == "around", (r.shape, r.reason)
    S = densify(r.waypoints)
    clear = box_dist_xy(S, *PILLAR).min()
    assert clear >= cfg["hull_radius_m"] + cfg["static_inflate_extra_m"] - 0.02, clear
    assert np.allclose(r.waypoints[0], [2.0, 3.0, 1.0]) and np.allclose(r.waypoints[-1], [6.0, 3.0, 1.0])


def _low_box():
    # 1.4 x 1.4 x 0.7 m, spanning the corridor between the pillar (y <= 3.2) and wall_yW (y = 6): going around means
    # the long way south of the pillar.
    return box(7, (3.3, 3.9, 0.0), (4.7, 5.3, 0.7), top_seen=True)


def test_plan_over_low_wide_box():
    sm, planner, cfg = world()
    ob = _low_box()
    r = planner.plan([1.5, 4.5, 1.0], goal_at([5.5, 4.5, 1.0]), [ob])
    assert r.ok and r.shape == "over", (r.shape, r.reason, r.waypoints)
    assert r.info["over_ids"] == [7]
    S = densify(r.waypoints)
    above = box_dist_xy(S, ob.lo, ob.hi) <= cfg["hull_radius_m"] + cfg["dynamic_inflate_extra_m"]
    need = ob.hi[2] + cfg["hull_half_height_m"] + cfg["dynamic_vmargin_m"]
    assert above.any() and S[above, 2].min() >= need - 1e-6, (S[above, 2].min(), need)
    # the same box with an unseen top cannot be flown over: around, and more expensive
    ob2 = _low_box()
    ob2.top_seen = False
    r2 = planner.plan([1.5, 4.5, 1.0], goal_at([5.5, 4.5, 1.0]), [ob2])
    assert r2.ok and r2.shape == "around", (r2.shape, r2.reason)
    assert r2.cost > r.cost
    S2 = densify(r2.waypoints)
    assert box_dist_xy(S2, ob2.lo, ob2.hi).min() >= cfg["hull_radius_m"] + cfg["dynamic_inflate_extra_m"] - 0.02


def test_plan_around_tall_narrow_box():
    sm, planner, cfg = world()
    ob = box(8, (1.85, 3.75, 0.0), (2.35, 4.25, 1.9), top_seen=False)     # 0.5 x 0.5 x 1.9, top unseen
    r = planner.plan([1.0, 4.0, 1.0], goal_at([3.2, 4.0, 1.0]), [ob])
    assert r.ok and r.shape == "around", (r.shape, r.reason)
    assert r.info["around_ids"] == [8]
    S = densify(r.waypoints)
    assert box_dist_xy(S, ob.lo, ob.hi).min() >= cfg["hull_radius_m"] + cfg["dynamic_inflate_extra_m"] - 0.02
    assert S[:, 2].max() < 1.0 + 0.5 * cfg["layer_dz_m"]


def test_plan_refusals():
    sm, planner, cfg = world()
    g = goal_at([6.0, 2.75, 1.0])                  # open floor between the pillar, the shelf and crate_c
    assert planner.plan([2.0, 4.5, 1.0], g, []).ok
    ring = [box(10, (4.8, 1.55, 0), (7.2, 1.75, 2.0), False), box(11, (4.8, 3.75, 0), (7.2, 3.95, 2.0), False),
            box(12, (4.8, 1.55, 0), (5.0, 3.95, 2.0), False), box(13, (7.0, 1.55, 0), (7.2, 3.95, 2.0), False)]
    r = planner.plan([2.0, 4.5, 1.0], g, ring)
    assert not r.ok and "no path" in r.reason, r.reason
    r = planner.plan([5.0, 4.5, 1.0], goal_at([2.0, 4.0, 1.0]), [box(14, (1.8, 3.8, 0), (2.2, 4.2, 1.9), False)])
    assert not r.ok and "goal inside" in r.reason, r.reason
    r = planner.plan([5.0, 4.5, 1.0], goal_at([4.0, 3.5, 1.0]), [])                 # next to the pillar
    assert not r.ok and "goal inside" in r.reason, r.reason
    r = planner.plan([5.0, 4.5, 1.0], goal_at([3.0, 4.5, 2.9]), [])
    assert not r.ok and "band" in r.reason, r.reason


def test_plan_start_inside_inflation():
    sm, planner, cfg = world()
    ob = box(20, (1.3, 3.7, 0), (1.7, 4.1, 2.0), False)                          # 0.4 m beside the start
    r = planner.plan([1.5, 4.5, 1.0], goal_at([5.5, 4.5, 1.0]), [ob])
    assert r.ok, r.reason
    assert 0 < r.info["start_escape_m"] <= cfg["max_start_escape_m"]
    assert np.allclose(r.waypoints[0], [1.5, 4.5, 1.0])



def test_plan_escape_after_emergency_stop():
    """Stopped by a looming emergency 0.15 m from a tall box face (inside even the hard keep-out): the planner
    hops back out of it and plans around (sim late_obstacle, before the fix: 'start inside inflation')."""
    sm, planner, cfg = world()
    ob = box(21, (5.0, 3.43, 0.0), (5.8, 4.03, 2.0), top_seen=False)
    start = [4.85, 3.8, 1.1]
    r = planner.plan(start, goal_at([6.6, 3.6, 1.1]), [ob])
    assert r.ok, r.reason
    assert 0 < r.info["start_escape_m"] <= cfg["max_start_escape_m"]
    S = densify(r.waypoints[1:])
    assert box_dist_xy(S, ob.lo, ob.hi).min() >= cfg["hull_radius_m"] + cfg["dynamic_inflate_extra_m"] - 0.15

def test_plan_timing():
    sm, planner, cfg = world()
    obs = [_low_box(), box(8, (1.85, 3.75, 0.0), (2.35, 4.25, 1.9), top_seen=False)]
    g = wall_goal(sm, (8.0, 4.5, 1.2), cfg)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        r = planner.plan([1.0, 3.0, 1.0], g, obs)
        times.append(time.perf_counter() - t0)
        assert r.ok, r.reason
    t0 = time.perf_counter()
    Planner(sm, cfg)
    t_init = time.perf_counter() - t0
    med, worst = float(np.median(times)), float(np.max(times))
    print(f"    planner: init {t_init * 1e3:.0f} ms, replan median {med * 1e3:.1f} ms, max {worst * 1e3:.1f} ms "
          f"({planner.L} layers x {planner.nx} x {planner.ny} cells, shape {r.shape})")
    assert worst < 0.3, worst


# ------------------------------------------------------------------------------------------------------------------
# obstacle memory


def test_obstacle_memory():
    mem = ObstacleMemory(load_nav_config())
    assert mem.update([box(100, (2, 2, 0), (2.5, 2.5, 1.0), False, t=1.0)], 1.0)
    v1 = mem.version
    a = mem.active(1.0)
    assert len(a) == 1 and a[0].id == 1
    # an overlapping, slightly shifted detection merges into id 1 (weighted mean), no new box
    mem.update([box(200, (2.06, 2.0, 0), (2.56, 2.5, 1.0), False, t=1.1)], 1.1)
    a = mem.active(1.1)
    assert len(a) == 1 and a[0].id == 1 and abs(a[0].lo[0] - 2.03) < 1e-6
    assert mem.version == v1                                           # 3 cm move < change_tol: no replan
    # a far detection is a new box; top_seen from a later detection allows "over"
    assert mem.update([box(300, (5, 4, 0), (5.4, 4.4, 0.6), False, t=2.0)], 2.0)
    assert len(mem.active(2.0)) == 2
    v2 = mem.version
    assert mem.update([box(301, (5.05, 4, 0), (5.45, 4.4, 0.62), True, t=2.1)], 2.1)
    assert mem.version > v2 and [o.top_seen for o in mem.active(2.1) if o.id == 2] == [True]
    # one detection bridging both boxes merges them; the lower id survives
    mem.update([box(400, (2.4, 2.4, 0), (5.1, 4.1, 0.5), False, t=3.0)], 3.0)
    a = mem.active(3.0)
    assert [o.id for o in a] == [1]
    # default memory: the whole sortie (a hovering Scout cannot re-see it); a finite memory expires
    assert len(mem.active(1000.0)) == 1
    short = ObstacleMemory({"memory_s": 2.0})
    short.update([box(1, (0, 0, 0), (1, 1, 1), t=0.0)], 0.0)
    v = short.version
    assert len(short.active(1.5)) == 1 and len(short.active(2.5)) == 0 and short.version > v


def _det(oid, center, half, top_seen, t):
    c, h = np.asarray(center, float), np.asarray(half, float)
    return box(oid, c - h, c + h, top_seen, t=t)


# The rest of a wall-to-wall barrier, as in the sim's low_barrier: a tall unseen-top wall north of the barrier piece
# the memory holds (x 4.6..5.4 from y 2.9 to the room wall). Around is closed; the only way is OVER the piece.
_BARRIER_REST = box(90, (4.6, 2.9, 0.0), (5.4, 5.98, 3.0), top_seen=False)


def _plan_across(mem_boxes):
    sm, planner, cfg = world()
    return planner.plan([2.8, 1.8, 1.0], goal_at([6.6, 2.0, 1.0]), list(mem_boxes) + [_BARRIER_REST])


def test_obstacle_memory_unseen_top_retired_by_later_seen_top():
    """DN-51 amended: one ceiling-extended (top_seen False) detection merged with the 0.6 m barrier must not outlive
    the later SEEN-top detections of it. The event sequence of the sim's low_barrier seed 0 (705e687 + DN-63), which
    left a box up to the ceiling with top_seen True: boxed in, landed after no_path_land_s."""
    mem = ObstacleMemory(load_nav_config())
    for k, (t, c, h) in enumerate([(9.95, (4.67, 1.66, 0.44), (0.59, 0.54, 0.46)),
                                   (10.05, (4.74, 1.54, 0.50), (0.65, 0.60, 0.52)),
                                   (10.15, (4.78, 1.51, 0.52), (0.66, 0.61, 0.54))]):
        mem.update([_det(500 + k, c, h, True, t)], t)
    r0 = _plan_across(mem.active(10.15))
    assert r0.ok and "over" in r0.shape, (r0.shape, r0.reason)
    # the top goes out of view: ceiling-extended, then "nothing near above" (top = highest member + pad)
    mem.update([_det(510, (4.87, 1.43, 1.50), (0.59, 0.54, 1.52), False, 10.25)], 10.25)
    mem.update([_det(511, (4.85, 1.44, 0.72), (0.59, 0.54, 0.74), False, 10.35)], 10.35)
    a = mem.active(10.35)
    assert len(a) == 1 and a[0].hi[2] > 3.0, [(o.id, o.hi) for o in a]           # conservative while unconfirmed
    r1 = _plan_across(a)
    assert not r1.ok and "boxed in" in r1.reason, (r1.shape, r1.reason)
    v = mem.version
    mem.update([_det(512, (4.72, 1.31, 0.53), (0.44, 0.43, 0.55), True, 10.45)], 10.45)
    mem.update([_det(513, (4.70, 1.35, 0.54), (0.45, 0.43, 0.56), True, 10.55)], 10.55)
    a = mem.active(10.55)
    assert len(a) == 1 and a[0].top_seen and abs(a[0].hi[2] - 1.10) < 1e-6, [(o.id, o.hi) for o in a]
    assert mem.version > v                                                         # the top dropped: replan
    r2 = _plan_across(a)
    assert r2.ok and "over" in r2.shape and a[0].id in r2.info["over_ids"], (r2.shape, r2.reason, r2.info)


def test_obstacle_memory_seen_low_top_keeps_tall_neighbour():
    """A merge bridges distinct objects within merge_gap_m. A seen top of the LOW one never lowers the merged TALL
    neighbour: an unseen top is retired only by a seen-top detection over its own footprint (its centre and at least
    unseen_top_cover_frac of its area), and seen tops are never lowered at all."""
    low = ((4.70, 1.35, 0.30), (0.40, 0.45, 0.30))                       # x 4.30..5.10, y 0.90..1.80, top 0.6
    tall = ((5.55, 1.35, 1.51), (0.25, 0.30, 1.51))                      # x 5.30..5.80: 0.2 m east, top 3.02 unseen
    mem = ObstacleMemory(load_nav_config())
    mem.update([_det(600, *low, True, 1.0)], 1.0)
    mem.update([_det(601, *tall, False, 1.1)], 1.1)
    for k in range(3):                                                   # repeated seen tops of the low box only
        mem.update([_det(602 + k, *low, True, 1.2 + 0.1 * k)], 1.2 + 0.1 * k)
    a = mem.active(1.5)
    assert len(a) == 1 and a[0].top_seen and a[0].hi[2] > 3.0, [(o.id, o.hi) for o in a]
    r = _plan_across(a)
    assert not (r.ok and a[0].id in r.info.get("over_ids", [])), (r.shape, r.reason, r.info)
    # one wide unseen detection bridging both, centred over the tall one: not retired by the low box's seen top
    mem2 = ObstacleMemory(load_nav_config())
    mem2.update([_det(700, (5.20, 1.35, 1.51), (0.60, 0.45, 1.51), False, 1.0)], 1.0)   # x 4.60..5.80
    for k in range(3):
        mem2.update([_det(701 + k, *low, True, 1.1 + 0.1 * k)], 1.1 + 0.1 * k)
    assert mem2.active(1.5)[0].hi[2] > 3.0
    # a SEEN tall top (a 2.0 m shelf) merged with the low box stays at 2.0
    mem3 = ObstacleMemory(load_nav_config())
    mem3.update([_det(800, (5.55, 1.35, 1.0), (0.25, 0.30, 1.0), True, 1.0)], 1.0)
    for k in range(3):
        mem3.update([_det(801 + k, *low, True, 1.1 + 0.1 * k)], 1.1 + 0.1 * k)
    a3 = mem3.active(1.5)
    assert len(a3) == 1 and abs(a3[0].hi[2] - 2.0) < 1e-6, [(o.id, o.hi) for o in a3]
    # order matters: an unseen top that comes AFTER the seen ones raises the box again (new evidence of a tall top)
    v = mem3.version
    mem3.update([_det(810, (4.72, 1.33, 1.51), (0.42, 0.45, 1.51), False, 2.0)], 2.0)
    assert mem3.active(2.0)[0].hi[2] > 3.0 and mem3.version > v


# ------------------------------------------------------------------------------------------------------------------
# follower


def test_follower_yaw_first():
    sm, _, cfg = world()
    f = Follower(cfg, sm)
    f.set_path(np.array([[2.0, 4.5, 1.0], [2.0, 2.4, 1.0]]), -np.pi / 2)
    rec = fly(f, [2.0, 4.5, 1.0], 0.0, t_max=20)
    moved = False
    for a, b in zip(rec[:-1], rec[1:]):
        rate = abs(np.degrees(wrap(b["ysp"] - a["ysp"]))) / 0.05
        assert rate <= cfg["max_yaw_rate_dps"] + 1e-6, rate
        herr = abs(np.degrees(wrap(-np.pi / 2 - b["yaw"])))         # the heading the step was decided with
        if np.linalg.norm(b["sp"][:2] - a["sp"][:2]) > 1e-9:
            moved = True
            assert herr <= cfg["max_heading_err_deg"] + 1e-6, herr
        if not moved:
            assert np.linalg.norm(a["sp"][:2] - [2.0, 4.5]) < 1e-9
    assert moved and rec[-1]["done"]
    assert any(r["info"]["mode"] == "yaw_first" for r in rec)


def test_follower_speed_and_accel_limits():
    sm, _, cfg = world()
    dt = 0.05
    for path, gy in [(np.array([[1.0, 4.5, 1.0], [5.5, 4.5, 1.0]]), 0.0),          # horizontal
                     (np.array([[1.0, 4.5, 0.6], [1.0, 4.5, 2.2]]), 0.0)]:         # vertical
        f = Follower(cfg, sm)
        f.set_path(path, gy)
        rec = fly(f, path[0], 0.0, t_max=40, dt=dt)
        assert rec[-1]["done"], rec[-1]["info"]
        sp = np.array([r["sp"] for r in rec])
        v = np.diff(sp, axis=0) / dt
        vh, vz = np.hypot(v[:, 0], v[:, 1]), np.abs(v[:, 2])
        assert vh.max() <= cfg["max_speed_mps"] + 1e-6 and vz.max() <= cfg["max_speed_v_mps"] + 1e-6
        assert vh.max() > 0.9 * cfg["max_speed_mps"] or vz.max() > 0.9 * cfg["max_speed_v_mps"]
        acc = np.linalg.norm(np.diff(v, axis=0), axis=1) / dt
        assert np.sum(acc > cfg["max_accel_mps2"] + 1e-6) <= 1, np.sort(acc)[-3:]  # the one tick landing on the goal
        assert acc.max() <= 2 * cfg["max_accel_mps2"] + 1e-6
        yaw = np.array([r["ysp"] for r in rec])
        if path[0, 2] != path[1, 2]:
            assert np.allclose(yaw, 0.0)                                         # vertical: no yawing


def test_follower_ttc_speed_law_near_wall():
    sm, _, cfg = world()
    f = Follower(cfg, sm)
    goal = np.array([7.2, 3.0, 1.2])
    f.set_path(np.array([[5.0, 3.0, 1.2], goal]), 0.0)
    rec = fly(f, [5.0, 3.0, 1.2], 0.0, t_max=30)
    assert rec[-1]["done"]
    for a, b in zip(rec[:-1], rec[1:]):
        vh = np.linalg.norm(b["sp"][:2] - a["sp"][:2]) / 0.05
        d = sm.distance_along(a["p"], [1, 0, 0], 5.0)
        assert vh <= max(0.0, d - cfg["stop_margin_m"]) / cfg["ttc_floor_s"] + 0.03, (a["p"], vh, d)
        # the TTC the looming detector would measure on the wall stays above its emergency threshold
        if vh > 0.05:
            assert d / vh > cfg["looming_ttc_s"]
    assert any(r["info"]["limit"] == "ttc" for r in rec)
    # a known obstacle ahead bounds the speed the same way
    f2 = Follower(cfg, sm)
    f2.set_path(np.array([[1.0, 4.5, 1.0], [2.0, 4.5, 1.0]]), 0.0)
    ob = box(1, (2.3, 4.2, 0.0), (2.6, 4.8, 1.9), False)
    rec2 = fly(f2, [1.0, 4.5, 1.0], 0.0, t_max=5, obstacles=[ob])
    assert all(r["info"]["d_ahead_m"] <= 1.31 for r in rec2[1:])
    assert max(r["info"]["v"] for r in rec2) <= (1.3 - cfg["stop_margin_m"]) / cfg["ttc_floor_s"] + 1e-6


def test_follower_reaches_wall_goal_around_pillar():
    sm, planner, cfg = world()
    g = wall_goal(sm, (8.0, 3.0, 1.2), cfg)
    r = planner.plan([2.0, 3.0, 1.0], g, [])
    assert r.ok and r.shape in ("around", "over+around"), (r.shape, r.reason)
    f = Follower(cfg, sm)
    f.set_path(r.waypoints, g.yaw)
    rec = fly(f, [2.0, 3.0, 1.0], 0.0, t_max=90)
    last = rec[-1]
    assert last["done"], last["info"]
    assert np.linalg.norm(last["p"] - g.position) <= cfg["arrive_tol_m"]
    assert abs(np.degrees(wrap(last["yaw"] - g.yaw))) <= cfg["arrive_tol_deg"]
    traj = np.array([x["p"] for x in rec])
    assert box_dist_xy(traj, *PILLAR).min() >= cfg["hull_radius_m"] + cfg["static_inflate_extra_m"] - 0.05
    sp = np.array([x["sp"] for x in rec])
    assert_steps(sp, cfg)
    print(f"    follower: wall goal around the pillar in {last['t']:.1f} s (sim), final error "
          f"{np.linalg.norm(last['p'] - g.position) * 100:.1f} cm")


def test_follower_flies_over_low_box():
    sm, planner, cfg = world()
    ob = _low_box()
    g = goal_at([5.5, 4.5, 1.0])
    r = planner.plan([1.5, 4.5, 1.0], g, [ob])
    assert r.ok and r.shape == "over"
    f = Follower(cfg, sm)
    f.set_path(r.waypoints, g.yaw)
    rec = fly(f, [1.5, 4.5, 1.0], 0.0, t_max=60, obstacles=[ob])
    assert rec[-1]["done"], rec[-1]["info"]
    traj = np.array([x["p"] for x in rec])
    over = box_dist_xy(traj, ob.lo, ob.hi) <= cfg["hull_radius_m"]
    assert over.any() and traj[over, 2].min() >= ob.hi[2] + cfg["hull_half_height_m"] + 0.2, traj[over, 2].min()
    assert_steps([x["sp"] for x in rec], cfg)


def test_follower_hold_and_replan_continuity():
    sm, _, cfg = world()
    f = Follower(cfg, sm)
    sp, y, done, info = f.step(0.0, body_pose([2.0, 4.5, 1.0], 0.3), np.zeros(3))
    assert info["mode"] == "hold" and np.allclose(sp, [2.0, 4.5, 1.0]) and abs(y - 0.3) < 1e-9 and not done
    f = Follower(cfg, sm)
    f.set_path(np.array([[1.0, 4.5, 1.0], [5.0, 4.5, 1.0]]), 0.0)

    def replan(t, p, fol):
        if abs(t - 3.0) < 1e-9:                    # new plan from the BODY position (lagging the reference)
            fol.set_path(np.array([p, [5.0, 4.5, 1.4]]), 0.0)

    rec = fly(f, [1.0, 4.5, 1.0], 0.0, t_max=30, on_step=replan)
    sp = np.array([r["sp"] for r in rec])
    assert_steps(sp, cfg)
    assert rec[-1]["done"] and abs(rec[-1]["p"][2] - 1.4) < 0.1


def _corridor_contact_s(p, d, v, cfg, L=8.0, W=6.0):
    """TRUE hull-contact time the looming emergency measures (DN-25) against the room walls: the nearest wall
    point within hull radius + corridor margin of the motion line, (along - hull radius) / speed."""
    a_h = cfg["hull_radius_m"] + cfg["corridor_margin_m"]
    side = np.array([-d[1], d[0]])
    best = np.inf
    for lat in np.linspace(-a_h, a_h, 71):
        q = p[:2] + lat * side
        for k, lim in ((0, L), (1, W)):
            if abs(d[k]) > 1e-6:
                best = min(best, ((lim if d[k] > 0 else 0.0) - q[k]) / d[k])
    return (best - cfg["hull_radius_m"]) / v


def _worst_corridor_contact(rec, cfg, dt=0.05):
    P = np.array([r["p"] for r in rec])
    V = np.diff(P, axis=0) / dt
    worst = np.inf
    for k in range(len(V)):
        v = float(np.hypot(V[k, 0], V[k, 1]))
        if v > 0.05:
            worst = min(worst, _corridor_contact_s(P[k], V[k, :2] / v, v, cfg))
    return worst


def test_follower_ttc_corridor_oblique_wall():
    """Diagonal approach (31 deg) to the mapped north wall, to a vertex 0.65 m from it, then along it (the sim's
    low_box / drift route around the box, SCOUT_NAV.md §10). The looming emergency judges the hull-contact time
    over its whole corridor, whose edge meets an oblique wall first: with the old single centre ray the TRUE
    corridor contact time fell to 1.65 s (estimated 1.26 s in the closed loop: a false looming_raw obstacle); the
    corridor law keeps it >= ttc_floor_s and still reaches the vertex (DN-57)."""
    sm, _, cfg = world()
    path = np.array([[2.0, 4.2, 1.1], [3.9, 5.35, 1.1], [5.2, 5.35, 1.1]])
    worst = {}
    for name, n_rays in (("centre_ray", 1), ("corridor", cfg["ttc_rays_lateral"])):
        f = Follower(dict(cfg, ttc_rays_lateral=n_rays), sm)
        f.set_path(path, 0.0)
        rec = fly(f, path[0], float(np.arctan2(1.15, 1.9)), t_max=60)
        assert rec[-1]["done"], (name, rec[-1]["info"])
        worst[name] = (_worst_corridor_contact(rec, cfg), rec[-1]["t"])
    print(f"    follower: oblique wall, worst true corridor contact centre ray {worst['centre_ray'][0]:.2f} s, "
          f"corridor {worst['corridor'][0]:.2f} s (arrival {worst['centre_ray'][1]:.1f} / {worst['corridor'][1]:.1f} s)")
    assert worst["centre_ray"][0] < 0.9 * cfg["ttc_floor_s"]                   # the geometry exposes the old law
    assert worst["corridor"][0] >= 0.95 * cfg["ttc_floor_s"], worst            # (5 % for the vehicle's lag)
    assert worst["corridor"][1] <= worst["centre_ray"][1] + 2.0, worst


def test_follower_short_hop_faces_next_segment():
    """A short sidestep (the planner's escape hop into a channel, 0.1-0.2 m toward a near wall in the sim) is flown
    facing the NEXT segment, not turned onto: the camera never creeps head-on toward the wall, and two yaw-in-place
    turns are saved (DN-58). A longer hop is still flown look-where-you-go."""
    sm, _, cfg = world()
    for hop, short in ((0.18, True), (0.5, False)):
        f = Follower(cfg, sm)
        path = np.array([[2.0, 4.0, 1.0], [2.0, 4.0 + hop, 1.0], [4.5, 4.0 + hop, 1.0]])
        f.set_path(path, 0.0)
        rec = fly(f, path[0], 0.0, t_max=40)
        assert rec[-1]["done"], rec[-1]["info"]
        yaw_max = max(abs(np.degrees(r["yaw"])) for r in rec)
        if short:
            assert yaw_max <= cfg["max_heading_err_deg"], yaw_max
            assert not any(r["info"]["mode"] == "yaw_first" for r in rec)
            P = np.array([r["p"] for r in rec])
            assert abs(P[P[:, 0] < 2.02, 1].max() - path[1, 1]) < 0.03             # the hop is flown, not cut
        else:
            assert yaw_max > 60.0, yaw_max


def test_follower_final_yaw_stays_in_the_heading_gate_while_moving():
    """DN-53 amended: a goal yaw 180 deg from the final segment (the return to the pad) is NOT turned onto while the
    Scout still moves: the camera stays within the heading gate of the motion; the rest of the turn is in place."""
    sm, _, cfg = world()
    f = Follower(cfg, sm)
    path = np.array([[4.5, 4.5, 1.0], [2.0, 4.5, 1.0]])                    # flown -x, goal yaw +x
    f.set_path(path, 0.0)
    rec = fly(f, path[0], np.pi, t_max=40)
    assert rec[-1]["done"], rec[-1]["info"]
    gate = cfg["max_heading_err_deg"]
    for a, b in zip(rec[:-1], rec[1:]):
        if np.linalg.norm(b["sp"][:2] - a["sp"][:2]) > 1e-9:               # the reference moved this tick
            assert abs(np.degrees(wrap(np.pi - b["yaw"]))) <= gate + 1.0, np.degrees(b["yaw"])
    assert abs(np.degrees(wrap(rec[-1]["yaw"]))) <= cfg["arrive_tol_deg"]


# ------------------------------------------------------------------------------------------------------------------
# safety gate


def test_safety_clamps():
    sm, _, cfg = world()
    gate = SafetyGate(cfg, sm)
    lo, hi = sm.geofence(cfg)
    p, y, act, why = gate.filter(0.0, [-1.0, 3.0, 1.0], 0.0, body_pose([0.5, 3.0, 1.0], 0.0), True, False)
    assert act == "CLAMP" and "geofence" in why and abs(p[0] - lo[0]) < 1e-9
    assert abs(lo[0] - (sm.bounds_lo[0] + cfg["geofence_margin_m"])) < 1e-9
    gate.reset()
    z_lo, z_hi = sm.band(cfg)
    p, _, act, why = gate.filter(0.0, [4.0, 1.5, 2.9], 0.0, body_pose([4.0, 1.5, 2.4], 0.0), True, False)
    assert act == "CLAMP" and "band" in why and abs(p[2] - z_hi) < 1e-9
    gate.reset()
    p, _, act, why = gate.filter(0.0, [4.0, 1.5, 0.1], 0.0, body_pose([4.0, 1.5, 0.5], 0.0), True, False)
    assert "band" in why and abs(p[2] - z_lo) < 1e-9
    gate.reset()
    p, _, act, why = gate.filter(0.0, [4.0, 1.5, 0.1], 0.0, body_pose([4.0, 1.5, 0.3], 0.0), True, False,
                                 band_check=False)                         # takeoff / landing
    assert act == "PASS" and abs(p[2] - 0.1) < 1e-9
    gate.reset()
    p, _, act, why = gate.filter(0.0, [4.0, 4.5, 1.0], 0.0, body_pose([2.0, 4.5, 1.0], 0.0), True, False)
    assert "max_step" in why and np.allclose(p, [2.6, 4.5, 1.0])
    gate.reset()
    body = body_pose([2.0, 4.5, 1.0], 0.0)
    gate.filter(0.0, [2.0, 4.5, 1.0], 0.0, body, True, False)
    p, _, act, why = gate.filter(0.05, [2.4, 4.5, 1.0], 0.0, body, True, False)
    assert "sp_rate" in why and abs(p[0] - (2.0 + cfg["max_sp_rate_mps"] * 0.05)) < 1e-9
    gate.reset()
    p, y, act, why = gate.filter(0.0, [2.1, 4.5, 1.0], 0.5, body, True, False)
    assert act == "PASS" and np.allclose(p, [2.1, 4.5, 1.0]) and abs(y - 0.5) < 1e-12


def test_safety_hold_latch():
    sm, _, cfg = world()
    gate = SafetyGate(cfg, sm)
    sp = [3.0, 4.5, 1.0]
    p, y, act, why = gate.filter(0.0, sp, 0.0, body_pose([2.5, 4.5, 1.0], 0.2), True, True)
    assert act == "HOLD" and why == "emergency" and np.allclose(p, [2.5, 4.5, 1.0]) and abs(y - 0.2) < 1e-9
    # the vehicle drifts during the hover: the hold stays latched (no creep)
    for t, drift in [(0.3, 0.1), (0.6, 0.2), (0.95, 0.25)]:
        p, y, act, _ = gate.filter(t, sp, 0.0, body_pose([2.5 + drift, 4.5, 1.0], 0.2 + drift), True, False)
        assert act == "HOLD" and np.allclose(p, [2.5, 4.5, 1.0]) and abs(y - 0.2) < 1e-9
    p, y, act, _ = gate.filter(1.05, sp, 0.0, body_pose([2.75, 4.5, 1.0], 0.0), True, False)
    assert act in ("PASS", "CLAMP") and cfg["emergency_hover_s"] == 1.0
    # a new emergency re-latches at the current pose; align invalid and no plan hold the same way
    gate.reset()
    p, _, act, why = gate.filter(0.0, sp, 0.0, body_pose([2.0, 4.0, 1.0], 0.0), False, False)
    assert act == "HOLD" and why == "align_invalid" and np.allclose(p, [2.0, 4.0, 1.0])
    p, _, act, _ = gate.filter(0.5, sp, 0.0, body_pose([2.3, 4.0, 1.0], 0.0), False, False)
    assert act == "HOLD" and np.allclose(p, [2.0, 4.0, 1.0])
    p, _, act, _ = gate.filter(0.55, sp, 0.0, body_pose([2.3, 4.0, 1.0], 0.0), True, False)
    assert act != "HOLD"
    gate.reset()
    p, _, act, why = gate.filter(0.0, None, 0.0, body_pose([2.0, 4.0, 1.0], 0.0), True, False)
    assert act == "HOLD" and why == "no_plan"


# ------------------------------------------------------------------------------------------------------------------


def main() -> int:
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    t_all = time.perf_counter()
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"PASS {name} ({time.perf_counter() - t0:.2f} s)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {e!r}")
    print(f"{len(tests) - failed}/{len(tests)} passed in {time.perf_counter() - t_all:.1f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

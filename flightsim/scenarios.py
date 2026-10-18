"""The six closed-loop scenarios of docs/SCOUT_NAV.md §8 (+ pass bars, provisional).

Geometry (default room, flightsim/world.py, DN-72): takeoff pose A = (1.0, 4.0) facing +x; goal wall point
B1 = (8.0, 3.6, 1.2) on wall_xL, so the hover goal is ~(7.2, 3.6, 1.2) and the straight line A->goal passes 0.6 m
north of the mapped pillar (3.8..4.2, 2.8..3.2) and south of crate_c: with no unmapped obstacle the direct path is
free. Unmapped obstacles sit ON that line between the pillar and crate_c (x ~ 4.4..5.8).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scenario:
    name: str
    description: str
    goal: str = "B1"                                    # wall point name (world.cfg wall_points)
    obstacles: list = field(default_factory=list)       # [(name, lo, hi)] unmapped, present from the start
    late_obstacle: dict | None = None                   # appears mid-flight ahead of the vehicle (see loop.py)
    reset_at_progress_m: float | None = None            # local-frame reset when this far along A->goal (NAV)
    takeoff_offset: tuple = (0.0, 0.0, 0.0)             # dx, dy (map m), dyaw (deg): TRUE placement vs the fixed A
    vehicle: dict = field(default_factory=dict)         # SimVehicle cfg overrides
    mission: dict = field(default_factory=dict)         # Mission cfg overrides (on top of scout_mission.yaml)
    loop: dict = field(default_factory=dict)            # loop cfg overrides (e.g. stop_at: done)
    timeout_s: float = 150.0
    expect: dict = field(default_factory=dict)          # pass bars, see metrics.evaluate

    def world_obstacles(self) -> list:
        obs = [tuple(o) for o in self.obstacles]
        if self.late_obstacle:
            sz = np.asarray(self.late_obstacle["size"], float)
            obs.append((self.late_obstacle["name"], (0.0, 0.0, 0.0), tuple(sz), False))
        return obs


ARRIVE = {"outcome": "arrived", "max_pos_err_m": 0.15, "max_yaw_err_deg": 10.0, "no_collision": True}

STRONG_DRIFT = {"estimator": {"bias_sigma_mps": 0.05, "bias_tau_s": 20.0, "vel_scale_sigma": 0.06,
                              "yaw_rate_bias_sigma_dps": 0.05, "yaw_rw_dps_sqrt_s": 0.02,
                              "hgt_step_tau_s": 1.5, "reset_delta": [0.3, -0.2, 0.0, 5.0]}}


def _box(cx, cy, sx, sy, sz):
    return (cx - sx / 2, cy - sy / 2, 0.0), (cx + sx / 2, cy + sy / 2, sz)


SCENARIOS = {
    "clear": Scenario(
        "clear", "no unmapped obstacle: arrives with zero emergencies and zero obstacles (false-alarm test)",
        expect=dict(ARRIVE, max_obstacles=0, max_emergencies=0)),
    "low_box": Scenario(
        "low_box", "a wide low box (1.4 x 1.4 x 0.7 m) on the path, room to pass beside it: over OR around, whichever "
                   "the planner's cost prefers (in this room around is shorter), and arrives",
        obstacles=[("low_box",) + _box(5.1, 3.74, 1.4, 1.4, 0.7)],
        expect=dict(ARRIVE)),
    "low_barrier": Scenario(
        "low_barrier", "a low barrier (0.8 m deep, 0.6 m tall) from wall to wall across the room: the only way is "
                       "OVER it",
        obstacles=[("low_barrier", (4.6, 0.02, 0.0), (5.4, 5.98, 0.6))],
        expect=dict(ARRIVE, shape_contains="over", geometric={"low_barrier": "over"})),
    "tall_box": Scenario(
        "tall_box", "a narrow tall box (0.5 x 0.5 x 1.9 m, person-sized) on the path: flies AROUND it",
        obstacles=[("tall_box",) + _box(5.1, 3.735, 0.5, 0.5, 1.9)],
        expect=dict(ARRIVE, shape_contains="around", geometric={"tall_box": "around"})),
    "late_obstacle": Scenario(
        "late_obstacle", "a 0.6 x 0.6 x 2.0 m box appears ~1 m ahead mid-flight: emergency hover, replan, arrives",
        late_obstacle={"name": "late_box", "size": [0.6, 0.6, 2.0], "ahead_m": 1.0, "trigger_progress_m": 3.2,
                       "min_speed_mps": 0.2, "clear_margin_m": 0.25},
        expect=dict(ARRIVE, min_emergencies=1)),
    "drift": Scenario(
        "drift", "strong odom drift (bias, scale, yaw) + a local-frame reset + a slow (1.5 s) height-step lag over "
                 "the low box: still arrives (the alignment works)",
        obstacles=[("low_box",) + _box(5.1, 3.74, 1.4, 1.4, 0.7)],
        reset_at_progress_m=2.0, vehicle=STRONG_DRIFT,
        expect=dict(ARRIVE)),
    "round_trip": Scenario(
        "round_trip", "take off from the raised pad box, fly to B, dwell, fly back and LAND on the pad box",
        mission={"on_arrival": "return_land", "dwell_s": 4.0}, loop={"stop_at": "done"}, timeout_s=200.0,
        expect=dict(ARRIVE, final_state="DONE", landed_on_pad=True, max_pad_xy_err_m=0.15)),
    "hover_test": Scenario(
        "hover_test", "hover-test mode: take off from the raised pad, rise 2 m, hold the same spot 10 s, land on the pad",
        mission={"mode": "hover_test", "goal": {"point_b": None}}, loop={"stop_at": "done"}, timeout_s=90.0,
        expect={"outcome": "done", "final_state": "DONE", "no_collision": True, "landed_on_pad": True,
                "max_pad_xy_err_m": 0.15, "min_hover_s": 10.0, "max_hover_xy_dev_m": 0.15, "max_hover_z_err_m": 0.10,
                "max_obstacles": 0, "max_emergencies": 0}),
    "wrong_takeoff": Scenario(
        "wrong_takeoff", "the Scout sits 0.5 m / 20 deg off the fixed takeoff pose: REFUSED, never arms",
        takeoff_offset=(0.0, 0.5, 20.0), timeout_s=40.0,
        expect={"outcome": "refused", "never_armed": True, "no_collision": True}),
}

ORDER = ["clear", "low_box", "low_barrier", "tall_box", "late_obstacle", "drift", "round_trip",
         "hover_test", "wrong_takeoff"]


def get(name) -> Scenario:
    if isinstance(name, Scenario):
        return name
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; known: {', '.join(ORDER)}")
    return SCENARIOS[name]

"""Print the synthetic relocalization results table (docs/SCOUT_LOC.md) as markdown + JSON.

    py -3.9 tools/eval_scout_loc_synthetic.py [--json out.json]

Synthetic only (localization/synthetic_room.py): proves the logic and shows sensitivities; it is not a real-world
accuracy figure. CPU-heavy for a few minutes (renders ~1000 frames); make sure no simulator is running on this laptop first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from localization import synthetic_eval as ev  # noqa: E402
from localization import synthetic_room as sr  # noqa: E402


def _row(name, s, note=""):
    return (f"| {name} | {s['fixes']}/{s['frames']} | {s['pos_med_m'] * 100:.1f} / {s['pos_p95_m'] * 100:.1f} / "
            f"{s['pos_max_m'] * 100:.1f} | {s['rot_med_deg']:.2f} / {s['rot_p95_deg']:.2f} / {s['rot_max_deg']:.2f} | "
            f"{s['inl_med']} / {s['inl_min']} | {s['ms_med']:.0f} | {note} |")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    t0 = time.time()
    ds = sr.build_dataset(0)
    res = {"dataset": {"keyframes": ds.db.n_keyframes, "landmarks": len(ds.db), "map_points": int(len(ds.map_xyz)),
                       "camera": ds.cam.to_dict(), "build_s": round(time.time() - t0, 1)},
           "landmark_depth": ev.landmark_depth_error(ds)}
    rows = []
    s = ev.summarize(ev.tracking(ds))
    res["tracking"] = s
    rows.append(_row("Tracking, held-out flight (15 % darker)", s))
    s = ev.summarize(ev.kidnapped(ds))
    res["global"] = s
    rows.append(_row("Global, no prior (every 5th frame alone)", s))
    for rot, tr in ((0.5, 0.01), (1.0, 0.02), (2.0, 0.05)):
        db = ev.rebuild_db(ds, extrinsic_error=ev.extrinsic_perturbation(rot, tr))
        s = ev.summarize(ev.tracking(ds, db=db))
        res[f"extrinsic_{rot}deg_{tr}m"] = s
        rows.append(_row(f"Lidar-camera calibration off by {rot} deg + {tr * 100:.0f} cm", s,
                         "map built with the wrong extrinsics"))
    rng = np.random.default_rng(5)
    noisy = ds.map_xyz + rng.normal(0, 0.02, ds.map_xyz.shape).astype(np.float32)   # 1 cm -> ~2.2 cm total
    s = ev.summarize(ev.tracking(ds, db=ev.rebuild_db(ds, map_xyz=noisy)))
    res["map_noise_2.2cm"] = s
    rows.append(_row("Map noise 1 cm -> ~2.2 cm", s))
    recs, (h0, h1) = ev.blank_wall(ds)
    last = max(r["stamp"] for r in recs if r["fresh"] and r["stamp"] < h1)
    first_false = next(r["stamp"] for r in recs if r["stamp"] > 1.0 and not r["loc_ok"])
    back = next(r["stamp"] for r in recs if r["stamp"] >= h1 and r["loc_ok"])
    res["blank_wall"] = {"hold_s": [h0, h1], "last_fix_s": last, "loc_ok_false_at_s": first_false,
                         "loc_ok_back_at_s": back, "fixes_during_hold": sum(r["fresh"] for r in recs
                                                                            if h0 <= r["stamp"] < h1)}
    recs_t, _ = ev.blank_wall(ds, with_tag=True)
    hold = [r for r in recs_t if h0 <= r["stamp"] < h1]
    res["blank_wall_with_tag"] = {"loc_ok_all_hold": all(r["loc_ok"] for r in hold),
                                  "pos_max_m": max(r.get("pos_err_m", np.nan) for r in hold),
                                  "rot_max_deg": max(r.get("rot_err_deg", np.nan) for r in hold)}
    s = ev.summarize(ev.tag_only(ds))
    res["tags_only_flight"] = s
    rows.append(_row("AprilTag only, 0.30 m tags (1 per wall), >= 40 px", s, "single-tag fixes"))

    print(f"Dataset: {res['dataset']['keyframes']} keyframes, {res['dataset']['landmarks']} landmarks, "
          f"{res['dataset']['map_points']} map points; landmark depth error median "
          f"{res['landmark_depth']['rel_med'] * 100:.2f} %, >5 % off: {res['landmark_depth']['frac_over_5pct'] * 100:.2f} %")
    print()
    print("| Case | Fixes | Position err cm (median / p95 / max) | Rotation err deg (median / p95 / max) | "
          "Inliers (median / min) | ms/frame | Note |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(r)
    b = res["blank_wall"]
    print()
    print(f"Texture-less wall: facing it {b['hold_s'][0]:.1f}-{b['hold_s'][1]:.1f} s; last fix {b['last_fix_s']:.1f} s; "
          f"loc_ok false at {b['loc_ok_false_at_s']:.1f} s; fixes while facing it: {b['fixes_during_hold']}; "
          f"loc_ok back at {b['loc_ok_back_at_s']:.1f} s.")
    bt = res["blank_wall_with_tag"]
    print(f"Same wall with a 0.20 m tag on it: loc_ok held through: {bt['loc_ok_all_hold']}; worst error "
          f"{bt['pos_max_m'] * 100:.1f} cm / {bt['rot_max_deg']:.2f} deg.")
    print(f"(total {time.time() - t0:.0f} s)")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main()

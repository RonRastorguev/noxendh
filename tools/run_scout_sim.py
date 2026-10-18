"""Run the Scout closed-loop flight sim (flightsim/, docs/SCOUT_NAV.md §8).

    py -3.9 tools/run_scout_sim.py --scenario all --seeds 0,1,2 --out data/sim_runs
    py -3.9 tools/run_scout_sim.py --scenario low_box --seeds 0

Prints a JSON summary. Exit 0 = every run passed, 1 = a run failed its bars, 2 = could not run (a flight-software
piece missing, a crash in the software under test, bad arguments).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="all", help="scenario name, comma list, or 'all'")
    ap.add_argument("--seeds", default="0", help="comma list of seeds")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "sim_runs"))
    ap.add_argument("--config", default=None, help="sim config YAML (default config/scout_sim.yaml)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        from flightsim import loop, scenarios
        names = scenarios.ORDER if a.scenario == "all" else [s.strip() for s in a.scenario.split(",") if s.strip()]
        for n in names:
            scenarios.get(n)
        seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
        cfg = loop.load_sim_config(a.config)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"status": "could_not_run", "error": repr(e)}))
        return 2
    out = {"status": "ok", "runs": [], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rc = 0
    for n in names:
        for s in seeds:
            try:
                m = loop.run_scenario(n, s, a.out, sim_cfg=cfg)
            except loop.ComponentsMissing as e:
                print(json.dumps({"status": "could_not_run", "error": str(e)}))
                return 2
            except Exception as e:  # noqa: BLE001 - the software under test raised: cannot evaluate
                out["runs"].append({"scenario": n, "seed": s, "passed": False, "error": repr(e),
                                    "traceback": traceback.format_exc(limit=6)})
                rc = 2
                continue
            r = {"scenario": n, "seed": s, "passed": m["passed"], "outcome": m["outcome"],
                 "final_state": m["final_state"], "reason": m["reason"],
                 "failed_checks": [c for c in m["checks"] if not c["ok"]],
                 "arrival": m["arrival"], "landing": m.get("landing"), "hover": m.get("hover"), "collision": m["collision"],
                 "min_clearance_m": m["min_clearance_m"],
                 "emergencies": m["emergencies"], "obstacles_created": m["obstacles_created"],
                 "obstacles_false": m["obstacles_false"], "plan_shapes": m["plan_shapes"],
                 "geometric": m["geometric"], "max_speed_xy_mps": m["max_speed_xy_mps"],
                 "align_xy_err_max_m": m["alignment"]["xy_err_m"]["max"],
                 "sim_s": m["timing"]["sim_s"], "wall_s": m["timing"]["wall_s"],
                 "wall_per_sim_s": m["timing"].get("wall_per_sim_s"), "run_dir": m.get("run_dir")}
            out["runs"].append(r)
            if not a.quiet:
                print(f"[{n} seed {s}] {'PASS' if m['passed'] else 'FAIL'} {m['outcome']} "
                      f"({m['timing']['sim_s']:.0f} sim s, {m['timing']['wall_s']:.0f} wall s)", file=sys.stderr)
            if not m["passed"] and rc == 0:
                rc = 1
    out["passed"] = sum(1 for r in out["runs"] if r.get("passed"))
    out["total"] = len(out["runs"])
    print(json.dumps(out, indent=1, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())

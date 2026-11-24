#!/usr/bin/env python
"""CHLA-Z glider pilot driver (not part of a release). Resumable: rerun any stage.

    python glider_pilot.py candidates [--refresh]   # ERDDAP search -> glider/candidates.parquet
    python glider_pilot.py screen                   # one-day sensor check per shortlisted deployment
    python glider_pilot.py fetch                    # selected deployments, CHLA 0-200 m, per month
    python glider_pilot.py fetch-deep               # CHLA below 300 m, for the dark offset
    python glider_pilot.py windows                  # one profile per 3 h solar window
    python glider_pilot.py combine                  # -> glider/CHLA_glider_profiles.parquet
    python glider_pilot.py pace --product Rrs       # also chlor_a, Kd; only granules on glider days
    python glider_pilot.py olci
    python glider_pilot.py merge                    # -> ..._plus_PACE / _plus_OLCI, both time rules

The selected deployments are in src/chla_z_glider_selection.json. Outputs go
to notebooks/_temp_data/glider/. Design note: claude/notes/glider-pilot.md.
Network units run in threads (the work is waiting on ERDDAP / Earthdata);
failed units are listed and the exit status is non-zero; rerun to retry.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import chla_z_build as b  # noqa: E402
import chla_z_glider as g  # noqa: E402


def run(fn, units, workers):
    failed = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, *u): u for u in units}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                msg = f.result()
                if not str(msg).startswith("skip"):
                    b.log(f"[{i}/{len(units)}] {msg}")
            except Exception as e:
                failed[futs[f]] = f"{type(e).__name__}: {e}"
                b.log(f"[{i}/{len(units)}] FAILED {futs[f]}: {str(e)[:300]}")
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--product", default="Rrs")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    failed = {}
    if a.stage == "candidates":
        c = g.candidates(refresh=a.refresh)
        s = g.shortlist(c)
        print(f"{len(c)} datasets; shortlist {len(s)} deployments")
        print(s.groupby("region").size().to_string())
    elif a.stage == "screen":
        s = g.shortlist(g.candidates())
        failed = run(g.screen_unit, [(d,) for d in s.datasetID], a.workers)
    elif a.stage == "fetch":
        units = [(d, m) for d in g.selection() for m in g.months_for(d)]
        failed = run(g.fetch_unit, units, a.workers)
    elif a.stage == "fetch-deep":
        units = [(d, m) for d in g.selection() for m in g.months_for(d)]
        failed = run(g.fetch_deep_unit, units, a.workers)
    elif a.stage == "windows":
        for d in g.selection():
            b.log(g.window_unit(d, force=a.force))
    elif a.stage == "combine":
        _, info = g.combine()
        print(json.dumps(info, indent=1))
    elif a.stage == "pace":
        units = [(a.product, sn, gr) for sn, gr in g.pace_granules(a.product)]
        failed = run(g.pace_unit, units, a.workers)
    elif a.stage == "olci":
        failed = run(g.olci_unit, [(d,) for d in g.olci_days()], a.workers)
    elif a.stage == "merge":
        print(json.dumps(g.merge(), indent=1, default=str))
    else:
        ap.error(f"unknown stage {a.stage}")
    if failed:
        print(f"{len(failed)} failed units; rerun to retry")
        sys.exit(1)


if __name__ == "__main__":
    main()

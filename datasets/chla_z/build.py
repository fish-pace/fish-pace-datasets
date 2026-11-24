#!/usr/bin/env python
"""Build the CHLA-Z training tables. Resumable: rerun any stage after a crash.

    python build.py argo-fetch           # Ifremer ERDDAP -> _temp_data/raw/ tiles -> monthly .nc
    python build.py argo                 # monthly .nc -> data/CHLA_argo_profiles.parquet
    python build.py ooi-fetch            # OOI ERDDAP -> _temp_data/ooi_raw/<id>/<YYYY-MM>.csv.gz
    python build.py ooi                  # raw -> data/CHLA_ooi_profiles.parquet (Sauzède corrected)
    python build.py glider-select        # Glider DAC search + one-day sensor screen (to choose deployments)
    python build.py glider-scan          # all DAC chlorophyll deployments, whole-record check, status vs the table
                                         # -> src/chla_z_glider_candidates.csv (issue #16)
    python build.py glider-fetch         # the deployments in src/chla_z_glider_selection.json, 0-200 m and below 300 m
    python build.py glider               # raw -> data/CHLA_glider_profiles.parquet
    python build.py pace --product Rrs   # one product's granule matchups (also chlor_a, Kd);
                                         # units are (platform, granule), --platform to limit
    python build.py pace-merge           # -> data/CHLA_{argo,ooi,glider}_profiles_plus_PACE.parquet
    python build.py olci                 # OLCI S3A/S3B daily matchups, one unit per day
    python build.py olci-merge           # -> data/CHLA_{argo,ooi,glider}_profiles_plus_OLCI.parquet

--platform argo|ooi|glider limits pace, pace-merge, olci and olci-merge to one
platform; the merges then rewrite only that platform's table. Glider: the
selection is a committed list; glider-select only helps choose a new one
(design note claude/notes/chla-z-glider.md).

Each unit runs in its own process (`max_tasks_per_child=1`), so memory is
returned between units, and an OOM-killed worker only loses its unit: the pool
is rebuilt and the remaining units continue. Failed units are listed at the end
and the exit status is non-zero; rerun the same command to retry them.
The released Argo monthly .nc files were fetched on 2026-11-06 by the code
that is now `argo-fetch` (then in argopy.ipynb Workflow 2). `argo-fetch` skips
tiles and months that already exist, so it does not refetch them; delete
_temp_data/raw/ and CHLA_argo_monthly_nc/ for a new snapshot. Use few workers
for it (2): the ERDDAP server drops requests under load.
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

import chla_z_build as b  # noqa: E402


def _call(fn_name, args):
    import chla_z_build as bb
    return getattr(bb, fn_name)(*args)


def run_units(fn_name, units, workers, rounds=3):
    """Run fn(*unit) for each unit in a fresh process; return failed units."""
    todo = list(units)
    failed = {}
    for r in range(rounds):
        if not todo:
            break
        b.log(f"{fn_name}: round {r + 1}, {len(todo)} units, {workers} workers")
        failed = {}
        done = set()
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                     max_tasks_per_child=1) as ex:
                futs = {ex.submit(_call, fn_name, u): u for u in todo}
                for i, fut in enumerate(as_completed(futs), 1):
                    u = futs[fut]
                    try:
                        msg = fut.result()
                        done.add(u)
                        if not str(msg).startswith("skip"):
                            b.log(f"[{i}/{len(todo)}] {msg}")
                    except BrokenProcessPool:
                        raise
                    except Exception as e:
                        failed[u] = f"{type(e).__name__}: {e}"
                        b.log(f"[{i}/{len(todo)}] FAILED {u}: {failed[u][:300]}")
        except BrokenProcessPool:
            b.log("worker died (OOM kill?); rebuilding the pool for the remaining units")
        todo = [u for u in todo if u not in done]
        if todo and r < rounds - 1:
            time.sleep(60)
    for u in todo:
        failed.setdefault(u, "not completed")
    return failed


def report_failures(stage, failed):
    path = b.TEMP_DIR / "logs" / f"build_{stage}_failed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({" ".join(map(str, k)): v for k, v in failed.items()}, indent=1))
    if failed:
        b.log(f"{len(failed)} {stage} units failed; see {path}. Rerun the same command.")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["argo-fetch", "argo", "ooi-fetch", "ooi", "glider-select",
                                      "glider-scan", "glider-fetch", "glider", "pace", "pace-merge", "olci", "olci-merge"])
    ap.add_argument("--product", choices=list(b.PACE_PRODUCTS))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, help="olci: only the first N days (for testing)")
    ap.add_argument("--force", action="store_true",
                    help="argo-fetch/argo/ooi: rebuild per-unit outputs even if they exist (raw downloads are kept)")
    ap.add_argument("--platform", choices=list(b.PLATFORMS),
                    help="pace/olci and their merges: only this platform (default: all)")
    a = ap.parse_args()
    platforms = [a.platform] if a.platform else list(b.PLATFORMS)
    import chla_z_tables as t

    if a.stage == "argo-fetch":
        tiles = [(m, lo, la) for m in b.MONTHS for lo, la in b.ARGO_TILES]
        failed = run_units("argo_tile_unit", tiles, a.workers)
        report_failures("argo-fetch", failed)
        failed = run_units("argo_merge_month_unit", [(m, a.force) for m in b.MONTHS], a.workers)
        report_failures("argo-merge", failed)
    elif a.stage == "argo":
        failed = run_units("argo_month_unit", [(m, a.force) for m in b.MONTHS], a.workers)
        report_failures("argo", failed)
        t.write_argo_profiles()
    elif a.stage == "ooi-fetch":
        ids = list(b.ooi_datasets())
        units = []
        for i in ids:  # coverage lookups are one small request each
            units += [(i, m) for m in b.ooi_months_for(i)]
        b.log(f"{len(ids)} datasets, {len(units)} dataset-months to fetch")
        failed = run_units("ooi_month_unit", units, a.workers)
        report_failures("ooi-fetch", failed)
    elif a.stage == "ooi":
        failed = run_units("ooi_dataset_unit", [(i, a.force) for i in b.ooi_datasets()], a.workers)
        report_failures("ooi", failed)
        t.write_ooi_profiles()
    elif a.stage == "glider-scan":
        import chla_z_glider_scan as gs
        d = gs.scan(refresh=True, workers=a.workers)
        gs.write_candidates(d)
        print(d.groupby(["status", "verdict"]).size().to_string())
    elif a.stage in ("glider-select", "glider-fetch", "glider"):
        import chla_z_glider as g
        # Network units are I/O-bound and small: threads, not processes.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def run_threads(fn, units):
            failed = {}
            with ThreadPoolExecutor(max_workers=a.workers) as ex:
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
        if a.stage == "glider-select":
            short = g.shortlist(g.candidates(refresh=a.force))
            report_failures("glider-screen", run_threads(g.screen_unit, [(d,) for d in short.datasetID]))
            print(short.merge(g.screen_table(), on="datasetID")
                  .groupby(["region", "verdict"]).size().to_string())
        elif a.stage == "glider-fetch":
            units = [(d, m) for d in g.selection() for m in g.months_for(d)]
            report_failures("glider-fetch", run_threads(g.fetch_unit, units))
            report_failures("glider-fetch-deep", run_threads(g.fetch_deep_unit, units))
        else:
            for d in g.selection():
                b.log(g.window_unit(d, force=a.force))
            t.write_glider_profiles()
    elif a.stage == "pace":
        if not a.product:
            ap.error("--product is required for pace")
        gl = b.pace_granule_list(a.product)
        units = [(a.product, g["short_name"], g["granule"], p) for p in platforms for g in gl]
        failed = run_units("pace_granule_unit", units, a.workers)
        report_failures(f"pace-{a.product}", failed)
    elif a.stage == "pace-merge":
        t.write_plus_pace(platforms)
    elif a.stage == "olci":
        b.olci_granule_index()
        units = []
        for p in platforms:
            days = b.olci_days(p)
            units += [(d, p) for d in (days[: a.limit] if a.limit else days)]
        failed = run_units("olci_day_unit", units, a.workers)
        report_failures("olci", failed)
    elif a.stage == "olci-merge":
        t.write_plus_olci(platforms)


if __name__ == "__main__":
    main()

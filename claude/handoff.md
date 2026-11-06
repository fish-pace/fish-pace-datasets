# Handoff — fish-pace-datasets

Read this first. Orientation, not a task list: Eli says what to work on.

## Repo state (2026-11-06, ~22:30 UTC)

- Branch `chla-z-rebuild`, commits `f43cf48` (fetch fixes, `requirements.txt`,
  GDAC note) and the one adding this handoff. **Not pushed.** No PR yet.
- Work is the CHLA-Z training-table rebuild, tracked in fish-pace/fish-pace-datasets#4.
  Poster Methods text lives in fish-pace/2026-go-bgc-hackweek#8.

## What shipped this session

- **Argo monthly files rebuilt** from Ifremer ERDDAP (snapshot 2026-11-06, argopy
  1.5.0, expert mode), 2024-03 to 2026-10, 32 months, all passed the merge check.
  In `datasets/chla_z/notebooks/_temp_data/CHLA_argo_monthly_nc/` (7.8 GB, shared-public,
  git-ignored) with `argo_profiles_index.parquet` (95,963 rows). August files kept in
  `CHLA_argo_monthly_nc_2026-08-20/`. Comparison table: `claude/notes/argo-gdac-vs-erddap.md`.
- GDAC-on-AWS was tried and **dropped** (Eli's decision); the note records why.
- Methods comment on the descending/ascending merge posted to hackweek#8.

## Open question Eli is thinking about: descending vs ascending profiles

Eli is worried about this and may want to **restrict Argo to ascending profiles**.
She recalls something about ascent being the better data; that is not verified yet.

Facts (from the rebuilt monthly files):
- Pipeline keys profiles on (`PLATFORM_NUMBER`, `CYCLE_NUMBER`), not `DIRECTION`.
  Cell 47 pools a cycle's D and A `CHLA_ADJUSTED` points before 10 m binning and
  takes TIME/LAT/LON from the first record = the descent in 99.9% of cases. Cell 42
  (index) does the same over the whole record. Monthly .nc files keep `DIRECTION`.
- 5,036 descending profiles (194 floats) vs 95,925 ascending. 4,998 cycles have both;
  descent is a median 4.95 days (10–90%: 1–10 d) and 10 km (90%: ≤50 km) from ascent.
- **Only 610 of 5,036 descending profiles (12%) have any good `CHLA_ADJUSTED` level
  (QC 1/2/5/8), vs 59,584 of 95,925 ascending (62%).** That fits "ascent is better"
  but is not a reason on its own.
- So for up to ~5,000 cycles the PACE match-up (24 h window) uses the descent's time
  and position while the good CHLA comes from the ascent.

Options to discuss (not decided): (a) keep only `DIRECTION == "A"` before binning
and indexing (simplest; loses ~600 cycles' descent levels, ~0.6% of the table);
(b) key on direction and keep D as separate profiles; (c) keep as is, just apply QC
first. Worth checking Argo / BGC-Argo documentation on descending profiles before
choosing.

## Known bug to fix next (Argo)

Cell 47 bins `df`, not `df_qc`: **the QC filter on {1, 2, 5, 8} is not applied**,
though hackweek#8's earlier Methods comment says it is. Fixing it (and the D/A
choice above) comes before running cells 47–48 to build `data/CHLA_argo_profiles.parquet`.

## Downstream, in order (Eli: OOI re-fetch is separate from PACE matchups)

1. Argo: D/A decision, QC fix in cell 47 (and cell 44 example), build the profile table.
2. Decide file names (overwrite vs new; old ones reachable via tag `2025_11_01`) and
   where the 7.8 GB monthly files live, if released at all.
3. `argo-matchups.ipynb` cell 2 still reads the index from fish-pace/2025-tutorials.
4. OOI (separately): rerun `ooi.ipynb` to 2026-11-01; save `df_dedup`, assert
   `profile_id` unique.
5. PACE matchups (later, both platforms): V3.2 collections `PACE_OCI_L3M_AOP`, `_BGC`,
   `_KD` (+ `_NRT`); old ones are gone from CMR.
6. Release: before/after table, README + STAC text (still says "no QC"), snapshot date
   and argopy version, PR to `main`.

## Working notes

- Venv: `~/.venvs/chla-z` (from `datasets/chla_z/requirements.txt`). Runner, lane
  script and logs: `~/.venvs/chla-z/run/`. `/tmp` is wiped on hub restart.
- Hub memory cap 7.8 GB; an OOM kill takes sibling processes too. Run one Argo month
  per process; don't run memory-heavy tests beside a fetch. Details in
  `claude/notes/argo-gdac-vs-erddap.md`.
- Install with `pip --no-cache-dir`; home fills up (pip/Claude caches).

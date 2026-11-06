# Argo BGC source: ERDDAP vs GDAC Sprof on AWS S3

**Decision (Eli, 2026-11-06 21:05 UTC): GDAC is dropped; the rebuild stays on
ERDDAP.** GDAC was only a fallback for when ERDDAP looked like a 90-hour run.
Once each worker had its own fresh process per month, ERDDAP ran at 3–6 tiles/min.
The rest of this note is kept as a record of what was tried, in case ERDDAP fails
again.

Status as of 2026-11-06. The CHLA-Z rebuild (branch `chla-z-rebuild`) fetches
BGC-Argo through argopy `src="erddap"` (Ifremer synthetic-BGC ERDDAP). Ifremer
is slow and drops connections, so Eli wants to test the GDAC Sprof files on the
official AWS mirror, `s3://argo-gdac-sandbox/pub`, as an alternative. Decision so
far (Eli, 2026-11-06): keep the ERDDAP run going; test GDAC on one month before
changing the method. Switching sources is a method change and must be recorded in
the README / STAC items / release notes.

## Why GDAC at all

- Same synthetic profiles ERDDAP serves; the official source.
- Charlie Turner's hackweek rebuild from Sprof raised CHLA level coverage from
  62% to 98% (slide 9; see issue #4). Most of that gain was the argopy
  standard-mode QC-8 trap, which `mode="expert"` already fixes on ERDDAP.
- Charlie's code is **not** in `fish-pace/2026-go-bgc-hackweek`
  (`contributor_folders/charlie` has only the OOI duplicate-rows walkthrough).
  His README says the Argo numbers came from "the `drift` project's experiments
  26–28", which is not in the fish-pace org and not on this hub. Ask Charlie for it.

## Rebuild result (2026-11-06, finished 21:52 UTC)

All 32 months (2024-03 to 2026-10) fetched from Ifremer ERDDAP on 2026-11-06,
argopy 1.5.0 expert mode; merged into `_temp_data/CHLA_argo_monthly_nc/`
(7.8 GB, every month passed the merge check) plus `argo_profiles_index.parquet`
(95,963 rows). August files kept in `CHLA_argo_monthly_nc_2026-08-20/`.

- Only tile ever missing: 60–120°E, 45–90°N (26 months; ERDDAP "no data").
- 2024 months match August almost exactly (same profiles, ~1.5% more points).
  From 2025-04 on, 2–4% more profiles than August (late-arriving data).
  2024-12 now has 2,796 profiles (August file was the all-zeros corrupt one);
  2026-08 now runs to 08-31 (August pull stopped at 08-20).
- **Index collapses A/D profiles (pre-existing, not from this rebuild).** Cell 42
  keys on (PLATFORM_NUMBER, CYCLE_NUMBER) with no DIRECTION, so 4,998 cycles that
  have both a descending and an ascending profile get one index row (TIME/LAT/LON
  of the earlier one). 862 of them span two months, which is why per-month unique
  float-cycles sum to 96,825 vs 95,963 index rows. Not fixed; Eli's call.

## ERDDAP baseline month: 2024-03

All 23 ERDDAP tiles for 2024-03 are copied into
`datasets/chla_z/notebooks/_temp_data/cmp_2024-03/erddap/_temp_data/raw/`
(`_temp_data` is shared-public, git-ignored). Merge them there with the runner's
`merge` stage to rebuild the monthly file. Numbers from the first merge (Nov 6):

| 2024-03 | August file | ERDDAP, Nov 6 |
|---|---|---|
| points | 447,426 | 455,030 |
| profiles / floats | 2,122 / 582 | 2,122 / 582 |
| levels with `CHLA_ADJUSTED_QC` in {1,2,5,8} | 173,657 | 173,386 |
| profiles with >=1 good level | 1,379 | 1,376 |
| `CHLA_DATA_MODE` A / R / D / empty | 263,714 / 3,612 / 1,053 / 179,047 | 263,864 / 3,612 / 365 / 187,189 |

- 99.2% of `CHLA_ADJUSTED` values identical between August and Nov on shared
  levels: the recalibration was already in the August pull for this month.
- ~35% of profiles (746 / 2,122) have no `CHLA_ADJUSTED` (empty data mode). The
  key GDAC question: does Sprof give more adjusted coverage?
- D-mode count fell 1,053 -> 365 between pulls; not explained yet.

## What happened trying argopy `src="gdac"` on S3 (2026-11-06)

Test: `get_bgc_file()` from `argopy.ipynb` cell 35 with `src="erddap"` swapped
for `src="gdac"` and `argopy.set_options(gdac="s3://argo-gdac-sandbox/pub")`,
one tile (`[-60, 0, 0, 45, 0, 200]`, 2024-03, expert mode). Problems, in order:

1. **Needs `boto3`**, which argopy treats as optional and `requirements.txt` does
   not list. `pip install boto3` pulled a botocore too new for s3fs's aiobotocore;
   pin both to the aiobotocore range (`boto3/botocore >=1.43.115,<1.43.118` for
   aiobotocore 3.10.0). Add to `requirements.txt` only if GDAC is adopted.
2. **"This host (s3://argo-gdac-sandbox/pub/idx) is not alive"** although the
   bucket lists fine anonymously. The hub runs under an AWS IRSA role
   (`AWS_ROLE_ARN`, `AWS_WEB_IDENTITY_TOKEN_FILE`), so with boto3 present argopy
   signs requests and they fail. Workaround: run with no credentials visible,
   `~/.venvs/chla-z/run/anon.sh` (unsets `AWS_*`, nulls config files, sets
   `AWS_EC2_METADATA_DISABLED=true`). Not in the repo; the notebook would need its
   own fix if adopted.
3. **`h5netcdf` without `h5py`**: recent h5netcdf no longer installs h5py, so
   opening Sprof failed. `h5py>=3.11` was added to `requirements.txt` (it is
   needed whenever h5netcdf is used, not just for GDAC).
4. **Out of memory.** The index search is fast (49 Sprof files in the tile; 610 MB
   compressed total, median 8 MB, max 59 MB). But argopy reads each float's
   *whole* Sprof (every cycle, level and parameter) and converts it to points
   before subsetting to the box. With argopy's default 6 threads it was
   OOM-killed after 6 min; with `max_workers=1` it still passed 4 GB and was
   stopped. Hub cap is 7.8 GB, shared with the ERDDAP workers. argopy also warns
   "BGC data support with the 'gdac' data source is still in Work In Progress".

Conclusion: argopy's GDAC fetcher is not usable for global tiles on this hub.

## GO-BGC 2026 data workshop (checked 2026-11-06)

`github.com/go-bgc/go-bgc-dataworkshop-2026.github.io` has no GDAC / Sprof / AWS
Argo code. Its one tutorial (`book/tutorials/machine-learning/argo_regression_modeling.ipynb`)
uses the same approach as ours: argopy `src="erddap"`, `ds="bgc"`, `mode="expert"`,
QC flags {1, 2, 5, 8}. It adds `parallel=True, chunks_maxsize={"time": 30}`, which
makes argopy split one box into several smaller ERDDAP requests, and it ships the
fetched data pre-downloaded with `rerun_Argopy = False` "to avoid JupyterHub crashing
with multiple users". Its S3 bucket (`s3://uw-escience-scratch-prod/`) is workshop
scratch space, not Argo data.

## Options if GDAC is still wanted

- **Own lazy Sprof reader** (most promising): use the S3 index
  (`ArgoIndex(host="s3")` or `argo_synthetic-profile_index.txt`) to find
  (float, cycle) in box+month, open each `*_Sprof.nc` lazily from S3, `isel` just
  those `N_PROF`, `PRES <= 200`, and only `CHLA`, `CHLA_ADJUSTED`, `*_QC`,
  `*_DATA_MODE`, `PRES`, position, time. Memory stays small. It replaces argopy's
  expert-mode processing, so it must be checked against the ERDDAP 2024-03
  baseline above before use.
- Get Charlie's `drift` code and reuse it.
- Run argopy GDAC on a bigger machine (not the hub).

## ERDDAP run status (for context)

Runner, logs, pids in `~/.venvs/chla-z/run/` (survives hub restarts; scratchpad
in `/tmp` does not). Venv `~/.venvs/chla-z` built from `requirements.txt`.
After the restart: ~50 tiles / 10 min at 19:50 UTC.

**Lesson (2026-11-06):** the first GDAC test (argopy's default 6 threads) hit
the hub's cgroup memory cap at ~20:11 and the OOM killer also took all 3 ERDDAP
workers (`oom_kill 4` in `/sys/fs/cgroup/memory.events`). It looked like a slow
server because the recorded PIDs were the wrapper subshells, which stayed as
zombies, so `ps -p` still "found" them. Now: never run a memory-heavy test next
to the workers; record the real python PIDs (`pgrep -f "run_argo_fetch.py.*tiles"`
into `~/.venvs/chla-z/run/workers.pids`); check `oom_kill` when a log just stops.
Workers restarted at 20:38 UTC.

**Memory grows within a worker (2026-11-06).** After the 20:38 restart, ERDDAP
ran at 3–6 tiles/min, but by 21:00 two workers had been OOM-killed and the third
was at 6.7 GB of the 7.8 GB cap after ~3 months in one process: memory
accumulates across months (argopy/xarray, cause not pinned down). Fix: `run/lane.sh`
runs **one python process per month**, retries a month up to 3 times on a
non-zero exit, and logs `exit N` plus the runner's `peak RSS MB`. Started 21:03 UTC
as 3 lanes over the 11 remaining months. Zombie PIDs fooled `ps -p` a second time;
check with `pgrep` and process state, not saved PIDs.

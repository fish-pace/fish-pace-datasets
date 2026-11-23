# Handoff — fish-pace-datasets

Read this first. Orientation, not a task list: Eli says what to work on.

## Repo state (2026-11-23)

- Branch `chla-z-rebuild`: the CHLA-Z training-table rebuild
  (fish-pace/fish-pace-datasets#4) is built; PR #9 is open for Eli's review.
  All four `datasets/chla_z/data/` tables are overwritten in place; the old
  ones are at tag `2025_11_01`. No release cut (Eli does that). Two new tables,
  `CHLA_{argo,ooi}_profiles_plus_OLCI.parquet` (2026-11-23): OLCI for every
  profile, flagged by PACE match.
- Decisions, deliberate oddities, run log, before/after tables:
  [notes/chla-z-rebuild-2026-11.md](notes/chla-z-rebuild-2026-11.md).
  Argo source comparison (GDAC vs ERDDAP, why ERDDAP):
  [notes/argo-gdac-vs-erddap.md](notes/argo-gdac-vs-erddap.md).
- Downstream: fish-pace/2026-go-bgc-hackweek branch `rebuilt-training-data`
  reads these tables with `DATA_RELEASE = "chla-z-rebuild"` (this branch name);
  switch it to the release tag once the release exists.

## Changed after review (Eli, 2026-11-23)

- OOI bins are now (A, B], like Argo.
- OOI rows are 3 h windows of local mean solar time (95,770 rows), not UTC
  days, so `solar_hour` varies within a site. Matchups are now kept per
  platform, so one platform can be re-matched alone.
- October 2026 PACE matchups are NRT only (standard V3.2 ends 2026-09-30);
  Eli is fine with that, and the Methods say so.
- A train/test split on the OOI table must be grouped by instrument: several
  windows of a day share one PACE/OLCI pixel.

## How to rebuild

`datasets/chla_z/build.py` (stages `argo`, `ooi-fetch`, `ooi`, `pace --product`,
`pace-merge`, `olci`, `olci-merge`; `--platform argo|ooi` for the matchup
stages); every stage resumes. Code in `src/chla_z_build.py` and
`src/chla_z_tables.py`; the notebooks call the same functions.

## Environment

- **`~` has a quota; `~/shared-public` does not** (ENOSPC while `df` shows free
  space). Venv: `~/shared-public/venvs/chla-z`, built with
  `/srv/conda/envs/notebook/bin/python3.13 -m venv` from `requirements.txt`.
- `notebooks/_temp_data` is a symlink into shared-public (git-ignored): Argo
  monthly .nc (7.8 GB, ERDDAP snapshot 2026-11-06), OOI raw CSVs, PACE
  per-granule matchups (`matchups_v3_2/`), logs.
- The run used a 16 GB / 4-core hub. Rrs matchups are CPU-bound (~19 h).
- **Clean up `~/shared-public` after the poster work** (Eli): keep what the
  poster needs. Not before.

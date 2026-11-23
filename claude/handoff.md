# Handoff — fish-pace-datasets

Read this first. Orientation, not a task list: Eli says what to work on.

## Repo state (2026-11-22)

- Branch `chla-z-rebuild`: the CHLA-Z training-table rebuild
  (fish-pace/fish-pace-datasets#4) is built; PR #9 is open for Eli's review.
  All four `datasets/chla_z/data/` tables are overwritten in place; the old
  ones are at tag `2025_11_01`. No release cut (Eli does that).
- Decisions, deliberate oddities, run log, before/after tables:
  [notes/chla-z-rebuild-2026-11.md](notes/chla-z-rebuild-2026-11.md).
  Argo source comparison (GDAC vs ERDDAP, why ERDDAP):
  [notes/argo-gdac-vs-erddap.md](notes/argo-gdac-vs-erddap.md).
- Downstream: fish-pace/2026-go-bgc-hackweek branch `rebuilt-training-data`
  reads these tables with `DATA_RELEASE = "chla-z-rebuild"` (this branch name);
  switch it to the release tag once the release exists.

## Flagged for Eli, not changed

- Argo bins are (A, B], OOI bins are [A, B). The metadata now says so.
- OOI `time` is the mean of the day's good samples: near noon at moorings, so
  `solar_hour` means little there.
- October 2026 PACE matchups are NRT only (standard V3.2 ends 2026-09-30).

## How to rebuild

`datasets/chla_z/build.py` (stages `argo`, `ooi-fetch`, `ooi`, `pace --product`,
`pace-merge`); every stage resumes. Code in `src/chla_z_build.py` and
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

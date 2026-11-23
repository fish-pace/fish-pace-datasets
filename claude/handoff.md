# Handoff — fish-pace-datasets

Read this first. Orientation, not a task list: Eli says what to work on.

## Repo state (2026-11-23)

- The CHLA-Z rebuild (#4) is **released**: tag
  [`2026_11_23`](https://github.com/fish-pace/fish-pace-datasets/releases/tag/2026_11_23)
  "Nov 2026" (PR #9, merged). Six tables in `datasets/chla_z/data/`, including
  `CHLA_{argo,ooi}_profiles_plus_OLCI.parquet`. The old tables are at
  `2025_11_01`. Branch `chla-z-rebuild` is deleted.
- Decisions, deliberate oddities, run log, before/after tables:
  [notes/chla-z-rebuild-2026-11.md](notes/chla-z-rebuild-2026-11.md).
  Argo source comparison (GDAC vs ERDDAP, why ERDDAP):
  [notes/argo-gdac-vs-erddap.md](notes/argo-gdac-vs-erddap.md).
- Downstream: fish-pace/2026-go-bgc-hackweek (PR #19) reads
  `DATA_RELEASE = "2026_11_23"`.
- My local `2025_11_01` tag is one commit behind GitHub's (GitHub's adds the
  Sauzède LUT); the four tables are identical in both. `git fetch --tags`
  warns about it.

## Next task Eli named (2026-11-23), not started

Branch `pipeline-notebooks`. Rework the notebooks in `datasets/chla_z/notebooks/`
(`argopy.ipynb`, `ooi.ipynb`, `argo-matchups.ipynb`, plus OLCI) into **example
notebooks that show users exactly how data retrieval, QC and matchups are
done**. They are solely data retrieval, QC and matchup methods, and must be
labelled as such (not model training). They run small real examples and point
to `build.py` and `src/chla_z_build.py` for the full pipeline. The build
runs from the script; the notebooks are for showing the method.

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

# Handoff — fish-pace-datasets

Read this first. Orientation, not a task list: Eli says what to work on.

## Repo state (2026-11-24)

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
- Open loose ends (not tasks): the repo has no `LICENSE` or README
  "Reuse and citation" section; `argo-fetch` has only been run on 2024-03.
- **Open: PR #14 (branch `glider-pilot`), the CHLA-Z glider tables**, waiting
  for Eli's review (below).
- My local `2025_11_01` tag is one commit behind GitHub's (GitHub's adds the
  Sauzède LUT); the four tables are identical in both. `git fetch --tags`
  warns about it.

## Method notebooks (PR #11, merged 2026-11-23; branch deleted)

- `datasets/chla_z/notebooks/01`–`04` show the data methods (Argo, OOI, PACE,
  OLCI) on small live samples and assert against the released tables;
  labelled as data methods, not model training. `05-compare-datasets` keeps
  the in situ vs satellite comparisons Eli wanted, with solar hour (NPQ) as
  the main point. Eli reviewed them: "look fine".
- `argopy.ipynb`, `ooi.ipynb`, `argo-matchups.ipynb` are retired, kept at tag
  `2026_11_23`; the new notebooks and READMEs say so.
- The global Argo download (was only `argopy.ipynb` Workflow 2, run by hand)
  is now `build.py argo-fetch`.
- `datasets/data_collection.json` CHLA-Z entries are generated from the STAC
  items by `src/update_data_collection.py`: edit the items, rerun the script.
- Detail and test results: the last section of
  [notes/chla-z-rebuild-2026-11.md](notes/chla-z-rebuild-2026-11.md).
- **Where the data lives:** the released tables are plain git files in
  `datasets/chla_z/data/` on GitHub (no LFS), versioned by release tag.
  Intermediates (24 GB) are only on the hub, in `_temp_data` (below).

## Glider tables (PR #14, open, 2026-11-24)

- IOOS Glider DAC chlorophyll as a third CHLA-Z platform, **for release like
  Argo and OOI** (Eli: it started as a pilot only because it is small). 20
  deployments in six regions, 8,923 rows (3 h solar windows, one glider
  profile each). `CHLA_glider_profiles{,_plus_PACE,_plus_OLCI}.parquet` in
  `datasets/chla_z/data/`, STAC items, README, `data_collection.json`.
- Code `src/chla_z_glider.py` (+ `chla_z_glider_selection.json`); `build.py`
  stages `glider-fetch`, `glider`, and `--platform glider` for the matchup
  stages and merges. Notebook `06-glider-retrieval-qc.ipynb`. Everything
  else: [notes/chla-z-glider.md](notes/chla-z-glider.md).
- Decided 2026-11-24: main `CHLA_A_B` is Sauzède only (as OOI); raw
  `CHLA_FLUOR_*` and `dark_offset` (below 300 m) are columns, offset not
  subtracted (it doubled night PACE/glider, 1.19 → 1.99; Argo ~1.25). PACE
  rule = the release's 24 h rule until #10 is decided.
- Needs a new release tag after merge (Eli cuts it).
- On #10 (2026-11-24): PACE daily files mostly start just *after* 00 UTC of
  their own day, so a UTC-day rule must pick the file by name; OOI is not
  dark-corrected.

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

`datasets/chla_z/build.py` (stages `argo-fetch`, `argo`, `ooi-fetch`, `ooi`,
`glider-select`, `glider-fetch`, `glider`, `pace --product`, `pace-merge`,
`olci`, `olci-merge`; `--platform argo|ooi|glider` for the matchup stages and
merges, which then rewrite only that platform's tables); every stage resumes. Code in `src/chla_z_build.py`
and `src/chla_z_tables.py`; the notebooks call the same functions.
`argo-fetch` skips the 2026-11-06 tiles already in `_temp_data/raw/`; delete
them for a new Argo snapshot. Don't use argopy `cache_index=True` (broken with
fsspec 2026.9).

## Environment

- **`~` has a quota; `~/shared-public` does not** (ENOSPC while `df` shows free
  space). Venv: `~/shared-public/venvs/chla-z`, built with
  `/srv/conda/envs/notebook/bin/python3.13 -m venv` from `requirements.txt`.
- `notebooks/_temp_data` is a symlink into shared-public (git-ignored, 24 GB):
  Argo tiles (`raw/`) and monthly .nc (7.8 GB, ERDDAP snapshot 2026-11-06),
  OOI raw CSVs, PACE per-granule matchups (`matchups_v3_2/`), OLCI
  (`matchups_olci/`), logs, and glider intermediates (`glider/`, ~270 MB).
- The venv has no `ipykernel`. Notebooks were tested as scripts in the venv
  (jupytext py:percent sources) and executed for stored outputs in the hub
  `python3` kernel.
- The run used a 16 GB / 4-core hub. Rrs matchups are CPU-bound (~19 h).
- **Clean up `~/shared-public` after the poster work** (Eli): keep what the
  poster needs. Not before.

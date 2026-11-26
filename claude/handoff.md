# Handoff — fish-pace-datasets

Read this first. Orientation, not a task list: Eli says what to work on.

## Repo state (2026-11-26)

- **Latest release: [`2026_11_26`](https://github.com/fish-pace/fish-pace-datasets/releases/tag/2026_11_26)**
  "Nov 2026 (gliders, 280 deployments)": glider tables rebuilt from the
  whole-DAC scan (PR #20, merged 2026-11-26; #19 closed; branch deleted).
  Argo and OOI identical to `2026_11_23`. Earlier: `2026_11_24` (20 pilot
  gliders; its `shore_km` is `inf` for 393 offshore rows), `2026_11_23`
  (Argo/OOI rebuild), `2025_11_01`.
- Release notes since 2026_11_23 say: no random row split (OOI, gliders);
  use a space-time grouped split, pointing to 2026-go-bgc-hackweek
  `claude/notes/paper-leakage-and-ooi-sites.md`.
- **Downstream:** 2026-go-bgc-hackweek poster v2 (branch
  `poster-v2-glider`, its #22) reads `2026_11_24`; `2026_11_26` has 14x the
  glider rows and fixes `shore_km`. Not changed there (other repo).
- #10 (PACE/OLCI time rule) closed: 24 h rule for PACE, UTC day for OLCI.
- Open issues: #13 (OOI biofouling/calibration the QC misses), #7 (BBP700
  idea). Open threads (not tasks): no `LICENSE` or README "Reuse and
  citation"; `argo-fetch` only run on 2024-03; clean up `~/shared-public`
  after the poster work, not before.
- My local `2025_11_01` tag is one commit behind GitHub's (adds the Sauzède
  LUT; tables identical); `git fetch --tags` errors on it.

## Notes

- [notes/chla-z-rebuild-2026-11.md](notes/chla-z-rebuild-2026-11.md): Argo/OOI
  decisions, deliberate oddities, run log, method notebooks 01–04, changes
  after Eli's review (OOI 3 h solar windows, (A, B] bins, Oct 2026 PACE = NRT).
  Split design / leakage is decided in the hackweek repo
  (`claude/notes/paper-leakage-and-ooi-sites.md`), not here.
- [notes/chla-z-glider.md](notes/chla-z-glider.md): glider data problems,
  deployments, scan, the #19 rebuild, dark offset, matchup numbers, open questions.
- [notes/argo-gdac-vs-erddap.md](notes/argo-gdac-vs-erddap.md): why ERDDAP.

## Gliders

- **Released `2026_11_26`:** 280 IOOS Glider DAC deployments (scan of every
  DAC chlorophyll dataset, #16; selection #19), 98,659 rows, 25,446 valid
  PACE / 26,856 OLCI matchups. Per-row `deep_offset_flag` and `scan_check`;
  whole-column bad windows and bad position fixes dropped; profile =
  profile_id + time. Notebook 05 compares all three platforms; 06 is the
  glider methods notebook.
- Decided (Eli): `CHLA_A_B` Sauzède only, raw `CHLA_FLUOR_*` and
  `dark_offset` as columns, offset not subtracted (2026-11-24); deep-offset
  deployments kept with a flag; all `check` deployments kept; whole-column
  bad windows dropped (2026-11-24, #19).
- **Where things are:** data problems in Glider DAC sets → "Known problems
  in Glider DAC data" in [notes/chla-z-glider.md](notes/chla-z-glider.md)
  (also the #19 comment of 2026-11-25); decisions and run log → "Rebuild
  with the scanned deployments" there. Open: Great Lakes gets an ocean
  Sauzède ratio; 3,795 rows fill < 3 bins.
- Adding deployments: re-run `glider-scan`, then `write_selection()`,
  `glider-fetch`, `glider`, matchups with `--platform glider`. A fix that
  moves rows needs their days ±1 re-matched (units are skipped if present).
- Hub: the superseded glider matchup backups were deleted 2026-11-26; only
  the live `glider/` unit folders remain.

## How to rebuild

`datasets/chla_z/build.py`: stages `argo-fetch`, `argo`, `ooi-fetch`, `ooi`,
`glider-select`, `glider-scan`, `glider-fetch`, `glider`, `pace --product`, `pace-merge`,
`olci`, `olci-merge`. `--platform argo|ooi|glider` on the matchup stages and
merges rewrites only that platform's tables. Every stage resumes. Code in
`src/chla_z_{build,tables,glider}.py`; notebooks call the same functions.
`data_collection.json` is generated from the STAC items by
`src/update_data_collection.py`. `argo-fetch` skips the 2026-11-06 tiles in
`_temp_data/raw/` (delete them for a new snapshot). Don't use argopy
`cache_index=True` (broken with fsspec 2026.9).

## Environment

- Released tables are plain git files (no LFS), versioned by tag.
  Intermediates are only on the hub: `notebooks/_temp_data` → shared-public
  (git-ignored, ~24 GB; glider part ~270 MB).
- **`~` has a quota; `~/shared-public` does not.** Venv
  `~/shared-public/venvs/chla-z` (python3.13, from `requirements.txt`; no
  `ipykernel`: test notebooks as jupytext scripts in the venv, execute for
  outputs in the hub `python3` kernel).
- Hub 16 GB / 4 cores, shared with Eli's other jobs. Argo+OOI Rrs matchups
  ~19 h; glider PACE+OLCI ~4.5 h at 4 workers (280 deployments); glider
  fetch ~50 min at 3 workers. Notebook 05 ~1.5 min, 06 ~15 s.

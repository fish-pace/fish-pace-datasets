# Handoff — fish-pace-datasets

Read this first. Orientation, not a task list: Eli says what to work on.

## Repo state (2026-11-25)

- **Latest release: [`2026_11_24`](https://github.com/fish-pace/fish-pace-datasets/releases/tag/2026_11_24)**
  "Nov 2026 (gliders)": Argo + OOI (identical to `2026_11_23`) plus the
  three glider tables (PR #14, merged 2026-11-24; branch deleted). Earlier:
  `2026_11_23` (Argo/OOI rebuild), `2025_11_01`.
- Both 2026 release notes say: no random row split on OOI (temporal
  autocorrelation); use a space-time grouped split (1° × 30 d), pointing to
  2026-go-bgc-hackweek `claude/notes/paper-leakage-and-ooi-sites.md`.
- **Downstream:** 2026-go-bgc-hackweek poster v2 (branch
  `poster-v2-glider`, its #22) reads `2026_11_24` (2026-11-24, not yet merged
  to its `main`); its `main` (poster v1) stays on `2026_11_23`.
- #10 (PACE/OLCI time rule) closed 2026-11-24: the release keeps the 24 h
  rule for PACE, UTC day for OLCI.
- Open threads (not tasks): no `LICENSE` or README "Reuse and citation";
  `argo-fetch` only run on 2024-03; clean up `~/shared-public` after the
  poster work, not before.
- My local `2025_11_01` tag is one commit behind GitHub's (adds the Sauzède
  LUT; tables identical); `git fetch --tags` warns about it.

## Notes

- [notes/chla-z-rebuild-2026-11.md](notes/chla-z-rebuild-2026-11.md): Argo/OOI
  decisions, deliberate oddities, run log, method notebooks 01–05, changes
  after Eli's review (OOI 3 h solar windows, (A, B] bins, Oct 2026 PACE = NRT).
  Split design / leakage is decided in the hackweek repo
  (`claude/notes/paper-leakage-and-ooi-sites.md`), not here.
- [notes/chla-z-glider.md](notes/chla-z-glider.md): glider deployments,
  processing, dark offset, matchup numbers, open questions.
- [notes/argo-gdac-vs-erddap.md](notes/argo-gdac-vs-erddap.md): why ERDDAP.

## Glider tables (released 2026_11_24)

- IOOS Glider DAC chlorophyll, third CHLA-Z platform, released like Argo/OOI
  (Eli: "pilot" only because it is small). 20 deployments, six regions, 8,923
  rows (3 h solar windows, one glider profile each); 2,289 valid PACE, 2,680
  OLCI matchups. Notebook `06-glider-retrieval-qc.ipynb`.
- Decided (Eli, 2026-11-24): `CHLA_A_B` Sauzède only (as OOI); raw
  `CHLA_FLUOR_*` and `dark_offset` are columns, offset not subtracted (night
  PACE/glider 1.19 → 1.99 if subtracted; Argo ~1.25). PACE = release 24 h
  rule.
- `glider-select` / `glider-fetch` were not re-run through `build.py`; the
  tables came from the pilot's downloads in `_temp_data/glider/`.

## Glider rebuild (#19, branch `glider-release`, 2026-11-25)

PR #18 (scan) merged. The glider table is rebuilt from the scan, pilot 20
included: 280 deployments, 98,586 rows, 25,442 valid PACE / 26,851 OLCI
matchups; per-row `deep_offset_flag` and `scan_check`; whole-column bad
windows dropped; finer scan regions. Fetch, build and all glider matchups
ran 2026-11-24/25; tables, README, STAC items and `data_collection.json` are
on the branch. **Not yet:** PR, merge, release tag (Eli decides), and
notebook 06 still shows the pilot. Details and decisions: "Rebuild with the
scanned deployments" in [notes/chla-z-glider.md](notes/chla-z-glider.md).

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
  ~19 h; glider PACE+OLCI ~4.5 h at 4 workers.

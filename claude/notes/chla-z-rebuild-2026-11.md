# CHLA-Z training tables: November 2026 rebuild

Rebuild of the four `datasets/chla_z/data/` tables with the 2026 GO-BGC Hackweek
fixes (fish-pace/fish-pace-datasets#4). Run on the PACE hub on 2026-11-21 by an
agent, from Eli's decisions below. The release before it is `2025_11_01`.

## Decisions (Eli Holmes, 2026-11-21)

| Question | Decision |
|---|---|
| Argo descending profiles | **Drop them**; ascending (`DIRECTION == "A"`) only |
| Date range | **Extend** to 2024-03-01 – 2026-10-31 for both platforms |
| File names | **Overwrite in place**; old files stay at tag `2025_11_01` |
| OOI Sauzède fluorescence:chla correction | **Apply** (`chla_ratio` column kept) |
| Release | Not part of this work; Eli reviews the PR and cuts it |
| Network code | Anything hitting ERDDAP must resume and retry unattended |

## How it is built

`src/chla_z_build.py` holds all processing; `src/chla_z_tables.py` writes the
files and their parquet metadata; `datasets/chla_z/build.py` is the driver. The
three notebooks call the same functions, so they describe what actually ran.

Every stage is a set of independent units, each written to its own file
atomically (`.part` then rename), skipped on rerun, and run in a fresh process
(`ProcessPoolExecutor(max_tasks_per_child=1)`), so an OOM kill loses one unit:

| Stage | Unit | Output under `notebooks/_temp_data/` |
|---|---|---|
| `argo` | month | `argo_profiles_monthly/argo_chla_profiles_YYYY-MM.parquet` + `.stats.json` |
| `ooi-fetch` | dataset × month | `ooi_raw/<id>/<YYYY-MM>.csv.gz` or `.empty` |
| `ooi` | dataset | `ooi_windows/<id>.parquet` or `.empty` (3 h windows; `ooi_daily/` holds the superseded UTC-day files) |
| `pace --product P` | platform × PACE granule | `matchups_v3_2/<P>/<platform>/<granule>.parquet` or `.empty` |
| `olci` | platform × UTC day | `matchups_olci/<platform>/<YYYYMMDD>.parquet` or `.empty` |

ERDDAP: retries with backoff on connection errors, timeouts, 408/429/5xx. A
404 is only "empty" if its body says "no matching results"; any other 404 or
4xx raises. Months outside a dataset's ERDDAP `time_coverage` are not asked
for. A CSV is parsed before it is kept, so a truncated download is refetched.

The Argo monthly `.nc` files were not refetched: they are the 2026-11-06
Ifremer ERDDAP pull (argopy 1.5.0, expert mode); see `argo-gdac-vs-erddap.md`.

## Things that look odd but are deliberate

- **Both platforms use (A, B] bins** (since 2026-11-23). The Argo code (2025
  `average_profile()`: `depth > z_min & depth <= z_max`) never matched its own
  metadata, which said `PRES >= A and PRES < B`; the code's behaviour is kept
  and the metadata fixed. OOI used `pd.cut(right=False)`, i.e. [A, B), until
  Eli asked for one rule. This is not a rare edge case: ERDDAP OOI depths are
  whole metres (2.3 M of 11 M good samples sit on a 10 m edge), and 26% of
  Argo PRES values are whole numbers (4% on a 10 m edge, 2025-07). A depth of
  0 is outside (0, 10], so 5 more fixed instruments that publish z = 0 got
  depth overrides (4 Coastal near-surface frames at 7 m, Irminger surface
  buoy at 1 m); they stay in the 0–10 m bin.
- **Argo profiles with no QC-passing level are dropped.** The 2025 table kept
  every profile, even all-NaN ones, because it never filtered on QC.
- **TIME/LAT/LON "first" is exact.** Within an ascending profile they are
  constant (checked on 2025-07: 0 of 3,183 profiles vary). `POSITION_QC` 8
  (interpolated) positions are kept, as before.
- **The vectorized binning equals the notebook function.** On 2025-07,
  `argo_bin_profiles()` and `summarize_profile_binned()` agree on every cell of
  1,972 profiles.
- **OOI lat/lon are per row, not per (row, bin).** The 2025 code took them per
  (day, bin) and pivoted on (date, lat, lon), so a position change within a day
  could split one day into two rows. Before the 3 h windows, 9,776 of the 9,782
  old OOI daily profiles were reproduced, and with the Sauzède ratio undone
  their surface CHLA equalled the old value (median ratio 1.000).
- **31 OOI datasets the 2025 notebook skipped as "no data in time range" were
  queried again** for the extended range. All still have no data, so the site
  list is the same 30. Three exclusions stay for date-independent reasons (no
  `qc_agg`; 2000 m and deeper).
- **OOI rows are 3 h windows of local mean solar time** (Eli, 2026-11-23).
  A UTC-day mean (2025, and the first rebuild) put every OOI time near 12 UTC:
  a solar hour of about 3–9 h at OOI longitudes, nearly constant per site, so
  `solar_hour` encoded longitude, not sampling time. Windows are 00–03, 03–06,
  ... h of solar time (UTC + lon/15 h, per sample); `time` is the mean time of
  the window's samples. 95,770 rows (was 13,664 days), spread evenly over the 8
  windows. All windows of one day usually match the same daily PACE and OLCI
  pixel; they differ in CHLA and solar hour. `profile_id` =
  `<dataset>_<solar YYYYMMDD>T<HH>`, so a few ids carry a solar date one day
  off the UTC date of `time`.
- **PACE band names come from round().** V3.2 AOP band centres are fractional;
  `round()` maps all 172 onto the 2025 integer names (the 2025 names were
  `int()` of V3.1 centres; 640–643 included). The code raises if that ever stops
  being one-to-one. Kd_490 is the KD band nearest 490 nm (489.674 nm).
- **Standard before NRT, then closest window centre**, per product, and
  chlor_a/Kd are attached only when they come from the same day and stream as
  the Rrs file. NRT only fills dates with no standard V3.2 file (April 2026 on;
  in practice mostly October 2026). Duplicate KD_NRT granules (two name
  patterns for some days) are handled by the same rule.
- **Matching is unchanged from 2025:** profile time within 24 h after the
  granule's `time_coverage_start`, nearest 4 km cell. The pixel read is one dask
  `vindex` per granule instead of one `.sel` per profile; the nearest-index
  rule is the same.

## Run log (2026-11-21)

- Hub: 16 GB, 4 cores. The Rrs matchups are CPU-bound (decompressing 172-band
  chunks): about 60 s of wall time per granule overall with 6 workers.
- Bug found and fixed during the run: OOI mean time was computed as an int64
  mean and read back as nanoseconds, but pandas 3 stores these as microseconds,
  so every OOI time landed in 1970. Fixed with `Series.mean()` on the datetimes;
  checked that every `profile_id` day equals its time's day.
- PACE: Rrs 1,137, chlor_a 1,138 and Kd 1,176 granules, 0 failed units in
  any stage. Rrs took ~19 h (2026-11-21 18:07 to 2026-11-22 13:24 UTC),
  sharing the 4 cores with chlor_a and Kd until 22:49.
- `pace-merge` (`logs/pace_merge_stats.json`): Argo 59,583 profiles → 56,912
  with a matching Rrs granule (2,084 NRT), 9,542 with any non-NaN Rrs; OOI
  13,664 → 13,102 (242 NRT), 3,187 with Rrs (UTC-day OOI, superseded; see
  the 2026-11-23 re-run below). chlor_a/Kd from a different granule than Rrs
  or missing (left NaN): 25 and 22 Argo rows.
- `argopy.ipynb` cell 19 read `datasets/chla-z/` (404 since the folder is
  `chla_z`); fixed so the notebook runs end to end.
- Tables written from the venv built from `requirements.txt`. The notebooks
  were then executed in the hub's notebook env (pandas 2.3.3); their rewrites
  of the tables were checked equal to the venv's and the venv's were kept.
- Things that changed counts and are expected:
  - (UTC-day OOI) `CHLA_0_10` dropped 6,101 → 5,325 although rows rose: the 9
    `DEPTH_OVERRIDE_M` instruments (12–187 m) published z = 0 and so filled
    the surface bin in 2025.
  - 2025 OOI `_plus_PACE` had 10,989 rows for 6,415 profile_ids (duplicates);
    now one row per profile_id.
  - The 2025 Argo and OOI `_plus_PACE` both have exactly 2,770 rows with Rrs.
    Checked: coincidence, the values differ.

### OOI re-run as 3 h windows (2026-11-23)

- Matchups were split into per-platform folders (`<product>/argo/`,
  `<product>/ooi/`, `matchups_olci/<platform>/`) so OOI could be re-matched
  alone. The existing Argo files were copied into `argo/` (script in
  `logs/migrate_argo.py`); the Argo `_plus_PACE` and `_plus_OLCI` tables then
  rebuilt identical (`assert_frame_equal`), and are unchanged after the final
  merges too.
- `ooi` stage: 95,770 windows from 30 instruments, median 48 samples per
  window, `profile_id` unique. Five more fixed instruments published z = 0
  (added to `DEPTH_OVERRIDE_M`: four at 7 m, GI01SUMO at 1 m, Eli
  2026-11-23); 14 overrides in all.
- `pace --platform ooi`: 1,137 Rrs, 1,138 chlor_a, 1,176 Kd granules, then
  `olci --platform ooi`: 975 days. 0 failed units; 1 retry (OLCI), recovered.
  ~3 h total.
- `pace-merge`: OOI 95,770 → 90,338 with a Rrs granule (1,570 NRT), 22,240
  with Rrs, from all 30 instruments; chlor_a/Kd from another granule or
  missing: 49 and 36 rows. Windows with Rrs are spread evenly over solar
  hour (2,100–3,000 per 3 h window). Median 5 windows per instrument-day
  with Rrs, which share the day's pixel.
  So a train/test split on the OOI table must be grouped by instrument
  (Eli, 2026-11-23).

### Before / after

**CHLA_argo_profiles.parquet**

| | 2025_11_01 | rebuild |
|---|---:|---:|
| rows | 15,833 | 59,583 |
| unique profile_id | 15,792 | 59,583 |
| floats / instruments | 537 | 907 |
| non-NaN CHLA bins | 196,309 | 1,166,607 |
| CHLA_0_10 non-NaN | 13,044 | 58,464 |
| first | 2024-03-01 | 2024-03-01 |
| last | 2025-11-29 | 2026-10-31 |

**CHLA_argo_profiles_plus_PACE.parquet**

| | 2025_11_01 | rebuild |
|---|---:|---:|
| rows | 14,916 | 56,912 |
| unique profile_id | 14,916 | 56,912 |
| floats / instruments | 534 | 906 |
| non-NaN CHLA bins | 185,080 | 1,114,318 |
| CHLA_0_10 non-NaN | 12,278 | 55,830 |
| first | 2024-03-05 | 2024-03-05 |
| last | 2025-11-29 | 2026-10-31 |
| rows with any Rrs | 2,770 | 9,542 |
| NRT PACE rows | 1,684 | 2,084 |

**CHLA_ooi_profiles.parquet** (3 h windows)

| | 2025_11_01 | rebuild |
|---|---:|---:|
| rows | 9,782 | 95,770 |
| unique profile_id | 9,782 | 95,770 |
| floats / instruments | 30 | 30 |
| non-NaN CHLA bins | 61,331 | 532,803 |
| CHLA_0_10 non-NaN | 6,101 | 40,948 |
| first | 2024-03-01 | 2024-03-01 |
| last | 2025-12-05 | 2026-10-31 |

**CHLA_ooi_profiles_plus_PACE.parquet** (3 h windows)

| | 2025_11_01 | rebuild |
|---|---:|---:|
| rows | 10,989 | 90,338 |
| unique profile_id | 6,415 | 90,338 |
| floats / instruments | 30 | 30 |
| non-NaN CHLA bins | 66,686 | 503,233 |
| CHLA_0_10 non-NaN | 7,013 | 38,887 |
| first | 2024-03-06 | 2024-03-05 |
| last | 2025-12-05 | 2026-10-31 |
| rows with any Rrs | 2,770 | 22,240 |
| NRT PACE rows | 2,207 | 1,570 |

## OLCI matchups (added 2026-11-23, Eli's request)

Eli asked for OLCI matched to *every* profile, with a flag for whether it also
has a PACE match. That way "OLCI has more data" and "OLCI vs PACE on the same
profiles" can be separated. Two new tables:
`CHLA_{argo,ooi}_profiles_plus_OLCI.parquet`. Stages `olci` (one unit per UTC
day) and `olci-merge`.

- **Rules are Nabir's** (hackweek `contributor_folders/nama/olci_l3_matchups.ipynb`),
  kept on purpose so his rows reproduce: the file of the profile's UTC day,
  nearest cell, S3A first; S3B is read at all of the day's points when any has
  NaN S3A Rrs_443; each band is S3A where finite, else S3B. So one row's 11
  bands can come from two sensors. `olci_sat` names the Rrs_443 sensor.
- **One difference:** a file that fails to open raises and the day is retried.
  The notebook stored NaN, which would make a network drop look like a masked
  pixel.
- **Check:** all 12,326 of Nabir's rows on days with OLCI files are reproduced
  exactly (0 cell mismatches, 0 `olci_sat` mismatches). His other 403 rows are
  October 2026 days after the last OLCI file (2026-10-01): NaN there, and not
  in the new table, whose rows are profiles on a day with an OLCI file.
- **Run:** 975 days, 6 workers, ~80 min, 0 retries, 0 failures (Argo and
  UTC-day OOI). OOI re-run for the 3 h windows: 975 days, 1 retry, 0 failures.

Coverage, March 2024 – September 2026 (both sensors available):

| | rows | OLCI all 11 bands | PACE Rrs valid | both |
|---|---:|---:|---:|---:|
| Argo | 57,544 | 12,535 (21.8%) | 9,187 (16.0%) | 4,935 |
| OOI (3 h windows) | 94,265 | 23,900 (25.4%) | 21,823 (23.2%) | 13,484 |

So OLCI does give more valid matchups than PACE, mostly for Argo, and only
40–60% of each sensor's valid profiles are valid in the other. PACE is
matched with a 24 h window from the file's `time_coverage_start`; OLCI with
the calendar day. That difference in rule is part of the comparison, not just
the sensors.

## Method notebooks and `argo-fetch` (2026-11-23, branch `pipeline-notebooks`)

Eli asked for the notebooks to show the data methods, not to build the tables.

- `notebooks/01`–`04` each run one small live sample through the same
  `src/chla_z_build.py` functions as `build.py`, and end with an `assert`
  against the released tables. If one fails later, the first suspect is
  upstream reprocessing since the release, not the code. Checked: Argo box
  80–60°W 30–40°N 2025-07, 27/27 profiles equal; OOI CE04OSPS July 2025, 247/249
  windows equal (the 2 are month-edge solar windows, which the build fills from
  the next month: expected, and asserted); PACE and OLCI 2025-07-15, all rows
  equal.
- `05-compare-datasets` keeps the comparison plots from the old notebooks
  (Eli wanted them kept), now for both platforms: in situ vs PACE chlor_a, bias
  vs solar hour (the NPQ signal: median bias ~0.1 at night, ~0.3 Argo / ~0.2 OOI
  around solar noon), bias vs 1/Kd at night, OLCI vs PACE.
- `argopy.ipynb`, `ooi.ipynb`, `argo-matchups.ipynb` are retired; they are kept
  at tag `2026_11_23`, and the new notebooks, README, STAC items and
  `data_collection.json` say so or point to the new ones.
- The global Argo download was only ever in `argopy.ipynb` Workflow 2, run by
  hand. It is now the `argo-fetch` stage (`argo_tile_unit`,
  `argo_merge_month_unit`), same query, tiles and file names, so the existing
  `_temp_data/raw/` tiles count as done. One change: no `cache_index=True`.
  With argopy 1.5.0 and fsspec 2026.9 it raises (`HTTPFileSystem` has no
  `save_cache`), and region fetches work without it.
- `datasets/data_collection.json` (a standalone flat collection; `catalog.json`
  does not link it) had the 2025 descriptions, PACE V3.1, no OOI-profiles or
  OLCI entries, and `raw.githubusercontent.com/.../blob/main/` data hrefs that
  did not resolve. Its CHLA-Z entries are now generated from
  `datasets/chla_z/items/*.json` by `src/update_data_collection.py` (the items
  are the source to edit; rerun the script after), keeping the
  ids it already published; three entries added. All data hrefs checked 200.
- `argo-fetch` test (venv, scratch dirs, 2024-03): 23 tiles + 1 empty
  (60–120°E, 45–90°N: no float there measures CHLA); every tile has the same
  N_POINTS as the 2026-11-06 tile; binning the merged month gives all 1,375
  released profiles identical, plus 3 profiles ERDDAP has added since.

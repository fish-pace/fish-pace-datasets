# CHLA-Z glider tables (2026-11-23/24)

IOOS Glider DAC chlorophyll profiles as a third CHLA-Z platform, to add coastal
regions where OOI has only a few sites. Started as a pilot; Eli then said it
is a **data release like Argo and OOI** ("only pilot because it was small, not
all glider data"): `CHLA_glider_profiles.parquet`, `_plus_PACE`, `_plus_OLCI`
in `datasets/chla_z/data/`, with STAC items and README sections. 20
deployments across six regions; the selection can grow.

- Code: `src/chla_z_glider.py` (selection, fetch, windows, dark offset,
  combine), `src/chla_z_glider_selection.json` (the deployments). The table
  is written by `chla_z_tables.write_glider_profiles`; matchups are the
  release code with platform "glider" (`chla_z_build.PLATFORMS`).
- Build: `build.py glider-select` (only to choose deployments), `glider-fetch`,
  `glider`, then `pace --product P --platform glider`, `olci --platform glider`,
  `pace-merge --platform glider`, `olci-merge --platform glider` (the merges
  with `--platform` rewrite only that platform's table).
- Methods notebook: `datasets/chla_z/notebooks/06-glider-retrieval-qc.ipynb`.
- Source: https://gliders.ioos.us/erddap (tabledap), fetched 2026-11-23/24.

## Decisions (Eli Holmes, 2026-11-23)

| Question | Decision |
|---|---|
| Scope | ~20 deployments across six regions, not all glider data; released like Argo and OOI (2026-11-24) |
| Rows | Like OOI: 3 h windows of local mean solar time |
| Code location | Glider stages in `build.py`; a glider module in `src/`; intermediates in `_temp_data/glider/` |
| Datasets without chlorophyll units | Ignore |
| Wonky chlorophyll | Pick another deployment |
| PACE time rule (2026-11-24) | **Same as the release** (24 h after `time_coverage_start`), so all three platforms change together when #10 is decided |
| Main `CHLA_A_B` (2026-11-24) | **Sauzède only**, as OOI; raw `CHLA_FLUOR_A_B` and `dark_offset` kept as columns, the offset **not** subtracted |

## Processing, and where it differs from OOI

Same as OOI: 3 h windows of local mean solar time (UTC + lon/15 h); 10 m bins
(A, B] in 0–200 m; Sauzède ratio applied and kept as `chla_ratio`; `profile_id`
= `<dataset>_<solar date>T<window start hour>`.

Different, on purpose:

- **One profile per window, not an average.** A glider moves 10–80 km a day
  (Florida Current Sprays the most), so averaging a window's profiles would mix
  positions. The kept profile has the most filled 10 m bins; ties go to the one
  nearest the window centre. `glider_profile_id` and `profiles_in_window` say
  which and from how many.
- **Time = mean `precise_time` of the kept samples**, not the profile `time`.
  On ru34 (delayed mode) the profile `time` of climb profiles is ~70 min after
  their chlorophyll samples, because chlorophyll is sampled on only part of
  each segment.
- **Position = mean `precise_lat/lon`** for Slocums; Sprays have no per-sample
  position, so the profile position. Measured on six deployments: within one
  10 m bin the glider moves 0.01–0.04 km (median); over a whole profile at most
  0.7 km (Slocum) or ~3 km (Spray in the Florida Current, estimated).
- **QC:** the dataset's own flag where it has one, keeping 1 and 2:
  `chlorophyll_a_qc` (Spray) or `chlorophyll_a_qartod_summary_flag` (Rutgers;
  only a gross-range test, so nearly everything is 1). Real-time Slocum sets
  often have no flag. The useful screening is ours (below).
- **Extra columns:** `region`, `platform_type`, `institution`, `solar_hour`,
  `max_depth`, `shore_km` (Natural Earth 10 m coastline, approximate).

## Selecting the deployments

1. ERDDAP advanced search: tabledap, `chlorophyll_a`, minTime 2024-03-01 →
   227 datasets, 173 deployments (56 have a `-delayed` copy).
2. Shortlist: units present, ≥ 14 days in 2024-03..2026-10, delayed copy where
   it exists → 147 deployments.
3. One-day screen from mid-deployment (`screen_stats`): flag "flat profile"
   (max 5 m-bin median within 0.05 of p10 and p90/p10 < 1.5), "deep offset"
   (median below 150 m > 0.2), "negatives > 5%". 114 passed, 21 deep offset,
   6 flat, 1 negatives, 5 no data that day / too few.
4. 20 chosen by hand from the passes (seasons spread, QC flag preferred).
5. **Whole records then plotted; two replaced** ("pick another", Eli):
   - sp071-20250501T1427 (off Hatteras): flat 0.04–0.07 at all depths incl.
     150 m, no structure for 3.5 months → sp070-20240912T1432.
   - sp055-20250806T2212 (California): surface below its own 150 m value for
     the last 4 weeks (0.05 vs 0.25), deep rising → drift/fouling →
     sp041-20250923T1716.
   - Kept: ru43-20260731 (summer shelf, surface ~0.1 under a strong DCM: normal).

### Scanning for more deployments

Done 2026-11-24 (#16): see "Scan of the whole DAC" below. The pilot's
`candidates` / `shortlist` / `screen_unit` stay as they were (they chose the
released 20).

### Regions are labels, not coverage

The six regions are coarse boxes on deployment centre (`region_of`). Two
earlier labels were wrong and were renamed: **no qualifying deployment is in
the Gulf of Mexico** (capex690 is at 12.6°N 55.6°W, open tropical Atlantic),
and "Gulf of Maine / New England" was ru34 on the southern New England shelf
(41.1°N 71°W).

| Region | Deployments | Windows |
|---|---:|---:|
| California Current (Spray, CUGN lines) | 5 | 3,688 |
| Off Cape Hatteras / South Atlantic Bight (Spray) | 4 | 1,817 |
| Mid-Atlantic Bight and slope (4 Slocum, 1 Spray) | 5 | 1,486 |
| Antarctic Peninsula (Slocum ru26d) | 2 | 955 |
| Southern New England shelf (Slocum ru34) | 3 | 844 |
| Tropical Atlantic (Slocum capex690) | 1 | 133 |
| **Total** | **20** | **8,923** |

## Scan of the whole DAC (#16, 2026-11-24)

Goal: a list of good candidate deployments for the next step (fetching them),
and a scan that can be re-run later without re-getting what is cached.

- **Result:** `src/chla_z_glider_candidates.csv`, one row per deployment with
  ≥ 7 days of record since 2024-03-01 (316: 20 in the table, 296 new).
  Columns: chosen `datasetID` (+ `alt_ids`, the other copies), `status`
  against the table, `verdict` / `fail` / `check`, region, institution,
  `chl_var`, units, `qc_var`, `days_in_months` (inside `b.MONTHS`), the
  whole-record stats, and `bad_days`.
- **Code:** `src/chla_z_glider_scan.py`; `build.py glider-scan` re-runs it all
  and rewrites the CSV. `restat()` recomputes stats from the cache only (for
  changing thresholds). Cache in `_temp_data/glider/scan/` (catalog, per-dataset
  info) and `_temp_data/glider/sampled/<id>/<day>.csv` (~31 MB).

### What the pilot search missed

The pilot searched only the variable name `chlorophyll_a`. ERDDAP's
`categorize/variableName` index shows the DAC also uses `chlorophyll`,
`fluorescence` (Oregon State Seagliders) and raw Slocum names
(`sci_flbbcd_chlor_units`, `sci_flbb_chlor_units`, `sci_bbfl2s_chlor_scaled`;
Skidaway, USF, UAF). PACE-era datasets: 227 → 471; deployments with units,
depth and ≥ 7 days: 316. New sources: OOI gliders (Pioneer now off Hatteras
`cp_`, Endurance `ce_`, Irminger `gi_`, Station Papa `gp_`), C-PROOF/DFO (BC),
UAF (Gulf of Alaska), Skidaway/USF (Georgia–NE Florida shelf), Oregon State,
SWFSC AMLR, BIOS, EPA Great Lakes. The OOI table in this repo has no OOI
gliders, so they are new data.

- Units are all mg m⁻³ equivalents (`ug L-1`, `ug l-1`, `ugr l-1`, `mg.m-3`,
  `microgram/L`...). `bios_jack-20240812T1856-delayed` says
  `micrograms m^-3`, but its values (surface 0.03, maximum 0.3) are mg m⁻³:
  a label typo.
- 156 of 316 have no chlorophyll QC variable; others use `chlorophyll_a_qc`,
  `chlorophyll_a_qartod_summary_flag`, `chlorophyll_qc` or `fluorescence_qc`.
- **Still nothing in the Gulf of Mexico.** The USF and Skidaway gliders are
  on the Atlantic shelf off Georgia / NE Florida (29–31.5°N, ~80°W).

### Deployment key and which copy

- Key = glider name lower-cased + start date (`deployment_key`): Stony Brook
  has `SBU02-20240822T1918-delayed` next to `sbu02-20240822T1916`.
- Delayed copy only if its span is ≥ 90% of the real-time one (stubs, as
  ru26d). **Except** delayed copies with > 1000 variables (Skidaway/USF raw
  Slocum sets, ~2,800 variables): used only when there is no real-time copy,
  because ERDDAP did not answer a whole-record query on them in 15 min.
- `status`: `new`, `in table`, `in table, grown` (maxTime > 2 d past the
  table's last profile, while that is before `END_EXCLUSIVE`),
  `in table, delayed copy now covers it` (a possible swap). Today: all 20 are
  plain `in table`.

### How the whole record is checked (and why this way)

- **One day every 4 days** (every day under 20 d of record, every 2nd under
  40 d), plus the last full day, as daily 10 m-bin means computed by ERDDAP
  (`orderByMean("time/1day,depth/10")`; `depth` comes back as the bin number).
  Days are aligned to the calendar from `START`, so a grown deployment adds
  days and keeps its cache. 6,415 days, ~1.5 h at 4 threads.
- **Not the whole record in one query:** tried first; ERDDAP opens every
  profile file, 1–10 min per deployment (~4 h for all) and the raw sets hung.
  A one-day query takes ~0.5 s because ERDDAP skips files by time. 73
  whole-record grids from that first try are in `_temp_data/glider/fullrec/`
  (not used).
- The DAC under load returns **empty-body 404s and truncated JSON**, which
  `_get` / `_info` retry (`chla_z_build._http_get` treats any 404 as final).
  Killing a client does not stop a long query on the server: active requests
  stayed at 20–25 for an hour after mine were stopped.

**Checks** (`fullrec_stats`). Fail:

| Check | Rule | Why |
|---|---|---|
| deep offset | median ≥ 300 m > 0.2 | as the pilot screen (sp034) |
| negative offset | > 5% of 0–200 m bins < −0.02 and 5th pct < −0.05 | DFO `eva035` (−0.4 at depth), some C-PROOF/OSU |
| flat | > 30% of days with 0–200 m range < 0.05 and max/min < 1.5, **only if it dives ≥ 100 m** | a 30–40 m mixed shelf is flat for real |
| no signal above deep value | median profile max < 0.15 and < 0.05 above the 150–200 m value | sp071 (the pilot's whole-record failure; its daily range was not flat enough for the flat rule) |
| surface < deep | surface (0–20 m) below the ≥ 300 m (else 150–200 m) value − 0.02 for ≥ 14 d | sp055 (caught: 25 d) |
| implausibly high | median profile max > 50 | `dfo-maria997` (~96) |

Check (look at a plot before using): `jumpy surface` (> 10% of steps with a
×5 change; mostly real upwelling off OR/WA), `sparse depths` (median share
of 0–100 m bins filled < 0.7; C-PROOF/DFO sets log chlorophyll at scattered
depths), `deep drift > 0.1`, `flat` on a shelf, and `N whole-column bad
day(s)` (gliders diving ≥ 300 m: the 150–200 m value > its median + 0.3, or
the whole profile < 0.02; days listed in `bad_days`). The last catches
California Sprays whose final days go high or ~0 at every depth (sensor
failure or recovery); a fetch can drop those days.

Plots of the `check` group and of BC, Alaska, California and OOI `ok`
deployments were looked at (2026-11-24, scratch contact sheets): Alaska, OOI
and most `ok` California look clean; most `jumpy surface` is real.

### Result (2026-11-24)

New deployments (296):

| Region | ok | check | fail | ok days in months |
|---|---:|---:|---:|---:|
| Mid-Atlantic Bight and slope | 87 | 3 | 1 | 3,114 |
| Pacific Northwest (OR/WA) | 30 | 12 | 2 | 1,820 |
| Gulf of Alaska / Bering | 14 | 1 | 0 | 1,462 |
| California Current | 23 | 13 | 29 | 1,382 |
| Irminger / subpolar N Atlantic | 3 | 2 | 0 | 792 |
| South Atlantic Bight / off Hatteras | 18 | 5 | 5 | 437 |
| British Columbia / Salish Sea | 10 | 9 | 11 | 395 |
| Open NW Atlantic (slope sea / Gulf Stream) | 4 | 0 | 0 | 248 |
| Southern New England / Gulf of Maine | 5 | 0 | 0 | 199 |
| Sargasso / Bermuda | 3 | 0 | 2 | 102 |
| Tropical Atlantic / Caribbean | 1 | 0 | 0 | 17 |
| Great Lakes | 1 | 0 | 0 | 13 |
| **Total** | **199** | **45** | **50** | **~10,000** |

(2 more: too few days.) 178 of the 199 `ok` have ≥ 14 days inside the
release months; 4 are entirely after 2026-10. Fails: deep offset 23 (mostly
California Sprays), negative offset 13, flat 10, no signal 7, surface < deep
5, implausibly high 2.

- **Deep offset is the release's own trade-off.** Released `sp020-20250122T1433`
  fails it on the whole record (its one-day screen passed); the release keeps
  `dark_offset` as a column instead of subtracting it. Whether the other 22
  deep-offset Sprays are usable on the same terms is Eli's call.
- Of the 20 in the table: 18 `ok`, sp203-20250813T1745 `check` (jumpy
  surface, looks real), sp020-20250122T1433 `fail` (deep offset, above).
- The regions here (`chla_z_glider_scan.region`) are finer than the
  release's `region_of`, which is unchanged.

### For the fetch (next step, not done)

- `chla_z_glider.fetch_unit` / `fetch_deep_unit` hard-code `chlorophyll_a`
  and read `candidates.parquet`; they need `chl_var` and `qc_var` from the
  scan catalog (`scan/catalog.parquet`), and Sprays-without-precise-lat logic
  as now.
- Fetch months are `b.MONTHS` (to 2026-10); `days_in_months` shows what that
  keeps.
- Drop `bad_days`; look at `check` rows first.

## Things that look odd but are deliberate

- **The deep (dark) offset is a column, not subtracted** (see "Dark offset").
  It is NaN for shelf Slocums and capex690, which never reach 300 m.
- **Sauzède ratio is large in the Antarctic (~2.9)** and 1.2–1.7 elsewhere;
  it is applied as for OOI.
- **Shelf profiles fill few bins** (ru34: ~35 m deep, bins below 40 m NaN).
- **Profiles start 2024-03-01** even where a deployment started earlier
  (sp065 in Feb 2024): fetch months are the CHLA-Z months.

## Dark offset (2026-11-24)

**Evidence.** On the start, middle and end days of each deployment that dives
deep, the California Current Sprays read 0.09–0.34 (fluorometer) from 150 to
500 m, flat with depth and drifting within a deployment (sp020 0.28 → 0.17 →
0.20; sp203 0.25 → 0.12 → 0.10). East Coast Sprays and ru26d read 0.03–0.10.
East Coast Sprays still have a little real CHLA at 150–200 m (0.04–0.09 vs
0.03–0.08 below 300 m), so the reference is below 300 m.

**Method** (`dark_series`, `assign_dark`, `combine`):
- `fetch-deep` stage: QC-passing CHLA below 300 m, raw ERDDAP CSV in
  `_temp_data/glider/raw_deep/`.
- `dark_smooth` = centred 3-day running median of all deep *samples* (at
  least 10 in the window), taken at each glider profile. Samples, not
  per-profile medians: real-time Slocum sets have ~2 deep CHLA samples per
  profile (ru26d-20250107 went from 3% to 99% corrected with this change).
- A window uses its own profile's value, else the nearest within 1.5 days.
- Published as `dark_offset`. If subtracted, it goes **before** the Sauzède
  division (Argo's order): `(CHLA_FLUOR_A_B - dark_offset) / chla_ratio`.

**Raw values kept** (Eli asked): `CHLA_FLUOR_A_B` = published fluorometer bin
means after QC; `CHLA_A_B = CHLA_FLUOR_A_B / chla_ratio`. The raw ERDDAP
samples also stay on the hub (`raw/`, `raw_deep/`).

**What subtracting it did (tested, then not adopted).** 6,867 of 8,923 rows
have a value (every deep-diving deployment 92–100%); subtracted, the 190–200
m bin was within ±0.01 of 0, about half the deepest bins slightly negative
(noise around 0, residual SD ~0.04). **But it moved the gliders away from
PACE and from Argo**, so Eli chose Sauzède only (2026-11-24). Night
(solar hour outside 6–18) median PACE chlor_a / in situ 0–10 m:

| | n | night ratio |
|---|---:|---:|
| Argo release (`CHLA_ADJUSTED`, dark-corrected) | 9,344 (all hours) | 1.23–1.30 |
| Deep-diving gliders, Sauzède only | 828 | 1.19 |
| Deep-diving gliders, dark + Sauzède | 827 | 1.99 |
| Shelf gliders (uncorrected), Sauzède only | 324 | 1.80 |

At 0.1–0.5 mg m⁻³ the dark value (~0.13) is about half the surface
fluorometer reading, so subtracting it halves the glider value. Possible
reasons, none tested: the deep reading is not the dark value at the surface
(ECO dark counts depend on temperature, and 300–500 m water is much colder);
the published glider CHLA already has a dark/blank removed and the deep signal
is something else; or Sauzède and a dark correction together over-correct for
these sensors. The raw columns mean it can be revisited without re-fetching.

## Matchups

**Released tables:** the release code and rules (24 h after
`time_coverage_start` for PACE, UTC-day file for OLCI), platform "glider".

### Time-rule comparison on the glider profiles (#10, superseded code)

Before the release decision, a separate matcher (commit d4cf74f, removed
since) compared both rules on the same profiles. PACE units kept every profile inside a granule's whole `time_coverage_start`–
`time_coverage_end` (~26 h); granules whose nominal day is a glider UTC day
±1 were sampled (758 per product, standard + NRT). `merge()` then applies
either rule: `window24` (the release: within 24 h after `time_coverage_start`,
standard before NRT, nearest window centre) or `utc_day` (file named for the
profile's UTC day, standard before NRT). OLCI uses the release's UTC-day rule.
Lookup is per cell (`isel`), not dask `vindex`: ~10 s per Rrs granule for
~20 points.

### Results of the rule comparison (2026-11-24, superseded code)

The first PACE run selected profiles by "inside the granule's coverage" only.
That lost ~300 profiles under `utc_day`: a file often starts a few minutes
*after* 00 UTC of its own day (64% of 741 files; e.g. PACE_OCI.20240420
starts 00:13), so a profile at 00:05 is outside it. Units now also take every
profile of the file's nominal UTC day. First run kept in `matchups_pace_v1/`.
Details and the file-coverage table are on #10 (comment of 2026-11-24).

Valid matchups (PACE: any Rrs band; OLCI: Rrs_443; "OLCI+PACE" uses utc_day):

| Region | profiles | PACE window24 | PACE utc_day | OLCI | OLCI+PACE |
|---|---:|---:|---:|---:|---:|
| Antarctic Peninsula | 955 | 88 | 88 | 72 | 28 |
| California Current | 3,688 | 1,181 | 1,209 | 1,197 | 780 |
| Mid-Atlantic Bight and slope | 1,486 | 401 | 410 | 461 | 284 |
| Off Cape Hatteras / South Atlantic Bight | 1,817 | 280 | 282 | 546 | 188 |
| Southern New England shelf | 844 | 340 | 344 | 362 | 270 |
| Tropical Atlantic | 133 | 5 | 5 | 42 | 0 |
| **Total** | **8,923** | **2,295** | **2,338** | **2,680** | **1,550** |

- **The rule barely matters here.** 8,386 profiles fall in a window24 granule,
  8,665 in a utc_day file; 124 of the shared ones get a different file, and
  only 19 of those have valid Rrs under both. utc_day adds 279 profiles
  (no window24 granule: partial-day files, gaps), 43 more valid matchups.
- **PACE V3.2 daily starts 2024-03-05**; glider profiles of 1–4 March 2024
  have no PACE file. No NRT rows (glider data end 2026-08-29).
- **Quenching shows clearly.** PACE chlor_a / glider CHLA_0_10 (utc_day,
  n ≈ 2,330). Sauzède only (before the dark correction): median 1.64; by
  solar hour 0–6 h 1.39, 6–9 h 1.67, 9–15 h 2.21, 15–18 h 1.61, 18–24 h 1.35.
  With the dark correction: 1.94, 2.51, 3.98, 2.36, 1.88 (see Dark offset).
- **Near-shore failures:** 28 matchups have PACE chlor_a 8–27 with Rrs(442)
  ≈ 0 or negative, median 12 km from shore (3–43 km), California Current
  Sprays. A coastal training set needs a near-shore or Rrs-quality filter.

### Released tables (build.py, 2026-11-24)

Matched with the release code and rules (`build.py pace --platform glider`,
three products, then `olci`, `pace-merge`, `olci-merge`; 4 workers, ~4.5 h,
no failed units). PACE uses the 24 h rule, so the numbers are close to the
`window24` column above (2,289 vs 2,295: the release matcher picks the nearest
granule slightly differently). "OLCI+PACE" here uses the release's
`pace_Rrs_valid`.

| Region | profiles | in a PACE granule | PACE valid Rrs | OLCI | OLCI+PACE |
|---|---:|---:|---:|---:|---:|
| Antarctic Peninsula | 955 | 914 | 88 | 72 | 28 |
| California Current | 3,688 | 3,546 | 1,176 | 1,197 | 757 |
| Mid-Atlantic Bight and slope | 1,486 | 1,334 | 402 | 461 | 274 |
| Off Cape Hatteras / South Atlantic Bight | 1,817 | 1,643 | 279 | 546 | 187 |
| Southern New England shelf | 844 | 820 | 339 | 362 | 265 |
| Tropical Atlantic | 133 | 133 | 5 | 42 | 0 |
| **Total** | **8,923** | **8,390** | **2,289** | **2,680** | **1,511** |

`_plus_PACE` has 8,390 rows (profiles in a granule, 2024-03-05 to
2026-08-29); `_plus_OLCI` keeps all 8,923. PACE/glider CHLA_0_10 by solar hour
in the released table: 1.39, 1.65, 2.21, 1.60, 1.35 (n 577, 295, 587, 271,
553); night 1.19 (n = 819), 1.99 with `dark_offset` subtracted. 29 matchups
have PACE chlor_a > 8 (3–43 km from shore, median 12 km).

## Not done / open

- Matchup rule: #10 closed by Eli 2026-11-24, keeping the release rules
  (PACE 24 h, OLCI UTC day); nothing to re-match.
- The dark offset (above): kept as a column; revisit if the cause is found.
- OOI is not dark-corrected either (noted on #10, 2026-11-24).
- No NPQ correction (as OOI); `solar_hour` is there to model it.
- No check of fluorometer calibration across deployments beyond the screen.

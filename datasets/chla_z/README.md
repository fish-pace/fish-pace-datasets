# CHLA in situ profile datasets

This dataset group contains chlorophyll-a (CHLA) vertical profile metrics derived
from in situ observations, with optional matchups to PACE satellite products.

## What is included

- CHLA(z) profile metrics from:
  - BGC-Argo floats (global)
  - OOI Profilers and Moorings
  - IOOS gliders (280 Slocum, Spray and Seaglider deployments, mostly coastal)
- Depth range: 0–200 m (10 m bins)
- Some datasets include matchups to:
  - PACE OCI Level-3 Daily products
  - Rrs (multi-band), chlor_a, Kd_490

## Which file should I use?

- **I want in situ profiles only**
  - `CHLA_argo_profiles.parquet`
  - `CHLA_ooi_profiles.parquet`
  - `CHLA_glider_profiles.parquet`

- **I want to compare in situ profiles to PACE**
  - `CHLA_argo_profiles_plus_PACE.parquet`
  - `CHLA_ooi_profiles_plus_PACE.parquet`
  - `CHLA_glider_profiles_plus_PACE.parquet`

- **I want to compare in situ profiles to OLCI, or OLCI to PACE**
  - `CHLA_argo_profiles_plus_OLCI.parquet`
  - `CHLA_ooi_profiles_plus_OLCI.parquet`
  - `CHLA_glider_profiles_plus_OLCI.parquet`

## Platforms

| Platform | CHLA measurement | Description |
|--------|------------------|-------------|
| BGC-Argo | Fluorescence-based chlorophyll-a | Autonomous profiling floats (global coverage) |
| OOI | Fluorescence-based chlorophyll-a | Fixed moorings and mobile profilers |
| Glider | Fluorescence-based chlorophyll-a | IOOS Glider DAC Slocum, Spray and Seagliders: US East and West coasts, British Columbia, Gulf of Alaska, Irminger Sea, Bermuda, Great Lakes, Antarctic Peninsula (none in the Gulf of Mexico) |

## Processing (rebuild of November 2026)

- **Dates:** 2024-03-01 to 2026-10-31.
- **BGC-Argo:** ascending profiles only (`DIRECTION == "A"`). `CHLA_ADJUSTED`
  levels with `CHLA_ADJUSTED_QC` in {1, 2, 5, 8}, filtered before binning. Bin
  `CHLA_A_B` is the mean of levels with A < PRES <= B, and `CHLA_A_B_N` is how
  many levels went into it. Profiles with no QC-passing level are dropped.
  Source: Ifremer ERDDAP via argopy 1.5.0 (expert mode), pulled 2026-11-06.
- **OOI:** samples with `qc_agg == 1`, one row per instrument per **3 h
  window of local mean solar time** (00–03, 03–06, ... h; solar time = UTC +
  lon/15 h). `time` is the mean time of the window's samples, so each row has
  its own solar hour. Bins are A < depth <= B, as for Argo. ERDDAP depths are
  whole metres, so the edge rule matters. Fixed-depth instruments that publish
  z = 0 get their true depth. CHLA is divided by the Sauzède et al.
  fluorescence:chlorophyll ratio for the site (column `chla_ratio`).
  `n_samples` is the number of samples in the window.
- **Gliders** (added 2026-11-24, extended 2026-11-25 to 280 deployments):
  every IOOS Glider DAC deployment with chlorophyll since 2024-03 was scanned
  on every 4th day of its record (`src/chla_z_glider_candidates.csv`); the
  table keeps those that passed or need a look (`scan_check`) and those that
  failed only for a deep offset, listed in `src/chla_z_glider_selection.json`.
  Deployments with a negative offset, flat or signal-free profiles, surface
  below deep values, or implausible values are left out. **QC columns:**
  `deep_offset_flag` is True for every row of a deployment whose fluorometer
  reads > 0.2 below 300 m (mostly California Current Sprays), so
  `df[~df.deep_offset_flag]` removes them; `scan_check` holds the scan's notes
  (e.g. "jumpy surface", "sparse depths"), empty when the deployment passed.
  Windows where the whole column failed (150–200 m value > deployment median
  + 0.3, or every bin < 0.02; gliders diving to 300 m) were removed. Samples
  pass the dataset's own CHLA flag where it has one (1, 2, or 0 = "no QC
  performed"); about half the datasets have no flag. Rows are 3 h
  windows of local solar time, as OOI, but a glider moves 10–80 km a day, so
  each window keeps **one** glider profile (the one with the most filled
  bins) instead of averaging. Bins A < depth <= B; shelf profiles fill only
  the bins above the bottom. `CHLA_A_B` is divided by the Sauzède ratio
  (`chla_ratio`), as OOI. The table also keeps the fluorometer values
  (`CHLA_FLUOR_A_B`) and `dark_offset`, the fluorometer reading below 300 m
  (NaN if the glider does not reach 300 m). `dark_offset` is **not**
  subtracted: on the first 20 deployments doing so doubled the night
  PACE/glider ratio (1.19 to 1.99; Argo about 1.25; 1.20 on the 280 without
  it). To apply it: `(CHLA_FLUOR_A_B - dark_offset) / chla_ratio`.
  `shore_km` is the distance to the coast; PACE fails near some coasts
  (chlor_a of 8–27 with Rrs(442) near zero), so filter near-shore rows. The
  one Great Lakes deployment (`region == "Great Lakes"`) gets the Sauzède
  ratio of the nearest ocean cell, which is not meant for lakes.

## Matchup details (for +PACE files)

PACE and OLCI Level-3 daily maps composite a day's Level-2 swaths and carry no
per-pixel observation time. A matchup pairs a profile with the daily map that
covers its time; it is not a match to the satellite overpass. Time-of-day
effects on in situ fluorescence (daytime quenching) are left to the model,
through the solar hour of the in situ sample.

- Products: PACE OCI L3M Daily 4 km **V3.2**: Rrs from `PACE_OCI_L3M_AOP`,
  chlor_a from `PACE_OCI_L3M_BGC`, Kd_490 from `PACE_OCI_L3M_KD` (band
  489.674 nm). NRT files are used only for days with no standard file.
- Time: a profile matches a granule when its time is within 24 h after the
  granule's `time_coverage_start`.
- Space: nearest 4 km grid cell.
- Rows: one per profile with a matching Rrs granule. Rrs is NaN where the
  pixel was masked (cloud, glint), so filter on Rrs before modelling.
- Rrs band names are rounded band centres (`pace_Rrs_413`, ...). The source
  file of each value is in `pace_Rrs_file`.

## Matchup details (for +OLCI files)

- Product: Sentinel-3A and -3B OLCI L3 mapped daily 4 km Rrs
  (`OLCIS3A_L3m_ERR_RRS`, `OLCIS3B_L3m_ERR_RRS`), 11 bands from 400 to 709 nm.
  Files exist up to 2026-10-01.
- Every profile is matched, with or without PACE. Rows are profiles on a day
  with an OLCI file. `pace_granule` and `pace_Rrs_valid` say whether the
  profile also has a PACE match. Select rows with finite `olci_Rrs_*` and
  `pace_Rrs_valid` to compare the two sensors on the same profiles.
- Time: the OLCI file of the profile's UTC day. Space: nearest 4 km cell.
- S3A is read first. When any of the day's profiles has NaN S3A `Rrs_443`, S3B
  is read for all of them, and each band takes S3A where finite, else S3B.
  `olci_sat` names the sensor that gave `Rrs_443`. These are the rules of
  Nabir Mamnun's `olci_l3_matchups.ipynb` (fish-pace/2026-go-bgc-hackweek),
  and this table reproduces his rows.

## How to rebuild

```bash
cd datasets/chla_z
python build.py argo-fetch --workers 2   # skips tiles already in _temp_data/raw/
python build.py argo
python build.py ooi-fetch && python build.py ooi
python build.py glider-fetch && python build.py glider   # selection: src/chla_z_glider_selection.json
python build.py pace --product Rrs   # also chlor_a, Kd
python build.py pace-merge
python build.py olci && python build.py olci-merge
```

Each stage is resumable. If it crashes or a server times out, run the same
command again. Code: `src/chla_z_build.py`, `src/chla_z_glider.py`,
`src/chla_z_tables.py`. Design notes: `claude/notes/chla-z-rebuild-2026-11.md`,
`claude/notes/chla-z-glider.md`. Add `--platform glider` to the `pace`,
`pace-merge`, `olci` and `olci-merge` stages to (re)build only the glider
tables. The previous files are at tag
`2025_11_01`.

## How to get started

The notebooks in `notebooks/` show how the tables are made, each on a small
real sample that it checks against the released table. They are data
methods, not model training; the full build runs from `build.py`.

| Notebook | What it shows |
|---|---|
| `01-argo-retrieval-qc.ipynb` | BGC-Argo retrieval, QC, 10 m depth bins |
| `02-ooi-retrieval-qc.ipynb` | OOI retrieval, QC, 3 h solar-time windows, Sauzède correction |
| `03-pace-matchups.ipynb` | PACE Rrs, chlor_a and Kd matchups |
| `04-olci-matchups.ipynb` | Sentinel-3 OLCI Rrs matchups |
| `05-compare-datasets.ipynb` | how in situ and satellite chlorophyll compare; the effect of solar hour; OLCI vs PACE |
| `06-glider-retrieval-qc.ipynb` | Glider selection, retrieval, QC, one profile per 3 h solar window, dark offset, Sauzède correction |

The earlier notebooks (`argopy.ipynb`, `ooi.ipynb`, `argo-matchups.ipynb`)
are kept at tag
[`2026_11_23`](https://github.com/fish-pace/fish-pace-datasets/tree/2026_11_23/datasets/chla_z/notebooks).

## Provenance and metadata

Formal metadata for each file is provided via STAC:
- See `collection.json`
- See individual item JSONs in `items/`

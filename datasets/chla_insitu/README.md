# CHLA in situ profile datasets

This dataset group contains chlorophyll-a (CHLA) vertical profile metrics derived
from in situ observations, with optional matchups to PACE satellite products.

## What is included

- CHLA(z) profile metrics from:
  - BGC-Argo floats (global)
  - OOI Profilers and Moorings
- Depth range: 0–200 m (10 m bins)
- Some datasets include matchups to:
  - PACE OCI Level-3 Daily products
  - Rrs (multi-band), chlor_a, Kd_490

## Which file should I use?

- **I want in situ profiles only**
  - `CHLA_argo_profiles.parquet`
  - `CHLA_ooi_profiles.parquet`

- **I want to compare in situ profiles to PACE**
  - `CHLA_argo_profiles_plus_PACE.parquet`
  - `CHLA_ooi_profiles_plus_PACE.parquet`

## Platforms

| Platform | CHLA measurement | Description |
|--------|------------------|-------------|
| BGC-Argo | Fluorescence-based chlorophyll-a | Autonomous profiling floats (global coverage) |
| OOI | Fluorescence-based chlorophyll-a | Fixed moorings and mobile profilers |

## Matchup details (for +PACE files)

- Time window: ±24 hours
- Spatial method: nearest 4 km PACE pixel
- Satellite products: PACE OCI L3 Rrs v3.1 Daily

## How to get started

Start with one of these notebooks:
- `notebooks/argopy.ipynb`
- `notebooks/argo-matchups.ipynb`
- `notebooks/ooi.ipynb`

## Provenance and metadata

Formal metadata for each file is provided via STAC:
- See `collection.json`
- See individual item JSONs in `items/`

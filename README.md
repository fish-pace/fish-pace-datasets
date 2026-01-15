# Fish-PACE Tutorial Datasets

This repository contains curated datasets used in Fish-PACE hackweek tutorials,
organized by scientific theme and documented using STAC.

## Quick start (5 minutes)

1. Browse available datasets interactively: [STAC Browser link here]()
2. Pick a dataset group (e.g. CHLA in situ profiles)
3. Open a starter notebook from that dataset’s folder

## Dataset groups

- **CHLA in situ profiles**  
  Chlorophyll-a vertical profiles from various platforms, with optional
  matchups to PACE satellite products.
  → `datasets/chla_insitu/`

- **SEAMAP data**  
  Species-resolved total fish biomass observations.
  → `datasets/SEAMAP/`

## How datasets are organized

Each dataset group contains:
- `README.md` — human-friendly description
- `collection.json` — STAC metadata for discovery
- `items/` — one STAC item per data file
- `data/` — the actual data files
- `notebooks/` — example/tutorial notebooks



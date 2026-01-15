# SEAMAP Dataset (Gulf of Mexico)

This dataset contains station-based biomass and environmental measurements from **SEAMAP surveys in the Gulf of Mexico**, harmonized into a single “master table” for analysis, visualization, and modeling exercises.

The dataset is distributed as a columnar Parquet file and indexed in the Fish-PACE STAC catalog for discovery and reuse.

---

## Dataset Overview

**Source:** Southeast Area Monitoring and Assessment Program (SEAMAP)  
**Region:** Gulf of Mexico  
**Data type:** In situ plankton and environmental observations  
**Format:** Parquet  
**Spatial support:** Station locations (lat/lon)  
**Temporal support:** Survey date (`Loc_date`)

Each row represents a sampling station with:
- Station and cruise metadata
- Environmental conditions
- Total live biomass
- Species- or taxon-specific biomass values stored in `BIO_*` columns

---

## Key Variables

### Station metadata
- `STATIONID`
- `CRUISEID`
- `Loc_date`
- `Latitude`, `Longitude`

### Environmental measurements
- Temperature: `TEMP_SSURF`, `TEMP_BOT`, `TEMPSURF`, `TEMPMID`, `TEMPMAX`
- Salinity: `SALSURF`, `SALMID`, `SALMAX`
- Chlorophyll: `CHLORSURF`, `CHLORMID`, `CHLORMAX`
- Oxygen: `OXYSURF`, `OXYMID`, `OXYMAX`
- Turbidity: `TURBSURF`, `TURBMID`, `TURBMAX`
- Wind: `WIND_SPD`, `WIND_DIR`
- Additional water quality and sampling metadata

### Biomass variables

- `TOT_LIVE` — total live biomass
- `BIO_*` — biomass by species or taxonomic group

Each `BIO_#########` column corresponds to a SEAMAP taxon code.  
Values represent biomass measured at the station for that taxon.

> **Note:** Taxonomic resolution varies by code. Users should aggregate or filter taxa based on analysis goals.

---

## Data Files

- **Parquet:**  
  `seamap_with_species.parquet`

The Parquet format enables efficient loading, filtering, and analysis in both Python and R.

---

## How to Load the Data

### Python

```python
import pandas as pd

url = "https://raw.githubusercontent.com/fish-pace/fish-pace-datasets/main/datasets/SEAMAP/data/seamap_with_species.parquet"
df = pd.read_parquet(url)

df.head()
```

To inspect metadata:

```python
import pyarrow.parquet as pq
import fsspec

with fsspec.open(url, "rb") as f:
    table = pq.read_table(f)

table.schema.metadata
```

---

### R

```r
library(arrow)

url <- "https://raw.githubusercontent.com/fish-pace/fish-pace-datasets/main/datasets/SEAMAP/data/seamap_with_species.parquet"
df <- read_parquet(url)

head(df)
```

---

## STAC Catalog

This dataset is indexed in the Fish-PACE STAC catalog:

- **Collection:** `seamap-gom`
- **Item:** `seamap_with_species`

The STAC metadata provides:
- Spatial and temporal extent
- Variable groupings
- Direct links to the data file
- Links to example notebooks

You can explore the dataset interactively using **STAC Browser**.

## How to access the data

All datasets in this group are provided as Parquet or NetCDF files and can be
loaded directly from GitHub using standard Python and R tools.

### Python (Parquet)

```python
import pandas as pd

url = "https://raw.githubusercontent.com/fish-pace/fish-pace-datasets/main/datasets/chla_insitu/data/CHLA_argo_profiles.parquet"
df = pd.read_parquet(url)
df.head()
```

### Inspect embedded metadata (optional, Python)

```python
import fsspec
import pyarrow.parquet as pq

with fsspec.open(url, "rb") as f:
    table = pq.read_table(f)

table.schema.metadata
```

### Python (NetCDF)

```python
import xarray as xr

url = "https://raw.githubusercontent.com/fish-pace/fish-pace-datasets/main/datasets/chla_insitu/data/argo_na_2024_03.nc"
ds = xr.open_dataset(url)
ds
```

### R (Parquet)

```r
library(arrow)

url <- "https://raw.githubusercontent.com/fish-pace/fish-pace-datasets/main/datasets/chla_insitu/data/CHLA_argo_profiles.parquet"
df <- read_parquet(url)
head(df)
```

### R (NetCDF)

```r
library(ncdf4)

url <- "https://raw.githubusercontent.com/fish-pace/fish-pace-datasets/main/datasets/chla_insitu/data/argo_na_2024_03.nc"
nc <- nc_open(url)
print(nc)
nc_close(nc)
```

> Tip: You do **not** need to download files locally — all examples stream data
> directly from GitHub.

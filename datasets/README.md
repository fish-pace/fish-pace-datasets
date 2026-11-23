## How to access the data

All datasets in this group are provided as Parquet or NetCDF files and can be
loaded directly from GitHub using standard Python and R tools.

### Python (Parquet)

```python
import pandas as pd

url = "https://raw.githubusercontent.com/fish-pace/fish-pace-datasets/main/datasets/chla_z/data/CHLA_argo_profiles.parquet"
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

# "#mode=bytes" makes netCDF read the file over plain HTTP; without it the
# URL is taken as an OPeNDAP server and the open fails.
url = "https://raw.githubusercontent.com/fish-pace/fish-pace-datasets/main/datasets/chla_z/data/fluor_to_chla_lut_127630.nc#mode=bytes"
ds = xr.open_dataset(url)
ds
```

### R (Parquet)

```r
library(arrow)

url <- "https://raw.githubusercontent.com/fish-pace/fish-pace-datasets/main/datasets/chla_z/data/CHLA_argo_profiles.parquet"
df <- read_parquet(url)
head(df)
```

### R (NetCDF)

```r
library(ncdf4)

# "#mode=bytes": read the file over plain HTTP, not as an OPeNDAP server
url <- "https://raw.githubusercontent.com/fish-pace/fish-pace-datasets/main/datasets/chla_z/data/fluor_to_chla_lut_127630.nc#mode=bytes"
nc <- nc_open(url)
print(nc)
nc_close(nc)
```

> Tip: You do **not** need to download files locally — all examples stream data
> directly from GitHub.

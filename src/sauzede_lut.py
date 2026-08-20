#!/usr/bin/env python3
"""Look up the Sauzède fluorescence:chla correction ratio for a lat/lon.

Fluorescence-based chlorophyll-a from WET Labs ECO sensors — the sensor family
flown on BGC-Argo floats and deployed on OOI moorings and profilers — reads high
against HPLC chlorophyll by a factor that varies regionally. Argo corrects for
this in `CHLA_ADJUSTED` using the lookup table shipped here; OOI CHLA in this
repository is factory-calibrated and *uncorrected*. To put the two on a common
basis:

    chla_corrected = chla_fluor / ratio

The LUT (`datasets/chla_z/data/fluor_to_chla_lut_127630.nc`) is the
climatological physiological ratio between fluorescence-derived and true
chlorophyll-a on a 1°x1° grid, plus its temporal standard deviation. The
correction is strongly site-dependent — for the OOI sites in `chla_z` it ranges
from 0.77 (Endurance OR offshore, where correction *raises* CHLA) to 2.49
(Irminger Sea).

Lookup follows the file's own `usage` attribute: nearest neighbour per axis,
ties broken toward the smaller coordinate, lon in [-180, 180).

Usage
-----
    from sauzede_lut import chla_scaling_factor, load_lut

    lut = load_lut()                       # load once, reuse
    ratio = chla_scaling_factor(59.9, -39.5, lut=lut)
    ratio, sd = chla_scaling_factor(lat_array, lon_array, lut=lut, with_sd=True)

`load_lut()` reads the copy in this repository when it is available and
otherwise streams it from GitHub, so the module works from a clone or from a
standalone notebook.

Data attribution
----------------
Sauzède, R. (2025). Real-Time Physiological Correction Ratio between
Fluorescence and Chlorophyll-a Dataset. SEANOE. doi:10.17882/105732.
Sorbonne Université/CNRS. Licensed CC-BY-4.0.

References: Sauzède et al., 2016 (doi:10.1002/2015JC011408); Renosh et al.,
2023 (doi:10.3390/rs15245663); Sauzède et al., 2025 (doi:10.48670/moi-00046).

Acknowledgment: This dataset was generated using E.U. Copernicus Marine Service
Products, which are derived from OneArgo data collected and made freely
available by the International Argo Program and its contributing national
programs (https://argo.ucsd.edu, https://www.ocean-ops.org). The Argo Program
is a component of the Global Ocean Observing System (GOOS).
"""

import argparse
import functools
import io
import urllib.request
from pathlib import Path

import numpy as np
import xarray as xr

LUT_FILENAME = "fluor_to_chla_lut_127630.nc"
LUT_PATH = Path(__file__).resolve().parent.parent / "datasets" / "chla_z" / "data" / LUT_FILENAME
LUT_URL = (
    "https://raw.githubusercontent.com/fish-pace/fish-pace-datasets/main/"
    f"datasets/chla_z/data/{LUT_FILENAME}"
)


def _open_lut(source: Path | str) -> xr.Dataset:
    """Open the LUT from a local path or an http(s) URL, -9999 fill decoded to NaN.

    The file is netCDF *classic* (CDF-1), not netCDF4/HDF5, so it cannot be read
    with the h5netcdf engine. Remote copies are read into memory (1 MB) and handed
    to xarray as a buffer, which selects a working engine on its own and needs no
    HTTP range-request support.
    """
    if str(source).startswith(("http://", "https://")):
        with urllib.request.urlopen(str(source)) as response:
            return xr.open_dataset(io.BytesIO(response.read()))
    return xr.open_dataset(source)


@functools.lru_cache(maxsize=1)
def _load_bundled_lut() -> xr.Dataset:
    """Open this repository's LUT — the local copy if present, else stream it."""
    return _open_lut(LUT_PATH if LUT_PATH.exists() else LUT_URL)


def load_lut(source: Path | str | None = None) -> xr.Dataset:
    """Load the LUT. With no argument, use this repository's copy (cached)."""
    if source is None:
        return _load_bundled_lut()
    return _open_lut(source)


def _nearest_index(coord: np.ndarray, first: int, n: int) -> np.ndarray:
    """Nearest integer grid index, ties toward the smaller value, clamped to grid."""
    idx = np.ceil(np.asarray(coord, dtype=float) - 0.5).astype(int) - first
    return np.clip(idx, 0, n - 1)


def chla_scaling_factor(
    lat: float | np.ndarray,
    lon: float | np.ndarray,
    lut: xr.Dataset | None = None,
    with_sd: bool = False,
) -> np.ndarray | float | tuple:
    """Return the fluorescence:chla ratio at (lat, lon); NaN where the LUT has no value.

    Accepts scalars or arrays (broadcast together). With with_sd=True returns
    (ratio, temporal_sd), where the standard deviation describes how much the
    ratio varies over time within that 1°x1° cell.

    Pass `lut=load_lut()` when calling repeatedly to avoid reopening the file.
    """
    if lut is None:
        lut = load_lut()
    lat, lon = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float))
    if np.any((lat < -90) | (lat > 90)):
        raise ValueError("latitude outside [-90, 90]")
    lon = ((lon + 180.0) % 360.0) - 180.0  # normalize to [-180, 180)
    ilat = _nearest_index(lat, int(lut.latitude[0]), lut.sizes["latitude"])
    ilon = _nearest_index(lon, int(lut.longitude[0]), lut.sizes["longitude"])
    ratio = lut.fluorescence_chlorophyll_ratio.values[ilat, ilon]
    if with_sd:
        return ratio[()], lut.sd_fluorescence_chlorophyll_ratio.values[ilat, ilon][()]
    return ratio[()]


def main(lat: float, lon: float, lut_path: str | None) -> None:
    lut = load_lut(lut_path)
    ratio, sd = chla_scaling_factor(lat, lon, lut=lut, with_sd=True)
    print(f"lat={lat} lon={lon}  fluorescence:chla ratio={ratio:.4f}  sd={sd:.4f}")
    if np.isnan(ratio):
        print("WARNING: no LUT value at this cell (land or no coverage)")


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument(
        "--lut-path",
        type=str,
        default=None,
        help="local path or http(s) URL to the LUT (default: this repository's copy)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_opt()))

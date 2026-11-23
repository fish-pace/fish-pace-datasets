"""Write the four CHLA-Z tables in datasets/chla_z/data/ with their parquet metadata.

Kept apart from chla_z_build.py so the processing and the description of it can
be reviewed separately. The text below starts from the metadata in the
2025_11_01 release; keys that the rebuild changed were rewritten, the rest
(sensor descriptions, PACE algorithm references) are carried over unchanged.
Each table also gets a `build_*` block recording the code commit, the source
snapshot dates and package versions, so a file can be traced back to how it
was made.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import chla_z_build as b

CREATOR = "Eli Holmes / NOAA https://orcid.org/0000-0001-9128-8393"
ARGO_SNAPSHOT = "2026-11-06"  # Ifremer ERDDAP fetch of the monthly files
ARGO_SOURCE = (
    "BGC-Argo synthetic profiles from the Ifremer BGC-Argo ERDDAP "
    "(https://erddap.ifremer.fr/erddap), fetched with argopy 1.5.0 "
    "(DataFetcher ds='bgc', src='erddap', mode='expert') on 2026-11-06. "
    "Argo (2000). Argo float data and metadata from Global Data Assembly Centre "
    "(Argo GDAC). SEANOE. https://doi.org/10.17882/42182"
)
PACE_SOURCE = (
    "PACE OCI Level 3 mapped daily 4 km, version 3.2: PACE_OCI_L3M_AOP (Rrs), "
    "PACE_OCI_L3M_BGC (chlor_a) and PACE_OCI_L3M_KD (Kd), and their _NRT "
    "collections where no standard granule exists, via NASA Earthdata (earthaccess). "
    "Collections V3.1 and earlier were removed from CMR before this rebuild."
)
OOI_SOURCE = (
    "NSF Ocean Observatories Initiative. (2014). Data Explorer ERDDAP "
    "https://erddap.dataexplorer.oceanobservatories.org/erddap, data from "
    "2024-03-01 to 2026-10-31. See profile_id for the dataset used."
)
COVERAGE = {
    "spatiotemporal_coverage_time_start": "2024-03-01T00:00:00Z",
    "spatiotemporal_coverage_time_end": "2026-10-31T23:59:59Z",
    "spatiotemporal_coverage_lat_min": "-90.0",
    "spatiotemporal_coverage_lat_max": "90.0",
    "spatiotemporal_coverage_lon_min": "-180.0",
    "spatiotemporal_coverage_lon_max": "180.0",
}

# ---- Argo text ------------------------------------------------------------

ARGO_CORE = {
    "creator": CREATOR,
    "profile_id_definition": "profile_id = PLATFORM_NUMBER (7 digits) + '_' + CYCLE_NUMBER (4 digits)",
    "PLATFORM_NUMBER_definition": "PLATFORM_NUMBER from BGC-Argo identifying the float.",
    "CYCLE_NUMBER_definition": (
        "CYCLE_NUMBER from BGC-Argo. Only the ascending profile (DIRECTION = 'A') "
        "of each cycle is used, so each row is one ascending profile."),
    "TIME_definition": "TIME in UTC of the ascending profile (constant within a profile).",
    "LATITUDE_definition": "LATITUDE of the ascending profile. POSITION_QC 1 and 8 (interpolated) are both kept.",
    "LONGITUDE_definition": "LONGITUDE of the ascending profile.",
    "CHLA_A_B_definition": (
        "Mean of the CHLA_ADJUSTED levels of the ascending profile with "
        "CHLA_ADJUSTED_QC in {1, 2, 5, 8} and A < PRES <= B (PRES in dbar, about m). "
        "NaN where no level passes."),
    "CHLA_A_B_N_definition": "Number of QC-passing CHLA_ADJUSTED levels used for CHLA_A_B.",
    "variable_LATITUDE_standard_name": "latitude",
    "variable_LATITUDE_units": "degrees_north",
    "variable_LONGITUDE_standard_name": "longitude",
    "variable_LONGITUDE_units": "degrees_east",
    "variable_TIME_standard_name": "time",
    "variable_TIME_units": "UTC",
    "variable_CHLA_A_B_standard_name": "mass_concentration_of_chlorophyll_a_in_sea_water",
    "variable_CHLA_A_B_units": "mg m-3",
    "variable_CHLA_A_B_N_long_name": "count of QC-passing CHLA_ADJUSTED levels in each depth bin",
    "variable_CHLA_A_B_N_units": "1",
    "CHLA_processing_description": (
        "CHLA is the BGC-Argo variable CHLA_ADJUSTED (mg m-3): dark-count and "
        "Sauzède fluorescence:chla corrections and non-photochemical quenching "
        "adjustment as applied by the Argo data centres. Descending profiles were "
        "dropped (about 5% of cycles also sample on descent; only 12% of those "
        "profiles had any QC-passing level). Levels were kept when "
        "CHLA_ADJUSTED_QC is 1, 2, 5 or 8, then averaged in 10 m pressure bins "
        "from 0 to 200 dbar. Profiles with no QC-passing level in 0-200 dbar "
        "were dropped. Most CHLA_ADJUSTED values were recalibrated by the Argo "
        "data centres in June-August 2026; this table uses the 2026-11-06 snapshot."),
    "CHLA_measurement_description": (
        "BGC-Argo chlorophyll-a (CHLA) is measured using a submersible chlorophyll "
        "fluorometer mounted on the float. The sensor emits blue light (~470 nm) and "
        "detects the resulting chlorophyll fluorescence near ~695 nm. Fluorescence "
        "intensity is converted onboard to chlorophyll-a concentration using factory "
        "calibration coefficients and reported in mg m-3 as the raw 'CHLA' variable. "
        "Additional processing recommended by the BGC-Argo community (e.g., "
        "non-photochemical quenching correction, dark-count correction, and "
        "delayed-mode quality-control adjustments) is applied in CHLA_ADJUSTED, "
        "which is the variable used here."),
    "license": "Open access (Argo Data Policy); unrestricted use with attribution.",
}

# ---- OOI text -------------------------------------------------------------

OOI_CORE = {
    "creator": CREATOR,
    "profile_id_definition": "profile_id = ERDDAP dataset_id + '_' + YYYYMMDD (UTC day).",
    "time_definition": (
        "Mean time (UTC) of the QC-passing observations in that day. Note: for "
        "moorings that sample all day this is close to 12:00 UTC."),
    "lat_definition": "lat in the ERDDAP dataset (first value of the day). OOI platforms are fixed.",
    "lon_definition": "lon in the ERDDAP dataset (first value of the day).",
    "CHLA_A_B_definition": (
        "Daily mean of QC-passing CHLA (qc_agg = 1) with A <= depth < B, where "
        "depth = -z in m, divided by chla_ratio (see CHLA_processing_description)."),
    "chla_ratio_definition": (
        "Sauzède fluorescence:chla ratio at the site (1 deg LUT, nearest cell). "
        "Raw fluorometer CHLA = CHLA_A_B * chla_ratio. NaN where the LUT has no "
        "value; those rows are uncorrected (ratio 1 applied)."),
    "description_definition": "OOI platform and instrument description for the dataset.",
    "variable_lat_standard_name": "latitude",
    "variable_lat_units": "degrees_north",
    "variable_lon_standard_name": "longitude",
    "variable_lon_units": "degrees_east",
    "variable_time_standard_name": "time",
    "variable_time_units": "UTC",
    "variable_CHLA_standard_name": "mass_concentration_of_chlorophyll_a_in_sea_water",
    "variable_CHLA_units": "mg m-3",
    "CHLA_processing_description": (
        "CHLA is the OOI variable 'mass_concentration_of_chlorophyll_a_in_sea_water' "
        "for moorings and 'mass_concentration_of_chlorophyll_a_in_sea_water_profiler_depth_enabled' "
        "for profilers (mg m-3), filtered to qc_agg = 1. For nine fixed-depth "
        "datasets that publish z = 0 the known sensor depth is used instead. "
        "Values were averaged per UTC day in 10 m depth bins from 0 to 200 m. "
        "To put OOI on the same calibration basis as Argo CHLA_ADJUSTED, values "
        "were divided by the Sauzède fluorescence:chla ratio (Sauzède, R. (2025), "
        "SEANOE, doi:10.17882/105732, CC-BY-4.0)."),
    "CHLA_measurement_description": (
        "OOI chlorophyll-a (CHLA) is measured with WET Labs ECO fluorometers (e.g., "
        "FLORT/FLORD family) deployed on Ocean Observatories Initiative platforms. "
        "These sensors emit blue light (~470 nm) to excite chlorophyll pigments in "
        "the water and detect the resulting fluorescence in the red band near ~695 nm. "
        "Raw fluorescence counts are converted to chlorophyll-a concentration using "
        "factory calibration coefficients and OOI processing, which applies standard "
        "scale factors, dark corrections, and automated quality-control tests to "
        "produce the 'fluorometric_chlor_a' data product (reported in approximately "
        "mg m-3). Here those values are then divided by the Sauzède ratio."),
    "license": "Open access; unrestricted use with attribution.",
}

# ---- PACE text (shared) ---------------------------------------------------

PACE_CORE = {
    "pace_Rrs_lat_definition": "The matched latitude from the PACE Level 3 grid.",
    "pace_Rrs_lon_definition": "The matched longitude from the PACE Level 3 grid.",
    "pace_Rrs_t_start_definition": "time_coverage_start attribute of the matched PACE Level 3 file (UTC).",
    "pace_Rrs_t_end_definition": "time_coverage_end attribute of the matched PACE Level 3 file (UTC).",
    "pace_Rrs_file_definition": (
        "File name of the matched PACE_OCI_L3M_AOP file. chlor_a and Kd_490 come "
        "from the BGC and KD files of the same day and stream (standard or NRT); "
        "where that file is missing they are NaN."),
    "pace_Rrs_A_definition": (
        "Rrs from PACE_OCI_L3M_AOP V3.2 daily 4 km for the band whose centre "
        "rounds to A nm. V3.2 band centres are fractional (e.g. 346.017); "
        "round() maps all 172 bands one-to-one onto the integer names used in "
        "the 2025 release."),
    "pace_chlor_a_definition": "chlor_a from PACE_OCI_L3M_BGC V3.2 daily 4 km.",
    "pace_Kd_490_definition": "Kd from PACE_OCI_L3M_KD V3.2 daily 4 km at the band nearest 490 nm ({kd_wl} nm).",
    "pace_matching_description": (
        "A profile matches a PACE file when its time is within 24 h after the "
        "file's time_coverage_start attribute (not the date in the file name). "
        "Its lat/lon is matched to the nearest PACE 4 km grid cell. When several "
        "files match, standard is preferred over NRT, then the file whose time "
        "window centre is closest to the profile time. Only profiles with a "
        "matching Rrs file are kept; Rrs may still be NaN (clouds, glint, "
        "flags). NRT files are used only from 2026-04-01, mostly for October 2026, "
        "where no standard V3.2 file exists yet."),
    "variable_pace_Rrs_lat_standard_name": "latitude",
    "variable_pace_Rrs_lat_units": "degrees_north",
    "variable_pace_Rrs_lon_standard_name": "longitude",
    "variable_pace_Rrs_lon_units": "degrees_east",
    "variable_pace_Rrs_t_start_units": "UTC",
    "variable_pace_Rrs_t_end_units": "UTC",
    "variable_pace_Rrs_prefix_definition": (
        "All columns whose names start with 'pace_Rrs_' followed by a number "
        "contain PACE OCI remote-sensing reflectance (Rrs) at that wavelength in nm."),
    "variable_pace_Rrs_long_name": "Remote sensing reflectance",
    "variable_pace_Rrs_standard_name": "surface_ratio_of_upwelling_radiance_emerging_from_sea_water_to_downwelling_radiative_flux_in_air",
    "variable_pace_Rrs_units": "sr-1",
    "Rrs_measurement_definition": (
        "Remote-sensing reflectance (Rrs) is the ratio of water-leaving radiance to "
        "downwelling irradiance just above the sea surface (sr-1). Rrs(λ) from PACE "
        "OCI is derived from top-of-atmosphere radiances after atmospheric correction "
        "that removes molecular scattering, aerosol scattering, and surface glint "
        "contributions. Rrs represents the spectral signature of the ocean water "
        "column and is the primary input for bio-optical algorithms."),
    "pace_chlor_a_long_name": "Chlorophyll Concentration, OCI Algorithm",
    "pace_chlor_a_standard_name": "mass_concentration_of_chlorophyll_a_in_sea_water",
    "pace_chlor_a_units": "mg m-3",
    "pace_chlor_a_measurement_definition": (
        "PACE OCI chlorophyll-a concentration (chlor_a) uses the OCI Chlorophyll "
        "Algorithm (CI algorithm) of Hu, Lee, and Franz (2012), a three-band "
        "reflectance-difference approach blended with band-ratio OCx for higher "
        "concentrations. Hu, C., Lee Z., and Franz, B.A. (2012). 'Chlorophyll-a "
        "algorithms for oligotrophic oceans: A novel approach based on three-band "
        "reflectance difference,' J. Geophys. Res., 117, C01011."),
    "pace_Kd_490_standard_name": "attenuation_of_downwelling_radiative_flux_in_sea_water",
    "pace_Kd_490_units": "m-1",
    "pace_Kd_490_measurement_definition": (
        "Diffuse attenuation coefficient for downwelling irradiance near 490 nm, "
        "retrieved from PACE OCI Rrs. Lee, Z.P., et al. (2005) doi:10.1029/2004JC002573; "
        "Lee, Z., et al. (2013) doi:10.1002/jgrc.20308."),
}


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "-C", str(b.REPO), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "-C", str(b.REPO), "diff", "--quiet", "--", "src", "datasets/chla_z/build.py"])
        return sha + ("+uncommitted" if dirty else "")
    except Exception:
        return "unknown"


def _versions() -> str:
    import importlib.metadata as im
    out = {}
    for p in ["argopy", "earthaccess", "xarray", "pandas", "numpy", "dask", "h5netcdf"]:
        try:
            out[p] = im.version(p)
        except im.PackageNotFoundError:
            pass
    return json.dumps(out)


def _build_block(extra: dict) -> dict:
    blk = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "build_code": "fish-pace/fish-pace-datasets src/chla_z_build.py via datasets/chla_z/build.py",
        "build_commit": _git_commit(),
        "build_package_versions": _versions(),
    }
    blk.update({k: (v if isinstance(v, str) else json.dumps(v, default=str)) for k, v in extra.items()})
    return blk


def _write(df: pd.DataFrame, name: str, meta: dict):
    table = pa.Table.from_pandas(df, preserve_index=False)
    md = dict(table.schema.metadata or {})
    md.update({k.encode(): str(v).encode() for k, v in meta.items()})
    table = table.replace_schema_metadata(md)
    b.atomic_write_parquet(df, b.DATA_DIR / name, table=table)
    b.log(f"wrote {name}: {len(df)} rows, {df.shape[1]} columns")


# ---- writers ---------------------------------------------------------------

def _argo_stats() -> dict:
    tot = {}
    for m in b.MONTHS:
        s = json.loads((b.ARGO_PROFILES_MONTHLY / f"argo_chla_profiles_{m}.stats.json").read_text())
        for k, v in s.items():
            if isinstance(v, int):
                tot[k] = tot.get(k, 0) + v
    return tot


def write_argo_profiles():
    prof = b.argo_combine()
    stats = _argo_stats()
    meta = {
        "title": "Global BGC-Argo CHLA profile metrics (0 to 200 m, 10 m bins)",
        "source": ARGO_SOURCE,
        "description": (
            "All BGC-Argo synthetic profiles with CHLA from 2024-03-01 to 2026-10-31. "
            "Ascending profiles only; CHLA_ADJUSTED with CHLA_ADJUSTED_QC in "
            "{1, 2, 5, 8}, averaged in 10 m pressure bins from 0 to 200 dbar. "
            "One row per ascending profile with at least one QC-passing level; "
            "bins without data are NaN."),
        **ARGO_CORE, **COVERAGE,
        **_build_block({"build_source_snapshot": ARGO_SNAPSHOT, "build_counts": stats}),
    }
    _write(prof, "CHLA_argo_profiles.parquet", meta)
    # Index used by older matchup code; now the same profiles as the table.
    idx = prof[["profile_id", "PLATFORM_NUMBER", "CYCLE_NUMBER", "TIME", "LATITUDE", "LONGITUDE"]]
    b.atomic_write_parquet(idx, b.ARGO_MONTHLY_NC / "argo_profiles_index.parquet")
    return prof


def write_ooi_profiles():
    combined, lut_info = b.ooi_combine()
    meta = {
        "title": "OOI CHLA daily profile metrics (0 to 200 m, 10 m bins)",
        "source": OOI_SOURCE,
        "description": (
            "OOI fluorometer CHLA from profilers and fixed-depth instruments, "
            "2024-03-01 to 2026-10-31, from the OOI Data Explorer ERDDAP. One row "
            "per dataset and UTC day. QC qc_agg = 1, daily means in 10 m depth "
            "bins, Sauzède fluorescence:chla correction applied."),
        **OOI_CORE, **COVERAGE,
        **_build_block({"build_ooi_fetch_dates": _ooi_fetch_dates(),
                        "build_sauzede_sites_without_lut": lut_info["sites_without_lut"]}),
    }
    _write(combined, "CHLA_ooi_profiles.parquet", meta)
    b.write_marker(b.TEMP_DIR / "logs" / "ooi_lut_by_site.json", lut_info)
    return combined


def _ooi_fetch_dates() -> str:
    days = set()
    for f in b.OOI_RAW.glob("*/*.csv.gz"):
        days.add(datetime.fromtimestamp(f.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d"))
    return f"{min(days)} to {max(days)}" if days else "none"


def _kd_wavelength() -> str:
    files = sorted((b.MATCH_DIR / "Kd").glob("*.parquet"))[:20]
    wl = {round(float(pd.read_parquet(f, columns=["pace_Kd_wavelength"]).iloc[0, 0]), 3) for f in files}
    return ", ".join(map(str, sorted(wl)))


def write_plus_pace():
    pace_meta = dict(PACE_CORE)
    pace_meta["pace_Kd_490_definition"] = pace_meta["pace_Kd_490_definition"].format(kd_wl=_kd_wavelength())
    out = {}
    for platform, fname, core, src, title in [
        ("argo", "CHLA_argo_profiles", ARGO_CORE, ARGO_SOURCE, "Global BGC-Argo"),
        ("ooi", "CHLA_ooi_profiles", OOI_CORE, OOI_SOURCE, "OOI"),
    ]:
        prof = pd.read_parquet(b.DATA_DIR / f"{fname}.parquet")
        prof_md = pq.read_schema(b.DATA_DIR / f"{fname}.parquet").metadata
        merged, stats = b.pace_merge(prof, platform)
        meta = {
            "title": f"{title} CHLA profile metrics (0 to 200 m, 10 m bins) with PACE Rrs, chlor_a and Kd_490 matchups",
            "source": f"{src} {PACE_SOURCE}",
            "description": prof_md[b"description"].decode() + " Each profile is matched to PACE V3.2 daily 4 km Rrs, chlor_a and Kd_490 (see pace_matching_description).",
            **core, **pace_meta, **COVERAGE,
            **_build_block({"build_pace_merge_counts": stats,
                            **{k.decode(): v.decode() for k, v in prof_md.items()
                               if k.decode().startswith("build_") and k.decode() not in
                               ("build_commit", "build_code", "build_package_versions")}}),
        }
        _write(merged, f"{fname}_plus_PACE.parquet", meta)
        out[platform] = stats
    b.write_marker(b.TEMP_DIR / "logs" / "pace_merge_stats.json", out)
    return out


# ---- before / after ------------------------------------------------------------

def _summ(df: pd.DataFrame, platform: str) -> dict:
    tcol = "TIME" if "TIME" in df.columns else "time"
    t = pd.to_datetime(df[tcol], utc=True)
    if platform == "argo":
        groups = df["PLATFORM_NUMBER"].nunique()
    else:
        groups = df["profile_id"].str.rsplit("_", n=1).str[0].nunique()
    chla = [c for c in b.CHLA_COLS if c in df.columns]
    out = {
        "rows": len(df),
        "unique profile_id": df["profile_id"].nunique(),
        "floats / instruments": groups,
        "non-NaN CHLA bins": int(df[chla].notna().sum().sum()),
        "CHLA_0_10 non-NaN": int(df["CHLA_0_10"].notna().sum()),
        "first": t.min().strftime("%Y-%m-%d"),
        "last": t.max().strftime("%Y-%m-%d"),
    }
    rrs = [c for c in df.columns if c.startswith("pace_Rrs_") and c[-1].isdigit()]
    if rrs:
        out["rows with any Rrs"] = int(df[rrs].notna().any(axis=1).sum())
        out["NRT PACE rows"] = int(df["pace_Rrs_file"].str.contains("NRT").sum())
    return out


def before_after(tag: str = "2025_11_01") -> str:
    """Markdown tables comparing the rebuilt files with release `tag`."""
    import io
    lines = []
    for name, platform in [("CHLA_argo_profiles", "argo"), ("CHLA_argo_profiles_plus_PACE", "argo"),
                           ("CHLA_ooi_profiles", "ooi"), ("CHLA_ooi_profiles_plus_PACE", "ooi")]:
        rel = f"datasets/chla_z/data/{name}.parquet"
        old = pd.read_parquet(io.BytesIO(subprocess.check_output(
            ["git", "-C", str(b.REPO), "show", f"{tag}:{rel}"])))
        new = pd.read_parquet(b.DATA_DIR / f"{name}.parquet")
        so, sn = _summ(old, platform), _summ(new, platform)
        lines += [f"**{name}.parquet**", "", f"| | {tag} | rebuild |", "|---|---:|---:|"]
        for k in sn:
            fo = so.get(k, "")
            fmt = lambda v: f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
            lines.append(f"| {k} | {fmt(fo)} | {fmt(sn[k])} |")
        lines.append("")
    return "\n".join(lines)


# ---- OLCI ------------------------------------------------------------------------

OLCI_SOURCE = (
    "Sentinel-3A and -3B OLCI Level 3 mapped daily 4 km Rrs (NASA OB.DAAC "
    "collections OLCIS3A_L3m_ERR_RRS and OLCIS3B_L3m_ERR_RRS), via NASA Earthdata "
    "(earthaccess). Granules exist from 2024-03-01 to 2026-10-01."
)
OLCI_CORE = {
    "olci_matching_description": (
        "Every in situ profile is matched to the OLCI daily file named for its "
        "UTC calendar day, at the nearest 4 km grid cell. S3A is read first; S3B "
        "is read at all of that day's profiles when any has NaN S3A Rrs_443, and "
        "each band takes S3A where finite, else S3B. These are the rules of "
        "2026-go-bgc-hackweek contributor_folders/nama/olci_l3_matchups.ipynb, "
        "whose rows this table reproduces. Rows: profiles on a day with at least "
        "one OLCI granule; OLCI Rrs is NaN where the pixel is masked."),
    "olci_sat_definition": "Sensor that gave olci_Rrs_443: 'S3A', 'S3B', or null if neither.",
    "olci_day_definition": "UTC day (YYYYMMDD) of the OLCI files used.",
    "olci_S3A_available_definition": "True if at least one S3A band file exists for olci_day.",
    "olci_S3B_available_definition": "True if at least one S3B band file exists for olci_day.",
    "olci_Rrs_A_definition": "OLCI Rrs at the A nm band (sr-1). Bands: 400, 412, 443, 490, 510, 560, 620, 665, 674, 681, 709.",
    "pace_granule_definition": "True if the profile has a row in the matching _plus_PACE table (a PACE V3.2 Rrs file covers its time).",
    "pace_Rrs_valid_definition": (
        "True if that PACE row has at least one finite Rrs band. Use olci_Rrs_* "
        "finite & pace_Rrs_valid to compare OLCI and PACE on the same profiles."),
}


def write_plus_olci():
    out = {}
    for platform, fname, core, src, title in [
        ("argo", "CHLA_argo_profiles", ARGO_CORE, ARGO_SOURCE, "Global BGC-Argo"),
        ("ooi", "CHLA_ooi_profiles", OOI_CORE, OOI_SOURCE, "OOI"),
    ]:
        prof = pd.read_parquet(b.DATA_DIR / f"{fname}.parquet")
        prof_md = pq.read_schema(b.DATA_DIR / f"{fname}.parquet").metadata
        merged, stats = b.olci_merge(prof, platform)
        meta = {
            "title": f"{title} CHLA profile metrics (0 to 200 m, 10 m bins) with Sentinel-3 OLCI Rrs matchups",
            "source": f"{src} {OLCI_SOURCE}",
            "description": prof_md[b"description"].decode() + " Each profile is matched to Sentinel-3 OLCI daily 4 km Rrs (see olci_matching_description), with flags for whether it also has a PACE match.",
            **core, **OLCI_CORE, **COVERAGE,
            "spatiotemporal_coverage_time_end": "2026-10-01T23:59:59Z",
            **_build_block({"build_olci_merge_counts": stats}),
        }
        _write(merged, f"{fname}_plus_OLCI.parquet", meta)
        out[platform] = stats
    b.write_marker(b.TEMP_DIR / "logs" / "olci_merge_stats.json", out)
    return out

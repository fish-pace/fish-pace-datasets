"""Build the CHLA-Z training tables (BGC-Argo and OOI CHLA profiles + PACE matchups).

This module is the single source of the processing used by the CLI driver
(`datasets/chla_z/build.py`) and by the notebooks in `datasets/chla_z/notebooks/`.
Design note: `claude/notes/chla-z-rebuild-2026-11.md`.

Every stage works on independent units (one Argo month, one OOI dataset-month,
one PACE granule) and writes one file per unit, atomically (`.part` + rename).
A unit whose output already exists is skipped, so any stage can be killed and
rerun and picks up where it stopped. Network calls retry with backoff, and a
dropped connection is never mistaken for "no data": only an explicit
"no matching results" answer from ERDDAP, or a PACE granule with no points in
its time window, is written as an `.empty` marker.

Decisions recorded in fish-pace/fish-pace-datasets#4 (Eli Holmes, 2026-11-21):

- BGC-Argo: ascending profiles only (DIRECTION == "A"); CHLA_ADJUSTED with
  CHLA_ADJUSTED_QC in {1, 2, 5, 8} applied *before* binning; profiles with no
  good level in 0-200 dbar are dropped.
- Dates: 2024-03-01 to 2026-10-31 (UTC) for both platforms.
- OOI: Sauzède fluorescence:chla correction applied (chla_ratio column kept).
- PACE: V3.2 daily 4 km collections AOP (Rrs), BGC (chlor_a), KD (Kd_490),
  standard preferred over NRT.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
DS_DIR = REPO / "datasets" / "chla_z"
DATA_DIR = DS_DIR / "data"
# _temp_data is git-ignored; on the PACE hub it is a symlink into shared-public.
TEMP_DIR = DS_DIR / "notebooks" / "_temp_data"
ARGO_MONTHLY_NC = TEMP_DIR / "CHLA_argo_monthly_nc"
ARGO_PROFILES_MONTHLY = TEMP_DIR / "argo_profiles_monthly"
OOI_RAW = TEMP_DIR / "ooi_raw"
OOI_DAILY = TEMP_DIR / "ooi_daily"
MATCH_DIR = TEMP_DIR / "matchups_v3_2"

START = "2024-03-01"
END_EXCLUSIVE = "2026-11-01"  # data through 2026-10-31 UTC
MONTHS = [p.strftime("%Y-%m") for p in pd.period_range(START, "2026-10", freq="M")]

QC_GOOD = (1, 2, 5, 8)
N_BINS = 20  # 0-200 m in 10 m bins
BIN_NAMES = [f"{10 * i}_{10 * (i + 1)}" for i in range(N_BINS)]
CHLA_COLS = [f"CHLA_{b}" for b in BIN_NAMES]

# Rrs band names in the 2025_11_01 tables (V3.1, int-truncated). V3.2 AOP band
# centres are fractional; round() maps all 172 onto these one-to-one (checked
# in `_rrs_labels`). The model's feature names depend on these.
RRS_LABELS = [
    346, 348, 351, 353, 356, 358, 361, 363, 366, 368, 371, 373, 375, 378, 380, 383, 385,
    388, 390, 393, 395, 398, 400, 403, 405, 408, 410, 413, 415, 418, 420, 422, 425, 427,
    430, 432, 435, 437, 440, 442, 445, 447, 450, 452, 455, 457, 460, 462, 465, 467, 470,
    472, 475, 477, 480, 482, 485, 487, 490, 492, 495, 497, 500, 502, 505, 507, 510, 512,
    515, 517, 520, 522, 525, 527, 530, 532, 535, 537, 540, 542, 545, 547, 550, 553, 555,
    558, 560, 563, 565, 568, 570, 573, 575, 578, 580, 583, 586, 588, 613, 615, 618, 620,
    623, 625, 627, 630, 632, 635, 637, 640, 641, 642, 643, 645, 646, 647, 648, 650, 651,
    652, 653, 655, 656, 657, 658, 660, 661, 662, 663, 665, 666, 667, 668, 670, 671, 672,
    673, 675, 676, 677, 678, 679, 681, 682, 683, 684, 686, 687, 688, 689, 691, 692, 693,
    694, 696, 697, 698, 699, 701, 702, 703, 704, 706, 707, 708, 709, 711, 712, 713, 714,
    717, 719,
]


def log(*args):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *args, flush=True)


def atomic_write_parquet(df: pd.DataFrame, path: Path, table=None):
    """Write via `.part` + rename so a killed process never leaves a half file
    that a rerun would mistake for a finished unit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    if table is not None:
        import pyarrow.parquet as pq
        pq.write_table(table, tmp)
    else:
        df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_marker(path: Path, info: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(info, default=str, indent=1))
    os.replace(tmp, path)


def backoff_sleep(attempt: int, base: float = 30.0, cap: float = 900.0):
    time.sleep(min(cap, base * 2 ** attempt) * (0.75 + 0.5 * random.random()))


# --------------------------------------------------------------------------
# BGC-Argo
# --------------------------------------------------------------------------

ARGO_KEY = ["PLATFORM_NUMBER", "CYCLE_NUMBER"]
ARGO_VARS = ARGO_KEY + ["DIRECTION", "TIME", "LATITUDE", "LONGITUDE", "PRES",
                        "CHLA_ADJUSTED", "CHLA_ADJUSTED_QC"]


def bin_index_argo(pres: np.ndarray) -> np.ndarray:
    """10 m bin index for pressure, using the (A, B] convention of the
    original `average_profile()` in argopy.ipynb (depth > z_min & depth <= z_max).
    Returns -1 outside (0, 200]."""
    pres = np.asarray(pres, dtype=float)
    idx = np.ceil(pres / 10.0) - 1
    ok = np.isfinite(pres) & (pres > 0) & (pres <= 200)
    return np.where(ok, idx, -1).astype(int)


def argo_bin_profiles(df: pd.DataFrame, qc_good=QC_GOOD) -> tuple[pd.DataFrame, dict]:
    """Points (one row per level) -> one row per ascending profile with
    CHLA_A_B means and CHLA_A_B_N counts. Returns (profiles, stats)."""
    df = df.copy()
    df["DIRECTION"] = df["DIRECTION"].astype(str).str.strip()
    df = df.dropna(subset=ARGO_KEY)
    df[ARGO_KEY] = df[ARGO_KEY].astype("int64")
    stats = {
        "points": int(len(df)),
        "profiles_all": int(df.groupby(ARGO_KEY + ["DIRECTION"]).ngroups),
        "profiles_descending": int(df[df.DIRECTION == "D"].groupby(ARGO_KEY).ngroups),
    }

    asc = df[df.DIRECTION == "A"]
    stats["profiles_ascending"] = int(asc.groupby(ARGO_KEY).ngroups)

    # Time and position are constant within an ascending profile (checked on
    # 2025-07: 0 of 3,183 profiles vary), so "first" is exact, not a choice.
    meta = (asc.dropna(subset=["TIME", "LATITUDE", "LONGITUDE"])
               .groupby(ARGO_KEY, as_index=False)
               .agg(TIME=("TIME", "first"), LATITUDE=("LATITUDE", "first"),
                    LONGITUDE=("LONGITUDE", "first")))

    good = asc[asc["CHLA_ADJUSTED_QC"].isin(qc_good)
               & np.isfinite(asc["CHLA_ADJUSTED"]) & np.isfinite(asc["PRES"])]
    stats["levels_ascending"] = int(len(asc))
    stats["levels_good_qc"] = int(len(good))
    b = bin_index_argo(good["PRES"].to_numpy())
    good = good.assign(bin=b)[b >= 0]

    g = good.groupby(ARGO_KEY + ["bin"])["CHLA_ADJUSTED"].agg(["mean", "count"])
    mean = g["mean"].unstack("bin").reindex(columns=range(N_BINS))
    cnt = g["count"].unstack("bin").reindex(columns=range(N_BINS)).fillna(0).astype("int64")

    wide = pd.DataFrame(index=mean.index)
    for i, name in enumerate(BIN_NAMES):
        wide[f"CHLA_{name}"] = mean[i].astype(float)
        wide[f"CHLA_{name}_N"] = cnt[i]
    wide = wide.reset_index()

    prof = meta.merge(wide, on=ARGO_KEY, how="inner")
    prof["TIME"] = pd.to_datetime(prof["TIME"], utc=True)
    prof.insert(0, "profile_id",
                prof["PLATFORM_NUMBER"].astype(str).str.zfill(7) + "_"
                + prof["CYCLE_NUMBER"].astype(str).str.zfill(4))
    stats["profiles_with_good_chla"] = int(len(prof))
    return prof, stats


def argo_month_unit(month: str, force: bool = False) -> str:
    """One month: monthly .nc -> profile parquet + stats json."""
    import xarray as xr
    out = ARGO_PROFILES_MONTHLY / f"argo_chla_profiles_{month}.parquet"
    if out.exists() and not force:
        return f"skip {month}"
    nc = ARGO_MONTHLY_NC / f"argo_bgc_global_{month}.nc"
    with xr.open_dataset(nc) as ds:
        df = ds[ARGO_VARS].to_dataframe().reset_index(drop=True)
    prof, stats = argo_bin_profiles(df)
    # A profile's TIME should fall in its file's month; record any that don't.
    in_month = prof["TIME"].dt.strftime("%Y-%m") == month
    stats["month"] = month
    stats["profiles_outside_month"] = int((~in_month).sum())
    write_marker(out.with_suffix(".stats.json"), stats)
    atomic_write_parquet(prof, out)
    return f"done {month}: {stats['profiles_with_good_chla']} profiles"


def argo_combine() -> pd.DataFrame:
    files = [ARGO_PROFILES_MONTHLY / f"argo_chla_profiles_{m}.parquet" for m in MONTHS]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Argo months not built: {missing}")
    prof = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # The ERDDAP query is by month on TIME, so a profile should appear once;
    # keep the check anyway because the monthly files are the only guarantee.
    dup = prof["profile_id"].duplicated(keep=False)
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicated Argo profile_id rows across months")
    t = prof["TIME"]
    prof = prof[(t >= pd.Timestamp(START, tz="UTC")) & (t < pd.Timestamp(END_EXCLUSIVE, tz="UTC"))]
    return prof.sort_values("TIME").reset_index(drop=True)


# --------------------------------------------------------------------------
# OOI
# --------------------------------------------------------------------------

OOI_ERDDAP = "https://erddap.dataexplorer.oceanobservatories.org/erddap"
CHL_MOORING = "mass_concentration_of_chlorophyll_a_in_sea_water"
CHL_PROFILER = "mass_concentration_of_chlorophyll_a_in_sea_water_profiler_depth_enabled"

# Some datasets publish an incorrect z (e.g. 0.0) even though the sensor sits
# at a known fixed depth (commit 9025711). Positive meters down.
DEPTH_OVERRIDE_M = {
    "ooi-gi01sumo-rid16-02-flortd000": 12.0,
    "ooi-gi03flmb-ris01-05-flortd000": 25.0,
    "ooi-gi03flma-ris01-05-flortd000": 26.0,
    "ooi-gi01sumo-rii11-02-flordg031": 40.0,
    "ooi-gi01sumo-rii11-02-flordg032": 80.0,
    "ooi-gi01sumo-rii11-02-flordg033": 130.0,
    "ooi-gp03flma-ris01-05-flortd000": 28.0,
    "ooi-gp03flmb-ris01-05-flortd000": 30.0,
    "ooi-rs03axps-pc03a-4c-flordd303": 187.0,
}

# Excluded for reasons that do not depend on the date range. The other
# exclusions in the 2025 notebook were "no data in time range" for a range
# that ended 2025-12; with the range extended, those are fetched and simply
# come back empty if there is still no data.
OOI_EXCLUDE = {
    "ooi-ce02shsp-sp002-07-flortj000": "no qc_agg variable",
    "ooi-rs01sbps-pc01a-4c-flordd103": "no qc_agg variable",
    "ooi-gp02hypm-wfp03-01-flordl000": "2000 m and deeper",
}


def ooi_datasets() -> dict:
    from chla_z_ooi_datasets import OOI_DATASETS
    return {k: v for k, v in OOI_DATASETS.items() if k not in OOI_EXCLUDE}


def ooi_chl_var(dataset_id: str) -> str:
    from chla_z_ooi_datasets import MOORING_CHL
    return CHL_MOORING if dataset_id in MOORING_CHL else CHL_PROFILER


def _month_bounds(month: str) -> tuple[str, str]:
    p = pd.Period(month, freq="M")
    return p.start_time.strftime("%Y-%m-%dT00:00:00Z"), (p + 1).start_time.strftime("%Y-%m-%dT00:00:00Z")


def ooi_url(dataset_id: str, month: str) -> str:
    chl = ooi_chl_var(dataset_id)
    t0, t1 = _month_bounds(month)
    return (f"{OOI_ERDDAP}/tabledap/{dataset_id}.csv?"
            f"time%2Clatitude%2Clongitude%2Cz%2C{chl}%2C{chl}_qc_agg"
            f"&time%3E={t0}&time%3C{t1}&z%3E=-200")


def _http_get(url: str, attempts: int = 6, timeout=(30, 900)):
    """GET with retry. Returns (status, bytes). 404 is returned, not retried:
    ERDDAP uses it for 'no matching results'. 4xx other than 404/408/429 are
    raised at once (a bad query will not fix itself)."""
    import requests
    last = None
    for a in range(attempts):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 404:
                return 404, r.content
            if r.status_code in (408, 429) or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}: {r.text[:300]}")
            if r.status_code >= 400:
                raise ValueError(f"HTTP {r.status_code} for {url}: {r.text[:500]}")
            return r.status_code, r.content
        except ValueError:
            raise
        except Exception as e:  # connection reset, timeout, 5xx
            last = e
            log(f"  retry {a + 1}/{attempts} after {type(e).__name__}: {str(e)[:200]}")
            if a < attempts - 1:
                backoff_sleep(a)
    raise RuntimeError(f"giving up on {url}: {last}")


def ooi_coverage(dataset_id: str) -> dict:
    """Dataset time coverage from ERDDAP info, cached, so months outside the
    deployment record are not requested at all."""
    path = OOI_RAW / dataset_id / "_info.json"
    if path.exists():
        return json.loads(path.read_text())
    status, body = _http_get(f"{OOI_ERDDAP}/info/{dataset_id}/index.json")
    if status == 404:
        info = {"dataset_id": dataset_id, "exists": False}
    else:
        rows = json.loads(body)["table"]["rows"]
        attrs = {r[2]: r[4] for r in rows if r[0] == "attribute" and r[1] == "NC_GLOBAL"}
        info = {"dataset_id": dataset_id, "exists": True,
                "time_coverage_start": attrs.get("time_coverage_start"),
                "time_coverage_end": attrs.get("time_coverage_end")}
    write_marker(path, info)
    return info


def ooi_month_unit(dataset_id: str, month: str) -> str:
    """Fetch one dataset-month as raw CSV (gzip). Empty months get a marker."""
    import gzip
    d = OOI_RAW / dataset_id
    out, empty = d / f"{month}.csv.gz", d / f"{month}.empty"
    if out.exists() or empty.exists():
        return f"skip {dataset_id} {month}"
    url = ooi_url(dataset_id, month)
    status, body = _http_get(url)
    if status == 404:
        text = body.decode(errors="replace")
        if "no matching results" not in text.lower():
            raise RuntimeError(f"404 that is not 'no matching results' for {url}: {text[:300]}")
        write_marker(empty, {"url": url, "reason": "no matching results",
                             "fetched": pd.Timestamp.now("UTC")})
        return f"empty {dataset_id} {month}"
    # Check it parses before committing it: a truncated body raises here and
    # the unit is retried on the next run instead of being kept.
    import io
    df = pd.read_csv(io.BytesIO(body), skiprows=[1])
    need = {"time", "latitude", "longitude", "z"}
    if not need <= set(df.columns):
        raise RuntimeError(f"unexpected columns {list(df.columns)} for {url}")
    d.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with gzip.open(tmp, "wb") as fh:
        fh.write(body)
    os.replace(tmp, out)
    return f"done {dataset_id} {month}: {len(df)} rows"


def ooi_months_for(dataset_id: str) -> list[str]:
    info = ooi_coverage(dataset_id)
    if not info.get("exists"):
        return []
    t0 = pd.to_datetime(info.get("time_coverage_start") or START, utc=True)
    t1 = pd.to_datetime(info.get("time_coverage_end") or END_EXCLUSIVE, utc=True)
    keep = []
    for m in MONTHS:
        p = pd.Period(m, freq="M")
        ms, me = p.start_time.tz_localize("UTC"), (p + 1).start_time.tz_localize("UTC")
        if ms <= t1 and me > t0:
            keep.append(m)
    return keep


def ooi_daily_profiles(df: pd.DataFrame, dataset_id: str, chl_var: str,
                       description: str | None) -> pd.DataFrame:
    """Raw ERDDAP rows -> one row per day with CHLA_A_B daily means.

    Same rules as `ooi_mooring_save_file()` in the 2025 notebook plus the
    hackweek fixes: QC qc_agg == 1, depth = -z (or DEPTH_OVERRIDE_M),
    bins [A, B) via pd.cut(right=False), time = mean observation time of the
    good samples that day (commit 839c602). One change: lat/lon are taken
    per day, not per (day, bin), so a day can never split into two rows.
    """
    qc_col = f"{chl_var}_qc_agg"
    df = df.rename(columns={chl_var: "CHLA"})
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["date"] = df["time"].dt.floor("D")
    df["depth"] = -df["z"]
    if dataset_id in DEPTH_OVERRIDE_M:
        df["depth"] = DEPTH_OVERRIDE_M[dataset_id]
    good = df[(df["depth"] >= 0) & (df["depth"] < 200) & (df[qc_col] == 1.0)
              & np.isfinite(df["CHLA"])].copy()
    if good.empty:
        return pd.DataFrame()
    edges = np.arange(0, 210, 10)
    good["bin"] = pd.cut(good["depth"], bins=edges, labels=BIN_NAMES, right=False)
    wide = (good.groupby(["date", "bin"], observed=True)["CHLA"].mean()
                .unstack("bin").reindex(columns=BIN_NAMES))
    wide.columns = CHLA_COLS
    day = good.groupby("date").agg(
        lat=("latitude", "first"), lon=("longitude", "first"),
        # Series.mean() on datetimes, not int64 casting: pandas 3 stores these
        # in microseconds, so an int64 mean re-read as ns lands in 1970.
        time=("time", "mean"))
    out = day.join(wide).reset_index()
    out.insert(0, "profile_id", dataset_id + "_" + out["date"].dt.strftime("%Y%m%d"))
    out = out.drop(columns=["date"])
    out["description"] = description
    return out[["profile_id", "time", "lat", "lon"] + CHLA_COLS + ["description"]]


def ooi_dataset_unit(dataset_id: str, force: bool = False) -> str:
    """All fetched months of one dataset -> daily profiles parquet."""
    out = OOI_DAILY / f"{dataset_id}.parquet"
    empty = OOI_DAILY / f"{dataset_id}.empty"
    if (out.exists() or empty.exists()) and not force:
        return f"skip {dataset_id}"
    for stale in (out, empty):  # with force, never keep a result from older code
        stale.unlink(missing_ok=True)
    months = ooi_months_for(dataset_id)
    d = OOI_RAW / dataset_id
    missing = [m for m in months
               if not (d / f"{m}.csv.gz").exists() and not (d / f"{m}.empty").exists()]
    if missing:
        raise RuntimeError(f"{dataset_id}: months not fetched yet {missing}")
    parts = []
    for m in months:
        f = d / f"{m}.csv.gz"
        if f.exists():
            # Daily bins never cross a month boundary (UTC days, UTC months),
            # so each month is binned on its own and memory stays small.
            raw = pd.read_csv(f, skiprows=[1])
            prof = ooi_daily_profiles(raw, dataset_id, ooi_chl_var(dataset_id),
                                      ooi_datasets()[dataset_id])
            if len(prof):
                parts.append(prof)
    if not parts:
        write_marker(empty, {"dataset_id": dataset_id, "months": months,
                             "reason": "no QC-passing CHLA in 0-200 m"})
        return f"empty {dataset_id}"
    prof = pd.concat(parts, ignore_index=True)
    atomic_write_parquet(prof, out)
    return f"done {dataset_id}: {len(prof)} days"


def ooi_combine() -> tuple[pd.DataFrame, dict]:
    """Per-dataset daily files -> one table with the Sauzède correction."""
    sys.path.insert(0, str(REPO / "src"))
    from sauzede_lut import load_lut, chla_scaling_factor
    files = sorted(OOI_DAILY.glob("*.parquet"))
    combined = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)
    dup = combined["profile_id"].duplicated(keep=False)
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicated OOI profile_id rows")

    # OOI CHLA is factory-calibrated fluorescence; Argo CHLA_ADJUSTED is
    # Sauzède-corrected. chla_corrected = chla_fluor / ratio, and keep
    # chla_ratio so raw = corrected * ratio. No LUT value -> ratio 1 (commit d17c522).
    lut = load_lut()  # this repo's copy, datasets/chla_z/data/fluor_to_chla_lut_127630.nc
    ratio = chla_scaling_factor(combined["lat"].values, combined["lon"].values, lut=lut)
    combined["chla_ratio"] = ratio
    apply = np.where(np.isnan(ratio), 1.0, ratio)
    for c in CHLA_COLS:
        combined[c] = combined[c] / apply
    site = combined["profile_id"].str.rsplit("_", n=1).str[0]
    no_lut = sorted(site[np.isnan(ratio)].unique().tolist())
    per_site = (combined.assign(site=site).groupby("site")["chla_ratio"]
                .agg(["min", "max"]).round(3).to_dict(orient="index"))
    return combined, {"sites_without_lut": no_lut, "ratio_by_site": per_site}


# --------------------------------------------------------------------------
# PACE matchups
# --------------------------------------------------------------------------

PACE_PRODUCTS = {
    "Rrs": {"short_names": ["PACE_OCI_L3M_AOP", "PACE_OCI_L3M_AOP_NRT"], "var": "Rrs"},
    "chlor_a": {"short_names": ["PACE_OCI_L3M_BGC", "PACE_OCI_L3M_BGC_NRT"], "var": "chlor_a"},
    "Kd": {"short_names": ["PACE_OCI_L3M_KD", "PACE_OCI_L3M_KD_NRT"], "var": "Kd", "wavelength": 490.0},
}
PACE_GRANULE_PATTERN = "*.DAY.*.4km.*"


def _earthaccess_login():
    import earthaccess
    auth = earthaccess.login(strategy="netrc")
    if not auth.authenticated:
        raise RuntimeError("Earthdata login failed (netrc)")
    return earthaccess


def pace_granule_list(product: str, refresh: bool = False) -> list[dict]:
    """CMR search once per product, cached, so every worker and every rerun
    uses the same granule list."""
    path = MATCH_DIR / product / "_granules.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    ea = _earthaccess_login()
    out = []
    for sn in PACE_PRODUCTS[product]["short_names"]:
        res = ea.search_data(short_name=sn, temporal=(START, END_EXCLUSIVE),
                             granule_name=PACE_GRANULE_PATTERN)
        for g in res:
            out.append({"short_name": sn, "granule": g["umm"]["GranuleUR"]})
    write_marker(path, out)
    return out


def pace_points(refresh: bool = False) -> pd.DataFrame:
    """Profile time/position for both platforms, from the rebuilt tables."""
    path = MATCH_DIR / "_points.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    a = pd.read_parquet(DATA_DIR / "CHLA_argo_profiles.parquet",
                        columns=["profile_id", "TIME", "LATITUDE", "LONGITUDE"])
    a.columns = ["profile_id", "time", "lat", "lon"]
    a["platform"] = "argo"
    o = pd.read_parquet(DATA_DIR / "CHLA_ooi_profiles.parquet",
                        columns=["profile_id", "time", "lat", "lon"])
    o["platform"] = "ooi"
    pts = pd.concat([a, o], ignore_index=True)
    pts["time"] = pd.to_datetime(pts["time"], utc=True)
    atomic_write_parquet(pts, path)
    return pts


def _granule_out(product: str, granule: str) -> Path:
    return MATCH_DIR / product / granule.replace(".nc", ".parquet")


def _rrs_labels(wl: np.ndarray) -> list[int]:
    lab = np.round(np.asarray(wl, dtype=float)).astype(int).tolist()
    if lab != RRS_LABELS:
        raise ValueError(f"PACE Rrs bands changed: {len(lab)} bands, first diffs "
                         f"{[(a, b) for a, b in zip(lab, RRS_LABELS) if a != b][:5]}")
    return lab


def _nearest_index(coord: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Nearest grid index for ascending or descending 1-D coords
    (same result as ds.sel(..., method='nearest') used in 2025)."""
    asc = coord[0] <= coord[-1]
    base, tgt = (coord, q) if asc else (-coord, -q)
    i = np.searchsorted(base, tgt)
    i0 = np.clip(i - 1, 0, len(coord) - 1)
    i1 = np.clip(i, 0, len(coord) - 1)
    return np.where(np.abs(coord[i1] - q) < np.abs(coord[i0] - q), i1, i0)


def pace_granule_unit(product: str, short_name: str, granule: str, attempts: int = 5) -> str:
    out = _granule_out(product, granule)
    empty = out.with_suffix(".empty")
    if out.exists() or empty.exists():
        return f"skip {granule}"
    pts = pace_points()
    cfg = PACE_PRODUCTS[product]
    var = cfg["var"]
    last = None
    for a in range(attempts):
        try:
            res = _match_one_granule(product, short_name, granule, pts, cfg, var)
            break
        except ValueError:
            raise  # band set changed etc.: not transient
        except Exception as e:
            last = e
            log(f"  {granule} retry {a + 1}/{attempts} after {type(e).__name__}: {str(e)[:200]}")
            if a < attempts - 1:
                backoff_sleep(a, base=20)
    else:
        raise RuntimeError(f"giving up on {granule}: {last}")
    if res is None:
        write_marker(empty, {"granule": granule, "reason": "no profiles in time window"})
        return f"empty {granule}"
    atomic_write_parquet(res, out)
    return f"done {granule}: {len(res)} points"


def _match_one_granule(product, short_name, granule, pts, cfg, var):
    import dask
    import xarray as xr
    ea = _earthaccess_login()
    found = ea.search_data(short_name=short_name, granule_name=granule)
    if len(found) != 1:
        raise RuntimeError(f"CMR returned {len(found)} results for {granule}")
    f = ea.open(found, pqdm_kwargs={"disable": True})[0]
    with xr.open_dataset(f, chunks={}, cache=False) as ds:
        t_start = pd.to_datetime(ds.attrs["time_coverage_start"], utc=True)
        t_end = pd.to_datetime(ds.attrs["time_coverage_end"], utc=True)
        # 2025 rule, unchanged: profiles within 24 h after the granule's
        # t_start attribute (not the date in the file name).
        sel = (pts["time"] >= t_start) & (pts["time"] < t_start + pd.Timedelta(hours=24))
        rec = pts[sel].reset_index(drop=True)
        if rec.empty:
            return None
        lat = ds["lat"].values
        lon = ds["lon"].values
        li = _nearest_index(lat, rec["lat"].to_numpy(float))
        lo = _nearest_index(lon, rec["lon"].to_numpy(float))
        da = ds[var]
        extra = {}
        if product == "Kd":
            wl = ds["wavelength"].values
            k = int(np.argmin(np.abs(wl - cfg["wavelength"])))
            if abs(wl[k] - cfg["wavelength"]) > 3:
                raise ValueError(f"no Kd band near 490 nm in {granule}: {wl}")
            da = da.isel(wavelength=k)
            extra["pace_Kd_wavelength"] = float(wl[k])
        with dask.config.set(scheduler="threads", num_workers=4):
            vals = np.asarray(da.data.vindex[li, lo].compute(), dtype=float)
        cols = {
            "platform": rec["platform"].to_numpy(),
            "profile_id": rec["profile_id"].to_numpy(),
            f"pace_{product}_file": granule,
            f"pace_{product}_t_start": t_start,
            f"pace_{product}_t_end": t_end,
            f"pace_{product}_lat": lat[li].astype(float),
            f"pace_{product}_lon": lon[lo].astype(float),
        }
        if product == "Rrs":
            for j, lab in enumerate(_rrs_labels(ds["wavelength"].values)):
                cols[f"pace_Rrs_{lab}"] = vals[:, j]
        elif product == "Kd":
            cols["pace_Kd_490"] = vals
        else:
            cols["pace_chlor_a"] = vals
        df = pd.DataFrame(cols)
        for k_, v_ in extra.items():
            df[k_] = v_
        return df


def _granule_day(name: pd.Series) -> pd.Series:
    return name.str.extract(r"PACE_OCI\.(\d{8})\.")[0]


def pace_select(product: str, pts: pd.DataFrame) -> pd.DataFrame:
    """All granule matches for a product -> one row per profile.
    Rule from 2025, applied the same way to every product: standard before NRT,
    then the granule whose time window centre is closest to the profile time,
    then file name (only to make ties deterministic, e.g. duplicate KD_NRT)."""
    files = sorted((MATCH_DIR / product).glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    fcol = f"pace_{product}_file"
    df["_nrt"] = df[fcol].str.contains("NRT")
    df = df.merge(pts[["profile_id", "time"]], on="profile_id", how="left")
    centre = df[f"pace_{product}_t_start"] + (df[f"pace_{product}_t_end"] - df[f"pace_{product}_t_start"]) / 2
    df["_dt"] = (df["time"] - centre).abs()
    df = (df.sort_values(["profile_id", "_nrt", "_dt", fcol])
            .drop_duplicates("profile_id", keep="first"))
    df["_day"] = _granule_day(df[fcol])
    return df.drop(columns=["time", "_dt"]).reset_index(drop=True)


def pace_merge(profiles: pd.DataFrame, platform: str) -> tuple[pd.DataFrame, dict]:
    """Rrs is the base; chlor_a and Kd_490 are attached only when they come
    from the same day and the same stream (standard/NRT) as the Rrs granule."""
    pts = pace_points()
    pts = pts[pts["platform"] == platform]
    rrs = pace_select("Rrs", pts)
    rrs = rrs[rrs["platform"] == platform]
    chl = pace_select("chlor_a", pts)
    kd = pace_select("Kd", pts)
    rrs_cols = [f"pace_Rrs_{x}" for x in RRS_LABELS]
    base = rrs[["profile_id", "_day", "_nrt", "pace_Rrs_file", "pace_Rrs_t_start",
                "pace_Rrs_t_end", "pace_Rrs_lat", "pace_Rrs_lon"] + rrs_cols]
    base = base.merge(chl[["profile_id", "_day", "_nrt", "pace_chlor_a"]],
                      on=["profile_id", "_day", "_nrt"], how="left")
    base = base.merge(kd[["profile_id", "_day", "_nrt", "pace_Kd_490"]],
                      on=["profile_id", "_day", "_nrt"], how="left")
    stats = {
        "profiles": int(len(profiles)),
        "matched_rrs_granule": int(len(base)),
        "nrt_rows": int(base["_nrt"].sum()),
        "rrs_all_nan": int(base[rrs_cols].isna().all(axis=1).sum()),
        "chlor_a_from_other_granule_or_missing": int(
            (~base["profile_id"].isin(chl.merge(rrs[["profile_id", "_day", "_nrt"]],
                                                on=["profile_id", "_day", "_nrt"])["profile_id"])).sum()),
        "kd_from_other_granule_or_missing": int(
            (~base["profile_id"].isin(kd.merge(rrs[["profile_id", "_day", "_nrt"]],
                                               on=["profile_id", "_day", "_nrt"])["profile_id"])).sum()),
    }
    base = base.drop(columns=["_day", "_nrt"])
    tcol = "TIME" if "TIME" in profiles.columns else "time"
    out = profiles.merge(base, on="profile_id", how="inner").sort_values(tcol).reset_index(drop=True)
    if out["profile_id"].duplicated().any():
        raise ValueError("duplicate profile_id after PACE merge")
    stats["rows_out"] = int(len(out))
    return out, stats


# ---- OLCI ----------------------------------------------------------------------
# Sentinel-3 OLCI L3 mapped daily 4 km Rrs (NASA OB.DAAC), matched for *every*
# in situ profile, with or without a PACE match, so that "OLCI has more data"
# and "OLCI vs PACE on the same profiles" can both be asked of one table.
#
# Matching rules are Nabir Mamnun's (2026-go-bgc-hackweek
# contributor_folders/nama/olci_l3_matchups.ipynb), kept so his rows reproduce:
# - the granule is the one named for the profile's UTC calendar day;
# - nearest 4 km cell (pandas get_indexer "nearest");
# - S3A is read first; S3B is read only for profiles whose S3A Rrs_443 is NaN;
#   each band takes S3A where finite, else S3B, and olci_sat says which sensor
#   gave Rrs_443 (so the 11 bands of one row can come from both sensors).
# One deliberate difference: a granule that fails to open raises (the day is
# retried) instead of being recorded as NaN, so a network drop cannot pass for
# a masked pixel.

OLCI_DIR = TEMP_DIR / "matchups_olci"
OLCI_BANDS = [400, 412, 443, 490, 510, 560, 620, 665, 674, 681, 709]
OLCI_SHORT = {"S3A": "OLCIS3A_L3m_ERR_RRS", "S3B": "OLCIS3B_L3m_ERR_RRS"}
_YMD_RE = re.compile(r"\.(\d{8})\.L3m\.DAY\.")


def olci_granule_index(refresh: bool = False) -> dict:
    """{sat: {YYYYMMDD: {band: https url}}}, from one CMR search per sat x band,
    cached so every worker and rerun uses the same list."""
    path = OLCI_DIR / "_granules.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    ea = _earthaccess_login()
    index = {sat: {} for sat in OLCI_SHORT}
    dups = []
    for sat, sn in OLCI_SHORT.items():
        for nm in OLCI_BANDS:
            res = ea.search_data(short_name=sn, temporal=(START, END_EXCLUSIVE),
                                 granule_name=f"*.DAY.*.Rrs_{nm}.4km.nc")
            for g in res:
                url = g.data_links(access="external")[0]
                m = _YMD_RE.search(url)
                if not m:
                    continue
                day = index[sat].setdefault(m.group(1), {})
                if str(nm) in day and day[str(nm)] != url:
                    dups.append([sat, m.group(1), nm, day[str(nm)], url])
                    url = max(day[str(nm)], url)  # deterministic; none seen so far
                day[str(nm)] = url
    if dups:
        log(f"OLCI: {len(dups)} duplicate day/band granules; kept the lexically last URL")
    write_marker(path, index)
    return index


def olci_points() -> pd.DataFrame:
    pts = pace_points().copy()
    pts["ymd"] = pts["time"].dt.strftime("%Y%m%d")
    return pts


def _olci_sample(urls: dict, lats, lons) -> dict:
    import xarray as xr
    ea = _earthaccess_login()
    out = {}
    for nm in OLCI_BANDS:
        url = urls.get(str(nm))
        if url is None:  # no granule for this band and day: genuinely missing
            out[nm] = np.full(len(lats), np.nan)
            continue
        f = ea.open([url], pqdm_kwargs={"disable": True})[0]
        with xr.open_dataset(f, engine="h5netcdf", chunks={}, cache=False) as ds:
            var = f"Rrs_{nm}" if f"Rrs_{nm}" in ds.data_vars else "Rrs"
            i = ds.get_index("lat").get_indexer(np.asarray(lats, float), method="nearest")
            j = ds.get_index("lon").get_indexer(np.asarray(lons, float), method="nearest")
            out[nm] = np.asarray(ds[var].isel(lat=xr.DataArray(i), lon=xr.DataArray(j)).values, float)
    return out


def olci_day_unit(ymd: str, attempts: int = 5) -> str:
    out = OLCI_DIR / f"{ymd}.parquet"
    empty = out.with_suffix(".empty")
    if out.exists() or empty.exists():
        return f"skip {ymd}"
    index = olci_granule_index()
    a_urls = index["S3A"].get(ymd, {})
    b_urls = index["S3B"].get(ymd, {})
    pts = olci_points()
    sub = pts[pts["ymd"] == ymd].reset_index(drop=True)
    if sub.empty or not (a_urls or b_urls):
        write_marker(empty, {"day": ymd, "profiles": int(len(sub)),
                             "reason": "no profiles" if sub.empty else "no OLCI granule"})
        return f"empty {ymd}"
    lats, lons = sub["lat"].to_numpy(), sub["lon"].to_numpy()
    last = None
    for a in range(attempts):
        try:
            s3a = _olci_sample(a_urls, lats, lons)
            need_b = ~np.isfinite(s3a[443])
            s3b = {nm: np.full(len(sub), np.nan) for nm in OLCI_BANDS}
            if need_b.any() and b_urls:
                # As in the notebook: S3B is sampled at *all* of the day's
                # points once any needs it, so it can also fill another band
                # where S3A has Rrs_443 but not that band.
                s3b = _olci_sample(b_urls, lats, lons)
            break
        except Exception as e:
            last = e
            log(f"  OLCI {ymd} retry {a + 1}/{attempts} after {type(e).__name__}: {str(e)[:200]}")
            if a < attempts - 1:
                backoff_sleep(a, base=20)
    else:
        raise RuntimeError(f"giving up on OLCI {ymd}: {last}")
    res = sub[["profile_id"]].copy()
    sat = np.full(len(sub), None, dtype=object)
    sat[np.isfinite(s3a[443])] = "S3A"
    sat[need_b & np.isfinite(s3b[443])] = "S3B"
    res["olci_sat"] = sat
    res["olci_day"] = ymd
    res["olci_S3A_available"] = bool(a_urls)
    res["olci_S3B_available"] = bool(b_urls)
    for nm in OLCI_BANDS:
        res[f"olci_Rrs_{nm}"] = np.where(np.isfinite(s3a[nm]), s3a[nm], s3b[nm])
    atomic_write_parquet(res, out)
    return f"done {ymd}: {len(res)} points, {int(np.isfinite(res['olci_Rrs_443']).sum())} with Rrs_443"


def olci_days() -> list[str]:
    return sorted(olci_points()["ymd"].unique())


def olci_merge(profiles: pd.DataFrame, platform: str) -> tuple[pd.DataFrame, dict]:
    """Every profile on a day with an OLCI granule, plus PACE flags."""
    files = sorted(OLCI_DIR.glob("*.parquet"))
    olci = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    plus = pd.read_parquet(DATA_DIR / f"CHLA_{platform}_profiles_plus_PACE.parquet")
    rrs_cols = [f"pace_Rrs_{x}" for x in RRS_LABELS]
    has_granule = set(plus["profile_id"])
    has_rrs = set(plus.loc[plus[rrs_cols].notna().any(axis=1), "profile_id"])
    tcol = "TIME" if "TIME" in profiles.columns else "time"
    out = profiles.merge(olci, on="profile_id", how="inner").sort_values(tcol).reset_index(drop=True)
    if out["profile_id"].duplicated().any():
        raise ValueError("duplicate profile_id after OLCI merge")
    out["pace_granule"] = out["profile_id"].isin(has_granule)
    out["pace_Rrs_valid"] = out["profile_id"].isin(has_rrs)
    ok = out[[f"olci_Rrs_{nm}" for nm in OLCI_BANDS]].notna().all(axis=1)
    stats = {
        "profiles": int(len(profiles)),
        "rows_out (day has an OLCI granule)": int(len(out)),
        "olci_all_11_bands": int(ok.sum()),
        "olci_all_11_and_pace_Rrs_valid": int((ok & out["pace_Rrs_valid"]).sum()),
        "olci_all_11_no_pace_Rrs": int((ok & ~out["pace_Rrs_valid"]).sum()),
        "pace_Rrs_valid_no_olci": int((~ok & out["pace_Rrs_valid"]).sum()),
    }
    return out, stats

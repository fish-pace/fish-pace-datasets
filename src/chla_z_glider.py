"""CHLA-Z glider table: IOOS Glider DAC chlorophyll profiles.

Coastal CHLA-Z (Eli Holmes, 2026-11-23/24): 20 glider deployments spread over
six regions (not every glider deployment; the list can grow), processed like
OOI (3 h windows of local mean solar time, 10 m bins (A, B] in 0-200 m,
Sauzède correction). Released like Argo and OOI: `datasets/chla_z/build.py`
stages `glider-select`, `glider-fetch`, `glider`, then the PACE and OLCI
stages with `--platform glider`. Design note: `claude/notes/chla-z-glider.md`.

Units are resumable like `chla_z_build` (one file per unit, atomic write,
skipped if present); intermediates go to `_temp_data/glider/`:

    candidates()               ERDDAP search + per-dataset metadata -> candidates.parquet
    screen_unit(id)            one mid-deployment day, sensor checks -> screen/<id>.json
    fetch_unit(id, month)      CHLA samples 0-200 m                  -> raw/<id>/<YYYY-MM>.csv.gz
    fetch_deep_unit(id, month) CHLA samples below 300 m (dark value) -> raw_deep/<id>/<YYYY-MM>.csv.gz
    window_unit(id)            one profile per 3 h solar window      -> windows/<id>.parquet
    combine()                  + Sauzède, dark_offset, region, ...   -> the table (chla_z_tables writes it)
"""

from __future__ import annotations

import gzip
import io
import json
import os
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

import chla_z_build as b

GLIDER_ERDDAP = "https://gliders.ioos.us/erddap"
GLIDER_DIR = b.TEMP_DIR / "glider"
CAND_PATH = GLIDER_DIR / "candidates.parquet"
SCREEN_DIR = GLIDER_DIR / "screen"
RAW_DIR = GLIDER_DIR / "raw"
WIN_DIR = GLIDER_DIR / "windows"
SELECTION_PATH = Path(__file__).resolve().parent / "chla_z_glider_selection.json"

WINDOW_H = b.OOI_WINDOW_H  # 3 h, the same as OOI
QC_KEEP = (1, 2)  # both flag schemes: 1 good, 2 not evaluated / probably good
QC_VARS = ("chlorophyll_a_qc", "chlorophyll_a_qartod_summary_flag")


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------

def region_of(lat: float, lon: float) -> str:
    """Pilot regions, by deployment centre. Boundaries are coarse on purpose:
    they only spread the selection, nothing is computed per region. No
    deployment of >= 14 days since 2024-03 is in the Gulf of Mexico
    (checked 2026-11-23); the one low-latitude deployment is in the open
    tropical Atlantic east of the Caribbean."""
    if lat < -30:
        return "Antarctic Peninsula"
    if lon < -110:
        return "California Current"
    if lat < 25:
        return "Tropical Atlantic"
    if lon > -60:
        return "Open Atlantic"
    if lat < 35:
        return "Off Cape Hatteras / South Atlantic Bight"
    if lat > 40.6 and lon > -72:
        return "Southern New England shelf"
    return "Mid-Atlantic Bight and slope"


def _info(dataset_id: str) -> dict:
    status, body = b._http_get(f"{GLIDER_ERDDAP}/info/{dataset_id}/index.json")
    if status == 404:
        raise RuntimeError(f"no info for {dataset_id}")
    rows = json.loads(body)["table"]["rows"]
    var = {r[1] for r in rows if r[0] == "variable"}
    att = {(r[1], r[2]): r[4] for r in rows if r[0] == "attribute"}
    qc = next((q for q in QC_VARS if q in var), None)
    return {"datasetID": dataset_id,
            "platform_type": att.get(("NC_GLOBAL", "platform_type")),
            "institution": att.get(("NC_GLOBAL", "institution")),
            "chla_units": att.get(("chlorophyll_a", "units")),
            "qc_var": qc,
            "has_precise_lat": "precise_lat" in var,
            "has_depth": "depth" in var}


def candidates(refresh: bool = False) -> pd.DataFrame:
    """All tabledap datasets with chlorophyll_a and data since START, one row
    each, with bounds and the metadata needed to choose and fetch them."""
    if CAND_PATH.exists() and not refresh:
        return pd.read_parquet(CAND_PATH)
    q = (f"{GLIDER_ERDDAP}/search/advanced.csv?page=1&itemsPerPage=1000&protocol=tabledap"
         f"&variableName=chlorophyll_a&minTime={b.START}")
    _, body = b._http_get(q)
    ids = pd.read_csv(io.BytesIO(body))["Dataset ID"].tolist()
    _, body = b._http_get(
        f"{GLIDER_ERDDAP}/tabledap/allDatasets.csv?datasetID,minLongitude,maxLongitude,"
        "minLatitude,maxLatitude,minTime,maxTime")
    alld = pd.read_csv(io.BytesIO(body), skiprows=[1])
    c = alld[alld.datasetID.isin(ids)].copy()
    c = c.merge(pd.DataFrame([_info(d) for d in c.datasetID]), on="datasetID")
    for k in ("minTime", "maxTime"):
        c[k] = pd.to_datetime(c[k], utc=True)
    c["deployment"] = c.datasetID.str.removesuffix("-delayed")
    c["delayed"] = c.datasetID.str.endswith("-delayed")
    c["lat"] = (c.minLatitude + c.maxLatitude) / 2
    c["lon"] = (c.minLongitude + c.maxLongitude) / 2
    c["region"] = [region_of(a, o) for a, o in zip(c.lat, c.lon)]
    t0 = c.minTime.clip(lower=pd.Timestamp(b.START, tz="UTC"))
    t1 = c.maxTime.clip(upper=pd.Timestamp(b.END_EXCLUSIVE, tz="UTC"))
    c["days_in_period"] = (t1 - t0).dt.total_seconds() / 86400
    c["fetched"] = pd.Timestamp.now("UTC")
    b.atomic_write_parquet(c, CAND_PATH)
    return c


def shortlist(c: pd.DataFrame, min_days: float = 14) -> pd.DataFrame:
    """One dataset per deployment (delayed mode preferred: it is the
    reprocessed copy), with units, inside the PACE period, >= min_days."""
    c = c[c.chla_units.notna() & c.has_depth & (c.days_in_period >= min_days)]
    c = c.sort_values("delayed", ascending=False).drop_duplicates("deployment")
    return c.sort_values(["region", "minTime"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Screening: is the chlorophyll sensor plausible?
# --------------------------------------------------------------------------

def _tabledap_csv(dataset_id: str, cols: list[str], cons: list[str]) -> pd.DataFrame | None:
    """None for ERDDAP's 'no matching results' 404; raises on anything else."""
    q = ",".join(cols) + "".join("&" + urllib.parse.quote(x, safe="=!<>") for x in cons)
    url = f"{GLIDER_ERDDAP}/tabledap/{dataset_id}.csv?{q}"
    status, body = b._http_get(url)
    if status == 404:
        if "no matching results" not in body.decode(errors="replace").lower():
            raise RuntimeError(f"404 that is not 'no matching results' for {url}")
        return None
    return pd.read_csv(io.BytesIO(body), skiprows=[1])


def screen_stats(d: pd.DataFrame) -> dict:
    """Sensor checks on one day of samples (all depths, no QC applied).

    The two failures seen on 2026-11-23 were sp040 (flat ~0.07 at all
    depths) and sp034 (0.43 below 100 m: dark offset not removed)."""
    d = d[np.isfinite(d.chlorophyll_a) & np.isfinite(d.depth)]
    out = {"n": int(len(d)), "n_profiles": int(d.profile_id.nunique()),
           "max_depth": float(d.depth.max()) if len(d) else np.nan}
    if len(d) < 50:
        out["verdict"] = "too few samples"
        return out
    zmax = out["max_depth"]
    top = d[d.depth <= min(20, zmax / 3)].chlorophyll_a
    q = d.chlorophyll_a.quantile([0.1, 0.5, 0.9])
    out.update(surface_median=float(top.median()), p10=float(q[0.1]), p50=float(q[0.5]),
               p90=float(q[0.9]), neg_frac=float((d.chlorophyll_a < 0).mean()))
    # Vertical maximum: highest 5 m-bin median anywhere in the profile.
    zb = (d.depth // 5) * 5
    prof = d.groupby(zb).chlorophyll_a.median()
    out["max_bin_median"] = float(prof.max())
    deep = d[d.depth >= 150].chlorophyll_a
    out["deep_median"] = float(deep.median()) if len(deep) >= 20 else np.nan
    reasons = []
    if out["max_bin_median"] - out["p10"] < 0.05 and out["p90"] / max(out["p10"], 1e-3) < 1.5:
        reasons.append("flat profile")
    if np.isfinite(out["deep_median"]) and out["deep_median"] > 0.2:
        reasons.append("deep offset > 0.2")
    if out["neg_frac"] > 0.05:
        reasons.append("negatives > 5%")
    out["verdict"] = "; ".join(reasons) or "ok"
    return out


def screen_unit(dataset_id: str) -> str:
    out = SCREEN_DIR / f"{dataset_id}.json"
    if out.exists():
        return f"skip {dataset_id}"
    c = candidates().set_index("datasetID").loc[dataset_id]
    t0 = max(c.minTime, pd.Timestamp(b.START, tz="UTC"))
    t1 = min(c.maxTime, pd.Timestamp(b.END_EXCLUSIVE, tz="UTC"))
    mid = (t0 + (t1 - t0) / 2).floor("D")
    d = _tabledap_csv(dataset_id, ["profile_id", "time", "depth", "chlorophyll_a"],
                      [f"time>={mid:%Y-%m-%dT%H:%M:%SZ}",
                       f"time<{mid + pd.Timedelta(days=1):%Y-%m-%dT%H:%M:%SZ}",
                       "chlorophyll_a!=NaN"])
    s = {"datasetID": dataset_id, "day": f"{mid:%Y-%m-%d}"}
    s.update(screen_stats(d) if d is not None else {"verdict": "no data that day"})
    b.write_marker(out, s)
    return f"done {dataset_id}: {s['verdict']}"


def screen_table() -> pd.DataFrame:
    return pd.DataFrame([json.loads(p.read_text()) for p in sorted(SCREEN_DIR.glob("*.json"))])


def selection() -> list[str]:
    """The deployments in the table, chosen once and committed (see design note)."""
    return json.loads(SELECTION_PATH.read_text())["datasets"]


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def months_for(dataset_id: str) -> list[str]:
    c = candidates().set_index("datasetID").loc[dataset_id]
    keep = []
    for m in b.MONTHS:
        p = pd.Period(m, freq="M")
        ms, me = p.start_time.tz_localize("UTC"), (p + 1).start_time.tz_localize("UTC")
        if ms <= c.maxTime and me > c.minTime:
            keep.append(m)
    return keep


def fetch_unit(dataset_id: str, month: str) -> str:
    """One dataset-month of CHLA samples in 0-200 m, as gzipped ERDDAP CSV."""
    d = RAW_DIR / dataset_id
    out, empty = d / f"{month}.csv.gz", d / f"{month}.empty"
    if out.exists() or empty.exists():
        return f"skip {dataset_id} {month}"
    c = candidates().set_index("datasetID").loc[dataset_id]
    cols = ["profile_id", "time", "latitude", "longitude", "precise_time", "depth", "chlorophyll_a"]
    if isinstance(c.qc_var, str):  # NaN when the dataset has no CHLA flag
        cols.append(c.qc_var)
    if c.has_precise_lat:
        cols += ["precise_lat", "precise_lon"]
    t0, t1 = b._month_bounds(month)
    df = _tabledap_csv(dataset_id, cols, [f"time>={t0}", f"time<{t1}", "depth<=200",
                                          "chlorophyll_a!=NaN"])
    if df is None:
        b.write_marker(empty, {"reason": "no matching results", "fetched": pd.Timestamp.now("UTC")})
        return f"empty {dataset_id} {month}"
    d.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with gzip.open(tmp, "wb") as fh:
        df.to_csv(fh, index=False)
    os.replace(tmp, out)
    return f"done {dataset_id} {month}: {len(df)} rows"


# --------------------------------------------------------------------------
# Profiles: one per 3 h window of local solar time
# --------------------------------------------------------------------------

def good_samples(df: pd.DataFrame, qc_var: str | None) -> pd.DataFrame:
    """QC-passing samples in (0, 200] m. Sample time is precise_time where
    present: the profile `time` can be ~70 min off the CHLA samples (ru34)."""
    df = df.copy()
    df["profile_time"] = pd.to_datetime(df["time"], utc=True)
    pt = pd.to_datetime(df["precise_time"], utc=True) if "precise_time" in df else None
    df["t"] = df["profile_time"] if pt is None else pt.fillna(df["profile_time"])
    ok = (df.depth > 0) & (df.depth <= 200) & np.isfinite(df.chlorophyll_a)
    if qc_var:
        ok &= df[qc_var].isin(QC_KEEP)
    return df[ok]


def window_profiles(good: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Samples -> one row per 3 h window of local mean solar time.

    OOI averages every sample in a window because the platform is fixed. A
    glider moves 10-80 km a day, so averaging several profiles would blur
    positions; instead each window keeps ONE glider profile: the one with the
    most filled 10 m bins, ties to the profile closest to the window centre.
    Its time is the mean sample time and lat/lon the mean sample position
    (precise_lat/lon where the dataset has them, else the profile position).
    """
    if good.empty:
        return pd.DataFrame()
    g = good.copy()
    if "precise_lat" in g:
        g["plat"] = g.precise_lat.fillna(g.latitude)
        g["plon"] = g.precise_lon.fillna(g.longitude)
    else:
        g["plat"], g["plon"] = g.latitude, g.longitude
    g["bin"] = b.bin_index_argo(g.depth.to_numpy(float))
    p = g.groupby("profile_id").agg(time=("t", "mean"), lat=("plat", "mean"), lon=("plon", "mean"),
                                    n_samples=("chlorophyll_a", "size"),
                                    n_bins=("bin", "nunique"), max_depth=("depth", "max"))
    solar = p.time + pd.to_timedelta(p.lon / 15.0, unit="h")
    win = (solar.dt.hour // WINDOW_H) * WINDOW_H
    p["key"] = solar.dt.floor("D").dt.strftime("%Y%m%d") + "T" + win.map("{:02d}".format)
    centre = solar.dt.floor("D") + pd.to_timedelta(win + WINDOW_H / 2, unit="h")
    p["off_centre"] = (solar - centre).abs()
    p = p.sort_values(["key", "n_bins", "off_centre"], ascending=[True, False, True])
    pick = p[~p.key.duplicated()]
    sel = g[g.profile_id.isin(pick.index)]
    wide = (sel.groupby(["profile_id", "bin"]).chlorophyll_a.mean().unstack("bin")
               .reindex(columns=range(b.N_BINS)))
    wide.columns = b.CHLA_COLS
    out = pick.join(wide).reset_index().rename(columns={"profile_id": "glider_profile_id"})
    out["profiles_in_window"] = out.key.map(p.key.value_counts())
    out.insert(0, "profile_id", dataset_id + "_" + out["key"])
    out["dataset_id"] = dataset_id
    return out[["profile_id", "time", "lat", "lon", "n_samples", "max_depth"] + b.CHLA_COLS
               + ["dataset_id", "glider_profile_id", "profiles_in_window"]]


def window_unit(dataset_id: str, force: bool = False) -> str:
    out, empty = WIN_DIR / f"{dataset_id}.parquet", WIN_DIR / f"{dataset_id}.empty"
    if (out.exists() or empty.exists()) and not force:
        return f"skip {dataset_id}"
    months = months_for(dataset_id)
    d = RAW_DIR / dataset_id
    missing = [m for m in months if not (d / f"{m}.csv.gz").exists() and not (d / f"{m}.empty").exists()]
    if missing:
        raise RuntimeError(f"{dataset_id}: months not fetched yet {missing}")
    qc = candidates().set_index("datasetID").loc[dataset_id, "qc_var"]
    qc = qc if isinstance(qc, str) else None
    # Pool months first: a solar-time window can straddle a UTC month edge.
    parts = [good_samples(pd.read_csv(f), qc) for f in sorted(d.glob("*.csv.gz"))]
    parts = [x for x in parts if len(x)]
    prof = window_profiles(pd.concat(parts, ignore_index=True), dataset_id) if parts else pd.DataFrame()
    if prof.empty:
        b.write_marker(empty, {"reason": "no QC-passing CHLA in 0-200 m"})
        return f"empty {dataset_id}"
    b.atomic_write_parquet(prof, out)
    return f"done {dataset_id}: {len(prof)} windows"


def shore_distance_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Distance to the Natural Earth 10 m coastline (approximate: degrees
    scaled by latitude, fine at the few-km level this flag is for)."""
    import cartopy.io.shapereader as shp
    import shapely
    geoms = list(shp.Reader(shp.natural_earth("10m", "physical", "coastline")).geometries())
    tree = shapely.STRtree(geoms)
    out = np.empty(len(lat))
    for i, (a, o) in enumerate(zip(lat, lon)):
        k = np.cos(np.deg2rad(a))
        # nearest in degree space, then exact distance in the local scaled frame
        cand = tree.query(shapely.Point(o, a).buffer(3.0))
        best = np.inf
        for j in cand:
            g = shapely.affinity.scale(geoms[j], xfact=k, yfact=1.0, origin=(0, 0))
            best = min(best, g.distance(shapely.Point(o * k, a)))
        out[i] = best * 111.2
    return out


# --------------------------------------------------------------------------
# Dark (deep) offset, Eli 2026-11-24
# --------------------------------------------------------------------------
# Below 300 m the fluorometer should read ~0. On 2026-11-24 the California
# Current Sprays read 0.09-0.34 there, flat from 150 to 500 m and drifting
# within a deployment (sp020 0.28 -> 0.17 -> 0.20); East Coast Sprays and
# ru26d 0.03-0.10. The offset is estimated per glider profile and smoothed in
# time, like Argo's dark correction, and published as `dark_offset`, but NOT
# subtracted (Eli, 2026-11-24): subtracting it doubled night PACE/glider
# (1.19 -> 1.99; Argo ~1.25), and it keeps CHLA on the same basis as OOI.
# 300 m, not 150 m: East Coast Sprays still have real CHLA at 150-200 m.
# Gliders that never reach 300 m (shelf Slocums, capex690) get NaN: their
# deepest water can hold real chlorophyll, so it is not a dark reference.

DEEP_DIR = GLIDER_DIR / "raw_deep"
DARK_MIN_DEPTH = 300.0
DARK_MIN_SAMPLES = 10       # deep samples in the 3-day window
DARK_SMOOTH = "3D"          # centred running median over deep samples
DARK_MAX_GAP = pd.Timedelta(days=1.5)  # farthest a window may borrow a dark value


def fetch_deep_unit(dataset_id: str, month: str) -> str:
    """One dataset-month of CHLA samples below DARK_MIN_DEPTH (raw ERDDAP)."""
    d = DEEP_DIR / dataset_id
    out, empty = d / f"{month}.csv.gz", d / f"{month}.empty"
    if out.exists() or empty.exists():
        return f"skip {dataset_id} {month}"
    c = candidates().set_index("datasetID").loc[dataset_id]
    cols = ["profile_id", "time", "precise_time", "depth", "chlorophyll_a"]
    if isinstance(c.qc_var, str):
        cols.append(c.qc_var)
    t0, t1 = b._month_bounds(month)
    df = _tabledap_csv(dataset_id, cols, [f"time>={t0}", f"time<{t1}",
                                          f"depth>={DARK_MIN_DEPTH:g}", "chlorophyll_a!=NaN"])
    if df is None:
        b.write_marker(empty, {"reason": "no matching results", "fetched": pd.Timestamp.now("UTC")})
        return f"empty {dataset_id} {month}"
    d.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    with gzip.open(tmp, "wb") as fh:
        df.to_csv(fh, index=False)
    os.replace(tmp, out)
    return f"done {dataset_id} {month}: {len(df)} rows"


def dark_series(dataset_id: str) -> pd.DataFrame:
    """Dark value per glider profile that reaches DARK_MIN_DEPTH.

    dark_smooth is the centred 3-day running median of all QC-passing samples
    below 300 m (not of per-profile medians: real-time Slocum sets have only
    ~2 deep CHLA samples per profile, ru26d-20250107), taken at the profile's
    mean deep-sample time; it needs DARK_MIN_SAMPLES samples in the window.
    dark_profile (the profile's own deep median) is kept for checking."""
    files = sorted((DEEP_DIR / dataset_id).glob("*.csv.gz"))
    cols = ["glider_profile_id", "t", "dark_profile", "n_deep", "dark_smooth"]
    if not files:
        return pd.DataFrame(columns=cols)
    qc = candidates().set_index("datasetID").loc[dataset_id, "qc_var"]
    qc = qc if isinstance(qc, str) else None
    deep = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    deep["profile_time"] = pd.to_datetime(deep["time"], utc=True)
    deep["t"] = pd.to_datetime(deep["precise_time"], utc=True).fillna(deep["profile_time"])
    ok = np.isfinite(deep.chlorophyll_a)
    if qc:
        ok &= deep[qc].isin(QC_KEEP)
    deep = deep[ok].sort_values("t").reset_index(drop=True)
    if deep.empty:
        return pd.DataFrame(columns=cols)
    deep["smooth"] = deep.rolling(DARK_SMOOTH, on="t", center=True,
                                  min_periods=DARK_MIN_SAMPLES).chlorophyll_a.median()
    p = (deep.groupby("profile_id")
         .agg(t=("t", "mean"), dark_profile=("chlorophyll_a", "median"),
              n_deep=("chlorophyll_a", "size"), dark_smooth=("smooth", "median"))
         .reset_index().rename(columns={"profile_id": "glider_profile_id"}))
    return p.dropna(subset=["dark_smooth"]).sort_values("t").reset_index(drop=True)[cols]


def assign_dark(win: pd.DataFrame, ds: pd.DataFrame) -> pd.Series:
    """Dark value for each window row: its own glider profile's smoothed
    value, else the nearest smoothed value within DARK_MAX_GAP, else NaN."""
    out = pd.Series(np.nan, index=win.index)
    if ds.empty:
        return out
    own = win.glider_profile_id.map(ds.set_index("glider_profile_id").dark_smooth)
    out[:] = own.to_numpy()
    miss = out.isna()
    if miss.any():
        q = win.loc[miss, ["time"]].reset_index().sort_values("time")
        near = pd.merge_asof(q, ds[["t", "dark_smooth"]].sort_values("t"), left_on="time",
                             right_on="t", direction="nearest", tolerance=DARK_MAX_GAP)
        out[near["index"].to_numpy()] = near.dark_smooth.to_numpy()
    return out


def combine() -> tuple[pd.DataFrame, dict]:
    """Window files -> the glider table.

    CHLA_A_B = CHLA_FLUOR_A_B / chla_ratio (Sauzède, as OOI). CHLA_FLUOR_A_B
    are the fluorometer bin means as published after QC (Eli asked for the
    raw values in the table). dark_offset is the fluorometer value below
    300 m (NaN where the glider does not reach 300 m); it is not subtracted,
    so a user who wants it removed computes
    (CHLA_FLUOR_A_B - dark_offset) / chla_ratio."""
    import sys
    sys.path.insert(0, str(b.REPO / "src"))
    from sauzede_lut import load_lut, chla_scaling_factor
    ids = selection()
    missing = [d for d in ids if not (WIN_DIR / f"{d}.parquet").exists()]
    if missing:
        raise RuntimeError(f"window files missing for {missing}; run the glider stage")
    t = pd.concat([pd.read_parquet(WIN_DIR / f"{d}.parquet") for d in ids], ignore_index=True)
    t = t.sort_values("time").reset_index(drop=True)
    if t.profile_id.duplicated().any():
        raise ValueError("duplicated profile_id")
    c = candidates().set_index("datasetID")
    t["region"] = t.dataset_id.map(c.region)
    t["platform_type"] = t.dataset_id.map(c.platform_type)
    t["institution"] = t.dataset_id.map(c.institution)
    solar = t.time + pd.to_timedelta(t.lon / 15.0, unit="h")
    t["solar_hour"] = (solar - solar.dt.floor("D")).dt.total_seconds() / 3600
    t["shore_km"] = shore_distance_km(t.lat.to_numpy(), t.lon.to_numpy())
    fluor_cols = [c_.replace("CHLA_", "CHLA_FLUOR_") for c_ in b.CHLA_COLS]
    for raw_col, col in zip(fluor_cols, b.CHLA_COLS):
        t[raw_col] = t[col]
    dark = pd.Series(np.nan, index=t.index)
    for d in ids:
        m = t.dataset_id == d
        dark[m] = assign_dark(t[m], dark_series(d)).to_numpy()
    t["dark_offset"] = dark
    ratio = chla_scaling_factor(t.lat.values, t.lon.values, lut=load_lut())
    t["chla_ratio"] = ratio
    apply = np.where(np.isnan(ratio), 1.0, ratio)
    for col in b.CHLA_COLS:
        t[col] = t[col] / apply
    order = (["profile_id", "time", "lat", "lon", "solar_hour", "n_samples", "max_depth"]
             + b.CHLA_COLS + ["chla_ratio"] + fluor_cols + ["dark_offset"]
             + ["dataset_id", "glider_profile_id", "profiles_in_window", "platform_type",
                "institution", "region", "shore_km"])
    t = t[order]
    per = t.groupby("dataset_id").agg(rows=("profile_id", "size"),
                                      dark_offset_rows=("dark_offset", "count"),
                                      dark_median=("dark_offset", "median"))
    return t, {"rows": len(t), "datasets": len(ids),
               "datasets_without_lut": sorted(t.loc[np.isnan(ratio), "dataset_id"].unique().tolist()),
               "by_dataset": per.round(3).to_dict(orient="index")}

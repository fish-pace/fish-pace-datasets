"""CHLA-Z glider table: IOOS Glider DAC chlorophyll profiles.

Coastal CHLA-Z (Eli Holmes): processed like OOI (3 h windows of local mean
solar time, 10 m bins (A, B] in 0-200 m, Sauzède correction). The 2026_11_24
release had 20 pilot deployments; since #19 the deployments and their
metadata (chlorophyll variable, QC variable, region, scan verdict) come from
the whole-DAC scan, `src/chla_z_glider_candidates.csv` (`chla_z_glider_scan`).
Released like Argo and OOI: `datasets/chla_z/build.py` stages `glider-scan`
(optional), `glider-fetch`, `glider`, then the PACE and OLCI stages with
`--platform glider`. Design note: `claude/notes/chla-z-glider.md`.

Units are resumable like `chla_z_build` (one file per unit, atomic write,
skipped if present); intermediates go to `_temp_data/glider/`:

    meta()                     the scan CSV, one row per deployment (committed)
    write_selection()          scan CSV -> chla_z_glider_selection.json (the release list)
    candidates(), screen_unit  the pilot's search and one-day screen (kept; chose the first 20)
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
# Flag schemes: 1 good, 2 not evaluated / probably good. 0 is "no QC performed"
# in the 0-9 scheme: kept like datasets with no flag at all. Only the Oregon
# State Seagliders (fluorescence_qc) use it, on every value (2026-11-24, #19).
QC_KEEP = (0, 1, 2)
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
    """The deployments in the table, committed (written by write_selection)."""
    return json.loads(SELECTION_PATH.read_text())["datasets"]


# --------------------------------------------------------------------------
# Deployment metadata and selection, from the whole-DAC scan (#16, #19)
# --------------------------------------------------------------------------

META_PATH = SELECTION_PATH.with_name("chla_z_glider_candidates.csv")
DEEP_OFFSET_FAIL = "deep offset > 0.2"
_META = None


def meta() -> pd.DataFrame:
    """The committed scan result, indexed by datasetID. Read from the CSV, not
    from the hub-only scan cache, so the fetch and the table are reproducible
    from the repo."""
    global _META
    if _META is None:
        m = pd.read_csv(META_PATH)
        for k in ("minTime", "maxTime"):
            m[k] = pd.to_datetime(m[k], utc=True)
        for k in ("qc_var", "fail", "check", "bad_days"):
            m[k] = m[k].astype(object).where(m[k].notna(), None)
        _META = m.set_index("datasetID")
    return _META


def select_from_scan(m: pd.DataFrame) -> pd.DataFrame:
    """Eli, 2026-11-24 (#19): every `ok` and `check` deployment, plus the
    deployments whose ONLY failure is the deep offset (kept, flagged per row:
    deep_offset_flag). Other failures (negative offset, flat, no signal,
    surface < deep, implausibly high, or deep offset together with one of
    those) stay out. Needs some record inside the fetch months."""
    keep = m.verdict.isin(["ok", "check"]) | (m.fail == DEEP_OFFSET_FAIL)
    return m[keep & (m.days_in_months > 0)]


def write_selection() -> list[str]:
    s = select_from_scan(meta())
    ids = s.reset_index().sort_values(["region", "minTime"]).datasetID.tolist()
    SELECTION_PATH.write_text(json.dumps({
        "chosen": ("2026-11-24 (#19), by chla_z_glider.write_selection from "
                   "src/chla_z_glider_candidates.csv: scan verdict ok or check, or a "
                   "fail whose only reason is the deep offset (flagged per row), with "
                   "days inside the fetch months. Includes the 20 pilot deployments of "
                   "release 2026_11_24. See claude/notes/chla-z-glider.md."),
        "datasets": ids}, indent=1) + "\n")
    return ids


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def months_for(dataset_id: str) -> list[str]:
    c = meta().loc[dataset_id]
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
    c = meta().loc[dataset_id]
    # The raw CSV keeps the dataset's own variable names (chlorophyll,
    # fluorescence, sci_flbbcd_chlor_units, ...); _standard() renames on read.
    cols = ["profile_id", "time", "latitude", "longitude", "precise_time", "depth", c.chl_var]
    if c.qc_var:  # None when the dataset has no CHLA flag
        cols.append(c.qc_var)
    if c.has_precise_lat:
        cols += ["precise_lat", "precise_lon"]
    t0, t1 = b._month_bounds(month)
    df = _tabledap_csv(dataset_id, cols, [f"time>={t0}", f"time<{t1}", "depth<=200",
                                          f"{c.chl_var}!=NaN"])
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

CHL_RANGE = (-5.0, 100.0)  # the scan's range (day_unit): outside it is a fill value


def _standard(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Raw CSV -> chlorophyll in a column named chlorophyll_a, whatever the
    dataset calls it."""
    v = meta().loc[dataset_id, "chl_var"]
    return df.rename(columns={v: "chlorophyll_a"}) if v != "chlorophyll_a" else df


def _pkey(df: pd.DataFrame) -> pd.Series:
    """One glider profile = DAC profile_id + its profile `time`. profile_id
    alone is not unique in 27 selected datasets (DFO / C-PROOF reuse it for
    profiles days apart; also SEA044, osu592, sg263, sp011, sp030), and
    grouping by it merged unrelated profiles: positions zigzagged ~45 km
    (dfo-bumblebee998-20260205), found in notebook 06, 2026-11-25. Not an
    issue for any of the pilot 20."""
    return df["profile_id"].astype(str) + "@" + df["time"].astype(str)


def good_samples(df: pd.DataFrame, qc_var: str | None) -> pd.DataFrame:
    """QC-passing samples in (0, 200] m. Sample time is precise_time where
    present: the profile `time` can be ~70 min off the CHLA samples (ru34)."""
    df = df.copy()
    df["profile_time"] = pd.to_datetime(df["time"], utc=True)
    pt = pd.to_datetime(df["precise_time"], utc=True) if "precise_time" in df else None
    df["t"] = df["profile_time"] if pt is None else pt.fillna(df["profile_time"])
    ok = (df.depth > 0) & (df.depth <= 200) & df.chlorophyll_a.between(*CHL_RANGE, inclusive="neither")
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
    g["pkey"] = _pkey(g)
    p = g.groupby("pkey").agg(time=("t", "mean"), lat=("plat", "mean"), lon=("plon", "mean"),
                                    gpid=("profile_id", "first"), n_samples=("chlorophyll_a", "size"),
                                    n_bins=("bin", "nunique"), max_depth=("depth", "max"))
    # A profile with no position at all cannot be placed (dfo-rosie713 profile 1:
    # latitude and precise_lat both NaN). Profile-level, so no other row changes.
    p = p[p.lat.notna() & p.lon.notna()]
    if p.empty:
        return pd.DataFrame()
    solar = p.time + pd.to_timedelta(p.lon / 15.0, unit="h")
    win = (solar.dt.hour // WINDOW_H) * WINDOW_H
    p["key"] = solar.dt.floor("D").dt.strftime("%Y%m%d") + "T" + win.map("{:02d}".format)
    centre = solar.dt.floor("D") + pd.to_timedelta(win + WINDOW_H / 2, unit="h")
    p["off_centre"] = (solar - centre).abs()
    p = p.sort_values(["key", "n_bins", "off_centre"], ascending=[True, False, True])
    pick = p[~p.key.duplicated()]
    sel = g[g.pkey.isin(pick.index)]
    wide = (sel.groupby(["pkey", "bin"]).chlorophyll_a.mean().unstack("bin")
               .reindex(columns=range(b.N_BINS)))
    wide.columns = b.CHLA_COLS
    out = pick.join(wide).reset_index().rename(columns={"gpid": "glider_profile_id", "pkey": "_pkey"})
    out["profiles_in_window"] = out.key.map(p.key.value_counts())
    out.insert(0, "profile_id", dataset_id + "_" + out["key"])
    out["dataset_id"] = dataset_id
    return out[["profile_id", "time", "lat", "lon", "n_samples", "max_depth"] + b.CHLA_COLS
               + ["dataset_id", "glider_profile_id", "profiles_in_window", "_pkey"]]


def window_unit(dataset_id: str, force: bool = False) -> str:
    out, empty = WIN_DIR / f"{dataset_id}.parquet", WIN_DIR / f"{dataset_id}.empty"
    if (out.exists() or empty.exists()) and not force:
        return f"skip {dataset_id}"
    months = months_for(dataset_id)
    d = RAW_DIR / dataset_id
    missing = [m for m in months if not (d / f"{m}.csv.gz").exists() and not (d / f"{m}.empty").exists()]
    if missing:
        raise RuntimeError(f"{dataset_id}: months not fetched yet {missing}")
    qc = meta().loc[dataset_id, "qc_var"]
    # Pool months first: a solar-time window can straddle a UTC month edge.
    parts = [good_samples(_standard(pd.read_csv(f), dataset_id), qc) for f in sorted(d.glob("*.csv.gz"))]
    parts = [x for x in parts if len(x)]
    prof = window_profiles(pd.concat(parts, ignore_index=True), dataset_id) if parts else pd.DataFrame()
    if prof.empty:
        b.write_marker(empty, {"reason": "no QC-passing CHLA in 0-200 m"})
        return f"empty {dataset_id}"
    b.atomic_write_parquet(prof, out)
    return f"done {dataset_id}: {len(prof)} windows"


GREAT_LAKES = ("Lake Superior", "Lake Michigan", "Lake Huron", "Lake Erie", "Lake Ontario")


def _shore_geoms(lakes: bool) -> list:
    import cartopy.io.shapereader as shp
    if not lakes:
        return list(shp.Reader(shp.natural_earth("10m", "physical", "coastline")).geometries())
    r = shp.Reader(shp.natural_earth("10m", "physical", "lakes"))
    return [rec.geometry.boundary for rec in r.records() if rec.attributes.get("name") in GREAT_LAKES]


def shore_distance_km(lat: np.ndarray, lon: np.ndarray, lakes: bool = False) -> np.ndarray:
    """Distance to the Natural Earth 10 m coastline, or with lakes=True to the
    Great Lakes shorelines (the coastline layer has no lake shores; the one
    Great Lakes deployment got ~840 km before, 2026-11-25). Approximate:
    degrees scaled by cos(lat), fine at the few-km level this column is for.

    The search widens until something is found within r (scaled degrees):
    anything nearer lies within r / cos(lat) unscaled, so the buffered query
    cannot miss it. Before 2026-11-25 the search stopped at 3 degrees and
    returned inf offshore (393 rows of the 2026_11_24 table)."""
    import shapely
    geoms = _shore_geoms(lakes)
    tree = shapely.STRtree(geoms)
    out = np.empty(len(lat))
    for i, (a, o) in enumerate(zip(lat, lon)):
        k = max(np.cos(np.deg2rad(a)), 0.05)
        p = shapely.Point(o * k, a)
        r, best = 3.0, np.inf
        while True:
            for j in tree.query(shapely.Point(o, a).buffer(r / k)):
                g = shapely.affinity.scale(geoms[j], xfact=k, yfact=1.0, origin=(0, 0))
                best = min(best, g.distance(p))
            if best <= r or r >= 180:
                break
            r *= 3
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
    c = meta().loc[dataset_id]
    if c.max_z < DARK_MIN_DEPTH:  # never reaches 300 m in the scan: nothing to ask for
        b.write_marker(empty, {"reason": f"scan max_z {c.max_z:g} m < {DARK_MIN_DEPTH:g} m"})
        return f"empty {dataset_id} {month} (shallow)"
    cols = ["profile_id", "time", "precise_time", "depth", c.chl_var]
    if c.qc_var:
        cols.append(c.qc_var)
    t0, t1 = b._month_bounds(month)
    df = _tabledap_csv(dataset_id, cols, [f"time>={t0}", f"time<{t1}",
                                          f"depth>={DARK_MIN_DEPTH:g}", f"{c.chl_var}!=NaN"])
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
    cols = ["_pkey", "t", "dark_profile", "n_deep", "dark_smooth"]
    if not files:
        return pd.DataFrame(columns=cols)
    qc = meta().loc[dataset_id, "qc_var"]
    deep = _standard(pd.concat([pd.read_csv(f) for f in files], ignore_index=True), dataset_id)
    deep["profile_time"] = pd.to_datetime(deep["time"], utc=True)
    deep["t"] = pd.to_datetime(deep["precise_time"], utc=True).fillna(deep["profile_time"])
    ok = deep.chlorophyll_a.between(*CHL_RANGE, inclusive="neither")
    if qc:
        ok &= deep[qc].isin(QC_KEEP)
    deep = deep[ok].sort_values("t").reset_index(drop=True)
    if deep.empty:
        return pd.DataFrame(columns=cols)
    deep["smooth"] = deep.rolling(DARK_SMOOTH, on="t", center=True,
                                  min_periods=DARK_MIN_SAMPLES).chlorophyll_a.median()
    deep["_pkey"] = _pkey(deep)
    p = (deep.groupby("_pkey")
         .agg(t=("t", "mean"), dark_profile=("chlorophyll_a", "median"),
              n_deep=("chlorophyll_a", "size"), dark_smooth=("smooth", "median"))
         .reset_index())
    return p.dropna(subset=["dark_smooth"]).sort_values("t").reset_index(drop=True)[cols]


def assign_dark(win: pd.DataFrame, ds: pd.DataFrame) -> pd.Series:
    """Dark value for each window row: its own glider profile's smoothed
    value, else the nearest smoothed value within DARK_MAX_GAP, else NaN."""
    out = pd.Series(np.nan, index=win.index)
    if ds.empty:
        return out
    own = win._pkey.map(ds.set_index("_pkey").dark_smooth)
    out[:] = own.to_numpy()
    miss = out.isna()
    if miss.any():
        q = win.loc[miss, ["time"]].reset_index().sort_values("time")
        near = pd.merge_asof(q, ds[["t", "dark_smooth"]].sort_values("t"), left_on="time",
                             right_on="t", direction="nearest", tolerance=DARK_MAX_GAP)
        out[near["index"].to_numpy()] = near.dark_smooth.to_numpy()
    return out


BAD_MID_ABOVE = 0.3   # 150-200 m value above the deployment median by this much
BAD_ALL_BELOW = 0.02  # or every bin of the profile below this
MID_COLS = b.CHLA_COLS[15:20]  # (150, 200] m


def bad_window_mask(t: pd.DataFrame, max_z: pd.Series) -> pd.Series:
    """Whole-column failure windows: the scan's bad-day rule
    (chla_z_glider_scan.fullrec_stats) applied to every window instead of
    every 4th day, so the days between sampled bad days are caught too (Eli,
    2026-11-24, #19: drop them). California Sprays near the end of a record
    go high, or ~0, at every depth (sensor failure or recovery).

    Only for deployments that dive >= 300 m (scan max_z), on the fluorometer
    (CHLA_FLUOR) 150-200 m mean: on a mixed shelf a bloom really does raise
    the whole column. Bad = 150-200 m mean > the deployment's median of it +
    BAD_MID_ABOVE, or every filled bin < BAD_ALL_BELOW with >= 3 bins."""
    fl = t[MID_COLS]
    mid = fl.mean(axis=1)
    allb = t[b.CHLA_COLS]
    deep = t.dataset_id.map(max_z).fillna(0) >= DARK_MIN_DEPTH
    med = mid.groupby(t.dataset_id).transform("median")
    bad = (mid > med + BAD_MID_ABOVE) | ((allb.max(axis=1) < BAD_ALL_BELOW) & (allb.count(axis=1) >= 3))
    return bad & deep


def drop_bad_windows(t: pd.DataFrame, max_z: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Must run before the Sauzède division (the rule is on fluorometer
    values). Returns the kept rows and the number dropped per dataset."""
    bad = bad_window_mask(t, max_z)
    return t[~bad].reset_index(drop=True), t[bad].groupby("dataset_id").size()


SPIKE_KMH = 10.0  # gliders move ~1 km/h; in the Gulf Stream core < ~9 km/h


def position_spike_mask(t: pd.DataFrame) -> pd.Series:
    """Rows whose position is a bad fix: implied speed > SPIKE_KMH both from
    the previous row and to the next row of the same deployment. Found
    running notebook 06 (2026-11-25): sp062-20250702 had a row at +37 E
    (its rows are off Hatteras), sp213 one at 17 N 43 W, sp030 one 600 km
    away, sg263 two on land, dfo-bumblebee998-20240509 repeated jumps to a
    fixed point near Vancouver Island; sp058-20241121 (pilot) one 35 km away
    and back in 3 h. Both sides, so a real fast drift (one long step) stays."""
    s = t.sort_values(["dataset_id", "time"])
    g = s.groupby("dataset_id")
    k = np.cos(np.deg2rad(s.lat))

    def speed(shift):
        dt = (s.time - g.time.shift(shift)).abs().dt.total_seconds() / 3600
        dk = 111.2 * np.hypot(s.lat - g.lat.shift(shift), (s.lon - g.lon.shift(shift)) * k)
        return dk / dt

    bad = (speed(1) > SPIKE_KMH) & (speed(-1) > SPIKE_KMH)
    return bad.reindex(t.index).fillna(False)


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
    no_data = [d for d in ids if (WIN_DIR / f"{d}.empty").exists()]
    missing = [d for d in ids if not (WIN_DIR / f"{d}.parquet").exists() and d not in no_data]
    if missing:
        raise RuntimeError(f"window files missing for {missing}; run the glider stage")
    ids = [d for d in ids if d not in no_data]
    t = pd.concat([pd.read_parquet(WIN_DIR / f"{d}.parquet") for d in ids], ignore_index=True)
    t = t.sort_values("time").reset_index(drop=True)
    # The fetch selects by profile `time`; rows use the samples' precise_time,
    # which can fall minutes outside the release period (12 rows, 2026-11-24).
    t = t[(t.time >= pd.Timestamp(b.START, tz="UTC"))
          & (t.time < pd.Timestamp(b.END_EXCLUSIVE, tz="UTC"))].reset_index(drop=True)
    if t.profile_id.duplicated().any():
        raise ValueError("duplicated profile_id")
    spike = position_spike_mask(t)
    spikes = t[spike].groupby("dataset_id").size()
    t = t[~spike].reset_index(drop=True)
    c = meta()
    # Region of the median position of the deployment's rows, with the scan's
    # region(). Not the scan CSV's label: that uses the ERDDAP bounding-box
    # centre, and some boxes hold a stray position (gp_276-20231024 has
    # maxLongitude -70.9 while it sits at Station Papa, 145 W; sp062-20250702
    # has a centre at 21.5 W while its rows are off Georgia), 2026-11-25.
    import chla_z_glider_scan as gs
    pos = t.groupby("dataset_id")[["lat", "lon"]].median()
    t["region"] = t.dataset_id.map(pd.Series([gs.region(a, o) for a, o in zip(pos.lat, pos.lon)],
                                             index=pos.index))
    t["platform_type"] = t.dataset_id.map(c.platform_type)
    t["institution"] = t.dataset_id.map(c.institution)
    # Per-deployment scan results, on every row so QC is a one-line filter
    # (Eli, 2026-11-24, #19).
    t["deep_offset_flag"] = t.dataset_id.map(c.fail).fillna("").str.contains(DEEP_OFFSET_FAIL, regex=False)
    t["scan_check"] = t.dataset_id.map(c.check).fillna("")
    t, bad = drop_bad_windows(t, c.max_z)
    solar = t.time + pd.to_timedelta(t.lon / 15.0, unit="h")
    t["solar_hour"] = (solar - solar.dt.floor("D")).dt.total_seconds() / 3600
    t["shore_km"] = shore_distance_km(t.lat.to_numpy(), t.lon.to_numpy())
    lake = (t.region == "Great Lakes").to_numpy()
    if lake.any():
        t.loc[lake, "shore_km"] = shore_distance_km(t.lat[lake].to_numpy(), t.lon[lake].to_numpy(), lakes=True)
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
             + ["deep_offset_flag", "scan_check"]
             + ["dataset_id", "glider_profile_id", "profiles_in_window", "platform_type",
                "institution", "region", "shore_km"])
    t = t[order]
    per = t.groupby("dataset_id").agg(rows=("profile_id", "size"),
                                      dark_offset_rows=("dark_offset", "count"),
                                      dark_median=("dark_offset", "median"))
    per["bad_windows_dropped"] = bad.reindex(per.index).fillna(0).astype(int)
    per["position_spikes_dropped"] = spikes.reindex(per.index).fillna(0).astype(int)
    return t, {"rows": len(t), "datasets": len(ids),
               "bad_windows_dropped": int(bad.sum()),
               "position_spikes_dropped": int(spikes.sum()),
               "datasets_without_qc_passing_data": no_data,
               "datasets_without_lut": sorted(t.loc[np.isnan(ratio), "dataset_id"].unique().tolist()),
               "by_dataset": per.round(3).to_dict(orient="index")}

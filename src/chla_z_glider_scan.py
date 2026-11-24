"""Scan the IOOS Glider DAC for glider chlorophyll deployments (issue #16).

Repeatable: every step caches per dataset in `_temp_data/glider/scan/` and
`_temp_data/glider/fullrec/`, so a later run fetches only datasets that are
new or have grown. Design note: claude/notes/chla-z-glider.md, "Scan
(2026-11-24)".

    catalog(refresh)        every DAC dataset with a chlorophyll variable and data since START
    deployments(cat)        one dataset per deployment (delayed copy only if it covers the real-time one)
    day_unit(id, var, day)  one day as 10 m bin means, by ERDDAP, every 4th day -> sampled/<id>/<day>.csv
    fullrec_stats(id)       sensor checks across the whole record (on the sampled days)
    scan(...)               all of the above + status against the released selection

Differences from the pilot's `chla_z_glider.candidates` (2026-11-23): the
pilot searched only the variable name `chlorophyll_a`. The DAC also uses
`chlorophyll` (most Spray-like and many Slocum sets, OOI gliders, C-PROOF,
Skidaway, UAF...), `fluorescence` (Oregon State Seagliders) and raw Slocum
`sci_*_chlor_*` names, which more than doubles the PACE-era datasets.
"""

from __future__ import annotations

import io
import json
import os
import time
import urllib.parse

import numpy as np
import pandas as pd

import chla_z_build as b
import chla_z_glider as g

SCAN_DIR = g.GLIDER_DIR / "scan"
INFO_DIR = SCAN_DIR / "info"
SAMPLE_DIR = g.GLIDER_DIR / "sampled"
SAMPLE_EVERY = 4  # days between sampled days
CATALOG_PATH = SCAN_DIR / "catalog.parquet"

# Chlorophyll variables, most preferred first. A dataset with several uses the first.
CHL_VARS = ("chlorophyll_a", "chlorophyll", "fluorescence", "sci_flbbcd_chlor_units",
            "sci_flbb_chlor_units", "sci_bbfl2s_chlor_scaled")
MIN_DAYS = 7  # days of record since START (the pilot used 14)


def _get(url: str, attempts: int = 5) -> tuple[int, bytes]:
    """The DAC returns spurious empty-body 404s and truncated bodies under load
    (2026-11-24: ~120 of 471 info requests at 6 threads). Retry those; a 404
    whose body says 'no matching results' is a real answer."""
    for a in range(attempts):
        status, body = b._http_get(url)
        if status != 404 or b"no matching results" in body.lower():
            return status, body
        time.sleep(5 * (a + 1))
    return status, body


def _info(dataset_id: str) -> dict:
    p = INFO_DIR / f"{dataset_id}.json"
    for _ in range(3):
        if not p.exists():
            status, body = _get(f"{g.GLIDER_ERDDAP}/info/{dataset_id}/index.json")
            if status == 404:
                raise RuntimeError(f"no info for {dataset_id}")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body)
        try:
            rows = json.loads(p.read_text())["table"]["rows"]
            break
        except json.JSONDecodeError:  # truncated download
            p.unlink()
    else:
        raise RuntimeError(f"unparseable info for {dataset_id}")
    var = {r[1] for r in rows if r[0] == "variable"}
    att = {(r[1], r[2]): r[4] for r in rows if r[0] == "attribute"}
    chl = next((v for v in CHL_VARS if v in var), None)
    qc = next((q for q in (f"{chl}_qc", "chlorophyll_a_qartod_summary_flag") if q in var), None)
    return {"datasetID": dataset_id, "chl_var": chl, "chl_units": att.get((chl, "units")),
            "qc_var": qc, "platform_type": att.get(("NC_GLOBAL", "platform_type")),
            "has_depth": "depth" in var, "has_precise_lat": "precise_lat" in var,
            "has_precise_time": "precise_time" in var, "title": att.get(("NC_GLOBAL", "title")), "n_vars": len(var)}


def catalog(refresh: bool = False) -> pd.DataFrame:
    """Every tabledap dataset with one of CHL_VARS and maxTime >= START.

    refresh re-reads the variable index and allDatasets (cheap) and fetches
    info only for datasets not seen before; delete scan/info/ to re-read all."""
    if CATALOG_PATH.exists() and not refresh:
        return pd.read_parquet(CATALOG_PATH)
    ids = set()
    for v in CHL_VARS:
        status, body = _get(f"{g.GLIDER_ERDDAP}/categorize/variableName/{v}/index.csv"
                            "?page=1&itemsPerPage=100000")
        if status == 200:
            ids |= set(pd.read_csv(io.BytesIO(body))["Dataset ID"])
    _, body = _get(f"{g.GLIDER_ERDDAP}/tabledap/allDatasets.csv?datasetID,institution,"
                   "minLongitude,maxLongitude,minLatitude,maxLatitude,minTime,maxTime")
    alld = pd.read_csv(io.BytesIO(body), skiprows=[1])
    c = alld[alld.datasetID.isin(ids)].copy()
    for k in ("minTime", "maxTime"):
        c[k] = pd.to_datetime(c[k], utc=True)
    c = c[c.maxTime >= pd.Timestamp(b.START, tz="UTC")]
    c = c.merge(pd.DataFrame([_info(d) for d in c.datasetID]), on="datasetID")
    x = c.datasetID.str.extract(r"^(.*)-(\d{8})T\d{4}(?:-delayed)?$")
    # Lower case: Stony Brook has SBU02-...-delayed next to sbu02-... real time,
    # with start times 2 min apart, so the key is glider + start date.
    c["glider"] = x[0].str.lower()
    c["deployment_key"] = c.glider + "-" + x[1]
    c["delayed"] = c.datasetID.str.endswith("-delayed")
    c["lat"] = (c.minLatitude + c.maxLatitude) / 2
    c["lon"] = (c.minLongitude + c.maxLongitude) / 2
    t0 = c.minTime.clip(lower=pd.Timestamp(b.START, tz="UTC"))
    c["days_since_start"] = (c.maxTime - t0).dt.total_seconds() / 86400
    c["span_days"] = (c.maxTime - c.minTime).dt.total_seconds() / 86400
    c["fetched"] = pd.Timestamp.now("UTC")
    b.atomic_write_parquet(c, CATALOG_PATH)
    return c


def deployments(cat: pd.DataFrame) -> pd.DataFrame:
    """One dataset per deployment: the delayed copy when its span is >= 90% of
    the real-time one (delayed copies can be stubs, e.g. ru26d-20250107 spans
    6 minutes), else the longest real-time copy. `alt_ids` lists the others.

    Exception: delayed copies with > 1000 variables (Skidaway and USF raw
    Slocum sets, ~2,800 variables) are used only when there is no real-time
    copy: ERDDAP did not answer a whole-record query on them within 15 min
    (2026-11-24), while the ~75-variable real-time copy answers in seconds."""
    c = cat[cat.chl_var.notna() & cat.chl_units.notna() & cat.has_depth]
    out = []
    for _, grp in c.groupby("deployment_key"):
        rt, dl = grp[~grp.delayed], grp[grp.delayed]
        if len(rt):
            dl = dl[dl.n_vars <= 1000]
        if len(dl) and (not len(rt) or dl.span_days.max() >= 0.9 * rt.span_days.max()):
            pick = dl.sort_values("span_days").iloc[-1]
        else:
            pick = rt.sort_values("span_days").iloc[-1]
        r = pick.copy()
        r["alt_ids"] = ",".join(sorted(set(grp.datasetID) - {pick.datasetID}))
        out.append(r)
    return pd.DataFrame(out).reset_index(drop=True)


def region(lat: float, lon: float) -> str:
    """Finer than chla_z_glider.region_of (which stays for the released table)."""
    if lat < -50:
        return "Antarctic Peninsula"
    if lat > 54 and lon > -30:
        return "Northern Europe"
    if 41 < lat < 49 and -93 < lon < -76:
        return "Great Lakes"
    if lon < -140 or (lat > 54 and lon < -125):
        return "Gulf of Alaska / Bering"
    if lon < -110 and lat >= 48:
        return "British Columbia / Salish Sea"
    if lon < -110 and lat >= 41:
        return "Pacific Northwest (OR/WA)"
    if lon < -110:
        return "California Current"
    if lat < 31 and lon < -81.5:
        return "Gulf of Mexico / West Florida"
    if lat < 25:
        return "Tropical Atlantic / Caribbean"
    if lon > -70 and lat < 36:
        return "Sargasso / Bermuda"
    if lon > -68 and lat < 42:
        return "Open NW Atlantic (slope sea / Gulf Stream)"
    if lon > -50 and lat > 50:
        return "Irminger / subpolar N Atlantic"
    if lat < 35:
        return "South Atlantic Bight / off Hatteras"
    if lat > 40.6 and lon > -72:
        return "Southern New England / Gulf of Maine"
    return "Mid-Atlantic Bight and slope"


# --------------------------------------------------------------------------
# Whole-record check, on sampled days
# --------------------------------------------------------------------------

def sample_days(min_time, max_time) -> list[str]:
    """Every n-th UTC day of the record since START, plus the last full day:
    n = 1 under 20 days, 2 under 40, else SAMPLE_EVERY (so short deployments
    still get >= 5 days for the stats). Fixed to the calendar (days since
    START), not to the deployment start, so a grown deployment keeps its
    cached days and only adds new ones."""
    t0 = max(pd.Timestamp(min_time), pd.Timestamp(b.START, tz="UTC")).normalize()
    t1 = pd.Timestamp(max_time).normalize()
    span = (t1 - t0).days
    n = 1 if span < 20 else 2 if span < 40 else SAMPLE_EVERY
    off = (t0 - pd.Timestamp(b.START, tz="UTC")).days % n
    first = t0 + pd.Timedelta(days=(n - off) % n)
    days = list(pd.date_range(first, t1, freq=f"{n}D"))
    last = t1 - pd.Timedelta(days=1)  # t1 itself is usually a partial day
    if last >= t0 and (not days or last > days[-1]):
        days.append(last)
    return [f"{d:%Y-%m-%d}" for d in days]


def day_unit(dataset_id: str, chl_var: str, day: str) -> str:
    """One UTC day of chlorophyll as 10 m bin means, aggregated by ERDDAP
    (orderByMean), -> sampled/<id>/<day>.csv. Values outside (-5, 100) are
    dropped first so fill values cannot dominate a mean; the returned `depth`
    is the bin number (depth // 10).

    Why days and not the whole record: a whole-record orderByMean makes ERDDAP
    open every profile file; it took 1-10 min per deployment and hung on the
    ~2,800-variable raw Slocum sets (2026-11-24), while a one-day query
    answers in ~0.5 s because ERDDAP skips files by time."""
    d = SAMPLE_DIR / dataset_id
    out, empty = d / f"{day}.csv", d / f"{day}.empty"
    if out.exists() or empty.exists():
        return f"skip {dataset_id} {day}"
    v = chl_var
    t1 = pd.Timestamp(day) + pd.Timedelta(days=1)
    q = (f"time,depth,{v}&time>={day}T00:00:00Z&time<{t1:%Y-%m-%d}T00:00:00Z&depth<=1000"
         f'&{v}>-5&{v}<100&orderByMean("time/1day,depth/10")')
    status, body = _get(f"{g.GLIDER_ERDDAP}/tabledap/{dataset_id}.csv?"
                        + urllib.parse.quote(q, safe="=&,/<>:"))
    d.mkdir(parents=True, exist_ok=True)
    if status == 404:
        empty.write_text("no matching results")
        return f"empty {dataset_id} {day}"
    tmp = out.with_name(out.name + ".part")
    tmp.write_bytes(body)
    os.replace(tmp, out)
    return f"done {dataset_id} {day}"


def load_grid(dataset_id: str) -> pd.DataFrame | None:
    """The sampled days of one deployment: columns time, bin, chl."""
    files = sorted((SAMPLE_DIR / dataset_id).glob("*.csv"))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f, skiprows=[1]).set_axis(["time", "bin", "chl"], axis=1)
                   for f in files], ignore_index=True)
    return d


def fullrec_stats(dataset_id: str, grid: pd.DataFrame | None = None) -> dict:
    """Checks on the daily x 10 m grid (sampled days, or `grid` = a whole
    record for comparison). "Runs" are in days. Thresholds were set so the two
    pilot deployments that failed on their whole record are caught:
    sp071-20250501T1427 (flat 0.04-0.07 at all depths) and sp055-20250806T2212
    (surface below its own 150 m value for its last 4 weeks)."""
    d = load_grid(dataset_id) if grid is None else grid
    if d is None:
        return {"verdict": "no data on sampled days"}
    d["z"] = d.bin * 10
    o = {"n_days": int(d.time.nunique()), "max_z": float(d.z.max()),
         "first_day": d.time.min()[:10], "last_day": d.time.max()[:10]}
    if o["n_days"] < 5:
        o["verdict"] = "too few days"
        return o
    up = d[(d.z >= 0) & (d.z < 200)]
    pv = up.pivot_table(index="time", columns="z", values="chl")
    surf = d[(d.z >= 0) & (d.z < 20)].groupby("time").chl.mean()
    deep = d[d.z >= 300].groupby("time").chl.mean()
    mid = d[(d.z >= 150) & (d.z < 200)].groupby("time").chl.mean()
    pmax, pmin = pv.max(axis=1), pv.min(axis=1)
    o.update(surf_med=surf.median(), surf_p10=surf.quantile(.1), surf_p90=surf.quantile(.9),
             surf_max=surf.max(), prof_max_med=pmax.median(),
             deep_med=deep.median() if len(deep) >= 5 else np.nan,
             mid_med=mid.median() if len(mid) >= 5 else np.nan,
             neg_frac=float((up.chl < -0.02).mean()), p05_0_200=up.chl.quantile(.05))
    # Flat day: the whole 0-200 m daily profile within 0.05 and max/min < 1.5
    # (the pilot's one-day screen rule, applied to every day).
    flat = ((pmax - pmin) < 0.05) & (pmax / pmin.clip(lower=1e-3) < 1.5) & (pv.count(axis=1) >= 3)
    o["flat_frac"] = float(flat.mean())
    # Vertical fill: share of 10 m bins in 0-100 m (down to each day's
    # deepest bin) with any value that day. Some DFO / C-PROOF sets log
    # chlorophyll only at scattered depths, which leaves windows with few bins.
    zday = up.groupby("time").z.max().clip(upper=90)
    have = up[up.z < 100].groupby("time").z.nunique()
    o["fill_0_100"] = float((have / (zday // 10 + 1)).clip(upper=1).median())
    ref = deep if len(deep) >= 5 else mid
    run = 0
    if len(ref) >= 5:
        sb = (surf < ref.reindex(surf.index) - 0.02).astype(int)
        o["surf_below_deep_frac"] = float(sb.mean())
        # longest stretch of consecutive sampled days with the surface below,
        # in days (works for daily and for every-4th-day grids)
        tt = pd.to_datetime(sb.index)
        step = pd.Series(tt).diff().dt.days.median() if len(tt) > 1 else 1
        grp = (sb != sb.shift()).cumsum()
        for _, s in sb[sb == 1].groupby(grp[sb == 1]):
            ts = pd.to_datetime(s.index)
            run = max(run, int((ts.max() - ts.min()).days + step))
    o["surf_below_deep_maxrun"] = run
    if len(deep) >= 20:
        o["deep_first10"], o["deep_last10"] = deep.iloc[:10].median(), deep.iloc[-10:].median()
    ratio = (surf / surf.shift()).clip(1e-3, 1e3)
    o["surf_jump_frac"] = float((np.maximum(ratio, 1 / ratio) > 5).mean())
    # fail: the sensor or its processing is wrong for this use.
    # check: could be real ocean (upwelling, a mixed shelf); look at the plot.
    fail, check = [], []
    if o["flat_frac"] > 0.3:
        # A flat profile is normal on a mixed shelf (30-40 m Georgia shelf
        # Slocums); sp071 was flat to 150 m.
        (fail if o["max_z"] >= 100 else check).append("flat")
    if np.isfinite(o["deep_med"]) and o["deep_med"] > 0.2:
        fail.append("deep offset > 0.2")
    # sp071: a 0.04-0.06 floor at every depth with a 0.09 "maximum"; its 150-200 m
    # value equals the surface. Oligotrophic but real (Bermuda: surface 0.03,
    # maximum 0.3) passes.
    base = o["mid_med"] if np.isfinite(o["mid_med"]) else o["p05_0_200"]
    if o["prof_max_med"] < 0.15 and o["prof_max_med"] - base < 0.05:
        fail.append("no signal above deep value")
    if o["neg_frac"] > 0.05 and o["p05_0_200"] < -0.05:
        fail.append("negative offset")
    if run >= 14:
        fail.append(f"surface < deep for {run} d")
    if o["prof_max_med"] > 50:
        fail.append("implausibly high")
    if o["surf_jump_frac"] > 0.1:
        check.append("jumpy surface")
    if o["fill_0_100"] < 0.7:
        check.append("sparse depths")
    # Whole-column failure days: often the last day or two of a California
    # Spray record goes high (or ~0) at every depth. Too short to move the
    # medians above; list them so a fetch can drop those days. Only for
    # gliders that dive >= 300 m, on the 150-200 m value: on a mixed shelf a
    # bloom really does raise the whole column.
    bad = pd.Series(False, index=pv.index)
    if o["max_z"] >= 300 and len(mid) >= 5:
        m = mid.reindex(pv.index)
        bad = (m > mid.median() + 0.3) | ((pmax < 0.02) & (pv.count(axis=1) >= 3))
    o["bad_days"] = ",".join(t[:10] for t in pv.index[bad])
    if bad.any():
        check.append(f"{int(bad.sum())} whole-column bad day(s)")
    if "deep_first10" in o and abs(o["deep_last10"] - o["deep_first10"]) > 0.1:
        check.append("deep drift > 0.1")
    o["fail"], o["check"] = "; ".join(fail), "; ".join(check)
    o["verdict"] = "fail" if fail else "check" if check else "ok"
    return o


# --------------------------------------------------------------------------
# Scan: everything, with status against the released table
# --------------------------------------------------------------------------

def _key(dataset_id: str) -> str:
    x = pd.Series([dataset_id]).str.extract(r"^(.*)-(\d{8})T\d{4}(?:-delayed)?$").iloc[0]
    return f"{x[0].lower()}-{x[1]}"


def table_status(dep: pd.DataFrame) -> pd.Series:
    """Per deployment, against the selection and the released glider table:
    'new'; 'in table'; 'in table, grown' (catalog maxTime > 2 d past the
    table's last profile, and the table's last month is not the last fetch
    month); 'in table, delayed copy now covers it' (a possible swap, not new data)."""
    sel = {_key(i): i for i in g.selection()}
    tpath = b.DATA_DIR / "CHLA_glider_profiles.parquet"
    last = {}
    if tpath.exists():
        t = pd.read_parquet(tpath, columns=["dataset_id", "time"])
        last = {_key(k): v for k, v in t.groupby("dataset_id").time.max().items()}
    out = []
    for _, r in dep.iterrows():
        k = r.deployment_key
        if k not in sel:
            out.append("new")
            continue
        s = ["in table"]
        if k in last and r.maxTime > last[k] + pd.Timedelta(days=2) \
                and last[k] < pd.Timestamp(b.END_EXCLUSIVE, tz="UTC") - pd.Timedelta(days=2):
            s.append("grown")
        if r.delayed and not sel[k].endswith("-delayed"):
            s.append("delayed copy now covers it")
        out.append(", ".join(s))
    return pd.Series(out, index=dep.index)


def scan(refresh: bool = True, workers: int = 3, min_days: float = MIN_DAYS) -> pd.DataFrame:
    """Catalog -> deployments -> whole-record fetch (cached) -> stats + status.
    Writes scan/scan_<UTC date>.parquet and returns it."""
    from concurrent.futures import ThreadPoolExecutor
    dep = deployments(catalog(refresh=refresh))
    dep = dep[dep.days_since_start >= min_days].sort_values("days_since_start")
    units = [(r.datasetID, r.chl_var, day) for _, r in dep.iterrows()
             for day in sample_days(r.minTime, r.maxTime)]
    b.log(f"{len(dep)} deployments, {len(units)} sampled days")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda u: _log(day_unit, *u), units))
    st = pd.DataFrame([fullrec_stats(i) for i in dep.datasetID], index=dep.index)
    dep = pd.concat([dep, st], axis=1)
    dep["region"] = [region(a, o) for a, o in zip(dep.lat, dep.lon)]
    dep["status"] = table_status(dep)
    out = SCAN_DIR / f"scan_{pd.Timestamp.now('UTC'):%Y-%m-%d}.parquet"
    b.atomic_write_parquet(dep.reset_index(drop=True), out)
    return dep


def _log(fn, *a):
    try:
        msg = fn(*a)
    except Exception as e:  # one deployment must not stop the scan
        msg = f"FAILED {a[0]}: {type(e).__name__}: {str(e)[:200]}"
    if msg.startswith("FAILED"):  # thousands of units: log only failures
        b.log(msg)
    return msg


CANDIDATES_PATH = g.SELECTION_PATH.with_name("chla_z_glider_candidates.csv")
CANDIDATE_COLS = [
    "deployment_key", "datasetID", "alt_ids", "status", "verdict", "fail", "check", "region",
    "institution", "platform_type", "chl_var", "chl_units", "qc_var", "has_precise_lat",
    "lat", "lon", "minTime", "maxTime", "days_since_start", "days_in_months",
    "n_days", "max_z", "fill_0_100", "surf_med", "prof_max_med", "mid_med", "deep_med", "flat_frac",
    "surf_below_deep_maxrun", "bad_days", "scanned"]


def restat(path=None) -> pd.DataFrame:
    """Recompute stats and status from the cached sampled days only (no
    network): for changing thresholds after a scan."""
    path = path or max(SCAN_DIR.glob("scan_*.parquet"))
    d = pd.read_parquet(path)
    keep = [c for c in d.columns if c in set(catalog().columns) | {"alt_ids", "region"}]
    d = d[keep]
    st = pd.DataFrame([fullrec_stats(i) for i in d.datasetID], index=d.index)
    d = pd.concat([d, st], axis=1)
    d["status"] = table_status(d)
    b.atomic_write_parquet(d, path)
    return d


def write_candidates(d: pd.DataFrame) -> pd.DataFrame:
    """The committed scan result: every deployment with >= MIN_DAYS since START,
    one row each. `days_in_months` is the part inside the CHLA-Z fetch months
    (chla_z_build.MONTHS); data after them need MONTHS extended first."""
    d = d.copy()
    t0 = d.minTime.clip(lower=pd.Timestamp(b.START, tz="UTC"))
    t1 = d.maxTime.clip(upper=pd.Timestamp(b.END_EXCLUSIVE, tz="UTC"))
    d["days_in_months"] = ((t1 - t0).dt.total_seconds() / 86400).clip(lower=0)
    d["scanned"] = f"{pd.Timestamp.now('UTC'):%Y-%m-%d}"
    order = {"ok": 0, "check": 1, "fail": 2}
    d = d.assign(_o=d.verdict.map(order).fillna(3)).sort_values(["_o", "region", "minTime"])
    d = d[[c for c in CANDIDATE_COLS if c in d.columns]]
    for c in ("lat", "lon", "days_since_start", "days_in_months", "surf_med", "prof_max_med",
              "mid_med", "deep_med", "flat_frac", "fill_0_100"):
        if c in d:
            d[c] = d[c].astype(float).round(3)
    d.to_csv(CANDIDATES_PATH, index=False)
    return d

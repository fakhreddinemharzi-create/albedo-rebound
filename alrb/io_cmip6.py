"""Monthly CMIP6 fields at the site grid cells, from the Pangeo CMIP6 Zarr
stores mirrored on AWS (``s3://cmip6-pds``), for the delta-change
projection of the observed meteorology."""
from __future__ import annotations

import json
import time
from functools import lru_cache

import numcodecs
import numpy as np
import pandas as pd
import requests

CATALOG = "https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.csv"
_S = requests.Session()


def _get(url, binary=True, tries=6):
    for k in range(tries):
        try:
            r = _S.get(url, timeout=600)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content if binary else r.text
        except Exception:
            time.sleep(2 ** k)
    raise RuntimeError(url)


def https(zstore: str) -> str:
    return zstore.replace("s3://cmip6-pds/", "https://cmip6-pds.s3.amazonaws.com/").rstrip("/") + "/"


@lru_cache(maxsize=None)
def meta(zstore):
    return json.loads(_get(https(zstore) + ".zmetadata", binary=False))["metadata"]


def _read_full(zstore, name, chunk_index):
    m = meta(zstore)
    za = m[f"{name}/.zarray"]
    sep = za.get("dimension_separator", ".")
    key = sep.join(str(c) for c in chunk_index) if chunk_index else "0"
    buf = _get(https(zstore) + f"{name}/{key}")
    if buf is None:
        return None
    raw = numcodecs.get_codec(za["compressor"]).decode(buf) if za["compressor"] else buf
    for flt in reversed(za.get("filters") or []):
        raw = numcodecs.get_codec(flt).decode(raw)
    return np.frombuffer(raw, dtype=np.dtype(za["dtype"])).reshape(za["chunks"], order=za.get("order", "C"))


def _coord(zstore, name):
    m = meta(zstore)
    za = m[f"{name}/.zarray"]
    n = za["shape"][0]
    ck = za["chunks"][0]
    parts = [_read_full(zstore, name, (c,) + (0,) * (len(za["shape"]) - 1)) for c in range((n + ck - 1) // ck)]
    return np.concatenate(parts)[:n]


def decode_time(zstore):
    m = meta(zstore)
    at = m["time/.zattrs"]
    units, cal = at["units"], at.get("calendar", "standard")
    t = _coord(zstore, "time").astype(float)
    step, _, ref = units.partition(" since ")
    ref = pd.Timestamp(ref.strip().split(" ")[0])
    scale = {"days": 1.0, "hours": 1 / 24.0}[step.strip()]
    days = t * scale
    if cal in ("noleap", "365_day"):
        # offsets in a 365-day calendar, counted from 1 January of the reference year
        cum = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        tot = days + cum[ref.month - 1] + (ref.day - 1) + ref.hour / 24.0
        yrs = ref.year + np.floor(tot / 365.0)
        doy = tot - (yrs - ref.year) * 365.0
        mon = np.searchsorted(cum, doy, side="right")
        return pd.PeriodIndex.from_fields(year=yrs.astype(int), month=np.clip(mon, 1, 12), freq="M")
    if cal == "360_day":
        tot = days + (ref.month - 1) * 30 + (ref.day - 1) + ref.hour / 24.0
        yrs = ref.year + np.floor(tot / 360.0)
        mon = np.floor((tot - (yrs - ref.year) * 360.0) / 30.0) + 1
        return pd.PeriodIndex.from_fields(year=yrs.astype(int), month=np.clip(mon, 1, 12).astype(int), freq="M")
    # second resolution: some ScenarioMIP stores extend to 2300, beyond the
    # nanosecond range of pandas timestamps
    t64 = np.datetime64(ref.strftime("%Y-%m-%dT%H:%M:%S"), "s") + np.round(days * 86400.0).astype("timedelta64[s]")
    yrs = t64.astype("datetime64[Y]").astype(int) + 1970
    mon = t64.astype("datetime64[M]").astype(int) % 12 + 1
    return pd.PeriodIndex.from_fields(year=yrs, month=mon, freq="M")


def points(zstore, var, lats, lons, period=None):
    """Monthly series (DataFrame, one column per site) at the nearest grid cells."""
    m = meta(zstore)
    za = m[f"{var}/.zarray"]
    lat = _coord(zstore, "lat") if "lat/.zarray" in m else _coord(zstore, "latitude")
    lon = _coord(zstore, "lon") if "lon/.zarray" in m else _coord(zstore, "longitude")
    if lat.ndim != 1:
        raise ValueError("curvilinear grid not supported")
    per = decode_time(zstore)
    nt = za["shape"][0]
    ct = za["chunks"][0]
    if za["chunks"][1:] != za["shape"][1:]:
        raise ValueError(f"spatially chunked store {za['chunks']}")
    ii = [int(np.argmin(np.abs(lat - a))) for a in lats]
    jj = [int(np.argmin(np.abs(((lon - (b % 360)) + 180) % 360 - 180))) for b in lons]
    sel = np.ones(nt, bool) if period is None else (per.year >= period[0]) & (per.year <= period[1])
    out = np.full((nt, len(lats)), np.nan)
    for c in range((nt + ct - 1) // ct):
        a, b = c * ct, min(nt, (c + 1) * ct)
        if not sel[a:b].any():
            continue
        blk = _read_full(zstore, var, (c, 0, 0))
        if blk is None:
            continue
        for k, (i, j) in enumerate(zip(ii, jj)):
            out[a:b, k] = blk[:b - a, i, j]
    df = pd.DataFrame(out[sel], index=per[sel])
    fill = za.get("fill_value")
    if isinstance(fill, (int, float)) and np.isfinite(fill):
        df = df.where(df != fill)
    df = df.where(np.abs(df) < 1e19)
    return df

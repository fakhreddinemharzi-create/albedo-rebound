"""Point time series from the NASA POWER analysis-ready Zarr stores on AWS
(``s3://nasa-power``): MERRA-2 meteorology and CERES SYN1deg radiation.

The stores are Zarr v2 with consolidated metadata and time-series-oriented
chunks (all times x a few grid cells), so a site series costs one or two
chunk reads per variable.  Chunks are fetched and decoded directly, which
avoids any dependence on the Zarr library version.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache

import numcodecs
import numpy as np
import pandas as pd
import requests

BASE = "https://nasa-power.s3.amazonaws.com"
STORES = {
    ("merra2", "hourly"): "merra2/temporal/power_merra2_hourly_temporal_utc.zarr",
    ("merra2", "daily"): "merra2/temporal/power_merra2_daily_temporal_utc.zarr",
    ("merra2", "monthly"): "merra2/temporal/power_merra2_monthly_temporal_utc.zarr",
    ("syn1deg", "hourly"): "syn1deg/temporal/power_syn1deg_hourly_temporal_utc.zarr",
    ("syn1deg", "daily"): "syn1deg/temporal/power_syn1deg_daily_temporal_utc.zarr",
    ("syn1deg", "monthly"): "syn1deg/temporal/power_syn1deg_monthly_temporal_utc.zarr",
}
_S = requests.Session()


def _get(url, binary=True, tries=6):
    for k in range(tries):
        try:
            r = _S.get(url, timeout=300)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.content if binary else r.text
        except Exception:
            time.sleep(2 ** k)
    raise RuntimeError(url)


@lru_cache(maxsize=None)
def meta(product, freq):
    return json.loads(_get(f"{BASE}/{STORES[(product, freq)]}/.zmetadata", binary=False))["metadata"]


@lru_cache(maxsize=None)
def _coord(product, freq, name):
    m = meta(product, freq)
    za = m[f"{name}/.zarray"]
    sep = za.get("dimension_separator", ".")
    n = za["shape"][0]
    ck = za["chunks"][0]
    parts = []
    for c in range((n + ck - 1) // ck):
        parts.append(_decode(_get(f"{BASE}/{STORES[(product, freq)]}/{name}/{c}"), za, (min(ck, n - c * ck),)))
    return np.concatenate(parts)[:n]


def _decode(buf, za, shape_valid):
    codec = numcodecs.get_codec(za["compressor"]) if za["compressor"] else None
    raw = codec.decode(buf) if codec else buf
    for flt in reversed(za.get("filters") or []):
        raw = numcodecs.get_codec(flt).decode(raw)
    arr = np.frombuffer(raw, dtype=np.dtype(za["dtype"]))
    full = arr.reshape(za["chunks"], order=za.get("order", "C"))
    sl = tuple(slice(0, s) for s in shape_valid)
    return full[sl]


def times(product, freq):
    m = meta(product, freq)
    units = m["time/.zattrs"]["units"]          # e.g. "hours since 2000-12-31 00:00:00"
    step, _, ref = units.partition(" since ")
    t = _coord(product, freq, "time")
    unit = {"hours": "h", "days": "D"}[step]
    return pd.Timestamp(ref, tz="UTC") + pd.to_timedelta(t, unit=unit)


def series(product, freq, var, lat, lon, t0=None, t1=None):
    """Nearest-grid-cell time series of ``var`` at (lat, lon)."""
    m = meta(product, freq)
    za = m[f"{var}/.zarray"]
    sep = za.get("dimension_separator", ".")
    lats = _coord(product, freq, "lat")
    lons = _coord(product, freq, "lon")
    i = int(np.argmin(np.abs(lats - lat)))
    j = int(np.argmin(np.abs(((lons - lon) + 180) % 360 - 180)))
    nt = za["shape"][0]
    ct, cy, cx = za["chunks"]
    tt = times(product, freq)
    lo = 0 if t0 is None else int(np.searchsorted(tt, pd.Timestamp(t0, tz="UTC")))
    hi = nt if t1 is None else int(np.searchsorted(tt, pd.Timestamp(t1, tz="UTC"), side="right"))
    out = np.full(hi - lo, np.nan, dtype=float)
    for c in range(lo // ct, (hi - 1) // ct + 1):
        key = sep.join([str(c), str(i // cy), str(j // cx)])
        buf = _get(f"{BASE}/{STORES[(product, freq)]}/{var}/{key}")
        if buf is None:
            continue
        valid_t = min(ct, nt - c * ct)
        blk = _decode(buf, za, (valid_t, cy, cx))
        v = blk[:, i % cy, j % cx].astype(float)
        a, b = c * ct, c * ct + valid_t
        s0, s1 = max(a, lo), min(b, hi)
        out[s0 - lo:s1 - lo] = v[s0 - a:s1 - a]
    fill = za.get("fill_value")
    if fill is not None and not (isinstance(fill, float) and np.isnan(fill)) and not isinstance(fill, str):
        out[out == fill] = np.nan
    out[out < -900] = np.nan
    return pd.Series(out, index=tt[lo:hi], name=var), (float(lats[i]), float(lons[j]))

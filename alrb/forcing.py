"""Hourly forcing series for one site: aggregation of the USCRN records,
quality control, gap filling and derived radiative quantities.

Time convention: the value stamped H (UTC) is the mean (or total) over the
hour ending at H, which is the USCRN convention and was confirmed for the
ISD solar field by the clear-sky timing test (Sect. 3.1 of the manuscript).
POWER hourly means are labelled by the hour they begin and are shifted by
one hour on import.
"""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import requests

from . import atmos, solar
from .config import RAW, YEARS

GHCN = "https://noaa-ghcn-pds.s3.amazonaws.com"


def load_isd(key):
    five, hour = [], []
    for y in YEARS + [YEARS[-1] + 1]:
        p5 = RAW / "isd" / "parquet" / f"{key}_{y}_5min.parquet"
        ph = RAW / "isd" / "parquet" / f"{key}_{y}_hourly.parquet"
        if p5.exists():
            five.append(pd.read_parquet(p5))
            hour.append(pd.read_parquet(ph))
    f = pd.concat(five)
    h = pd.concat(hour)
    f = f[~f.index.duplicated()].sort_index()
    h = h[~h.index.duplicated()].sort_index()
    return f, h


def ghcn_station(sid):
    for k in range(6):
        try:
            r = requests.get(f"{GHCN}/csv/by_station/{sid}.csv", timeout=300)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content), dtype={"ID": str, "DATA_VALUE": float},
                             usecols=["ID", "DATE", "ELEMENT", "DATA_VALUE", "Q_FLAG"])
            df = df[df.Q_FLAG.isna()]
            df["DATE"] = pd.to_datetime(df.DATE.astype(str), format="%Y%m%d")
            return df.pivot_table(index="DATE", columns="ELEMENT", values="DATA_VALUE")
        except Exception:
            time.sleep(2 ** k)
    raise RuntimeError(sid)


def _fill_gaps(x: pd.Series, max_interp=3, clim_key=None):
    """Linear interpolation for gaps <= max_interp hours, then a month-hour
    climatology of the same series for longer gaps.  Returns (filled, flag)."""
    flag = x.isna().astype(np.int8)
    y = x.interpolate(limit=max_interp, limit_area="inside")
    if y.isna().any():
        k = clim_key if clim_key is not None else [x.index.month, x.index.hour]
        clim = x.groupby(k).transform("mean")
        y = y.fillna(clim)
        y = y.ffill().bfill()
    flag[(flag == 1) & x.isna() & y.notna()] = 1
    return y, flag


def build(site, power_m2: pd.DataFrame, power_ceres: pd.DataFrame, ghcn: pd.DataFrame | None):
    f, h = load_isd(site.key)
    t0 = pd.Timestamp(f"{YEARS[0]}-01-01 01:00", tz="UTC")
    t1 = pd.Timestamp(f"{YEARS[-1] + 1}-01-01 00:00", tz="UTC")
    idx = pd.date_range(t0, t1, freq="h")

    # ---- five-minute fields -> hour ending at H
    cnt = f.ta.resample("h", label="right", closed="right").count()
    ta5 = f.ta.resample("h", label="right", closed="right").mean().where(cnt >= 9)
    rh5 = f.rh.resample("h", label="right", closed="right").mean()
    rhc = f.rh.resample("h", label="right", closed="right").count()
    rh5 = rh5.where(rhc >= 9)
    prc = f.pr.resample("h", label="right", closed="right").count()
    pr5 = f.pr.resample("h", label="right", closed="right").sum(min_count=1).where(prc >= 9)
    ws = f.ws.dropna()
    ws = ws[ws.index.minute == 0]
    df = pd.DataFrame(index=idx)
    df["ta"] = ta5.reindex(idx)
    df["ta"] = df["ta"].fillna(h.ta_h.reindex(idx))
    df["rh"] = rh5.reindex(idx)
    df["ws"] = ws.reindex(idx)
    # hourly precipitation: the network's hourly total (AA1) reproduces the
    # GHCN-D daily totals of the same gauge to within 0.5 %; the five-minute
    # sums (CB1) do not, and serve only as a fallback
    df["pr"] = h.pr_h.reindex(idx).fillna(pr5.reindex(idx))
    df["sw"] = h.sw.reindex(idx)

    # ---- geometry
    g0, muw, daylit = solar.hourly_toa(idx, site.lat, site.lon, ending=True)
    df["g0"] = g0
    df["mu"] = muw
    df["daylit"] = daylit

    # ---- quality control
    qc = {}
    sw = df.sw.copy()
    night = df.g0 < 1.0
    sw[night & (sw.abs() < 25)] = 0.0
    bad = (sw < -5) | (sw > np.minimum(1.1 * df.g0 + 30.0, 1250.0))
    # stuck sensor: >= 4 identical non-zero daytime hours
    run = sw.diff().eq(0) & (sw > 0)
    stuck = run.rolling(3).sum().eq(3)
    for k in range(3):
        stuck |= stuck.shift(-1, fill_value=False)
    qc["sw_rejected"] = int((bad | stuck).sum())
    sw[bad | stuck] = np.nan
    df["sw"] = sw.clip(lower=0)
    ta = df.ta.copy()
    jump = ta.diff().abs() > 10
    ta[(ta < -50) | (ta > 55) | jump] = np.nan
    df["ta"] = ta
    df.loc[(df.rh < 1) | (df.rh > 100.5), "rh"] = np.nan
    df["rh"] = df.rh.clip(upper=100)
    df.loc[(df.ws < 0) | (df.ws > 40), "ws"] = np.nan
    df.loc[(df.pr < 0) | (df.pr > 150), "pr"] = np.nan

    # ---- gap filling
    miss = {c: float(df[c].isna().mean()) for c in ["sw", "ta", "rh", "ws", "pr"]}
    kt = (df.sw / df.g0).where(df.g0 > 20)
    kt_f, _ = _fill_gaps(kt.clip(0, 1.1), 3)
    sw_f = df.sw.copy()
    need = sw_f.isna()
    sw_f[need] = (kt_f * df.g0)[need]
    sw_f[df.g0 <= 20] = sw_f[df.g0 <= 20].fillna(0.0)
    df["sw"] = sw_f.fillna(0.0)
    for c in ["ta", "rh", "ws"]:
        df[c], _ = _fill_gaps(df[c], 3)
    df["pr"] = df.pr.fillna(0.0)

    # ---- derived: vapour pressure, pressure, clear sky, cloud, long wave
    df["ea"] = atmos.vapour_pressure(df.ta, df.rh)
    df["p"] = atmos.pressure_from_elevation(site.elev)
    df["cs"] = clear_sky(idx, site)
    ratio = (df.sw / df.cs).where((df.cs > 50) & (df.mu > np.sin(np.radians(12))))
    cloud = (1.0 - ratio.clip(0, 1))
    cloud = cloud.interpolate(limit_area=None).ffill().bfill()
    df["cloud"] = cloud.clip(0, 1)
    df["lw"] = atmos.longwave_down(df.ta, df.ea, df.cloud)

    # ---- POWER (hour-beginning means -> hour-ending stamps)
    for src, cols in ((power_m2, {"TQV": "tqv", "TO3": "o3", "PRECSNO": "snowfall", "SNODP": "snodp",
                                  "FRSNO": "frsno", "PS": "ps_m2"}),
                      (power_ceres, {"ALLSKY_SFC_LW_DWN": "lw_ceres", "AOD_55": "aod", "ALLSKY_SRF_ALB": "alb_ceres",
                                     "CLOUD_AMT": "cloud_ceres", "ALLSKY_SFC_SW_DWN": "sw_ceres",
                                     "TOA_SW_DWN": "g0_ceres"})):
        s2 = src.copy()
        s2.index = s2.index + pd.Timedelta(hours=1)
        for a, b in cols.items():
            if a in s2:
                v = s2[a].reindex(idx)
                df[b] = v.interpolate(limit=6).ffill().bfill() if b not in ("sw_ceres", "g0_ceres", "lw_ceres") else v
    df["lw_ceres_f"] = df["lw_ceres"].interpolate(limit=3).fillna(df["lw"])
    # ---- daily precipitation cross-check against GHCN-D (same station)
    qc["missing"] = miss
    if ghcn is not None and "PRCP" in ghcn:
        loc = df.pr.copy()
        loc.index = loc.index + pd.Timedelta(hours=site.tz)       # local standard time
        d = loc.resample("D").sum()
        g = ghcn.PRCP.reindex(d.index.tz_localize(None)) / 10.0
        both = pd.concat([d.reset_index(drop=True), g.reset_index(drop=True)], axis=1, keys=["crn", "ghcn"]).dropna()
        qc["prcp_crn_sum"] = float(both.crn.sum())
        qc["prcp_ghcn_sum"] = float(both.ghcn.sum())
    return df.astype("float32"), qc


def clear_sky(idx, site, nsub=12):
    """Hour-ending mean clear-sky global irradiance (Ineichen-Perez, with the
    Linke turbidity climatology distributed with pvlib)."""
    import pvlib
    loc = pvlib.location.Location(site.lat, site.lon, altitude=site.elev)
    frac = (np.arange(nsub) + 0.5) / nsub - 1.0
    base = idx.values.astype("datetime64[s]").astype(np.int64)
    ts = pd.to_datetime((base[:, None] + frac[None, :] * 3600.0).ravel(), unit="s", utc=True)
    cs = loc.get_clearsky(ts, model="ineichen")
    return cs["ghi"].to_numpy().reshape(len(idx), nsub).mean(axis=1)

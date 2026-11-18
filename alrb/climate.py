"""Delta-change projection of the observed hourly forcing over the analysis
period (1 May 2025 - 30 April 2075).

Monthly change factors per model and scenario are taken relative to the
model's own 2011-2031 window, which is centred on the 2017-2024 record, so
that the warming already contained in the observations is not counted
twice.  Future windows are 21-year running means centred on each analysis
year.  Temperature and relative humidity change additively; shortwave
irradiance and precipitation multiplicatively (irradiance capped by the
clear-sky envelope).  Incoming long-wave radiation is scaled with the
fourth power of the absolute air temperature at constant emissivity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW, SITE, YEARS
from . import atmos

MODELS = ["ACCESS-CM2", "CanESM5", "EC-Earth3", "GFDL-ESM4", "INM-CM5-0", "IPSL-CM6A-LR",
          "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
BASE = (2011, 2031)


def _series(model, exp, var, key):
    f = RAW / "cmip6" / f"{model}_{exp}_{var}.parquet"
    if not f.exists():
        return None
    s = pd.read_parquet(f)[key]
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s


def monthly_deltas(model, ssp, key, years=range(2025, 2076)):
    """DataFrame indexed by (year, month) with dT, rsds ratio, pr ratio, dRH."""
    out = {}
    for var in ["tas", "rsds", "pr", "hurs"]:
        h = _series(model, "historical", var, key)
        f = _series(model, ssp, var, key)
        if h is None or f is None:
            return None
        s = pd.concat([h, f])
        s = s[~s.index.duplicated()]
        clim = lambda a, b: s[(s.index.year >= a) & (s.index.year <= b)].groupby(s.index.month[(s.index.year >= a) & (s.index.year <= b)]).mean()
        base = clim(*BASE)
        rows = {}
        for y in years:
            fut = clim(y - 10, y + 10)
            if var in ("tas", "hurs"):
                rows[y] = (fut - base).to_numpy()
            else:
                rows[y] = (fut / base.replace(0, np.nan)).fillna(1.0).clip(0.3, 3.0).to_numpy()
        out[var] = pd.DataFrame(rows).T
        out[var].columns = range(1, 13)
    return out


def projected_forcing(df_obs: pd.DataFrame, key: str, deltas, start="2025-05-01 01:00", nyears=50):
    """Hourly forcing for the analysis period: analysis year k re-uses the
    observed hours of record year (k mod 8), counted from 1 May, with the
    deltas of calendar year 2025 + k applied month by month."""
    s = SITE[key]
    rec = df_obs.loc["2017-05-01 01:00":"2025-01-01 00:00"]
    rec_years = []
    for k in range(8):
        a = pd.Timestamp(f"{2017 + k}-05-01 01:00", tz="UTC")
        b = pd.Timestamp(f"{2018 + k}-05-01 00:00", tz="UTC")
        chunk = df_obs.loc[a:b]
        rec_years.append(chunk)
    parts = []
    for k in range(nyears):
        src = rec_years[k % 8].copy()
        # the last record year (May 2024 - Apr 2025) lacks Jan-Apr 2025: fill from the first year
        if len(src) < 8700:
            need = 8760 - len(src)
            src = pd.concat([src, rec_years[0].iloc[len(src):len(src) + need]])
        yr = 2025 + k
        if deltas is not None:
            mon = ((src.index + pd.Timedelta(hours=s.tz)).month).to_numpy()
            dT = deltas["tas"].loc[yr].reindex(range(1, 13)).to_numpy()[mon - 1]
            rsr = deltas["rsds"].loc[yr].to_numpy()[mon - 1]
            prr = deltas["pr"].loc[yr].to_numpy()[mon - 1]
            drh = deltas["hurs"].loc[yr].to_numpy()[mon - 1]
            src = src.copy()
            t_old = src["ta"].to_numpy()
            src["ta"] = t_old + dT
            src["sw"] = np.minimum(src["sw"].to_numpy() * rsr, np.maximum(src["cs"].to_numpy(), src["sw"].to_numpy()))
            src["pr"] = src["pr"].to_numpy() * prr
            src["rh"] = np.clip(src["rh"].to_numpy() + drh, 5, 100)
            src["ea"] = atmos.vapour_pressure(src["ta"].to_numpy(), src["rh"].to_numpy())
            scale = ((src["ta"].to_numpy() + 273.15) / (t_old + 273.15)) ** 4
            for c in ("lw", "lw_ceres_f"):
                src[c] = src[c].to_numpy() * scale
        idx = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.DateOffset(years=k), periods=len(src), freq="h")
        src.index = idx
        parts.append(src.iloc[:8760])
    out = pd.concat(parts)
    return out

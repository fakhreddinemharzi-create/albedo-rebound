"""Annual and winter statistics of the stored sweep runs (complete calendar
years and complete winters only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW, SITE, YEARS

WINTER_SHIFT = pd.Timedelta(days=213)      # winters run 1 August - 31 July


def load_run(key, alb, tag="ref"):
    z = np.load(RAW / "sweep" / tag / f"{key}_a{alb:.2f}.npz")
    idx = pd.date_range(pd.Timestamp(str(z["t0"])), periods=int(z["n"]), freq="h")
    return z, idx


def local_index(idx, key):
    return idx + pd.Timedelta(hours=SITE[key].tz)


def full_years(s: pd.Series):
    return s[(s.index.year >= YEARS[0]) & (s.index.year <= YEARS[-1])]


def winter_id(index):
    return (index - WINTER_SHIFT).year


def complete_winters(s: pd.Series):
    w = winter_id(s.index)
    return s[(w >= YEARS[0]) & (w <= YEARS[-1] - 1)], w[(w >= YEARS[0]) & (w <= YEARS[-1] - 1)]


def summarise(key, alb, tag="ref"):
    z, idx = load_run(key, alb, tag)
    li = local_index(idx, key)
    ts = pd.Series(z["ts"], index=li)
    fr = pd.Series(z["frost"], index=li)
    yrs = full_years(ts)
    ann_max = yrs.groupby(yrs.index.year).max()
    ann_min = yrs.groupby(yrs.index.year).min()
    fw, w = complete_winters(fr)
    wmax = fw.groupby(w).max()
    daily_frozen = (fw.resample("D").max() > 0)
    dfw = daily_frozen.groupby(winter_id(daily_frozen.index)).sum()
    dfw = dfw[(dfw.index >= YEARS[0]) & (dfw.index <= YEARS[-1] - 1)]
    depths = z["depths"]
    Tm = {f"{d:.3f}": float(np.mean(z["T"][:, i])) for i, d in enumerate(depths)}
    return dict(ts_mean=float(yrs.mean()), ts_max=float(ann_max.mean()), ts_min=float(ann_min.mean()),
                frost_max=float(wmax.mean()) if len(wmax) else 0.0,
                frost_max_extreme=float(wmax.max()) if len(wmax) else 0.0,
                frozen_days=float(dfw.mean()) if len(dfw) else 0.0, T_mean=Tm)

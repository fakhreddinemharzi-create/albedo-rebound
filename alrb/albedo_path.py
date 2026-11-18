"""Albedo of a decaying, periodically renewed treatment on the solver's
sub-step clock."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sawtooth(index_hourly_utc: pd.DatetimeIndex, nsub: int, tz: int, first_application: str,
             alpha0: float, alphaf: float, tau: float, Tr: float, alpha_inf: float | None = None):
    """Albedo at every sub-step.  Applications on the local date
    ``first_application`` and every Tr years after; untreated before."""
    if alpha_inf is None:
        alpha_inf = alpha0
    base = index_hourly_utc.values.astype("datetime64[s]").astype(np.int64)
    frac = (np.arange(nsub) + 0.5) / nsub - 1.0
    ts = (base[:, None] + frac[None, :] * 3600.0).ravel() + tz * 3600.0
    t0 = pd.Timestamp(first_application).value // 10 ** 9
    yrs = (ts - t0) / (365.25 * 86400.0)
    a = np.full(ts.shape, alpha0, float)
    on = yrs >= 0
    ph = np.mod(yrs[on], Tr)
    a[on] = alpha_inf + (alphaf - alpha_inf) * np.exp(-ph / tau)
    return a

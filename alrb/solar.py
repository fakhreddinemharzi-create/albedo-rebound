"""Solar geometry and extraterrestrial irradiance.

Declination, equation of time and orbital eccentricity follow the Fourier
series of Spencer (1971) as given by Iqbal (1983, Ch. 1).  Hourly means of
the extraterrestrial irradiance on a horizontal plane are integrated
numerically over the averaging interval (one-minute steps), so the hourly
clearness index is defined consistently with the interval the pyranometer
averages over, including the partial hours at sunrise and sunset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import S0


def _gamma(doy, hour_utc):
    return 2.0 * np.pi * (doy - 1 + (hour_utc - 12.0) / 24.0) / 365.0


def spencer(doy, hour_utc=12.0):
    """Declination (rad), equation of time (min), eccentricity factor E0."""
    g = _gamma(np.asarray(doy, float), np.asarray(hour_utc, float))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g)
                    - 0.040849 * np.sin(2 * g))
    e0 = (1.000110 + 0.034221 * np.cos(g) + 0.001280 * np.sin(g) + 0.000719 * np.cos(2 * g)
          + 0.000077 * np.sin(2 * g))
    return dec, eot, e0


def cos_zenith(times_utc: pd.DatetimeIndex, lat, lon):
    """Cosine of the solar zenith angle at instants ``times_utc``."""
    t = pd.DatetimeIndex(times_utc)
    doy = t.dayofyear.to_numpy()
    hr = (t.hour + t.minute / 60.0 + t.second / 3600.0).to_numpy()
    dec, eot, e0 = spencer(doy, hr)
    tst = hr * 60.0 + eot + 4.0 * lon            # true solar time (min)
    ha = np.radians(tst / 4.0 - 180.0)
    phi = np.radians(lat)
    mu = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha)
    return mu, e0


def hourly_toa(index_utc: pd.DatetimeIndex, lat, lon, ending=True, nsub=60):
    """Mean extraterrestrial irradiance on a horizontal plane (W m-2) and
    the irradiance-weighted mean cosine of the zenith angle over each hour.

    ``ending=True``: stamp H labels the interval (H-1 h, H]; otherwise [H, H+1 h).
    """
    idx = pd.DatetimeIndex(index_utc)
    off = -1.0 if ending else 0.0
    # sub-step mid-points
    frac = (np.arange(nsub) + 0.5) / nsub
    base = idx.values.astype("datetime64[s]").astype(np.int64)
    ts = base[:, None] + ((off + frac) * 3600.0)[None, :]
    tt = pd.to_datetime(ts.ravel(), unit="s", utc=True)
    mu, e0 = cos_zenith(tt, lat, lon)
    mu = mu.reshape(len(idx), nsub)
    e0 = e0.reshape(len(idx), nsub)
    g0 = S0 * e0 * np.clip(mu, 0.0, None)
    g0m = g0.mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        muw = np.where(g0m > 0, (g0 * np.clip(mu, 0, None)).sum(1) / np.maximum(g0.sum(1), 1e-12), 0.0)
    daylit = (mu > 0).mean(axis=1)
    return g0m, muw, daylit


def air_mass(mu):
    """Relative optical air mass (Kasten and Young, 1989), for mu = cos(zenith)."""
    mu = np.clip(mu, 1e-3, 1.0)
    zen = np.degrees(np.arccos(mu))
    return 1.0 / (mu + 0.50572 * (96.07995 - zen) ** -1.6364)


def erbs_diffuse_fraction(kt):
    """Hourly diffuse fraction of global irradiance (Erbs et al., 1982)."""
    kt = np.asarray(kt, float)
    k = np.where(kt <= 0.22, 1.0 - 0.09 * kt,
                 np.where(kt <= 0.80,
                          0.9511 - 0.1604 * kt + 4.388 * kt ** 2 - 16.638 * kt ** 3 + 12.336 * kt ** 4,
                          0.165))
    return np.clip(k, 0.0, 1.0)

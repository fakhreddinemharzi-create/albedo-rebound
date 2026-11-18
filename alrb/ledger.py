"""Fifty-year greenhouse-gas ledger of a reflective treatment, per square
metre, under four metric conventions.

    G_net = G_rf + G_co + G_st            (G_en examined separately)

* G_rf  radiative credit of the delivered albedo increase: daily kernel of
        the site (hourly APRP kernel, zero while the plowed street is snow
        covered) x albedo increment of the decaying, periodically renewed
        treatment, times an urban-form factor (share of the patch forcing
        that escapes the street) and the efficacy of albedo forcing;
* G_co  embodied burden of the applications (pulses at the renewal dates;
        the last coat is allocated in proportion to its service inside the
        analysis period);
* G_st  change of structural renewals, as a flow g_st (1/L_eff - 1/L_0),
        with L_eff the harmonic time-average of the life under the albedo
        path (Miner-type damage accumulates at the rate of the current state).

The analysis starts on 1 May (application date) of the first observed year;
the eight observed years are cycled over the 50-yr period (stationary
climate) unless a projected record is supplied.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from numba import njit

from . import metrics, kernel, driver
from .config import A_EARTH, RAW, SITE, H_ANALYSIS

DAYS_PER_YEAR = 365.25


@dataclass
class Params:
    alpha0: float = 0.14
    alphaf: float = 0.45
    alpha_inf: float = 0.14
    tau: float = 0.91
    Tr: float = 3.0
    g_co: float = 2.70
    g_st: float = 9.0
    kmult: float = 1.0        # kernel multiplier (structural uncertainty of the kernel)
    efficacy: float = 1.0
    snow_days: float = 1.0
    psi: float = 1.0          # urban-form factor (open surface = 1)
    hmult: float = 1.0
    H: float = H_ANALYSIS


def daily_kernel(key, snow_days=1.0, kind="aprp"):
    """Daily mean kernel (W m-2 per unit albedo, <= 0) over the observed
    record, local days, with the snow mask of the plowed street.

    The record is cut to seven whole years starting on 1 May 2017 (the
    date of the first application), so that cycling it over the analysis
    period keeps the seasonal phase of the kernel aligned with the renewal
    dates."""
    df = driver.load_forcing(key)
    K, Ks, Kc, kt, mu = kernel.hourly_kernel(df)
    Kh = {"aprp": K, "simple": Ks, "clear": Kc}[kind]
    mask = driver.snow_mask_pavement(df, key, days_after=snow_days) if snow_days is not None else 0.0
    Kh = Kh * (1.0 - mask)
    loc = pd.Series(Kh, index=df.index + pd.Timedelta(hours=SITE[key].tz))
    d = loc.resample("D").mean()
    d = d[(d.index >= "2017-05-01") & (d.index < "2024-05-01")]
    return d


def daily_kernel_bank(key, days=(0.0, 1.0, 2.0, 3.0), kind="aprp"):
    ks = [daily_kernel(key, d, kind) for d in days]
    idx = ks[0].index
    return np.array(days), np.vstack([k.reindex(idx).fillna(0.0).to_numpy() for k in ks]), idx


@njit(cache=True)
def _annual_forcing(Kd, ndays_rec, alpha0, alphaf, alpha_inf, tau, Tr, H, days_per_year):
    """Annual-mean local forcing (W m-2, per unit area of treated surface) for
    years 0..H-1 of the analysis period (daily resolution)."""
    ny = int(H + 0.5)
    out = np.zeros(ny)
    cnt = np.zeros(ny)
    nd = int(H * days_per_year)
    trd = Tr * days_per_year
    taud = tau * days_per_year
    for d in range(nd):
        ph = d % trd
        da = (alpha_inf - alpha0) + (alphaf - alpha_inf) * np.exp(-ph / taud)
        y = int(d / days_per_year)
        if y >= ny:
            break
        out[y] += Kd[d % ndays_rec] * da
        cnt[y] += 1.0
    for y in range(ny):
        if cnt[y] > 0:
            out[y] /= cnt[y]
    return out


def applications(Tr, H):
    """Application times and allocation weights (last coat pro rata)."""
    n = int(np.floor(H / Tr - 1e-12)) + 1
    t = np.arange(n) * Tr
    w = np.ones(n)
    last_service = H - t[-1]
    w[-1] = min(1.0, last_service / Tr)
    return t, w


def delivered_fraction(Tr, tau):
    """phi = (tau / Tr)(1 - exp(-Tr / tau))."""
    return tau / Tr * (1.0 - np.exp(-Tr / tau))


def harmonic_life(life_fun, p: Params):
    """L_eff with 1/L_eff = <1/L(alpha(t))> over a renewal cycle."""
    ph = np.linspace(0.0, p.Tr, 2001)
    a = p.alpha0 + (p.alpha_inf - p.alpha0) + (p.alphaf - p.alpha_inf) * np.exp(-ph / p.tau)
    return 1.0 / np.trapezoid(1.0 / life_fun(a), ph) * p.Tr


def evaluate(Kd, p: Params, life_fun=None, L0=20.0, TH=100.0, metrics_list=("static", "dynamic", "gtp", "stock")):
    """Ledger terms (kg CO2e m-2; + adds to the atmosphere) for one parameter
    set.  Kd: daily kernel (W m-2 per unit albedo) of the record."""
    f_loc = _annual_forcing(Kd * p.kmult, len(Kd), p.alpha0, p.alphaf, p.alpha_inf, p.tau, p.Tr, p.H, DAYS_PER_YEAR)
    f_glob = f_loc * p.psi / A_EARTH                     # W m-2 global, for 1 m2
    t_app, w_app = applications(p.Tr, p.H)
    m_app = p.g_co * w_app
    if life_fun is not None:
        L_eff = harmonic_life(life_fun, p)
        rate = p.g_st * (1.0 / L_eff - 1.0 / L0)         # kg yr-1
    else:
        L_eff, rate = L0, 0.0
    ny = len(f_glob)
    t_st = np.arange(ny) + 0.5
    m_st = np.full(ny, rate)
    out = {}
    for m in metrics_list:
        eq_co = metrics.equivalents(t_app, m_app, f_glob, TH=TH, efficacy=p.efficacy)
        eq_st = metrics.equivalents(t_st, m_st, np.zeros(1), TH=TH)
        G_co = eq_co[m][0]
        G_rf = eq_co[m][1]
        G_st = eq_st[m][0]
        out[m] = dict(G_rf=G_rf, G_co=G_co, G_st=G_st, G_net=G_rf + G_co + G_st,
                      R=(G_co + G_st) / abs(G_rf) if G_rf != 0 else np.inf)
    out["L_eff"] = float(L_eff)
    out["mean_delta_alpha"] = float(np.mean([
        (p.alpha_inf - p.alpha0) + (p.alphaf - p.alpha_inf) * delivered_fraction(p.Tr, p.tau)]))
    out["phi"] = float(delivered_fraction(p.Tr, p.tau))
    out["f_loc_mean"] = float(np.mean(f_loc))
    return out


def kappa(Kd, TH=100.0, efficacy=1.0, psi=1.0):
    """Static-metric credit per unit albedo increase sustained for one year
    (kg CO2e m-2 per albedo-year): |mean kernel| x 1 yr / A_E / AGWP(TH)."""
    return float(abs(np.mean(Kd)) * psi * efficacy / A_EARTH / metrics.agwp(TH)[0])

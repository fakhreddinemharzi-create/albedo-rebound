"""Climate metrics that put a time-varying albedo forcing and pulsed
greenhouse-gas emissions on a common, time-consistent footing.

CO2 impulse response: Joos et al. (2013), as adopted by IPCC AR5
    IRF(t) = a0 + sum_i a_i exp(-t / tau_i),
    a = (0.2173, 0.2240, 0.2824, 0.2763), tau = (394.4, 36.54, 4.304) yr.
Radiative efficiency: 1.37e-5 W m-2 ppb-1 (AR5), i.e. 1.756e-15 W m-2 kg-1.
Temperature response to forcing (AR5; Boucher and Reddy, 2008):
    R_T(t) = sum_j (c_j / d_j) exp(-t / d_j),  c = (0.631, 0.429) K (W m-2)-1,
    d = (8.4, 409.5) yr.
The global-mean forcing of a patch of area a (m2) with local forcing f
(W m-2) is f a / A_earth.
"""
from __future__ import annotations

import numpy as np

from .config import A_EARTH

A0 = 0.2173
AI = np.array([0.2240, 0.2824, 0.2763])
TAUI = np.array([394.4, 36.54, 4.304])
RE_PPB = 1.37e-5
KG_PER_PPB = 1.7726e20 * 44.01e-3 * 1e-9          # moles of dry air x molar mass x 1 ppb
RE_KG = RE_PPB / KG_PER_PPB
CJ = np.array([0.631, 0.429])
DJ = np.array([8.4, 409.5])


def irf(t):
    t = np.asarray(t, float)
    return A0 + np.sum(AI[:, None] * np.exp(-np.atleast_1d(t)[None, :] / TAUI[:, None]), axis=0)


def agwp(h):
    """Absolute GWP of CO2 (W m-2 yr kg-1) for horizon(s) h >= 0."""
    h = np.maximum(np.asarray(h, float), 0.0)
    return RE_KG * (A0 * h + np.sum(AI[:, None] * TAUI[:, None] * (1 - np.exp(-np.atleast_1d(h)[None, :] / TAUI[:, None])), axis=0))


def agtp(h):
    """Absolute GTP of CO2 (K kg-1) at horizon(s) h >= 0 (analytic convolution)."""
    h = np.atleast_1d(np.maximum(np.asarray(h, float), 0.0))
    out = A0 * np.sum(CJ[:, None] * (1 - np.exp(-h[None, :] / DJ[:, None])), axis=0)
    for a, t in zip(AI, TAUI):
        for c, d in zip(CJ, DJ):
            out += a * t * c / (t - d) * (np.exp(-h / t) - np.exp(-h / d))
    return RE_KG * out


def temp_response_kernel(t):
    """R_T(t), K per (W m-2) per year of forcing."""
    t = np.atleast_1d(np.asarray(t, float))
    return np.sum((CJ / DJ)[:, None] * np.exp(-t[None, :] / DJ[:, None]), axis=0)


def global_forcing(local_wm2, area_m2=1.0):
    return np.asarray(local_wm2) * area_m2 / A_EARTH


# --------------------------------------------------------------- ledgers
def equivalents(t_emis, m_emis, f_years, TH=100.0, efficacy=1.0, dt=1.0):
    """CO2-equivalents (kg) of pulsed emissions and of an annual-mean global
    forcing path, under four conventions.

    t_emis, m_emis : emission times (yr from the start) and masses (kg CO2e)
    f_years        : global-mean albedo forcing for years 0..H-1 (W m-2), < 0 cools
    TH             : time horizon (yr) of the metric, fixed at the start (t = 0)

    Returns dict of (burden, credit) pairs:
      static   : emissions at face value; forcing integrated over the analysis
                 period and divided by AGWP(TH)  (conventional GWP-based LCA)
      dynamic  : cumulative forcing of both up to TH, divided by AGWP(TH)
                 (Levasseur et al., 2010)
      gtp      : temperature change at TH, divided by AGTP(TH)
      stock    : forcing-equivalent atmospheric stock (annual-mean forcing / RE),
                 emissions at face value
    """
    t_emis = np.asarray(t_emis, float)
    m_emis = np.asarray(m_emis, float)
    f = np.asarray(f_years, float) * efficacy
    ty = (np.arange(len(f)) + 0.5) * dt
    out = {}
    # static
    out["static"] = (float(m_emis.sum()), float(np.sum(f) * dt / agwp(TH)[0]))
    # dynamic (Levasseur): forcing counted only inside the window [0, TH]
    inside = ty < TH
    crf_e = float(np.sum(m_emis * agwp(TH - t_emis)))
    crf_a = float(np.sum(f[inside]) * dt)
    out["dynamic"] = (crf_e / agwp(TH)[0], crf_a / agwp(TH)[0])
    # gtp
    dT_e = float(np.sum(m_emis * agtp(TH - t_emis)))
    dT_a = float(np.sum(f[inside] * temp_response_kernel(TH - ty[inside])) * dt)
    out["gtp"] = (dT_e / agtp(TH)[0], dT_a / agtp(TH)[0])
    # stock (timeless forcing equivalence of the mean forcing over the period)
    out["stock"] = (float(m_emis.sum()), float(np.mean(f) / RE_KG))
    return out

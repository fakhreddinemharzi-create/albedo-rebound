"""Distress transfer functions and damage accumulation.

* Bottom-up fatigue of the bound layer, Asphalt Institute (Shook et al.,
  1982; lab-to-field shift of Finn et al., 1977 included):
      N_f = 0.0795 eps_t^-3.291 |E*|^-0.854      (|E*| in psi)
* Subgrade rutting, Asphalt Institute:
      N_d = 1.365e-9 eps_v^-4.477
* Permanent deformation of the bound layer (MEPDG, NCHRP 1-37A global model;
  ARA 2004; Schwartz et al., 2012):
      eps_p / eps_r = k_z 10^-3.35412 T^1.5606 N^0.4791   (T in F)
      k_z = (C1 + C2 D) 0.328196^D,
      C1 = -0.1039 h^2 + 2.4868 h - 17.342,  C2 = 0.0172 h^2 - 1.7331 h + 27.428
  (h, D in inches), accumulated across changing conditions by the strain-
  hardening (equivalent-repetition) method.
Miner's rule for the two fatigue-type criteria.
"""
from __future__ import annotations

import numpy as np
from numba import njit

PSI = 6894.757

# MEPDG default hourly truck distribution (% of daily traffic, local hour 0-23)
HOURLY_TRAFFIC = np.array([2.3, 2.3, 2.3, 2.3, 2.3, 2.3, 5.0, 5.0, 5.0, 5.0, 5.9, 5.9,
                           5.9, 5.9, 5.9, 5.9, 4.6, 4.6, 4.6, 4.6, 3.1, 3.1, 3.1, 3.1]) / 100.0


def kz(h_ac_m, depth_m):
    h = h_ac_m / 0.0254
    D = depth_m / 0.0254
    c1 = -0.1039 * h ** 2 + 2.4868 * h - 17.342
    c2 = 0.0172 * h ** 2 - 1.7331 * h + 27.428
    return (c1 + c2 * D) * 0.328196 ** D


@njit(cache=True)
def accumulate(n_h, et, e_ac2, ev, er1, er2, t1, t2, kz1, kz2, h1, h2, rut_limit, dmax, nmax_years):
    """Damage and rutting for a repeated record.

    n_h: repetitions per hour (record length L hours); the record is cycled
    until every criterion is reached or ``nmax_years`` elapse.
    Returns (years to fatigue failure, to subgrade failure, to AC-rut limit,
             annual fatigue damage, annual subgrade damage, AC rut after 1 record).
    """
    L = n_h.shape[0]
    yrs_per_rec = L / 8766.0
    # Miner rates over one record
    df = 0.0
    dd = 0.0
    for i in range(L):
        if n_h[i] <= 0.0:
            continue
        if et[i] > 0.0:
            nf = 0.0795 * et[i] ** -3.291 * (e_ac2[i] * 1e6 / 6894.757) ** -0.854
            df += n_h[i] / nf
        if ev[i] > 0.0:
            nd = 1.365e-9 * ev[i] ** -4.477
            dd += n_h[i] / nd
    df_year = df / yrs_per_rec
    dd_year = dd / yrs_per_rec
    life_f = 1.0 / df_year if df_year > 0 else 1e9
    life_d = 1.0 / dd_year if dd_year > 0 else 1e9
    # AC rutting: strain hardening across the cycled record
    ep1 = 0.0
    ep2 = 0.0
    k1 = 10.0 ** -3.35412
    rut_first = 0.0
    life_r = 1e9
    t_years = 0.0
    done = False
    for cyc in range(int(nmax_years / yrs_per_rec) + 1):
        for i in range(L):
            if n_h[i] <= 0.0:
                continue
            for s in range(2):
                ers = er1[i] if s == 0 else er2[i]
                tf = (t1[i] if s == 0 else t2[i]) * 9.0 / 5.0 + 32.0
                if tf < 1.0:
                    tf = 1.0
                kzs = kz1 if s == 0 else kz2
                if ers <= 0.0:
                    continue
                a = kzs * k1 * tf ** 1.5606 * ers
                ep = ep1 if s == 0 else ep2
                neq = (ep / a) ** (1.0 / 0.4791) if ep > 0.0 else 0.0
                ep_new = a * (neq + n_h[i]) ** 0.4791
                if s == 0:
                    ep1 = ep_new
                else:
                    ep2 = ep_new
            rut = ep1 * h1 + ep2 * h2
            if (not done) and rut >= rut_limit:
                life_r = t_years + (i + 1) / 8766.0
                done = True
        t_years += yrs_per_rec
        if cyc == 0:
            rut_first = ep1 * h1 + ep2 * h2
        if done:
            break
    return life_f, life_d, life_r, df_year, dd_year, rut_first

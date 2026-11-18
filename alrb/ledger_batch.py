"""Vectorised ledger for large samples (Monte Carlo over streets).

Evaluates the ledger of ``ledger.evaluate`` for arrays of parameter sets in
one compiled, parallel pass.  The terms are returned separately and for an
open street with unit efficacy, so that the urban-form factor, the efficacy
and a kernel multiplier, which scale the radiative term only, can be applied
afterwards.  The structural term compares the harmonic-mean life of the
treated surface with the life of the untreated surface at its own albedo
alpha0 (``L0=None``, the default) or with a fixed life ``L0``.  It agrees
with ``ledger.evaluate`` to round-off for the same parameters.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from . import metrics
from .config import A_EARTH

DAYS_PER_YEAR = 365.25
_A0 = metrics.A0
_AI = metrics.AI
_TAUI = metrics.TAUI
_CJ = metrics.CJ
_DJ = metrics.DJ
_RE = metrics.RE_KG


@njit(cache=True)
def _agwp(h):
    h = max(h, 0.0)
    s = _A0 * h
    for i in range(3):
        s += _AI[i] * _TAUI[i] * (1.0 - np.exp(-h / _TAUI[i]))
    return _RE * s


@njit(cache=True)
def _agtp(h):
    h = max(h, 0.0)
    s = 0.0
    for j in range(2):
        s += _A0 * _CJ[j] * (1.0 - np.exp(-h / _DJ[j]))
    for i in range(3):
        for j in range(2):
            s += _AI[i] * _TAUI[i] * _CJ[j] / (_TAUI[i] - _DJ[j]) * (np.exp(-h / _TAUI[i]) - np.exp(-h / _DJ[j]))
    return _RE * s


@njit(cache=True)
def _rt(t):
    s = 0.0
    for j in range(2):
        s += _CJ[j] / _DJ[j] * np.exp(-t / _DJ[j])
    return s


@njit(cache=True)
def _interp(x, xp, fp):
    n = len(xp)
    if x <= xp[0]:
        return fp[0]
    if x >= xp[n - 1]:
        return fp[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    w = (x - xp[lo]) / (xp[hi] - xp[lo])
    return fp[lo] + w * (fp[hi] - fp[lo])


@njit(cache=True)
def _life(alpha, hmult, al, hs, L):
    """L(alpha, hmult): linear in alpha, then in the convective multiplier."""
    nh = len(hs)
    if nh == 1:
        return _interp(alpha, al, L[0])
    if hmult <= hs[0]:
        return _interp(alpha, al, L[0])
    if hmult >= hs[nh - 1]:
        return _interp(alpha, al, L[nh - 1])
    k = 0
    while hs[k + 1] < hmult:
        k += 1
    w = (hmult - hs[k]) / (hs[k + 1] - hs[k])
    return (1.0 - w) * _interp(alpha, al, L[k]) + w * _interp(alpha, al, L[k + 1])


@njit(parallel=True, cache=True)
def _batch(Kd, cell, alpha0, alphaf, alpha_inf, tau, Tr, g_co, g_st, hmult, life_id, st_scale,
           al, hs, Ltab, L0, H, TH, a_earth, nquad):
    n = len(cell)
    out = np.zeros((n, 9))
    ny = int(H + 0.5)
    nd = int(H * DAYS_PER_YEAR)
    nrec = Kd.shape[1]
    agwp_th = _agwp(TH)
    agtp_th = _agtp(TH)
    for i in prange(n):
        # radiative term (open street, unit efficacy): annual-mean forcing
        f = np.zeros(ny)
        cnt = np.zeros(ny)
        trd = Tr[i] * DAYS_PER_YEAR
        taud = tau[i] * DAYS_PER_YEAR
        c = cell[i]
        for d in range(nd):
            ph = d % trd
            da = (alpha_inf[i] - alpha0[i]) + (alphaf[i] - alpha_inf[i]) * np.exp(-ph / taud)
            y = int(d / DAYS_PER_YEAR)
            if y >= ny:
                break
            f[y] += Kd[c, d % nrec] * da
            cnt[y] += 1.0
        s_static = 0.0
        s_gtp = 0.0
        for y in range(ny):
            if cnt[y] > 0:
                f[y] /= cnt[y]
            fg = f[y] / a_earth
            ty = y + 0.5
            if ty < TH:
                s_static += fg
                s_gtp += fg * _rt(TH - ty)
        out[i, 0] = s_static / agwp_th               # G_rf static (= dynamic while H <= TH)
        out[i, 1] = s_gtp / agtp_th                  # G_rf GTP
        # coats: pulses at k Tr, the last allocated pro rata
        napp = int(np.floor(H / Tr[i] - 1e-12)) + 1
        for k in range(napp):
            t = k * Tr[i]
            w = 1.0
            if k == napp - 1:
                w = min(1.0, (H - t) / Tr[i])
            m = g_co[i] * w
            out[i, 2] += m
            out[i, 3] += m * _agwp(TH - t) / agwp_th
            out[i, 4] += m * _agtp(TH - t) / agtp_th
        # structure: flow g_st (1/L_eff - 1/L0) with the harmonic-mean life
        j = life_id[i]
        acc = 0.0
        for q in range(nquad):
            ph = Tr[i] * q / (nquad - 1)
            a = alpha_inf[i] + (alphaf[i] - alpha_inf[i]) * np.exp(-ph / tau[i])
            v = 1.0 / _life(a, hmult[i], al, hs, Ltab[j])
            acc += v if (q > 0 and q < nquad - 1) else 0.5 * v
        L_eff = (nquad - 1) / acc
        L_base = L0 if L0 > 0 else _life(alpha0[i], hmult[i], al, hs, Ltab[j])
        rate = g_st[i] * (1.0 / L_eff - 1.0 / L_base) * st_scale[i]
        for y in range(ny):
            ty = y + 0.5
            out[i, 5] += rate
            out[i, 6] += rate * _agwp(TH - ty) / agwp_th
            out[i, 7] += rate * _agtp(TH - ty) / agtp_th
        out[i, 8] = L_eff
    return out


def evaluate(Kd, cell, alpha0, alphaf, alpha_inf, tau, Tr, g_co, g_st, hmult, life_id, al, hs, Ltab,
             st_scale=None, L0=None, H=50.0, TH=100.0, nquad=2001):
    """Ledger terms for n parameter sets.

    Kd      : (ncell, ndays) daily kernels (W m-2 per unit albedo)
    cell    : (n,) row of Kd of every sample
    life_id : (n,) row of Ltab (ntab, nh, nal), the life tables L(alpha) at
              the convective multipliers hs
    st_scale: (n,) factor on the structural term (1 for an open street)
    L0      : life of the untreated surface (yr); None: L(alpha0) from the
              life table of the sample

    Returns a dict of arrays: G_rf (static = dynamic, gtp) for psi = E = 1,
    G_co and G_st (static, dynamic, gtp), and L_eff.
    """
    n = len(cell)
    f8 = lambda x: np.ascontiguousarray(np.broadcast_to(np.asarray(x, float), (n,)))  # noqa: E731
    st = np.ones(n) if st_scale is None else f8(st_scale)
    o = _batch(np.ascontiguousarray(Kd, float), np.ascontiguousarray(cell, np.int64), f8(alpha0), f8(alphaf),
               f8(alpha_inf), f8(tau), f8(Tr), f8(g_co), f8(g_st), f8(hmult),
               np.ascontiguousarray(np.broadcast_to(np.asarray(life_id, np.int64), (n,))), st,
               np.ascontiguousarray(al, float), np.ascontiguousarray(hs, float), np.ascontiguousarray(Ltab, float),
               -1.0 if L0 is None else float(L0), H, TH, A_EARTH, nquad)
    return dict(G_rf=dict(static=o[:, 0], dynamic=o[:, 0], gtp=o[:, 1]),
                G_co=dict(static=o[:, 2], dynamic=o[:, 3], gtp=o[:, 4]),
                G_st=dict(static=o[:, 5], dynamic=o[:, 6], gtp=o[:, 7]), L_eff=o[:, 8])

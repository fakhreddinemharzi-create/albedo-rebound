"""Layer stiffness from the thermal state.

Bound layer
    Dynamic modulus from the Witczak 1-37A predictive equation (Andrei et al.,
    1999; ARA, 2004), with the binder viscosity from the A-VTS relation
        log log eta(cP) = A + VTS log T_R  (T_R in Rankine),
    capped at 2.7e10 P, at a loading frequency of 10 Hz.  Mixture
    volumetrics are those of a typical dense-graded surface mix; the binder
    is a PG 64-22 (A = 10.98, VTS = -3.680).

Unbound layers
    Equilibrium (unfrozen) resilient modulus, frozen modulus of 6895 MPa
    (1e6 psi; Witczak et al., 2000b), weighted in log space by the layer-mean
    ice fraction, and thaw weakening with linear recovery
        M_R = M_R,unf (RF + (1 - RF) t / T_R),  0 <= t <= T_R,
    after each thaw (RF, T_R as in the design framework).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

PSI = 6894.757


@dataclass
class ACMix:
    p200: float = 6.0      # % passing No. 200
    p4: float = 40.0       # cumulative % retained on No. 4
    p38: float = 20.0      # cumulative % retained on 3/8 in
    p34: float = 0.0       # cumulative % retained on 3/4 in
    va: float = 7.0        # air voids (%)
    vbeff: float = 11.0    # effective binder content (% by volume)
    A: float = 10.98
    VTS: float = -3.680
    freq: float = 10.0


@njit(cache=True)
def _witczak(Tc, p200, p4, p38, p34, va, vbeff, A, VTS, f):
    tr = (Tc * 9.0 / 5.0 + 32.0) + 459.67
    ll = A + VTS * np.log10(tr)
    eta_cp = 10.0 ** (10.0 ** ll)
    eta_p = eta_cp / 100.0
    if eta_p > 2.7e10:
        eta_p = 2.7e10
    eta = eta_p / 1e6                          # 10^6 P
    c = (-1.249937 + 0.029232 * p200 - 0.001767 * p200 ** 2 - 0.002841 * p4 - 0.058097 * va
         - 0.802208 * vbeff / (vbeff + va))
    num = 3.871977 - 0.0021 * p4 + 0.003958 * p38 - 0.000017 * p38 ** 2 + 0.005470 * p34
    loge = c + num / (1.0 + np.exp(-0.603313 - 0.313351 * np.log10(f) - 0.393532 * np.log10(eta)))
    return 10.0 ** loge * 1e5 * 6894.757 / 1e6          # MPa


@njit(cache=True)
def ac_modulus_arr(T, p200, p4, p38, p34, va, vbeff, A, VTS, f):
    out = np.empty(T.shape[0])
    for i in range(T.shape[0]):
        out[i] = _witczak(T[i], p200, p4, p38, p34, va, vbeff, A, VTS, f)
    return out


def ac_modulus(T, mix: ACMix = ACMix()):
    T = np.atleast_1d(np.asarray(T, float))
    return ac_modulus_arr(T, mix.p200, mix.p4, mix.p38, mix.p34, mix.va, mix.vbeff, mix.A, mix.VTS, mix.freq)


@dataclass
class Unbound:
    name: str
    mr: float           # equilibrium unfrozen modulus (MPa)
    rf: float           # thaw reduction factor
    tr_days: float      # recovery period (days)
    nu: float = 0.35


BASE_U = Unbound("base", 250.0, 0.65, 90.0)
SUBBASE_U = Unbound("subbase", 150.0, 0.60, 90.0)
SUBGRADE_U = Unbound("subgrade", 70.0, 0.55, 120.0, nu=0.45)
MR_FROZEN = 6895.0


@njit(cache=True)
def unbound_series(ice, mr, rf, tr_h, mr_frozen, thaw_hi, thaw_lo):
    """Hourly modulus of an unbound layer from its layer-mean ice fraction.
    A thaw is registered when the ice fraction falls below ``thaw_lo``
    after having exceeded ``thaw_hi``; recovery then runs over tr_h hours."""
    n = ice.shape[0]
    out = np.empty(n)
    frozen_before = False
    t_since = 1e9
    lmu = np.log(mr)
    lmf = np.log(mr_frozen)
    for i in range(n):
        f = ice[i]
        if f >= thaw_hi:
            frozen_before = True
            t_since = 1e9
        elif f < thaw_lo and frozen_before:
            frozen_before = False
            t_since = 0.0
        # recovering modulus
        if t_since < tr_h:
            m_u = mr * (rf + (1.0 - rf) * t_since / tr_h)
        else:
            m_u = mr
        # partially frozen layer: log-linear in the ice fraction
        out[i] = np.exp((1.0 - f) * np.log(m_u) + f * lmf)
        t_since += 1.0
    return out

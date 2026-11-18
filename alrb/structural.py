"""From the hourly thermal state of the pavement column to layer moduli,
critical strains, damage and service life."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import damage, stiffness, strain_table
from .config import PROC, SITE, RAW

H_AC = 0.10
RUT_LIMIT = 0.00635          # 0.25 in of permanent deformation in the bound layer
_TAB = None


def table():
    global _TAB
    if _TAB is None:
        z = np.load(PROC / "strain_table.npz")
        _TAB = (z["tab"], z["X1"], z["X2"], z["X3"], z["X4"])
    return _TAB


@dataclass
class State:
    E: np.ndarray            # (n, 5) moduli, MPa
    T1: np.ndarray           # bound-layer temperature at 25 mm
    T2: np.ndarray           # at 75 mm
    strains: np.ndarray      # (n, 4): eps_t, eps_v, eps_r25, eps_r75


def hourly_state(T, lice, depths, mix=stiffness.ACMix(), units=(stiffness.BASE_U, stiffness.SUBBASE_U,
                                                                 stiffness.SUBGRADE_U)):
    """T: hourly mean temperatures at the recorded depths; lice: layer-mean
    ice fractions (base, subbase, upper subgrade)."""
    i25 = int(np.argmin(np.abs(depths - 0.025)))
    i75 = int(np.argmin(np.abs(depths - 0.075)))
    T1 = np.asarray(T[:, i25], float)
    T2 = np.asarray(T[:, i75], float)
    E1 = stiffness.ac_modulus(T1, mix)
    E2 = stiffness.ac_modulus(T2, mix)
    Eu = []
    for q, u in enumerate(units):
        Eu.append(stiffness.unbound_series(np.asarray(lice[:, q], float), u.mr, u.rf, u.tr_days * 24.0,
                                           stiffness.MR_FROZEN, 0.5, 0.05))
    E = np.column_stack([E1, E2] + Eu)
    tab, x1, x2, x3, x4 = table()
    st = strain_table.interp_cubic(tab, x1, x2, x3, x4, E[:, 0].copy(), E[:, 1].copy(), E[:, 2].copy(),
                             E[:, 3].copy(), E[:, 4].copy())
    return State(E, T1, T2, st)


def traffic_weights(index_utc, tz):
    """Share of the daily traffic in each hour (MEPDG default distribution,
    local standard time)."""
    hl = ((index_utc + pd.Timedelta(hours=tz)).hour).to_numpy()
    return damage.HOURLY_TRAFFIC[hl]


def lives(state: State, weights, n_day, rut_limit=RUT_LIMIT, nmax_years=200.0):
    n_h = weights * n_day
    kz1 = damage.kz(H_AC, 0.025)
    kz2 = damage.kz(H_AC, 0.075)
    st = state.strains
    return damage.accumulate(n_h, st[:, 0].copy(), state.E[:, 1].copy(), st[:, 1].copy(), st[:, 2].copy(),
                             st[:, 3].copy(), state.T1, state.T2, kz1, kz2, 0.05, 0.05, rut_limit, 1.0, nmax_years)


def run_site(key, alb, tag="ref", n_day=1.0):
    z = np.load(RAW / "sweep" / tag / f"{key}_a{alb:.2f}.npz")
    idx = pd.date_range(pd.Timestamp(str(z["t0"])), periods=int(z["n"]), freq="h")
    st = hourly_state(z["T"], z["lice"], z["depths"])
    w = traffic_weights(idx, SITE[key].tz)
    return st, w, lives(st, w, n_day)

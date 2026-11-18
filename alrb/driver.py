"""Forcing preparation and run control for the column model."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import atmos, column, solar
from .config import PROC, SITE

NSUB = 3                      # sub-steps per hour (20 min)
DT = 3600.0 / NSUB


def load_forcing(key: str) -> pd.DataFrame:
    df = pd.read_parquet(PROC / f"forcing_{key}.parquet")
    return df.astype(float)


@dataclass
class SubForcing:
    sw: np.ndarray
    lw: np.ndarray
    ta: np.ndarray
    qa: np.ndarray
    ws: np.ndarray
    pr: np.ndarray
    p: np.ndarray
    snow: np.ndarray
    index: pd.DatetimeIndex      # hourly stamps
    nsub: int = NSUB


def substeps(df: pd.DataFrame, key: str, nsub: int = NSUB, lw_col: str = "lw",
             snow_mask: np.ndarray | None = None) -> SubForcing:
    """Sub-hourly forcing.  Shortwave: the hourly total is distributed over
    the sub-steps in proportion to the extraterrestrial irradiance at their
    mid-points, which conserves the measured energy exactly and follows the
    solar geometry inside the hour.  State variables (T, humidity, LW) are
    interpolated linearly between hour centres."""
    s = SITE[key]
    idx = df.index
    nh = len(idx)
    frac = (np.arange(nsub) + 0.5) / nsub - 1.0          # offsets (h) within the hour ending at H
    base = idx.values.astype("datetime64[s]").astype(np.int64)
    ts = pd.to_datetime((base[:, None] + frac[None, :] * 3600.0).ravel(), unit="s", utc=True)
    mu, e0 = solar.cos_zenith(ts, s.lat, s.lon)
    g = (1361.0 * e0 * np.clip(mu, 0, None)).reshape(nh, nsub)
    gm = g.mean(axis=1, keepdims=True)
    sw_h = df["sw"].to_numpy()[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(gm > 1e-6, g / gm, 1.0)
    sw = (sw_h * w).ravel()
    # hour-centre interpolation
    tc = np.arange(nh) - 0.5                               # centre of hour k, in hours from stamp 0
    tsub = (np.arange(nh)[:, None] + frac[None, :]).ravel()

    def interp(x):
        return np.interp(tsub, tc, np.asarray(x, float))

    ta = interp(df["ta"])
    ea = interp(df["ea"])
    p = np.repeat(df["p"].to_numpy(), nsub)
    qa = atmos.specific_humidity(ea, p)
    lw = interp(df[lw_col])
    ws = np.repeat(df["ws"].to_numpy(), nsub)
    pr = np.repeat(df["pr"].to_numpy(), nsub)
    if snow_mask is None:
        snow = np.zeros(nh * nsub)
    else:
        snow = np.repeat(np.asarray(snow_mask, float), nsub)
    return SubForcing(sw, lw, ta, qa, ws, pr, p, snow, idx, nsub)


def snow_mask_pavement(df: pd.DataFrame, key: str, days_after: float = 1.0, thresh_mm: float = 1.0):
    """Hours at which a plowed street is taken as snow- or slush-covered:
    from the start of a local calendar day whose snowfall (water equivalent,
    MERRA-2) exceeds ``thresh_mm`` until ``days_after`` days after it ends."""
    s = SITE[key]
    sf = df["snowfall"].copy()                # kg m-2 s-1 (POWER: mm/hour); convert robustly
    # POWER PRECSNO is given in mm/hour for hourly data
    loc = sf.copy()
    loc.index = loc.index + pd.Timedelta(hours=s.tz)
    daily = loc.resample("D").sum()
    snowy = daily[daily > thresh_mm].index
    mask_loc = pd.Series(0.0, index=loc.index)
    for d in snowy:
        mask_loc[(mask_loc.index >= d) & (mask_loc.index < d + pd.Timedelta(days=1.0 + days_after))] = 1.0
    return mask_loc.to_numpy()


@dataclass
class RunResult:
    index: pd.DatetimeIndex
    ts: np.ndarray            # hourly mean skin temperature
    ts_end: np.ndarray        # skin temperature at the end of each hour
    T: np.ndarray             # hourly mean temperature at recorded depths
    ice: np.ndarray
    flux: np.ndarray          # Rn, H, LE, G, absorbed SW
    w: np.ndarray
    frost: np.ndarray
    lice: np.ndarray          # layer-mean ice fraction (base, subbase, upper subgrade)
    depths: np.ndarray
    nfail: int
    meta: dict = field(default_factory=dict)


REC_DEPTHS_PAV = np.array([0.0125, 0.025, 0.05, 0.075, 0.10, 0.20, 0.225, 0.35, 0.50, 0.65, 0.70, 0.85, 1.0, 1.5, 2.0, 3.0])


def prepare(struct: column.Structure, rec_depths, frost_layers=(1, 2, 3), upper_sg=1.0):
    edges, dz, zc, mat = struct.grid()
    Ttab, Hs, Ks, Is = struct.tables()
    rec = np.array([int(np.argmin(np.abs(zc - d))) for d in rec_depths])
    lay = -np.ones(len(zc), np.int64)
    for q, L in enumerate(frost_layers):
        top = struct.layers[L].top
        bot = struct.layers[L].bottom
        if q == len(frost_layers) - 1:
            bot = min(bot, top + upper_sg)
        lay[(zc >= top) & (zc < bot)] = q
    return dict(edges=edges, dz=dz, zc=zc, mat=mat, Ttab=Ttab, Hs=Hs, Ks=Ks, Is=Is, rec=rec, lay=lay,
                nlay=len(frost_layers))


def initial_state(pre, t_init):
    H0 = np.array([column._h_of_t(pre["Hs"][m], pre["Ttab"], t_init) for m in pre["mat"]])
    return H0


def run(pre, sub: SubForcing, alb, eps=0.95, zr=1.5, z0m=0.005, z0h=None, hmult=1.0,
        surface=0, wcap=1.0, rs_day=50.0, rs_night=200.0, alb_snow=0.6,
        H0=None, ts0=None, w0=0.0, t_init=10.0, zmax_frost=4.0, t_bot=None):
    n = len(sub.sw)
    alb_arr = np.full(n, float(alb)) if np.ndim(alb) == 0 else np.asarray(alb, float)
    if len(alb_arr) != n:
        raise ValueError("albedo series must have one value per sub-step")
    if H0 is None:
        H0 = initial_state(pre, t_init)
        ts0 = t_init
    z0h = z0m / 10.0 if z0h is None else z0h
    out = column.run(pre["dz"], pre["zc"], pre["mat"], pre["Ttab"], pre["Hs"], pre["Ks"],
                     alb_arr, eps, zr, z0m, z0h, hmult,
                     sub.sw, sub.lw, sub.ta, sub.qa, sub.ws, sub.pr, sub.p, sub.snow,
                     DT if sub.nsub == NSUB else 3600.0 / sub.nsub, sub.nsub, surface, wcap, rs_day, rs_night, alb_snow,
                     H0, ts0, w0, pre["rec"], pre["Is"], pre["lay"], pre["nlay"], zmax_frost,
                     t_bot is not None,
                     np.zeros(1) if t_bot is None else (np.full(n, float(t_bot)) if np.ndim(t_bot) == 0
                                                        else np.asarray(t_bot, float)))
    (ts, ts_end, T, ice, flux, w, it, frost, lice, H, tsf, wf, nfail) = out
    res = RunResult(sub.index, ts, ts_end, T, ice, flux, w, frost, lice, pre["zc"][pre["rec"]], int(nfail),
                    meta=dict(mean_iter=float(np.mean(it))))
    return res, (H, tsf, wf)


def slice_sub(sub: SubForcing, i0: int, i1: int) -> SubForcing:
    """Hours [i0, i1) of a sub-step forcing record."""
    k0, k1 = i0 * sub.nsub, i1 * sub.nsub
    return SubForcing(sub.sw[k0:k1], sub.lw[k0:k1], sub.ta[k0:k1], sub.qa[k0:k1], sub.ws[k0:k1],
                      sub.pr[k0:k1], sub.p[k0:k1], sub.snow[k0:k1], sub.index[i0:i1], sub.nsub)


def spun_up_run(pre, sub: SubForcing, alb, spin_years=3, t_init=None, equilibrate=True, return_state=False,
                w0=0.0, **kw):
    """Spin-up.  One year from a uniform profile; the ground below 1 m is then
    reset to the annual-mean temperature simulated at 1 m (which carries the
    thermal offset of the frozen/thawed conductivity contrast), and the first
    year is cycled ``spin_years`` further times before the record is run.
    Without the reset the deep ground of a strongly cooled surface is still
    drifting after several years (Sect. 4.1)."""
    nh1 = 8760
    first = slice_sub(sub, 0, nh1)
    if t_init is None:
        t_init = float(np.mean(sub.ta)) + (2.0 if kw.get("surface", 0) == 0 else 1.0)
    a1 = alb if np.ndim(alb) == 0 else np.asarray(alb)[: nh1 * sub.nsub]
    kw1 = dict(kw)
    if kw.get("t_bot") is not None and np.ndim(kw["t_bot"]) > 0:
        kw1["t_bot"] = np.asarray(kw["t_bot"])[: nh1 * sub.nsub]
    r0, state = run(pre, first, a1, t_init=t_init, w0=w0, **kw1)
    if equilibrate:
        q1 = int(np.argmin(np.abs(pre["zc"][pre["rec"]] - 1.0)))
        t1 = float(np.mean(r0.T[:, q1]))
        H = state[0].copy()
        deep = pre["zc"] >= 1.0
        for i in np.where(deep)[0]:
            H[i] = column._h_of_t(pre["Hs"][pre["mat"][i]], pre["Ttab"], t1)
        state = (H, state[1], state[2])
    for _ in range(spin_years):
        _, state = run(pre, first, a1, H0=state[0], ts0=state[1], w0=state[2], **kw1)
    res, st = run(pre, sub, alb, H0=state[0], ts0=state[1], w0=state[2], **kw)
    if return_state:
        return res, state, st
    return res

"""Verification of the column solver.

1. Periodic surface temperature over a homogeneous half-space against the
   steady-periodic solution (Carslaw and Jaeger, 1959, Sect. 2.6).
2. Two-phase freezing of a half-space (Neumann solution; Carslaw and
   Jaeger, 1959, Sect. 11.2; Lunardini, 1991; used as a benchmark for
   cold-regions models by Kurylyk et al., 2014).
3. Energy conservation over a year of real forcing with freezing.
4. Convergence with time step and grid spacing (reported quantities).
Writes results/verification_thermal.json and the profile data for the figure.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import erf, erfc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import column, driver  # noqa: E402
from alrb.config import RES  # noqa: E402


def const_forcing(nh, nsub, ta_fun, dt):
    n = nh * nsub
    t = (np.arange(n) + 0.5) * dt
    idx = pd.date_range("2000-01-01 01:00", periods=nh, freq="h", tz="UTC")
    z = np.zeros(n)
    return driver.SubForcing(sw=z.copy(), lw=z.copy(), ta=ta_fun(t), qa=np.full(n, 1e-3), ws=np.full(n, 5.0),
                             pr=z.copy(), p=np.full(n, 1013.0), snow=z.copy(), index=idx, nsub=nsub), t


def uniform_grid(dz0, zbot, growth=1.0, dzmax=0.05):
    e = [0.0]
    d = dz0
    while e[-1] < zbot:
        e.append(e[-1] + d)
        d = min(d * growth, dzmax)
    e = np.array(e)
    return np.diff(e), 0.5 * (e[1:] + e[:-1])


def run_custom(dz, zc, Ttab, H, K, ice, sub, t_init, rec):
    n = len(dz)
    mat = np.zeros(n, np.int64)
    H0 = np.full(n, np.interp(t_init, Ttab, H))
    lay = -np.ones(n, np.int64)
    out = column.run(dz, zc, mat, Ttab, H[None, :], K[None, :], np.zeros(len(sub.sw)), 0.0, 1.5, 0.01, 0.001, 1e5,
                     sub.sw, sub.lw, sub.ta, sub.qa, sub.ws, sub.pr, sub.p, sub.snow,
                     3600.0 / sub.nsub, sub.nsub, 0, 0.0, 50.0, 200.0, 0.6, H0, t_init, 0.0, rec, ice[None, :],
                     lay, 1, 5.0)
    return out


def harmonic():
    k, C = 1.5, 2.1e6
    kap = k / C
    P = 86400.0
    w = 2 * np.pi / P
    Ttab = np.linspace(-60, 90, 3001)
    H = C * Ttab
    K = np.full_like(Ttab, k)
    ice = np.zeros_like(Ttab)
    dz, zc = uniform_grid(0.002, 2.0, 1.08, 0.05)
    nsub = 12
    dt = 3600.0 / nsub
    ndays = 12
    sub, t = const_forcing(24 * ndays, nsub, lambda t: 20.0 + 10.0 * np.sin(w * t), dt)
    depths = np.array([0.02, 0.05, 0.10, 0.20, 0.40])
    rec = np.array([int(np.argmin(np.abs(zc - d))) for d in depths])
    out = run_custom(dz, zc, Ttab, H, K, ice, sub, 20.0, rec)
    Tm = out[2]                          # hourly means at the recorded cells
    zr = zc[rec]
    d = np.sqrt(2 * kap / w)
    th = (np.arange(24 * ndays) + 0.5) * 3600.0
    # exact hourly mean of the periodic solution
    err = []
    prof = {}
    for q, z in enumerate(zr):
        a = np.exp(-z / d)
        # mean over each hour of 10 a sin(w t - z/d)
        t0 = th - 1800.0
        t1 = th + 1800.0
        ex = 20.0 + 10.0 * a * (-(np.cos(w * t1 - z / d) - np.cos(w * t0 - z / d)) / (w * 3600.0))
        last = slice(24 * (ndays - 2), None)
        err.append(float(np.max(np.abs(Tm[last, q] - ex[last]))))
        prof[f"{z:.3f}"] = dict(model=Tm[last, q].tolist(), exact=ex[last].tolist())
    amp_ratio = [float((Tm[-24:, q].max() - Tm[-24:, q].min()) / (2 * 10 * np.exp(-z / d) *
                                                                   np.sinc(3600.0 / P))) for q, z in enumerate(zr)]
    return dict(depths=zr.tolist(), max_abs_err=err, amp_ratio=amp_ratio, damping_depth=float(d)), prof


def neumann():
    kf, ku = 2.0, 1.2
    Cf, Cu = 1.9e6, 2.6e6
    thw = 0.30
    L = 1000.0 * 3.34e5 * thw
    Ti, Ts_, Tf = 5.0, -10.0, 0.0
    delta = 0.02
    Ttab = np.concatenate([np.linspace(-60, -delta, 3000), np.linspace(-delta, 0, 400)[1:], np.linspace(0, 90, 3000)[1:]])
    H = np.where(Ttab < -delta, Cf * Ttab - L,
                 np.where(Ttab <= 0, -L * (-Ttab / delta) + Cf * Ttab * 0 + (Cu * Ttab), Cu * Ttab))
    # continuous at -delta: Cf(-delta) - L vs -L + Cu(-delta) -> adjust to be exact
    H = np.where(Ttab < -delta, Cf * (Ttab + delta) + (-L - Cu * delta), H)
    K = np.where(Ttab < -delta, kf, np.where(Ttab <= 0, kf + (ku - kf) * (Ttab + delta) / delta, ku))
    ice = np.where(Ttab < -delta, 1.0, np.where(Ttab <= 0, -Ttab / delta, 0.0))
    kapf, kapu = kf / Cf, ku / Cu
    nu = np.sqrt(kapf / kapu)

    def f(lam):
        return (np.exp(-lam ** 2) / erf(lam)
                - (ku / kf) * nu * (Ti - Tf) / (Tf - Ts_) * np.exp(-lam ** 2 * nu ** 2) / erfc(lam * nu)
                - lam * L * np.sqrt(np.pi) / (Cf * (Tf - Ts_)))
    lam = brentq(f, 1e-4, 3.0)
    dz, zc = uniform_grid(0.002, 4.0, 1.02, 0.01)
    nsub = 12
    dt = 3600.0 / nsub
    ndays = 30
    sub, _ = const_forcing(24 * ndays, nsub, lambda t: np.full_like(t, Ts_), dt)
    depths = np.array([0.05, 0.10, 0.20, 0.30, 0.50])
    rec = np.array([int(np.argmin(np.abs(zc - d))) for d in depths])
    n = len(dz)
    mat = np.zeros(n, np.int64)
    H0 = np.full(n, np.interp(Ti, Ttab, H))
    lay = np.zeros(n, np.int64)
    out = column.run(dz, zc, mat, Ttab, H[None, :], K[None, :], np.zeros(len(sub.sw)), 0.0, 1.5, 0.01, 0.001, 1e6,
                     sub.sw, sub.lw, sub.ta, sub.qa, sub.ws, sub.pr, sub.p, sub.snow, dt, nsub, 0, 0.0, 50.0, 200.0,
                     0.6, H0, Ti, 0.0, rec, ice[None, :], lay, 1, 3.9)
    frost = out[7]
    th = (np.arange(24 * ndays) + 1.0) * 3600.0
    X = 2 * lam * np.sqrt(kapf * th)
    sel = th > 2 * 86400
    front_err = float(np.max(np.abs(frost[sel] - X[sel])))
    front_rel = float(np.max(np.abs(frost[sel] - X[sel]) / X[sel]))
    # temperature profile at the end, exact vs model (hourly mean of the last hour ~ end state)
    Tm = out[2][-1]
    t_end = th[-1] - 1800.0
    zr = zc[rec]
    Xe = 2 * lam * np.sqrt(kapf * t_end)
    ex = np.where(zr < Xe, Ts_ + (Tf - Ts_) * erf(zr / (2 * np.sqrt(kapf * t_end))) / erf(lam),
                  Ti - (Ti - Tf) * erfc(zr / (2 * np.sqrt(kapu * t_end))) / erfc(lam * nu))
    return dict(lam=float(lam), front_max_abs_err=front_err, front_max_rel_err=front_rel,
                T_depths=zr.tolist(), T_model=Tm.tolist(), T_exact=ex.tolist(),
                T_max_abs_err=float(np.max(np.abs(Tm - ex)))), dict(t_h=(th / 3600).tolist(), model=frost.tolist(),
                                                                    exact=X.tolist())


def conservation_and_convergence():
    key = "goodridge"
    df = driver.load_forcing(key).loc["2018-07-01 01:00":"2019-07-01 00:00"]
    out = {}
    runs = {}
    for label, nsub, st_kw in [("ref", 3, {}), ("dt_60min", 1, {}), ("dt_5min", 12, {}),
                               ("fine_grid", 3, dict(dz_top=0.002, growth=1.10, dz_max_frost=0.04)),
                               ("coarse_grid", 3, dict(dz_top=0.008, growth=1.25, dz_max_frost=0.16))]:
        st = column.pavement_structure()
        for k2, v in st_kw.items():
            setattr(st, k2, v)
        pre = driver.prepare(st, driver.REC_DEPTHS_PAV)
        sub = driver.substeps(df, key, nsub=nsub)
        res = {}
        for a in (0.14, 0.45):
            H0 = driver.initial_state(pre, float(df.ta.mean()) + 2.0)
            r, st_end = driver.run(pre, sub, a, H0=H0, ts0=float(df.ta.mean()) + 2.0)
            # spin-up: repeat the year twice from the end state
            for _ in range(2):
                r, st_end = driver.run(pre, sub, a, H0=st_end[0], ts0=st_end[1], w0=st_end[2])
            res[a] = r
            if label == "ref":
                # energy conservation of the final year: sum dz (H_end - H_start) vs integral of G
                Hs = driver.run(pre, sub, a, H0=st_end[0], ts0=st_end[1], w0=st_end[2])
                r2, st2 = Hs
                dE = float(np.sum(pre["dz"] * (st2[0] - st_end[0])))
                Gint = float(np.sum(r2.flux[:, 3]) * 3600.0)
                out[f"energy_alb{a}"] = dict(storage_change_MJ=dE / 1e6, surface_flux_MJ=Gint / 1e6,
                                             rel_error=abs(dE - Gint) / max(np.sum(np.abs(r2.flux[:, 3])) * 3600.0, 1))
        r1, r2 = res[0.14], res[0.45]
        runs[label] = dict(ts_mean=float(r1.ts.mean()), ts_max=float(r1.ts.max()), frost_max=float(r1.frost.max()),
                           dts_mean=float((r2.ts - r1.ts).mean()), dts_max=float(r2.ts.max() - r1.ts.max()),
                           dfrost=float(r2.frost.max() - r1.frost.max()),
                           dT085=float((r2.T[:, 11] - r1.T[:, 11]).mean()),
                           nfail=int(r1.nfail + r2.nfail), ncell=int(len(pre["zc"])))
    out["convergence"] = runs
    return out


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    h, hprof = harmonic()
    print("harmonic", h)
    nm, nfront = neumann()
    print("neumann", {k: v for k, v in nm.items() if not isinstance(v, list)})
    cc = conservation_and_convergence()
    print(json.dumps(cc, indent=1))
    (RES / "verification_thermal.json").write_text(json.dumps(dict(harmonic=h, neumann=nm, **cc), indent=1))
    (RES / "verification_thermal_profiles.json").write_text(json.dumps(dict(harmonic=hprof, neumann_front=nfront)))

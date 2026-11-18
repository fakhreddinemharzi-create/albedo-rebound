"""Sensitivity of the thermal and structural response to the lower boundary.

The reference column (Sect. 4.2) is 8 m deep with its base held at the mean
deep temperature of the untreated street, which represents lateral conduction
to the untreated ground around a street of finite width.  Two alternatives
bound that choice: an adiabatic base at 8 m, which lets the whole column
equilibrate with the treated surface (an infinitely wide treated area, the
upper bound of the deep cooling), and a 15 m column anchored in the same way
(a weaker anchor).  For three sites the untreated (0.14) and treated (0.45)
columns are run under each option, and the frost depth, the temperatures at
three depths and the service life under the reference traffic are compared.

usage: python 22_boundary_test.py [--workers 3]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import column, driver, structural  # noqa: E402
from alrb.config import RES, SITE  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--workers", type=int, default=3)
_A = _ap.parse_args()

KEYS = ("goodridge", "durham", "tucson")
ALBS = (0.14, 0.45)
Z0M = 0.0021
TAG = "v3"


def summary(r, key, n_ref):
    idx = r.index
    fr = pd.Series(r.frost, index=idx)
    daily = fr.resample("D").max()
    season = (daily.index - pd.Timedelta(days=213)).year
    winters = daily.groupby(season).max()
    keep = (winters.index >= idx[0].year) & (winters.index < idx[-1].year)
    fdays = (daily > 0).groupby(season).sum()
    st = structural.hourly_state(r.T, r.lice, r.depths)
    w = structural.traffic_weights(idx, SITE[key].tz)
    lv = structural.lives(st, w, n_ref)
    depths = np.asarray(r.depths)
    out = dict(frost_max_mean=float(winters[keep].mean()) if keep.any() else 0.0,
               frozen_days_mean=float(fdays[keep].mean()) if keep.any() else 0.0,
               life_fatigue=float(lv[0]), life_subgrade=float(lv[1]), life=float(min(lv[0], lv[1])),
               E_sg_mean=float(np.mean(st.E[:, 4])))
    for d in (0.05, 0.85, 3.0):
        q = int(np.argmin(np.abs(depths - d)))
        out[f"T_{d:.2f}"] = float(np.mean(r.T[:, q]))
    out["depths"] = [float(x) for x in depths]
    out["T_profile"] = [float(x) for x in r.T.mean(0)]
    # mean seasonal course of the daily maximum frost depth, by day of the frost season (from 1 August)
    doy = ((daily.index - pd.Timedelta(days=212)).dayofyear).to_numpy()
    out["frost_season"] = [float(daily[doy == d].mean()) if np.any(doy == d) else 0.0 for d in range(1, 366)]
    return out


def one(key):
    n_ref = json.loads((RES / f"structural_{TAG}.json").read_text())[key]["n_ref"]
    df = driver.load_forcing(key)
    mask = driver.snow_mask_pavement(df, key, days_after=1.0)
    sub = driver.substeps(df, key, snow_mask=mask, lw_col="lw")
    res = {}
    for label, zbot, anchored in (("anchored8", 8.0, True), ("adiabatic8", 8.0, False), ("anchored15", 15.0, True)):
        pre = driver.prepare(column.pavement_structure(zbot=zbot), driver.REC_DEPTHS_PAV)
        t_bot = None
        runs = {}
        r_ad = driver.spun_up_run(pre, sub, 0.14, spin_years=4, z0m=Z0M)
        if anchored:
            t_bot = float(np.mean(r_ad.T[:, -1]))
        for a in ALBS:
            if not anchored and a == 0.14:
                r = r_ad
            else:
                r = driver.spun_up_run(pre, sub, a, spin_years=4, z0m=Z0M, t_bot=t_bot)
            runs[f"{a:.2f}"] = summary(r, key, n_ref)
        runs["t_bot"] = t_bot
        res[label] = runs
        print(key, label, {a: (round(v["frost_max_mean"], 2), round(v["life"], 2))
                           for a, v in runs.items() if a != "t_bot"}, flush=True)
    return key, res


if __name__ == "__main__":
    out = {}
    with ProcessPoolExecutor(_A.workers) as ex:
        for key, res in ex.map(one, KEYS):
            out[key] = res
    (RES / "boundary_test_v3.json").write_text(json.dumps(out, indent=1))
    for key, res in out.items():
        for label, runs in res.items():
            u, t = runs["0.14"], runs["0.45"]
            print(f"{key:10s} {label:11s} frost {u['frost_max_mean']:.2f}->{t['frost_max_mean']:.2f} m  "
                  f"dT(0.85) {t['T_0.85'] - u['T_0.85']:+.2f}  dT(3) {t['T_3.00'] - u['T_3.00']:+.2f}  "
                  f"life ratio {t['life'] / u['life']:.3f}")

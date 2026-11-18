"""Pavement column at every site for the albedo grid (8-yr observed record),
with snow masking of the plowed surface.  Stores compact hourly series for
the structural and radiative stages, and per-run summaries.

usage: python 30_pavement_sweep.py TAG [--hmult 1] [--snow 1] [--z0m 0.0021] [--lw lw_ceres_f]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import column, driver  # noqa: E402
from alrb.config import ALBEDO_GRID, RAW, RES, SITES  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("tag")
_ap.add_argument("--hmult", type=float, default=1.0)
_ap.add_argument("--snow", type=float, default=1.0)
_ap.add_argument("--z0m", type=float, default=0.0021)
_ap.add_argument("--lw", default="lw_ceres_f")
_ap.add_argument("--albedos", default="")
_ap.add_argument("--workers", type=int, default=4)
_ap.add_argument("--deep", default="dirichlet", choices=["dirichlet", "adiabatic"])
_ap.add_argument("--zbot", type=float, default=8.0)
_A = _ap.parse_args()
TAG, HMULT, SNOW_DAYS, Z0M, LW = _A.tag, _A.hmult, _A.snow, _A.z0m, _A.lw
GRID = [float(x) for x in _A.albedos.split(",")] if _A.albedos else ALBEDO_GRID
OUT = RAW / "sweep" / TAG


def summarise(r, idx):
    ts = pd.Series(r.ts, index=idx)
    fr = pd.Series(r.frost, index=idx)
    winters = fr.groupby((fr.index - pd.Timedelta(days=213)).year).max()
    winters = winters[(winters.index >= idx[0].year) & (winters.index < idx[-1].year)]
    frozen_days = (fr.resample("D").max() > 0).groupby((fr.resample("D").max().index - pd.Timedelta(days=213)).year).sum()
    frozen_days = frozen_days[(frozen_days.index >= idx[0].year) & (frozen_days.index < idx[-1].year)]
    ann_max = ts.groupby(ts.index.year).max()
    return dict(ts_mean=float(ts.mean()), ts_max_mean=float(ann_max.mean()), ts_min=float(ts.min()),
                T_mean={f"{d:.3f}": float(v) for d, v in zip(r.depths, r.T.mean(0))},
                frost_max_mean=float(winters.mean()) if len(winters) else 0.0,
                frost_max_max=float(winters.max()) if len(winters) else 0.0,
                frozen_days_mean=float(frozen_days.mean()) if len(frozen_days) else 0.0,
                nfail=int(r.nfail))


def one_site(site):
    df = driver.load_forcing(site.key)
    mask = driver.snow_mask_pavement(df, site.key, days_after=SNOW_DAYS)
    sub = driver.substeps(df, site.key, snow_mask=mask, lw_col=LW)
    pre = driver.prepare(column.pavement_structure(zbot=_A.zbot), driver.REC_DEPTHS_PAV)
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {}
    t_bot = None
    if _A.deep == "dirichlet":
        # deep equilibrium of the untreated street (adiabatic base), then every albedo is anchored
        # to it at z_bot: lateral conduction to the untreated surroundings of a ~10 m wide street
        r0 = driver.spun_up_run(pre, sub, 0.14, spin_years=4, z0m=Z0M, hmult=HMULT)
        t_bot = float(np.mean(r0.T[:, -1]))
    for a in GRID:
        r = driver.spun_up_run(pre, sub, a, spin_years=4, z0m=Z0M, hmult=HMULT, t_bot=t_bot)
        np.savez_compressed(OUT / f"{site.key}_a{a:.2f}.npz", ts=r.ts.astype(np.float32),
                            ts_end=r.ts_end.astype(np.float32), T=r.T.astype(np.float32),
                            lice=r.lice.astype(np.float32), frost=r.frost.astype(np.float32),
                            flux=r.flux.astype(np.float32), w=r.w.astype(np.float32), depths=r.depths,
                            t0=str(r.index[0]), n=len(r.index))
        summ[f"{a:.2f}"] = summarise(r, r.index)
        summ[f"{a:.2f}"]["t_bot"] = t_bot
    np.save(OUT / f"{site.key}_snowmask.npy", mask.astype(np.float32))
    return site.key, summ


if __name__ == "__main__":
    out = {}
    with ProcessPoolExecutor(_A.workers) as ex:
        for key, summ in ex.map(one_site, SITES):
            out[key] = summ
            print(key, "done", flush=True)
    (RES / f"sweep_{TAG}.json").write_text(json.dumps(dict(hmult=HMULT, snow_days=SNOW_DAYS, z0m=Z0M, lw=LW,
                                                           sites=out), indent=1))

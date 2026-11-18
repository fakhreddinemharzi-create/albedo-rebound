"""Fifty-year runs (1 May 2025 - 30 April 2075) under projected climates:
nine CMIP6 models x SSP2-4.5 / SSP5-8.5 delta changes applied to the
observed record, untreated and treated (reference treatment, decaying and
renewed every 3 years from 1 May 2025), plus a stationary-climate control.
Per run: annual thermal summaries, annual fatigue and subgrade damage at the
site's reference traffic, annual kernel.

usage: python 50_climate_change.py [tag_structural] [workers]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import albedo_path, climate, column, damage, driver, kernel, structural  # noqa: E402
from alrb.config import RAW, RES, SITES, SITE, REF  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "v3"
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
Z0M, LW = 0.0021, "lw"
OUT = RAW / "climate_runs"


def annual(x, idx, fun="mean"):
    s = pd.Series(x, index=idx)
    y = ((idx - pd.Timestamp("2025-05-01", tz="UTC")).days // 365.25).astype(int)
    g = s.groupby(y)
    return getattr(g, fun)().to_numpy()


def job(args):
    key, model, ssp = args
    dest = OUT / f"{key}_{model}_{ssp}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    s = SITE[key]
    obs = driver.load_forcing(key)
    deltas = None if model == "none" else climate.monthly_deltas(model, ssp, key)
    if model != "none" and deltas is None:
        return None
    df = climate.projected_forcing(obs, key, deltas)
    mask = driver.snow_mask_pavement(df, key, days_after=1.0)
    sub = driver.substeps(df, key, snow_mask=mask, lw_col=LW)
    pre = driver.prepare(column.pavement_structure(zbot=8.0), driver.REC_DEPTHS_PAV)
    js = json.loads((RES / f"structural_{TAG}.json").read_text())[key]
    n_ref = js["n_ref"]
    w = structural.traffic_weights(df.index, s.tz)
    res = dict(site=key, model=model, ssp=ssp)
    K, _, _, kt, mu = kernel.hourly_kernel(df)
    res["kernel_annual"] = (annual(K * (1 - mask), df.index) / 100.0).tolist()
    res["ta_annual"] = annual(df.ta.to_numpy(), df.index).tolist()
    t_bot = None
    for lab, alb in (("untreated", REF.alpha0),
                     ("treated", albedo_path.sawtooth(df.index, sub.nsub, s.tz, "2025-05-01", REF.alpha0, REF.alphaf,
                                                      REF.tau, REF.Tr))):
        r = driver.spun_up_run(pre, sub, alb, spin_years=4, z0m=Z0M, t_bot=t_bot)
        if lab == "untreated":
            # the treated street is anchored to the (warming) deep ground of its untreated surroundings
            deep = pd.Series(r.T[:, -1]).rolling(8760, min_periods=1, center=True).mean().to_numpy()
            t_bot = np.repeat(deep, sub.nsub)
        st = structural.hourly_state(r.T, r.lice, r.depths)
        n_h = w * n_ref
        # annual Miner damage (fatigue, subgrade)
        et, ev, E2 = st.strains[:, 0], st.strains[:, 1], st.E[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            df_h = np.where(et > 0, n_h / (0.0795 * np.maximum(et, 1e-12) ** -3.291 * (E2 * 1e6 / 6894.757) ** -0.854), 0.0)
            dd_h = np.where(ev > 0, n_h / (1.365e-9 * np.maximum(ev, 1e-12) ** -4.477), 0.0)
        res[lab] = dict(ts_mean=annual(r.ts, df.index).tolist(), ts_max=annual(r.ts, df.index, "max").tolist(),
                        frost_max=annual(r.frost, df.index, "max").tolist(),
                        frozen_hours=annual((r.frost > 0).astype(float), df.index, "sum").tolist(),
                        dfat=annual(df_h, df.index, "sum").tolist(), dsg=annual(dd_h, df.index, "sum").tolist(),
                        T085=annual(r.T[:, 11], df.index).tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s.key, "none", "stationary") for s in SITES]
    jobs += [(s.key, m, ssp) for ssp in ("ssp245", "ssp585") for m in climate.MODELS for s in SITES]
    n = 0
    with ProcessPoolExecutor(WORKERS) as ex:
        for r in ex.map(job, jobs):
            n += 1
            if r is None:
                continue
            u, t = r["untreated"], r["treated"]
            print(f"[{n}/{len(jobs)}] {r['site']:12s} {r['model']:14s} {r['ssp']:10s} Ta {np.mean(r['ta_annual'][:10]):5.1f}->"
                  f"{np.mean(r['ta_annual'][-10:]):5.1f} | frost U {np.mean(u['frost_max'][:10]):.2f}->{np.mean(u['frost_max'][-10:]):.2f} "
                  f"| Dfat U {np.sum(u['dfat']):.2f} T {np.sum(t['dfat']):.2f}", flush=True)

"""Calibration (2023) and out-of-sample evaluation (2022, 2024) of the
surface energy balance against GOES-R ABI land-surface temperature at the
ten USCRN sites, with the column configured as the natural (grass)
surface around each station.

Two parameters are calibrated per site on 2023 only: the momentum
roughness length and the capacity of the root-zone water store.  The
snow-free albedo is not calibrated (skin temperature alone cannot separate
it from the turbulent exchange); it is the median CERES SYN1deg all-sky
surface albedo of the site's grid cell over snow-free daylight hours.  Everything else (radiation, conduction, stability, the
long-wave scheme) is shared with the pavement configuration.
Satellite retrievals used: DQF <= 1 (high or medium quality, which implies
clear or probably clear), excluding hours with snow on the ground
(MERRA-2 snow depth > 1 cm in the preceding 72 h).
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import column, driver, io_goes  # noqa: E402
from alrb.config import RAW, RES, SITES, CAL_YEAR  # noqa: E402

GOES = RAW / "goes"


def goes_obs(site):
    fs = sorted(GOES.glob(f"lst_{site.goes}_*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[(d.site == site.key) & (d.dqf_c <= 1) & d.lst_c.notna()]
    d = d[(d.time >= "2022-01-01") & (d.time < "2025-01-01")]
    # scan time -> the model state at the end of the hour that contains it
    stamp = d.time.dt.ceil("h")
    s = pd.Series(d.lst_c.to_numpy() - 273.15, index=stamp)
    s = s[~s.index.duplicated()]
    return s


def snow_free(df, s_index):
    snow = (df["snodp"] > 0.01)
    snow72 = snow.rolling(72, min_periods=1).max().astype(bool)
    return ~snow72.reindex(s_index).fillna(True).to_numpy()


LW = sys.argv[1] if len(sys.argv) > 1 else "lw"


class Problem:
    def __init__(self, site, years=(2022, 2023, 2024)):
        self.site = site
        df = driver.load_forcing(site.key)
        sf = (df["snodp"] < 0.001) & (df["g0"] > 200) & df["alb_ceres"].between(0.05, 0.5)
        self.alb = float(np.clip(df.loc[sf, "alb_ceres"].median(), 0.10, 0.35))
        self.df = df.loc[f"{years[0]}-01-01 01:00":f"{years[-1] + 1}-01-01 00:00"]
        snow = (self.df["snodp"] > 0.01).to_numpy().astype(float)
        self.sub = driver.substeps(self.df, site.key, snow_mask=snow, lw_col="lw" if LW == "lw" else "lw_ceres_f")
        self.obs = goes_obs(site)
        keep = snow_free(self.df, self.obs.index)
        self.obs = self.obs[keep]
        self.pos = self.df.index.get_indexer(self.obs.index)
        ok = self.pos >= 0
        self.obs = self.obs[ok]
        self.pos = self.pos[ok]
        self.year = self.obs.index.year.to_numpy()
        self.pre = driver.prepare(column.natural_structure(), driver.REC_DEPTHS_PAV, frost_layers=(0, 1, 2))

    def run(self, x, years_out=None):
        lz0, wcap = x
        res = driver.spun_up_run(self.pre, self.sub, self.alb, spin_years=1, z0m=10 ** lz0, surface=1, wcap=wcap,
                                 eps=0.97, alb_snow=0.70, w0=0.5 * wcap)
        return res

    def cost(self, x, year=CAL_YEAR):
        lz0, wcap = x
        if not (-3.3 <= lz0 <= -0.7 and 20 <= wcap <= 500):
            return 1e3
        res = self.run(x)
        mod = res.ts_end[self.pos]
        sel = self.year == year
        return float(np.sqrt(np.mean((mod[sel] - self.obs.to_numpy()[sel]) ** 2)))


def stats(mod, obs):
    e = mod - obs
    return dict(n=int(len(e)), rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)),
                mae=float(np.mean(np.abs(e))), r=float(np.corrcoef(mod, obs)[0, 1]))


def one(site):
    P = Problem(site)
    x0 = np.array([np.log10(0.02), 150.0])
    best = minimize(P.cost, x0, method="Nelder-Mead",
                    options=dict(xatol=1e-3, fatol=2e-3, maxfev=45, initial_simplex=np.array(
                        [x0, [np.log10(0.06), 150.0], [np.log10(0.02), 320.0]])))
    x = best.x
    res = P.run(x)
    mod = res.ts_end[P.pos]
    obs = P.obs.to_numpy()
    out = dict(params=dict(albedo=P.alb, z0m=float(10 ** x[0]), wcap=float(x[1])), nfev=int(best.nfev), lw=LW,
               vza=float(io_goes.view_zenith(site.lat, site.lon, io_goes.LON0[site.goes])))
    for y in (2022, 2023, 2024):
        sel = P.year == y
        if sel.sum() > 50:
            out[str(y)] = stats(mod[sel], obs[sel])
            # day / night split
            hr_local = (P.obs.index[sel] + pd.Timedelta(hours=site.tz)).hour
            day = (hr_local >= 9) & (hr_local <= 17)
            out[str(y)]["day"] = stats(mod[sel][day], obs[sel][day])
            out[str(y)]["night"] = stats(mod[sel][~day], obs[sel][~day])
    val = P.year != CAL_YEAR
    out["validation"] = stats(mod[val], obs[val])
    # diurnal composites (local hour) for the figure, validation years only
    hl = (P.obs.index + pd.Timedelta(hours=site.tz)).hour
    comp = pd.DataFrame(dict(h=hl[val], m=mod[val], o=obs[val])).groupby("h").mean()
    out["diurnal"] = dict(hour=comp.index.tolist(), model=comp.m.round(3).tolist(), obs=comp.o.round(3).tolist())
    pd.DataFrame(dict(time=P.obs.index, obs=obs, model=mod)).to_parquet(RAW / "goes" / f"pairs_{site.key}_{LW}.parquet")
    return site.key, out


if __name__ == "__main__":
    allr = {}
    with ProcessPoolExecutor(2) as ex:
        for key, r in ex.map(one, SITES):
            allr[key] = r
            v = r["validation"]
            print(f"{key:13s} alb {r['params']['albedo']:.3f} z0 {r['params']['z0m']:.4f} W {r['params']['wcap']:.0f} | "
                  f"cal {r[str(CAL_YEAR)]['rmse']:.2f} | val rmse {v['rmse']:.2f} bias {v['bias']:+.2f} r {v['r']:.3f} n {v['n']}",
                  flush=True)
    val = [r["validation"] for r in allr.values()]
    n = np.array([v["n"] for v in val])
    pooled = dict(rmse=float(np.sqrt(np.sum(n * np.array([v["rmse"] for v in val]) ** 2) / n.sum())),
                  bias=float(np.sum(n * np.array([v["bias"] for v in val])) / n.sum()), n=int(n.sum()))
    print("pooled validation", pooled)
    (RES / f"natural_validation_{LW}.json").write_text(json.dumps(dict(sites=allr, pooled_validation=pooled), indent=1))

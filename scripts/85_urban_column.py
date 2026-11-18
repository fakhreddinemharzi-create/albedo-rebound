"""Thermal and structural response of streets in urban canyons and in a
warmer urban atmosphere (revision; Sect. 6.9).

The reference column (30_pavement_sweep.py, 40_structural.py) is run with the
forcing of a street floor (alrb.urban): the shortwave the floor absorbs in a
canyon of aspect ratio h (radiosity, east-west street, walls of albedo 0.30),
the longwave from the sky and the walls, the canyon wind and an urban
heat-island offset of the air temperature.  For every configuration and site
the deep boundary is anchored, as in the reference, to the untreated street
of the same configuration; the service life at every albedo of the grid
follows from the hourly states exactly as in 40_structural.py.

Configurations (name: aspect ratio, wind factor, heat-island offset):
  open      0     1                 0     (reference street, reduced albedo grid)
  h0.25     0.25  1                 0     canyon geometry only
  h0.5      0.5   1                 0
  h1        1     1                 0
  h2        2     1                 0
  uhi3      0     1                 3 K   open street in a warmer city
  urban0.5  0.5   (2/pi)exp(-h/4)   3 K   street of an open mid-rise district
  urban1    1     (2/pi)exp(-h/4)   3 K   street of a compact district

usage: python 85_urban_column.py [workers]
Writes results/urban_column_v3.json.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import column, driver, structural, urban  # noqa: E402
from alrb.config import RES, SITE, SITES  # noqa: E402

WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ALB = [0.14, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
CONFIGS = {
    "open": (0.0, 1.0, 0.0),
    "h0.25": (0.25, 1.0, 0.0),
    "h0.5": (0.5, 1.0, 0.0),
    "h1": (1.0, 1.0, 0.0),
    "h2": (2.0, 1.0, 0.0),
    "uhi3": (0.0, 1.0, 3.0),
    "urban0.5": (0.5, "teb", 3.0),
    "urban1": (1.0, "teb", 3.0),
}
Z0M, ZBOT = 0.0021, 8.0


def lives_of(r, key):
    """Fatigue and subgrade lives (years at one repetition per day) from the
    hourly states of a run, as 40_structural.py does from the stored sweep."""
    st = structural.hourly_state(r.T.astype(np.float32), r.lice.astype(np.float32), r.depths)
    idx = pd.date_range(r.index[0], periods=len(r.index), freq="h")
    w = structural.traffic_weights(idx, SITE[key].tz)
    lv = structural.lives(st, w, 1.0)
    return float(lv[0]), float(lv[1]), float(np.mean(0.5 * (st.E[:, 0] + st.E[:, 1])))


def one(job):
    key, name = job
    h, wind, uhi = CONFIGS[name]
    wind = urban.teb_wind(h) if wind == "teb" else wind
    df = driver.load_forcing(key)
    mask = driver.snow_mask_pavement(df, key, days_after=1.0)
    sub = driver.substeps(df, key, snow_mask=mask, lw_col="lw")
    pre = driver.prepare(column.pavement_structure(zbot=ZBOT), driver.REC_DEPTHS_PAV)
    cf = urban.CanyonForcing(df, key, sub, h, axis_deg=90.0)
    f0 = cf.apply(0.14, uhi=uhi, wind=wind)
    r0 = driver.spun_up_run(pre, f0, 0.14, spin_years=4, z0m=Z0M, hmult=1.0)
    t_bot = float(np.mean(r0.T[:, -1]))
    out = dict(h=h, wind=wind, uhi=uhi, t_bot=t_bot, albedo={})
    for a in ALB:
        f = cf.apply(a, uhi=uhi, wind=wind)
        r = driver.spun_up_run(pre, f, a, spin_years=4, z0m=Z0M, hmult=1.0, t_bot=t_bot)
        lf, ld, eac = lives_of(r, key)
        fr = pd.Series(r.frost, index=r.index)
        winters = fr.groupby((fr.index - pd.Timedelta(days=213)).year).max()
        winters = winters[(winters.index >= r.index[0].year) & (winters.index < r.index[-1].year)]
        out["albedo"][f"{a:.2f}"] = dict(life_fatigue=lf, life_subgrade=ld, ts_mean=float(np.mean(r.ts)),
                                         ts_max_mean=float(pd.Series(r.ts, index=r.index).groupby(r.index.year).max().mean()),
                                         sw_abs=float(np.mean(r.flux[:, 4])), E_ac_mean=eac,
                                         frost_max_mean=float(winters.mean()) if len(winters) else 0.0,
                                         nfail=int(r.nfail))
    return key, name, out


if __name__ == "__main__":
    jobs = [(s.key, n) for n in CONFIGS for s in SITES]
    res = {s.key: {} for s in SITES}
    with ProcessPoolExecutor(WORKERS) as ex:
        for key, name, out in ex.map(one, jobs):
            res[key][name] = out
            a0, a1 = out["albedo"]["0.14"], out["albedo"]["0.45"]
            l0 = min(a0["life_fatigue"], a0["life_subgrade"])
            l1 = min(a1["life_fatigue"], a1["life_subgrade"])
            print(f"{name:9s} {key:13s} Ts {a0['ts_mean']:6.2f} -> {a1['ts_mean']:6.2f} (dTs {a1['ts_mean'] - a0['ts_mean']:+.2f})"
                  f" | life ratio 0.45/0.14 {l1 / l0:.4f} | SWabs {a0['sw_abs']:6.1f} | nfail {a1['nfail']}", flush=True)
    (RES / "urban_column_v3.json").write_text(json.dumps(dict(albedos=ALB, configs={k: list(map(str, v)) for k, v in CONFIGS.items()},
                                                              sites=res), indent=1))

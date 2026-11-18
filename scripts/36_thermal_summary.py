"""Thermal and frost response to albedo at every site (complete years and
winters of the 2017-2024 record)."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import summaries  # noqa: E402
from alrb.config import ALBEDO_GRID, RES, SITES  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "v3"


def one(site):
    out = dict(albedo={})
    series = {}
    for a in ALBEDO_GRID:
        out["albedo"][f"{a:.2f}"] = summaries.summarise(site.key, a, TAG)
        if a in (0.14, 0.45):
            z, idx = summaries.load_run(site.key, a, TAG)
            li = summaries.local_index(idx, site.key)
            series[a] = (z, li)
    (z0, li), (z1, _) = series[0.14], series[0.45]
    depths = z0["depths"]
    yrs = (li.year >= 2017) & (li.year <= 2024)
    dT = (z1["T"] - z0["T"])[yrs]
    dts = (z1["ts"] - z0["ts"])[yrs]
    mon = li[yrs].month
    hr = li[yrs].hour
    out["depths"] = depths.tolist()
    out["dT_depth_mean"] = dT.mean(0).tolist()
    out["dT_depth_p95"] = np.percentile(np.abs(dT), 95, axis=0).tolist()
    out["dts_monthly"] = pd.Series(dts).groupby(mon).mean().tolist()
    i25 = int(np.argmin(np.abs(depths - 0.025)))
    i85 = int(np.argmin(np.abs(depths - 0.85)))
    out["dT25_monthly"] = pd.Series(dT[:, i25]).groupby(mon).mean().tolist()
    out["dT85_monthly"] = pd.Series(dT[:, i85]).groupby(mon).mean().tolist()
    jul = mon == 7
    jan = mon == 1
    out["dts_diurnal_jul"] = pd.Series(dts[jul]).groupby(hr[jul]).mean().tolist()
    out["dts_diurnal_jan"] = pd.Series(dts[jan]).groupby(hr[jan]).mean().tolist()
    ts0 = pd.Series(z0["ts"][yrs])
    out["ts0_diurnal_jul"] = ts0[jul].groupby(hr[jul]).mean().tolist()
    # hourly bound-layer temperature distributions (for the stiffness discussion)
    out["T25_quantiles_014"] = np.percentile(z0["T"][yrs, i25], [1, 5, 25, 50, 75, 95, 99]).tolist()
    out["T25_quantiles_045"] = np.percentile(z1["T"][yrs, i25], [1, 5, 25, 50, 75, 95, 99]).tolist()
    # space-time field for the 3-D rendering: day-of-year x depth mean dT, and frost depth climatology
    doy = li[yrs].dayofyear
    df = pd.DataFrame(dT, columns=[f"{d:.3f}" for d in depths])
    df["doy"] = np.minimum(doy, 365)
    clim = df.groupby("doy").mean()
    out["dT_doy_depth"] = clim.to_numpy().tolist()
    f0 = pd.Series(z0["frost"][yrs]).groupby(np.minimum(doy, 365)).mean()
    f1 = pd.Series(z1["frost"][yrs]).groupby(np.minimum(doy, 365)).mean()
    out["frost_doy_014"] = f0.tolist()
    out["frost_doy_045"] = f1.tolist()
    return site.key, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(4) as ex:
        for k, o in ex.map(one, SITES):
            res[k] = o
            a0, a1 = o["albedo"]["0.14"], o["albedo"]["0.45"]
            print(f"{k:13s} dTs {a1['ts_mean'] - a0['ts_mean']:+.2f} dTsmax {a1['ts_max'] - a0['ts_max']:+.2f} "
                  f"dT085 {o['dT_depth_mean'][11]:+.2f} frost {a0['frost_max']:.2f}->{a1['frost_max']:.2f} "
                  f"frozen {a0['frozen_days']:.0f}->{a1['frozen_days']:.0f} Tsmin {a0['ts_min']:.1f}->{a1['ts_min']:.1f}",
                  flush=True)
    (RES / f"thermal_{TAG}.json").write_text(json.dumps(res))

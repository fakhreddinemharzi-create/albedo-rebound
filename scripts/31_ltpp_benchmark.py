"""Pavement-specific check of the untreated column against the empirical
pavement-temperature models fitted to the LTPP Seasonal Monitoring Program
(Mohseni, 1998, as implemented in LTPPBind) and the earlier SHRP model:

  SHRP high (20 mm):  T = (T_air - 0.00618 Lat^2 + 0.2289 Lat + 42.2) 0.9545 - 17.78
  LTPP high (depth H mm, 50 % reliability):
                      T = 54.32 + 0.78 T_air - 0.0025 Lat^2 - 15.14 log10(H + 25)
  LTPP low  (depth H mm, 50 % reliability):
                      T = -1.56 + 0.72 T_air - 0.004 Lat^2 + 6.26 log10(H + 25)

T_air (high) is the mean over years of the annual maximum 7-day mean of the
daily maximum air temperature; T_air (low) the mean annual minimum air
temperature.  The modelled counterparts use the same statistics of the
simulated pavement temperature at 20 mm (high) and at the surface (low).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import driver, summaries  # noqa: E402
from alrb.config import RES, SITES  # noqa: E402


def seven_day_high(daily_max: pd.Series):
    s = summaries.full_years(daily_max)
    roll = s.rolling(7, min_periods=6).mean()
    return roll.groupby(roll.index.year).max()


def main(tag="ref"):
    out = {}
    for s in SITES:
        df = driver.load_forcing(s.key)
        li = summaries.local_index(df.index, s.key)
        ta = pd.Series(df.ta.to_numpy(), index=li)
        z, idx = summaries.load_run(s.key, 0.14, tag)
        # (depths are those of the cells nearest to the requested recording depths)
        depths = z["depths"]
        T = z["T"]
        # temperature at 20 mm by linear interpolation between recorded depths
        t20 = T[:, 0] + (T[:, 1] - T[:, 0]) * (0.020 - depths[0]) / (depths[1] - depths[0])
        t20 = pd.Series(t20, index=li)
        tsurf = pd.Series(z["ts"], index=li)
        air7 = seven_day_high(ta.resample("D").max())
        pav7 = seven_day_high(t20.resample("D").max())
        airmin = summaries.full_years(ta).groupby(summaries.full_years(ta).index.year).min()
        pavmin = summaries.full_years(tsurf).groupby(summaries.full_years(tsurf).index.year).min()
        lat = s.lat
        shrp = (air7.mean() - 0.00618 * lat ** 2 + 0.2289 * lat + 42.2) * 0.9545 - 17.78
        ltpp_hi = 54.32 + 0.78 * air7.mean() - 0.0025 * lat ** 2 - 15.14 * np.log10(20 + 25)
        ltpp_lo = -1.56 + 0.72 * airmin.mean() - 0.004 * lat ** 2 + 6.26 * np.log10(0 + 25)
        out[s.key] = dict(air_7d_high=float(air7.mean()), air_low=float(airmin.mean()),
                          model_20mm_7d_high=float(pav7.mean()), shrp_20mm=float(shrp), ltpp_20mm=float(ltpp_hi),
                          model_surface_low=float(pavmin.mean()), ltpp_surface_low=float(ltpp_lo),
                          model_20mm_7d_high_sd=float(pav7.std()), air_7d_high_sd=float(air7.std()))
        o = out[s.key]
        print(f"{s.key:13s} air7 {o['air_7d_high']:5.1f} | 20mm model {o['model_20mm_7d_high']:5.1f} "
              f"LTPP {o['ltpp_20mm']:5.1f} SHRP {o['shrp_20mm']:5.1f} | low: air {o['air_low']:6.1f} "
              f"model {o['model_surface_low']:6.1f} LTPP {o['ltpp_surface_low']:6.1f}")
    d_hi = np.array([o["model_20mm_7d_high"] - o["ltpp_20mm"] for o in out.values()])
    d_sh = np.array([o["model_20mm_7d_high"] - o["shrp_20mm"] for o in out.values()])
    d_lo = np.array([o["model_surface_low"] - o["ltpp_surface_low"] for o in out.values()])
    stats = dict(high_vs_ltpp_bias=float(d_hi.mean()), high_vs_ltpp_rmse=float(np.sqrt((d_hi ** 2).mean())),
                 high_vs_shrp_bias=float(d_sh.mean()), high_vs_shrp_rmse=float(np.sqrt((d_sh ** 2).mean())),
                 low_vs_ltpp_bias=float(d_lo.mean()), low_vs_ltpp_rmse=float(np.sqrt((d_lo ** 2).mean())))
    print(stats)
    (RES / f"ltpp_benchmark_{tag}.json").write_text(json.dumps(dict(sites=out, stats=stats), indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ref")

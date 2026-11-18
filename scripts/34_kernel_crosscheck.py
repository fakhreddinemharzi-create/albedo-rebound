"""Site kernels: hourly APRP from the measured irradiance versus (i) the
simple transmittance kernel, (ii) clear-sky, (iii) the same APRP applied to
monthly means, and (iv) monthly APRP kernels from nine CMIP6 historical
simulations (1995-2014) at the site grid cells."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import driver, kernel  # noqa: E402
from alrb.config import RAW, RES, SITES  # noqa: E402

MODELS = ["ACCESS-CM2", "CanESM5", "EC-Earth3", "GFDL-ESM4", "INM-CM5-0", "IPSL-CM6A-LR",
          "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"]


def cmip_kernels():
    out = {}
    for m in MODELS:
        try:
            v = {k: pd.read_parquet(RAW / "cmip6" / f"{m}_historical_{k}.parquet") for k in ["rsdt", "rsut", "rsds", "rsus"]}
        except FileNotFoundError:
            continue
        K, mu, gam = kernel.aprp_monthly(v["rsdt"], v["rsut"], v["rsds"], v["rsus"])
        out[m] = dict(K=K.mean().to_dict(), mu=mu.mean().to_dict(), gamma=gam.mean().to_dict(),
                      Kmon=K.groupby(K.index.str[5:7]).mean().to_dict())
    return out


if __name__ == "__main__":
    cm = cmip_kernels()
    res = {}
    for s in SITES:
        df = driver.load_forcing(s.key)
        yrs = (df.index.year >= 2017) & (df.index.year <= 2024)
        K, Ks, Kc, kt, mu = kernel.hourly_kernel(df)
        mask = driver.snow_mask_pavement(df, s.key, days_after=1.0)
        d = pd.DataFrame(dict(g0=df.g0, sw=df.sw, mu=mu, K=K), index=df.index)[yrs]
        mo = d.resample("MS").mean()
        Kmon_self = (-mo.g0 * (mo.sw / mo.g0) ** 2 / mo.mu)
        mon_cycle = pd.Series(K[yrs], index=df.index[yrs]).groupby(df.index[yrs].month).mean()
        cmK = [cm[m]["K"][s.key] for m in cm]
        res[s.key] = dict(
            K_aprp_hourly=float(K[yrs].mean() / 100), K_aprp_snowmasked=float((K * (1 - mask))[yrs].mean() / 100),
            K_simple=float(Ks[yrs].mean() / 100), K_clear=float(Kc[yrs].mean() / 100),
            K_aprp_monthlymeans=float(Kmon_self.mean() / 100),
            K_cmip6_mean=float(np.mean(cmK) / 100), K_cmip6_min=float(np.min(cmK) / 100),
            K_cmip6_max=float(np.max(cmK) / 100), kt=float(d.sw.sum() / d.g0.sum()),
            mu_weighted=float(np.average(mu[yrs], weights=df.g0[yrs] + 1e-9)),
            snow_frac=float(mask[yrs].mean()),
            monthly_cycle=(mon_cycle / 100).round(4).tolist(),
            cmip6_monthly={m: [cm[m]["Kmon"][s.key][f"{i:02d}"] / 100 for i in range(1, 13)] for m in cm})
        r = res[s.key]
        print(f"{s.key:13s} hourly {r['K_aprp_hourly']:.3f} masked {r['K_aprp_snowmasked']:.3f} simple {r['K_simple']:.3f} "
              f"clear {r['K_clear']:.3f} monthly {r['K_aprp_monthlymeans']:.3f} | CMIP6 {r['K_cmip6_mean']:.3f} "
              f"[{r['K_cmip6_max']:.3f}, {r['K_cmip6_min']:.3f}]")
    a = np.array([[r["K_aprp_hourly"], r["K_cmip6_mean"], r["K_simple"]] for r in res.values()])
    summ = dict(rel_cmip6_minus_obs=float(np.mean(a[:, 1] / a[:, 0] - 1)), r_cmip6_obs=float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]),
                rel_simple_minus_obs=float(np.mean(a[:, 2] / a[:, 0] - 1)), models=list(cm))
    print(summ)
    (RES / "kernels.json").write_text(json.dumps(dict(sites=res, summary=summ), indent=1))

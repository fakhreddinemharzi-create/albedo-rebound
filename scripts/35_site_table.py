"""Climate descriptors of the ten sites from their own 2017-2024 record:
mean annual air temperature, freezing index (mean over complete winters of
the sum of negative daily-mean air temperatures), precipitation, mean
irradiance, clearness index, snow-masked hours, and the Koppen-Geiger class
from the monthly climatology (rules of Peel et al., 2007)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import driver, summaries  # noqa: E402
from alrb.config import RES, SITES  # noqa: E402


def koppen(T, P, lat):
    """T (C) and P (mm) monthly climatologies (Jan..Dec)."""
    T = np.asarray(T)
    P = np.asarray(P)
    MAT, MAP = T.mean(), P.sum()
    north = lat >= 0
    summer = [3, 4, 5, 6, 7, 8] if north else [9, 10, 11, 0, 1, 2]
    winter = [9, 10, 11, 0, 1, 2] if north else [3, 4, 5, 6, 7, 8]
    Ps = P[summer].sum()
    thr = 2 * MAT + (28 if Ps >= 0.7 * MAP else (0 if P[winter].sum() >= 0.7 * MAP else 14))
    Thot, Tcold = T.max(), T.min()
    if Thot < 10:
        return "E"
    if MAP < 10 * thr:
        a = "B" + ("W" if MAP < 5 * thr else "S")
        return a + ("h" if MAT >= 18 else "k")
    Psdry, Pwdry = P[summer].min(), P[winter].min()
    Pswet, Pwwet = P[summer].max(), P[winter].max()
    if Tcold >= 18:
        return "A"
    main = "C" if Tcold > 0 else "D"
    if Psdry < 40 and Psdry < Pwwet / 3:
        s2 = "s"
    elif Pwdry < Pswet / 10:
        s2 = "w"
    else:
        s2 = "f"
    nwarm = int(np.sum(T >= 10))
    if Thot >= 22:
        s3 = "a"
    elif nwarm >= 4:
        s3 = "b"
    elif main == "D" and Tcold < -38:
        s3 = "d"
    else:
        s3 = "c"
    return main + s2 + s3


if __name__ == "__main__":
    kern = json.loads((RES / "kernels.json").read_text())["sites"]
    out = {}
    for s in SITES:
        df = driver.load_forcing(s.key)
        li = summaries.local_index(df.index, s.key)
        d = pd.DataFrame(dict(ta=df.ta.to_numpy(), pr=df.pr.to_numpy(), sw=df.sw.to_numpy(), g0=df.g0.to_numpy(),
                              ws=df.ws.to_numpy(), rh=df.rh.to_numpy()), index=li)
        d = d[(d.index.year >= 2017) & (d.index.year <= 2024)]
        daily = d.ta.resample("D").mean()
        fw, w = summaries.complete_winters(daily)
        fi = (-fw.clip(upper=0)).groupby(w).sum()
        mon = d.resample("MS").agg({"ta": "mean", "pr": "sum"})
        Tm = mon.ta.groupby(mon.index.month).mean().to_numpy()
        Pm = mon.pr.groupby(mon.index.month).mean().to_numpy()
        out[s.key] = dict(name=s.name, state=s.state, lat=s.lat, lon=s.lon, elev=s.elev, wban=s.wban,
                          maat=float(d.ta.mean()), fi=float(fi.mean()), fi_max=float(fi.max()),
                          precip=float(d.pr.sum() / 8.0), sw=float(d.sw.mean()), kt=float(d.sw.sum() / d.g0.sum()),
                          ws=float(d.ws.mean()), rh=float(d.rh.mean()), koppen=koppen(Tm, Pm, s.lat),
                          T_monthly=Tm.round(2).tolist(), P_monthly=Pm.round(1).tolist(),
                          kernel=kern[s.key]["K_aprp_hourly"], kernel_masked=kern[s.key]["K_aprp_snowmasked"],
                          snow_frac=kern[s.key]["snow_frac"])
        o = out[s.key]
        print(f"{s.key:13s} {o['koppen']:4s} MAAT {o['maat']:5.1f} FI {o['fi']:6.0f} P {o['precip']:6.0f} "
              f"SW {o['sw']:5.1f} kt {o['kt']:.3f} K {o['kernel']:.3f}")
    (RES / "sites.json").write_text(json.dumps(out, indent=1))

"""Ledger of the reference treatment on streets in canyons and in a warmer
city (revision; Sect. 6.9), from the simulations of 85_urban_column.py.

For every configuration and site the radiative credit is that of the
reference ledger times the kernel-weighted urban-form factor of the street
(east-west, walls of albedo 0.30; 64_canyon.py), and the structural term
follows from the service lives simulated with the forcing of the street,
through the harmonic-mean life of Eq. (6).  The ledger is otherwise that of
62_ledger.py.  For comparison the structural term is also kept at its
open-street value, the approximation of the original Section 6.9.

Writes results/urban_ledger_v3.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import ledger  # noqa: E402
from alrb.config import RES, SITES  # noqa: E402

METRICS = ("static", "dynamic", "gtp")


def main():
    u = json.loads((RES / "urban_column_v3.json").read_text())
    cy = json.loads((RES / "canyon_v3.json").read_text())
    lg = json.loads((RES / "ledger_v3.json").read_text())
    hg = np.array(cy["h"])
    al = np.array(u["albedos"])
    out = {}
    for s in SITES:
        K1 = ledger.daily_kernel(s.key, 1.0).to_numpy()
        res = {}
        Lopen = None
        for name, cfg in u["sites"][s.key].items():
            L = np.array([min(cfg["albedo"][f"{a:.2f}"]["life_fatigue"], cfg["albedo"][f"{a:.2f}"]["life_subgrade"]) for a in al])
            L = L / L[0] * 20.0
            if name == "open":
                Lopen = L.copy()
            h = float(cfg["h"])
            psi = 1.0 if h == 0 else float(np.interp(h, hg, cy["sites"][s.key]["EW"]["psi"]))
            p = ledger.Params(psi=psi)
            r = ledger.evaluate(K1, p, lambda x, L=L: np.interp(x, al, L), metrics_list=METRICS)
            a0, a1 = cfg["albedo"]["0.14"], cfg["albedo"]["0.45"]
            res[name] = dict(h=h, wind=float(cfg["wind"]), uhi=float(cfg["uhi"]), psi=psi, L_eff=r["L_eff"],
                             life45=float(L[-1] / L[0]), dTs=a1["ts_mean"] - a0["ts_mean"],
                             dTsmax=a1["ts_max_mean"] - a0["ts_max_mean"], ts14=a0["ts_mean"], sw14=a0["sw_abs"],
                             frost14=a0["frost_max_mean"], dfrost=a1["frost_max_mean"] - a0["frost_max_mean"],
                             **{m: {k: float(r[m][k]) for k in ("G_rf", "G_co", "G_st", "G_net", "R")} for m in METRICS})
        # open-street structural term with the canyon credit (approximation of the original text)
        for name, d in res.items():
            r = ledger.evaluate(K1, ledger.Params(psi=d["psi"]), lambda x: np.interp(x, al, Lopen), metrics_list=METRICS)
            d["R_open_structure"] = {m: float(r[m]["R"]) for m in METRICS}
            d["st_factor"] = d["static"]["G_st"] / res["open"]["static"]["G_st"]
        # check: the open configuration reproduces the reference ledger
        ref = lg[s.key]["reference"]
        res["check"] = {m: abs(res["open"][m]["R"] / ref[m]["R"] - 1) for m in METRICS}
        out[s.key] = res
        print(f"{s.key:13s} check {max(res['check'].values()):.1e} | " + " | ".join(
            f"{n} dTs {d['dTs']:+.2f} st {d['st_factor']:.2f} R {d['dynamic']['R']:.2f} ({d['R_open_structure']['dynamic']:.2f})"
            for n, d in res.items() if n != "check"), flush=True)
    # validation of the floor forcing: kernel-weighted floor absorption change against the fate curve (Durham)
    import pandas as pd
    from alrb import driver, kernel, urban
    from alrb.config import SITE
    fc = json.loads((RES / "canyon_fatecurve_v3.json").read_text())
    key = "durham"
    df = driver.load_forcing(key)
    mask = driver.snow_mask_pavement(df, key, days_after=1.0)
    sub = driver.substeps(df, key, snow_mask=mask, lw_col="lw")
    K = kernel.hourly_kernel(df)[0] * (1.0 - mask)
    loc = df.index + pd.Timedelta(hours=SITE[key].tz)
    keep = np.asarray((loc >= "2017-05-01") & (loc < "2024-05-01"))
    dev = [abs(-urban.CanyonForcing(df, key, sub, h).kernel_weighted_floor_change(K, keep) / fl - 1)
           for h, fl in zip(fc["h"], fc["floor"])]
    out["validation"] = dict(site=key, max_abs_pct=100 * float(max(dev)))
    print("floor forcing against the fate curve: max deviation", f"{100 * max(dev):.3f} %")
    (RES / "urban_ledger_v3.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Summary of the fifty-year runs under projected climates (Sect. 6.9).

For every site, scenario (stationary control, SSP2-4.5, SSP5-8.5) and CMIP6
model: warming, frost depth and frozen season of the untreated and treated
street in the first and last decade, fatigue damage summed over the fifty
years (the number of structural renewals it implies), the mean kernel, and
from these the static balance of the reference treatment relative to the
stationary control:
    G_rf,cc = G_rf,ref * <K>_cc / <K>_stationary
    G_st,cc = g_st * (sum D_treated - sum D_untreated)
    R_cc    = (G_co + G_st,cc) / |G_rf,cc|
Writes results/climate_v3.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import palette as C  # noqa: E402
from alrb.climate import MODELS  # noqa: E402
from alrb.config import RAW, REF, RES  # noqa: E402

TAG = "v3"
D = RAW / "climate_runs"


def load(key, model, ssp):
    p = D / f"{key}_{model}_{ssp}.json"
    return json.loads(p.read_text()) if p.exists() else None


def metrics(r, lg_static, k_stat):
    u, t = r["untreated"], r["treated"]
    ta = np.array(r["ta_annual"])
    K = np.array(r["kernel_annual"])
    fu, ft = np.array(u["frost_max"]), np.array(t["frost_max"])
    du, dt = np.array(u["dfat"]), np.array(t["dfat"])
    su, stt = np.array(u["dsg"]), np.array(t["dsg"])
    hu, ht = np.array(u["frozen_hours"]) / 24.0, np.array(t["frozen_hours"]) / 24.0
    out = dict(ta0=float(ta[:10].mean()), ta1=float(ta[-10:].mean()), dta=float(ta[-10:].mean() - ta[:10].mean()),
               K=float(K.mean()), K0=float(K[:10].mean()), K1=float(K[-10:].mean()),
               frost_u0=float(fu[:10].mean()), frost_u1=float(fu[-10:].mean()),
               frost_t0=float(ft[:10].mean()), frost_t1=float(ft[-10:].mean()),
               fdays_u0=float(hu[:10].mean()), fdays_u1=float(hu[-10:].mean()),
               Dfat_u=float(du.sum()), Dfat_t=float(dt.sum()), Dsg_u=float(su.sum()), Dsg_t=float(stt.sum()),
               dfat_u0=float(du[:10].mean()), dfat_u1=float(du[-10:].mean()),
               benefit0=float(1 - dt[:10].sum() / du[:10].sum()), benefit1=float(1 - dt[-10:].sum() / du[-10:].sum()),
               benefit=float(1 - dt.sum() / du.sum()),
               ts_u0=float(np.mean(u["ts_mean"][:10])), ts_u1=float(np.mean(u["ts_mean"][-10:])),
               tsmax_u1=float(np.mean(u["ts_max"][-10:])),
               series=dict(frost_u=fu.tolist(), frost_t=ft.tolist(), dfat_u=du.tolist(), dfat_t=dt.tolist(),
                           ta=ta.tolist(), K=K.tolist()))
    # governing distress is fatigue at every site (Sect. 6.2); renewals = accumulated Miner damage
    g_st = REF.g_st * (out["Dfat_t"] - out["Dfat_u"])
    g_rf = lg_static["G_rf"] * out["K"] / k_stat
    out.update(G_st=g_st, G_rf=g_rf, G_co=lg_static["G_co"],
               R=(lg_static["G_co"] + g_st) / abs(g_rf), G_net=lg_static["G_co"] + g_st + g_rf)
    return out


if __name__ == "__main__":
    lg = json.loads((RES / f"ledger_{TAG}.json").read_text())
    res = dict(models=MODELS, sites={})
    nrun = 0
    for key in C.SITE_ORDER:
        st = load(key, "none", "stationary")
        if st is None:
            continue
        k_stat = float(np.mean(st["kernel_annual"]))
        lgs = lg[key]["reference"]["static"]
        site = dict(stationary=metrics(st, lgs, k_stat))
        for ssp in ("ssp245", "ssp585"):
            runs = {}
            for m in MODELS:
                r = load(key, m, ssp)
                if r is not None:
                    runs[m] = metrics(r, lgs, k_stat)
                    nrun += 1
            if not runs:
                continue
            ens = {}
            for q in ("dta", "ta1", "K", "frost_u0", "frost_u1", "frost_t0", "frost_t1", "fdays_u0", "fdays_u1",
                      "Dfat_u", "Dfat_t", "benefit0", "benefit1", "benefit", "G_st", "G_rf", "R", "G_net", "ts_u1",
                      "tsmax_u1", "dfat_u1"):
                v = np.array([runs[m][q] for m in runs])
                ens[q] = dict(med=float(np.median(v)), min=float(v.min()), max=float(v.max()))
            ens["n"] = len(runs)
            # relative changes against the stationary control
            s0 = site["stationary"]
            v = np.array([100 * (runs[m]["K"] / s0["K"] - 1) for m in runs])
            ens["dK_pct"] = dict(med=float(np.median(v)), min=float(v.min()), max=float(v.max()))
            v = np.array([100 * (runs[m]["Dfat_u"] / s0["Dfat_u"] - 1) for m in runs])
            ens["dDu_pct"] = dict(med=float(np.median(v)), min=float(v.min()), max=float(v.max()))
            v = np.array([runs[m]["R"] / s0["R"] for m in runs])
            ens["R_ratio"] = dict(med=float(np.median(v)), min=float(v.min()), max=float(v.max()))
            site[ssp] = dict(ensemble=ens, runs=runs)
        res["sites"][key] = site
        s0 = site["stationary"]
        msg = f"{key:13s} stationary R {s0['R']:.2f} G_st {s0['G_st']:+.2f}"
        for ssp in ("ssp245", "ssp585"):
            if ssp in site:
                e = site[ssp]["ensemble"]
                msg += (f" | {ssp} n={e['n']} dTa {e['dta']['med']:.1f} dK {e['dK_pct']['med']:+.1f}% "
                        f"dDu {e['dDu_pct']['med']:+.0f}% G_st {e['G_st']['med']:+.2f} R {e['R']['med']:.2f}")
        print(msg)
    res["nruns"] = nrun
    (RES / f"climate_{TAG}.json").write_text(json.dumps(res))

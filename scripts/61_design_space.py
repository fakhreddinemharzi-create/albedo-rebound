"""Design space of reflective treatments: break-even embodied carbon per
application, g_co*(tau), below which the treatment is a net sink, for each
site, metric, laid albedo and renewal interval.  Because the ledger is
linear in g_co, g_co* = -(G_rf + G_st) / (dG/dg_co) exactly.

usage: python 61_design_space.py [tag]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import ledger  # noqa: E402
from alrb.config import ALBEDO_GRID, RES, SITES, REF  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "v3"
METRICS = ["static", "dynamic", "gtp", "stock"]
TAUS = np.logspace(np.log10(0.2), np.log10(30.0), 46)
ALPHAF = [0.30, 0.45, 0.60]
TRS = [1.0, 3.0, 10.0, 50.0]


def one(site):
    Kd = ledger.daily_kernel(site.key, 1.0).to_numpy()
    js = json.loads((RES / f"structural_{TAG}.json").read_text())[site.key]
    al = np.array(ALBEDO_GRID)
    L = np.array([js["albedo"][f"{a:.2f}"]["life_ref"] for a in al])
    lf = lambda a: np.interp(a, al, L)
    out = {m: {} for m in METRICS}
    for af in ALPHAF:
        for Tr in TRS:
            g = {m: [] for m in METRICS}
            for tau in TAUS:
                p0 = ledger.Params(alphaf=af, tau=tau, Tr=Tr, g_co=0.0)
                p1 = ledger.Params(alphaf=af, tau=tau, Tr=Tr, g_co=1.0)
                r0 = ledger.evaluate(Kd, p0, lf)
                r1 = ledger.evaluate(Kd, p1, lf)
                for m in METRICS:
                    b = r1[m]["G_net"] - r0[m]["G_net"]
                    g[m].append(float(-r0[m]["G_net"] / b) if b > 0 else float("nan"))
            for m in METRICS:
                out[m][f"af{af:.2f}_Tr{Tr:g}"] = g[m]
    # reference treatment: break-even and ratio
    ref = {}
    for m in METRICS:
        p0 = ledger.Params(g_co=0.0)
        p1 = ledger.Params(g_co=1.0)
        r0 = ledger.evaluate(Kd, p0, lf)
        r1 = ledger.evaluate(Kd, p1, lf)
        b = r1[m]["G_net"] - r0[m]["G_net"]
        gstar = -r0[m]["G_net"] / b
        ref[m] = dict(g_co_star=float(gstar), ratio=float(REF.g_co / gstar))
    # break-even decay time for the reference coat (g_co = 2.70, alpha_f = 0.45), long renewal
    tstar = {}
    for m in METRICS:
        for Tr in TRS:
            gs = np.array(out[m][f"af0.45_Tr{Tr:g}"])
            ok = np.isfinite(gs)
            tstar[f"{m}_Tr{Tr:g}"] = float(np.interp(REF.g_co, gs[ok], TAUS[ok])) if ok.sum() > 2 and gs[ok].max() > REF.g_co else float("nan")
    return site.key, dict(g_star=out, ref=ref, tau_star=tstar)


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(4) as ex:
        for key, r in ex.map(one, SITES):
            res[key] = r
            print(key, {m: round(r["ref"][m]["g_co_star"], 3) for m in METRICS},
                  {k: round(v, 2) for k, v in r["tau_star"].items() if k.endswith("Tr50")}, flush=True)
    (RES / f"design_space_{TAG}.json").write_text(json.dumps(dict(taus=TAUS.tolist(), alphaf=ALPHAF, trs=TRS,
                                                                  sites=res), indent=1))

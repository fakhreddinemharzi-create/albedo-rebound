"""Reference ledger and deterministic variants at every site.

Writes results/ledger_<tag>.json with
  reference      : G_rf, G_co, G_st, G_net, R under the four metrics (TH = 100 yr)
  horizons       : dynamic and GTP metrics for TH = 20, 50, 100, 200, 500 yr
  variants       : static metric with the simple / clear-sky kernel, snow cover
                   0 and 3 days, urban-form factor 0.5, efficacy 0.6, full
                   reconstruction (g_st = 70), rutting-governed structure,
                   renewal every 1 and 10 years
  renewal_sweep  : G_net(T_r) for T_r in 1..20 yr, static and GTP metrics
  coefficient    : CO2e credit of a sustained 0.01 albedo increase on 1 m2 for
                   50 yr under each metric (the quantity published as 'k')
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import ledger, metrics  # noqa: E402
from alrb.config import ALBEDO_GRID, A_EARTH, RES, SITES  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "v3"
METRICS = ["static", "dynamic", "gtp", "stock"]


def life_fun(js, key="life_ref"):
    al = np.array(ALBEDO_GRID)
    L = np.array([js["albedo"][f"{a:.2f}"][key] for a in al])
    L = L / L[list(al).index(0.14)] * 20.0
    return lambda a: np.interp(a, al, L)


def terms(r, m):
    return {k: float(r[m][k]) for k in ("G_rf", "G_co", "G_st", "G_net", "R")}


def one(site):
    js = json.loads((RES / f"structural_{TAG}.json").read_text())[site.key]
    lf = life_fun(js)
    lr = life_fun(js, "life_rut")
    K1 = ledger.daily_kernel(site.key, 1.0).to_numpy()
    out = {}
    base = ledger.Params()
    r = ledger.evaluate(K1, base, lf)
    out["reference"] = {m: terms(r, m) for m in METRICS}
    out["reference"]["L_eff"] = r["L_eff"]
    out["reference"]["phi"] = r["phi"]
    out["reference"]["mean_delta_alpha"] = r["mean_delta_alpha"]
    out["reference"]["f_loc_mean"] = r["f_loc_mean"]
    out["kappa"] = {str(th): ledger.kappa(K1, TH=th) for th in (20, 100, 500)}
    # horizons
    out["horizons"] = {}
    for th in (20, 50, 100, 200, 500):
        rr = ledger.evaluate(K1, base, lf, TH=th, metrics_list=("dynamic", "gtp", "static"))
        out["horizons"][str(th)] = {m: terms(rr, m) for m in ("dynamic", "gtp", "static")}
    # variants (static metric unless stated)
    V = {}
    for name, kind in (("kernel_simple", "simple"), ("kernel_clear", "clear")):
        Kv = ledger.daily_kernel(site.key, 1.0, kind=kind).to_numpy()
        V[name] = terms(ledger.evaluate(Kv, base, lf), "static")
    for d in (0.0, 3.0):
        Kv = ledger.daily_kernel(site.key, d).to_numpy()
        V[f"snow_{d:g}d"] = terms(ledger.evaluate(Kv, base, lf), "static")
    V["psi_0.5"] = terms(ledger.evaluate(K1, ledger.Params(psi=0.5), lf), "static")
    V["efficacy_0.6"] = terms(ledger.evaluate(K1, ledger.Params(efficacy=0.6), lf), "static")
    V["reconstruction_g70"] = terms(ledger.evaluate(K1, ledger.Params(g_st=70.0), lf), "static")
    V["rutting_governed"] = terms(ledger.evaluate(K1, base, lr), "static")
    V["rutting_governed_g70"] = terms(ledger.evaluate(K1, ledger.Params(g_st=70.0), lr), "static")
    V["Tr_1"] = terms(ledger.evaluate(K1, ledger.Params(Tr=1.0), lf), "static")
    V["Tr_10"] = terms(ledger.evaluate(K1, ledger.Params(Tr=10.0), lf), "static")
    V["no_structure"] = terms(ledger.evaluate(K1, base, None), "static")
    out["variants"] = V
    # renewal sweep
    trs = np.round(np.concatenate([np.arange(1.0, 5.01, 0.25), np.arange(5.5, 20.01, 0.5)]), 3)
    sw = {m: [] for m in ("static", "dynamic", "gtp")}
    for tr in trs:
        rr = ledger.evaluate(K1, ledger.Params(Tr=float(tr)), lf, metrics_list=("static", "dynamic", "gtp"))
        for m in sw:
            sw[m].append(terms(rr, m))
    out["renewal_sweep"] = dict(Tr=trs.tolist(), **sw)
    # coefficient: sustained 0.01 albedo increase for H = 50 yr on 1 m2
    f_loc = np.mean(K1) * 0.01
    f_years = np.full(50, f_loc / A_EARTH)
    eq = metrics.equivalents(np.array([0.0]), np.array([0.0]), f_years, TH=100.0)
    out["coefficient"] = {m: float(-eq[m][1]) for m in METRICS}
    eq20 = metrics.equivalents(np.array([0.0]), np.array([0.0]), f_years, TH=20.0)
    out["coefficient"]["dynamic_TH20"] = float(-eq20["dynamic"][1])
    # temperature-based equivalence of a sustained change evaluated while it is sustained (80 yr)
    f80 = np.full(80, f_loc / A_EARTH)
    eq80 = metrics.equivalents(np.array([0.0]), np.array([0.0]), f80, TH=80.0)
    out["coefficient"]["gtp_sustained_80"] = float(-eq80["gtp"][1])
    # stock with an airborne fraction of 0.55 (forcing equivalence to an emitted, not an airborne, mass)
    out["coefficient"]["stock_af055"] = float(-f_loc / A_EARTH / (metrics.RE_KG * 0.55))
    return site.key, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(4) as ex:
        for key, o in ex.map(one, SITES):
            res[key] = o
            rf = o["reference"]
            print(f"{key:13s} " + " | ".join(f"{m}: net {rf[m]['G_net']:6.2f} R {rf[m]['R']:5.2f}" for m in METRICS)
                  + f" | coef static {o['coefficient']['static']:.2f} gtp80 {o['coefficient']['gtp_sustained_80']:.2f}"
                  f" stockAF {o['coefficient']['stock_af055']:.2f}", flush=True)
    (RES / f"ledger_{TAG}.json").write_text(json.dumps(res, indent=1))

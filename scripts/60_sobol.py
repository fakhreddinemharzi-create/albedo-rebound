"""Uncertainty propagation and variance-based sensitivity of the net
balance (Saltelli sampling, Sobol first-order and total-effect indices;
SALib) at every site and under every metric.

Two designs: (A) all twelve parameters including the renewal interval;
(B) renewal interval fixed at the reference 3 years (physical and inventory
uncertainty only).

usage: python 60_sobol.py [tag] [N]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import ledger  # noqa: E402
from alrb.config import ALBEDO_GRID, RES, SITES  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "v3"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
METRICS = ["static", "dynamic", "gtp", "stock"]

PARAMS = [
    ("alpha0", 0.09, 0.18),
    ("alphaf", 0.25, 0.60),
    ("log_tau", np.log10(0.5), np.log10(3.0)),
    ("Tr", 1.0, 10.0),
    ("g_co", 1.63, 3.39),
    ("log_g_st", np.log10(6.0), np.log10(70.0)),
    ("kmult", 0.85, 1.15),
    ("efficacy", 0.60, 1.10),
    ("snow_days", 0.0, 3.0),
    ("psi", 0.60, 1.00),
    ("hmult", 0.60, 1.60),
    ("resid", 0.00, 0.05),
]


def life_functions(key):
    """L(alpha, hmult) from the structural runs at hmult 0.6, 1.0, 1.6."""
    tabs = {}
    for h, tag in ((0.6, f"{TAG}_h06"), (1.0, TAG), (1.6, f"{TAG}_h16")):
        f = RES / f"structural_{tag}.json"
        if f.exists():
            js = json.loads(f.read_text())[key]
            tabs[h] = np.array([js["albedo"][f"{a:.2f}"]["life_ref"] for a in ALBEDO_GRID]) / js["albedo"]["0.14"]["life_ref"] * 20.0
    hs = np.array(sorted(tabs))
    L = np.array([tabs[h] for h in hs])
    al = np.array(ALBEDO_GRID)

    def fun(alpha, hmult):
        Lh = np.array([np.interp(alpha, al, L[i]) for i in range(len(hs))])
        if len(hs) == 1:
            return Lh[0]
        return np.array([np.interp(hmult, hs, Lh[:, j]) for j in range(Lh.shape[1])]) if Lh.ndim > 1 else np.interp(hmult, hs, Lh)
    return fun


def run_site(args):
    key, design = args
    days, bank, idx = ledger.daily_kernel_bank(key)
    lf = life_functions(key)
    names = [p[0] for p in PARAMS]
    bounds = [[p[1], p[2]] for p in PARAMS]
    if design == "B":
        names = [n for n in names if n != "Tr"]
        bounds = [b for p, b in zip(PARAMS, bounds) if p[0] != "Tr"]
    problem = dict(num_vars=len(names), names=names, bounds=bounds)
    X = sobol_sample.sample(problem, N, calc_second_order=False, seed=7)
    Y = {m: np.empty(len(X)) for m in METRICS}
    comps = {m: np.empty((len(X), 3)) for m in METRICS}
    for i, x in enumerate(X):
        v = dict(zip(names, x))
        Tr = v.get("Tr", 3.0)
        p = ledger.Params(alpha0=v["alpha0"], alphaf=v["alphaf"], alpha_inf=v["alpha0"] + v["resid"],
                          tau=10 ** v["log_tau"], Tr=Tr, g_co=v["g_co"], g_st=10 ** v["log_g_st"],
                          kmult=v["kmult"], efficacy=v["efficacy"], snow_days=v["snow_days"], psi=v["psi"],
                          hmult=v["hmult"])
        # daily kernel for this snow-cover duration (linear between the stored durations)
        w = np.interp(v["snow_days"], days, np.arange(len(days)))
        j0 = int(np.floor(w))
        j1 = min(j0 + 1, len(days) - 1)
        f = w - j0
        K = (1 - f) * bank[j0] + f * bank[j1]
        life = lambda a, h=v["hmult"]: lf(a, h)
        r = ledger.evaluate(K, p, life)
        for m in METRICS:
            Y[m][i] = r[m]["G_net"]
            comps[m][i] = (r[m]["G_rf"], r[m]["G_co"], r[m]["G_st"])
    out = {}
    for m in METRICS:
        Si = sobol_analyze.analyze(problem, Y[m], calc_second_order=False, seed=7)
        y = Y[m]
        out[m] = dict(names=names, S1=Si["S1"].tolist(), S1_conf=Si["S1_conf"].tolist(), ST=Si["ST"].tolist(),
                      ST_conf=Si["ST_conf"].tolist(),
                      q=np.percentile(y, [5, 25, 50, 75, 95]).tolist(), mean=float(y.mean()),
                      p_source=float(np.mean(y > 0)),
                      hist=np.histogram(y, bins=60, range=(np.percentile(y, 0.5), np.percentile(y, 99.5)))[0].tolist(),
                      hist_edges=np.histogram_bin_edges(y, bins=60, range=(np.percentile(y, 0.5), np.percentile(y, 99.5))).tolist(),
                      comp_median=np.median(comps[m], axis=0).tolist())
    return key, design, out


if __name__ == "__main__":
    jobs = [(s.key, d) for d in ("A", "B") for s in SITES]
    res = {"A": {}, "B": {}}
    with ProcessPoolExecutor(4) as ex:
        for key, d, out in ex.map(run_site, jobs):
            res[d][key] = out
            print(d, key, {m: (round(out[m]["q"][2], 2), round(out[m]["p_source"], 3)) for m in METRICS}, flush=True)
    (RES / f"sobol_{TAG}.json").write_text(json.dumps(dict(N=N, params=PARAMS, results=res), indent=1))

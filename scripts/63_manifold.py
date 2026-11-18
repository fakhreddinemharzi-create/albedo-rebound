"""Break-even embodied carbon per application, g_co*(tau, T_r), on a fine
grid of coating decay time and renewal interval (laid albedo 0.45), for
every site under the static and dynamic GWP100.  The ledger is linear in
g_co, so g_co* = -G_net(g_co = 0) / (dG_net/dg_co) exactly.

usage: python 63_manifold.py [tag]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import ledger  # noqa: E402
from alrb.config import ALBEDO_GRID, RES, SITES  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "v3"
TAUS = np.logspace(np.log10(0.2), np.log10(20.0), 34)
TRS = np.logspace(0.0, np.log10(25.0), 28)
METRICS = ("static", "dynamic")


def one(site):
    Kd = ledger.daily_kernel(site.key, 1.0).to_numpy()
    js = json.loads((RES / f"structural_{TAG}.json").read_text())[site.key]
    al = np.array(ALBEDO_GRID)
    L = np.array([js["albedo"][f"{a:.2f}"]["life_ref"] for a in al])
    lf = lambda a: np.interp(a, al, L)
    G = {m: np.full((len(TAUS), len(TRS)), np.nan) for m in METRICS}
    for i, tau in enumerate(TAUS):
        for j, tr in enumerate(TRS):
            r0 = ledger.evaluate(Kd, ledger.Params(tau=float(tau), Tr=float(tr), g_co=0.0), lf, metrics_list=METRICS)
            r1 = ledger.evaluate(Kd, ledger.Params(tau=float(tau), Tr=float(tr), g_co=1.0), lf, metrics_list=METRICS)
            for m in METRICS:
                b = r1[m]["G_net"] - r0[m]["G_net"]
                if b > 0:
                    G[m][i, j] = -r0[m]["G_net"] / b
    return site.key, {m: G[m].tolist() for m in METRICS}


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(4) as ex:
        for key, g in ex.map(one, SITES):
            res[key] = g
            a = np.array(g["dynamic"])
            print(key, "g* dynamic at tau=0.91,Tr=3 ~", round(float(a[np.argmin(np.abs(TAUS - 0.91)),
                                                                        np.argmin(np.abs(TRS - 3.0))]), 3), flush=True)
    (RES / f"manifold_{TAG}.json").write_text(json.dumps(dict(taus=TAUS.tolist(), trs=TRS.tolist(), sites=res)))

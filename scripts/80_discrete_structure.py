"""Discrete renewals of the structure against the flow of Eq. (gst).

The ledger enters the change of structural renewals as a continuous flow
g_st (1/L_eff - 1/L_0).  Real renewals are discrete events.  Here the two
schedules are written out: the untreated structure, of age a0 when the
first coat is laid, is renewed at (L_0 - a0) + k L_0 and the treated one at
(1 - a0/L_0) L_eff + k L_eff, and the difference of the two series of pulses
is converted with the same metrics as the coats.  Two allocations of the
events are evaluated for every initial age:

  prorata : the last renewal of each schedule is allocated to the analysis
            period in proportion to its service inside it (the allocation
            the ledger uses for the coats);
  cashflow: every renewal inside the analysis period counts in full and
            those after it not at all.

Writes results/discrete_structure_v3.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import metrics  # noqa: E402
from alrb.config import REF, RES, SITES  # noqa: E402

TAG = "v3"
L0 = 20.0
H = REF.H
TH = 100.0
METRICS = ("static", "dynamic", "gtp")
NAGE = 400


def schedule(first, L, mode):
    """Renewal times inside the analysis period and their allocation."""
    t = np.arange(first, H, L)
    w = np.ones(len(t))
    if mode == "prorata" and len(t):
        w[-1] = min(1.0, (H - t[-1]) / L)
    return t, w


def discrete(a0, L_eff, g_st, mode):
    tu, wu = schedule(L0 - a0, L0, mode)
    tt, wt = schedule((1.0 - a0 / L0) * L_eff, L_eff, mode)
    t = np.concatenate([tt, tu])
    m = g_st * np.concatenate([wt, -wu])
    eq = metrics.equivalents(t, m, np.zeros(1), TH=TH)
    return {k: eq[k][0] for k in METRICS}


def flow(L_eff, g_st):
    ny = int(H + 0.5)
    eq = metrics.equivalents(np.arange(ny) + 0.5, np.full(ny, g_st * (1.0 / L_eff - 1.0 / L0)), np.zeros(1), TH=TH)
    return {k: eq[k][0] for k in METRICS}


def main():
    lg = json.loads((RES / f"ledger_{TAG}.json").read_text())
    ages = (np.arange(NAGE) + 0.5) / NAGE * L0
    out = {}
    for s in SITES:
        ref = lg[s.key]["reference"]
        L_eff = ref["L_eff"]
        o = dict(L_eff=L_eff)
        for g_st, name in ((REF.g_st, "g9"), (70.0, "g70")):
            fl = flow(L_eff, g_st)
            o[name] = dict(flow=fl)
            for mode in ("prorata", "cashflow"):
                G = {k: np.array([discrete(a, L_eff, g_st, mode)[k] for a in ages]) for k in METRICS}
                d = {}
                for k in METRICS:
                    burden = ref[k]["G_co"]
                    credit = abs(ref[k]["G_rf"])
                    # the reference ratio uses g_st = 9; rescale the structural term for the variant
                    R_flow = (burden + fl[k]) / credit
                    R = (burden + G[k]) / credit
                    d[k] = dict(mean=float(G[k].mean()), min=float(G[k].min()), max=float(G[k].max()),
                                R_flow=float(R_flow), R_mean=float(R.mean()), R_min=float(R.min()),
                                R_max=float(R.max()),
                                dR_min_pct=float(100 * (R.min() / R_flow - 1)),
                                dR_max_pct=float(100 * (R.max() / R_flow - 1)),
                                share_avoided=float(np.mean(G[k] < -0.5 * g_st)),
                                share_sink=float(np.mean(burden + G[k] < credit)))
                o[name][mode] = d
            if name == "g9":
                o["curve"] = dict(age=ages[::8].tolist(),
                                  prorata_dynamic=[discrete(a, L_eff, g_st, "prorata")["dynamic"] for a in ages[::8]],
                                  cashflow_dynamic=[discrete(a, L_eff, g_st, "cashflow")["dynamic"] for a in ages[::8]])
        out[s.key] = o
        p, c = o["g9"]["prorata"], o["g9"]["cashflow"]
        print(f"{s.key:13s} L_eff {L_eff:5.2f} | flow st {o['g9']['flow']['static']:6.3f} dyn {o['g9']['flow']['dynamic']:6.3f}"
              f" | prorata dyn mean {p['dynamic']['mean']:6.3f} [{p['dynamic']['min']:6.3f},{p['dynamic']['max']:6.3f}]"
              f" dR [{p['dynamic']['dR_min_pct']:+.2f},{p['dynamic']['dR_max_pct']:+.2f}]%"
              f" | cash dyn mean {c['dynamic']['mean']:6.3f} [{c['dynamic']['min']:6.2f},{c['dynamic']['max']:5.2f}]"
              f" dR [{c['dynamic']['dR_min_pct']:+.1f},{c['dynamic']['dR_max_pct']:+.1f}]% avoided {c['dynamic']['share_avoided']:.3f}"
              f" | g70 cash sink share {o['g70']['cashflow']['dynamic']['share_sink']:.3f}"
              f" prorata dR [{o['g70']['prorata']['dynamic']['dR_min_pct']:+.2f},{o['g70']['prorata']['dynamic']['dR_max_pct']:+.2f}]%",
              flush=True)
    (RES / f"discrete_structure_{TAG}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Figure: how the treatment acts on the unbound layers and on the timing of
damage -- one winter at Goodridge (untreated vs albedo 0.45) and the
monthly distribution of fatigue and subgrade damage at three sites."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import damage, plotting as P, palette as C, structural, summaries  # noqa: E402
from alrb.config import RAW, RES, FIG, SITE  # noqa: E402

TAG = "v3"
KEY = "goodridge"
W0, W1 = "2018-10-15", "2019-06-30"


def load(key, a):
    z = np.load(RAW / "sweep" / TAG / f"{key}_a{a:.2f}.npz")
    idx = pd.date_range(pd.Timestamp(str(z["t0"])), periods=int(z["n"]), freq="h")
    st = structural.hourly_state(z["T"], z["lice"], z["depths"])
    return z, idx, st


def rates(st, idx, key, n_ref):
    w = structural.traffic_weights(idx, SITE[key].tz) * n_ref
    et, ev, E2 = st.strains[:, 0], st.strains[:, 1], st.E[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        df = np.where(et > 0, w / (0.0795 * np.maximum(et, 1e-12) ** -3.291 * (E2 * 1e6 / 6894.757) ** -0.854), 0.0)
        dd = np.where(ev > 0, w / (1.365e-9 * np.maximum(ev, 1e-12) ** -4.477), 0.0)
    return df, dd


def main():
    P.setup()
    sj = json.loads((RES / f"structural_{TAG}.json").read_text())
    fig = plt.figure(figsize=(P.DOUBLE, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    runs = {a: load(KEY, a) for a in (0.14, 0.45)}
    cols = {0.14: C.WARM, 0.45: C.COOL}
    labs = {0.14: r"$\alpha=0.14$", 0.45: r"$\alpha=0.45$"}
    # ------------------------------------------------------------ (a) frost depth and snow
    ax = fig.add_subplot(gs[0, 0])
    mask = np.load(RAW / "sweep" / TAG / f"{KEY}_snowmask.npy").astype(float)
    for a, (z, idx, st) in runs.items():
        li = summaries.local_index(idx, KEY)
        sel = (li >= W0) & (li <= W1)
        fr = pd.Series(z["frost"][sel], index=li[sel]).resample("D").max()
        ax.plot(fr.index, fr.values, color=cols[a], lw=1.0, label=labs[a])
    z, idx, st = runs[0.14]
    li = summaries.local_index(idx, KEY)
    sel = (li >= W0) & (li <= W1)
    sm = pd.Series(mask[: len(idx)][sel], index=li[sel]).resample("D").max()
    for d, v in sm.items():
        if v > 0.5:
            P.vspan(ax, d, d + pd.Timedelta(days=1), color=C.PALE, lw=0, zorder=0)
    ax.set_ylim(2.6, 0)
    ax.set_ylabel("Frost depth (m)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.set_xlim(pd.Timestamp(W0), pd.Timestamp(W1))
    ax.legend(loc="lower left", fontsize=5.8, borderaxespad=0.3)
    ax.text(0.83, 0.45, "shaded: street snow-covered", transform=ax.transAxes, ha="right", va="center",
            fontsize=5.5, color=C.GREY)
    P.panel_label(ax, "a")
    # ------------------------------------------------------------ (b) unbound moduli
    ax = fig.add_subplot(gs[0, 1])
    for a, (z, idx, st) in runs.items():
        li = summaries.local_index(idx, KEY)
        sel = (li >= W0) & (li <= W1)
        for j, ls, nm in ((2, "-", "base"), (4, (0, (3, 1.5)), "subgrade")):
            e = pd.Series(st.E[sel, j], index=li[sel]).resample("D").mean()
            ax.plot(e.index, e.values, color=cols[a], lw=0.9, ls=ls)
    ax.set_yscale("log")
    ax.set_ylim(30, 1.2e4)
    ax.set_yticks([50, 100, 250, 1000, 6895])
    ax.set_yticklabels(["50", "100", "250", "1000", "6895"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel(r"Resilient modulus (MPa)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.set_xlim(pd.Timestamp(W0), pd.Timestamp(W1))
    ax.plot([], [], color=C.GREY, lw=0.9, label="base")
    ax.plot([], [], color=C.GREY, lw=0.9, ls=(0, (3, 1.5)), label="subgrade")
    ax.legend(loc="lower left", fontsize=5.8, borderaxespad=0.3, ncol=2)
    ax.text(0.99, 0.96, "frozen", transform=ax.transAxes, ha="right", va="top", fontsize=5.8)
    P.panel_label(ax, "b")
    # ------------------------------------------------------------ (c) monthly fatigue damage
    ax = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    for i, key in enumerate(("goodridge", "durham", "tucson")):
        n_ref = sj[key]["n_ref"]
        res = {}
        for a in (0.14, 0.45):
            z, idx, st = load(key, a)
            li = summaries.local_index(idx, key)
            df, dd = rates(st, idx, key, n_ref)
            yrs = (li.year >= 2017) & (li.year <= 2024)
            mon = li[yrs].month
            ny = len(np.unique(li[yrs].year))
            res[a] = (np.bincount(mon - 1, weights=df[yrs], minlength=12) / ny,
                      np.bincount(mon - 1, weights=dd[yrs], minlength=12) / ny)
        m = np.arange(1, 13)
        col = C.site_colors()[key]
        for a, ls in ((0.14, "-"), (0.45, (0, (3, 1.5)))):
            ax.plot(m, 100 * res[a][0], color=col, lw=0.9, ls=ls, marker="o", ms=1.8)
            ax2.plot(m, 100 * res[a][1], color=col, lw=0.9, ls=ls, marker="o", ms=1.8)
    for a_ in (ax, ax2):
        a_.set_xticks(np.arange(1, 13))
        a_.set_xticklabels(list("JFMAMJJASOND"))
        a_.set_xlim(0.5, 12.5)
        a_.set_xlabel("Month")
    ax.set_ylabel(r"Fatigue damage per month (\% of life)")
    ax2.set_ylabel(r"Subgrade damage per month (\% of life)")
    ax.set_ylim(0, 1.25)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 3)
    ax2.set_yticks([0.001, 0.01, 0.1, 1])
    ax2.set_yticklabels(["0.001", "0.01", "0.1", "1"])
    ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    hs = [matplotlib.lines.Line2D([], [], color=C.site_colors()[k], lw=1.0) for k in ("goodridge", "durham", "tucson")]
    hs += [matplotlib.lines.Line2D([], [], color=C.GREY, lw=0.9), matplotlib.lines.Line2D([], [], color=C.GREY, lw=0.9,
                                                                                          ls=(0, (3, 1.5)))]
    ax.legend(hs, ["Goodridge", "Durham", "Tucson", r"$\alpha=0.14$", r"$\alpha=0.45$"], loc="upper left",
              fontsize=5.6, ncol=2, borderaxespad=0.3, columnspacing=0.8)
    P.panel_label(ax, "c")
    P.panel_label(ax2, "d")
    P.save(fig, FIG / "fig_frostmech")


if __name__ == "__main__":
    main()

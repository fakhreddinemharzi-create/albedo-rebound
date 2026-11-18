"""Figure: uncertainty of the net balance and its variance decomposition
(Saltelli sampling, N = 1024 base samples; SALib).  Design A samples all
twelve parameters including the renewal interval; design B fixes the
renewal interval at the reference 3 years."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm as ccm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import plotting as P, palette as C  # noqa: E402
from alrb.config import RES, FIG  # noqa: E402

TAG = "v3"
M = "dynamic"
LAB = {"alpha0": r"$\alpha_0$", "alphaf": r"$\alpha_f$", "log_tau": r"$\tau$", "Tr": r"$T_r$",
       "g_co": r"$g_{\mathrm{co}}$", "log_g_st": r"$g_{\mathrm{st}}$", "kmult": r"kernel",
       "efficacy": "efficacy", "snow_days": "snow", "psi": r"$\psi$", "hmult": r"$h$ mult.",
       "resid": r"$\alpha_\infty-\alpha_0$"}


def ridges(ax, res, keys, cols, xlim):
    for i, k in enumerate(keys[::-1]):
        r = res[k][M]
        e = np.array(r["hist_edges"])
        h = np.array(r["hist"], float)
        h = h / h.max() * 0.85
        xc = 0.5 * (e[1:] + e[:-1])
        ax.fill_between(xc, i, i + h, color=cols[k], alpha=0.75, lw=0)
        ax.plot(xc, i + h, color=C.INK, lw=0.35)
        q = r["q"]
        ax.plot([q[2]], [i + 0.06], "|", color="white", ms=4.5, mew=1.0)
        ax.text(xlim[1] - 0.5, i + 0.35, rf"{100 * r['p_source']:.0f}\,\%", fontsize=5.4, ha="right", va="center")
    ax.axvline(0, color=C.INK, lw=0.6)
    ax.set_yticks(np.arange(len(keys)) + 0.35)
    ax.set_yticklabels([C.SITE_SHORT[k] for k in keys[::-1]], fontsize=5.8)
    ax.set_ylim(-0.1, len(keys) + 0.25)
    ax.set_xlim(*xlim)
    ax.grid(axis="y", lw=0)
    ax.set_xlabel(r"Net balance, dynamic GWP$_{100}$ (kg CO$_2$e\,m$^{-2}$)")


def heat(ax, res, keys, names, fig, cbar=True):
    Z = np.array([[res[k][M]["ST"][names.index(n)] for k in keys] for n in names])
    im = ax.imshow(Z, aspect="auto", cmap=ccm.lapaz_r, vmin=0, vmax=0.8, interpolation="nearest")
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([C.SITE_SHORT[k] for k in keys], rotation=90, fontsize=5.6)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([LAB[n] for n in names], fontsize=5.8)
    ax.grid(False)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if Z[i, j] >= 0.05:
                ax.text(j, i, f"{Z[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=4.4,
                        color="white" if Z[i, j] > 0.4 else C.INK)
    return im


def main():
    P.setup()
    d = json.loads((RES / f"sobol_{TAG}.json").read_text())
    A, B = d["results"]["A"], d["results"]["B"]
    keys = C.SITE_ORDER
    cols = C.site_colors()
    fig = plt.figure(figsize=(P.DOUBLE, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05])
    # ------------------------------------------------------------ (a), (b) distributions
    ax = fig.add_subplot(gs[0, 0])
    ridges(ax, B, keys, cols, (-15, 50))
    ax.set_title(r"B: $T_r$ fixed at 3\,yr", fontsize=6.2, pad=2)
    P.panel_label(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    ridges(ax, A, keys, cols, (-15, 50))
    ax.set_title(r"A: $T_r$ sampled, 1--10\,yr", fontsize=6.2, pad=2)
    P.panel_label(ax, "b")
    # ------------------------------------------------------------ (c) probability of a net source
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(keys))
    for m in ("static", "dynamic", "stock", "gtp"):
        ax.plot(x, [100 * B[k][m]["p_source"] for k in keys], "o-", ms=2.6, lw=0.9, color=C.METRIC_COLORS[m],
                label=C.METRIC_LABELS[m])
        ax.plot(x, [100 * A[k][m]["p_source"] for k in keys], "o", ms=2.6, lw=0.8, ls=(0, (2, 1.5)),
                color=C.METRIC_COLORS[m], mfc="white")
    ax.plot([], [], "o-", color=C.GREY, ms=2.6, lw=0.9, label="design B")
    ax.plot([], [], "o", color=C.GREY, ms=2.6, lw=0.8, ls=(0, (2, 1.5)), mfc="white", label="design A")
    ax.set_xticks(x)
    ax.set_xticklabels([C.SITE_SHORT[k] for k in keys], rotation=90, fontsize=5.8)
    ax.set_ylim(74, 101)
    ax.set_ylabel(r"Probability of a net source (\%)")
    ax.legend(loc="lower left", fontsize=5.3, ncol=2, borderaxespad=0.3, columnspacing=0.6, handlelength=1.6)
    P.panel_label(ax, "c")
    # ------------------------------------------------------------ (d), (e) total-effect indices
    namesA = A["durham"][M]["names"]
    namesB = B["durham"][M]["names"]
    ax = fig.add_subplot(gs[1, 0])
    im = heat(ax, B, keys, namesB, fig)
    ax.set_title(r"B: total-effect index $S_T$", fontsize=6.2, pad=2)
    P.panel_label(ax, "d")
    ax = fig.add_subplot(gs[1, 1])
    heat(ax, A, keys, namesA, fig)
    ax.set_title(r"A: total-effect index $S_T$", fontsize=6.2, pad=2)
    P.panel_label(ax, "e")
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.9)
    cb.set_label(r"$S_T$", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.4)
    # ------------------------------------------------------------ (f) first-order vs total, median over sites
    ax = fig.add_subplot(gs[1, 2])
    S1 = np.median([[B[k][M]["S1"][i] for i in range(len(namesB))] for k in keys], axis=0)
    ST = np.median([[B[k][M]["ST"][i] for i in range(len(namesB))] for k in keys], axis=0)
    order = np.argsort(ST)[::-1][:7]
    y = np.arange(len(order))[::-1]
    ax.barh(y + 0.18, ST[order], 0.34, color=C.CREDIT, label=r"total effect $S_T$")
    ax.barh(y - 0.18, np.clip(S1[order], 0, None), 0.34, color=C.GOLD, label=r"first order $S_1$")
    ax.set_yticks(y)
    ax.set_yticklabels([LAB[namesB[i]] for i in order], fontsize=5.8)
    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Sensitivity index (median of the sites)")
    ax.grid(axis="y", lw=0)
    ax.legend(loc="lower right", fontsize=5.5, borderaxespad=0.3)
    ax.set_title(r"B, dynamic GWP$_{100}$", fontsize=6.2, pad=2)
    P.panel_label(ax, "f")
    P.save(fig, FIG / "fig_uncertainty")


if __name__ == "__main__":
    main()

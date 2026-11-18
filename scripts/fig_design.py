"""Figure: design space of reflective treatments -- break-even embodied
carbon per application g_co* against coating durability, laid albedo and
renewal interval, and the break-even frontiers of the ten sites."""
import json
import sys
from pathlib import Path

import contourpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import plotting as P, palette as C  # noqa: E402
from alrb.config import RES, FIG, REF  # noqa: E402

TAG = "v3"
M = "dynamic"


def band(ax, x, Y, color, label, ls="-"):
    Y = np.array(Y)
    ax.fill_between(x, np.nanmin(Y, 0), np.nanmax(Y, 0), color=color, alpha=0.18, lw=0)
    ax.plot(x, np.nanmedian(Y, 0), color=color, lw=1.0, ls=ls, label=label)


def logaxes(ax, xt, yt):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v:g}" for v in xt])
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v:g}" for v in yt])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def main():
    P.setup()
    ds = json.loads((RES / f"design_space_{TAG}.json").read_text())
    mf = json.loads((RES / f"manifold_{TAG}.json").read_text())
    taus = np.array(ds["taus"])
    keys = C.SITE_ORDER
    cols = C.site_colors()
    S = ds["sites"]
    fig, axs = plt.subplots(2, 2, figsize=(P.DOUBLE, 4.6), constrained_layout=True)
    inv = (1.63, 3.39)
    # ------------------------------------------------------------ (a) laid albedo
    ax = axs[0, 0]
    for af, c in ((0.30, C.WARM), (0.45, C.CREDIT), (0.60, C.FROST)):
        band(ax, taus, [S[k]["g_star"][M][f"af{af:.2f}_Tr3"] for k in keys], c, rf"$\alpha_f={af:.2f}$")
    P.hspan(ax, *inv, color=C.BURDEN, alpha=0.10, lw=0)
    ax.axhline(REF.g_co, color=C.BURDEN, lw=0.8)
    ax.axvline(REF.tau, color=C.GREY, lw=0.5, ls=(0, (2, 2)))
    logaxes(ax, [0.2, 0.5, 1, 2, 5, 10, 30], [0.1, 0.3, 1, 3, 10])
    ax.set_xlim(0.2, 30)
    ax.set_ylim(0.08, 20)
    ax.set_xlabel(r"Decay time of the coating, $\tau$ (yr)")
    ax.set_ylabel(r"Break-even $g^{*}_{\mathrm{co}}$ (kg CO$_2$e\,m$^{-2}$)")
    ax.text(29.0, 1.68, r"coating inventories (1.63--3.39)", fontsize=5.6, color=C.BURDEN, va="bottom", ha="right")
    ax.text(REF.tau * 1.05, 0.09, r"$\tau$ ref.", fontsize=5.6, color=C.GREY, va="bottom")
    ax.legend(loc="lower right", fontsize=5.8, borderaxespad=0.3, title=r"$T_r=3$\,yr", title_fontsize=5.8)
    P.panel_label(ax, "a")
    # ------------------------------------------------------------ (b) renewal interval
    ax = axs[0, 1]
    for tr, c, ls in ((1, C.WARM, "-"), (3, C.CREDIT, "-"), (10, C.STRUCT, "-"), (50, C.INK, (0, (3, 1.5)))):
        band(ax, taus, [S[k]["g_star"][M][f"af0.45_Tr{tr:g}"] for k in keys], c,
             rf"$T_r={tr:g}$\,yr" + (" (single coat)" if tr == 50 else ""), ls=ls)
    P.hspan(ax, *inv, color=C.BURDEN, alpha=0.10, lw=0)
    ax.axhline(REF.g_co, color=C.BURDEN, lw=0.8)
    logaxes(ax, [0.2, 0.5, 1, 2, 5, 10, 30], [0.1, 0.3, 1, 3, 10])
    ax.set_xlim(0.2, 30)
    ax.set_ylim(0.08, 20)
    ax.set_xlabel(r"Decay time of the coating, $\tau$ (yr)")
    ax.set_ylabel(r"Break-even $g^{*}_{\mathrm{co}}$ (kg CO$_2$e\,m$^{-2}$)")
    ax.legend(loc="lower right", fontsize=5.8, borderaxespad=0.3, title=r"$\alpha_f=0.45$", title_fontsize=5.8)
    P.panel_label(ax, "b")
    # ------------------------------------------------------------ (c) frontiers
    ax = axs[1, 0]
    mt, mr = np.array(mf["taus"]), np.array(mf["trs"])
    for k in keys:
        G = np.array(mf["sites"][k][M], float)
        gen = contourpy.contour_generator(x=np.log10(mt), y=np.log10(mr), z=G.T)
        for ln in gen.lines(REF.g_co):
            ax.plot(10 ** ln[:, 0], 10 ** ln[:, 1], color=cols[k], lw=1.0)
    ax.plot(REF.tau, REF.Tr, "*", ms=7, color=C.GOLD, mec=C.INK, mew=0.4, zorder=5)
    ax.annotate("reference\ntreatment", xy=(REF.tau, REF.Tr), xytext=(0.3, 9.0), fontsize=5.6,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=C.INK))
    logaxes(ax, [0.5, 1, 2, 5, 10, 20], [1, 2, 5, 10, 20])
    ax.set_xlim(0.2, 20)
    ax.set_ylim(1, 25)
    ax.set_xlabel(r"Decay time of the coating, $\tau$ (yr)")
    ax.set_ylabel(r"Renewal interval, $T_r$ (yr)")
    ax.text(12, 1.25, "net sink", fontsize=6, ha="right", va="bottom", color=C.CREDIT)
    ax.text(0.25, 1.25, "net source", fontsize=6, ha="left", va="bottom", color=C.BURDEN)
    ax.text(0.97, 0.97, r"frontier $g^{*}_{\mathrm{co}}=2.70$" "\n" r"dynamic GWP$_{100}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=5.6)
    P.panel_label(ax, "c")
    # ------------------------------------------------------------ (d) required reduction per coat
    ax = axs[1, 1]
    x = np.arange(len(keys))
    for i, m in enumerate(("static", "dynamic", "stock", "gtp")):
        r = [S[k]["ref"][m]["ratio"] for k in keys]
        ax.plot(x, r, "o-", ms=2.8, lw=0.9, color=C.METRIC_COLORS[m], label=C.METRIC_LABELS[m])
    ax.axhline(1.0, color=C.INK, lw=0.6)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 40)
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.set_yticklabels(["1", "2", "5", "10", "20"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels([C.SITE_SHORT[k] for k in keys], fontsize=6)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylabel(r"$g_{\mathrm{co}}/g^{*}_{\mathrm{co}}$, reference treatment")
    ax.legend(loc="upper right", fontsize=5.5, ncol=2, borderaxespad=0.3, columnspacing=0.8)
    ax.text(len(keys) - 0.5, 1.06, "break-even", ha="right", va="bottom", fontsize=5.6)
    P.panel_label(ax, "d")
    hs = [matplotlib.lines.Line2D([], [], color=cols[k], lw=1.2) for k in keys]
    fig.legend(hs, [C.SITE_SHORT[k] for k in keys], loc="outside lower center", ncol=10, fontsize=5.8,
               handlelength=1.6, columnspacing=1.1, frameon=False)
    P.save(fig, FIG / "fig_design")


if __name__ == "__main__":
    main()

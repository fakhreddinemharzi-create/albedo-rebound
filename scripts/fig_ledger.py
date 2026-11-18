"""Figure: the life-cycle carbon balance of the reference treatment at the
ten sites (50-year analysis period)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import plotting as P, palette as C  # noqa: E402
from alrb.config import RES, FIG  # noqa: E402

TAG = "v3"
VARIANTS = [("kernel_clear", "clear-sky kernel"), ("kernel_simple", "simple kernel"),
            ("snow_0d", "snow masking 0 d"), ("snow_3d", "snow masking 3 d"),
            ("psi_0.5", r"half of reflection intercepted ($\psi=0.5$)"), ("efficacy_0.6", "efficacy 0.6"),
            ("no_structure", "no structural term"), ("reconstruction_g70", r"full reconstruction, 70\,kg\,m$^{-2}$"),
            ("rutting_governed", "rutting-governed street"),
            ("rutting_governed_g70", r"rutting-governed, 70\,kg\,m$^{-2}$"),
            ("Tr_10", r"renewal every 10\,yr"), ("Tr_1", r"renewal every year")]


def main():
    P.setup()
    L = json.loads((RES / f"ledger_{TAG}.json").read_text())
    keys = C.SITE_ORDER
    cols = C.site_colors()
    fig = plt.figure(figsize=(P.DOUBLE, 4.7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0])
    # ------------------------------------------------------------ (a) terms per site, static GWP100
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(keys))
    rf = np.array([L[k]["reference"]["static"]["G_rf"] for k in keys])
    co = np.array([L[k]["reference"]["static"]["G_co"] for k in keys])
    stt = np.array([L[k]["reference"]["static"]["G_st"] for k in keys])
    ax.bar(x, co, 0.62, color=C.BURDEN, label=r"coating, $G_{\mathrm{co}}$")
    ax.bar(x, rf, 0.62, color=C.CREDIT, label=r"albedo, $G_{\mathrm{rf}}$")
    ax.bar(x, stt, 0.62, bottom=rf, color=C.STRUCT, label=r"structure, $G_{\mathrm{st}}$")
    for m, mk, face, dx in (("static", "D", C.INK, -0.17), ("dynamic", "o", "white", 0.0), ("stock", "s", "white", 0.17)):
        net = [L[k]["reference"][m]["G_net"] for k in keys]
        ax.plot(x + dx, net, mk, ms=3.0, mfc=face, mec=C.INK, mew=0.6, ls="none",
                label={"static": r"net, static GWP$_{100}$", "dynamic": r"net, dynamic GWP$_{100}$",
                       "stock": "net, stock equivalence"}[m])
    ax.axhline(0, color=C.INK, lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([C.SITE_SHORT[k] for k in keys], fontsize=6)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylim(-24, 68)
    ax.set_ylabel(r"kg CO$_2$e\,m$^{-2}$ over 50\,yr")
    ax.legend(loc="upper center", fontsize=5.5, ncol=2, borderaxespad=0.3, columnspacing=0.8, handlelength=1.4)
    P.panel_label(ax, "a")
    # ------------------------------------------------------------ (b) burden-to-credit ratio
    ax = fig.add_subplot(gs[0, 1])
    for m in ("static", "dynamic", "stock", "gtp"):
        R = [L[k]["reference"][m]["R"] for k in keys]
        ax.plot(x, R, "o-", ms=2.8, lw=0.9, color=C.METRIC_COLORS[m], label=C.METRIC_LABELS[m])
    ax.axhline(1.0, color=C.INK, lw=0.6)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 60)
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1", "2", "5", "10", "20", "50"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels([C.SITE_SHORT[k] for k in keys], fontsize=6)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylabel(r"$R=(G_{\mathrm{co}}+G_{\mathrm{st}})/|G_{\mathrm{rf}}|$")
    ax.text(len(keys) - 0.5, 1.06, "break-even", ha="right", va="bottom", fontsize=5.6, color=C.INK)
    ax.legend(loc="upper right", fontsize=5.5, ncol=2, borderaxespad=0.3, columnspacing=0.8)
    P.panel_label(ax, "b")
    # ------------------------------------------------------------ (c) variants
    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(len(VARIANTS))[::-1]
    for yi, (v, lab) in zip(y, VARIANTS):
        d = np.array([L[k]["variants"][v]["G_net"] - L[k]["reference"]["static"]["G_net"] for k in keys])
        c = C.BURDEN if np.median(d) > 0 else C.CREDIT
        ax.barh(yi, np.median(d), 0.62, color=c, alpha=0.85)
        ax.plot([d.min(), d.max()], [yi, yi], color=C.INK, lw=0.7)
        ax.plot([d.min(), d.max()], [yi, yi], "|", color=C.INK, ms=3.5, mew=0.7)
    ax.axvline(0, color=C.INK, lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([v[1] for v in VARIANTS], fontsize=5.6)
    ax.set_xlim(-70, 85)
    ax.set_xlabel(r"Change in net balance, static GWP$_{100}$ (kg CO$_2$e\,m$^{-2}$)")
    ax.grid(axis="y", lw=0)
    ax.text(0.99, 0.985, "bars: median of the ten sites\nwhiskers: range", transform=ax.transAxes, ha="right",
            va="top", fontsize=5.4, color=C.GREY)
    P.panel_label(ax, "c")
    # ------------------------------------------------------------ (d) renewal interval
    ax = fig.add_subplot(gs[1, 1])
    for k in keys:
        rs = L[k]["renewal_sweep"]
        tr = np.array(rs["Tr"])
        ax.plot(tr, [q["G_net"] for q in rs["static"]], color=cols[k], lw=0.9)
    rs = L["durham"]["renewal_sweep"]
    ax.plot(rs["Tr"], [q["G_net"] for q in rs["dynamic"]], color=C.INK, lw=0.8, ls=(0, (3, 1.5)))
    ax.axhline(0, color=C.INK, lw=0.5)
    ax.axvline(3.0, color=C.GREY, lw=0.5, ls=(0, (2, 2)))
    ax.set_xscale("log")
    ax.set_xlim(1, 20)
    ax.set_xticks([1, 2, 3, 5, 10, 20])
    ax.set_xticklabels(["1", "2", "3", "5", "10", "20"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(-5, 125)
    ax.set_xlabel(r"Renewal interval, $T_r$ (yr)")
    ax.set_ylabel(r"Net balance (kg CO$_2$e\,m$^{-2}$)")
    ax.text(3.1, 118, "reference", fontsize=5.6, color=C.GREY, va="top")
    ax.plot([], [], color=C.GREY, lw=0.9, label=r"static GWP$_{100}$, each site")
    ax.plot([], [], color=C.INK, lw=0.8, ls=(0, (3, 1.5)), label=r"dynamic GWP$_{100}$, Durham")
    ax.legend(loc="upper right", fontsize=5.6, borderaxespad=0.3)
    P.panel_label(ax, "d")
    hs = [matplotlib.lines.Line2D([], [], color=cols[k], lw=1.2) for k in keys]
    fig.legend(hs, [C.SITE_SHORT[k] for k in keys], loc="outside lower center", ncol=10, fontsize=5.8,
               handlelength=1.6, columnspacing=1.1, frameon=False)
    P.save(fig, FIG / "fig_ledger")


if __name__ == "__main__":
    main()

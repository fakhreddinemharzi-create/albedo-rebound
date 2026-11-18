"""Figure: the balance under projected climates (2025-2075, nine CMIP6
models, SSP2-4.5 and SSP5-8.5, delta change on the observed record)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import plotting as P, palette as C  # noqa: E402
from alrb.config import FIG, RES  # noqa: E402

TAG = "v3"
SSP_COL = {"stationary": C.GREY, "ssp245": "#2E86AB", "ssp585": "#C73E1D"}
SSP_LAB = {"stationary": "stationary", "ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
YEARS = np.arange(2025, 2075)


def smooth(v, w=5):
    return pd.Series(v).rolling(w, center=True, min_periods=1).mean().to_numpy()


def ens_series(site, ssp, var):
    runs = site[ssp]["runs"]
    arr = np.array([smooth(runs[m]["series"][var]) for m in runs])
    return np.median(arr, 0), arr.min(0), arr.max(0)


def dots(ax, cj, getter, ylab, keys, dx=0.24, ms=2.6):
    x = np.arange(len(keys))
    for j, ssp in enumerate(("stationary", "ssp245", "ssp585")):
        xs, ys, lo, hi = [], [], [], []
        for i, k in enumerate(keys):
            v = getter(cj["sites"][k], ssp)
            if v is None:
                continue
            xs.append(i + (j - 1) * dx)
            ys.append(v[0])
            lo.append(v[0] - v[1])
            hi.append(v[2] - v[0])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", ms=ms, color=SSP_COL[ssp], elinewidth=0.7, capsize=0,
                    mew=0, label=SSP_LAB[ssp])
    ax.set_xticks(x)
    ax.set_xticklabels([C.SITE_SHORT[k] for k in keys], rotation=90)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylabel(ylab)


def main():
    cj = json.loads((RES / f"climate_{TAG}.json").read_text())
    keys = [k for k in C.SITE_ORDER if k in cj["sites"]]
    P.setup()
    fig = plt.figure(figsize=(P.DOUBLE, 4.55), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    # (a) frost depth at Goodridge
    ax = fig.add_subplot(gs[0, 0])
    g = cj["sites"]["goodridge"]
    st = g["stationary"]["series"]
    ax.plot(YEARS, smooth(st["frost_u"]), color=SSP_COL["stationary"], lw=0.9)
    ax.plot(YEARS, smooth(st["frost_t"]), color=SSP_COL["stationary"], lw=0.9, ls=(0, (3, 1.5)))
    for ssp in ("ssp245", "ssp585"):
        med, lo, hi = ens_series(g, ssp, "frost_u")
        ax.fill_between(YEARS, lo, hi, color=SSP_COL[ssp], alpha=0.18, lw=0)
        ax.plot(YEARS, med, color=SSP_COL[ssp], lw=0.9)
        med_t, _, _ = ens_series(g, ssp, "frost_t")
        ax.plot(YEARS, med_t, color=SSP_COL[ssp], lw=0.9, ls=(0, (3, 1.5)))
    ax.set_ylim(2.55, 1.15)
    ax.set_xlim(2025, 2074)
    ax.set_ylabel("Annual maximum frost depth (m)")
    ax.set_xlabel("Year")
    hs = [matplotlib.lines.Line2D([], [], color=SSP_COL[s], lw=1.0) for s in ("stationary", "ssp245", "ssp585")]
    hs += [matplotlib.lines.Line2D([], [], color=C.INK, lw=0.9), matplotlib.lines.Line2D([], [], color=C.INK, lw=0.9,
                                                                                         ls=(0, (3, 1.5)))]
    ax.legend(hs, ["stationary", "SSP2-4.5", "SSP5-8.5", r"$\alpha=0.14$", "coated"], loc="lower left",
              fontsize=5.4, ncol=2, borderaxespad=0.3, columnspacing=0.8, handlelength=1.6)
    ax.text(0.97, 0.95, "Goodridge", transform=ax.transAxes, ha="right", va="top", fontsize=6.0)
    P.panel_label(ax, "a")
    # (b) fatigue damage of the untreated street over fifty years, relative to the stationary control
    ax = fig.add_subplot(gs[0, 1])

    def get_du(site, ssp):
        if ssp == "stationary":
            return (0.0, 0.0, 0.0)
        if ssp not in site:
            return None
        e = site[ssp]["ensemble"]["dDu_pct"]
        return (e["med"], e["min"], e["max"])
    dots(ax, cj, get_du, r"Change of fatigue damage (\%)", keys)
    ax.axhline(0.0, color=C.INK, lw=0.5)
    ax.set_ylim(-3, 25)
    ax.legend(loc="upper right", fontsize=5.4, borderaxespad=0.3, handletextpad=0.2, ncol=3, columnspacing=0.6)
    ax.text(0.97, 0.80, "untreated street,\n2025--2075", transform=ax.transAxes, fontsize=5.6, color=C.GREY,
            ha="right", va="top")
    P.panel_label(ax, "b")
    # (c) kernel change
    ax = fig.add_subplot(gs[0, 2])

    def get_dk(site, ssp):
        if ssp == "stationary":
            return (0.0, 0.0, 0.0)
        if ssp not in site:
            return None
        e = site[ssp]["ensemble"]["dK_pct"]
        return (e["med"], e["min"], e["max"])
    dots(ax, cj, get_dk, r"Change of the mean kernel (\%)", keys)
    ax.axhline(0.0, color=C.INK, lw=0.5)
    ax.set_ylim(-9, 9)
    P.panel_label(ax, "c")
    # (d) fatigue damage avoided by the coating over the fifty years
    ax = fig.add_subplot(gs[1, 0])

    def get_b(site, ssp):
        if ssp == "stationary":
            v = 100 * site["stationary"]["benefit"]
            return (v, v, v)
        if ssp not in site:
            return None
        e = site[ssp]["ensemble"]["benefit"]
        return (100 * e["med"], 100 * e["min"], 100 * e["max"])
    dots(ax, cj, get_b, r"Fatigue damage avoided (\%)", keys)
    ax.set_ylim(0, 12)
    ax.text(0.97, 0.95, "coated vs untreated,\n2025--2075", transform=ax.transAxes, fontsize=5.6,
            color=C.GREY, ha="right", va="top")
    P.panel_label(ax, "d")
    # (e) structural term
    ax = fig.add_subplot(gs[1, 1])

    def get_gst(site, ssp):
        if ssp == "stationary":
            v = site["stationary"]["G_st"]
            return (v, v, v)
        if ssp not in site:
            return None
        e = site[ssp]["ensemble"]["G_st"]
        return (e["med"], e["min"], e["max"])
    dots(ax, cj, get_gst, r"Structural term $G_{st}$ (kg\,CO$_2$e\,m$^{-2}$)", keys)
    ax.set_ylim(-2.8, 0.0)
    P.panel_label(ax, "e")
    # (f) rebound ratio
    ax = fig.add_subplot(gs[1, 2])

    def get_r(site, ssp):
        if ssp == "stationary":
            v = site["stationary"]["R"]
            return (v, v, v)
        if ssp not in site:
            return None
        e = site[ssp]["ensemble"]["R"]
        return (e["med"], e["min"], e["max"])
    dots(ax, cj, get_r, r"Static rebound ratio $R$", keys)
    ax.set_ylim(0, 5.5)
    ax.axhline(1.0, color=C.INK, lw=0.5, ls=(0, (3, 1.5)))
    P.panel_label(ax, "f")
    P.save(fig, FIG / "fig_climate")


if __name__ == "__main__":
    main()

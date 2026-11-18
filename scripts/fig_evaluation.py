"""Figure: evaluation against observations and independent products --
GOES skin temperature (natural-surface twin), LTPP pavement-temperature
models (untreated pavement) and CMIP6 radiative kernels."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import plotting as P, palette as C  # noqa: E402
from alrb.config import RES, FIG, RAW, SITE, CAL_YEAR  # noqa: E402

LW = "lw"
TAG = "v3"


def main():
    P.setup()
    nv = json.loads((RES / f"natural_validation_{LW}.json").read_text())
    lt = json.loads((RES / f"ltpp_benchmark_{TAG}.json").read_text())
    kn = json.loads((RES / "kernels.json").read_text())
    ks = C.SITE_ORDER
    cols = C.site_colors()
    fig = plt.figure(figsize=(P.DOUBLE, 4.45), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    # ------------------------------------------------------------ (a) density scatter, validation years
    ax = fig.add_subplot(gs[0, 0])
    xs, ys = [], []
    for k in ks:
        d = pd.read_parquet(RAW / "goes" / f"pairs_{k}_{LW}.parquet")
        d = d[pd.DatetimeIndex(d.time).year != CAL_YEAR]
        xs.append(d.obs.to_numpy())
        ys.append(d.model.to_numpy())
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    hb = ax.hexbin(x, y, gridsize=55, extent=(-40, 70, -40, 70), bins="log", cmap=cm.lapaz_r, mincnt=1,
                   linewidths=0.0)
    ax.plot([-40, 70], [-40, 70], color=C.INK, lw=0.6)
    ax.set_xlim(-40, 70)
    ax.set_ylim(-40, 70)
    ax.set_aspect("equal")
    ax.set_xlabel(r"GOES-R skin temperature ($^{\circ}$C)")
    ax.set_ylabel(r"Modelled skin temperature ($^{\circ}$C)")
    pv = nv["pooled_validation"]
    r = np.corrcoef(x, y)[0, 1]
    ax.text(0.04, 0.96, f"held-out years\n$n$ = {pv['n']:,}\nRMSE {pv['rmse']:.2f} K\nbias {pv['bias']:+.2f} K\n$r$ = {r:.3f}",
            transform=ax.transAxes, va="top", fontsize=5.9)
    cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02, aspect=25)
    cb.set_label("hours per bin", fontsize=6)
    cb.ax.tick_params(labelsize=5.6)
    P.panel_label(ax, "a")
    # ------------------------------------------------------------ (b) RMSE by site
    ax = fig.add_subplot(gs[0, 1])
    xx = np.arange(len(ks))
    cal = [nv["sites"][k][str(CAL_YEAR)]["rmse"] for k in ks]
    val = [nv["sites"][k]["validation"]["rmse"] for k in ks]
    day = [nv["sites"][k]["2024"]["day"]["rmse"] for k in ks]
    nig = [nv["sites"][k]["2024"]["night"]["rmse"] for k in ks]
    ax.bar(xx - 0.2, cal, 0.38, color=C.GREY, label=f"calibration ({CAL_YEAR})")
    ax.bar(xx + 0.2, val, 0.38, color=C.CREDIT, label="held-out (2022, 2024)")
    ax.plot(xx + 0.2, day, "^", ms=2.6, color=C.WARM, label="2024, day")
    ax.plot(xx + 0.2, nig, "v", ms=2.6, color=C.INK, label="2024, night")
    ax.set_xticks(xx)
    ax.set_xticklabels([C.SITE_SHORT[k] for k in ks], rotation=90, fontsize=5.8)
    ax.set_ylabel("RMSE (K)")
    ax.set_ylim(0, 6.2)
    ax.legend(loc="upper left", fontsize=5.6, ncol=2, borderaxespad=0.3, columnspacing=0.7, handlelength=1.2)
    P.panel_label(ax, "b")
    # ------------------------------------------------------------ (c) diurnal composites
    ax = fig.add_subplot(gs[0, 2])
    for k, c in (("goodridge", cols["goodridge"]), ("durham", cols["durham"]), ("tucson", cols["tucson"])):
        dd = nv["sites"][k]["diurnal"]
        h = np.array(dd["hour"])
        ax.plot(h, dd["obs"], "o", ms=2.2, mfc="white", mec=c, mew=0.6)
        ax.plot(h, dd["model"], color=c, lw=0.9, label=C.SITE_LABEL[k].split(",")[0])
    ax.set_xlabel("Local standard time (h)")
    ax.set_ylabel(r"Mean clear-sky $T_s$ ($^{\circ}$C)")
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_ylim(-8, 52)
    ax.legend(loc="upper left", fontsize=5.8, borderaxespad=0.3)
    ax.text(0.97, 0.04, "lines: model; circles: GOES-R", transform=ax.transAxes, fontsize=5.6, ha="right")
    P.panel_label(ax, "c")
    # ------------------------------------------------------------ (d) LTPP benchmark (high)
    ax = fig.add_subplot(gs[1, 0])
    for k in ks:
        o = lt["sites"][k]
        ax.plot(o["ltpp_20mm"], o["model_20mm_7d_high"], "o", ms=3.2, color=cols[k], mec=C.INK, mew=0.4)
        ax.plot(o["shrp_20mm"], o["model_20mm_7d_high"], "s", ms=2.6, mfc="white", mec=cols[k], mew=0.6)
    ax.plot([40, 66], [40, 66], color=C.INK, lw=0.6)
    ax.set_xlim(42, 66)
    ax.set_ylim(42, 66)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Empirical model ($^{\circ}$C)")
    ax.set_ylabel(r"Column, untreated ($^{\circ}$C)")
    st = lt["stats"]
    ax.plot([], [], "o", ms=3.2, color=C.GREY, mec=C.INK, mew=0.4, label=f"LTPP, bias {st['high_vs_ltpp_bias']:+.1f} K")
    ax.plot([], [], "s", ms=2.6, mfc="white", mec=C.GREY, mew=0.6, label=f"SHRP, bias {st['high_vs_shrp_bias']:+.1f} K")
    ax.legend(loc="lower right", fontsize=5.8, borderaxespad=0.3, handletextpad=0.3)
    ax.text(0.04, 0.96, "7-day maximum\nat 20 mm", transform=ax.transAxes, va="top", fontsize=5.9)
    P.panel_label(ax, "d")
    # ------------------------------------------------------------ (e) LTPP benchmark (low)
    ax = fig.add_subplot(gs[1, 1])
    for k in ks:
        o = lt["sites"][k]
        ax.plot(o["ltpp_surface_low"], o["model_surface_low"], "o", ms=3.2, color=cols[k], mec=C.INK, mew=0.4)
        ax.plot(o["air_low"], o["model_surface_low"], "x", ms=2.6, color=cols[k], mew=0.7)
    ax.plot([-40, 5], [-40, 5], color=C.INK, lw=0.6)
    ax.set_xlim(-38, 5)
    ax.set_ylim(-38, 5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Reference ($^{\circ}$C)")
    ax.set_ylabel(r"Column, untreated ($^{\circ}$C)")
    ax.plot([], [], "o", ms=3.2, color=C.GREY, mec=C.INK, mew=0.4, label=f"LTPP low, bias {st['low_vs_ltpp_bias']:+.1f} K")
    ax.plot([], [], "x", ms=2.6, color=C.GREY, mew=0.7, label="annual minimum air")
    ax.legend(loc="lower right", fontsize=5.8, borderaxespad=0.3, handletextpad=0.3)
    ax.text(0.04, 0.96, "annual minimum\nat the surface", transform=ax.transAxes, va="top", fontsize=5.9)
    P.panel_label(ax, "e")
    # ------------------------------------------------------------ (f) kernels vs CMIP6
    ax = fig.add_subplot(gs[1, 2])
    s = kn["sites"]
    xx = np.arange(len(ks))
    ax.errorbar(xx + 0.15, [-s[k]["K_cmip6_mean"] for k in ks],
                yerr=[[-s[k]["K_cmip6_mean"] + s[k]["K_cmip6_max"] for k in ks],
                      [-s[k]["K_cmip6_min"] + s[k]["K_cmip6_mean"] for k in ks]],
                fmt="s", ms=2.6, mfc="white", mec=C.INK, ecolor=C.GREY, elinewidth=0.6, capsize=1.2,
                label="CMIP6, monthly APRP")
    ax.plot(xx - 0.15, [-s[k]["K_aprp_monthlymeans"] for k in ks], "D", ms=2.4, color=C.GOLD,
            label="this study, monthly means")
    ax.plot(xx - 0.15, [-s[k]["K_aprp_hourly"] for k in ks], "o", ms=3.0, color=C.CREDIT, label="this study, hourly")
    ax.plot(xx, [-s[k]["K_simple"] for k in ks], "_", ms=5, mew=1.0, color=C.BURDEN, label=r"simple, $T_a=0.854$")
    ax.set_xticks(xx)
    ax.set_xticklabels([C.SITE_SHORT[k] for k in ks], rotation=90, fontsize=5.8)
    ax.set_ylabel(r"$|\bar K|$ (W\,m$^{-2}$ per 0.01)")
    ax.set_ylim(0.6, 2.9)
    ax.legend(loc="upper left", fontsize=5.5, borderaxespad=0.3, ncol=1, handletextpad=0.3, labelspacing=0.25)
    P.panel_label(ax, "f")
    P.save(fig, FIG / "fig_evaluation")


if __name__ == "__main__":
    main()

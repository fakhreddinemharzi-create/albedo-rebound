"""Figure: sensitivity to the lower boundary of the column (Sect. 5.7).

(a) Mean change of temperature with depth caused by the treatment (albedo
0.45 against 0.14), normalised by its value at 50 mm, for the three lower
boundaries at three sites, with the steady attenuation below the centre of
a 10 m wide strip, (2/pi) arctan(W/2z); (b) mean seasonal course of the
frost depth at Goodridge; (c) ratio of the treated to the untreated service
life.
"""
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
VAR = {"anchored8": ("anchored at 8 m", "-"), "adiabatic8": ("adiabatic at 8 m", (0, (3, 1.5))),
       "anchored15": ("anchored at 15 m", (0, (1, 1.2)))}
KEYS = ("goodridge", "durham", "tucson")


def main():
    bt = json.loads((RES / f"boundary_test_{TAG}.json").read_text())
    P.setup()
    fig = plt.figure(figsize=(P.DOUBLE, 2.45), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.95])
    # (a) normalised profiles
    ax = fig.add_subplot(gs[0, 0])
    z = np.linspace(0.02, 3.0, 200)
    ax.plot(2 / np.pi * np.arctan(10.0 / (2 * z)), z, color=C.GREY, lw=2.2, alpha=0.45, solid_capstyle="round",
            label=r"strip, $W=10$\,m")
    for k in KEYS:
        for v, (lab, ls) in VAR.items():
            u, t = bt[k][v]["0.14"], bt[k][v]["0.45"]
            d = np.array(u["depths"])
            dT = np.array(t["T_profile"]) - np.array(u["T_profile"])
            i5 = int(np.argmin(np.abs(d - 0.05)))
            ax.plot(dT / dT[i5], d, color=C.site_colors()[k], lw=0.9, ls=ls)
    ax.set_ylim(3.0, 0.0)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_xlim(0.4, 1.05)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"$\Delta T(z)/\Delta T(0.05\,\mathrm{m})$")
    ax.set_ylabel("Depth (m)")
    hs = [matplotlib.lines.Line2D([], [], color=C.GREY, lw=2.2, alpha=0.45)]
    hs += [matplotlib.lines.Line2D([], [], color=C.INK, lw=0.9, ls=ls) for _, ls in VAR.values()]
    ax.legend(hs, [r"strip, $W=10$\,m"] + [lab for lab, _ in VAR.values()], loc="upper left", fontsize=5.4,
              borderaxespad=0.3, handlelength=1.8)
    P.panel_label(ax, "a")
    # (b) frost season at Goodridge
    ax = fig.add_subplot(gs[0, 1])
    day0 = pd.Timestamp("2021-08-01")
    days = pd.date_range(day0, periods=365, freq="D")
    u = np.array(bt["goodridge"]["anchored8"]["0.14"]["frost_season"])
    ax.plot(days, u, color=C.WARM, lw=1.1, label=r"untreated ($\alpha=0.14$)")
    for v, (lab, ls) in VAR.items():
        t = np.array(bt["goodridge"][v]["0.45"]["frost_season"])
        ax.plot(days, t, color=C.COOL, lw=0.9, ls=ls, label=rf"$\alpha=0.45$, {lab}")
    ax.set_ylim(2.6, 0.0)
    ax.set_xlim(pd.Timestamp("2021-10-15"), pd.Timestamp("2022-07-15"))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.set_ylabel("Mean frost depth (m)")
    ax.legend(loc="lower left", fontsize=5.4, borderaxespad=0.3, handlelength=1.8)
    ax.text(0.97, 0.05, "Goodridge", transform=ax.transAxes, fontsize=6.0, ha="right", va="bottom")
    P.panel_label(ax, "b")
    # (c) life ratio
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(len(KEYS))
    for j, (v, (lab, ls)) in enumerate(VAR.items()):
        r = [bt[k][v]["0.45"]["life"] / bt[k][v]["0.14"]["life"] for k in KEYS]
        ax.bar(x + (j - 1) * 0.26, r, width=0.24, color=[C.site_colors()[k] for k in KEYS],
               alpha=[1.0, 0.65, 0.35][j], edgecolor=C.INK, linewidth=0.4, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([C.SITE_SHORT[k] for k in KEYS])
    ax.set_ylim(1.0, 1.42)
    ax.set_ylabel(r"Life ratio $L(0.45)/L(0.14)$")
    hs = [matplotlib.patches.Patch(facecolor=C.GREY, alpha=a, edgecolor=C.INK, linewidth=0.4) for a in (1.0, 0.65,
                                                                                                        0.35)]
    ax.legend(hs, [lab for lab, _ in VAR.values()], loc="upper right", fontsize=5.4, borderaxespad=0.3)
    P.panel_label(ax, "c")
    P.save(fig, FIG / "fig_boundary")


if __name__ == "__main__":
    main()

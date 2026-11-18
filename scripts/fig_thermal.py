"""Figure: thermal response of the pavement column to a reflective surface
(albedo 0.14 -> 0.45, reference structure, 2017-2024 complete years)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import driver, plotting as P, palette as C, summaries  # noqa: E402
from alrb.config import RES, FIG, RAW  # noqa: E402

TAG = "v3"


def main():
    P.setup()
    th = json.loads((RES / f"thermal_{TAG}.json").read_text())
    sites = json.loads((RES / "sites.json").read_text())
    cols = C.site_colors()
    keys = C.SITE_ORDER
    fig, axs = plt.subplots(2, 3, figsize=(P.DOUBLE, 4.45), constrained_layout=True)
    # ------------------------------------------------------------ (a) monthly surface cooling
    ax = axs[0, 0]
    m = np.arange(1, 13)
    for k in keys:
        ax.plot(m, th[k]["dts_monthly"], color=cols[k], lw=0.9, marker="o", ms=1.8)
    ax.set_xticks(m)
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(-7.0, 0.0)
    ax.set_xlabel("Month")
    ax.set_ylabel(r"$\Delta \bar T_s$ (K)")
    ax.text(0.5, 0.04, r"monthly mean, $\alpha$ 0.14 $\rightarrow$ 0.45", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=5.8)
    P.panel_label(ax, "a")
    # ------------------------------------------------------------ (b) diurnal cycle
    ax = axs[0, 1]
    h = np.arange(24)
    for k in ("goodridge", "durham", "tucson"):
        ax.plot(h, th[k]["dts_diurnal_jul"], color=cols[k], lw=1.0, label=f"{C.SITE_SHORT[k]}, July")
        ax.plot(h, th[k]["dts_diurnal_jan"], color=cols[k], lw=0.9, ls=(0, (3, 1.6)), label=f"{C.SITE_SHORT[k]}, January")
    ax.set_xlim(0, 23)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_ylim(-13.5, 0.8)
    ax.set_xlabel("Local standard time (h)")
    ax.set_ylabel(r"$\Delta T_s$ (K)")
    ax.legend(loc="lower left", fontsize=5.4, ncol=2, borderaxespad=0.3, columnspacing=0.7, handlelength=1.6,
              labelspacing=0.25)
    P.panel_label(ax, "b")
    # ------------------------------------------------------------ (c) depth profile
    ax = axs[0, 2]
    for k in keys:
        d = np.array(th[k]["depths"])
        dT = np.array(th[k]["dT_depth_mean"])
        _, iu = np.unique(d, return_index=True)
        ax.plot(dT[iu], d[iu], color=cols[k], lw=0.9)
    ax.set_yscale("log")
    ax.set_ylim(3.2, 0.01)
    ax.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
    ax.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1", "3"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(-5.0, -1.5)
    ax.set_xlabel(r"Annual mean $\Delta T$ (K)")
    ax.set_ylabel("Depth (m)")
    for zz, lab in ((0.10, "AC"), (0.35, "base"), (0.65, "subbase")):
        ax.axhline(zz, color=C.GREY, lw=0.4, ls=(0, (2, 2)))
        ax.text(-1.58, zz * 0.93, lab, ha="right", va="bottom", fontsize=5.4, color=C.GREY)
    P.panel_label(ax, "c")
    # ------------------------------------------------------------ (d) frost depth
    ax = axs[1, 0]
    fk = [k for k in keys if th[k]["albedo"]["0.14"]["frost_max"] > 0.01]
    y = np.arange(len(fk))[::-1]
    for yi, k in zip(y, fk):
        a0, a1 = th[k]["albedo"]["0.14"], th[k]["albedo"]["0.45"]
        ax.plot([a0["frost_max"], a1["frost_max"]], [yi, yi], color=C.GREY, lw=0.9, zorder=1)
        ax.plot(a0["frost_max"], yi, "o", ms=3.4, color=C.WARM, zorder=2)
        ax.plot(a1["frost_max"], yi, "o", ms=3.4, color=C.COOL, zorder=2)
        ax.text(2.62, yi, rf"{a0['frozen_days']:.0f}$\rightarrow${a1['frozen_days']:.0f}", va="center", ha="right",
                fontsize=5.4)
    ax.set_yticks(y)
    ax.set_yticklabels([C.SITE_SHORT[k] for k in fk], fontsize=6)
    ax.set_xlim(0, 2.65)
    ax.set_ylim(-0.7, len(fk) - 0.1)
    ax.set_xlabel("Mean annual maximum frost depth (m)")
    ax.plot([], [], "o", ms=3.4, color=C.WARM, label=r"$\alpha=0.14$")
    ax.plot([], [], "o", ms=3.4, color=C.COOL, label=r"$\alpha=0.45$")
    ax.legend(loc="center", bbox_to_anchor=(0.62, 0.42), fontsize=5.6, ncol=1, borderaxespad=0.3, handletextpad=0.2)
    ax.text(2.62, len(fk) - 0.55, "frozen days", ha="right", va="center", fontsize=5.4, color=C.GREY)
    ax.grid(axis="y", lw=0)
    P.panel_label(ax, "d")
    # ------------------------------------------------------------ (e) AC temperature at 25 mm
    ax = axs[1, 1]
    x = np.arange(len(keys))
    for xi, k in zip(x, keys):
        for off, qk, c in ((-0.18, "T25_quantiles_014", C.WARM), (0.18, "T25_quantiles_045", C.COOL)):
            q = th[k][qk]      # 1, 5, 25, 50, 75, 95, 99
            ax.plot([xi + off] * 2, [q[0], q[6]], color=c, lw=0.6)
            ax.plot([xi + off] * 2, [q[1], q[5]], color=c, lw=1.6, solid_capstyle="butt")
            ax.plot([xi + off] * 2, [q[2], q[4]], color=c, lw=3.2, solid_capstyle="butt")
            ax.plot([xi + off - 0.11, xi + off + 0.11], [q[3], q[3]], color="white", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([C.SITE_SHORT[k] for k in keys], rotation=90, fontsize=5.8)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylim(-32, 62)
    ax.set_ylabel(r"Asphalt temperature at 25\,mm ($^{\circ}$C)")
    ax.plot([], [], color=C.WARM, lw=3.0, label=r"$\alpha=0.14$")
    ax.plot([], [], color=C.COOL, lw=3.0, label=r"$\alpha=0.45$")
    ax.legend(loc="upper left", fontsize=5.6, ncol=2, borderaxespad=0.3, columnspacing=0.8, handlelength=1.2)
    P.panel_label(ax, "e")
    # ------------------------------------------------------------ (f) surface cooling vs absorbed shortwave
    ax = axs[1, 2]
    X, Y = [], []
    for k in keys:
        df = driver.load_forcing(k)
        li = summaries.local_index(df.index, k)
        mask = np.load(RAW / "sweep" / TAG / f"{k}_snowmask.npy").astype(float)
        sel = (li.year >= 2017) & (li.year <= 2024)
        sw = (df.sw.to_numpy() * (1 - mask[: len(df)]))[sel]
        mon = li[sel].month
        swm = np.bincount(mon - 1, weights=sw, minlength=12) / np.bincount(mon - 1, minlength=12)
        x = 0.31 * swm
        y = np.array(th[k]["dts_monthly"])
        ax.plot(x, y, "o", ms=2.3, color=cols[k], mec=C.INK, mew=0.2, alpha=0.95)
        X += list(x)
        Y += list(y)
    X, Y = np.array(X), np.array(Y)
    lam = -float(np.sum(X * X) / np.sum(X * Y))
    xx = np.array([0, 110])
    ax.plot(xx, -xx / lam, color=C.GREY, lw=0.7, zorder=0)
    r1 = float(np.corrcoef(X, Y)[0, 1])
    ax.set_xlim(0, 105)
    ax.set_ylim(-7.0, 0.0)
    ax.set_xlabel(r"$\Delta\alpha\,\langle S_{\downarrow}\rangle$, snow-free (W\,m$^{-2}$)")
    ax.set_ylabel(r"Monthly $\Delta \bar T_s$ (K)")
    ax.text(0.96, 0.95, rf"$\Delta \bar T_s = -\Delta\alpha\langle S_{{\downarrow}}\rangle/\Lambda$" "\n"
            rf"$\Lambda={lam:.1f}$\,W\,m$^{{-2}}$\,K$^{{-1}}$, $r={r1:.2f}$", transform=ax.transAxes, ha="right",
            va="top", fontsize=5.8)
    P.panel_label(ax, "f")
    # shared colour key for the sites (cold -> hot)
    hs = [matplotlib.lines.Line2D([], [], color=cols[k], lw=1.2, marker="o", ms=2.2) for k in keys]
    fig.legend(hs, [C.SITE_SHORT[k] for k in keys], loc="outside lower center", ncol=10, fontsize=5.8,
               handlelength=1.6, columnspacing=1.1, frameon=False)
    P.save(fig, FIG / "fig_thermal")
    return dict(Lambda=lam, r=r1)


if __name__ == "__main__":
    print(main())

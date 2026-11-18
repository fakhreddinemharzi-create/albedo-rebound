"""Figure: the radiative kernel of a surface-albedo change at the sites,
its delivery by a decaying coating, and the reconciliation of the published
CO2-equivalence coefficients."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm as ccm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import driver, kernel, ledger, plotting as P, palette as C, summaries  # noqa: E402
from alrb.config import RES, FIG, REF  # noqa: E402

TAG = "v3"
# Published equivalence coefficients (kg CO2 per m2 per 0.01 albedo increase)
LIT = [("Akbari et al. (2009)", 2.55, 2.55), ("Menon et al. (2010)", 3.3, 3.3),
       ("Akbari et al. (2012)", 6.5, 7.5), ("Xu (2018)", 0.90, 1.77), ("VanCuren (2010)", 0.0, 1.1)]


def heat(key):
    df = driver.load_forcing(key)
    K, _, _, _, _ = kernel.hourly_kernel(df)
    mask = driver.snow_mask_pavement(df, key, days_after=1.0)
    li = summaries.local_index(df.index, key)
    s = pd.DataFrame(dict(K=-K * (1 - mask) / 100.0, m=li.month, h=li.hour))
    s = s[(li.year >= 2017) & (li.year <= 2024)]
    return s.pivot_table(index="h", columns="m", values="K", aggfunc="mean").to_numpy()


def main():
    P.setup()
    kj = json.loads((RES / "kernels.json").read_text())["sites"]
    lg = json.loads((RES / f"ledger_{TAG}.json").read_text())
    cols = C.site_colors()
    keys = C.SITE_ORDER
    fig, axs = plt.subplots(2, 3, figsize=(P.DOUBLE, 4.45), constrained_layout=True)
    # ------------------------------------------------------------ (a) monthly cycle
    ax = axs[0, 0]
    m = np.arange(1, 13)
    for k in keys:
        ax.plot(m, -np.array(kj[k]["monthly_cycle"]), color=cols[k], lw=0.9, marker="o", ms=1.8)
    ax.set_xticks(m)
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0, 3.4)
    ax.set_xlabel("Month")
    ax.set_ylabel(r"$|K|$ (W\,m$^{-2}$ per 0.01 albedo)")
    ax.text(0.5, 0.96, "hourly APRP kernel, monthly mean", transform=ax.transAxes, ha="center", va="top",
            fontsize=5.8)
    P.panel_label(ax, "a")
    # ------------------------------------------------------------ (b) hour x month at Goodridge
    ax = axs[0, 1]
    Hm = heat("goodridge")
    im = ax.imshow(Hm, origin="lower", aspect="auto", cmap=ccm.lajolla_r, vmin=0, vmax=8,
                   extent=(0.5, 12.5, -0.5, 23.5), interpolation="nearest")
    ax.set_xticks(m)
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_yticks([0, 6, 12, 18, 23])
    ax.set_xlabel("Month")
    ax.set_ylabel("Local standard time (h)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.95)
    cb.set_label(r"$|K|$, snow-masked (W\,m$^{-2}$ per 0.01)", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.6)
    ax.text(0.03, 0.96, "Goodridge, MN", transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color=C.INK)
    P.panel_label(ax, "b")
    # ------------------------------------------------------------ (c) physics: kernel vs clearness index
    ax = axs[0, 2]
    X, Y = [], []
    for k in keys:
        df = driver.load_forcing(k)
        K, _, _, kt, mu = kernel.hourly_kernel(df)
        g0 = df.g0.to_numpy()
        sel = g0 > 150.0
        X.append(np.minimum(kt[sel], mu[sel]))
        Y.append(-K[sel] / g0[sel])
    X, Y = np.concatenate(X), np.concatenate(Y)
    hb = ax.hexbin(X, Y, gridsize=46, extent=(0, 1, 0, 1), bins="log", cmap=ccm.oslo_r, mincnt=1, linewidths=0.1)
    kk = np.linspace(0, 1, 101)
    for mu_, ls in ((0.80, (0, (3, 1.5))), (0.90, "-")):
        ax.plot(kk[kk <= mu_], kk[kk <= mu_] ** 2 / mu_, color=C.CREDIT, lw=0.9, ls=ls)
    ax.plot(kk, 0.854 * kk, color=C.BURDEN, lw=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"Clearness index, $k_t$ (--)")
    ax.set_ylabel(r"$|K|/S_{\mathrm{TOA}}$ (--)")
    ax.plot([], [], color=C.CREDIT, lw=0.9, label=r"$k_t^2/\mu$, $\mu=0.9$")
    ax.plot([], [], color=C.CREDIT, lw=0.9, ls=(0, (3, 1.5)), label=r"$k_t^2/\mu$, $\mu=0.8$")
    ax.plot([], [], color=C.BURDEN, lw=0.9, label=r"$0.854\,k_t$ (transmittance)")
    ax.legend(loc="upper left", fontsize=5.5, borderaxespad=0.3)
    ax.grid(False)
    cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.95)
    cb.set_label("daytime hours, ten sites", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.4)
    P.panel_label(ax, "c")
    # ------------------------------------------------------------ (d) kernel formulations
    ax = axs[1, 0]
    x = np.arange(len(keys))
    series = [("K_aprp_snowmasked", "hourly APRP, snow-masked (used)", C.CREDIT),
              ("K_aprp_monthlymeans", "APRP on monthly means", C.STRUCT),
              ("K_simple", r"simple, $T_a=0.854$", C.WARM),
              ("K_clear", "clear sky", C.GREY)]
    w = 0.2
    for i, (q, lab, c) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, [-kj[k][q] / -kj[k]["K_aprp_hourly"] for k in keys], w, color=c, label=lab)
    ax.axhline(1.0, color=C.INK, lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([C.SITE_SHORT[k] for k in keys], rotation=90, fontsize=5.8)
    ax.set_ylim(0.6, 2.75)
    ax.set_ylabel(r"Ratio to hourly APRP (--)")
    ax.legend(loc="upper right", fontsize=5.3, ncol=1, borderaxespad=0.3, handlelength=1.2, labelspacing=0.3)
    P.panel_label(ax, "d")
    # ------------------------------------------------------------ (e) delivered fraction
    ax = axs[1, 1]
    tau = np.logspace(np.log10(0.2), np.log10(20), 200)
    Tr = np.logspace(0, np.log10(20), 200)
    TT, RR = np.meshgrid(tau, Tr)
    PHI = ledger.delivered_fraction(RR, TT)
    cs = ax.contourf(TT, RR, PHI, levels=np.linspace(0, 1, 11), cmap=ccm.oslo_r, vmin=-0.1, vmax=1.1)
    cl = ax.contour(TT, RR, PHI, levels=[0.2, 0.4, 0.6, 0.8], colors=[C.INK, C.INK, "white", "white"], linewidths=0.4)
    ax.clabel(cl, fmt="%.1f", fontsize=5.4, inline_spacing=2, manual=[(1.6, 8.0), (3.6, 8.0), (7.0, 8.0), (15.0, 8.0)])
    ax.plot(REF.tau, REF.Tr, "*", ms=7, color=C.GOLD, mec=C.INK, mew=0.4)
    ax.annotate(rf"reference: $\phi={ledger.delivered_fraction(REF.Tr, REF.tau):.2f}$", xy=(REF.tau, REF.Tr),
                xytext=(0.26, 11.0), fontsize=5.6, color="white",
                arrowprops=dict(arrowstyle="-", lw=0.4, color="white"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.set_yticklabels(["1", "2", "5", "10", "20"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"Decay time of the coating, $\tau$ (yr)")
    ax.set_ylabel(r"Renewal interval, $T_r$ (yr)")
    ax.grid(False)
    cb = fig.colorbar(cs, ax=ax, pad=0.02, shrink=0.95)
    cb.set_label(r"Delivered fraction $\phi$", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.6)
    P.panel_label(ax, "e")
    # ------------------------------------------------------------ (f) reconciliation of coefficients
    ax = axs[1, 2]
    conv = [("static", r"GWP$_{100}$, static"), ("dynamic", r"GWP$_{100}$, dynamic"),
            ("stock_af055", r"stock, AF 0.55"), ("gtp_sustained_80", r"GTP, sustained 80\,yr")]
    y = 0
    ylabels, yt = [], []
    for q, lab in conv:
        v = np.array([lg[k]["coefficient"][q] for k in keys])
        ax.plot([v.min(), v.max()], [y, y], color=C.CREDIT, lw=3.0, solid_capstyle="butt", alpha=0.35)
        ax.scatter(v, np.full(len(v), y), s=9, c=[cols[k] for k in keys], edgecolors=C.INK, linewidths=0.3, zorder=3)
        ylabels.append(lab)
        yt.append(y)
        y -= 1
    y -= 0.4
    for lab, lo, hi in LIT:
        if lab.startswith("VanCuren"):
            ax.annotate("", xy=(0.12, y), xytext=(hi, y), arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.2",
                                                                          color=C.BURDEN, lw=1.6, shrinkA=0, shrinkB=0))
            ax.plot([hi, hi], [y - 0.18, y + 0.18], color=C.BURDEN, lw=1.0)
        elif hi > lo:
            ax.plot([lo, hi], [y, y], color=C.BURDEN, lw=3.0, solid_capstyle="butt", alpha=0.8)
        else:
            ax.plot([lo], [y], "D", ms=3.4, color=C.BURDEN)
        ylabels.append(lab)
        yt.append(y)
        y -= 1
    ax.set_yticks(yt)
    ax.set_yticklabels(ylabels, fontsize=5.6)
    ax.set_xlim(0, 8)
    ax.set_ylim(y + 0.4, 0.7)
    ax.axhline(-len(conv) + 0.3, color=C.GREY, lw=0.4)
    ax.text(7.9, -0.5, "this study,\nten sites", ha="right", va="center", fontsize=5.6, color=C.CREDIT)
    ax.text(7.9, yt[-1] + 0.2, "published", ha="right", va="center", fontsize=5.6, color=C.BURDEN)
    ax.set_xlabel(r"kg CO$_2$e\,m$^{-2}$ per 0.01 albedo")
    ax.grid(axis="y", lw=0)
    P.panel_label(ax, "f")
    P.save(fig, FIG / "fig_kernel")


if __name__ == "__main__":
    main()

"""Figure: the season of the credit and the timing of applications.

(a-d) Monthly snow-masked kernel over the contiguous United States in
January, April, July and October (grid calibrated to the stations).
(e) Credit of a coat renewed every three years as a function of the month
of application, relative to the mean over all application dates, at the
ten stations (daily kernel of the seven ledger years).
(f) Share of the credit of a coat applied at the best date that is lost
when it is applied on 1 October instead; (g) best month of application
(grid, monthly kernel climatology).  Writes results/season_v3.json.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmcrameri import cm
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import ledger, plotting as P, palette as C  # noqa: E402
from alrb.config import FIG, REF, RES, SITES  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_conus import albers, cells, polygons  # noqa: E402

TAG = "v3"
W = np.exp(-np.arange(int(REF.Tr * 365.25)) / (REF.tau * 365.25))


def timing(k, dates):
    """Relative credit of a coat applied on the first day of each month."""
    kk = np.concatenate([k, k, k, k])
    base = np.mean(k) * W.sum()
    out = []
    for m in range(1, 13):
        starts = np.where((dates.month == m) & (dates.day == 1))[0]
        out.append(np.mean([np.sum(kk[i:i + len(W)] * W) for i in starts]) / base)
    return np.array(out)


def draw(ax, polys, vals, cmap, norm, states, neigh, border):
    fc = cmap(norm(vals))
    ax.add_collection(PolyCollection(neigh, facecolor="#F4F5F6", edgecolor="#C9CDD2", linewidth=0.3))
    ax.add_collection(PolyCollection(polys, facecolor=fc, edgecolor=fc, linewidth=0.3))
    ax.add_collection(PolyCollection(states, facecolor="none", edgecolor="white", linewidth=0.3))
    ax.add_collection(PolyCollection(border, facecolor="none", edgecolor="#8A9199", linewidth=0.35))
    ax.set_xlim(-2380, 2280)
    ax.set_ylim(-1720, 1480)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    # ------------------------------------------------------------------ data
    st = {}
    for s in SITES:
        K = ledger.daily_kernel(s.key, 1.0)
        st[s.key] = timing(K.to_numpy(), K.index)
    z = np.load(RES / f"conus_ledger_{TAG}.npz")
    zk = np.load(RES / "conus_kernel_v3.npz")
    ratio = float(z["ratio"])
    lat, lon, land = z["lat"], z["lon"], z["land"]
    ok = land & np.isfinite(z["R_static"])
    days = pd.date_range("2017-05-01", "2024-04-30", freq="D")
    mon = days.month.to_numpy() - 1
    Kmon = zk["Kmon"] * ratio
    pen = np.full(land.shape, np.nan)
    best = np.full(land.shape, np.nan)
    fmay = np.full(land.shape, np.nan)
    for a, b in zip(*np.where(ok)):
        f = timing(Kmon[mon, a, b], days)
        pen[a, b] = 100 * (1 - f[9] / f.max())
        best[a, b] = int(np.argmax(f)) + 1
        fmay[a, b] = 100 * (1 - f[4] / f.max())
    res = dict(stations={k: v.tolist() for k, v in st.items()},
               grid=dict(penalty_oct=dict(min=float(np.nanmin(pen)), max=float(np.nanmax(pen)),
                                          med=float(np.nanmedian(pen))),
                         loss_may=dict(max=float(np.nanmax(fmay)), med=float(np.nanmedian(fmay))),
                         best_month_share={str(m): float(np.mean(best[ok] == m)) for m in range(1, 13)}))
    (RES / f"season_{TAG}.json").write_text(json.dumps(res, indent=1))
    # ------------------------------------------------------------------ figure
    polys, idx = cells(lat, lon, ok)
    ia = np.array([i for i, _ in idx])
    ib = np.array([j for _, j in idx])
    lower = lambda p: p.get("adm0_a3") == "USA" and p.get("name") not in ("Alaska", "Hawaii")  # noqa: E731
    states = polygons("ne_50m_admin_1_states_provinces_lakes.geojson", lower)
    neigh = polygons("ne_50m_admin_0_countries_lakes.geojson", lambda p: p.get("ADM0_A3") in ("CAN", "MEX"))
    border = polygons("ne_50m_admin_0_countries_lakes.geojson", lambda p: p.get("ADM0_A3") == "USA")
    P.setup()
    fig = plt.figure(figsize=(P.DOUBLE, 4.3))
    top = fig.add_gridspec(1, 4, left=0.0, right=1.0, bottom=0.60, top=0.965, wspace=0.02)
    norm = matplotlib.colors.Normalize(0.0, 3.2)
    cmap = cm.lajolla_r
    for i, (m, name) in enumerate(((1, "January"), (4, "April"), (7, "July"), (10, "October"))):
        ax = fig.add_subplot(top[0, i])
        draw(ax, polys, -Kmon[m - 1][ia, ib], cmap, norm, states, neigh, border)
        ax.set_title(rf"\textbf{{({'abcd'[i]})}} {name}", loc="left", fontsize=7.0, pad=1.5)
    cax = fig.add_axes([0.30, 0.575, 0.40, 0.016])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.8, 1.6, 2.4, 3.2])
    cb.ax.tick_params(labelsize=5.8, length=2, pad=1.2)
    cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.set_label(r"Monthly mean $|K|$ (W\,m$^{-2}$ per 0.01 albedo), street snow-masked", fontsize=6.0,
                 labelpad=1.5)
    bot = fig.add_gridspec(1, 3, left=0.065, right=0.995, bottom=0.075, top=0.455, wspace=0.08,
                           width_ratios=[1.0, 1.05, 1.05])
    # (e) station timing curves
    ax = fig.add_subplot(bot[0, 0])
    months = np.arange(1, 13)
    for k in C.SITE_ORDER:
        ax.plot(months, 100 * (st[k] - 1), color=C.site_colors()[k], lw=0.9, marker="o", ms=1.6,
                label=C.SITE_SHORT[k])
    ax.axhline(0, color=C.INK, lw=0.5)
    ax.plot([5, 5], [-22, 19], color=C.GREY, lw=0.6, ls=(0, (3, 1.5)))
    ax.set_xticks(months)
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(-22, 30)
    ax.set_xlabel("Month of application")
    ax.set_ylabel(r"Credit relative to the mean (\%)")
    ax.legend(ncol=5, fontsize=5.0, loc="upper center", borderaxespad=0.25, handlelength=1.0, columnspacing=0.5,
              handletextpad=0.25, labelspacing=0.2)
    P.panel_label(ax, "e")
    # (f) October penalty
    ax = fig.add_subplot(bot[0, 1])
    norm2 = matplotlib.colors.Normalize(10, 35)
    cmap2 = cm.acton_r
    draw(ax, polys, pen[ia, ib], cmap2, norm2, states, neigh, border)
    ax.set_title(r"\textbf{(f)} credit lost by applying on 1 October", loc="left", fontsize=6.6, pad=1.5)
    c2 = ax.inset_axes([0.05, 0.075, 0.36, 0.035])
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap2, norm=norm2), cax=c2, orientation="horizontal")
    cb.set_ticks([10, 20, 30])
    cb.ax.tick_params(labelsize=5.6, length=2, pad=1.2)
    cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.set_label(r"\% of the best date", fontsize=5.8, labelpad=1.2)
    cb.ax.xaxis.set_label_position("top")
    # (g) best month
    ax = fig.add_subplot(bot[0, 2])
    cols = {3: "#8FB9D6", 4: "#2E86AB", 5: "#1B3A5C", 6: "#C9A227"}
    bv = best[ia, ib]
    cm3 = matplotlib.colors.ListedColormap([cols[3], cols[4], cols[5], cols[6]])
    norm3 = matplotlib.colors.BoundaryNorm([2.5, 3.5, 4.5, 5.5, 6.5], 4)
    draw(ax, polys, bv, cm3, norm3, states, neigh, border)
    ax.set_title(r"\textbf{(g)} best month of application", loc="left", fontsize=6.6, pad=1.5)
    hs = [matplotlib.patches.Patch(color=cols[m], label=n) for m, n in ((3, "March"), (4, "April"), (5, "May"),
                                                                          (6, "June")) if np.any(bv == m)]
    ax.legend(handles=hs, loc="lower left", fontsize=5.6, borderaxespad=0.4, handlelength=1.0, ncol=2,
              columnspacing=0.6)
    P.save(fig, FIG / "fig_season")
    print(json.dumps(res["grid"], indent=0))


if __name__ == "__main__":
    main()

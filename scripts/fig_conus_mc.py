"""Figure: streets of varied form and material in the cells of the contiguous
United States (Monte Carlo of 82_conus_mc.py) and the distortion of the
reference rebound ratio when the parameters of a street deviate from the
reference."""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import plotting as P, palette as C  # noqa: E402
from alrb.config import FIG, RES  # noqa: E402

spec = importlib.util.spec_from_file_location("fig_conus", Path(__file__).with_name("fig_conus.py"))
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)

TAG = "v3"
CASES = [("lcz", "paved", "LCZ\nform"), ("lowrise", "paved", "assumed\nlow-rise"), ("compact", "paved", "assumed\ncompact"),
         ("open", "paved", "open\nstreets"), ("open", "sunny", "open streets,\nsunniest cells")]


def main():
    z = np.load(RES / f"conus_mc_{TAG}.npz")
    js = json.loads((RES / f"conus_mc_{TAG}.json").read_text())
    zl = np.load(RES / f"conus_ledger_{TAG}.npz")
    lat, lon = z["lat"], z["lon"]
    ok = np.zeros(zl["land"].shape, bool)
    ok[z["ia"], z["ib"]] = True
    polys, idx = fc.cells(lat, lon, ok)
    order = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(z["ia"], z["ib"]))}
    sel = np.array([order[i] for i in idx])
    lower = lambda p: p.get("adm0_a3") == "USA" and p.get("name") not in ("Alaska", "Hawaii")  # noqa: E731
    states = fc.polygons("ne_50m_admin_1_states_provinces_lakes.geojson", lower)
    neigh = fc.polygons("ne_50m_admin_0_countries_lakes.geojson", lambda p: p.get("ADM0_A3") in ("CAN", "MEX"))
    border = fc.polygons("ne_50m_admin_0_countries_lakes.geojson", lambda p: p.get("ADM0_A3") == "USA")
    P.setup()
    fig = plt.figure(figsize=(P.DOUBLE, 4.55))
    gm = fig.add_gridspec(1, 2, left=0.0, right=0.995, bottom=0.455, top=0.99, wspace=0.02)
    # (a) median rebound ratio of the sampled streets, renewal every three years
    ax = fig.add_subplot(gm[0, 0])
    norm = matplotlib.colors.Normalize(2.0, 6.0)
    fc.draw_map(ax, polys, z["B_lcz_R50"][sel], cm.bilbao, norm, states, neigh, border)
    fc.colourbar(fig, ax, cm.bilbao, norm, r"Median dynamic $R$ of the streets of a cell", [2, 3, 4, 5, 6])
    ax.text(0.005, 0.995, r"\textbf{(a)}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
    # (b) share of the streets that are net sources, renewal interval sampled
    ax = fig.add_subplot(gm[0, 1])
    v = 100 * z["A_lcz_psource"][sel]
    lo = 5 * np.floor(v.min() / 5)
    norm = matplotlib.colors.Normalize(lo, 100.0)
    fc.draw_map(ax, polys, v, cm.lajolla, norm, states, neigh, border)
    fc.colourbar(fig, ax, cm.lajolla, norm, r"Treated streets that are net sources (\%), $T_r$ sampled",
                 list(np.arange(lo, 100.1, 5 if 100 - lo <= 30 else 10)))
    ax.text(0.005, 0.995, r"\textbf{(b)}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
    gs = fig.add_gridspec(1, 3, left=0.062, right=0.99, bottom=0.085, top=0.405, wspace=0.40)
    # (c) distortion against the aspect ratio
    ax = fig.add_subplot(gs[0, 0])
    x, q = z["dist_h_x"], z["dist_h_q"]
    ax.fill_between(x, q[0], q[2], color=C.BURDEN, alpha=0.25, linewidth=0)
    ax.plot(x, q[1], color=C.BURDEN)
    ax.set_yscale("log")
    ax.set_xlim(0, 2)
    ax.set_ylim(0.9, 20)
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 2, 5, 10, 20]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["1", "2", "5", "10", "20"]))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"Aspect ratio $H/W$ of the street")
    ax.set_ylabel(r"$R/R_{\mathrm{ref}}$")
    ax.text(0.05, 0.93, "median and 5--95\\,\\% of the cells", transform=ax.transAxes, ha="left", va="top",
            fontsize=6.0, color=C.GREY)
    P.panel_label(ax, "c")
    # (d) distortion against the decay time and the renewal interval
    ax = fig.add_subplot(gs[0, 1])
    for name, col, lab in (("tau", C.CREDIT, r"decay time $\tau$"), ("Tr", C.INK, r"renewal interval $T_r$")):
        x, q = z[f"dist_{name}_x"], z[f"dist_{name}_q"]
        if name == "Tr":                      # whole years: every coat is laid on 1 May
            k = np.isclose(x, np.round(x))
            x, q = x[k], q[:, k]
        ax.fill_between(x, q[0], q[2], color=col, alpha=0.25, linewidth=0)
        ax.plot(x, q[1], color=col, label=lab, marker="o" if name == "Tr" else None, ms=2.0)
    ax.axhline(1.0, color=C.GREY, lw=0.5, zorder=0)
    ax.plot([0.91, 3.0], [1.0, 1.0], ls="none", marker="o", ms=3.0, mfc="white", mec=C.INK, mew=0.6, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 10)
    ax.set_ylim(0.3, 1.8)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 3, 5, 10]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(["0.5", "1", "2", "3", "5", "10"]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$\tau$ or $T_r$ (yr)")
    ax.set_ylabel(r"$R/R_{\mathrm{ref}}$")
    ax.legend(loc="upper right", fontsize=6.0, borderaxespad=0.3, handlelength=1.4)
    P.panel_label(ax, "d")
    # (e) national balance per square metre of treated street
    ax = fig.add_subplot(gs[0, 2])
    for k, (mix, wn, _) in enumerate(CASES):
        for design, dy, col in (("B", 0.16, C.BURDEN), ("A", -0.16, C.CREDIT)):
            d = js["part1"][design][mix][wn]["dynamic"]
            y = len(CASES) - 1 - k + dy
            ax.plot([d["net"][0], d["net"][4]], [y, y], color=col, lw=1.0, solid_capstyle="butt")
            ax.plot([d["net"][1], d["net"][3]], [y, y], color=col, lw=3.0, solid_capstyle="butt")
            ax.plot(d["net"][2], y, marker="o", ms=3.2, mfc="white", mec=col, mew=0.8, ls="none")
    ax.axvline(0.0, color=C.GREY, lw=0.5, zorder=0)
    ax.set_yticks(range(len(CASES)))
    ax.set_yticklabels([c[2] for c in CASES[::-1]], linespacing=0.95)
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_ylim(-0.55, len(CASES) + 0.25)
    lo = min(js["part1"][d][m][w]["dynamic"]["net"][0] for d in ("A", "B") for m, w, _ in CASES)
    hi = max(js["part1"][d][m][w]["dynamic"]["net"][4] for d in ("A", "B") for m, w, _ in CASES)
    ax.set_xlim(10 * np.floor(lo / 10) - 2, 10 * np.ceil(hi / 10) + 12)
    ax.set_xlabel(r"National balance (kg\,CO$_2$e\,m$^{-2}$)")
    ax.plot([], [], color=C.BURDEN, lw=3.0, label=r"$T_r=3$\,yr")
    ax.plot([], [], color=C.CREDIT, lw=3.0, label=r"$T_r$ 1--10\,yr")
    ax.legend(loc="upper right", fontsize=6.0, borderaxespad=0.3, handlelength=1.2, ncol=2, columnspacing=1.0)
    P.panel_label(ax, "e")
    P.save(fig, FIG / "fig_conus_mc")


if __name__ == "__main__":
    main()

"""Figure: structural response to the surface albedo (reference structure,
traffic calibrated to a 20-year untreated life at every site)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import plotting as P, palette as C  # noqa: E402
from alrb.config import RES, FIG, ALBEDO_GRID  # noqa: E402

TAG = "v3"


def main():
    P.setup()
    st = json.loads((RES / f"structural_{TAG}.json").read_text())
    tv = json.loads((RES / f"timevarying_check_{TAG}.json").read_text())
    cols = C.site_colors()
    keys = C.SITE_ORDER
    al = np.array(ALBEDO_GRID)
    get = lambda k, q: np.array([st[k]["albedo"][f"{a:.2f}"][q] for a in al])
    fig, axs = plt.subplots(2, 3, figsize=(P.DOUBLE, 4.35), constrained_layout=True)
    # ------------------------------------------------------------ (a) bound-layer modulus
    ax = axs[0, 0]
    for k in keys:
        ax.plot(al, get(k, "E_ac_mean") / 1000, color=cols[k], lw=0.9, marker="o", ms=1.6)
    ax.axvline(0.14, color=C.GREY, lw=0.5, ls=(0, (2, 2)))
    ax.axvline(0.45, color=C.GREY, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(0.03, 0.67)
    ax.set_ylim(3.5, 15.5)
    ax.set_xlabel(r"Surface albedo, $\alpha$ (--)")
    ax.set_ylabel(r"Mean asphalt modulus $|E^*|$ (GPa)")
    ax.text(0.145, 15.1, r"$\alpha_0$", fontsize=5.8, ha="left", va="top", color=C.GREY)
    ax.text(0.455, 15.1, r"$\alpha_f$", fontsize=5.8, ha="left", va="top", color=C.GREY)
    P.panel_label(ax, "a")
    # ------------------------------------------------------------ (b) tensile strain
    ax = axs[0, 1]
    for k in keys:
        ax.plot(al, get(k, "eps_t_trafficmean") * 1e6, color=cols[k], lw=0.9, marker="o", ms=1.6)
    ax.set_xlim(0.03, 0.67)
    ax.set_ylim(80, 275)
    ax.set_xlabel(r"Surface albedo, $\alpha$ (--)")
    ax.set_ylabel(r"Traffic-weighted $\varepsilon_t$ ($\mu\varepsilon$)")
    P.panel_label(ax, "b")
    # ------------------------------------------------------------ (c) fatigue life
    ax = axs[0, 2]
    for k in keys:
        ax.plot(al, get(k, "life_fatigue"), color=cols[k], lw=0.9, marker="o", ms=1.6)
    ax.axhline(20, color=C.GREY, lw=0.5)
    ax.set_xlim(0.03, 0.67)
    ax.set_ylim(17, 34)
    ax.set_xlabel(r"Surface albedo, $\alpha$ (--)")
    ax.set_ylabel("Fatigue life (years)")
    ax.text(0.36, 19.6, r"calibration: 20\,yr at $\alpha_0$", fontsize=5.6, color=C.GREY, va="top")
    P.panel_label(ax, "c")
    # ------------------------------------------------------------ (d) subgrade
    ax = axs[1, 0]
    for k in keys:
        ax.plot(al, get(k, "life_subgrade") / get(k, "life_subgrade")[2], color=cols[k], lw=0.9, marker="o", ms=1.6)
    ax.axhline(1.0, color=C.GREY, lw=0.5)
    ax.set_xlim(0.03, 0.67)
    ax.set_ylim(0.85, 1.6)
    ax.set_xlabel(r"Surface albedo, $\alpha$ (--)")
    ax.set_ylabel(r"Subgrade life, $L_d(\alpha)/L_d(\alpha_0)$")
    ax2 = ax.inset_axes([0.10, 0.60, 0.40, 0.34])
    for k in ("goodridge", "wolfpoint", "ithaca"):
        ax2.plot(al, get(k, "E_sg_mean"), color=cols[k], lw=0.8)
    ax2.set_xlim(0.03, 0.67)
    ax2.set_ylim(60, 340)
    ax2.set_xticks([0.1, 0.3, 0.5])
    ax2.set_yticks([100, 200, 300])
    ax2.tick_params(labelsize=5.0, length=1.5, pad=1.0)
    ax2.set_title(r"mean subgrade modulus (MPa)", fontsize=5.2, pad=1.5)
    ax2.grid(False)
    P.panel_label(ax, "d")
    # ------------------------------------------------------------ (e) rutting-governed scenario
    ax = axs[1, 1]
    for k in keys:
        L = np.minimum(get(k, "life_rut"), 150.0)
        ok = L < 149.9
        ax.plot(al[ok], L[ok], color=cols[k], lw=0.9, marker="o", ms=1.6)
    ax.set_yscale("log")
    ax.set_xlim(0.03, 0.67)
    ax.set_ylim(8, 160)
    ax.set_yticks([10, 20, 50, 100, 150])
    ax.set_yticklabels(["10", "20", "50", "100", "150"])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"Surface albedo, $\alpha$ (--)")
    ax.set_ylabel("Rutting life, scenario (years)")
    ax.text(0.05, 0.95, r"6.35\,mm limit, MEPDG" "\n" r"permanent deformation", transform=ax.transAxes, fontsize=5.6,
            va="top")
    P.panel_label(ax, "e")
    # ------------------------------------------------------------ (f) harmonic-life check
    ax = axs[1, 2]
    x = np.arange(len(keys))
    eh = [100 * tv[k]["rel_err_harmonic"] for k in keys]
    eq = [100 * tv[k]["rel_err_quasistatic"] for k in keys]
    em = [100 * (tv[k]["life_at_mean_albedo"] / tv[k]["life_direct"] - 1) for k in keys]
    ax.bar(x - 0.26, eh, 0.25, color=C.STRUCT, label=r"harmonic $L_{\mathrm{eff}}$")
    ax.bar(x, eq, 0.25, color=C.CREDIT, label="quasi-static, hourly")
    ax.bar(x + 0.26, em, 0.25, color=C.GREY, label=r"life at mean $\alpha$")
    ax.axhline(0, color=C.INK, lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([C.SITE_SHORT[k] for k in keys], rotation=90, fontsize=5.8)
    ax.set_ylim(-0.5, 2.9)
    ax.set_ylabel(r"Difference from direct run (\%)")
    ax.legend(loc="upper right", fontsize=5.6, borderaxespad=0.3, ncol=1)
    P.panel_label(ax, "f")
    hs = [matplotlib.lines.Line2D([], [], color=cols[k], lw=1.2, marker="o", ms=2.2) for k in keys]
    fig.legend(hs, [C.SITE_SHORT[k] for k in keys], loc="outside lower center", ncol=10, fontsize=5.8,
               handlelength=1.6, columnspacing=1.1, frameon=False)
    P.save(fig, FIG / "fig_structural")


if __name__ == "__main__":
    main()

"""Figure: verification of the thermal and structural solvers."""
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


def main():
    P.setup()
    vt = json.loads((RES / "verification_thermal.json").read_text())
    vp = json.loads((RES / "verification_thermal_profiles.json").read_text())
    vm = json.loads((RES / "verification_mechanics.json").read_text())
    fig, axs = plt.subplots(2, 3, figsize=(P.DOUBLE, 4.3), constrained_layout=True)
    # ------------------------------------------------------------ (a) periodic conduction
    ax = axs[0, 0]
    cols = [C.WARM, C.STRUCT, C.CREDIT]
    keys = list(vp["harmonic"])[::2][:3]
    for c, k in zip(cols, keys):
        d = vp["harmonic"][k]
        t = np.arange(len(d["exact"])) + 0.5
        ax.plot(t, d["exact"], color=c, lw=0.9, label=f"{float(k) * 1000:.0f} mm")
        ax.plot(t[::3], np.array(d["model"])[::3], "o", ms=2.2, mfc="white", mec=c, mew=0.6)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(r"$T$ ($^{\circ}$C)")
    ax.set_xlim(0, 48)
    ax.set_ylim(8, 33)
    ax.set_xticks([0, 12, 24, 36, 48])
    ax.legend(loc="upper right", ncol=3, fontsize=5.8, handlelength=1.2, columnspacing=0.8, borderaxespad=0.3)
    err = max(vt["harmonic"]["max_abs_err"])
    ax.text(0.03, 0.05, rf"line: exact; circles: model; max.\ error {err:.3f} K", transform=ax.transAxes,
            fontsize=5.8, va="bottom")
    P.panel_label(ax, "a")
    # ------------------------------------------------------------ (b) Neumann freezing front
    ax = axs[0, 1]
    nf = vp["neumann_front"]
    t = np.array(nf["t_h"]) / 24.0
    ax.plot(t, np.array(nf["exact"]) * 100, color=C.FROST, lw=1.0, label="Neumann solution")
    ax.plot(t[6::18], np.array(nf["model"])[6::18] * 100, "o", ms=2.3, mfc="white", mec=C.INK, mew=0.6,
            label="column model")
    ax.set_xlabel("Time (d)")
    ax.set_ylabel("Frost depth (cm)")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=5.8, borderaxespad=0.3)
    nm = vt["neumann"]
    ax.text(0.04, 0.95, rf"$\lambda={nm['lam']:.4f}$; front error $\leq$ {nm['front_max_abs_err'] * 1000:.0f} mm"
            "\n" rf"profile error $\leq$ {nm['T_max_abs_err']:.3f} K", transform=ax.transAxes, fontsize=5.8, va="top")
    P.panel_label(ax, "b")
    # ------------------------------------------------------------ (c) Boussinesq profiles
    ax = axs[0, 2]
    b = vm["boussinesq"]
    z = np.array(b["z"]) * 100
    q = 690e3
    ax.plot(-np.array(b["sz_exact"]) / q, z, color=C.CREDIT, lw=1.0, label=r"$-\sigma_z/q$")
    ax.plot(-np.array(b["sr_exact"]) / q, z, color=C.WARM, lw=1.0, label=r"$-\sigma_r/q$")
    ax.plot(-np.array(b["sz_let"])[::2] / q, z[::2], "o", ms=2.2, mfc="white", mec=C.CREDIT, mew=0.6)
    ax.plot(-np.array(b["sr_let"])[::2] / q, z[::2], "s", ms=2.0, mfc="white", mec=C.WARM, mew=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Normalised stress (--)")
    ax.set_ylabel("Depth (cm)")
    ax.set_xlim(-0.1, 1.05)
    ax.legend(loc="lower right", fontsize=6, borderaxespad=0.3)
    ax.text(0.97, 0.52, "lines: closed form\nmarkers: layered solution", transform=ax.transAxes, fontsize=5.8,
            ha="right", va="center")
    P.panel_label(ax, "c")
    # ------------------------------------------------------------ (d) layered solution vs FE
    ax = axs[1, 0]
    mk = {"ordinary": ("o", C.STRUCT), "frozen foundation": ("^", C.FROST), "hot bound layer": ("s", C.WARM)}
    allv = []
    for name, d in vm["fe"].items():
        x = np.array([p["fe"] for p in d["points"]]) * 1e6
        y = np.array([p["let"] for p in d["points"]]) * 1e6
        allv += list(np.abs(x))
        ax.plot(np.abs(x), np.abs(y), mk[name][0], ms=3.0, mfc="white", mec=mk[name][1], mew=0.7, label=name)
    lim = [2, 2000]
    ax.plot(lim, lim, color=C.GREY, lw=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks([10, 100, 1000])
    ax.set_yticks([10, 100, 1000])
    rel = max(abs(p["let"] / p["fe"] - 1) for d in vm["fe"].values() for p in d["points"])
    ax.text(0.97, 0.05, rf"max.\ difference {100 * rel:.1f}\,\%", transform=ax.transAxes, fontsize=5.8,
            ha="right", va="bottom")
    ax.set_xlabel(r"Finite elements, $|\varepsilon|$ ($\mu\varepsilon$)")
    ax.set_ylabel(r"Layered solution, $|\varepsilon|$ ($\mu\varepsilon$)")
    ax.legend(loc="upper left", fontsize=5.8, borderaxespad=0.3, handletextpad=0.2)
    P.panel_label(ax, "d")
    # ------------------------------------------------------------ (e) strain table accuracy
    ax = axs[1, 1]
    tb = vm["table"]
    ks = C.SITE_ORDER
    xx = np.arange(len(ks))
    ax.bar(xx - 0.18, [100 * (tb[k]["fatigue_ratio"] - 1) for k in ks], 0.34, color=C.CREDIT, label="fatigue damage")
    ax.bar(xx + 0.18, [100 * (tb[k]["subgrade_ratio"] - 1) for k in ks], 0.34, color=C.WARM, label="subgrade damage")
    ax.axhline(0, color=C.INK, lw=0.5)
    ax.set_xticks(xx)
    ax.set_xticklabels([C.SITE_SHORT[k] for k in ks], rotation=90, fontsize=5.8)
    ax.set_ylabel(r"Table $-$ direct (\%)")
    ax.set_ylim(-0.5, 0.5)
    ax.legend(loc="lower right", fontsize=5.8, ncol=2, borderaxespad=0.3, columnspacing=0.8)
    P.panel_label(ax, "e")
    # ------------------------------------------------------------ (f) convergence
    ax = axs[1, 2]
    conv = vt["convergence"]
    ref = conv["ref"]
    variants = [("dt_60min", r"$\Delta t$ 60 min"), ("dt_5min", r"$\Delta t$ 5 min"), ("coarse_grid", "coarse grid"),
                ("fine_grid", "fine grid")]
    qs = [("dts_mean", r"$\Delta\bar T_s$"), ("dts_max", r"$\Delta T_{s,\max}$"), ("dT085", r"$\Delta\bar T_{0.85}$")]
    w = 0.2
    for i, (qk, ql) in enumerate(qs):
        vals = [100 * (conv[v][qk] / ref[qk] - 1) for v, _ in variants]
        ax.bar(np.arange(len(variants)) + (i - 1) * w, vals, w, color=[C.CREDIT, C.WARM, C.STRUCT][i], label=ql)
    ax.axhline(0, color=C.INK, lw=0.5)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels([v[1] for v in variants], fontsize=5.8)
    ax.set_ylabel(r"Change from reference (\%)")
    ax.set_ylim(-1.2, 1.2)
    ax.legend(loc="upper right", fontsize=5.8, ncol=3, borderaxespad=0.3, columnspacing=0.6, handlelength=1.0)
    P.panel_label(ax, "f")
    P.save(fig, FIG / "fig_verification")


if __name__ == "__main__":
    main()

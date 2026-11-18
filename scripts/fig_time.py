"""Figure: the time structure of the balance -- delivered albedo, forcing
and temperature response of the credit and of the burden over 150 years,
and the CO2 equivalents as functions of the metric horizon."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import ledger, metrics, plotting as P, palette as C  # noqa: E402
from alrb.config import A_EARTH, RES, FIG, REF  # noqa: E402

SITE_KEY = "durham"
TAG = "v3"


def main():
    P.setup()
    Kd = ledger.daily_kernel(SITE_KEY, 1.0).to_numpy()
    p = ledger.Params()
    # daily albedo increment over the first 12 years (for panel a)
    days = np.arange(int(12 * 365.25))
    ph = np.mod(days / 365.25, p.Tr)
    da = (p.alpha_inf - p.alpha0) + (p.alphaf - p.alpha_inf) * np.exp(-ph / p.tau)
    # annual forcing of the albedo path and emissions
    f_loc = ledger._annual_forcing(Kd, len(Kd), p.alpha0, p.alphaf, p.alpha_inf, p.tau, p.Tr, p.H, 365.25)
    f_alb = f_loc / A_EARTH                                    # W m-2 per m2 treated
    t_app, w_app = ledger.applications(p.Tr, p.H)
    m_app = p.g_co * w_app
    T = 150
    t = np.arange(T * 12 + 1) / 12.0
    F_alb = np.where(t < p.H, np.interp(np.minimum(t, p.H - 1e-9), np.arange(len(f_alb)) + 0.5, f_alb), 0.0)
    F_co = np.zeros_like(t)
    for tk, mk in zip(t_app, m_app):
        on = t >= tk
        F_co[on] += mk * metrics.RE_KG * metrics.irf(t[on] - tk)
    dt = t[1] - t[0]
    Rt = metrics.temp_response_kernel(t)

    def conv(F):
        return np.convolve(F, Rt)[: len(t)] * dt
    T_alb, T_co = conv(F_alb), conv(F_co)
    fig, axs = plt.subplots(2, 2, figsize=(P.DOUBLE, 3.9), constrained_layout=True)
    # ------------------------------------------------------------ (a) delivered albedo
    ax = axs[0, 0]
    yrs = days / 365.25
    ax.plot(yrs, p.alpha0 + da, color=C.INK, lw=0.8, label="treated surface")
    phi = ledger.delivered_fraction(p.Tr, p.tau)
    mean_a = p.alpha0 + (p.alphaf - p.alpha0) * phi
    ax.axhline(p.alphaf, color=C.BURDEN, lw=0.7, ls=(0, (4, 2)), label=r"as laid, $\alpha_f$")
    ax.axhline(mean_a, color=C.CREDIT, lw=0.9, label=rf"delivered mean ($\phi={phi:.2f}$)")
    ax.axhline(p.alpha0, color=C.GREY, lw=0.7, label=r"untreated, $\alpha_0$")
    ax.fill_between(yrs, p.alpha0, p.alpha0 + da, color=C.CREDIT, alpha=0.12, lw=0)
    ax.set_xlim(0, 12)
    ax.set_ylim(0.08, 0.62)
    ax.set_xlabel("Years since first application")
    ax.set_ylabel(r"Surface albedo (--)")
    ax.legend(loc="upper right", fontsize=5.8, ncol=2, borderaxespad=0.3, columnspacing=0.8, handlelength=1.4)
    P.panel_label(ax, "a")
    # ------------------------------------------------------------ (b) forcing
    ax = axs[0, 1]
    sc = 1e15
    # renewal-cycle means (the annual values follow the sawtooth of panel a)
    ncyc = int(np.ceil(p.H / p.Tr))
    cyc = np.floor(np.minimum(t, p.H - 1e-9) / p.Tr).astype(int)
    Fm = np.array([F_alb[(cyc == c) & (t < p.H)].mean() for c in range(ncyc)])
    F_alb_cm = np.where(t < p.H, Fm[np.minimum(cyc, ncyc - 1)], 0.0)
    ax.fill_between(t, 0, F_alb * sc, color=C.CREDIT, alpha=0.18, lw=0, step="mid")
    ax.plot(t, F_alb_cm * sc, color=C.CREDIT, lw=1.0, label="albedo credit (cycle mean)")
    ax.plot(t, F_co * sc, color=C.BURDEN, lw=1.0, label=r"CO$_2$ of the applications")
    ax.plot(t, (F_alb_cm + F_co) * sc, color=C.INK, lw=0.9, ls=(0, (3, 1.5)), label="net (cycle mean)")
    ax.axhline(0, color=C.INK, lw=0.4)
    ax.axvline(p.H, color=C.GREY, lw=0.5)
    ax.set_xlim(0, T)
    ax.set_ylim(-60, 75)
    ax.set_xlabel("Years since first application")
    ax.set_ylabel(r"Global forcing per m$^2$ ($10^{-15}$\,W\,m$^{-2}$)")
    ax.legend(loc="upper right", fontsize=5.8, borderaxespad=0.3)
    ax.annotate("end of the\nanalysis period", xy=(p.H, -48), xytext=(p.H + 12, -48), fontsize=5.6, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.4, color=C.GREY))
    ax.text(3, -57, "annual values", fontsize=5.6, color=C.CREDIT, va="bottom")
    P.panel_label(ax, "b")
    # ------------------------------------------------------------ (c) temperature
    ax = axs[1, 0]
    sk = 1e15
    ax.plot(t, T_alb * sk, color=C.CREDIT, lw=0.9, label="albedo credit")
    ax.plot(t, T_co * sk, color=C.BURDEN, lw=0.9, label=r"CO$_2$ of the applications")
    ax.plot(t, (T_alb + T_co) * sk, color=C.INK, lw=0.8, ls=(0, (3, 1.5)), label="net")
    ax.axhline(0, color=C.INK, lw=0.4)
    ax.axvline(p.H, color=C.GREY, lw=0.5)
    ax.set_xlim(0, T)
    ax.set_xlabel("Years since first application")
    ax.set_ylabel(r"Global temperature per m$^2$ ($10^{-15}$\,K)")
    ax.legend(loc="lower right", fontsize=5.8, borderaxespad=0.3)
    P.panel_label(ax, "c")
    # ------------------------------------------------------------ (d) equivalents vs horizon
    ax = axs[1, 1]
    js = json.loads((RES / f"ledger_{TAG}.json").read_text())[SITE_KEY]["horizons"]
    ths = np.array(sorted(int(k) for k in js))
    for m, ls in (("dynamic", "-"), ("gtp", (0, (3, 1.5)))):
        rf = np.array([-js[str(h)][m]["G_rf"] for h in ths])
        co = np.array([js[str(h)][m]["G_co"] for h in ths])
        ax.plot(ths, co, color=C.BURDEN, ls=ls, lw=0.9, marker="o", ms=2.2)
        ax.plot(ths, rf, color=C.CREDIT, ls=ls, lw=0.9, marker="o", ms=2.2)
    st = [-js["100"]["static"]["G_rf"], js["100"]["static"]["G_co"]]
    ax.axhline(st[1], color=C.BURDEN, lw=0.5, alpha=0.6)
    ax.axhline(st[0], color=C.CREDIT, lw=0.5, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlim(18, 560)
    ax.set_ylim(0, 74)
    ax.set_xticks([20, 50, 100, 200, 500])
    ax.set_xticklabels(["20", "50", "100", "200", "500"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Metric horizon (years)")
    ax.set_ylabel(r"kg CO$_2$e\,m$^{-2}$")
    ax.plot([], [], color=C.BURDEN, lw=0.9, label=r"burden $G_{\mathrm{co}}$")
    ax.plot([], [], color=C.CREDIT, lw=0.9, label=r"credit $|G_{\mathrm{rf}}|$")
    ax.plot([], [], color=C.GREY, lw=0.9, ls="-", label="dynamic GWP")
    ax.plot([], [], color=C.GREY, lw=0.9, ls=(0, (3, 1.5)), label="dynamic GTP")
    ax.plot([], [], color=C.GREY, lw=0.5, label=r"static GWP$_{100}$")
    ax.legend(loc="upper left", fontsize=5.6, ncol=3, borderaxespad=0.3, columnspacing=0.7, handlelength=1.6)
    P.panel_label(ax, "d")
    P.save(fig, FIG / "fig_time")


if __name__ == "__main__":
    main()

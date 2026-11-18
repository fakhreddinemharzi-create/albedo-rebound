"""Figure: the ten sites -- map (Albers equal-area), climate space and the
monthly cycle of the radiative kernel."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from cmcrameri import cm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import plotting as P, palette as C  # noqa: E402
from alrb.config import RES, FIG, RAW  # noqa: E402

NE = RAW / "ne"


def albers(lon, lat, lon0=-96.0, lat0=37.5, lat1=29.5, lat2=45.5):
    phi, lam = np.radians(lat), np.radians(lon)
    p1, p2, p0 = np.radians(lat1), np.radians(lat2), np.radians(lat0)
    n = 0.5 * (np.sin(p1) + np.sin(p2))
    Cc = np.cos(p1) ** 2 + 2 * n * np.sin(p1)
    rho0 = np.sqrt(Cc - 2 * n * np.sin(p0)) / n
    rho = np.sqrt(Cc - 2 * n * np.sin(phi)) / n
    th = n * (lam - np.radians(lon0))
    return 6371.0 * rho * np.sin(th), 6371.0 * (rho0 - rho * np.cos(th))


def polygons(fname, keep):
    gj = json.loads((NE / fname).read_text())
    polys = []
    for f in gj["features"]:
        if not keep(f["properties"]):
            continue
        g = f["geometry"]
        rings = []
        if g["type"] == "Polygon":
            rings = [g["coordinates"][0]]
        elif g["type"] == "MultiPolygon":
            rings = [p[0] for p in g["coordinates"]]
        for r in rings:
            a = np.array(r)
            if len(a) < 4:
                continue
            x, y = albers(a[:, 0], a[:, 1])
            polys.append(np.column_stack([x, y]))
    return polys


def main():
    P.setup()
    sites = json.loads((RES / "sites.json").read_text())
    kern = json.loads((RES / "kernels.json").read_text())["sites"]
    keys = C.SITE_ORDER
    fig = plt.figure(figsize=(P.DOUBLE, 2.75), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.62, 1.0, 1.0])
    # ------------------------------------------------------------ (a) map
    ax = fig.add_subplot(gs[0])
    lower = lambda p: p.get("adm0_a3") == "USA" and p.get("name") not in ("Alaska", "Hawaii")
    states = polygons("ne_50m_admin_1_states_provinces_lakes.geojson", lower)
    neigh = polygons("ne_50m_admin_0_countries_lakes.geojson", lambda p: p.get("ADM0_A3") in ("CAN", "MEX"))
    ax.add_collection(PolyCollection(neigh, facecolor="#F4F5F6", edgecolor="#C9CDD2", linewidth=0.3))
    ax.add_collection(PolyCollection(states, facecolor="#ECEEF0", edgecolor="white", linewidth=0.45))
    kv = np.array([-kern[k]["K_aprp_hourly"] for k in keys])
    norm = matplotlib.colors.Normalize(vmin=0.9, vmax=1.8)
    xs, ys = albers(np.array([sites[k]["lon"] for k in keys]), np.array([sites[k]["lat"] for k in keys]))
    sc = ax.scatter(xs, ys, c=kv, cmap=cm.lajolla_r, norm=norm, s=26, edgecolor=C.INK, linewidth=0.5, zorder=5)
    offs = {"goodridge": (8, 6), "wolfpoint": (6, -7), "ithaca": (6, 5), "lajunta": (-6, 7), "wooster": (-8, 7),
            "bowlinggreen": (-10, -9), "monahans": (-6, -9), "durham": (8, -6), "merced": (6, 7), "tucson": (-10, -8)}
    ha = {"goodridge": "left", "wolfpoint": "left", "ithaca": "left", "lajunta": "right", "wooster": "right",
          "bowlinggreen": "right", "monahans": "right", "durham": "left", "merced": "left", "tucson": "right"}
    for k, x, y in zip(keys, xs, ys):
        ax.annotate(C.SITE_LABEL[k].split(",")[0], (x, y), xytext=offs[k], textcoords="offset points",
                    ha=ha[k], va="center", fontsize=6.3, color=C.INK)
    ax.set_xlim(-2380, 2280)
    ax.set_ylim(-1720, 1480)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")
    cax = ax.inset_axes([0.05, 0.075, 0.36, 0.035])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label(r"$|\bar K|$ (W\,m$^{-2}$ per 0.01 albedo)", fontsize=6.0, labelpad=1.5)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5.6, length=2, pad=1.5)
    cb.set_ticks([1.0, 1.2, 1.4, 1.6, 1.8])
    P.panel_label(ax, "a", x=0.04, y=1.0, dx=0, dy=0)
    # ------------------------------------------------------------ (b) climate space
    ax = fig.add_subplot(gs[1])
    cols = C.site_colors()
    for k in keys:
        s = sites[k]
        ax.scatter(s["sw"], s["fi"] + 1, s=10 + s["precip"] / 30, color=cols[k], edgecolor=C.INK, linewidth=0.4,
                   zorder=4)
        dx, dy, hh = {"goodridge": (-4, 2, "right"), "wolfpoint": (4, -1, "left"), "merced": (-4, 3, "right"),
                      "tucson": (4, 3, "left"), "monahans": (4, 1, "left"), "durham": (-4, 2, "right")}.get(k, (4, 1, "left"))
        ax.annotate(C.SITE_SHORT[k], (s["sw"], s["fi"] + 1), xytext=(dx, dy), textcoords="offset points",
                    fontsize=5.8, color=C.INK, va="bottom", ha=hh)
    ax.set_yscale("log")
    ax.set_ylim(0.6, 5000)
    ax.set_xlim(125, 245)
    ax.set_xlabel(r"Mean global irradiance (W\,m$^{-2}$)")
    ax.set_ylabel(r"Freezing index $+1$ ($^{\circ}$C\,d)")
    for ps, lab in ((300, "300 mm"), (1500, "1500 mm")):
        ax.scatter([], [], s=10 + ps / 30, color="white", edgecolor=C.INK, linewidth=0.4, label=lab)
    ax.legend(loc="lower left", title="precipitation", fontsize=5.8, title_fontsize=5.8, handletextpad=0.3,
              borderaxespad=0.3, labelspacing=0.6)
    P.panel_label(ax, "b")
    # ------------------------------------------------------------ (c) monthly air temperature
    ax = fig.add_subplot(gs[2])
    m = np.arange(1, 13)
    for k in keys:
        ax.plot(m, sites[k]["T_monthly"], color=cols[k], lw=0.9, label=C.SITE_SHORT[k])
    ax.axhline(0, color=C.INK, lw=0.5)
    ax.set_xticks([1, 3, 5, 7, 9, 11])
    ax.set_xticklabels(["Jan", "Mar", "May", "Jul", "Sep", "Nov"])
    ax.set_xlim(0.7, 12.3)
    ax.set_ylim(-16, 44)
    ax.set_ylabel(r"Monthly mean air temperature ($^{\circ}$C)")
    ax.legend(ncol=2, fontsize=5.4, loc="upper left", handlelength=1.1, columnspacing=0.6, labelspacing=0.2,
              borderaxespad=0.25)
    P.panel_label(ax, "c")
    P.save(fig, FIG / "fig_sites")


if __name__ == "__main__":
    main()

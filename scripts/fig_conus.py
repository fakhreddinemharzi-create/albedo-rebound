"""Figure: the credit and the balance of the reference treatment across the
contiguous United States (1-degree cells, CERES SYN1deg hourly irradiance
2017-2024, calibrated to the station network)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import plotting as P, palette as C  # noqa: E402
from alrb.config import FIG, RAW, RES, SITES  # noqa: E402

NE = RAW / "ne"
TAG = "v3"


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
        rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]
        for r in rings:
            a = np.array(r)
            if len(a) < 4:
                continue
            x, y = albers(a[:, 0], a[:, 1])
            polys.append(np.column_stack([x, y]))
    return polys


def cells(lat, lon, mask):
    out, idx = [], []
    for a in range(len(lat)):
        for b in range(len(lon)):
            if not mask[a, b]:
                continue
            la = np.array([-0.5, -0.5, 0.5, 0.5]) + lat[a]
            lo = np.array([-0.5, 0.5, 0.5, -0.5]) + lon[b]
            # densify the edges so the cells follow the projection
            t = np.linspace(0, 1, 5)[:-1]
            pl, pa = [], []
            for k in range(4):
                pl += list(lo[k] + (lo[(k + 1) % 4] - lo[k]) * t)
                pa += list(la[k] + (la[(k + 1) % 4] - la[k]) * t)
            x, y = albers(np.array(pl), np.array(pa))
            out.append(np.column_stack([x, y]))
            idx.append((a, b))
    return out, idx


def draw_map(ax, polys, vals, cmap, norm, states, neigh, border):
    ax.add_collection(PolyCollection(neigh, facecolor="#F4F5F6", edgecolor="#C9CDD2", linewidth=0.3))
    fc = cmap(norm(vals))
    ax.add_collection(PolyCollection(polys, facecolor=fc, edgecolor=fc, linewidth=0.3))
    ax.add_collection(PolyCollection(states, facecolor="none", edgecolor="white", linewidth=0.35))
    ax.add_collection(PolyCollection(border, facecolor="none", edgecolor="#8A9199", linewidth=0.4))
    ax.set_xlim(-2380, 2280)
    ax.set_ylim(-1720, 1480)
    ax.set_aspect("equal")
    ax.axis("off")


def colourbar(fig, ax, cmap, norm, label, ticks):
    cax = ax.inset_axes([0.05, 0.075, 0.34, 0.035])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(label, fontsize=6.0, labelpad=1.5)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.tick_params(labelsize=5.6, length=2, pad=1.5)
    cb.set_ticks(ticks)
    cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    return cb


def main():
    z = np.load(RES / f"conus_ledger_{TAG}.npz")
    zk = np.load(RES / "conus_kernel_v3.npz")
    cj = json.loads((RES / "conus_kernel_v3.json").read_text())["sites"]
    summ = json.loads((RES / f"conus_ledger_{TAG}.json").read_text())
    lat, lon, land = z["lat"], z["lon"], z["land"]
    ok = land & np.isfinite(z["R_static"])
    polys, idx = cells(lat, lon, ok)
    ia = np.array([i for i, _ in idx])
    ib = np.array([j for _, j in idx])
    lower = lambda p: p.get("adm0_a3") == "USA" and p.get("name") not in ("Alaska", "Hawaii")  # noqa: E731
    states = polygons("ne_50m_admin_1_states_provinces_lakes.geojson", lower)
    neigh = polygons("ne_50m_admin_0_countries_lakes.geojson", lambda p: p.get("ADM0_A3") in ("CAN", "MEX"))
    border = polygons("ne_50m_admin_0_countries_lakes.geojson", lambda p: p.get("ADM0_A3") == "USA")
    kern = json.loads((RES / "kernels.json").read_text())["sites"]
    sx, sy = albers(np.array([s.lon for s in SITES]), np.array([s.lat for s in SITES]))
    P.setup()
    fig = plt.figure(figsize=(P.DOUBLE, 4.75))
    gs = fig.add_gridspec(2, 2, left=0.0, right=0.995, bottom=0.075, top=0.985, wspace=0.02, hspace=0.06)
    # (a) kernel
    ax = fig.add_subplot(gs[0, 0])
    norm = matplotlib.colors.Normalize(0.7, 2.0)
    cmap = cm.lajolla_r
    draw_map(ax, polys, -z["K"][ia, ib], cmap, norm, states, neigh, border)
    ax.scatter(sx, sy, c=[-kern[s.key]["K_aprp_snowmasked"] for s in SITES], cmap=cmap, norm=norm, s=22,
               edgecolor=C.INK, linewidth=0.6, zorder=5)
    colourbar(fig, ax, cmap, norm, r"$|\bar K|$ (W\,m$^{-2}$ per 0.01 albedo)", [0.8, 1.2, 1.6, 2.0])
    ax.text(0.99, 0.02, "points: stations", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6,
            color=C.GREY)
    ax.text(0.005, 0.995, r"\textbf{(a)}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
    # (b) static rebound ratio
    ax = fig.add_subplot(gs[0, 1])
    norm = matplotlib.colors.Normalize(2.0, 5.0)
    cmap = cm.bilbao
    draw_map(ax, polys, z["R_static"][ia, ib], cmap, norm, states, neigh, border)
    ax.scatter(sx, sy, s=6, color="white", edgecolor=C.INK, linewidth=0.5, zorder=5)
    colourbar(fig, ax, cmap, norm, r"Static rebound ratio $R$", [2, 3, 4, 5])
    ax.text(0.005, 0.995, r"\textbf{(b)}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
    # (c) break-even emissions per coat
    ax = fig.add_subplot(gs[1, 0])
    norm = matplotlib.colors.Normalize(0.7, 1.6)
    cmap = cm.devon
    draw_map(ax, polys, z["gstar_dynamic"][ia, ib], cmap, norm, states, neigh, border)
    ax.scatter(sx, sy, s=6, color="white", edgecolor=C.INK, linewidth=0.5, zorder=5)
    colourbar(fig, ax, cmap, norm, r"Break-even $g^{*}_{\mathrm{co}}$, dynamic (kg\,CO$_2$e\,m$^{-2}$)",
              [0.8, 1.0, 1.2, 1.4, 1.6])
    ax.text(0.005, 0.995, r"\textbf{(c)}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
    # (d) validation at the station cells
    ax = fig.add_subplot(gs[1, 1])
    box = ax.get_position()
    ax.set_position([box.x0 + 0.10, box.y0 + 0.055, box.width - 0.16, box.height - 0.075])
    st_k = np.array([-cj[s.key]["station_Km"] for s in SITES])
    gr_k = np.array([-cj[s.key]["grid_Km"] for s in SITES])
    v = summ["validation"]
    st_r = np.array([v[s.key]["R_static_station"] for s in SITES])
    mp_r = np.array([v[s.key]["R_static_map"] for s in SITES])
    ax.plot([0.7, 5.5], [0.7, 5.5], color=C.GREY, lw=0.6, zorder=1)
    ax.scatter(st_k, gr_k, s=12, marker="o", facecolor="none", edgecolor=C.BURDEN, linewidth=0.8, zorder=3,
               label=r"$|\bar K|$, satellite grid (uncalibrated)")
    ax.scatter(st_r, mp_r, s=12, marker="s", color=C.CREDIT, linewidth=0, zorder=3,
               label=r"$R$ (static), calibrated map")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.7, 5.5)
    ax.set_ylim(0.7, 5.5)
    for a_ in (ax.xaxis, ax.yaxis):
        a_.set_major_locator(matplotlib.ticker.FixedLocator([1, 2, 3, 4, 5]))
        a_.set_major_formatter(matplotlib.ticker.FixedFormatter(["1", "2", "3", "4", "5"]))
        a_.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Station value")
    ax.set_ylabel("Grid value at the station cell")
    ax.legend(loc="upper left", fontsize=5.6, borderaxespad=0.3, handletextpad=0.3)
    P.panel_label(ax, "d")
    P.save(fig, FIG / "fig_conus")


if __name__ == "__main__":
    main()

"""Figure: the urban-form factor of a reflective street in a canyon.

(a) Path-traced view of an open street and of two canyons (H/W = 0.5 and
1.5) whose floor and inner walls are coloured by the change of absorbed
shortwave radiation caused by raising the albedo of the street, per unit
albedo change and per unit of the global irradiance of an open surface
(kernel-weighted mean over the record at Durham, east-west street, walls of
albedo 0.30).  Arrows above each street are proportional to the part of
the additional reflection that leaves the canyon, psi.
(b) psi against aspect ratio (all sites and orientations); (c) the
radiosity model against path tracing and a photon Monte Carlo; (d) seasonal
cycle of psi at H/W = 1; (e) the static rebound ratio in a canyon.

usage: python fig_canyon.py [samples per pixel]
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from cmcrameri import cm as ccm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import canyon, pathtrace as pt, plotting as P, palette as C  # noqa: E402
from alrb.config import FIG, RES  # noqa: E402

TAG = "v3"
KEY = "durham"
AXIS = 90.0                       # east-west street
W = 10.0                          # street width (m)
LX = 30.0                         # length of the street segments shown (m)
DEPTH = 7.0                       # depth of the building blocks (m)
TP = np.radians(np.linspace(0.0, 89.75, 360))
VMIN, VMAX = -1.0, 0.5
CACHE = FIG / "_render"
CACHE.mkdir(parents=True, exist_ok=True)
# one east-west street whose canyon deepens eastwards: name, H/W, x of the segment start
SCENES = [("open", 0.0, 0.0), ("shallow", 0.5, LX + 8.0), ("deep", 1.5, 2 * LX + 16.0)]
SUN = (0.30, 0.62, -0.72)         # direction of travel: the sun is in the south-west
XEND = 3 * LX + 16.0


def fate_maps(inp):
    out, maps = {}, {}
    for name, h, _ in SCENES:
        if h == 0.0:
            out[f"{h:.1f}"] = dict(floor=-1.0, walls=0.0, escape=1.0)
            maps[name] = None
            continue
        c = canyon.build(h, 60)
        dd, db = canyon.absorbed_change(c, 0.14, 0.45, 0.30, TP)
        seg = canyon.weighted_segments(inp, dd, db, TP, AXIS)
        surf = c.kind != 2
        L, k = c.length[surf], c.kind[surf]
        mid = 0.5 * (c.a + c.b)[surf]
        f = float(np.sum((seg * L)[k == 0]))
        w = float(np.sum((seg * L)[k != 0]))
        out[f"{h:.1f}"] = dict(floor=f, walls=w, escape=-(f + w))
        maps[name] = dict(seg=seg, kind=k, mid=mid)
    return out, maps


def norm01(v):
    """Two-slope normalisation of the absorbed-flux change to [0, 1]."""
    v = np.asarray(v, float)
    return np.where(v < 0, 0.5 * (1 - v / VMIN), 0.5 + 0.5 * v / VMAX)


def strip(p0, p1, u_vals, s_vals, x0, nx=4):
    """Rectangular surface from p0 to p1 in the (y, z) cross section,
    extruded along x over [x0, x0 + LX]; s_vals at cross-section fractions u_vals."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    us = np.linspace(0.0, 1.0, 121)
    xs = np.linspace(x0, x0 + LX, nx)
    X, U = np.meshgrid(xs, us, indexing="ij")
    Y = p0[0] + (p1[0] - p0[0]) * U
    Z = p0[1] + (p1[1] - p0[1]) * U
    grid = pv.StructuredGrid(X, Y, Z)
    s = np.interp(U, u_vals, s_vals)
    grid.point_data["s"] = s.ravel(order="F")
    surf = grid.extract_surface(algorithm="dataset_surface")
    return surf, np.asarray(surf.point_data["s"])


def box_edges(b, r=0.05):
    x0, x1, y0, y1, z0, z1 = b
    c = [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
    out = None
    for i in range(8):
        for j in range(i + 1, 8):
            d = np.abs(np.subtract(c[i], c[j]))
            if (d > 1e-9).sum() == 1:
                t = pt.tube([c[i], c[j]], r, n=8)
                out = t if out is None else out + t
    return out


def scene_shapes(maps, fate):
    shapes = {}
    neutral = pt.principled("#dcd7cd", roughness=0.9, specular=0.15)
    cmap = ccm.vik
    ground = pv.Box(bounds=(-8.0, XEND + 8.0, -DEPTH - 10.0, W + DEPTH + 10.0, -0.25, -0.02))
    shapes["ground"] = pt.mesh(ground, pt.principled("#cdc9c0", roughness=0.95, specular=0.1), smooth=False)
    anchors = {}
    ghosts, edges = None, None
    for name, h, x0 in SCENES:
        H = h * W
        m = maps[name]
        if m is None:
            surf, sv = strip((0.0, 0.0), (W, 0.0), [0.0, 1.0], [norm01(-1.0)] * 2, x0)
        else:
            fl = m["kind"] == 0
            # cross-section x (0 = sun-facing wall at the north kerb) -> world y = W (1 - x)
            xm = m["mid"][fl, 0]
            order = np.argsort(1 - xm)
            surf, sv = strip((0.0, 0.0), (W, 0.0), (1 - xm)[order], norm01(m["seg"][fl])[order], x0)
        shapes[f"floor_{name}"] = pt.colour_mesh(surf, sv, cmap, 0.0, 1.0, roughness=0.85, specular=0.1)
        if m is not None:
            # sun-facing (north) wall, facing south, coloured; the building behind it neutral
            sel = m["kind"] == 3
            zm = m["mid"][sel, 1] / h
            order = np.argsort(zm)
            surf, sv = strip((W, 0.0), (W, H), zm[order], norm01(m["seg"][sel])[order], x0)
            shapes[f"wall_{name}"] = pt.colour_mesh(surf, sv, cmap, 0.0, 1.0, roughness=0.85, specular=0.1)
            shapes[f"blockN_{name}"] = pt.mesh(pv.Box(bounds=(x0, x0 + LX, W + 0.03, W + DEPTH, 0.0, H)), neutral,
                                               smooth=False)
            # the south-side building, drawn as a translucent ghost with outlined edges
            b = (x0, x0 + LX, -DEPTH, -0.03, 0.0, H)
            g = pv.Box(bounds=b)
            ghosts = g if ghosts is None else ghosts + g
            e = box_edges(b, 0.06)
            edges = e if edges is None else edges + e
        psi = fate[f"{h:.1f}"]["escape"]
        base = np.array([x0 + 0.5 * LX, 0.5 * W, H + 0.8])
        top = base + np.array([0.0, 0.0, 12.0 * psi])
        a = pt.arrow(base, top, 0.26, 0.70, min(1.8, 0.45 * 12.0 * psi))
        shapes[f"arrow_{name}"] = pt.mesh(a, pt.principled(C.CREDIT, roughness=0.4, specular=0.3), smooth=True)
        anchors[name] = dict(top=top, base=base, H=H, x0=x0)
    shapes["ghosts"] = pt.mesh(ghosts, pt.translucent("#e8e4dc", 0.16), smooth=False)
    shapes["ghost_edges"] = pt.mesh(edges, pt.principled("#8a8478", roughness=0.7), smooth=True)
    return shapes, anchors


_BOX = [(x, y, z) for x in (0.0, XEND) for y in (-DEPTH, W + DEPTH) for z in (0.0, 16.5)]
CAM = pt.fit_camera(_BOX, (-0.10, -0.80, 0.60), 170.0, 2900, 1040, box=(0.02, 0.98, 0.30, 0.985))


def render(spp, shapes):
    sig = hashlib.sha1(repr((CAM.origin, CAM.target, CAM.fov, CAM.width, CAM.height, SUN,
                             sorted(shapes.items(), key=lambda kv: kv[0]))).encode()).hexdigest()[:10]
    f = CACHE / f"canyon_{sig}_{spp}.npy"
    if f.exists():
        return np.load(f)
    img = pt.render(shapes, CAM, spp=spp, sun_dir=SUN, sun_irradiance=1.6, sky=(0.78, 0.82, 0.90),
                    sky_strength=1.25, max_depth=8, transparent=True)
    np.save(f, img.astype(np.float32))
    return img


def label_clear(fig, ax, alpha, texts, thr=0.02):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    probs = []
    Hh, Ww = alpha.shape
    for t in texts:
        bb = matplotlib.text.Text.get_window_extent(t, renderer=r).expanded(1.04, 1.1)
        (x0, y1), (x1, y0) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        xa, xb = int(max(0, np.floor(min(x0, x1)))), int(min(Ww, np.ceil(max(x0, x1))))
        ya, yb = int(max(0, np.floor(min(y0, y1)))), int(min(Hh, np.ceil(max(y0, y1))))
        if xb <= xa or yb <= ya:
            continue
        frac = float((alpha[ya:yb, xa:xb] > 0.5).mean())
        if frac > thr:
            probs.append(dict(type="label_over_render", text=t.get_text()[:40], frac=round(frac, 3)))
    return probs


LCZ = [(0.10, 0.30, "sparse,\nlarge low-rise"), (0.30, 0.75, "open low-\nto mid-rise"),
       (0.75, 2.00, "compact low- to mid-rise"), (2.00, 3.00, "compact\nhigh-rise")]


def main(spp=192):
    cy = json.loads((RES / f"canyon_{TAG}.json").read_text())
    inp = canyon.site_inputs(KEY)
    fate, maps = fate_maps(inp)
    (RES / f"canyon_fate_{TAG}.json").write_text(json.dumps(fate, indent=1))
    shapes, anchors = scene_shapes(maps, fate)
    img = render(spp, shapes)
    a = img[..., 3:4]
    rgb = img[..., :3] * a + (1 - a)
    alpha = a[..., 0]
    Hp, Wp = alpha.shape

    P.setup()
    fig = plt.figure(figsize=(P.DOUBLE, 4.95))
    top = 0.515
    ax = fig.add_axes([0.0, top, 1.0, 1.0 - top])
    ax.imshow(rgb, interpolation="lanczos")
    ax.set_xlim(0, Wp)
    ax.set_ylim(Hp, 0)
    ax.axis("off")
    pr = lambda p: CAM.project([p])[0][0]  # noqa: E731
    fs = 6.2
    texts = []
    titles = {"open": "open street", "shallow": r"canyon, $H/W=0.5$", "deep": r"canyon, $H/W=1.5$"}
    xpos = {"open": 0.40, "shallow": 0.635, "deep": 0.865}
    for name, h, _ in SCENES:
        fz = fate[f"{h:.1f}"]
        tip = pr(anchors[name]["top"])
        if h == 0.0:
            body = (rf"\textbf{{{titles[name]}}}" "\n" r"$\psi=1$: all of the additional" "\n"
                    r"reflection leaves the street")
        else:
            body = (rf"\textbf{{{titles[name]}}}" "\n" rf"$\psi={fz['escape']:.2f}$ leaves the canyon;" "\n"
                    rf"walls absorb {fz['walls']:.2f}, floor $-${-fz['floor']:.2f}")
        t = ax.annotate(body, xy=tip, xytext=(xpos[name] * Wp, 0.035 * Hp), textcoords="data", fontsize=fs,
                        ha="center", va="top", color=C.INK, linespacing=1.15,
                        arrowprops=dict(arrowstyle="-", lw=0.45, color=C.INK, shrinkA=1.0, shrinkB=2.0))
        texts.append(t)
    ax.text(0.004, 0.995, r"\textbf{(a)}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
    cax = fig.add_axes([0.035, 1.0 - 0.085, 0.24, 0.016])
    sm = matplotlib.cm.ScalarMappable(cmap=ccm.vik, norm=matplotlib.colors.TwoSlopeNorm(0.0, VMIN, VMAX))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([-1.0, -0.5, 0.0, 0.25, 0.5])
    cb.set_ticklabels([r"$-1$", r"$-0.5$", "0", "0.25", "0.5"])
    cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.ax.tick_params(labelsize=5.6, length=2, pad=1.2)
    cb.set_label(r"Change of absorbed shortwave per unit $\Delta\alpha\,S^{\downarrow}$", fontsize=5.8,
                 labelpad=1.5)
    cb.ax.xaxis.set_label_position("top")
    t = ax.text(0.035 * Wp, 0.225 * Hp, r"arrows $\propto\psi$; Durham, east--west street;" "\n"
                r"south-side buildings drawn transparent; walls $\alpha_w=0.30$",
                fontsize=5.6, color=C.GREY, ha="left", va="top", linespacing=1.15)
    texts.append(t)
    extra = label_clear(fig, ax, alpha, texts)

    # ---------------------------------------------------------------- bottom row
    h = np.array(cy["h"])
    gs = fig.add_gridspec(1, 4, left=0.065, right=0.99, bottom=0.085, top=top - 0.075, wspace=0.42)
    # (b) psi against aspect ratio
    ax1 = fig.add_subplot(gs[0, 0])
    allpsi = np.array([cy["sites"][k][o]["psi"] for k in C.SITE_ORDER for o in ("EW", "NS", "diag")])
    ax1.fill_between(h, allpsi.min(0), allpsi.max(0), color=C.CREDIT, alpha=0.25, lw=0,
                     label="sites, orientations")
    ax1.plot(h, np.median(allpsi, 0), color=C.CREDIT, lw=1.1, label="median")
    ax1.plot(h, cy["diffuse"]["0.30"], color=C.INK, lw=0.8, ls=(0, (3, 1.5)), label="overcast sky")
    ax1.plot(h, cy["fgs"], color=C.GREY, lw=0.8, ls=(0, (1, 1.2)), label=r"sky-view factor $F_{gs}$")
    for i, (x0, x1, name) in enumerate(LCZ):
        P.vspan(ax1, x0, x1, color=C.PALE if i % 2 == 0 else "#f5f6f7", lw=0, zorder=0)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel(r"Aspect ratio $H/W$")
    ax1.set_ylabel(r"Urban-form factor $\psi$")
    ax1.legend(loc="upper right", fontsize=5.4, borderaxespad=0.3, handlelength=1.8)
    P.panel_label(ax1, "b", dy=8.0)
    # (c) verification
    ax2 = fig.add_subplot(gs[0, 1])
    pts = cy.get("path_tracing", [])
    mc = cy.get("photon_mc", [])
    ax2.plot([0, 0.75], [0, 0.75], color=C.GREY, lw=0.6, zorder=1)
    if pts:
        ax2.plot([p["rad_psi"] for p in pts], [p["pt_psi"] for p in pts], "o", ms=3.2, mfc="none",
                 mec=C.CREDIT, mew=0.8, label="path tracing (3-D)", zorder=3)
    if mc:
        ax2.plot([p["rad_psi"] for p in mc], [p["mc_psi"] for p in mc], "x", ms=3.0, color=C.BURDEN, mew=0.8,
                 label="photon Monte Carlo", zorder=4)
    ax2.set_xlim(0, 0.75)
    ax2.set_ylim(0, 0.75)
    ax2.set_xticks([0, 0.25, 0.5, 0.75])
    ax2.set_yticks([0.25, 0.5, 0.75])
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"$\psi$, radiosity model")
    ax2.set_ylabel(r"$\psi$, independent solution")
    ax2.legend(loc="upper left", fontsize=5.4, borderaxespad=0.3, handletextpad=0.3)
    P.panel_label(ax2, "c", dy=8.0)
    # (d) seasonal cycle at H/W = 1
    ax3 = fig.add_subplot(gs[0, 2])
    m = np.arange(1, 13)
    for k in ("goodridge", "tucson"):
        col = C.site_colors()[k]
        for o, ls in (("EW", "-"), ("NS", (0, (3, 1.5)))):
            v = np.array(cy["sites"][k][o]["monthly"]["1.0"], float)
            ax3.plot(m, v, color=col, lw=0.9, ls=ls, marker="o", ms=1.6)
    ax3.set_xticks(m)
    ax3.set_xticklabels(list("JFMAMJJASOND"))
    ax3.set_xlim(0.5, 12.5)
    ax3.set_ylim(0.0, 0.52)
    ax3.set_xlabel("Month")
    ax3.set_ylabel(r"$\psi$ at $H/W=1$")
    hs = [matplotlib.lines.Line2D([], [], color=C.site_colors()[k], lw=1.0) for k in ("goodridge", "tucson")]
    hs += [matplotlib.lines.Line2D([], [], color=C.GREY, lw=0.9),
           matplotlib.lines.Line2D([], [], color=C.GREY, lw=0.9, ls=(0, (3, 1.5)))]
    ax3.legend(hs, ["Goodridge", "Tucson", "east--west", "north--south"], loc="upper center", ncol=2,
               fontsize=5.3, borderaxespad=0.3, columnspacing=0.8, handlelength=1.6)
    P.panel_label(ax3, "d", dy=8.0)
    # (e) static rebound ratio in a canyon
    ax4 = fig.add_subplot(gs[0, 3])
    for k in C.SITE_ORDER:
        s = cy["sites"][k]
        R = s["R_open"] / np.array(s["EW"]["psi"])
        ax4.plot(h, R, color=C.site_colors()[k], lw=0.8)
    ax4.axhline(1.0, color=C.INK, lw=0.6, ls=(0, (3, 1.5)))
    ax4.set_yscale("log")
    ax4.set_ylim(0.5, 300)
    ax4.set_yticks([1, 10, 100])
    ax4.set_yticklabels(["1", "10", "100"])
    ax4.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax4.set_xlim(0, 3)
    ax4.set_xlabel(r"Aspect ratio $H/W$")
    ax4.set_ylabel(r"Static rebound ratio $R$")
    ax4.text(2.95, 0.8, r"$R=1$", fontsize=5.6, ha="right", va="top", color=C.INK)
    ax4.text(0.08, 200, "ten sites,\neast--west", fontsize=5.6, ha="left", va="top", color=C.GREY)
    P.panel_label(ax4, "e", dy=8.0)
    for e in extra:
        print("   ", e)
    probs = P.save(fig, FIG / "fig_canyon", pdf_dpi=450, strict=False)
    if (probs or extra) and not __import__("os").environ.get("ALRB_AUDIT_REPORT_ONLY"):
        raise RuntimeError("fig_canyon: layout problems")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 256)

"""Figure: the break-even manifold.  Embodied carbon per application below
which a coat (laid albedo 0.45) is a net carbon sink under the dynamic
GWP100, g_co*(tau, T_r), at the least and the most favourable sites,
path-traced.  The surface is coloured by the ratio of g_co* to the
reference coating (2.70 kg CO2e per m2 and application), and the tube is
the break-even frontier where the two are equal.

usage: python fig_manifold3d.py [samples per pixel]
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
import contourpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import pathtrace as pt, plotting as P, palette as C  # noqa: E402
from alrb.config import RES, FIG, REF  # noqa: E402

TAG = "v3"
SITES = ["ithaca", "tucson"]
METRIC = "dynamic"
LX, LY = 8.0, 6.0
GLO, GHI = 0.1, 30.0                      # g* axis (logarithmic)
RLIM = 1.2                                # colour range of log10(g*/2.70)
CMAP = ccm.vik_r
ZH = 5.0                                  # scene height of the g* axis


def zm(g):
    return ZH * (np.log10(np.clip(g, GLO, GHI)) - np.log10(GLO)) / (np.log10(GHI) - np.log10(GLO))
CACHE = FIG / "_render"
W, H = 1500, 1200


def xm(tau):
    return LX * (np.log10(tau) - np.log10(0.2)) / (np.log10(20.0) - np.log10(0.2))


def ym(tr):
    return LY * np.log10(tr) / np.log10(25.0)


def geometry(taus, trs, G, gref_exact=None):
    X, Y = np.meshgrid(xm(taus), ym(trs), indexing="ij")
    grid = pv.StructuredGrid(X, Y, zm(G))
    grid.point_data["g"] = np.log10(np.clip(G, GLO, GHI) / REF.g_co).ravel(order="F")
    surf = grid.extract_surface(algorithm="dataset_surface")
    shapes = {"surf": pt.colour_mesh(surf, np.asarray(surf.point_data["g"]), CMAP, -RLIM, RLIM,
                                     bsdf_kind="principled", roughness=0.5, specular=0.35)}
    z0 = float(zm(REF.g_co))
    # break-even frontier: contour g* = g_co on the surface
    gen = contourpy.contour_generator(x=np.log10(taus), y=np.log10(trs), z=np.nan_to_num(G, nan=0.0).T)
    front = None
    fpts = []
    for c in gen.lines(REF.g_co):
        p = np.column_stack([xm(10 ** c[:, 0]), ym(10 ** c[:, 1]), np.full(len(c), z0 + 0.03)])
        if len(p) > 2:
            t = pt.tube(p, 0.06, n=12)
            front = t if front is None else front + t
            fpts.append(p)
    if front is not None:
        shapes["front"] = pt.mesh(front, pt.principled(C.INK, roughness=0.4), smooth=True)
    # reference design point on the surface
    it = np.interp(np.log10(REF.tau), np.log10(taus), np.arange(len(taus)))
    ir = np.interp(np.log10(REF.Tr), np.log10(trs), np.arange(len(trs)))
    from scipy.ndimage import map_coordinates
    gref = float(map_coordinates(G, [[it], [ir]], order=1)[0]) if gref_exact is None else float(gref_exact)
    pref = np.array([xm(REF.tau), ym(REF.Tr), float(zm(gref))])
    shapes["ref"] = pt.mesh(pv.Sphere(radius=0.16, center=pref, theta_resolution=40, phi_resolution=40),
                            pt.principled(C.GOLD, roughness=0.3, specular=0.5), smooth=True)
    # floor with grid
    floor = pv.Box(bounds=(0, LX, 0, LY, -0.10, -0.02))
    shapes["floor"] = pt.mesh(floor, pt.principled("#f3f1ec", roughness=0.95), smooth=False)
    gl = None
    for t in (0.2, 0.5, 1, 2, 5, 10, 20):
        seg = pt.tube([(xm(t), 0, -0.015), (xm(t), LY, -0.015)], 0.012, n=8)
        gl = seg if gl is None else gl + seg
    for r in (1, 2, 5, 10, 20):
        gl = gl + pt.tube([(0, ym(r), -0.015), (LX, ym(r), -0.015)], 0.012, n=8)
    for g in (0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        z = float(zm(g))
        gl = gl + pt.tube([(-0.12, 0.0, z), (0.0, 0.0, z)], 0.014, n=8)
    gl = gl + pt.tube([(0.0, 0.0, 0.0), (0.0, 0.0, ZH)], 0.012, n=8)
    shapes["grid"] = pt.mesh(gl, pt.principled("#b9b6ae", roughness=0.8), smooth=True)
    return shapes, gref, pref, fpts


_PTS = [(x, y, z) for x in (0, LX) for y in (0, LY) for z in (0, ZH)]
_PTS += [(-1.0, 0, ZH + 0.6), (LX * 0.5, -1.3, 0), (LX + 2.2, LY * 0.55, 0), (-0.5, -0.8, 0)]
CAM = pt.fit_camera(_PTS, (-0.30, -0.86, 0.46), 40.0, W, H, box=(0.05, 0.95, 0.06, 0.88))


def main(spp=256):
    js = json.loads((RES / f"manifold_{TAG}.json").read_text())
    dsp = json.loads((RES / f"design_space_{TAG}.json").read_text())["sites"]
    taus, trs = np.array(js["taus"]), np.array(js["trs"])
    sites = json.loads((RES / "sites.json").read_text())
    P.setup()
    strip = 0.13
    fig = plt.figure(figsize=(P.DOUBLE, P.DOUBLE / 2 * H / W / (1 - strip)))
    fs = 5.8
    info = {}
    for i, key in enumerate(SITES):
        G = np.array(js["sites"][key][METRIC], float)
        shapes, gref, pref, fpts = geometry(taus, trs, G, dsp[key]["ref"][METRIC]["g_co_star"])
        sig = hashlib.sha1(repr((CAM, sorted(shapes.items(), key=lambda kv: kv[0]))).encode()).hexdigest()[:10]
        f = CACHE / f"mf_{key}_{sig}_{spp}.npy"
        if f.exists():
            img = np.load(f)
        else:
            img = pt.render(shapes, CAM, spp=spp, sun_dir=(0.35, 0.55, -0.76), sun_irradiance=2.2,
                            sky=(0.80, 0.84, 0.90), sky_strength=1.0, max_depth=6, transparent=True)
            np.save(f, img.astype(np.float32))
        a = img[..., 3:4]
        rgb = img[..., :3] * a + (1 - a)
        ax = fig.add_axes([i * 0.5, strip, 0.5, 1 - strip])
        ax.imshow(rgb, interpolation="lanczos")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        pr = lambda p: CAM.project([p])[0][0]
        for t, lab in ((0.2, "0.2"), (0.5, "0.5"), (1, "1"), (2, "2"), (5, "5"), (10, "10"), (20, "20")):
            x, y = pr((xm(t), -0.25, 0.0))
            ax.text(x, y, lab, fontsize=fs, ha="center", va="top")
        x, y = pr((LX * 0.5, -0.9, 0.0))
        ax.text(x, y, r"Decay time $\tau$ (yr)", fontsize=fs + 0.2, ha="center", va="top")
        for r in (2, 5, 10, 20):
            x, y = pr((LX + 0.3, ym(r), 0.0))
            ax.text(x, y, f"{r}", fontsize=fs, ha="left", va="center")
        x, y = pr((LX + 0.9, LY * 0.55, 0.0))
        ax.text(x, y, "Renewal\n" r"$T_r$ (yr)", fontsize=fs + 0.2, ha="left", va="center")
        for g, lab in ((0.3, "0.3"), (1.0, "1"), (3.0, "3"), (10.0, "10"), (30.0, "30")):
            x, y = pr((-0.25, 0.0, float(zm(g))))
            ax.text(x, y, lab, fontsize=fs, ha="right", va="center")
        x, y = pr((-0.25, 0.0, ZH + 0.45))
        ax.text(x, y, r"$g_{\mathrm{co}}^{*}$ (kg\,m$^{-2}$)", fontsize=fs + 0.2, ha="center", va="bottom")
        s = sites[key]
        ax.text(0.99, 0.99, rf"\textbf{{({'ab'[i]})}} {C.SITE_LABEL[key]} ({s['koppen']})" "\n"
                rf"reference coat: $g^{{*}}_{{\mathrm{{co}}}}={gref:.2f}$ vs 2.70",
                transform=ax.transAxes, fontsize=fs + 0.4, ha="right", va="top", linespacing=1.3)
        info[key] = gref
    cax = fig.add_axes([0.06, 0.06, 0.24, 0.022])
    sm = matplotlib.cm.ScalarMappable(cmap=CMAP, norm=matplotlib.colors.LogNorm(10 ** -RLIM, 10 ** RLIM))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0.1, 0.3, 1, 3, 10])
    cb.set_ticklabels(["0.1", "0.3", "1", "3", "10"])
    cb.ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    cb.ax.tick_params(labelsize=5.2, length=1.8, pad=1.0)
    cb.set_label(r"$g^{*}_{\mathrm{co}}/g_{\mathrm{co}}$: red, reference coat a net source; blue, a net sink",
                 fontsize=5.4, labelpad=1.2)
    cb.ax.xaxis.set_label_position("top")
    hs = [matplotlib.lines.Line2D([], [], color=C.INK, lw=1.8),
          matplotlib.lines.Line2D([], [], color=C.GOLD, marker="o", ms=4, lw=0)]
    fig.legend(hs, [r"break-even frontier, $g^{*}_{\mathrm{co}}=g_{\mathrm{co}}=2.70$\,kg\,m$^{-2}$",
                    r"reference treatment ($\tau=0.91$\,yr, $T_r=3$\,yr)"], loc="lower left",
               bbox_to_anchor=(0.42, 0.01), fontsize=5.4, frameon=False, ncol=1, handlelength=1.8)
    P.save(fig, FIG / "fig_manifold3d", pdf_dpi=450)
    return info


if __name__ == "__main__":
    print(main(int(sys.argv[1]) if len(sys.argv) > 1 else 256))

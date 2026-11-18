"""Figure: space-time structure of the cooling under a reflective surface.

For three climates, the mean cooling -dT(day of year, depth) produced by
raising the albedo from 0.14 to 0.45 (2017-2024) is drawn as a path-traced
surface over the (day, log depth) plane; on the floor, the mean frost-front
depth by day of year under the untreated (orange) and the reflective (blue)
surface.  Axes are built as geometry and labelled after projection.

usage: python fig_spacetime3d.py [samples per pixel]
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
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import pathtrace as pt, plotting as P, palette as C  # noqa: E402
from alrb.config import RES, FIG  # noqa: E402

TAG = "v3"
SITES = ["goodridge", "durham", "tucson"]
LX, LY, SZ = 10.0, 6.0, 0.75            # scene size: day axis, depth axis, metres per kelvin
DMIN, DMAX = 0.01, 3.0
VMAX = 6.0
CACHE = FIG / "_render"
CACHE.mkdir(parents=True, exist_ok=True)
W, H = 1500, 1150


def ymap(d):
    """Depth -> scene y (deep at the front, y = 0; surface at the back)."""
    d = np.clip(np.asarray(d, float), DMIN, DMAX)
    return LY * (np.log10(DMAX) - np.log10(d)) / (np.log10(DMAX) - np.log10(DMIN))


def xmap(doy):
    return LX * (np.asarray(doy, float) - 0.5) / 365.0


def field(th):
    d = np.array(th["depths"])
    F = np.array(th["dT_doy_depth"])            # 365 x ndepth
    d, iu = np.unique(d, return_index=True)
    F = -F[:, iu]
    dd = np.logspace(np.log10(DMIN), np.log10(DMAX), 70)
    # extend the top value to 0.01 m (the first recorded depth is 0.0115 m)
    d = np.concatenate([[DMIN], d]) if d[0] > DMIN else d
    F = np.concatenate([F[:, :1], F], axis=1) if F.shape[1] < len(d) else F
    G = np.array([PchipInterpolator(np.log10(d), row)(np.log10(dd)) for row in F])
    # light smoothing along the day axis (7-day circular running mean)
    k = 7
    Gp = np.concatenate([G[-k:], G, G[:k]])
    G = np.array([Gp[i:i + 2 * k + 1].mean(0) for i in range(365)])
    return dd, G


def surface_mesh(dd, G):
    doy = np.arange(1, 366)
    X, Yd = np.meshgrid(xmap(doy), ymap(dd), indexing="ij")
    Z = SZ * G
    grid = pv.StructuredGrid(X, Yd, Z)
    grid.point_data["v"] = G.ravel(order="F")
    return grid.extract_surface(algorithm="dataset_surface")


def axes_geometry():
    """Floor, frame, grid lines and tick marks."""
    zfloor = -0.02
    floor = pv.Box(bounds=(0, LX, 0, LY, zfloor - 0.08, zfloor))
    grid = None
    for m0 in np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]):
        x = xmap(m0 + 0.5)
        t = pt.tube([(x, 0, zfloor + 0.004), (x, LY, zfloor + 0.004)], 0.012, n=8)
        grid = t if grid is None else grid + t
    for d in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        y = float(ymap(d))
        grid = grid + pt.tube([(0, y, zfloor + 0.004), (LX, y, zfloor + 0.004)], 0.012, n=8)
    # back walls (x = 0 side and y = LY side) with horizontal K grid
    wall_b = pv.Box(bounds=(0, LX, LY, LY + 0.06, zfloor - 0.08, SZ * VMAX))
    wall_l = pv.Box(bounds=(-0.06, 0, 0, LY + 0.06, zfloor - 0.08, SZ * VMAX))
    for v in range(0, int(VMAX) + 1, 2):
        z = SZ * v
        grid = grid + pt.tube([(0, LY - 0.004, z), (LX, LY - 0.004, z)], 0.012, n=8)
        grid = grid + pt.tube([(0.004, 0, z), (0.004, LY, z)], 0.012, n=8)
        grid = grid + pt.tube([(-0.12, 0.0, z), (0.0, 0.0, z)], 0.014, n=8)       # tick on the front-left edge
    return dict(floor=pt.mesh(floor, pt.principled("#f3f1ec", roughness=0.95), smooth=False),
                wall_b=pt.mesh(wall_b, pt.principled("#f7f6f2", roughness=0.95), smooth=False),
                wall_l=pt.mesh(wall_l, pt.principled("#f7f6f2", roughness=0.95), smooth=False),
                grid=pt.mesh(grid, pt.principled("#b9b6ae", roughness=0.8), smooth=True))


def fronts(th, dd, G):
    """Frost fronts drawn on the cooling surface (height of the surface at
    the front depth on each day)."""
    out = {}
    ld = np.log10(dd)
    for key, col in (("frost_doy_014", C.WARM), ("frost_doy_045", C.COOL)):
        f = np.array(th[key])
        doy = np.arange(1, 366)
        # frost season as one continuous run around the new year (Jul -> Jun ordering)
        on = f > 0.02
        if on.sum() < 3:
            continue
        segs, cur = [], []
        for i in range(365):
            if on[i]:
                cur.append(i)
            elif cur:
                segs.append(cur)
                cur = []
        if cur:
            segs.append(cur)
        mesh = None
        for sgi in segs:
            if len(sgi) < 3:
                continue
            zz = np.array([np.interp(np.log10(max(f[i], DMIN)), ld, G[i]) for i in sgi]) * SZ + 0.07
            pts = np.column_stack([xmap(doy[sgi]), ymap(f[sgi]), zz])
            t = pt.tube(pts, 0.075, n=12)
            mesh = t if mesh is None else mesh + t
        if mesh is not None:
            out[key] = pt.mesh(mesh, pt.principled(col, roughness=0.35, specular=0.4), smooth=True)
    return out


_PTS = [(x, y, z) for x in (0, LX) for y in (0, LY) for z in (0, SZ * VMAX)]
_PTS += [(LX + 2.6, LY * 0.5, 0.0), (LX + 1.0, 0.0, 0.0), (LX * 0.5, -2.0, 0.0), (-1.2, 0.0, SZ * VMAX),
         (-1.2, 0.0, 0.0)]
CAM = pt.fit_camera(_PTS, (0.62, -0.60, 0.50), 40.0, W, H, box=(0.03, 0.97, 0.10, 0.97))


def render_site(key, th, spp):
    dd, G = field(th)
    surf = surface_mesh(dd, G)
    shapes = {"surf": pt.colour_mesh(surf, np.asarray(surf.point_data["v"]), ccm.lapaz_r, 0.0, VMAX,
                                     bsdf_kind="principled", roughness=0.55, specular=0.35)}
    shapes.update(axes_geometry())
    shapes.update(fronts(th, dd, G))
    sig = hashlib.sha1(repr((CAM, sorted(shapes.items(), key=lambda kv: kv[0]))).encode()).hexdigest()[:10]
    f = CACHE / f"st_{key}_{sig}_{spp}.npy"
    if f.exists():
        img = np.load(f)
    else:
        img = pt.render(shapes, CAM, spp=spp, sun_dir=(-0.35, 0.55, -0.76), sun_irradiance=2.2,
                        sky=(0.80, 0.84, 0.90), sky_strength=1.0, max_depth=6, transparent=True)
        np.save(f, img.astype(np.float32))
    return img, dict(G=G, dd=dd)


def main(spp=256):
    th = json.loads((RES / f"thermal_{TAG}.json").read_text())
    sites = json.loads((RES / "sites.json").read_text())
    P.setup()
    strip = 0.14
    fig = plt.figure(figsize=(P.DOUBLE, P.DOUBLE / 3 * H / W / (1 - strip)))
    pad = 0.0
    wax = 1 / 3
    fs = 5.6
    for i, key in enumerate(SITES):
        img, info = render_site(key, th[key], spp)
        a = img[..., 3:4]
        rgb = img[..., :3] * a + (1 - a)
        ax = fig.add_axes([i * wax + pad, strip, wax - 2 * pad, 1 - strip])
        ax.imshow(rgb, interpolation="lanczos")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        pr = lambda p: CAM.project([p])[0][0]
        # day-of-year ticks along the front edge (y = 0)
        for m0, lab in zip([0, 59, 120, 181, 243, 304], ["Jan", "Mar", "May", "Jul", "Sep", "Nov"]):
            x, y = pr((xmap(m0 + 1), -0.25, 0.0))
            ax.text(x, y, lab, fontsize=fs, ha="center", va="top")
        x, y = pr((LX * 0.64, -2.0, 0.0))
        ax.text(x, y, "Day of year", fontsize=fs + 0.2, ha="center", va="top")
        # depth ticks along the right edge (x = LX)
        for d, lab in ((0.01, "0.01"), (0.1, "0.1"), (1.0, "1"), (3.0, "3")):
            x, y = pr((LX + 0.3, float(ymap(d)), 0.0))
            ax.text(x, y, lab, fontsize=fs, ha="left", va="center")
        x, y = pr((LX + 1.25, LY * 0.5, 0.0))
        ax.text(x, y, "Depth (m)", fontsize=fs + 0.2, ha="left", va="center")
        # vertical ticks on the front-left edge
        for v in range(2, int(VMAX) + 1, 2):
            x, y = pr((-0.25, 0.0, SZ * v))
            ax.text(x, y, f"{v}", fontsize=fs, ha="right", va="center")
        x, y = pr((-0.25, 0.0, SZ * VMAX + 0.55))
        ax.text(x, y, r"Cooling (K)", fontsize=fs + 0.2, ha="center", va="bottom")
        s = sites[key]
        G = info["G"]
        ax.text(0.99, 0.99, rf"\textbf{{({'abc'[i]})}} {C.SITE_LABEL[key]} ({s['koppen']})" "\n"
                rf"max.\ cooling {G.max():.1f}\,K", transform=ax.transAxes, fontsize=fs + 0.4, ha="right", va="top",
                linespacing=1.3)
    # shared keys
    cax = fig.add_axes([0.30, 0.055, 0.16, 0.022])
    sm = matplotlib.cm.ScalarMappable(cmap=ccm.lapaz_r, norm=matplotlib.colors.Normalize(0, VMAX))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 2, 4, 6])
    cb.ax.tick_params(labelsize=5.2, length=1.8, pad=1.0)
    cb.set_label(r"Cooling, $\alpha$ 0.14 $\rightarrow$ 0.45 (K)", fontsize=5.4, labelpad=1.2)
    cb.ax.xaxis.set_label_position("top")
    h1 = matplotlib.lines.Line2D([], [], color=C.WARM, lw=2.0)
    h2 = matplotlib.lines.Line2D([], [], color=C.COOL, lw=2.0)
    fig.legend([h1, h2], [r"frost front, $\alpha=0.14$", r"frost front, $\alpha=0.45$"], loc="lower left",
               bbox_to_anchor=(0.52, 0.0), fontsize=5.4, frameon=False, handlelength=1.6, ncol=2,
               columnspacing=1.2)
    P.save(fig, FIG / "fig_spacetime3d", pdf_dpi=450)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 256)

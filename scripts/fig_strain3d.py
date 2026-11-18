"""Figure: the structural mechanism in three dimensions.  Layered-elastic
strain under the standard dual-tyre axle on a July afternoon at Tucson
(mean of 13:00-16:00 local time, 2017-2024), untreated (albedo 0.14) and
treated (0.45).  The cut face through the tyre centres is coloured by the
transverse horizontal strain (tension positive); the surface carries the
deflection basin, exaggerated.  Path-traced.

usage: python fig_strain3d.py [samples per pixel]
"""
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from cmcrameri import cm as ccm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import let, pathtrace as pt, plotting as P, palette as C, stiffness, strain_table as stt, summaries  # noqa: E402
from alrb.config import FIG  # noqa: E402

TAG = "v3"
KEY = "tucson"
YW = stt.S / 2                   # tyre centres at y = +/- s/2
XL, YL, ZL = 0.55, 0.62, 0.90    # block: x in [0, XL], y in [-YL, YL], depth ZL
EXAG = 250.0                     # deflection exaggeration
SLIM = 300.0                     # colour range (microstrain)
W, H = 1500, 1150
CACHE = FIG / "_render"


def state(a):
    z, idx = summaries.load_run(KEY, a, TAG)
    li = summaries.local_index(idx, KEY)
    m = (li.month == 7) & (li.hour >= 13) & (li.hour <= 16) & (li.year >= 2017) & (li.year <= 2024)
    d = z["depths"]
    T1 = float(z["T"][m, int(np.argmin(np.abs(d - 0.025)))].mean())
    T2 = float(z["T"][m, int(np.argmin(np.abs(d - 0.075)))].mean())
    E1, E2 = stiffness.ac_modulus([T1, T2])
    E = np.array([E1, E2, stiffness.BASE_U.mr, stiffness.SUBBASE_U.mr, stiffness.SUBGRADE_U.mr]) * 1e6
    return T1, T2, E


LAYER_COLS = ["#2a2a2a", "#8f8672", "#b3a584", "#7a6450"]
LAYER_BOTS = [0.10, 0.35, 0.65, 10.0]


def layer_colour(depth):
    out = np.empty((len(depth), 3))
    for i, d in enumerate(depth):
        k = int(np.searchsorted(LAYER_BOTS, d, side="right"))
        out[i] = matplotlib.colors.to_rgb(LAYER_COLS[min(k, 3)])
    return out


def respond(E, X, Y, Z, ks, ws):
    """Sum over the two tyres of (transverse strain along y, vertical strain,
    vertical displacement) at points (X, Y, Z), Z depth > 0."""
    lay = np.array([let.layer_of(z, stt.H) for z in Z])
    eyy = np.zeros(len(X))
    ezz = np.zeros(len(X))
    w = np.zeros(len(X))
    for yt in (-YW, YW):
        dy = Y - yt
        r = np.sqrt(X ** 2 + dy ** 2)
        out = let.response(ks, ws, stt.H, E, stt.NU, stt.Q, stt.A, r, Z, lay)
        er, et = out[:, 5], out[:, 6]
        with np.errstate(invalid="ignore", divide="ignore"):
            c2 = np.where(r > 1e-9, (dy / np.maximum(r, 1e-9)) ** 2, 1.0)
        eyy += er * c2 + et * (1 - c2)
        ezz += out[:, 4]
        w += out[:, 7]
    return eyy, ezz, w


def face(E, ks, ws, P0, U, V, nu_, nv_, zfun):
    """Structured face spanned by P0 + u U + v V, deformed vertically."""
    u = np.linspace(0, 1, nu_)
    v = np.linspace(0, 1, nv_)
    Uu, Vv = np.meshgrid(u, v, indexing="ij")
    P = np.asarray(P0)[None, None, :] + Uu[..., None] * np.asarray(U) + Vv[..., None] * np.asarray(V)
    X, Y, D = P[..., 0].ravel(), P[..., 1].ravel(), np.maximum(-P[..., 2].ravel(), 0.004)
    eyy, ezz, w = respond(E, X, Y, D, ks, ws)
    grid = pv.StructuredGrid(P[..., 0], P[..., 1], P[..., 2] - EXAG * w.reshape(Uu.shape))
    return grid, eyy.reshape(Uu.shape), w.reshape(Uu.shape), D.reshape(Uu.shape)


def geometry(E):
    ks, ws = let.kgrid(stt.A, 0.004)
    shapes = {}
    # cut face x = 0 (y from +YL to -YL, depth from 0 to ZL), coloured by transverse strain
    zfrac = np.concatenate([np.linspace(0, 0.10 / ZL, 40), np.linspace(0.105 / ZL, 1.0, 110)])
    ys = np.linspace(-YL, YL, 121)
    Y, Zd = np.meshgrid(ys, zfrac * ZL, indexing="ij")
    Dd = np.maximum(Zd, 0.004)
    eyy, ezz, w = respond(E, np.zeros(Y.size), Y.ravel(), Dd.ravel(), ks, ws)
    eyy, w = eyy.reshape(Y.shape), w.reshape(Y.shape)
    cut = pv.StructuredGrid(np.zeros_like(Y), Y, -Zd - EXAG * w)
    cut.point_data["e"] = (eyy * 1e6).ravel(order="F")
    surf = cut.extract_surface(algorithm="dataset_surface")
    shapes["cut"] = pt.colour_mesh(surf, np.asarray(surf.point_data["e"]), ccm.vik, -SLIM, SLIM,
                                   bsdf_kind="principled", roughness=0.7, specular=0.2)
    # top surface
    top, _, wtop, _ = face(E, ks, ws, (0.0, -YL, 0.0), (XL, 0, 0), (0, 2 * YL, 0), 46, 121, None)
    shapes["top"] = pt.mesh(top.extract_surface(algorithm="dataset_surface"), pt.principled("#474747", roughness=0.6,
                                                                                           specular=0.35), smooth=True)
    # side face y = -YL and back face x = XL, coloured by layer
    for nm, P0, U, V, nu_, nv_ in (("side", (0.0, -YL, 0.0), (XL, 0, 0), (0, 0, -ZL), 30, 90),
                                   ("back", (XL, -YL, 0.0), (0, 2 * YL, 0), (0, 0, -ZL), 60, 90)):
        g, _, _, D = face(E, ks, ws, P0, U, V, nu_, nv_, None)
        sf = g.extract_surface(algorithm="dataset_surface")
        sf.point_data["d"] = D.ravel(order="F")
        sf2 = sf.extract_surface(algorithm="dataset_surface")
        shapes[nm] = layered_mesh(sf2, None)
    # layer interfaces on the cut face, following the deformation
    ifc = None
    for d in (0.10, 0.35, 0.65):
        k = int(np.argmin(np.abs(zfrac * ZL - d)))
        pts = np.column_stack([np.full(len(ys), -0.004), ys, -Zd[:, k] - EXAG * w[:, k]])
        t = pt.tube(pts, 0.0022, n=10)
        ifc = t if ifc is None else ifc + t
    shapes["ifc"] = pt.mesh(ifc, pt.principled("#f2efe6", roughness=0.5), smooth=True)
    # tyre contact patches and load arrows
    patches, arrows = None, None
    for yt in (-YW, YW):
        disc = pv.Disc(center=(0.0, yt, 0.0), inner=0.0, outer=stt.A, normal=(0, 0, 1), c_res=64)
        disc = disc.clip(normal=(1, 0, 0), origin=(0.0, 0, 0), invert=False).extract_surface(algorithm="dataset_surface")
        wc = float(np.interp(yt, ys, w[:, 0]))
        disc.points[:, 2] = 0.010 - EXAG * wc
        patches = disc if patches is None else patches + disc
        for dx in (0.035, 0.075):
            a = pt.arrow((dx, yt, 0.36), (dx, yt, 0.016 - EXAG * wc), 0.011, 0.032, 0.07, n=24)
            arrows = a if arrows is None else arrows + a
    shapes["patches"] = pt.mesh(patches.triangulate(), pt.principled(C.GOLD, roughness=0.5), smooth=False)
    shapes["arrows"] = pt.mesh(arrows, pt.principled(C.BURDEN, roughness=0.4, specular=0.4), smooth=True)
    info = dict(eyy=eyy, ys=ys, zs=zfrac * ZL, wmax=float(w.max()))
    return shapes, info


def layered_mesh(sf, depth):
    """Mesh coloured by the layer at each vertex (sharp boundaries follow the
    vertex spacing, fine enough at the chosen resolution)."""
    sf = sf.triangulate()
    depth = np.asarray(sf.point_data["d"])
    V = np.asarray(sf.points, float)
    F = sf.faces.reshape(-1, 4)[:, 1:]
    Cc = pt.srgb_to_linear(layer_colour(depth))
    import hashlib as _h
    path = pt.CACHE / f"l_{_h.sha1(V.tobytes() + Cc.tobytes()).hexdigest()[:16]}.ply"
    if not path.exists():
        pt.write_ply(path, V, F, None, Cc)
    return {"type": "ply", "filename": str(path), "face_normals": True,
            "bsdf": pt.twosided(pt.vertex_colour("principled", roughness=0.9, specular=0.15))}


_PTS = [(x, y, z) for x in (0.0, XL) for y in (-YL, YL) for z in (-ZL, 0.36)]
_PTS += [(-0.05, YL + 0.35, -0.05), (-0.05, -YL - 0.1, 0.40)]
CAM = pt.fit_camera(_PTS, (-0.78, -0.42, 0.46), 6.0, W, H, box=(0.10, 0.92, 0.04, 0.96))


def main(spp=256):
    P.setup()
    strip = 0.14
    fig = plt.figure(figsize=(P.DOUBLE, P.DOUBLE / 2 * H / W / (1 - strip)))
    fs = 5.8
    res = {}
    for i, a in enumerate((0.14, 0.45)):
        T1, T2, E = state(a)
        shapes, info = geometry(E)
        sig = hashlib.sha1(repr((CAM, sorted(shapes.items(), key=lambda kv: kv[0]))).encode()).hexdigest()[:10]
        f = CACHE / f"sf_{a:.2f}_{sig}_{spp}.npy"
        if f.exists():
            img = np.load(f)
        else:
            img = pt.render(shapes, CAM, spp=spp, sun_dir=(0.45, 0.35, -0.82), sun_irradiance=2.4,
                            sky=(0.82, 0.85, 0.90), sky_strength=1.0, max_depth=6, transparent=True)
            np.save(f, img.astype(np.float32))
        al = img[..., 3:4]
        rgb = img[..., :3] * al + (1 - al)
        ax = fig.add_axes([i * 0.5, strip, 0.5, 1 - strip])
        ax.imshow(rgb, interpolation="lanczos")
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        pr = lambda p: CAM.project([p])[0][0]
        eyy = info["eyy"] * 1e6
        zs, ys = info["zs"], info["ys"]
        kb = int(np.argmin(np.abs(zs - 0.10)))
        emax = float(eyy[:, kb - 1].max())
        jy = int(np.argmax(eyy[:, kb - 1]))
        res[a] = emax
        # labels anchored to the geometry
        for d, lab in ((0.05, "asphalt"), (0.225, "base"), (0.50, "subbase"), (0.78, "subgrade")):
            x, y = pr((-0.01, YL, -d))
            ax.text(x - 14, y, lab, fontsize=fs, ha="right", va="center")
        x, y = pr((-0.004, ys[jy], -0.10 - EXAG * 0.0))
        ax.annotate(rf"$\varepsilon_t={emax:.0f}\,\mu\varepsilon$ at the" "\n" "bottom of the asphalt", xy=(x, y),
                    xytext=(W * 0.04, H * 0.93), fontsize=fs + 0.3, ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=C.INK, shrinkA=1, shrinkB=1))
        ax.plot([x], [y], "o", ms=1.8, color=C.INK, mec="white", mew=0.3)
        title = "untreated" if a == 0.14 else "treated"
        ax.text(0.02, 0.99, rf"\textbf{{({'ab'[i]})}} {title}, $\alpha={a:.2f}$" "\n"
                rf"asphalt {T1:.1f}/{T2:.1f}\,$^{{\circ}}$C, $|E^*|$ {E[0] / 1e9:.2f}/{E[1] / 1e9:.2f}\,GPa",
                transform=ax.transAxes, fontsize=fs + 0.4, ha="left", va="top", linespacing=1.3)
    cax = fig.add_axes([0.30, 0.065, 0.22, 0.024])
    sm = matplotlib.cm.ScalarMappable(cmap=ccm.vik, norm=matplotlib.colors.Normalize(-SLIM, SLIM))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([-300, -150, 0, 150, 300])
    cb.ax.tick_params(labelsize=5.2, length=1.8, pad=1.0)
    cb.set_label(r"Transverse strain on the cut face ($\mu\varepsilon$, tension $+$)", fontsize=5.4, labelpad=1.2)
    cb.ax.xaxis.set_label_position("top")
    fig.text(0.56, 0.075, rf"dual tyres, 2 $\times$ 20\,kN at 690\,kPa; deflection exaggerated {EXAG:g}$\times$;"
             "\nJuly, 13:00--16:00, Tucson, Arizona", fontsize=5.4, ha="left", va="center", color=C.GREY,
             linespacing=1.3)
    P.save(fig, FIG / "fig_strain3d", pdf_dpi=450)
    return res


if __name__ == "__main__":
    print(main(int(sys.argv[1]) if len(sys.argv) > 1 else 256))

"""Figure 1: path-traced cutaway of a street with one coated lane.

Schematic scene, but every quantity drawn to scale comes from the study:
the layer thicknesses of the reference structure; the untreated and laid
albedos, used directly as the diffuse reflectance of the two lanes; and, on
the cut face, the simulated mean temperature field at Goodridge, Minnesota,
in mid-March (untreated profile under the near lane, coated profile under
the far lane), whose 0 degC isotherm is the frost front.  Depths below the
surface are exaggerated 2.5 times.

usage: python fig_street3d.py [samples per pixel]
"""
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
from alrb import pathtrace as pt, plotting as P, palette as C, summaries  # noqa: E402
from alrb.config import FIG, REF  # noqa: E402

TAG = "v3"
EZ = 2.5                                   # vertical exaggeration below ground
ZMAX = 3.2                                 # real depth shown (m)
LAYERS = [("AC", 0.0, 0.10, "#262626"), ("base", 0.10, 0.35, "#8f8672"), ("subbase", 0.35, 0.65, "#b3a584"),
          ("subgrade", 0.65, ZMAX, "#7a6450")]
X0, X1 = -20.0, 16.0                       # street extent along x
YC = 5.0                                   # half-width of the carriageway
YS = 8.0                                   # outer edge of the sidewalks
DOY, HALF = 75, 3                          # mid-March, +/- 3 days, winters 2018-2024
TLIM = 6.0                                 # colour range of the cut face (degC)
SUN = (0.42, 0.55, -0.72)                  # direction of travel of sunlight
CACHE = FIG / "_render"
CACHE.mkdir(parents=True, exist_ok=True)


def zs(d):
    """Scene z of a real depth d (m)."""
    return -EZ * np.asarray(d, float)


def profiles():
    out = {}
    for a in (0.14, 0.45):
        z, idx = summaries.load_run("goodridge", a, TAG)
        li = summaries.local_index(idx, "goodridge")
        m = (np.abs(li.dayofyear.values - DOY) <= HALF) & (li.year >= 2018)
        T = z["T"][m].mean(0)
        d = np.asarray(z["depths"], float)
        d, iu = np.unique(d, return_index=True)
        T = T[iu]
        d = np.concatenate([[0.0], d, [ZMAX]])
        slope = (T[-1] - T[-2]) / (d[-2] - d[-3])
        T = np.concatenate([[T[0]], T, [T[-1] + slope * (ZMAX - d[-2])]])
        out[a] = PchipInterpolator(d, T)
    return out


def blend(y, w=1.3):
    return 0.5 * (1 + np.tanh(y / w))


def cut_face(prof):
    ny, nz = 161, 241
    ys = np.linspace(-YS, YS, ny)
    ds = np.linspace(0.0, ZMAX, nz)
    Y, D = np.meshgrid(ys, ds, indexing="ij")
    s = blend(Y)
    T = (1 - s) * prof[0.14](D) + s * prof[0.45](D)
    grid = pv.StructuredGrid(np.full_like(Y, X0 - 0.004), Y, zs(D))
    grid.point_data["T"] = T.ravel(order="F")
    surf = grid.extract_surface(algorithm="dataset_surface")
    shape = pt.colour_mesh(surf, np.asarray(surf.point_data["T"]), ccm.vik, -TLIM, TLIM, bsdf_kind="principled",
                           roughness=0.8, specular=0.15)
    # 0 degC isotherm
    zf = []
    for j in range(ny):
        col = T[j]
        k = np.where(np.diff(np.sign(col)) != 0)[0]
        k = k[-1] if len(k) else nz - 2
        f = col[k] / (col[k] - col[k + 1])
        zf.append(ds[k] + f * (ds[k + 1] - ds[k]))
    zf = np.array(zf)
    return shape, ys, zf


def block(prof):
    shapes = {}
    for name, dtop, dbot, col in LAYERS:
        b = pv.Box(bounds=(X0, X1, -YS, YS, zs(dbot), zs(dtop) - (0.0005 if name == "AC" else 0.0)))
        shapes[f"layer_{name}"] = pt.mesh(b, pt.principled(col, roughness=0.9 if name != "AC" else 0.75,
                                                           specular=0.2), smooth=False)
    face, ys, zf = cut_face(prof)
    shapes["cutface"] = face
    # frost front (0 degC isotherm) and layer interfaces on the cut face
    pts = np.column_stack([np.full(len(ys), X0 - 0.05), ys, zs(zf)])
    shapes["front"] = pt.mesh(pt.tube(pts, 0.075), pt.principled(C.FROST, roughness=0.35), smooth=True)
    ifc = None
    for d in (0.10, 0.35, 0.65):
        t = pt.tube([(X0 - 0.03, -YS, zs(d)), (X0 - 0.03, YS, zs(d))], 0.03, n=10)
        ifc = t if ifc is None else ifc + t
        t2 = pt.tube([(X0, -YS - 0.03, zs(d)), (X1, -YS - 0.03, zs(d))], 0.03, n=10)
        ifc = ifc + t2
    shapes["interfaces"] = pt.mesh(ifc, pt.principled("#f2efe6", roughness=0.5), smooth=True)
    return shapes, dict(ys=ys, zf=zf)


def surfaces():
    s = {}
    s["lane_u"] = pt.mesh(pv.Box(bounds=(X0, X1, -YC, 0.0, -0.0004, 0.0)),
                          pt.principled({"linear": [REF.alpha0] * 3}, roughness=0.9, specular=0.12), smooth=False)
    coat = REF.alphaf
    s["lane_c"] = pt.mesh(pv.Box(bounds=(X0, X1, 0.0, YC, -0.0004, 0.0003)),
                          pt.principled({"linear": [coat, 0.985 * coat, 0.95 * coat]}, roughness=0.8, specular=0.12),
                          smooth=False)
    marks = None
    for x in np.arange(X0 + 1.0, X1 - 1.0, 6.0):
        b = pv.Box(bounds=(x, x + 3.0, -0.08, 0.08, 0.0, 0.004))
        marks = b if marks is None else marks + b
    for yy in (-YC + 0.25, YC - 0.25):
        marks = marks + pv.Box(bounds=(X0, X1, yy - 0.06, yy + 0.06, 0.0, 0.004))
    s["marks"] = pt.mesh(marks, pt.principled("#f4f1e8", roughness=0.6), smooth=False)
    for sgn, nm in ((1, "far"), (-1, "near")):
        y0, y1 = sorted((sgn * YC, sgn * YS))
        s[f"walk_{nm}"] = pt.mesh(pv.Box(bounds=(X0, X1, y0, y1, 0.0, 0.15)),
                                  pt.principled("#cfc9bd", roughness=0.9), smooth=False)
    return s


def building(x0, x1, y0, y1, h, colour):
    body = pv.Box(bounds=(x0, x1, y0, y1, 0.0, h))
    parapet = pv.Box(bounds=(x0 - 0.12, x1 + 0.12, y0 - 0.12, y1 + 0.12, h, h + 0.35))
    wins = None
    nfl = int((h - 1.0) // 3.2)
    for k in range(nfl):
        z0 = 1.5 + 3.2 * k
        for xw in np.arange(x0 + 0.9, x1 - 1.3, 2.1):
            w = pv.Box(bounds=(xw, xw + 1.2, y0 - 0.05, y0 + 0.01, z0, z0 + 1.6))
            wins = w if wins is None else wins + w
    return body + parapet, wins


def city(rng):
    s = {}
    cols = ["#dcd3c4", "#cfc3b1", "#e4ddd1", "#c9bba8", "#d6cec4", "#c2b6a5"]
    plinth = pv.Box(bounds=(X0, X1, YS, YS + 13.0, -1.2, 0.0))
    s["plinth"] = pt.mesh(plinth, pt.principled("#8d8578", roughness=0.95), smooth=False)
    x, i = X0 + 9.0, 0
    bodies, windows = None, None
    while x < X1 - 2.0:
        w = min(rng.uniform(6.5, 10.0), X1 - x)
        h = rng.uniform(8.0, 14.5)
        body, wins = building(x + 0.15, x + w - 0.15, YS + 1.0, YS + 12.5, h, cols[i % len(cols)])
        s[f"b{i}"] = pt.mesh(body, pt.principled(cols[i % len(cols)], roughness=0.85, specular=0.2), smooth=False)
        if wins is not None:
            windows = wins if windows is None else windows + wins
        x += w
        i += 1
    s["windows"] = pt.mesh(windows, pt.principled("#3b4450", roughness=0.15, specular=0.6), smooth=False)
    trunks, crowns = None, None
    for (tx, ty) in [(-15.0, 6.7), (-5.0, 6.7), (5.0, 6.7), (14.0, 6.7)]:
        tr = pv.Cylinder(center=(tx, ty, 1.6), direction=(0, 0, 1), radius=0.16, height=3.2, resolution=20)
        cr = pv.Icosphere(radius=1.9, center=(tx, ty, 4.4), nsub=4)
        p = np.asarray(cr.points) - (tx, ty, 4.4)
        n = p / np.linalg.norm(p, axis=1, keepdims=True)
        bump = 1 + 0.08 * np.sin(4 * n[:, 0] + 1.3 * tx) * np.cos(3 * n[:, 1] - 2 * n[:, 2]) \
            + 0.05 * np.sin(9 * n[:, 2] + 2 * n[:, 0])
        cr.points = np.array((tx, ty, 4.4)) + n * 1.9 * bump[:, None] * np.array([1.0, 1.0, 0.88])
        trunks = tr if trunks is None else trunks + tr
        crowns = cr if crowns is None else crowns + cr
    s["trunks"] = pt.mesh(trunks, pt.principled("#5b4636", roughness=0.9), smooth=True)
    s["crowns"] = pt.mesh(crowns, pt.principled("#6a9150", roughness=0.8, specular=0.15), smooth=True)
    return s


def truck():
    s = {}
    y0, y1 = 1.3, 3.8
    x0 = -1.0
    s["trailer"] = pt.mesh(pv.Box(bounds=(x0, x0 + 8.6, y0, y1, 1.15, 3.9)),
                           pt.principled("#eceae4", roughness=0.45, specular=0.4), smooth=False)
    s["cab"] = pt.mesh(pv.Box(bounds=(x0 + 8.8, x0 + 11.1, y0 + 0.05, y1 - 0.05, 1.0, 3.25)),
                       pt.principled("#4a5a70", roughness=0.35, specular=0.5), smooth=False)
    s["cabwin"] = pt.mesh(pv.Box(bounds=(x0 + 11.08, x0 + 11.12, y0 + 0.3, y1 - 0.3, 2.15, 3.0)),
                          pt.principled("#2b3440", roughness=0.1, specular=0.7), smooth=False)
    wheels = None
    for xw in (x0 + 1.2, x0 + 2.4, x0 + 7.4, x0 + 10.2):
        for yw in (y0 + 0.24, y1 - 0.24):
            c = pv.Cylinder(center=(xw, yw, 0.5), direction=(0, 1, 0), radius=0.5, height=0.42, resolution=48)
            wheels = c if wheels is None else wheels + c
    s["wheels"] = pt.mesh(wheels, pt.principled("#1d1d1d", roughness=0.7), smooth=True)
    return s, np.array([x0 + 2.4, y0 + 0.02, 0.5])


def rays():
    s, anchors = {}, {}
    u = -np.asarray(SUN, float)
    u /= np.linalg.norm(u)
    hits = {"u": [(-16.0, -2.6), (-9.0, -2.6)], "c": [(-16.0, 2.6), (-9.0, 2.6)]}
    inc = None
    for grp in ("u", "c"):
        for hx, hy in hits[grp]:
            h = np.array([hx, hy, 0.03])
            a = pt.arrow(h + u * 12.0, h + u * 0.05, 0.08, 0.30, 0.9)
            inc = a if inc is None else inc + a
    anchors["inc"] = np.array([hits["c"][0][0], hits["c"][0][1], 0.0]) + u * 12.0
    d = np.array([0.0, -0.22, 1.0])
    d /= np.linalg.norm(d)
    for grp, col in (("u", C.WARM), ("c", C.CREDIT)):
        mesh = None
        L = 26.0 * (REF.alpha0 if grp == "u" else REF.alphaf)
        for hx, hy in hits[grp]:
            h = np.array([hx + 1.2, hy, 0.05])
            a = pt.arrow(h, h + d * L, 0.10, 0.34, 1.0)
            mesh = a if mesh is None else mesh + a
            anchors.setdefault(grp, []).append(h + d * L)
        s[f"refl_{grp}"] = pt.mesh(mesh, pt.principled(col, roughness=0.45, specular=0.3), smooth=True)
    s["inc"] = pt.mesh(inc, pt.principled(C.GOLD, roughness=0.45, specular=0.3), smooth=True)
    return s, anchors


_BOX = [(x, y, z) for x in (X0, X1) for y in (-YS, YS + 13.0) for z in (zs(ZMAX), 0.0)]
_BOX += [(x, YS + 1.0, 15.0) for x in (X0 + 9.0, X1)] + [(x, YS + 12.5, 15.0) for x in (X0 + 9.0, X1)]
CAM = pt.fit_camera(_BOX, (-0.80, -0.45, 0.42), 125.0, 3000, 1750, box=(0.12, 0.80, 0.03, 0.97))


def build_and_render(spp):
    prof = profiles()
    rng = np.random.default_rng(3)
    shapes = {}
    blk, fr = block(prof)
    shapes.update(blk)
    shapes.update(surfaces())
    shapes.update(city(rng))
    tr, wheel = truck()
    shapes.update(tr)
    r, anchors = rays()
    shapes.update(r)
    import hashlib
    sig = hashlib.sha1(repr((CAM.origin, CAM.target, CAM.fov, CAM.width, CAM.height, SUN, EZ, DOY, TLIM,
                             sorted(shapes.items(), key=lambda kv: kv[0]))).encode()).hexdigest()[:10]
    f = CACHE / f"street_{sig}_{spp}.npy"
    if f.exists():
        img = np.load(f)
    else:
        img = pt.render(shapes, CAM, spp=spp, sun_dir=SUN, sun_irradiance=3.0, sky=(0.72, 0.79, 0.90),
                        sky_strength=1.0, max_depth=6, transparent=True)
        np.save(f, img.astype(np.float32))
    return img, dict(prof=prof, fr=fr, anchors=anchors, wheel=wheel)


def over_white(img):
    a = img[..., 3:4]
    return img[..., :3] * a + (1 - a), a[..., 0]


def label_clear(fig, ax, alpha, texts, thr=0.02):
    """Problems for labels drawn over rendered (opaque) pixels."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    probs = []
    H, W = alpha.shape
    for t in texts:
        bb = matplotlib.text.Text.get_window_extent(t, renderer=r).expanded(1.04, 1.1)   # text only
        (x0, y1), (x1, y0) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        xa, xb = int(max(0, np.floor(min(x0, x1)))), int(min(W, np.ceil(max(x0, x1))))
        ya, yb = int(max(0, np.floor(min(y0, y1)))), int(min(H, np.ceil(max(y0, y1))))
        if xb <= xa or yb <= ya:
            continue
        frac = float((alpha[ya:yb, xa:xb] > 0.5).mean())
        if frac > thr:
            probs.append(dict(type="label_over_render", text=t.get_text()[:40], frac=round(frac, 3)))
    return probs


def main(spp=320):
    img, info = build_and_render(spp)
    rgb, alpha = over_white(img)
    H, W = alpha.shape
    P.setup()
    fig = plt.figure(figsize=(P.DOUBLE, P.DOUBLE * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="lanczos")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    pr = lambda p: CAM.project([p])[0][0]
    fs = 6.3
    texts = []

    def lab(text, anchor, xy, ha="left", va="center", colour=C.INK):
        x, y = pr(anchor)
        t = ax.annotate(text, xy=(x, y), xytext=xy, textcoords="data", fontsize=fs, ha=ha, va=va, color=colour,
                        arrowprops=dict(arrowstyle="-", lw=0.45, color=C.INK, shrinkA=1.5, shrinkB=0.0,
                                        connectionstyle="arc3,rad=0"), annotation_clip=False)
        ax.plot([x], [y], "o", ms=1.7, color=C.INK, mec="white", mew=0.3)
        texts.append(t)
        return t

    prof, fr = info["prof"], info["fr"]
    F = lambda fx, fy: (fx * W, fy * H)
    # ---------------------------------------------------------------- key for the radiation streams
    kx, ky, dy = 0.030, 0.060, 0.052
    for i, (col, txt) in enumerate([(C.GOLD, r"incident shortwave"),
                                    (C.CREDIT, r"reflected by the coated lane ($\alpha_f=0.45$)"),
                                    (C.WARM, r"reflected by the untreated lane ($\alpha_0=0.14$)")]):
        y = (ky + i * dy) * H
        ax.annotate("", xy=((kx + 0.045) * W, y), xytext=(kx * W, y),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", lw=1.6, color=col,
                                    shrinkA=0, shrinkB=0))
        t = ax.text((kx + 0.055) * W, y, txt, fontsize=fs, va="center", ha="left", color=C.INK)
        texts.append(t)
    t = ax.text(kx * W, (ky + 3 * dy + 0.004) * H,
                r"arrow lengths $\propto$ albedo; the TOA forcing" "\n" r"of the difference is $K\,\Delta\alpha$",
                fontsize=fs - 0.4, va="top", ha="left", color=C.GREY)
    texts.append(t)
    # ---------------------------------------------------------------- surfaces
    lab(r"coated lane: $\alpha_f=0.45$ when laid, decaying" "\n"
        r"with $\tau=0.91$\,yr, renewed every $T_r=3$\,yr", (-17.0, 3.6, 0.0), F(0.030, 0.40))
    lab(rf"untreated asphalt, $\alpha_0={REF.alpha0:.2f}$", (6.0, -3.2, 0.0), F(0.790, 0.415))
    lab(r"traffic loading: tensile strain at the" "\n" r"bottom of the asphalt, vertical" "\n"
        r"strain at the top of the subgrade", info["wheel"], F(0.640, 0.105))
    # ---------------------------------------------------------------- structure (near side face)
    names = [("AC", r"asphalt concrete, 100\,mm", 0.05), ("base", r"granular base, 250\,mm", 0.225),
             ("subbase", r"granular subbase, 300\,mm", 0.50), ("subgrade", r"frost-susceptible subgrade", 1.4)]
    for k, (nm, txt, d) in enumerate(names):
        a = (X0 + 13.0, -YS - 0.02, zs(d))
        lab(txt, a, F(0.790, 0.700 + 0.058 * k))
    # ---------------------------------------------------------------- cut face: frost fronts
    iu = int(np.argmin(np.abs(fr["ys"] + 4.0)))
    ic = int(np.argmin(np.abs(fr["ys"] - 4.0)))
    pu = (X0 - 0.05, fr["ys"][iu], zs(fr["zf"][iu]))
    pc = (X0 - 0.05, fr["ys"][ic], zs(fr["zf"][ic]))
    lab(r"0\,$^{\circ}$C isotherm (frost front)" "\n" rf"under the coated lane: {fr['zf'][ic]:.2f}\,m", pc,
        F(0.030, 0.660), colour=C.FROST)
    lab(r"0\,$^{\circ}$C isotherm (frost front)" "\n" rf"under the untreated lane: {fr['zf'][iu]:.2f}\,m", pu,
        F(0.030, 0.775), colour=C.FROST)
    t = ax.text(W * 0.985, H * 0.985, rf"depths $\times${EZ:g}; mean field for 13--19 March, Goodridge, Minnesota",
                fontsize=5.6, ha="right", va="bottom", color=C.GREY)
    texts.append(t)
    # colour bar for the cut face
    cax = fig.add_axes([0.035, 0.085, 0.20, 0.022])
    sm = matplotlib.cm.ScalarMappable(cmap=ccm.vik, norm=matplotlib.colors.Normalize(-TLIM, TLIM))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([-6, -3, 0, 3, 6])
    cb.ax.tick_params(labelsize=5.6, length=2, pad=1.2)
    cb.set_label(r"Temperature on the cut face ($^{\circ}$C)", fontsize=5.8, labelpad=1.5)
    cb.ax.xaxis.set_label_position("top")
    extra = label_clear(fig, ax, alpha, texts)
    if extra:
        for e in extra:
            print("   ", e)
    probs = P.save(fig, FIG / "fig_street3d", pdf_dpi=450, strict=False)
    if (probs or extra) and not __import__("os").environ.get("ALRB_AUDIT_REPORT_ONLY"):
        raise RuntimeError("fig_street3d: layout problems")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 320)

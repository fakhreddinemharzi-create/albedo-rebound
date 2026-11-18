"""Physically based path tracing of the three-dimensional figures with
Mitsuba 3 (Jakob et al., 2022): unidirectional path tracing with a
directional sun and a uniform sky, so that shading, soft sky light and
inter-reflection are computed rather than faked.

Geometry is built with PyVista (or plain arrays), written as binary PLY
(with optional per-vertex colours, read by Mitsuba as the ``vertex_color``
attribute), and the camera model is replicated here so that any world point
can be projected to image pixels; the figures are then composited in
Matplotlib, where every label is typeset with the manuscript fonts and
passes the same overlap audit as the data figures."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import mitsuba as mi

for _v in ("llvm_ad_rgb", "scalar_rgb"):
    try:
        mi.set_variant(_v)
        break
    except Exception:  # pragma: no cover - depends on the host
        continue

CACHE = Path(os.environ.get("ALRB_PT_CACHE", "/tmp/alrb_pathtrace"))
CACHE.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- colours
def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def lin(h_or_rgb):
    """Linear reflectance from a hex string, an sRGB triple, or
    ``{"linear": [r, g, b]}`` (used as given, e.g. an albedo)."""
    if isinstance(h_or_rgb, dict):
        return [float(v) for v in h_or_rgb["linear"]]
    rgb = hex_rgb(h_or_rgb) if isinstance(h_or_rgb, str) else np.asarray(h_or_rgb, float)
    return srgb_to_linear(rgb).tolist()


# ---------------------------------------------------------------- materials
def diffuse(c):
    return {"type": "diffuse", "reflectance": {"type": "rgb", "value": lin(c)}}


def plastic(c, alpha=0.25, ior=1.5):
    return {"type": "roughplastic", "diffuse_reflectance": {"type": "rgb", "value": lin(c)},
            "alpha": alpha, "int_ior": ior}


def principled(c, roughness=0.6, specular=0.4, metallic=0.0):
    return {"type": "principled", "base_color": {"type": "rgb", "value": lin(c)},
            "roughness": roughness, "specular": specular, "metallic": metallic}


def vertex_colour(kind="diffuse", **kw):
    tex = {"type": "mesh_attribute", "name": "vertex_color"}
    if kind == "diffuse":
        return {"type": "diffuse", "reflectance": tex}
    if kind == "plastic":
        return {"type": "roughplastic", "diffuse_reflectance": tex, "alpha": kw.get("alpha", 0.3)}
    return {"type": "principled", "base_color": tex, "roughness": kw.get("roughness", 0.55),
            "specular": kw.get("specular", 0.3)}


def glass_tint(c, alpha=0.05, ior=1.2):
    """Thin translucent sheet (used for the frost fronts)."""
    return {"type": "thindielectric", "int_ior": ior,
            "specular_transmittance": {"type": "rgb", "value": lin(c)}}


def translucent(c, opacity=0.45):
    return {"type": "mask", "opacity": {"type": "rgb", "value": [opacity] * 3}, "bsdf": diffuse(c)}


def emitter_mat(c, strength=4.0):
    return lin(c), strength


# ---------------------------------------------------------------- geometry
def write_ply(path, V, F, N=None, C=None):
    V = np.asarray(V, np.float32)
    F = np.asarray(F, np.int32)
    head = ["ply", "format binary_little_endian 1.0", f"element vertex {len(V)}",
            "property float x", "property float y", "property float z"]
    cols = [V]
    if N is not None:
        head += ["property float nx", "property float ny", "property float nz"]
        cols.append(np.asarray(N, np.float32))
    if C is not None:
        head += ["property float r", "property float g", "property float b"]
        cols.append(np.asarray(C, np.float32))
    head += [f"element face {len(F)}", "property list uchar int vertex_indices", "end_header"]
    vert = np.ascontiguousarray(np.hstack(cols), dtype="<f4")
    face = np.empty(len(F), dtype=[("n", "u1"), ("i", "<i4", (3,))])
    face["n"] = 3
    face["i"] = F
    with open(path, "wb") as fh:
        fh.write(("\n".join(head) + "\n").encode("ascii"))
        fh.write(vert.tobytes())
        fh.write(face.tobytes())


def _key(*arrs):
    h = hashlib.sha1()
    for a in arrs:
        if a is not None:
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def mesh(poly_or_VF, bsdf, colours=None, smooth=True, flip=False, emitter=None):
    """Mitsuba shape dictionary from a PyVista PolyData (triangulated here)
    or a (V, F) pair; ``colours`` are sRGB per-vertex colours in [0, 1]."""
    if isinstance(poly_or_VF, tuple):
        V, F = poly_or_VF
        V = np.asarray(V, float)
        F = np.asarray(F, int)
        N = None
    else:
        import pyvista as pv  # noqa: F401
        poly = poly_or_VF.triangulate()
        if smooth:
            poly = poly.compute_normals(split_vertices=True, feature_angle=35.0, auto_orient_normals=False,
                                        consistent_normals=False)
            N = np.asarray(poly.point_data["Normals"])
        else:
            N = None
        if colours is not None and len(colours) != poly.n_points:
            # colours supplied on the original points: carry through the split via the original ids
            raise ValueError("colour array does not match the triangulated mesh")
        V = np.asarray(poly.points, float)
        F = poly.faces.reshape(-1, 4)[:, 1:]
    if flip:
        F = F[:, ::-1]
        if N is not None:
            N = -N
    C = None if colours is None else srgb_to_linear(np.clip(np.asarray(colours, float)[:, :3], 0, 1))
    path = CACHE / f"m_{_key(V, F, N, C)}.ply"
    if not path.exists():
        write_ply(path, V, F, N, C)
    d = {"type": "ply", "filename": str(path), "bsdf": bsdf, "face_normals": N is None}
    if emitter is not None:
        rgb, s = emitter
        d["emitter"] = {"type": "area", "radiance": {"type": "rgb", "value": (np.asarray(rgb) * s).tolist()}}
    return d


def twosided(bsdf):
    return {"type": "twosided", "bsdf": bsdf}


def colour_mesh(poly, scalars, cmap, vmin, vmax, bsdf_kind="principled", two_sided=True, **kw):
    """Mesh whose vertex colours follow a colormap (colours are carried with
    the vertices through the normal-splitting step)."""
    import matplotlib
    poly = poly.triangulate()
    poly.point_data["_s"] = np.asarray(scalars, float)
    poly = poly.compute_normals(split_vertices=True, feature_angle=35.0, auto_orient_normals=False,
                                consistent_normals=False)
    s = np.asarray(poly.point_data["_s"])
    cm = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    rgb = cm(np.clip((s - vmin) / (vmax - vmin), 0, 1))[:, :3]
    V = np.asarray(poly.points, float)
    F = poly.faces.reshape(-1, 4)[:, 1:]
    N = np.asarray(poly.point_data["Normals"])
    C = srgb_to_linear(rgb)
    path = CACHE / f"c_{_key(V, F, N, C)}.ply"
    if not path.exists():
        write_ply(path, V, F, N, C)
    b = vertex_colour(bsdf_kind, **kw)
    return {"type": "ply", "filename": str(path), "bsdf": twosided(b) if two_sided else b, "face_normals": False}


def tube(points, radius, n=16):
    import pyvista as pv
    line = pv.lines_from_points(np.asarray(points, float))
    return line.tube(radius=radius, n_sides=n, capping=True)


def arrow(start, end, shaft_r, head_r, head_len, n=24):
    """Arrow as a PolyData (cylinder shaft + cone head)."""
    import pyvista as pv
    s, e = np.asarray(start, float), np.asarray(end, float)
    d = e - s
    L = np.linalg.norm(d)
    u = d / L
    shaft_end = s + u * max(L - head_len, 1e-6)
    shaft = pv.Cylinder(center=(s + shaft_end) / 2, direction=u, radius=shaft_r, height=np.linalg.norm(shaft_end - s),
                        resolution=n, capping=True)
    head = pv.Cone(center=shaft_end + u * head_len / 2, direction=u, height=head_len, radius=head_r, resolution=n,
                   capping=True)
    return shaft.triangulate() + head.triangulate()


# ---------------------------------------------------------------- camera
@dataclass
class Camera:
    origin: tuple
    target: tuple
    up: tuple = (0.0, 0.0, 1.0)
    fov: float = 30.0            # horizontal field of view (degrees)
    width: int = 2400
    height: int = 1500
    ortho_scale: float | None = None   # half-width of the view for an orthographic camera
    _frame: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        o, t, up = (np.asarray(v, float) for v in (self.origin, self.target, self.up))
        d = t - o
        d /= np.linalg.norm(d)
        left = np.cross(up, d)
        left /= np.linalg.norm(left)
        nup = np.cross(d, left)
        self._frame = np.vstack([left, nup, d])

    def to_world(self):
        return mi.ScalarTransform4f().look_at(origin=list(map(float, self.origin)),
                                              target=list(map(float, self.target)),
                                              up=list(map(float, self.up)))

    def sensor(self, spp=256, rfilter="gaussian", rgba=False):
        film = {"type": "hdrfilm", "width": self.width, "height": self.height, "rfilter": {"type": rfilter},
                "pixel_format": "rgba" if rgba else "rgb"}
        sampler = {"type": "multijitter", "sample_count": spp}
        if self.ortho_scale is not None:
            s = self.ortho_scale
            asp = self.height / self.width
            tw = self.to_world() @ mi.ScalarTransform4f().scale([s, s * asp, 1.0])
            return {"type": "orthographic", "to_world": tw, "film": film, "sampler": sampler}
        return {"type": "perspective", "fov": self.fov, "fov_axis": "x", "to_world": self.to_world(),
                "film": film, "sampler": sampler, "near_clip": 0.01, "far_clip": 1e4}

    def project(self, P):
        """World points -> image pixel coordinates (x right, y down) and depth."""
        P = np.atleast_2d(np.asarray(P, float))
        q = (P - np.asarray(self.origin, float)) @ self._frame.T     # (left, up, forward)
        if self.ortho_scale is not None:
            s = self.ortho_scale
            xn = q[:, 0] / s
            yn = q[:, 1] / (s * self.height / self.width)
        else:
            tx = np.tan(np.radians(self.fov) / 2.0)
            xn = q[:, 0] / (q[:, 2] * tx)
            yn = q[:, 1] / (q[:, 2] * tx) * (self.width / self.height)
        px = self.width * (1.0 - xn) / 2.0
        py = self.height * (1.0 - yn) / 2.0
        return np.column_stack([px, py]), q[:, 2]


def fit_camera(points, direction, distance, width, height, box=(0.1, 0.9, 0.05, 0.95), up=(0, 0, 1),
               iters=30):
    """Camera looking along ``-direction`` from ``distance`` whose view puts
    the projected bounding box of ``points`` in the image fraction ``box``
    (x0, x1, y0, y1; the tighter of the two axes decides the field of view)."""
    P = np.atleast_2d(np.asarray(points, float))
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    target = P.mean(axis=0)
    fov = 30.0
    for _ in range(iters):
        cam = Camera(origin=tuple(target + distance * d), target=tuple(target), up=up, fov=fov, width=width,
                     height=height)
        px, _ = cam.project(P)
        fx = (px[:, 0].min() / width, px[:, 0].max() / width)
        fy = (px[:, 1].min() / height, px[:, 1].max() / height)
        # shift the target in the image plane to centre the box
        cx = 0.5 * (fx[0] + fx[1]) - 0.5 * (box[0] + box[1])
        cy = 0.5 * (fy[0] + fy[1]) - 0.5 * (box[2] + box[3])
        tx = np.tan(np.radians(fov) / 2)
        left, nup = cam._frame[0], cam._frame[1]
        # content right of the desired centre -> look further right (towards -left); lower -> look down
        target = target - left * (cx * 2 * tx * distance) - nup * (cy * 2 * tx * distance * height / width)
        sx = (fx[1] - fx[0]) / (box[1] - box[0])
        sy = (fy[1] - fy[0]) / (box[3] - box[2])
        scale = max(sx, sy)
        fov = float(np.degrees(2 * np.arctan(tx * scale)))
    return Camera(origin=tuple(target + distance * d), target=tuple(target), up=up, fov=fov, width=width,
                  height=height)


# ---------------------------------------------------------------- render
def render(shapes: dict, cam: Camera, spp=256, sun_dir=(-0.45, 0.55, -0.7), sun_irradiance=3.2,
           sun_colour=(1.0, 0.96, 0.9), sky=(0.62, 0.70, 0.82), sky_strength=0.85, max_depth=8, envmap=None,
           seed=0, denoise=True, transparent=False):
    """Path-trace ``shapes`` and return an sRGB float image in [0, 1]
    (RGBA with straight alpha when ``transparent``: the sky still lights the
    scene but is not seen, so the figure background can be white)."""
    scene = {"type": "scene", "integrator": {"type": "path", "max_depth": max_depth, "hide_emitters": transparent},
             "sensor": cam.sensor(spp, rgba=transparent)}
    if sun_irradiance > 0:
        scene["sun"] = {"type": "directional", "direction": list(map(float, sun_dir)),
                        "irradiance": {"type": "rgb", "value": (np.asarray(sun_colour) * sun_irradiance).tolist()}}
    if envmap is not None:
        scene["sky"] = {"type": "envmap", "filename": str(envmap), "scale": sky_strength}
    else:
        scene["sky"] = {"type": "constant", "radiance": {"type": "rgb", "value": (np.asarray(sky) * sky_strength).tolist()}}
    scene.update(shapes)
    sc = mi.load_dict(scene)
    img = mi.render(sc, spp=spp, seed=seed)
    raw = np.asarray(img, dtype=np.float32)
    arr = raw[..., :3]
    alpha = raw[..., 3:4] if transparent and raw.shape[-1] >= 4 else None
    if alpha is not None:
        # the film stores premultiplied colour: un-premultiply before tone mapping
        arr = np.where(alpha > 1e-4, arr / np.maximum(alpha, 1e-4), 0.0)
    if denoise:
        arr = _bilateral(arr)
    out = _tonemap(arr)
    if alpha is not None:
        return np.concatenate([out, np.clip(alpha, 0, 1)], axis=-1)
    return out


def _tonemap(lin_rgb, exposure=1.0):
    x = np.clip(lin_rgb * exposure, 0, None)
    # gentle highlight roll-off (Reinhard on luminance above 0.8), then sRGB gamma
    L = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    e = np.maximum(L - 0.8, 0.0)
    Lr = np.where(L > 0.8, 0.8 + e / (1 + e / 0.6), L)
    scale = np.where(L > 1e-8, Lr / np.maximum(L, 1e-8), 1.0)
    x = np.clip(x * scale[..., None], 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def _bilateral(img, sigma_s=1.2, sigma_r=0.08, radius=2):
    """Light edge-preserving smoothing of path-tracing noise (CPU fallback
    for the OptiX denoiser)."""
    H, W, _ = img.shape
    out = np.zeros_like(img)
    wsum = np.zeros((H, W, 1), np.float32)
    lum = img.mean(axis=2, keepdims=True)
    pad = np.pad(img, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    padl = np.pad(lum, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ws = np.exp(-(dx * dx + dy * dy) / (2 * sigma_s ** 2))
            sh = pad[radius + dy:radius + dy + H, radius + dx:radius + dx + W]
            shl = padl[radius + dy:radius + dy + H, radius + dx:radius + dx + W]
            wr = np.exp(-((shl - lum) ** 2) / (2 * sigma_r ** 2)) * ws
            out += wr * sh
            wsum += wr
    return out / wsum

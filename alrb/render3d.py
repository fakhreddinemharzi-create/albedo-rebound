"""Off-screen 3-D rendering (PyVista/VTK; Sullivan and Kaszynski, 2019) with
physically based shading, composited into Matplotlib so that every label
uses the manuscript fonts and passes the same overlap audit.

``Scene.render`` returns the RGB image and a function that projects world
coordinates to image pixel coordinates, used to anchor annotations."""
from __future__ import annotations

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True


class Scene:
    def __init__(self, size=(2400, 1500), background="white"):
        self.p = pv.Plotter(off_screen=True, window_size=size, lighting="none")
        self.p.set_background(background)
        self.size = size

    def add(self, mesh, **kw):
        return self.p.add_mesh(mesh, **kw)

    def lights(self, sun_dir=(0.6, -0.5, 0.8), intensity=1.0, fill=0.35, shadows=True):
        sun = pv.Light(light_type="scene light", intensity=intensity)
        sun.set_direction_angle(elev=float(np.degrees(np.arcsin(sun_dir[2] / np.linalg.norm(sun_dir)))),
                                azim=float(np.degrees(np.arctan2(sun_dir[1], sun_dir[0]))))
        sun.positional = False
        sun.position = tuple(100 * np.asarray(sun_dir, float))
        sun.focal_point = (0, 0, 0)
        self.p.add_light(sun)
        f = pv.Light(light_type="headlight", intensity=fill)
        self.p.add_light(f)
        if shadows:
            self.p.enable_shadows()

    def camera(self, position, focal, up=(0, 0, 1), view_angle=30.0, parallel=False):
        c = self.p.camera
        c.position = position
        c.focal_point = focal
        c.up = up
        c.view_angle = view_angle
        if parallel:
            self.p.enable_parallel_projection()

    def render(self, ssaa=True):
        if ssaa:
            try:
                self.p.enable_anti_aliasing("ssaa")
            except Exception:
                pass
        img = self.p.screenshot(return_img=True, transparent_background=False)
        ren = self.p.renderer
        W, H = self.size

        def project(xyz):
            xyz = np.atleast_2d(np.asarray(xyz, float))
            out = []
            for q in xyz:
                ren.SetWorldPoint(q[0], q[1], q[2], 1.0)
                ren.WorldToDisplay()
                x, y, _ = ren.GetDisplayPoint()
                out.append((x, H - y))            # image coordinates (origin top-left)
            return np.array(out)
        return img, project

    def close(self):
        self.p.close()


def box(x0, x1, y0, y1, z0, z1):
    return pv.Box(bounds=(x0, x1, y0, y1, z0, z1))


def structured_block(x, y, z, values):
    """Rectilinear grid with point data ``values`` (shape nx, ny, nz)."""
    g = pv.RectilinearGrid(x, y, z)
    g.point_data["v"] = np.asarray(values, float).ravel(order="F")
    return g

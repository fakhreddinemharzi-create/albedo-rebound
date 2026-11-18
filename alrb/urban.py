"""Forcing of a street in an urban canyon for the pavement column.

The column (``column.run``) absorbs ``(1 - albedo) * sw`` of shortwave and
``eps * lw`` of longwave and exchanges heat with the air at ``ta`` through a
transfer coefficient that depends on the wind ``ws``.  For the floor of an
infinitely long canyon of aspect ratio h = H/W the four inputs are changed
as follows.

* Shortwave: the floor-mean absorbed flux follows from the radiosity of the
  cross section (``canyon``): for unit diffuse sky irradiance and for unit
  horizontal beam irradiance at every projected zenith angle, with the floor
  at its own albedo (or at the snow albedo while the street is snow covered)
  and walls of albedo ``aw``.  The hourly irradiance is split into beam and
  diffuse parts with the diffuse fraction of Erbs et al. (1982) and the sun
  position is evaluated at the sub-steps of the column, so that the column
  absorbs exactly the floor-mean flux of the canyon.
* Longwave: the floor sees the sky through its sky-view factor
  F = sqrt(1 + h^2) - h and the walls, of emissivity ``eps_w`` at the air
  temperature (plus ``dtw``), elsewhere.
* Wind: the near-surface wind is multiplied by ``wind`` (for example the
  canyon factor (2/pi) exp(-h/4) of Masson, 2000).
* Air: an urban heat-island offset ``uhi`` (K) is added to the air
  temperature at constant specific humidity, and the sky longwave is scaled
  with the fourth power of the absolute air temperature at constant
  emissivity, as in the projected-climate runs.
"""
from __future__ import annotations

import numpy as np

from . import canyon, kernel, solar
from .config import SITE
from .driver import SubForcing

SIGMA = 5.670374419e-8
T0K = 273.15
TP = np.radians(np.linspace(0.0, 89.75, 360))
ALB_SNOW = 0.6           # albedo of snow on the street; must equal the alb_snow default of driver.run


def teb_wind(h):
    """Canyon wind factor (2/pi) exp(-h/4) of the Town Energy Balance."""
    return 2.0 / np.pi * np.exp(-h / 4.0) if h > 0 else 1.0


def floor_tables(h, albedos, aw=0.30, n_floor=80, tp_grid=TP):
    """Floor-mean absorbed flux (W m-2 of floor) per unit diffuse sky
    irradiance and per unit horizontal beam irradiance at every projected
    zenith angle, for every floor albedo in ``albedos``."""
    c = canyon.build(float(h), n_floor)
    surf = c.kind != 2
    Fs = c.F[np.ix_(surf, surf)]
    f_open = c.F[np.ix_(surf, c.kind == 2)].sum(axis=1)
    ns = c.normal[surf]
    kinds = c.kind[surf]
    L = c.length[surf]
    floor = kinds == 0
    wfl = L[floor] / L[floor].sum()
    src = np.empty((surf.sum(), len(tp_grid)))
    for k, tp in enumerate(tp_grid):
        frac, _ = canyon._sunlit(c, tp)
        src[:, k] = frac[surf] * np.clip(ns[:, 0] * np.tan(tp) + ns[:, 1], 0.0, None)
    out = {}
    for a in albedos:
        rho = np.where(kinds == 0, a, aw)
        B = np.eye(surf.sum()) - Fs * rho[None, :]          # E = E0 + F rho E
        E_dif = np.linalg.solve(B, f_open)
        E_beam = np.linalg.solve(B, src)
        out[round(float(a), 4)] = ((1.0 - a) * float(E_dif[floor] @ wfl), (1.0 - a) * (wfl @ E_beam[floor]))
    return out


class CanyonForcing:
    """Pre-computed geometry of the sun and the irradiance split for one site;
    ``apply`` returns the forcing of a street floor of given albedo."""

    def __init__(self, df, key, sub: SubForcing, h, axis_deg=90.0, aw=0.30, eps_w=0.90):
        s = SITE[key]
        self.h = float(h)
        self.sub = sub
        _, _, _, kt, _ = kernel.hourly_kernel(df)
        kd = np.repeat(np.clip(solar.erbs_diffuse_fraction(kt), 0.0, 1.0), sub.nsub)
        kd = np.where(np.isfinite(kd), kd, 1.0)
        zen, az, _ = canyon.solar_angles(df.index, s.lat, s.lon, nsub=sub.nsub, ending=True)
        zen, az = zen.ravel(), az.ravel()
        up = zen < np.radians(89.9)
        tp = np.arctan(np.tan(np.minimum(zen, np.radians(89.9))) * np.abs(np.sin(az - np.radians(axis_deg))))
        self.tp = np.minimum(tp, TP[-1])
        self.D = np.where(up, kd, 1.0) * sub.sw
        self.B = sub.sw - self.D
        self.fsky = float(np.sqrt(1.0 + h * h) - h) if h > 0 else 1.0
        self.aw, self.eps_w = aw, eps_w
        self._tables = {}

    def _absorbed(self, a):
        key = round(float(a), 4)
        if key not in self._tables:
            self._tables.update(floor_tables(self.h, [a], self.aw))
        ad, ab = self._tables[key]
        return ad * self.D + np.interp(self.tp, TP, ab) * self.B

    def apply(self, alb, uhi=0.0, wind=1.0, dtw=0.0) -> SubForcing:
        sub = self.sub
        if self.h > 0:
            sw = self._absorbed(alb) / (1.0 - alb)
            sw_snow = self._absorbed(ALB_SNOW) / (1.0 - ALB_SNOW)
            sw = np.where(sub.snow > 0.5, sw_snow, sw)
        else:
            sw = sub.sw.copy()
        ta = sub.ta + uhi
        lsky = sub.lw * ((ta + T0K) / (sub.ta + T0K)) ** 4
        lwall = self.eps_w * SIGMA * (ta + dtw + T0K) ** 4 + (1.0 - self.eps_w) * lsky
        lw = self.fsky * lsky + (1.0 - self.fsky) * lwall
        return SubForcing(sw, lw, ta, sub.qa, sub.ws * wind, sub.pr, sub.p, sub.snow, sub.index, sub.nsub)

    def kernel_weighted_floor_change(self, K_hourly, keep_hours, a0=0.14, a1=0.45):
        """Kernel-weighted change of the floor-mean absorbed shortwave per
        unit albedo change and per unit global irradiance of an open surface
        (the quantity of the canyon fate curve), for validation."""
        n = self.sub.nsub
        d = (self._absorbed(a1) - self._absorbed(a0)) / (a1 - a0)
        dh = d.reshape(-1, n).sum(1)
        swh = self.sub.sw.reshape(-1, n).sum(1)
        ok = keep_hours & (swh > n * 1.0)
        wK = np.where(ok, K_hourly, 0.0)
        return float(np.sum(wK * np.where(ok, dh / np.maximum(swh, 1e-9), 0.0)) / np.sum(wK))

"""Thermal and hydraulic properties of the pavement and soil layers, the
soil freezing characteristic, and the enthalpy tables used by the column
solver.

* Water retention: van Genuchten (1980), m = 1 - 1/n.
* Freezing characteristic: generalised Clapeyron equation; below the
  depressed freezing point the unfrozen water content is read from the same
  retention curve at the matric head h_f = L_f T / (g T0)
  (Spaans and Baker, 1996; Hansson et al., 2004; Dall'Amico et al., 2011).
* Thermal conductivity: Johansen (1977) normalisation with the
  Cote and Konrad (2005) Kersten function and dry conductivity, frozen and
  unfrozen branches weighted by the ice fraction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import LF, G, T0K, RHO_W

C_W = 4.18e6        # volumetric heat capacity of water (J m-3 K-1)
C_I = 1.88e6        # of ice, per unit volume of water-equivalent ice mass (rho_w c_i)
K_W, K_I = 0.57, 2.20


@dataclass
class Material:
    name: str
    porosity: float           # (-)
    theta_r: float            # residual water content (-)
    vg_alpha: float           # van Genuchten alpha (1/m)
    vg_n: float               # van Genuchten n (-)
    c_solid: float            # volumetric heat capacity of the solid skeleton (J m-3 K-1), bulk
    k_solid: float            # conductivity of the solid grains (W m-1 K-1)
    chi: float = 0.75         # Cote-Konrad dry-conductivity parameters
    eta: float = 1.2
    kappa_u: float = 4.6      # Kersten-function parameter, unfrozen
    kappa_f: float = 1.7      # frozen
    bound: bool = False       # bituminous (no pore water, fixed k)
    k_bound: float = 1.5
    c_bound: float = 2.1e6

    def theta(self, h):
        """Volumetric water content at matric head h (m, negative in suction)."""
        m = 1.0 - 1.0 / self.vg_n
        se = (1.0 + np.abs(self.vg_alpha * np.minimum(h, 0.0)) ** self.vg_n) ** (-m)
        return self.theta_r + (self.porosity - self.theta_r) * se

    def head(self, theta):
        m = 1.0 - 1.0 / self.vg_n
        se = np.clip((theta - self.theta_r) / (self.porosity - self.theta_r), 1e-9, 1.0)
        return -((se ** (-1.0 / m) - 1.0) ** (1.0 / self.vg_n)) / self.vg_alpha

    def unfrozen(self, T, theta_w):
        """Unfrozen water content at temperature T (C) for total water theta_w."""
        h0 = self.head(theta_w) if theta_w < self.porosity - 1e-9 else 0.0
        tstar = G * T0K * h0 / LF                   # depressed freezing point (C, <= 0)
        hf = LF * np.minimum(T, 0.0) / (G * T0K)
        thu = np.where(T < tstar, self.theta(hf), theta_w)
        return np.minimum(thu, theta_w), tstar

    def conductivity(self, theta_w, theta_u):
        if self.bound:
            return np.full_like(np.asarray(theta_u, float), self.k_bound)
        n = self.porosity
        sr = np.clip(theta_w / n, 0.0, 1.0)
        k_dry = self.chi * 10.0 ** (-self.eta * n)
        k_sat_u = self.k_solid ** (1 - n) * K_W ** n
        ice = np.clip(theta_w - theta_u, 0.0, None)
        # frozen saturated conductivity with the unfrozen water retained
        k_sat_f = self.k_solid ** (1 - n) * K_I ** (n - theta_u) * K_W ** theta_u
        ke_u = self.kappa_u * sr / (1.0 + (self.kappa_u - 1.0) * sr)
        ke_f = self.kappa_f * sr / (1.0 + (self.kappa_f - 1.0) * sr)
        k_u = (k_sat_u - k_dry) * ke_u + k_dry
        k_f = (k_sat_f - k_dry) * ke_f + k_dry
        fi = np.where(theta_w > 0, ice / np.maximum(theta_w, 1e-12), 0.0)
        return (1.0 - fi) * k_u + fi * k_f


def enthalpy_table(mat: Material, theta_w: float, T=None):
    """Volumetric enthalpy H(T) (J m-3, zero for unfrozen water at 0 C),
    conductivity k(T) and unfrozen water theta_u(T) on a temperature grid."""
    if T is None:
        T = temperature_grid()
    if mat.bound:
        H = mat.c_bound * T
        return T, H, np.full_like(T, mat.k_bound), np.zeros_like(T)
    thu, _ = mat.unfrozen(T, theta_w)
    ice = theta_w - thu
    C = mat.c_solid + C_W * thu + C_I * ice
    # enthalpy: sensible heat of each phase + latent heat deficit of the ice
    H = mat.c_solid * T + C_W * thu * T + C_I * ice * T - RHO_W * LF * ice
    # enforce strict monotonicity (numerical safety at the freezing point)
    H = np.maximum.accumulate(H + 1e-9 * np.arange(len(H)))
    k = mat.conductivity(theta_w, thu)
    return T, H, k, thu


def temperature_grid():
    """Non-uniform grid, 0.002 K spacing in the freezing range."""
    return np.unique(np.concatenate([
        np.arange(-60.0, -5.0, 0.05),
        np.arange(-5.0, -0.5, 0.005),
        np.arange(-0.5, 0.5, 0.0005),
        np.arange(0.5, 90.0, 0.05),
    ]))


# ------------------------------------------------------------ reference set
ASPHALT = Material("asphalt concrete", 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, bound=True, k_bound=1.5, c_bound=2.1e6)
BASE = Material("crushed-stone base", porosity=0.20, theta_r=0.01, vg_alpha=8.0, vg_n=2.2,
                c_solid=1.60e6, k_solid=3.0, chi=1.70, eta=1.80, kappa_u=4.6, kappa_f=1.7)
SUBBASE = Material("granular subbase", porosity=0.25, theta_r=0.02, vg_alpha=5.0, vg_n=1.9,
                   c_solid=1.55e6, k_solid=2.8, chi=0.75, eta=1.20, kappa_u=4.6, kappa_f=1.7)
SUBGRADE = Material("silty-clay subgrade", porosity=0.43, theta_r=0.089, vg_alpha=1.0, vg_n=1.23,
                    c_solid=1.25e6, k_solid=2.2, chi=0.75, eta=1.20, kappa_u=1.9, kappa_f=0.85)
TOPSOIL = Material("loam topsoil", porosity=0.43, theta_r=0.078, vg_alpha=3.6, vg_n=1.56,
                   c_solid=1.20e6, k_solid=2.5, chi=0.75, eta=1.20, kappa_u=3.55, kappa_f=0.95)

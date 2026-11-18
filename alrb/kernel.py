"""Top-of-atmosphere radiative kernel of a surface-albedo change, hour by
hour, from the measured surface irradiance.

Single-layer atmosphere (Taylor et al., 2007; also Winton, 2005; Donohoe
and Battisti, 2011): the atmosphere absorbs a fraction 1 - mu of the
incoming beam on its first pass and scatters a fraction gamma back to
space; with regional surface albedo a_r,
    Q = S_sfc / S_toa = mu (1 - gamma) / (1 - a_r gamma),
    A_p = mu gamma + mu (1 - gamma)^2 a_r / (1 - a_r gamma).
A small treated patch does not alter the regional multiple reflection, so
the change of planetary albedo per unit patch albedo is
    dA_p / da = mu (1 - gamma)^2 / (1 - a_r gamma)^2 = Q^2 / mu,
and the local instantaneous forcing kernel is
    K = -S_toa k_t^2 / mu    (W m-2 per unit albedo, k_t = Q).
The clearness index k_t is measured; mu is the first-pass transmission of
the non-scattering absorbers: water vapour and ozone after Lacis and Hansen
(1974), with the precipitable water and ozone column of MERRA-2, and
aerosol absorption from the CERES AOD with a single-scattering albedo of
0.92.  Clouds enter through k_t, i.e. through gamma.

The simple kernel used in several LCA studies (Lenton and Vaughan, 2009;
Cherubini et al., 2012), K = -S_sfc T_a with T_a = 0.854, is computed for
comparison.
"""
from __future__ import annotations

import numpy as np

from . import solar

OMEGA_AER = 0.92
T_A_SIMPLE = 0.854


def lh_water(y_cm):
    """Lacis and Hansen (1974) water-vapour absorptivity for path y (cm)."""
    y = np.maximum(np.asarray(y_cm, float), 0.0)
    return 2.9 * y / ((1.0 + 141.5 * y) ** 0.635 + 5.925 * y)


def lh_ozone(x_cm):
    """Lacis and Hansen (1974) ozone absorptivity (visible + UV) for path x (cm NTP)."""
    x = np.maximum(np.asarray(x_cm, float), 0.0)
    vis = 0.02118 * x / (1.0 + 0.042 * x + 0.000323 * x ** 2)
    uv = 1.082 * x / (1.0 + 138.6 * x) ** 0.805 + 0.0658 * x / (1.0 + (103.6 * x) ** 3)
    return vis + uv


def first_pass_transmission(mu, pw_mm, o3_du, aod, p_hpa):
    """mu of the single-layer model: 1 - (water vapour + ozone + aerosol absorption)."""
    m = solar.air_mass(mu)
    mo = 35.0 / np.sqrt(1224.0 * np.clip(mu, 1e-3, 1) ** 2 + 1.0)       # ozone magnification (L&H)
    y = (np.asarray(pw_mm) / 10.0) * m * (np.asarray(p_hpa) / 1013.25)
    x = (np.asarray(o3_du) / 1000.0) * mo
    a_aer = (1.0 - OMEGA_AER) * (1.0 - np.exp(-np.asarray(aod) * m))
    return np.clip(1.0 - lh_water(y) - lh_ozone(x) - a_aer, 0.4, 1.0)


def hourly_kernel(df):
    """Hourly APRP kernel (W m-2 per unit albedo, negative), the simple
    kernel, and the clearness index / first-pass transmission used."""
    g0 = df["g0"].to_numpy()
    mu_z = df["mu"].to_numpy()
    kt = np.where(g0 > 5.0, df["sw"].to_numpy() / np.maximum(g0, 1e-9), 0.0)
    kt = np.clip(kt, 0.0, 1.0)
    mu_a = first_pass_transmission(np.where(mu_z > 0, mu_z, 0.05), df["tqv"].to_numpy(), df["o3"].to_numpy(),
                                   df["aod"].fillna(0.1).to_numpy(), df["p"].to_numpy())
    q = np.minimum(kt, mu_a)
    K = -g0 * q ** 2 / mu_a
    K_simple = -df["sw"].to_numpy() * T_A_SIMPLE
    K_clear = -g0 * np.minimum(df["cs"].to_numpy() / np.maximum(g0, 1e-9), mu_a) ** 2 / mu_a
    K[g0 <= 5.0] = 0.0
    K_clear[g0 <= 5.0] = 0.0
    return K, K_simple, K_clear, kt, mu_a


def aprp_monthly(rsdt, rsut, rsds, rsus):
    """Monthly APRP kernel from all-sky fluxes (e.g. CMIP6 climatologies):
    returns (kernel, mu, gamma)."""
    A = rsut / rsdt
    a = rsus / rsds
    Q = rsds / rsdt
    mu = A + Q * (1.0 - a)
    gam = (mu - Q) / (mu - a * Q)
    K = -rsdt * mu * (1.0 - gam) ** 2 / (1.0 - a * gam) ** 2
    return K, mu, gam

"""Near-surface atmosphere: psychrometrics, incoming long-wave radiation,
turbulent transfer and potential evaporation."""
from __future__ import annotations

import numpy as np

from .config import SIGMA, T0K, CP_AIR, LV

KARMAN = 0.41
RD = 287.05


def esat(tc):
    """Saturation vapour pressure over water (hPa), Tetens/Murray form."""
    tc = np.asarray(tc, float)
    return 6.1078 * np.exp(17.27 * tc / (tc + 237.3))


def esat_ice(tc):
    tc = np.asarray(tc, float)
    return 6.1078 * np.exp(21.875 * tc / (tc + 265.5))


def vapour_pressure(tc, rh):
    """Actual vapour pressure (hPa) from air temperature (C) and RH (%),
    with the saturation reference over ice below 0 C."""
    es = np.where(np.asarray(tc) < 0, esat_ice(tc), esat(tc))
    return np.clip(rh, 1.0, 100.0) / 100.0 * es


def pressure_from_elevation(z):
    """Standard-atmosphere surface pressure (hPa)."""
    return 1013.25 * (1.0 - 2.25577e-5 * z) ** 5.25588


def eps_clear_brutsaert(tc, ea):
    """Clear-sky atmospheric emissivity, Brutsaert (1975): 1.24 (e_a/T_a)^(1/7)."""
    return 1.24 * (ea / (np.asarray(tc) + T0K)) ** (1.0 / 7.0)


def eps_all_sky(eps_clr, cloud):
    """Crawford and Duchon (1999): eps = c + (1 - c) eps_clr."""
    c = np.clip(cloud, 0.0, 1.0)
    return c + (1.0 - c) * eps_clr


def longwave_down(tc, ea, cloud):
    eps = eps_all_sky(eps_clear_brutsaert(tc, ea), cloud)
    return eps * SIGMA * (np.asarray(tc) + T0K) ** 4


def rho_air(tc, p_hpa):
    return p_hpa * 100.0 / (RD * (np.asarray(tc) + T0K))


def qsat(tc, p_hpa):
    """Saturation specific humidity (kg/kg)."""
    e = esat(tc)
    return 0.622 * e / (p_hpa - 0.378 * e)


def specific_humidity(ea, p_hpa):
    return 0.622 * ea / (p_hpa - 0.378 * ea)


def slope_esat(tc):
    """d e_s / dT (hPa K-1)."""
    tc = np.asarray(tc, float)
    return 4098.0 * esat(tc) / (tc + 237.3) ** 2


def psychrometric(p_hpa):
    return CP_AIR * p_hpa / (0.622 * LV)

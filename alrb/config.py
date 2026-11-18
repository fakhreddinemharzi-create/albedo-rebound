"""Single source of truth for the study: sites, periods, paths and the
reference parameter set.

Every number used in the manuscript is either defined here or computed by
the pipeline from the data; the manuscript reads the computed values from
``results/*.json`` through generated LaTeX macros.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("ALRB_RAW", ROOT / "data" / "raw"))   # downloads and model runs (about 1 GB), not versioned
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = ROOT / "figures"
MS = ROOT / "manuscript"

YEARS = list(range(2017, 2025))          # observed meteorology, 8 full years
GOES_YEARS = [2022, 2023, 2024]          # satellite skin-temperature reference
CAL_YEAR = 2023                          # calibration year of the natural surface
H_ANALYSIS = 50.0                        # analysis period (yr)


@dataclass(frozen=True)
class Site:
    key: str
    name: str
    state: str
    wban: str
    lat: float
    lon: float
    elev: float
    tz: int            # local standard time offset from UTC (h)
    koppen: str
    goes: str          # GOES satellite used for the LST reference


SITES = [
    Site("goodridge", "Goodridge", "MN", "04994", 48.3055, -95.8744, 350.5, -6, "Dfb", "goes16"),
    Site("wolfpoint", "Wolf Point", "MT", "94059", 48.4886, -105.2097, 805.6, -7, "BSk", "goes16"),
    Site("ithaca", "Ithaca", "NY", "64758", 42.4400, -76.2461, 374.3, -5, "Dfb", "goes16"),
    Site("lajunta", "La Junta", "CO", "03063", 37.8639, -103.8225, 1336.9, -7, "BSk", "goes16"),
    Site("wooster", "Wooster", "OH", "54856", 40.7642, -81.9103, 335.9, -5, "Dfa", "goes16"),
    Site("bowlinggreen", "Bowling Green", "KY", "63849", 37.2503, -86.2325, 240.8, -6, "Cfa", "goes16"),
    Site("monahans", "Monahans", "TX", "03047", 31.6219, -102.8072, 830.3, -6, "BWh", "goes16"),
    Site("durham", "Durham", "NC", "03758", 35.9706, -79.0931, 171.3, -5, "Cfa", "goes16"),
    Site("merced", "Merced", "CA", "93243", 37.2381, -120.8825, 23.8, -8, "Csa", "goes18"),
    Site("tucson", "Tucson", "AZ", "53131", 32.2394, -111.1697, 833.0, -7, "BWh", "goes18"),
]
SITE = {s.key: s for s in SITES}


# ---------------------------------------------------------------- constants
SIGMA = 5.670374419e-8      # Stefan-Boltzmann (W m-2 K-4)
LF = 3.34e5                 # latent heat of fusion (J kg-1)
LV = 2.45e6                 # latent heat of vaporisation (J kg-1)
RHO_W = 1000.0              # water density (kg m-3)
RHO_I = 917.0               # ice density (kg m-3)
G = 9.80665                 # gravity (m s-2)
T0K = 273.15
CP_AIR = 1005.0
A_EARTH = 5.10072e14        # surface area of the Earth (m2)
S0 = 1361.0                 # total solar irradiance (W m-2), Kopp & Lean (2011)


@dataclass
class Treatment:
    """Reference treatment and inventory (Table 2 of the manuscript)."""
    alpha0: float = 0.14      # weathered bituminous surface
    alphaf: float = 0.45      # coating as laid
    alpha_inf: float = 0.14   # asymptote the coating decays to (default: untreated)
    tau: float = 0.91         # decay time constant (yr), fitted to Ko et al. (2022)
    Tr: float = 3.0           # renewal interval (yr)
    g_co: float = 2.70        # kg CO2e m-2 per application
    g_st: float = 9.0         # kg CO2e m-2 per structural renewal (50 mm mill and fill)
    H: float = H_ANALYSIS


REF = Treatment()

# albedo grid of the thermal/structural sweeps
ALBEDO_GRID = [0.05, 0.10, 0.14, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

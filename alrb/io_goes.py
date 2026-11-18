"""GOES-R ABI Level-2 land-surface (skin) temperature, CONUS sector
(ABI-L2-LSTC, hourly, 2 km), from the NOAA open-data buckets on AWS.

Site pixels are located on the ABI fixed grid with the navigation equations
of the GOES-R Product User Guide (Vol. 5, Sect. 4.2.8).  For every hourly
file the 3 x 3 pixel block centred on each site is extracted together with
the data-quality flag (DQF: 0 high, 1 medium, 2 low, 3 no retrieval) and the
product-quality indicator (PQI: cloud condition, surface type, day/night).
"""
from __future__ import annotations

import io
import re
import time

import h5py
import numpy as np
import requests

BUCKETS = {"goes16": "https://noaa-goes16.s3.amazonaws.com",
           "goes18": "https://noaa-goes18.s3.amazonaws.com"}
R_EQ, R_POL = 6378137.0, 6356752.31414
H_SAT = 35786023.0 + R_EQ
ECC = 0.0818191910435
LON0 = {"goes16": -75.0, "goes18": -137.0}


def latlon_to_scan(lat, lon, lon0):
    """Geodetic latitude/longitude (deg) -> ABI fixed-grid scan angles (x, y) in rad."""
    phi = np.radians(lat)
    lam = np.radians(lon)
    lam0 = np.radians(lon0)
    phic = np.arctan((R_POL ** 2 / R_EQ ** 2) * np.tan(phi))
    rc = R_POL / np.sqrt(1.0 - ECC ** 2 * np.cos(phic) ** 2)
    sx = H_SAT - rc * np.cos(phic) * np.cos(lam - lam0)
    sy = -rc * np.cos(phic) * np.sin(lam - lam0)
    sz = rc * np.sin(phic)
    y = np.arctan(sz / sx)
    x = np.arcsin(-sy / np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
    return x, y


def view_zenith(lat, lon, lon0):
    """Satellite view zenith angle (deg) at a ground point (spherical approximation)."""
    phi = np.radians(lat)
    dl = np.radians(lon - lon0)
    cosg = np.cos(phi) * np.cos(dl)
    g = np.arccos(cosg)
    r, h = R_EQ, H_SAT
    d = np.sqrt(r ** 2 + h ** 2 - 2 * r * h * cosg)
    return np.degrees(np.arcsin(h * np.sin(g) / d))


def list_keys(sat: str, year: int, doy: int) -> list[str]:
    b = BUCKETS[sat]
    out = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": f"ABI-L2-LSTC/{year}/{doy:03d}/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        for k in range(5):
            try:
                t = requests.get(b, params=params, timeout=60).text
                break
            except Exception:
                time.sleep(2 ** k)
        out += re.findall(r"<Key>([^<]+)</Key>", t)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", t)
        if not m:
            return out
        token = m.group(1)


class Extractor:
    """Locate the sites on the grid of the first file, then extract blocks."""

    def __init__(self, sat, sites):
        self.sat = sat
        self.sites = sites
        self.idx = None
        self.session = requests.Session()

    def _locate(self, h):
        xs = h["x"]
        ys = h["y"]
        xsc, xof = float(xs.attrs["scale_factor"][0]), float(xs.attrs["add_offset"][0])
        ysc, yof = float(ys.attrs["scale_factor"][0]), float(ys.attrs["add_offset"][0])
        nx, ny = xs.shape[0], ys.shape[0]
        idx = []
        for s in self.sites:
            x, y = latlon_to_scan(s.lat, s.lon, LON0[self.sat])
            j = int(round((x - xof) / xsc))
            i = int(round((y - yof) / ysc))
            if not (1 <= i < ny - 1 and 1 <= j < nx - 1):
                raise ValueError(f"{s.key} outside the {self.sat} CONUS sector")
            idx.append((i, j))
        self.idx = idx

    def fetch(self, key):
        url = f"{BUCKETS[self.sat]}/{key}"
        for k in range(6):
            try:
                b = self.session.get(url, timeout=120).content
                h = h5py.File(io.BytesIO(b), "r")
                break
            except Exception:
                time.sleep(2 ** k)
        else:
            return None
        if self.idx is None:
            self._locate(h)
        lst = h["LST"]
        sc = float(lst.attrs["scale_factor"][0])
        of = float(lst.attrs["add_offset"][0])
        t = float(h["t"][()])
        rows = []
        for (i, j) in self.idx:
            raw = lst[i - 1:i + 2, j - 1:j + 2].astype(float)
            dq = h["DQF"][i - 1:i + 2, j - 1:j + 2]
            pq = h["PQI"][i - 1:i + 2, j - 1:j + 2]
            val = np.where(raw == 65535, np.nan, raw * sc + of)
            rows.append((val[1, 1], int(dq[1, 1]), int(pq[1, 1]),
                         np.nanmean(np.where(dq <= 1, val, np.nan)) if np.any(dq <= 1) else np.nan,
                         int(np.sum(dq <= 1))))
        return t, rows

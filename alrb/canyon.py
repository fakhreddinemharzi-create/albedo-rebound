"""Urban-form factor of a reflective street in an infinitely long canyon.

A street of width W between walls of height H (aspect ratio h = H/W) is
treated in two dimensions: the floor, the two walls and the opening are
divided into segments, the view factors between segments follow Hottel's
crossed-string rule (exact for any two segments of the convex cross
section), and all surfaces reflect diffusely (radiosity).  The sky is
isotropic and enters through the opening; the direct beam enters with the
zenith angle projected on the cross section, tan(theta_p) = tan(theta)
|sin(A - A_street)|, and lights the parts of the floor and of one wall that
see the sun through the opening.

The urban-form factor is the change of the shortwave flux leaving the
opening per unit change of floor albedo, divided by the change an open
horizontal surface of the same albedo change would reflect:
    psi = d(escaping flux) / (d alpha_floor * S_down),
with S_down the global irradiance on an open horizontal surface.  Because
the reflected light leaving the opening is diffuse, as it is above an open
surface, the top-of-atmosphere kernel applies to it unchanged and the
canyon kernel is psi * K.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Canyon:
    h: float                     # aspect ratio H/W
    a: np.ndarray                # segment start points (n, 2), counter-clockwise
    b: np.ndarray                # segment end points
    kind: np.ndarray             # 0 floor, 1 right wall, 2 opening, 3 left wall
    length: np.ndarray
    normal: np.ndarray           # inward unit normals
    F: np.ndarray                # view factors between segments (row: from)


def build(h: float, n_floor: int = 80, n_wall: int | None = None) -> Canyon:
    """Cross section in units of the street width; floor from (0,0) to (1,0),
    right wall to (1,h), opening back to (0,h), left wall down to (0,0)."""
    n_wall = n_wall or max(8, int(round(n_floor * h)))
    pts, kinds = [], []

    def edge(p0, p1, n, k):
        t = np.linspace(0.0, 1.0, n + 1)
        seg = p0[None, :] + (p1 - p0)[None, :] * t[:, None]
        for i in range(n):
            pts.append((seg[i], seg[i + 1]))
            kinds.append(k)

    c = [np.array(v, float) for v in ((0, 0), (1, 0), (1, h), (0, h))]
    edge(c[0], c[1], n_floor, 0)
    edge(c[1], c[2], n_wall, 1)
    edge(c[2], c[3], n_floor, 2)
    edge(c[3], c[0], n_wall, 3)
    a = np.array([p[0] for p in pts])
    b = np.array([p[1] for p in pts])
    kind = np.array(kinds)
    d = b - a
    length = np.hypot(d[:, 0], d[:, 1])
    normal = np.column_stack([-d[:, 1], d[:, 0]]) / length[:, None]      # inward for CCW order
    # crossed strings: points of segments i < j appear as a_i, b_i, a_j, b_j around the boundary
    dist = lambda p, q: np.hypot(p[:, None, 0] - q[None, :, 0], p[:, None, 1] - q[None, :, 1])  # noqa: E731
    aa, bb, ba, ab = dist(a, a), dist(b, b), dist(b, a), dist(a, b)
    n = len(a)
    F = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    # for i < j: F_ij L_i = (|a_i a_j| + |b_i b_j| - |b_i a_j| - |a_i b_j|) / 2
    num = 0.5 * (aa + bb - ba - ab)
    F[iu] = num[iu] / length[iu[0]]
    # reciprocity for i > j
    F.T[iu] = num[iu] / length[iu[1]]
    same = kind[:, None] == kind[None, :]
    F[same] = 0.0
    F = np.clip(F, 0.0, None)
    return Canyon(h, a, b, kind, length, normal, F)


def _sunlit(c: Canyon, theta_p: float, nsub: int = 24):
    """Fraction of each segment lit by a beam whose direction towards the sun,
    in the cross section, is (sin theta_p, cos theta_p); theta_p > 0 means
    the sun is on the side of the right wall (x > 1)."""
    u, v = np.sin(theta_p), np.cos(theta_p)
    t = (np.arange(nsub) + 0.5) / nsub
    P = c.a[:, None, :] + (c.b - c.a)[:, None, :] * t[None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        x_hit = P[..., 0] + (c.h - P[..., 1]) * u / max(v, 1e-9)
    lit = (x_hit >= -1e-12) & (x_hit <= 1.0 + 1e-12)
    facing = (c.normal[:, 0] * u + c.normal[:, 1] * v) > 0
    return lit.mean(axis=1) * facing, facing


def responses(c: Canyon, alb_floor: float, alb_wall: float, theta_p_grid=None):
    """Escaping flux per unit street width for (i) unit diffuse sky irradiance
    and (ii) unit horizontal beam irradiance at each projected zenith angle."""
    rho = np.where(c.kind == 0, alb_floor, np.where(c.kind == 2, 0.0, alb_wall))
    surf = c.kind != 2
    Fs = c.F[np.ix_(surf, surf)]
    rs = rho[surf]
    M = np.linalg.solve(np.eye(surf.sum()) - rs[:, None] * Fs, np.diag(rs))   # J = M E0
    f_open = c.F[np.ix_(surf, c.kind == 2)].sum(axis=1)                        # F(segment -> opening)
    esc_w = (c.length[surf] * f_open) @ M                                       # escaping per unit E0
    diffuse = float(esc_w @ f_open)            # isotropic sky: E0 = D * F(segment -> opening)
    if theta_p_grid is None:
        theta_p_grid = np.radians(np.linspace(0.0, 89.5, 180))
    beam = np.empty(len(theta_p_grid))
    ns = c.normal[surf]
    for k, tp in enumerate(theta_p_grid):
        frac, _ = _sunlit(c, tp)
        # horizontal beam irradiance 1: the beam vector is (tan tp, 1) per unit horizontal irradiance
        cosi = ns[:, 0] * np.tan(tp) + ns[:, 1]
        beam[k] = float(esc_w @ (frac[surf] * np.clip(cosi, 0.0, None)))
    return diffuse, np.asarray(theta_p_grid), beam


def floor_sky_view(h):
    return np.sqrt(1.0 + h * h) - h


def solar_angles(index_utc, lat, lon, nsub=4, ending=True):
    """Zenith and azimuth (rad, azimuth clockwise from north) at ``nsub``
    sub-steps of each hour; and the extraterrestrial horizontal irradiance
    weights of the sub-steps."""
    from .solar import spencer
    idx = np.asarray(index_utc.values.astype("datetime64[s]").astype(np.int64))
    off = -1.0 if ending else 0.0
    frac = (np.arange(nsub) + 0.5) / nsub
    ts = idx[:, None] + ((off + frac) * 3600.0)[None, :]
    import pandas as pd
    tt = pd.to_datetime(ts.ravel(), unit="s", utc=True)
    doy = tt.dayofyear.to_numpy()
    hr = (tt.hour + tt.minute / 60.0 + tt.second / 3600.0).to_numpy()
    dec, eot, e0 = spencer(doy, hr)
    ha = np.radians((hr * 60.0 + eot + 4.0 * lon) / 4.0 - 180.0)
    phi = np.radians(lat)
    mu = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha)
    zen = np.arccos(np.clip(mu, -1, 1))
    az = np.arctan2(-np.sin(ha) * np.cos(dec), np.sin(dec) * np.cos(phi) - np.cos(dec) * np.sin(phi) * np.cos(ha))
    az = np.mod(az, 2 * np.pi)
    w = e0 * np.clip(mu, 0, None)
    shp = (len(idx), nsub)
    return zen.reshape(shp), az.reshape(shp), w.reshape(shp)


def absorbed_change(c: Canyon, a0: float, a1: float, alb_wall: float, theta_p_grid):
    """Change of the absorbed shortwave flux density on every surface segment
    (W m-2 per unit albedo change of the floor, per W m-2 of horizontal
    irradiance) when the floor albedo is raised from a0 to a1: for isotropic
    sky irradiance, and for unit horizontal beam irradiance at every
    projected zenith angle.  Negative on the floor (it absorbs less),
    positive on the walls (they intercept part of the additional reflection)."""
    surf = c.kind != 2
    Fs = c.F[np.ix_(surf, surf)]
    f_open = c.F[np.ix_(surf, c.kind == 2)].sum(axis=1)
    ns = c.normal[surf]
    kinds = c.kind[surf]
    out = []
    for af in (a0, a1):
        rho = np.where(kinds == 0, af, alb_wall)
        A = np.eye(surf.sum()) - rho[:, None] * Fs
        # irradiance: E = E0 + F J, J = rho E  ->  (I - F rho) E = E0
        B = np.eye(surf.sum()) - Fs * rho[None, :]
        E_dif = np.linalg.solve(B, f_open)
        E_beam = np.empty((len(theta_p_grid), surf.sum()))
        for k, tp in enumerate(theta_p_grid):
            frac, _ = _sunlit(c, tp)
            cosi = np.clip(ns[:, 0] * np.tan(tp) + ns[:, 1], 0.0, None)
            E_beam[k] = np.linalg.solve(B, frac[surf] * cosi)
        del A
        out.append(((1 - rho) * E_dif, (1 - rho)[None, :] * E_beam))
    d_dif = (out[1][0] - out[0][0]) / (a1 - a0)
    d_beam = (out[1][1] - out[0][1]) / (a1 - a0)
    return d_dif, d_beam


def site_inputs(key, t0="2017-05-01", t1="2024-05-01"):
    """Hourly kernel (snow-masked), global, diffuse and beam irradiance and
    sun position at 15-min sub-steps over the ledger's seven-year record."""
    import pandas as pd
    from . import driver, kernel, solar
    from .config import SITE
    df = driver.load_forcing(key)
    K, _, _, kt, _ = kernel.hourly_kernel(df)
    mask = driver.snow_mask_pavement(df, key, days_after=1.0)
    K = K * (1.0 - mask)
    loc = df.index + pd.Timedelta(hours=SITE[key].tz)
    keep = (loc >= t0) & (loc < t1)
    sw = df["sw"].to_numpy()
    D = solar.erbs_diffuse_fraction(kt) * sw
    zen, az, w = solar_angles(df.index, SITE[key].lat, SITE[key].lon, nsub=4)
    wsum = w.sum(1, keepdims=True)
    wn = np.where(wsum > 0, w / np.maximum(wsum, 1e-12), 0.25)
    return dict(K=K[keep], sw=sw[keep], D=D[keep], B=(sw - D)[keep], zen=zen[keep], az=az[keep], wn=wn[keep],
                month=loc[keep].month.to_numpy())


def weighted_segments(inp, d_dif, d_beam, tp_grid, axis_deg):
    """Kernel-weighted mean of a per-segment response (per unit global
    irradiance), e.g. the absorbed-flux change of ``absorbed_change``."""
    ok = inp["sw"] > 1.0
    wK = np.where(ok, inp["K"], 0.0)
    tp = np.arctan(np.tan(np.minimum(inp["zen"], np.radians(89.9)))
                   * np.abs(np.sin(inp["az"] - np.radians(axis_deg))))
    tp = np.minimum(tp, tp_grid[-1])
    # beam weight of every theta_p bin (linear interpolation weights)
    k = np.clip(np.searchsorted(tp_grid, tp) - 1, 0, len(tp_grid) - 2)
    fr = (tp - tp_grid[k]) / (tp_grid[k + 1] - tp_grid[k])
    up = inp["zen"] < np.radians(90.0)
    wb = (wK / np.maximum(inp["sw"], 1e-9) * inp["B"])[:, None] * inp["wn"] * up
    hist = np.zeros(len(tp_grid))
    np.add.at(hist, k.ravel(), (wb * (1 - fr)).ravel())
    np.add.at(hist, (k + 1).ravel(), (wb * fr).ravel())
    wd = np.sum(wK / np.maximum(inp["sw"], 1e-9) * inp["D"])
    return (wd * d_dif + hist @ d_beam) / np.sum(wK)

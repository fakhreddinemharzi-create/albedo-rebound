"""One-dimensional surface-energy-balance / heat-conduction column with
freezing and thawing.

Numerics
--------
* Finite volumes on a graded grid; the skin temperature T_s is an
  additional unknown closed by the surface energy balance
      (1 - a) S + eps L - eps sigma T_s^4 - h (T_s - T_a) - LE - G = 0,
  with G the conductive flux into the first cell.
* Each cell carries its volumetric enthalpy H (J m-3) as the primary
  variable, with T(H) and k(T) read from monotone material tables that
  contain the latent heat of the pore water through the soil freezing
  characteristic.  The fully implicit (backward Euler) step is solved by
  Newton iteration on (T_s, H_1..H_N); the Jacobian is tridiagonal.
  Solving for enthalpy keeps the latent heat exactly conserved even when
  the freezing interval is narrower than the temperature change of a step
  (the known failure of apparent-heat-capacity schemes).
* Turbulent exchange: bulk transfer between the surface and the screen
  height with Monin-Obukhov-type stability functions of the bulk
  Richardson number (Louis, 1979), plus a free-convection floor for a
  heated horizontal surface in calm air.
* Surfaces: 'pavement' (sealed; a 1 mm depression store that fills with
  rain and evaporates at the potential rate) or 'natural' (grass reference
  surface: Penman-Monteith-type resistance to evaporation with a
  soil-moisture bucket).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit

from . import materials as M
from .config import SIGMA, T0K, LV, CP_AIR

KARMAN = 0.41


# ----------------------------------------------------------------- numerics
@njit(cache=True)
def _tdma(a, b, c, d, n):
    """Solve a tridiagonal system (a: sub, b: diag, c: super) in place."""
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


@njit(cache=True)
def _t_of_h(Htab, Ttab, h):
    """Inverse of a monotone enthalpy table: T(H) and dT/dH."""
    n = Htab.shape[0]
    if h <= Htab[0]:
        d = (Ttab[1] - Ttab[0]) / (Htab[1] - Htab[0])
        return Ttab[0] + (h - Htab[0]) * d, d
    if h >= Htab[n - 1]:
        d = (Ttab[n - 1] - Ttab[n - 2]) / (Htab[n - 1] - Htab[n - 2])
        return Ttab[n - 1] + (h - Htab[n - 1]) * d, d
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if Htab[mid] <= h:
            lo = mid
        else:
            hi = mid
    d = (Ttab[hi] - Ttab[lo]) / (Htab[hi] - Htab[lo])
    return Ttab[lo] + (h - Htab[lo]) * d, d


@njit(cache=True)
def _interp_T(tab, Ttab, t):
    n = Ttab.shape[0]
    if t <= Ttab[0]:
        return tab[0]
    if t >= Ttab[n - 1]:
        return tab[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if Ttab[mid] <= t:
            lo = mid
        else:
            hi = mid
    w = (t - Ttab[lo]) / (Ttab[hi] - Ttab[lo])
    return tab[lo] * (1 - w) + tab[hi] * w


@njit(cache=True)
def _h_of_t(Htab, Ttab, t):
    return _interp_T(Htab, Ttab, t)


@njit(cache=True)
def _esat(tc):
    return 6.1078 * np.exp(17.27 * tc / (tc + 237.3))


@njit(cache=True)
def _qsat(tc, p):
    e = _esat(tc)
    return 0.622 * e / (p - 0.378 * e)


@njit(cache=True)
def _transfer(ts, ta, u, p, zr, z0m, z0h, hmult):
    """Sensible-heat transfer coefficient h (W m-2 K-1) and the aerodynamic
    conductance g_a = C_H U (m s-1)."""
    u = max(u, 0.3)
    rho = p * 100.0 / (287.05 * (ta + T0K))
    lm = np.log(zr / z0m)
    lh = np.log(zr / z0h)
    chn = KARMAN * KARMAN / (lm * lh)
    rib = 9.81 * zr * (ta - ts) / ((0.5 * (ta + ts) + T0K) * u * u)
    if rib < 0.0:
        c = 7.4 * 9.4 * chn * np.sqrt(zr / z0m)
        f = 1.0 - 9.4 * rib / (1.0 + c * np.sqrt(-rib))
    else:
        f = 1.0 / (1.0 + 4.7 * rib) ** 2
    ga = hmult * chn * u * f
    h = rho * CP_AIR * ga
    # free convection above a heated horizontal surface (turbulent regime)
    dT = ts - ta
    if dT > 0.0:
        hfree = 1.52 * dT ** (1.0 / 3.0)
        h = (h ** 3 + hfree ** 3) ** (1.0 / 3.0)
        ga = h / (rho * CP_AIR)
    return h, ga, rho


@njit(cache=True)
def run(dz, zc, mat, Ttab, Htabs, Ktabs,
        alb, eps, zr, z0m, z0h, hmult,
        sw, lw, ta, qa, ws, pr, p, snow,
        dt, nsub, surface, wcap, rs_day, rs_night, alb_snow,
        H0, ts0, w0, rec, ice_tabs, lay, nlay, zmax_frost, bot_dirichlet, t_bot):
    """Integrate the column over the forcing record.

    Forcing arrays have one value per sub-step (len = nhour * nsub).
    ``alb`` has one value per sub-step as well (time-varying albedo).
    Returns hourly-mean outputs and the final state.
    """
    n = dz.shape[0]
    nstep = sw.shape[0]
    nhour = nstep // nsub
    # distances between cell centres and face conductances
    dzc = np.empty(n)            # centre-to-centre distance to the next cell
    for i in range(n - 1):
        dzc[i] = zc[i + 1] - zc[i]
    dzc[n - 1] = 1.0
    H = H0.copy()
    Hold = H0.copy()
    T = np.empty(n)
    dTdH = np.empty(n)
    K = np.empty(n)
    ts = ts0
    wst = w0
    # outputs
    out_ts = np.zeros(nhour)
    out_ts_end = np.zeros(nhour)
    out_T = np.zeros((nhour, rec.shape[0]))
    out_ice = np.zeros((nhour, rec.shape[0]))
    out_flux = np.zeros((nhour, 5))       # Rn, H, LE, G, absorbed SW
    out_w = np.zeros(nhour)
    out_iter = np.zeros(nhour)
    out_frost = np.zeros(nhour)           # deepest depth with ice fraction > 5 %
    out_lice = np.zeros((nhour, nlay))    # layer-mean ice fraction
    lw_sum = np.zeros(nlay)
    for i in range(n):
        if lay[i] >= 0:
            lw_sum[lay[i]] += dz[i]
    a = np.empty(n + 1)
    b = np.empty(n + 1)
    c = np.empty(n + 1)
    r = np.empty(n + 1)
    nfail = 0
    for i in range(n):
        T[i], dTdH[i] = _t_of_h(Htabs[mat[i]], Ttab, H[i])
    for k in range(nstep):
        hr = k // nsub
        for i in range(n):
            Hold[i] = H[i]
        # surface properties for this step
        a_s = alb[k]
        if snow[k] > 0.5:
            a_s = alb_snow
        wet_frac = 0.0
        if surface == 0:
            wet_frac = min(1.0, wst / 0.2) if wst > 0.0 else 0.0
        # Newton iterations
        it = 0
        conv = False
        le = 0.0
        hsens = 0.0
        g_top = 0.0
        while it < 30:
            it += 1
            for i in range(n):
                T[i], dTdH[i] = _t_of_h(Htabs[mat[i]], Ttab, H[i])
                K[i] = _interp_T(Ktabs[mat[i]], Ttab, T[i])
            hc, ga, rho = _transfer(ts, ta[k], ws[k], p[k], zr, z0m, z0h, hmult)
            # latent heat flux and derivative
            qs = _qsat(ts, p[k])
            dq = qs - qa[k]
            dqs = qs * 17.27 * 237.3 / (ts + 237.3) ** 2
            if surface == 0:
                if dq > 0.0 and wet_frac > 0.0:
                    le = rho * LV * ga * dq * wet_frac
                    dle = rho * LV * ga * dqs * wet_frac
                else:
                    le = 0.0
                    dle = 0.0
            else:
                # grass reference surface: aerodynamic + canopy resistance
                day = sw[k] > 10.0
                rs = rs_day if day else rs_night
                beta = min(1.0, wst / (0.7 * wcap)) if wcap > 0 else 0.0
                if ts < 0.0 or snow[k] > 0.5:
                    beta = 0.0
                if beta > 1e-3 and dq > 0.0:
                    ra = 1.0 / ga
                    gs = 1.0 / (ra + rs / beta)
                    le = rho * LV * gs * dq
                    dle = rho * LV * gs * dqs
                else:
                    le = 0.0
                    dle = 0.0
            tsk = ts + T0K
            rn = (1.0 - a_s) * sw[k] + eps * lw[k] - eps * SIGMA * tsk ** 4
            hsens = hc * (ts - ta[k])
            k0 = K[0] / (0.5 * dz[0])
            g_top = k0 * (ts - T[0])
            # residuals
            r[0] = rn - hsens - le - g_top
            b[0] = -4.0 * eps * SIGMA * tsk ** 3 - hc - dle - k0
            c[0] = k0 * dTdH[0]
            a[0] = 0.0
            for i in range(n):
                kup = k0 if i == 0 else 2.0 * K[i - 1] * K[i] / (K[i - 1] + K[i]) / dzc[i - 1]
                if i < n - 1:
                    kdn = 2.0 * K[i] * K[i + 1] / (K[i] + K[i + 1]) / dzc[i]
                    tdn = T[i + 1]
                elif bot_dirichlet:
                    # ground anchored to the untreated equilibrium at the base
                    kdn = K[i] / (0.5 * dz[i])
                    tdn = t_bot[k]
                else:
                    kdn = 0.0
                    tdn = T[i]
                tup = ts if i == 0 else T[i - 1]
                fin = kup * (tup - T[i])          # heat entering from above
                fout = kdn * (T[i] - tdn)         # heat leaving downward
                j = i + 1
                r[j] = -(dz[i] * (H[i] - Hold[i]) / dt - fin + fout)
                b[j] = -(dz[i] / dt + (kup + kdn) * dTdH[i])
                if i == 0:
                    a[j] = kup                    # d/dTs of (-(-fin))
                else:
                    a[j] = kup * dTdH[i - 1]
                c[j] = kdn * dTdH[i + 1] if i < n - 1 else 0.0
            # convergence on the energy residuals (W m-2)
            rmax = 0.0
            for j in range(n + 1):
                if abs(r[j]) > rmax:
                    rmax = abs(r[j])
            if rmax < 1e-3 and it > 1:
                conv = True
                break
            # Newton step: J dx = -F, rows of J are those of F = [R_s, -R_i]
            for j in range(n + 1):
                r[j] = -r[j]
            dx = _tdma(a, b, c, r, n + 1)
            # damping of the skin-temperature update
            dts = dx[0]
            if dts > 5.0:
                dts = 5.0
            elif dts < -5.0:
                dts = -5.0
            ts += dts
            mx = abs(dts)
            for i in range(n):
                H[i] += dx[i + 1]
                v = abs(dx[i + 1] * dTdH[i])
                if v > mx:
                    mx = v
            if mx < 1e-7:
                conv = True
                break
        if not conv:
            nfail += 1
        for i in range(n):
            T[i], dTdH[i] = _t_of_h(Htabs[mat[i]], Ttab, H[i])
        # water store
        ev = le / LV * dt            # kg m-2 = mm
        wst += pr[k] / nsub          # hourly total spread over the sub-steps
        wst -= ev
        if wst < 0.0:
            wst = 0.0
        if wst > wcap:
            wst = wcap
        # accumulate hourly means
        out_ts[hr] += ts / nsub
        out_flux[hr, 0] += ((1.0 - a_s) * sw[k] + eps * lw[k] - eps * SIGMA * (ts + T0K) ** 4) / nsub
        out_flux[hr, 1] += hsens / nsub
        out_flux[hr, 2] += le / nsub
        out_flux[hr, 3] += g_top / nsub
        out_flux[hr, 4] += (1.0 - a_s) * sw[k] / nsub
        out_iter[hr] += it / nsub
        for q in range(rec.shape[0]):
            ci = rec[q]
            out_T[hr, q] += T[ci] / nsub
        if (k + 1) % nsub == 0:
            out_ts_end[hr] = ts
            out_w[hr] = wst
            for q in range(rec.shape[0]):
                ci = rec[q]
                out_ice[hr, q] = _interp_T(ice_tabs[mat[ci]], Ttab, T[ci])
            fd = 0.0
            for i in range(n):
                if zc[i] > zmax_frost:
                    break
                fi = _interp_T(ice_tabs[mat[i]], Ttab, T[i])
                if fi > 0.05:
                    fd = zc[i] + 0.5 * dz[i]
                if lay[i] >= 0:
                    out_lice[hr, lay[i]] += fi * dz[i] / lw_sum[lay[i]]
            out_frost[hr] = fd
    return (out_ts, out_ts_end, out_T, out_ice, out_flux, out_w, out_iter, out_frost, out_lice,
            H, ts, wst, nfail)


# ----------------------------------------------------------------- set-up
@dataclass
class Layer:
    material: M.Material
    top: float
    bottom: float
    theta_w: float = 0.0


@dataclass
class Structure:
    layers: list
    zbot: float = 10.0
    dz_top: float = 0.004
    growth: float = 1.18
    dz_max: float = 0.6
    dz_max_frost: float = 0.08      # resolution of the frost-susceptible zone
    z_frost: float = 3.2

    def grid(self):
        """Graded grid honouring every layer interface."""
        edges = [0.0]
        z = 0.0
        dz = self.dz_top
        interfaces = sorted({l.bottom for l in self.layers} | {self.zbot})
        while z < self.zbot - 1e-9:
            nxt = min(i for i in interfaces if i > z + 1e-9)
            cap = self.dz_max_frost if z < self.z_frost else self.dz_max
            step = min(dz, cap, nxt - z)
            # avoid slivers at an interface
            if nxt - (z + step) < 0.3 * step:
                step = nxt - z
            z += step
            edges.append(z)
            dz = min(dz * self.growth, self.dz_max_frost if z < self.z_frost else self.dz_max)
        edges = np.array(edges)
        dzs = np.diff(edges)
        zc = 0.5 * (edges[:-1] + edges[1:])
        mat = np.zeros(len(zc), np.int64)
        for i, zz in enumerate(zc):
            for j, l in enumerate(self.layers):
                if l.top <= zz < l.bottom or (j == len(self.layers) - 1 and zz >= l.top):
                    mat[i] = j
        return edges, dzs, zc, mat

    def tables(self):
        Ttab = M.temperature_grid()
        Hs, Ks, Is = [], [], []
        for l in self.layers:
            _, H, k, thu = M.enthalpy_table(l.material, l.theta_w, Ttab)
            Hs.append(H)
            Ks.append(k)
            Is.append((l.theta_w - thu) / max(l.theta_w, 1e-12) if l.theta_w > 0 else np.zeros_like(Ttab))
        return Ttab, np.array(Hs), np.array(Ks), np.array(Is)


def pavement_structure(h_ac=0.10, h_base=0.25, h_sub=0.30, S_base=0.45, S_sub=0.55, gwt=3.0, zbot=10.0,
                       k_ac=1.5, c_ac=2.1e6):
    """Reference low-volume street: asphalt concrete over crushed-stone base,
    granular subbase and a fine-grained subgrade in equilibrium with a water
    table at depth ``gwt``."""
    ac = M.Material("asphalt concrete", 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, bound=True, k_bound=k_ac, c_bound=c_ac)
    z1, z2, z3 = h_ac, h_ac + h_base, h_ac + h_base + h_sub
    # subgrade water content in equilibrium with the water table (1 m below
    # the formation level as the representative suction)
    th_sg = float(M.SUBGRADE.theta(-(gwt - (z3 + 0.5))))
    layers = [Layer(ac, 0.0, z1, 0.0),
              Layer(M.BASE, z1, z2, S_base * M.BASE.porosity),
              Layer(M.SUBBASE, z2, z3, S_sub * M.SUBBASE.porosity),
              Layer(M.SUBGRADE, z3, gwt, th_sg),
              Layer(M.SUBGRADE, gwt, zbot + 1.0, M.SUBGRADE.porosity)]
    return Structure(layers, zbot=zbot)


def natural_structure(theta_top=0.25, theta_deep=0.30, zbot=10.0):
    layers = [Layer(M.TOPSOIL, 0.0, 0.30, theta_top),
              Layer(M.TOPSOIL, 0.30, 1.0, theta_deep),
              Layer(M.SUBGRADE, 1.0, zbot + 1.0, 0.36)]
    return Structure(layers, zbot=zbot, dz_top=0.01)

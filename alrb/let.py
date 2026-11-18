"""Multilayer linear-elastic response of a pavement under circular loads
(Burmister's problem), solved in the Hankel-transform domain.

Formulation
-----------
Love's stress function phi = J0(k r) f(z) with, in layer i
(z_{i-1} <= z <= z_i),
    f = (A + B z1) exp(k z1) + (C + D z2) exp(-k z2),
    z1 = z - z_i <= 0,  z2 = z - z_{i-1} >= 0,
so that both exponentials are bounded by one inside the layer (stable for
any wavenumber).  With the scaled unknowns A' = k^3 A, B' = k^2 B,
C' = k^3 C, D' = k^2 D the transformed stresses are dimensionless:
    S_z (J0) = E1[-A' + (1-2v - k z1) B'] + E2[C' + (1-2v + k z2) D']
    T   (J1) = E1[ A' + (2v + k z1) B']   + E2[C' + (-2v + k z2) D']
    2G k W (J0) = E1[-A' + (2(1-2v) - k z1) B'] + E2[-C' + (-2(1-2v) - k z2) D']
    2G k U (J1) = E1[ A' + (1 + k z1) B']      + E2[-C' + (1 - k z2) D']
    P = (Lf)'   = 2 (B' E1 + D' E2)
with E1 = exp(k z1), E2 = exp(-k z2).  The last layer is a half-space
(A' = B' = 0).  Bonded interfaces carry continuity of S_z, T, W and U; the
surface is loaded by a unit J0 amplitude of vertical stress and is free of
shear.  A uniform pressure q over radius a has the amplitude -q a J1(k a).

The system (4n - 2 unknowns) is solved once per quadrature wavenumber and
shared by every evaluation point.  Hooke's law gives the strains.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from scipy.special import jn_zeros

try:
    from scipy.special import j0 as _j0, j1 as _j1
except ImportError:  # pragma: no cover
    pass


@njit(cache=True)
def _bessel_j0(x):
    # Numerical Recipes rational approximations (|error| < 1e-8)
    ax = abs(x)
    if ax < 8.0:
        y = x * x
        a1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))))
        a2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y * 1.0))))
        return a1 / a2
    z = 8.0 / ax
    y = z * z
    xx = ax - 0.785398164
    a1 = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)))
    a2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)))
    return np.sqrt(0.636619772 / ax) * (np.cos(xx) * a1 - z * np.sin(xx) * a2)


@njit(cache=True)
def _bessel_j1(x):
    ax = abs(x)
    if ax < 8.0:
        y = x * x
        a1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1 + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))))
        a2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))))
        return a1 / a2
    z = 8.0 / ax
    y = z * z
    xx = ax - 2.356194491
    a1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))))
    a2 = 0.04687499995 + y * (-0.2002690873e-3 + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)))
    ans = np.sqrt(0.636619772 / ax) * (np.cos(xx) * a1 - z * np.sin(xx) * a2)
    return ans if x > 0 else -ans


@njit(cache=True)
def _rows(k, z1, z2, nu):
    """Coefficient rows (A', B', C', D') of S_z, T, 2Gk W, 2Gk U, P."""
    e1 = np.exp(k * z1)
    e2 = np.exp(-k * z2)
    R = np.zeros((5, 4))
    R[0, 0] = -e1
    R[0, 1] = e1 * (1 - 2 * nu - k * z1)
    R[0, 2] = e2
    R[0, 3] = e2 * (1 - 2 * nu + k * z2)
    R[1, 0] = e1
    R[1, 1] = e1 * (2 * nu + k * z1)
    R[1, 2] = e2
    R[1, 3] = e2 * (-2 * nu + k * z2)
    R[2, 0] = -e1
    R[2, 1] = e1 * (2 * (1 - 2 * nu) - k * z1)
    R[2, 2] = -e2
    R[2, 3] = e2 * (-2 * (1 - 2 * nu) - k * z2)
    R[3, 0] = e1
    R[3, 1] = e1 * (1 + k * z1)
    R[3, 2] = -e2
    R[3, 3] = e2 * (1 - k * z2)
    R[4, 1] = 2 * e1
    R[4, 3] = 2 * e2
    return R


@njit(cache=True)
def solve_k(k, h, E, nu):
    """Coefficients (n x 4) for wavenumber k, unit surface J0 amplitude.
    h: thicknesses of the n-1 finite layers; E, nu: n values."""
    n = E.shape[0]
    zt = np.zeros(n)            # top of each layer
    for i in range(1, n):
        zt[i] = zt[i - 1] + h[i - 1]
    nun = 4 * n - 2
    M = np.zeros((nun, nun))
    rhs = np.zeros(nun)
    hs = 4 * (n - 1)            # first column of the half-space unknowns (C', D')
    # surface rows (layer 0 at z = 0)
    if n > 1:
        R = _rows(k, -h[0], 0.0, nu[0])
        for j in range(4):
            M[0, j] = R[0, j]
            M[1, j] = R[1, j]
    else:
        R = _rows(k, 0.0, 0.0, nu[0])
        for j in range(2, 4):
            M[0, hs + j - 2] = R[0, j]
            M[1, hs + j - 2] = R[1, j]
    rhs[0] = 1.0
    row = 2
    for i in range(n - 1):
        # interface between layer i (bottom: z1 = 0, z2 = h_i) and layer i+1 (top)
        Ra = _rows(k, 0.0, h[i], nu[i])
        last = i + 1 == n - 1
        if not last:
            Rb = _rows(k, -h[i + 1], 0.0, nu[i + 1])
        else:
            Rb = _rows(k, 0.0, 0.0, nu[i + 1])
        G_a = E[i] / (2 * (1 + nu[i]))
        G_b = E[i + 1] / (2 * (1 + nu[i + 1]))
        for q in range(4):
            sa = 1.0 if q < 2 else 1.0 / G_a
            sb = 1.0 if q < 2 else 1.0 / G_b
            for j in range(4):
                M[row, 4 * i + j] += sa * Ra[q, j]
                if not last:
                    M[row, 4 * (i + 1) + j] -= sb * Rb[q, j]
                elif j >= 2:
                    M[row, hs + j - 2] -= sb * Rb[q, j]
            row += 1
    x = np.linalg.solve(M, rhs)
    coef = np.zeros((n, 4))
    for i in range(n - 1):
        for j in range(4):
            coef[i, j] = x[4 * i + j]
    coef[n - 1, 2] = x[4 * (n - 1)]
    coef[n - 1, 3] = x[4 * (n - 1) + 1]
    return coef, zt


def kgrid(a, zmin, nint_min=60, npts=12, extra=40.0):
    """Gauss-Legendre nodes between consecutive zeros of J1(k a), up to a
    wavenumber at which exp(-k zmin) has decayed below exp(-extra)."""
    kmax = extra / max(zmin, 1e-3)
    nz = max(nint_min, int(kmax * a / np.pi) + 2)
    zeros = np.concatenate([[0.0], jn_zeros(1, nz) / a])
    xg, wg = np.polynomial.legendre.leggauss(npts)
    ks, ws = [], []
    for lo, hi in zip(zeros[:-1], zeros[1:]):
        ks.append(0.5 * (hi - lo) * xg + 0.5 * (hi + lo))
        ws.append(0.5 * (hi - lo) * wg)
    return np.concatenate(ks), np.concatenate(ws)


@njit(cache=True)
def response(ks, ws, h, E, nu, q, a, rpts, zpts, lay):
    """Stresses, strains and vertical displacement at (r, z) points.

    Returns array (npt, 8): sz, sr, st, trz, ez, er, et, w  (tension +)."""
    npt = rpts.shape[0]
    n = E.shape[0]
    out = np.zeros((npt, 8))
    for m in range(ks.shape[0]):
        k = ks[m]
        coef, zt = solve_k(k, h, E, nu)
        load = -q * a * _bessel_j1(k * a) * ws[m]
        for p in range(npt):
            i = lay[p]
            z = zpts[p]
            if i < n - 1:
                z1 = z - (zt[i] + h[i])
            else:
                z1 = 0.0
            z2 = z - zt[i]
            R = _rows(k, z1, z2, nu[i])
            S = 0.0
            T = 0.0
            W = 0.0
            U = 0.0
            P = 0.0
            for j in range(4):
                c = coef[i, j]
                S += R[0, j] * c
                T += R[1, j] * c
                W += R[2, j] * c
                U += R[3, j] * c
                P += R[4, j] * c
            G = E[i] / (2 * (1 + nu[i]))
            W = W / (2 * G * k)
            kQ = U                      # k * (2 G U) / ... : kQ = [..] dimensionless
            r = rpts[p]
            j0 = _bessel_j0(k * r)
            j1 = _bessel_j1(k * r)
            if r > 1e-12:
                j1r = j1 / r
            else:
                j1r = 0.5 * k
            # sigma_r = nu J0 P + (k J0 - J1/r) Q,  sigma_t = nu J0 P + (J1/r) Q, with k Q = kQ
            Q = kQ / k
            out[p, 0] += load * S * j0
            out[p, 1] += load * (nu[i] * j0 * P + (k * j0 - j1r) * Q)
            out[p, 2] += load * (nu[i] * j0 * P + j1r * Q)
            out[p, 3] += load * T * j1
            out[p, 7] += load * W * j0
    for p in range(npt):
        i = lay[p]
        Ei = E[i]
        v = nu[i]
        sz, sr, st = out[p, 0], out[p, 1], out[p, 2]
        out[p, 4] = (sz - v * (sr + st)) / Ei
        out[p, 5] = (sr - v * (st + sz)) / Ei
        out[p, 6] = (st - v * (sr + sz)) / Ei
    return out


def layer_of(z, h):
    zt = np.concatenate([[0.0], np.cumsum(h)])
    for i in range(len(h)):
        if zt[i] <= z < zt[i + 1]:
            return i
    return len(h)

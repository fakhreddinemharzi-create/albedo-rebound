"""Critical pavement strains under the standard dual-wheel axle, tabulated
on the modulus ratios of the five-layer structure.

Because the elastic problem is linear, multiplying every modulus by c
divides every strain by c; the strains therefore depend on the subgrade
modulus E5 only through the factor q/E5 and otherwise on the four ratios
    x1 = log10(E1/E5), x2 = log10(E2/E1), x3 = log10(E3/E5), x4 = log10(E4/E5)
(E1, E2: upper and lower halves of the bound layer; E3 base; E4 subbase).
The dual-wheel strains are superposed from single-wheel responses at the
radial offsets 0, s/2 and s along the axle line.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from . import let

Q = 690e3                 # tyre pressure (Pa), 100 psi
LOAD = 20.0e3             # load per tyre (N): 80 kN single axle with dual tyres
A = np.sqrt(LOAD / (np.pi * Q))
S = 0.343                 # dual spacing (m), 13.5 in
H = np.array([0.05, 0.05, 0.25, 0.30])
NU = np.array([0.35, 0.35, 0.35, 0.35, 0.45])
X1 = np.round(np.arange(-1.4, 3.0001, 0.2), 4)
X2 = np.round(np.arange(-0.8, 0.8001, 0.2), 4)
X3 = np.round(np.arange(-1.75, 2.2501, 0.25), 4)
X4 = np.round(np.arange(-1.75, 2.2501, 0.25), 4)
EPS_Z = 1e-7


def _points():
    zs = [0.025, 0.075, H[:2].sum() - EPS_Z, H.sum() + EPS_Z]
    lays = [0, 1, 1, 4]
    rs = [0.0, S / 2, S]
    R, Z, L = [], [], []
    for z, l in zip(zs, lays):
        for r in rs:
            R.append(r)
            Z.append(z)
            L.append(l)
    return np.array(R), np.array(Z), np.array(L)


_R, _Z, _L = _points()
_KS, _WS = let.kgrid(A, 0.025)


def critical(E):
    """Critical strains (tension +) for the moduli E (5 values, any unit).
    Returns (eps_t bottom of bound layer, eps_v top of subgrade,
             eps_r at 25 mm, eps_r at 75 mm) as positive magnitudes
    (eps_t may be negative if the bottom of the bound layer is in compression)."""
    out = let.response(_KS, _WS, H, np.asarray(E, float), NU, Q, A, _R, _Z, _L)
    er, ez, et = out[:, 5], out[:, 4], out[:, 6]
    # indices: 4 depths x 3 offsets (0, s/2, s)
    def blk(d):
        return slice(3 * d, 3 * d + 3)
    # bottom of the bound layer (depth index 2)
    b = blk(2)
    er_b, et_b = er[b], et[b]
    xx0 = er_b[0] + er_b[2]
    yy0 = et_b[0] + et_b[2]
    xx1 = 2 * er_b[1]
    yy1 = 2 * et_b[1]
    eps_t = max(xx0, yy0, xx1, yy1)
    b = blk(3)
    ez_s = ez[b]
    eps_v = max(-(ez_s[0] + ez_s[2]), -2 * ez_s[1])
    b = blk(0)
    e1 = max(-(ez[b][0] + ez[b][2]), -2 * ez[b][1])
    b = blk(1)
    e2 = max(-(ez[b][0] + ez[b][2]), -2 * ez[b][1])
    return np.array([eps_t, eps_v, e1, e2])


def build(n_jobs=4):
    """Tabulate critical strains for E5 = 1 (strain units of q/E5 per Pa ->
    here E5 = 1 Pa, so the stored numbers must be divided by E5 in Pa)."""
    from concurrent.futures import ProcessPoolExecutor
    grid = [(i, j, k, l) for i in range(len(X1)) for j in range(len(X2)) for k in range(len(X3))
            for l in range(len(X4))]
    tab = np.zeros((len(X1), len(X2), len(X3), len(X4), 4))
    chunks = np.array_split(np.arange(len(grid)), n_jobs * 8)
    with ProcessPoolExecutor(n_jobs) as ex:
        for ids, vals in ex.map(_work, [(c, grid) for c in chunks]):
            for q, v in zip(ids, vals):
                i, j, k, l = grid[q]
                tab[i, j, k, l] = v
    return tab


def _work(args):
    ids, grid = args
    vals = []
    for q in ids:
        i, j, k, l = grid[q]
        e1 = 10 ** X1[i]
        E = np.array([e1, e1 * 10 ** X2[j], 10 ** X3[k], 10 ** X4[l], 1.0])
        vals.append(critical(E))
    return ids, vals


@njit(cache=True)
def _loc(x, g):
    n = g.shape[0]
    if x <= g[0]:
        return 0, 0.0
    if x >= g[n - 1]:
        return n - 2, 1.0
    i = int((x - g[0]) / (g[1] - g[0]))
    if i > n - 2:
        i = n - 2
    return i, (x - g[i]) / (g[i + 1] - g[i])


@njit(cache=True)
def interp(tab, x1g, x2g, x3g, x4g, E1, E2, E3, E4, E5):
    """Quadrilinear interpolation of the table for hourly moduli (MPa).
    Returns (n, 4) strains."""
    n = E1.shape[0]
    out = np.zeros((n, 4))
    for p in range(n):
        x1 = np.log10(E1[p] / E5[p])
        x2 = np.log10(E2[p] / E1[p])
        x3 = np.log10(E3[p] / E5[p])
        x4 = np.log10(E4[p] / E5[p])
        i, a = _loc(x1, x1g)
        j, b = _loc(x2, x2g)
        k, c = _loc(x3, x3g)
        l, d = _loc(x4, x4g)
        for q in range(4):
            acc = 0.0
            for di in range(2):
                wi = a if di else 1 - a
                for dj in range(2):
                    wj = b if dj else 1 - b
                    for dk in range(2):
                        wk = c if dk else 1 - c
                        for dl in range(2):
                            wl = d if dl else 1 - d
                            acc += wi * wj * wk * wl * tab[i + di, j + dj, k + dk, l + dl, q]
            out[p, q] = acc / (E5[p] * 1e6)
    return out


@njit(cache=True)
def _cr_w(t):
    """Catmull-Rom (Keys, 1981, a = -0.5) weights for offsets -1, 0, 1, 2."""
    t2 = t * t
    t3 = t2 * t
    return (-0.5 * t3 + t2 - 0.5 * t,
            1.5 * t3 - 2.5 * t2 + 1.0,
            -1.5 * t3 + 2.0 * t2 + 0.5 * t,
            0.5 * t3 - 0.5 * t2)


@njit(cache=True)
def interp_cubic(tab, x1g, x2g, x3g, x4g, E1, E2, E3, E4, E5):
    """Four-dimensional cubic-convolution interpolation (clamped at the
    table edges).  Returns (n, 4) strains."""
    n = E1.shape[0]
    out = np.zeros((n, 4))
    n1, n2, n3, n4 = x1g.shape[0], x2g.shape[0], x3g.shape[0], x4g.shape[0]
    for p in range(n):
        xs = (np.log10(E1[p] / E5[p]), np.log10(E2[p] / E1[p]), np.log10(E3[p] / E5[p]), np.log10(E4[p] / E5[p]))
        i, a = _loc(xs[0], x1g)
        j, b = _loc(xs[1], x2g)
        k, c = _loc(xs[2], x3g)
        l, d = _loc(xs[3], x4g)
        wa = _cr_w(a)
        wb = _cr_w(b)
        wc = _cr_w(c)
        wd = _cr_w(d)
        for q in range(4):
            acc = 0.0
            for di in range(4):
                ii = min(max(i - 1 + di, 0), n1 - 1)
                for dj in range(4):
                    jj = min(max(j - 1 + dj, 0), n2 - 1)
                    w2 = wa[di] * wb[dj]
                    for dk in range(4):
                        kk = min(max(k - 1 + dk, 0), n3 - 1)
                        w3 = w2 * wc[dk]
                        for dl in range(4):
                            ll = min(max(l - 1 + dl, 0), n4 - 1)
                            acc += w3 * wd[dl] * tab[ii, jj, kk, ll, q]
            out[p, q] = acc / (E5[p] * 1e6)
    return out

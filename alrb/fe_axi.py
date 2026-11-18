"""Independent check of the layered elastic solution: axisymmetric
finite elements (8-node serendipity quadrilaterals, 3 x 3 Gauss
integration) on a graded mesh, rigid base and radial roller at the far
boundary.  Used only for verification (Appendix B)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_XI = np.array([-1, 1, 1, -1, 0, 1, 0, -1], float)
_ETA = np.array([-1, -1, 1, 1, -1, 0, 1, 0], float)
_G, _W = np.polynomial.legendre.leggauss(3)


def _shape(xi, eta):
    N = np.zeros(8)
    dN = np.zeros((2, 8))
    for i in range(4):
        a, b = _XI[i], _ETA[i]
        N[i] = 0.25 * (1 + a * xi) * (1 + b * eta) * (a * xi + b * eta - 1)
        dN[0, i] = 0.25 * a * (1 + b * eta) * (2 * a * xi + b * eta)
        dN[1, i] = 0.25 * b * (1 + a * xi) * (a * xi + 2 * b * eta)
    for i in range(4, 8):
        a, b = _XI[i], _ETA[i]
        if a == 0:
            N[i] = 0.5 * (1 - xi ** 2) * (1 + b * eta)
            dN[0, i] = -xi * (1 + b * eta)
            dN[1, i] = 0.5 * b * (1 - xi ** 2)
        else:
            N[i] = 0.5 * (1 + a * xi) * (1 - eta ** 2)
            dN[0, i] = 0.5 * a * (1 - eta ** 2)
            dN[1, i] = -eta * (1 + a * xi)
    return N, dN


def graded(x_fine, h_fine, x_end, growth=1.12, breaks=()):
    """1-D node coordinates: spacing h_fine up to x_fine, then geometric growth;
    ``breaks`` are coordinates that must coincide with element edges."""
    xs = [0.0]
    h = h_fine
    targets = sorted(set(list(breaks) + [x_end]))
    while xs[-1] < x_end - 1e-12:
        x = xs[-1]
        step = h_fine if x < x_fine else h
        nxt = min(t for t in targets if t > x + 1e-12)
        if x + step > nxt - 0.25 * step:
            step = nxt - x
        xs.append(x + step)
        if x >= x_fine:
            h *= growth
    return np.array(xs)


def solve(h, E, nu, q, a, R=30.0, Z=30.0, hf=0.004, rf=0.6, zf=1.0, growth=1.12):
    zb = np.concatenate([[0.0], np.cumsum(h)])
    rs = graded(rf, hf, R, growth, breaks=(a,))
    zs = graded(zf, hf, Z, growth, breaks=tuple(zb[1:]))
    ner, nez = len(rs) - 1, len(zs) - 1
    # node numbering on the (2 ner + 1) x (2 nez + 1) lattice, skipping element centres
    nr2, nz2 = 2 * ner + 1, 2 * nez + 1
    rr = np.empty(nr2)
    rr[0::2] = rs
    rr[1::2] = 0.5 * (rs[:-1] + rs[1:])
    zz = np.empty(nz2)
    zz[0::2] = zs
    zz[1::2] = 0.5 * (zs[:-1] + zs[1:])
    idx = -np.ones((nz2, nr2), np.int64)
    cnt = 0
    for j in range(nz2):
        for i in range(nr2):
            if (i % 2 == 1) and (j % 2 == 1):
                continue
            idx[j, i] = cnt
            cnt += 1
    ndof = 2 * cnt
    rows, cols, vals = [], [], []
    F = np.zeros(ndof)
    zmid = 0.5 * (zs[:-1] + zs[1:])
    lay = np.searchsorted(zb[1:], zmid, side="right")
    lay = np.minimum(lay, len(E) - 1)
    Dmats = []
    for e_, v_ in zip(E, nu):
        c = e_ / ((1 + v_) * (1 - 2 * v_))
        Dmats.append(c * np.array([[1 - v_, v_, v_, 0], [v_, 1 - v_, v_, 0], [v_, v_, 1 - v_, 0],
                                   [0, 0, 0, (1 - 2 * v_) / 2]]))
    pre = [[_shape(x, y) for y in _G] for x in _G]
    elems = []
    for ej in range(nez):
        for ei in range(ner):
            i0, j0 = 2 * ei, 2 * ej
            nodes = [idx[j0, i0], idx[j0, i0 + 2], idx[j0 + 2, i0 + 2], idx[j0 + 2, i0],
                     idx[j0, i0 + 1], idx[j0 + 1, i0 + 2], idx[j0 + 2, i0 + 1], idx[j0 + 1, i0]]
            xy = np.array([[rr[i0], zz[j0]], [rr[i0 + 2], zz[j0]], [rr[i0 + 2], zz[j0 + 2]], [rr[i0], zz[j0 + 2]],
                           [rr[i0 + 1], zz[j0]], [rr[i0 + 2], zz[j0 + 1]], [rr[i0 + 1], zz[j0 + 2]],
                           [rr[i0], zz[j0 + 1]]])
            D = Dmats[lay[ej]]
            Ke = np.zeros((16, 16))
            for gi in range(3):
                for gj in range(3):
                    N, dN = pre[gi][gj]
                    J = dN @ xy
                    detJ = np.linalg.det(J)
                    dNx = np.linalg.solve(J, dN)
                    r = N @ xy[:, 0]
                    B = np.zeros((4, 16))
                    B[0, 0::2] = dNx[0]
                    B[1, 1::2] = dNx[1]
                    B[2, 0::2] = N / max(r, 1e-12)
                    B[3, 0::2] = dNx[1]
                    B[3, 1::2] = dNx[0]
                    Ke += B.T @ D @ B * 2 * np.pi * r * detJ * _W[gi] * _W[gj]
            dofs = np.empty(16, np.int64)
            dofs[0::2] = 2 * np.array(nodes)
            dofs[1::2] = 2 * np.array(nodes) + 1
            rows.append(np.repeat(dofs, 16))
            cols.append(np.tile(dofs, 16))
            vals.append(Ke.ravel())
            elems.append((nodes, xy, lay[ej]))
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ndof, ndof))
    # surface load: consistent nodal forces of a uniform pressure q (downward = +z) on r < a
    for ei in range(ner):
        r0, r1 = rs[ei], rs[ei + 1]
        if r0 >= a - 1e-12:
            break
        nodes = [idx[0, 2 * ei], idx[0, 2 * ei + 2], idx[0, 2 * ei + 1]]
        for gx, gw in zip(_G, _W):
            s = gx
            Nl = np.array([0.5 * s * (s - 1), 0.5 * s * (s + 1), 1 - s * s])
            r = 0.5 * (r0 + r1) + 0.5 * (r1 - r0) * s
            for nd, nv in zip(nodes, Nl):
                F[2 * nd + 1] += q * nv * 2 * np.pi * r * gw * 0.5 * (r1 - r0)
    fixed = set()
    for i in range(nr2):              # rigid base
        if idx[-1, i] >= 0:
            fixed.update({2 * idx[-1, i], 2 * idx[-1, i] + 1})
    for j in range(nz2):              # axis and far boundary: no radial displacement
        for i in (0, nr2 - 1):
            if idx[j, i] >= 0:
                fixed.add(2 * idx[j, i])
    fixed = np.array(sorted(fixed))
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), F[free])
    return dict(u=u, idx=idx, rr=rr, zz=zz, elems=elems, Dmats=Dmats, E=E, nu=nu, ndof=ndof)


def strain_at(sol, r, z, layer_hint=None):
    """Strains (tension +, e_r, e_z, e_theta) at (r, z), evaluated inside the
    element that contains the point (one-sided at interfaces: the element
    whose layer is ``layer_hint``)."""
    best = None
    for nodes, xy, ly in sol["elems"]:
        if layer_hint is not None and ly != layer_hint:
            continue
        rmin, rmax = xy[:, 0].min(), xy[:, 0].max()
        zmin, zmax = xy[:, 1].min(), xy[:, 1].max()
        if rmin - 1e-12 <= r <= rmax + 1e-12 and zmin - 1e-12 <= z <= zmax + 1e-12:
            best = (nodes, xy)
            break
    nodes, xy = best
    xi = 2 * (r - xy[0, 0]) / (xy[1, 0] - xy[0, 0]) - 1
    eta = 2 * (z - xy[0, 1]) / (xy[3, 1] - xy[0, 1]) - 1
    N, dN = _shape(xi, eta)
    J = dN @ xy
    dNx = np.linalg.solve(J, dN)
    ue = np.array([sol["u"][2 * n] for n in nodes])
    we = np.array([sol["u"][2 * n + 1] for n in nodes])
    rr = N @ xy[:, 0]
    er = dNx[0] @ ue
    ez = dNx[1] @ we
    et = (N @ ue) / rr if rr > 1e-9 else er
    return er, ez, et, N @ we


def deflection_centre(sol):
    return sol["u"][2 * sol["idx"][0, 0] + 1]

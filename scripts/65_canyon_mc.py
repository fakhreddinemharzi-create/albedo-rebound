"""Independent photon Monte Carlo of the infinite canyon (Sect. 5.7).

Photons enter through the opening, either from an isotropic sky (positions
uniform across the opening, directions cosine-weighted) or as a beam in the
cross section; at the floor and walls they are absorbed with probability
1 - albedo or reflected diffusely.  The fraction that leaves through the
opening is compared with the radiosity solution of alrb.canyon for the
configurations of the path-tracing check.  Results are added to
results/canyon_v3.json under "photon_mc".
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import canyon  # noqa: E402
from alrb.config import RES  # noqa: E402

A0, A1, AW = 0.14, 0.45, 0.30
N = 4_000_000


def cosdir(n, rng, normal):
    u1, u2 = rng.random(n), rng.random(n)
    r, phi = np.sqrt(u1), 2 * np.pi * u2
    lx, ly, lz = r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)
    nx, nz = normal
    tx, tz = nz, -nx
    return lx * tx + lz * nx, ly, lx * tz + lz * nz


def escape(h, af, aw, theta_p=None, n=N, seed=1):
    rng = np.random.default_rng(seed)
    x, z = rng.random(n), np.full(n, float(h))
    if theta_p is None:
        dx, _, dz = cosdir(n, rng, (0.0, -1.0))
    else:
        th = np.radians(theta_p)
        dx, dz = np.full(n, -np.sin(th)), np.full(n, -np.cos(th))
    alive = np.ones(n, bool)
    esc = 0
    normals = {0: (1.0, 0.0), 1: (-1.0, 0.0), 2: (0.0, 1.0)}
    for _ in range(400):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        X, Z, DX, DZ = x[idx], z[idx], dx[idx], dz[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.vstack([np.where(DX < 0, -X / DX, np.inf), np.where(DX > 0, (1 - X) / DX, np.inf),
                           np.where(DZ < 0, -Z / DZ, np.inf), np.where(DZ > 0, (h - Z) / DZ, np.inf)])
        k = np.argmin(T, axis=0)
        t = T[k, np.arange(len(idx))]
        Xn, Zn = X + t * DX, Z + t * DZ
        out = k == 3
        esc += int(out.sum())
        rho = np.where(k == 2, af, aw)
        refl = ~out & (rng.random(len(idx)) < rho)
        alive[idx[~refl]] = False
        r = np.flatnonzero(refl)
        for kk, nn in normals.items():
            sel = r[k[r] == kk]
            if not len(sel):
                continue
            ddx, _, ddz = cosdir(len(sel), rng, nn)
            g = idx[sel]
            x[g], z[g] = np.clip(Xn[sel], 0.0, 1.0), np.clip(Zn[sel], 0.0, h)
            dx[g], dz[g] = ddx, ddz
    return esc / n


if __name__ == "__main__":
    p = RES / "canyon_v3.json"
    res = json.loads(p.read_text())
    out = []
    for h in (0.5, 1.0, 2.0):
        c = canyon.build(h, 80)
        for tp in (None, 0.0, 30.0, 55.0):
            grid = np.radians([0.0 if tp is None else tp])
            e = [escape(h, a, AW, tp, seed=11 + i) for i, a in enumerate((A0, A1))]
            rad = []
            for a in (A0, A1):
                d, _, b = canyon.responses(c, a, AW, grid)
                rad.append(d if tp is None else float(b[0]))
            rec = dict(h=h, theta_p=tp, mc_esc=e, rad_esc=rad, mc_psi=(e[1] - e[0]) / (A1 - A0),
                       rad_psi=(rad[1] - rad[0]) / (A1 - A0))
            out.append(rec)
            print(f"MC h={h} tp={tp}: esc {e[0]:.4f}/{e[1]:.4f} vs {rad[0]:.4f}/{rad[1]:.4f}; psi {rec['mc_psi']:.4f} "
                  f"vs {rec['rad_psi']:.4f}", flush=True)
    res["photon_mc"] = out
    p.write_text(json.dumps(res, indent=1))

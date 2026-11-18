"""Verification of the structural chain.

1. Layered elastic solution for a homogeneous half-space against the
   closed-form axis solution for a uniform circular load (Ahlvin and Ulery,
   1962; Huang, 2004): sigma_z, sigma_r and the deflection with depth.
2. Five-layer structures (ordinary, frozen foundation, hot bound layer)
   against an independent axisymmetric finite-element model (8-node
   quadrilaterals).
3. Critical-strain table (4-D cubic interpolation) against direct
   layered-elastic solutions on 1500 trafficked hours per site, expressed as
   the ratio of fatigue and subgrade damage and the error of the rutting
   strains.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import fe_axi, let, strain_table as stt, structural  # noqa: E402
from alrb.config import RES, SITES  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "v3"


def boussinesq():
    q, a, E, nu = 690e3, stt.A, 100e6, 0.35
    h = np.array([0.1, 0.25, 0.3])
    z = np.concatenate([np.linspace(0.002, 0.05, 6), np.linspace(0.075, 1.2, 20)])
    lay = np.array([let.layer_of(zz, h) for zz in z])
    ks, ws = let.kgrid(a, z.min())
    out = let.response(ks, ws, h, np.full(4, E), np.full(4, nu), q, a, np.zeros_like(z), z, lay)
    R = np.sqrt(a ** 2 + z ** 2)
    sz = -q * (1 - z ** 3 / R ** 3)
    sr = -q / 2 * ((1 + 2 * nu) - 2 * (1 + nu) * z / R + (z / R) ** 3)
    w = q * a * (1 + nu) / E * (a / R + (1 - 2 * nu) * (R - z) / a)
    return dict(z=z.tolist(), sz_let=out[:, 0].tolist(), sz_exact=sz.tolist(), sr_let=out[:, 1].tolist(),
                sr_exact=sr.tolist(), w_let=out[:, 7].tolist(), w_exact=w.tolist(),
                max_rel_err_sz=float(np.max(np.abs(out[:, 0] - sz) / q)),
                max_rel_err_sr=float(np.max(np.abs(out[:, 1] - sr) / q)),
                max_rel_err_w=float(np.max(np.abs(out[:, 7] - w) / w.max())))


def fe_cases():
    q, a = 690e3, stt.A
    h = stt.H
    nu = stt.NU
    cases = {"ordinary": [6000, 4000, 250, 150, 70], "frozen foundation": [20000, 18000, 6895, 6895, 150],
             "hot bound layer": [600, 900, 250, 150, 70]}
    pts = [("eps_r AC top", 0.0, 0.025, 0), ("eps_z AC top", 0.0, 0.025, 0), ("eps_r AC bottom", 0.0, 0.1 - 1e-7, 1),
           ("eps_z AC bottom", 0.0, 0.1 - 1e-7, 1), ("eps_r subgrade", 0.0, 0.65 + 1e-7, 4),
           ("eps_z subgrade", 0.0, 0.65 + 1e-7, 4), ("eps_t AC bottom r=s/2", stt.S / 2, 0.1 - 1e-7, 1),
           ("eps_z subgrade r=s/2", stt.S / 2, 0.65 + 1e-7, 4)]
    out = {}
    for name, E in cases.items():
        E = np.array(E, float) * 1e6
        sol = fe_axi.solve(h, E, nu, q, a, hf=0.005)
        r = np.array([p[1] for p in pts])
        z = np.array([p[2] for p in pts])
        lay = np.array([p[3] for p in pts])
        ks, ws = let.kgrid(a, 0.02)
        o = let.response(ks, ws, h, E, nu, q, a, r, z, lay)
        rows = []
        for (lab, rr, zz, ll), oo in zip(pts, o):
            er, ez, et, _ = fe_axi.strain_at(sol, rr, zz, ll)
            comp = "z" if "eps_z" in lab else ("t" if "eps_t" in lab else "r")
            v_let = {"r": oo[5], "z": oo[4], "t": oo[6]}[comp]
            v_fe = {"r": er, "z": ez, "t": et}[comp]
            rows.append(dict(label=lab, let=float(v_let), fe=float(v_fe)))
        out[name] = dict(E_MPa=(E / 1e6).tolist(), points=rows, ndof=int(sol["ndof"]))
        print(name, [(r_["label"], round(r_["let"] * 1e6, 2), round(r_["fe"] * 1e6, 2)) for r_ in rows], flush=True)
    return out


def table_accuracy():
    res = {}
    for s in SITES:
        st, w, _ = structural.run_site(s.key, 0.14, TAG)
        rng = np.random.default_rng(0)
        ii = rng.choice(np.where(w > 0)[0], 1500, replace=False)
        dr = np.array([stt.critical(st.E[i] * 1e6) for i in ii])
        tb = st.strains[ii]
        Ew = st.E[ii, 1] * 1e6 / 6894.757
        fat = lambda et: np.where(et > 0, w[ii] / (0.0795 * np.maximum(et, 1e-12) ** -3.291 * Ew ** -0.854), 0)
        sg = lambda ev: w[ii] / (1.365e-9 * ev ** -4.477)
        res[s.key] = dict(fatigue_ratio=float(fat(tb[:, 0]).sum() / fat(dr[:, 0]).sum()),
                          subgrade_ratio=float(sg(tb[:, 1]).sum() / sg(dr[:, 1]).sum()),
                          rut25_err=float(np.sum(np.abs(tb[:, 2] - dr[:, 2])) / np.sum(np.abs(dr[:, 2]))),
                          rut75_err=float(np.sum(np.abs(tb[:, 3] - dr[:, 3])) / np.sum(np.abs(dr[:, 3]))))
        print(s.key, {k: round(v, 4) for k, v in res[s.key].items()}, flush=True)
    return res


if __name__ == "__main__":
    b = boussinesq()
    print({k: v for k, v in b.items() if k.startswith("max")})
    t = table_accuracy()
    f = fe_cases()
    (RES / "verification_mechanics.json").write_text(json.dumps(dict(boussinesq=b, fe=f, table=t), indent=1))

"""Service life against surface albedo at every site.

Reference criterion: the first of bottom-up fatigue and subgrade rutting
(Asphalt Institute transfer functions, Miner's rule), with the daily
traffic set at each site so that the untreated pavement (albedo 0.14)
lasts exactly 20 years.  Scenario: a rutting-governed street, with the
traffic set so that the untreated bound layer reaches 6.35 mm of permanent
deformation (MEPDG model) in 20 years.

usage: python 40_structural.py [tag]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import structural  # noqa: E402
from alrb.config import ALBEDO_GRID, RES, SITES  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "v3"
L_DESIGN = 20.0
A0 = 0.14


def one(site):
    states = {}
    for a in ALBEDO_GRID:
        st, w, lv = structural.run_site(site.key, a, TAG, n_day=1.0)
        states[a] = (st, w, lv)
    # reference calibration: Miner lives scale exactly as 1/N
    lf0, ld0 = states[A0][2][0], states[A0][2][1]
    n_ref = min(lf0, ld0) / L_DESIGN
    gov = "fatigue" if lf0 < ld0 else "subgrade"
    # rutting-governed calibration (non-linear): solve L_rut(N) = 20
    st0, w0, _ = states[A0]

    def f(logn):
        return structural.lives(st0, w0, 10 ** logn, nmax_years=100.0)[2] - L_DESIGN
    try:
        n_rut = 10 ** brentq(f, -3.0, 4.0, xtol=1e-4)
    except ValueError:
        n_rut = float("nan")
    out = dict(n_ref=float(n_ref), governing=gov, n_rut=float(n_rut), albedo={})
    for a in ALBEDO_GRID:
        st, w, lv = states[a]
        lf, ld = lv[0] / n_ref, lv[1] / n_ref
        lr = structural.lives(st, w, n_rut, nmax_years=150.0)[2] if np.isfinite(n_rut) else float("nan")
        E = st.E
        s = st.strains
        wts = w / w.sum()
        out["albedo"][f"{a:.2f}"] = dict(
            life_fatigue=float(lf), life_subgrade=float(ld), life_ref=float(min(lf, ld)), life_rut=float(lr),
            E_ac_mean=float(np.mean(0.5 * (E[:, 0] + E[:, 1]))), E_sg_mean=float(np.mean(E[:, 4])),
            E_base_mean=float(np.mean(E[:, 2])),
            eps_t_trafficmean=float(np.sum(wts * s[:, 0])), eps_v_trafficmean=float(np.sum(wts * s[:, 1])),
            fatigue_damage_per_year=float(lv[3] * n_ref), subgrade_damage_per_year=float(lv[4] * n_ref))
    return site.key, out


if __name__ == "__main__":
    res = {}
    with ProcessPoolExecutor(4) as ex:
        for key, out in ex.map(one, SITES):
            res[key] = out
            r0 = out["albedo"]["0.14"]
            r1 = out["albedo"]["0.45"]
            print(f"{key:13s} N_ref {out['n_ref']:6.1f}/d ({out['governing']}) life 0.45: {r1['life_ref']:.2f} "
                  f"(ratio {r1['life_ref'] / r0['life_ref']:.3f}) | N_rut {out['n_rut']:.2f}/d rut life "
                  f"{r1['life_rut']:.2f} (ratio {r1['life_rut'] / r0['life_rut']:.3f})", flush=True)
    (RES / f"structural_{TAG}.json").write_text(json.dumps(res, indent=1))

"""Pavement-structure variants (revision; Sect. 6.9): how much the structural
term changes when the pavement of a street differs from the reference
structure (100 mm asphalt, 250 mm base, 300 mm subbase on a silty-clay
subgrade with the water table at 3 m).

Mechanical variants, from the reference thermal runs (30_pavement_sweep.py):
  sg_weak, sg_stiff     subgrade resilient modulus 35 and 140 MPa (70)
  gran_weak, gran_stiff base and subbase moduli x0.6 and x1.6 (250, 150 MPa)
  mix_stiff, mix_soft   in-place air voids of the asphalt 4 and 9 % (7 %)
Thermal and mechanical variants, with new column runs:
  ac_thin, ac_thick     asphalt 50 and 200 mm (new strain tables)
  wt_shallow, wt_deep   water table at 1.5 and 6 m (wetter and drier
                        subgrade: water content, frost, thaw weakening)

For every variant and site the life is computed at the albedos 0.14-0.45,
the effective life follows from the harmonic mean over the reference albedo
path (Eq. 6), and the structural flow 1/L_eff - 1/L_0 is compared with that
of the reference structure.  Lives are relative (Miner's rule), so they do
not depend on the traffic level.

usage: python 87_structure_variants.py [workers]
Writes results/structure_variants_v3.json.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import column, driver, ledger, stiffness, strain_table as stt, structural  # noqa: E402
from alrb.config import RAW, RES, SITE, SITES  # noqa: E402

WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ALB = [0.14, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
MECH = {
    "sg_weak": dict(units=(stiffness.BASE_U, stiffness.SUBBASE_U, replace(stiffness.SUBGRADE_U, mr=35.0))),
    "sg_stiff": dict(units=(stiffness.BASE_U, stiffness.SUBBASE_U, replace(stiffness.SUBGRADE_U, mr=140.0))),
    "gran_weak": dict(units=(replace(stiffness.BASE_U, mr=150.0), replace(stiffness.SUBBASE_U, mr=90.0), stiffness.SUBGRADE_U)),
    "gran_stiff": dict(units=(replace(stiffness.BASE_U, mr=400.0), replace(stiffness.SUBBASE_U, mr=240.0), stiffness.SUBGRADE_U)),
    "mix_stiff": dict(mix=stiffness.ACMix(va=4.0)),
    "mix_soft": dict(mix=stiffness.ACMix(va=9.0)),
}
THERM = {
    "ac_thin": dict(h_ac=0.05),
    "ac_thick": dict(h_ac=0.20),
    "wt_shallow": dict(gwt=1.5),
    "wt_deep": dict(gwt=6.0),
}
Z0M, ZBOT = 0.0021, 8.0
TABDIR = RAW / "strain_tables"


def _points(H):
    zs = [H[0] / 2, H[0] + H[1] / 2, H[:2].sum() - stt.EPS_Z, H.sum() + stt.EPS_Z]
    lays = [0, 1, 1, 4]
    R, Z, L = [], [], []
    for z, l in zip(zs, lays):
        for r in (0.0, stt.S / 2, stt.S):
            R.append(r)
            Z.append(z)
            L.append(l)
    return np.array(R), np.array(Z), np.array(L)


def table_for(h_ac):
    """Critical-strain table for an asphalt layer of thickness h_ac (two halves)."""
    if abs(h_ac - 0.10) < 1e-9:
        return structural.table()
    TABDIR.mkdir(parents=True, exist_ok=True)
    f = TABDIR / f"strain_table_ac{int(round(h_ac * 1000)):03d}.npz"
    if not f.exists():
        stt.H = np.array([h_ac / 2, h_ac / 2, 0.25, 0.30])
        stt._R, stt._Z, stt._L = _points(stt.H)
        tab = stt.build(WORKERS)
        np.savez_compressed(f, tab=tab, X1=stt.X1, X2=stt.X2, X3=stt.X3, X4=stt.X4, H=stt.H)
        # direct check at random states
        rng = np.random.default_rng(3)
        err = []
        for _ in range(60):
            E5 = 10 ** rng.uniform(np.log10(35), np.log10(500))
            E = np.array([E5 * 10 ** rng.uniform(0.5, 2.5), 0, E5 * 10 ** rng.uniform(0.0, 1.0), E5 * 10 ** rng.uniform(0.0, 0.8), E5])
            E[1] = E[0] * 10 ** rng.uniform(-0.6, 0.0)
            d = stt.critical(E * 1e6)
            i = stt.interp_cubic(tab, stt.X1, stt.X2, stt.X3, stt.X4, E[:1], E[1:2], E[2:3], E[3:4], E[4:5])[0]
            err.append(np.abs(i[:2] - d[:2]) / np.abs(d[:2]))
        print(f"strain table h_ac={h_ac}: max rel. error of eps_t, eps_v {np.max(err, axis=0)}", flush=True)
    z = np.load(f)
    return z["tab"], z["X1"], z["X2"], z["X3"], z["X4"]


def lives_from(T, lice, depths, index, key, h_ac=0.10, mix=stiffness.ACMix(),
               units=(stiffness.BASE_U, stiffness.SUBBASE_U, stiffness.SUBGRADE_U)):
    i1 = int(np.argmin(np.abs(depths - h_ac / 4)))
    i2 = int(np.argmin(np.abs(depths - 3 * h_ac / 4)))
    T1 = np.asarray(T[:, i1], float)
    T2 = np.asarray(T[:, i2], float)
    E1 = stiffness.ac_modulus(T1, mix)
    E2 = stiffness.ac_modulus(T2, mix)
    Eu = [stiffness.unbound_series(np.asarray(lice[:, q], float), u.mr, u.rf, u.tr_days * 24.0, stiffness.MR_FROZEN,
                                   0.5, 0.05) for q, u in enumerate(units)]
    E = np.column_stack([E1, E2] + Eu)
    tab, x1, x2, x3, x4 = table_for(h_ac)
    s = stt.interp_cubic(tab, x1, x2, x3, x4, E[:, 0].copy(), E[:, 1].copy(), E[:, 2].copy(), E[:, 3].copy(), E[:, 4].copy())
    st = structural.State(E, T1, T2, s)
    idx = pd.date_range(index[0], periods=len(index), freq="h")
    w = structural.traffic_weights(idx, SITE[key].tz)
    lv = structural.lives(st, w, 1.0)
    return float(min(lv[0], lv[1])), float(lv[0]), float(lv[1])


def flow(L):
    """1/L_eff - 1/L_0 on the reference albedo path, lives relative to 0.14."""
    al = np.array(ALB)
    Ln = np.asarray(L) / L[0] * 20.0
    Leff = ledger.harmonic_life(lambda x: np.interp(x, al, Ln), ledger.Params())
    return 1.0 / Leff - 1.0 / 20.0, Leff


def mech_job(job):
    key, name = job
    out = []
    for a in ALB:
        z = np.load(RAW / "sweep" / "v3" / f"{key}_a{a:.2f}.npz")
        idx = pd.date_range(pd.Timestamp(str(z["t0"])), periods=int(z["n"]), freq="h")
        kw = MECH.get(name, {}) if name != "ref" else {}
        out.append(lives_from(z["T"], z["lice"], z["depths"], idx, key, **kw))
    return key, name, out


def therm_job(job):
    key, name = job
    p = THERM[name]
    h_ac = p.get("h_ac", 0.10)
    df = driver.load_forcing(key)
    mask = driver.snow_mask_pavement(df, key, days_after=1.0)
    sub = driver.substeps(df, key, snow_mask=mask, lw_col="lw")
    rec = np.unique(np.concatenate([driver.REC_DEPTHS_PAV, [h_ac / 4, 3 * h_ac / 4]]))
    pre = driver.prepare(column.pavement_structure(zbot=ZBOT, **p), rec)
    r0 = driver.spun_up_run(pre, sub, 0.14, spin_years=4, z0m=Z0M, hmult=1.0)
    t_bot = float(np.mean(r0.T[:, -1]))
    out = []
    for a in ALB:
        r = driver.spun_up_run(pre, sub, a, spin_years=4, z0m=Z0M, hmult=1.0, t_bot=t_bot)
        out.append(lives_from(r.T.astype(np.float32), r.lice.astype(np.float32), r.depths, r.index, key, h_ac=h_ac))
    return key, name, out


if __name__ == "__main__":
    for h in sorted({p.get("h_ac", 0.10) for p in THERM.values()}):
        table_for(h)                                      # build the strain tables before the workers start
    res = {s.key: {} for s in SITES}
    with ProcessPoolExecutor(WORKERS) as ex:
        jobs = [(s.key, n) for n in ["ref"] + list(MECH) for s in SITES]
        for key, name, L in ex.map(mech_job, jobs):
            res[key][name] = L
        jobs = [(s.key, n) for n in THERM for s in SITES]
        for key, name, L in ex.map(therm_job, jobs):
            res[key][name] = L
            print(name, key, "done", flush=True)
    names = list(MECH) + list(THERM)
    out = dict(albedos=ALB, variants=names, sites={})
    for s in SITES:
        r = res[s.key]
        Lref = [x[0] for x in r["ref"]]
        f0, Le0 = flow(Lref)
        d = dict(ref=dict(L_eff=Le0, flow=f0, life45=Lref[-1] / Lref[0]), factor={}, L_eff={}, life45={})
        for n in names:
            L = [x[0] for x in r[n]]
            f, Le = flow(L)
            d["factor"][n] = f / f0
            d["L_eff"][n] = Le
            d["life45"][n] = L[-1] / L[0]
        out["sites"][s.key] = d
        print(f"{s.key:13s} ref L45/L14 {d['ref']['life45']:.3f} | factors " +
              " ".join(f"{n} {d['factor'][n]:.2f}" for n in names), flush=True)
    (RES / "structure_variants_v3.json").write_text(json.dumps(out, indent=1))

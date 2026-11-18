"""Direct check of the harmonic-life treatment of a time-varying albedo:
the column is driven by the actual decaying, renewed albedo (reference
treatment, applications on 1 May every 3 years from 2017) and the damage is
accumulated on that run; the result is compared with
    1/L_eff = < 1/L(alpha(t)) >
built from the constant-albedo runs."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import albedo_path, column, driver, structural, damage  # noqa: E402
from alrb.config import ALBEDO_GRID, RAW, RES, SITES, SITE, REF  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "v3"
Z0M, LW = 0.0021, "lw"


def one(site):
    df = driver.load_forcing(site.key)
    mask = driver.snow_mask_pavement(df, site.key, days_after=1.0)
    sub = driver.substeps(df, site.key, snow_mask=mask, lw_col=LW)
    pre = driver.prepare(column.pavement_structure(zbot=8.0), driver.REC_DEPTHS_PAV)
    a = albedo_path.sawtooth(df.index, sub.nsub, site.tz, "2017-05-01", REF.alpha0, REF.alphaf, REF.tau, REF.Tr)
    t_bot = json.loads((RES / f"sweep_{TAG}.json").read_text())["sites"][site.key]["0.14"]["t_bot"]
    r = driver.spun_up_run(pre, sub, a, spin_years=4, z0m=Z0M, t_bot=t_bot)
    st = structural.hourly_state(r.T, r.lice, r.depths)
    w = structural.traffic_weights(df.index, site.tz)
    js = json.loads((RES / f"structural_{TAG}.json").read_text())[site.key]
    n_ref = js["n_ref"]
    # analysis window: from the first application
    sel = (df.index >= pd.Timestamp("2017-05-01 07:00", tz="UTC"))
    lv = damage.accumulate(w[sel] * n_ref, st.strains[sel, 0].copy(), st.E[sel, 1].copy(), st.strains[sel, 1].copy(),
                           st.strains[sel, 2].copy(), st.strains[sel, 3].copy(), st.T1[sel], st.T2[sel],
                           damage.kz(0.1, 0.025), damage.kz(0.1, 0.075), 0.05, 0.05, 0.00635, 1.0, 1.0)
    life_direct = min(lv[0], lv[1])
    # harmonic prediction from the constant-albedo lives, on the same hours
    al = np.array(ALBEDO_GRID)
    L = np.array([js["albedo"][f"{x:.2f}"]["life_ref"] for x in al])
    ah = a.reshape(-1, sub.nsub).mean(axis=1)[sel]
    # damage-rate weighting: Miner rates add, so average 1/L over time
    life_harm = 1.0 / np.mean(1.0 / np.interp(ah, al, L))
    life_meanalb = float(np.interp(ah.mean(), al, L))
    # quasi-static, hour by hour: damage rate of the constant-albedo runs interpolated at each hour's albedo
    rates_f, rates_d = [], []
    for x in al:
        z = np.load(RAW / "sweep" / TAG / f"{site.key}_a{x:.2f}.npz")
        sx = structural.hourly_state(z["T"], z["lice"], z["depths"])
        n_h = (w * n_ref)[sel]
        et, ev, E2 = sx.strains[sel, 0], sx.strains[sel, 1], sx.E[sel, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            rates_f.append(np.where(et > 0, n_h / (0.0795 * np.maximum(et, 1e-12) ** -3.291 * (E2 * 1e6 / 6894.757) ** -0.854), 0))
            rates_d.append(np.where(ev > 0, n_h / (1.365e-9 * np.maximum(ev, 1e-12) ** -4.477), 0))
    rates_f = np.array(rates_f)
    rates_d = np.array(rates_d)
    j = np.clip(np.searchsorted(al, ah) - 1, 0, len(al) - 2)
    f = (ah - al[j]) / (al[j + 1] - al[j])
    cols = np.arange(len(ah))
    df_qs = (1 - f) * rates_f[j, cols] + f * rates_f[j + 1, cols]
    dd_qs = (1 - f) * rates_d[j, cols] + f * rates_d[j + 1, cols]
    yrs = sel.sum() / 8766.0
    life_qs = float(min(yrs / df_qs.sum(), yrs / dd_qs.sum()))
    return site.key, dict(life_direct=float(life_direct), life_harmonic=float(life_harm), life_quasistatic=life_qs,
                          life_at_mean_albedo=life_meanalb, rel_err_harmonic=float(life_harm / life_direct - 1),
                          rel_err_quasistatic=float(life_qs / life_direct - 1),
                          mean_albedo=float(ah.mean()), frost_max=float(r.frost.max()))


if __name__ == "__main__":
    out = {}
    with ProcessPoolExecutor(4) as ex:
        for k, v in ex.map(one, SITES):
            out[k] = v
            print(k, {q: round(x, 4) for q, x in v.items()}, flush=True)
    (RES / f"timevarying_check_{TAG}.json").write_text(json.dumps(out, indent=1))

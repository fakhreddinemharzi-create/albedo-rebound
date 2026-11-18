"""Every number quoted in the manuscript, generated from the result files.

Writes
  manuscript/generated/numbers.tex   \\defval{key}{value} for \\val{key}
  manuscript/generated/tab_sites.tex
  manuscript/generated/tab_results.tex
  manuscript/generated/tab_vv.tex
  manuscript/generated/tab_metrics.tex
"""
import json
import sys
from pathlib import Path

import contourpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import driver, palette as C, summaries  # noqa: E402
from alrb.config import MS, RAW, RES, SITES, REF  # noqa: E402

TAG = "v3"
OUT = MS / "generated"
OUT.mkdir(parents=True, exist_ok=True)
V = {}
KEYS = C.SITE_ORDER


def f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:.{nd}f}"
    return s.replace("-", "\\vminus ") if s.startswith("-") else s


def put(k, v, nd=2):
    V[k] = f(v, nd) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else str(v)


def rng(k, arr, nd=2):
    a = np.asarray(arr, float)
    put(k + ".min", float(np.nanmin(a)), nd)
    put(k + ".max", float(np.nanmax(a)), nd)
    put(k + ".med", float(np.nanmedian(a)), nd)
    put(k + ".mean", float(np.nanmean(a)), nd)


def load(name):
    return json.loads((RES / name).read_text())


def main():
    sites = load("sites.json")
    th = load(f"thermal_{TAG}.json")
    st = load(f"structural_{TAG}.json")
    lg = load(f"ledger_{TAG}.json")
    ds = load(f"design_space_{TAG}.json")
    mf = load(f"manifold_{TAG}.json")
    sb = load(f"sobol_{TAG}.json")
    kj = load("kernels.json")
    nv = load("natural_validation_lw.json")
    lt = load(f"ltpp_benchmark_{TAG}.json")
    vt = load("verification_thermal.json")
    vm = load("verification_mechanics.json")
    tv = load(f"timevarying_check_{TAG}.json")
    qc = load("forcing_qc.json")
    sw = load(f"sweep_{TAG}.json")
    # ------------------------------------------------------------------ sites and data
    put("nsites", "ten")
    rng("site.maat", [sites[k]["maat"] for k in KEYS], 1)
    rng("site.fi", [sites[k]["fi"] for k in KEYS], 0)
    rng("site.precip", [sites[k]["precip"] for k in KEYS], 0)
    rng("site.sw", [sites[k]["sw"] for k in KEYS], 0)
    rng("site.kt", [sites[k]["kt"] for k in KEYS], 2)
    rng("site.lat", [sites[k]["lat"] for k in KEYS], 1)
    rng("site.snowfrac", [100 * sites[k]["snow_frac"] for k in KEYS], 0)
    for k in KEYS:
        put(f"site.{k}.fi", sites[k]["fi"], 0)
        put(f"site.{k}.maat", sites[k]["maat"], 1)
    miss = [max(v for v in qc[k]["missing"].values()) for k in KEYS]
    rng("qc.missing", [100 * m for m in miss], 2)
    rng("qc.swbias", [qc[k]["sw_vs_ceres_bias"] for k in KEYS], 1)
    rng("qc.swr", [qc[k]["sw_vs_ceres_r"] for k in KEYS], 2)
    rng("qc.lwbias", [qc[k]["lw_vs_ceres_bias"] for k in KEYS], 1)
    rng("qc.lwrmse", [qc[k]["lw_vs_ceres_rmse"] for k in KEYS], 1)
    rng("qc.lwr", [qc[k]["lw_vs_ceres_r"] for k in KEYS], 2)
    pr = [abs(qc[k]["prcp_crn_sum"] / qc[k]["prcp_ghcn_sum"] - 1) * 100 for k in KEYS if qc[k].get("prcp_ghcn_sum")]
    rng("qc.prcpdiff", pr, 1)
    # ------------------------------------------------------------------ verification
    put("ver.harm.maxerr", max(vt["harmonic"]["max_abs_err"]), 3)
    put("ver.harm.amp", min(vt["harmonic"]["amp_ratio"]), 3)
    put("ver.neu.front", 1000 * vt["neumann"]["front_max_abs_err"], 0)
    put("ver.neu.frontrel", 100 * vt["neumann"]["front_max_rel_err"], 1)
    put("ver.neu.T", vt["neumann"]["T_max_abs_err"], 4)
    put("ver.neu.lam", vt["neumann"]["lam"], 4)
    put("ver.energy", max(abs(vt["energy_alb0.14"]["rel_error"]), abs(vt["energy_alb0.45"]["rel_error"])), 0)
    V["ver.energy"] = f"{max(abs(vt['energy_alb0.14']['rel_error']), abs(vt['energy_alb0.45']['rel_error'])):.0e}"
    cv = vt["convergence"]
    chg = [abs(cv[v][q] / cv["ref"][q] - 1) * 100 for v in ("dt_60min", "dt_5min", "fine_grid", "coarse_grid")
           for q in ("dts_mean", "dts_max", "dT085")]
    put("ver.conv.max", max(chg), 2)
    put("ver.conv.ncell", cv["ref"]["ncell"], 0)
    put("ver.conv.ncellfine", cv["fine_grid"]["ncell"], 0)
    b = vm["boussinesq"]
    put("ver.bous.sz", 100 * b["max_rel_err_sz"], 4)
    put("ver.bous.w", 100 * b["max_rel_err_w"], 4)
    rel = [abs(p["let"] / p["fe"] - 1) * 100 for d in vm["fe"].values() for p in d["points"]]
    put("ver.fe.max", max(rel), 1)
    put("ver.fe.med", float(np.median(rel)), 1)
    put("ver.fe.ndof", list(vm["fe"].values())[0]["ndof"], 0)
    tb = vm["table"]
    put("ver.tab.fat", max(abs(tb[k]["fatigue_ratio"] - 1) for k in KEYS) * 100, 2)
    put("ver.tab.sg", max(abs(tb[k]["subgrade_ratio"] - 1) for k in KEYS) * 100, 2)
    put("ver.tab.rut", max(max(tb[k]["rut25_err"], tb[k]["rut75_err"]) for k in KEYS) * 100, 1)
    # ------------------------------------------------------------------ evaluation
    pv = nv["pooled_validation"]
    put("eval.lst.rmse", pv["rmse"], 2)
    put("eval.lst.bias", pv["bias"], 2)
    put("eval.lst.n", f"{pv['n']:,}".replace(",", "\\,"))
    vr = [nv["sites"][k]["validation"]["rmse"] for k in KEYS if "validation" in nv["sites"][k]]
    rng("eval.lst.site", vr, 2)
    ls = lt["stats"]
    put("eval.ltpp.hibias", ls["high_vs_ltpp_bias"], 1)
    put("eval.ltpp.hirmse", ls["high_vs_ltpp_rmse"], 1)
    put("eval.ltpp.shrpbias", ls["high_vs_shrp_bias"], 1)
    put("eval.ltpp.lobias", ls["low_vs_ltpp_bias"], 1)
    put("eval.ltpp.lormse", ls["low_vs_ltpp_rmse"], 1)
    put("eval.tv.harm", max(abs(tv[k]["rel_err_harmonic"]) for k in KEYS) * 100, 1)
    put("eval.tv.qs", max(abs(tv[k]["rel_err_quasistatic"]) for k in KEYS) * 100, 2)
    put("eval.tv.mean", max(abs(tv[k]["life_at_mean_albedo"] / tv[k]["life_direct"] - 1) for k in KEYS) * 100, 1)
    rng("eval.tv.direct", [tv[k]["life_direct"] for k in KEYS], 2)
    sm = kj["summary"]
    put("eval.cmip.bias", 100 * sm["rel_cmip6_minus_obs"], 0)
    put("eval.cmip.r", sm["r_cmip6_obs"], 2)
    put("eval.cmip.n", len(sm["models"]), 0)
    put("eval.simple.bias", 100 * sm["rel_simple_minus_obs"], 0)
    # ------------------------------------------------------------------ thermal
    a0 = lambda k, q: th[k]["albedo"]["0.14"][q]
    a1 = lambda k, q: th[k]["albedo"]["0.45"][q]
    dts = {k: a1(k, "ts_mean") - a0(k, "ts_mean") for k in KEYS}
    dtx = {k: a1(k, "ts_max") - a0(k, "ts_max") for k in KEYS}
    d85 = {k: th[k]["dT_depth_mean"][11] for k in KEYS}
    rng("th.dts", [-v for v in dts.values()], 1)
    rng("th.dtsmax", [-v for v in dtx.values()], 1)
    rng("th.dt85", [-v for v in d85.values()], 1)
    for k in KEYS:
        put(f"th.{k}.dts", -dts[k], 2)
        put(f"th.{k}.frost0", a0(k, "frost_max"), 2)
        put(f"th.{k}.frost1", a1(k, "frost_max"), 2)
        put(f"th.{k}.fdays0", a0(k, "frozen_days"), 0)
        put(f"th.{k}.fdays1", a1(k, "frozen_days"), 0)
        put(f"th.{k}.dfrost", 100 * (a1(k, "frost_max") - a0(k, "frost_max")), 0)
    fk = [k for k in KEYS if a0(k, "frost_max") > 0.01]
    rng("th.dfrostcm", [100 * (a1(k, "frost_max") - a0(k, "frost_max")) for k in fk], 0)
    rng("th.dfdays", [a1(k, "frozen_days") - a0(k, "frozen_days") for k in fk], 0)
    q25 = {k: np.array(th[k]["T25_quantiles_045"]) - np.array(th[k]["T25_quantiles_014"]) for k in KEYS}
    rng("th.dT25p99", [-q25[k][6] for k in KEYS], 1)
    rng("th.dT25p01", [-q25[k][0] for k in KEYS], 1)
    # effective surface exchange coefficient (monthly regression through the origin)
    X, Y = [], []
    for k in KEYS:
        df = driver.load_forcing(k)
        li = summaries.local_index(df.index, k)
        mask = np.load(RAW / "sweep" / TAG / f"{k}_snowmask.npy").astype(float)
        sel = (li.year >= 2017) & (li.year <= 2024)
        s_ = (df.sw.to_numpy() * (1 - mask[: len(df)]))[sel]
        mon = li[sel].month
        swm = np.bincount(mon - 1, weights=s_, minlength=12) / np.bincount(mon - 1, minlength=12)
        X += list(0.31 * swm)
        Y += list(th[k]["dts_monthly"])
    X, Y = np.array(X), np.array(Y)
    put("th.Lambda", -float(np.sum(X * X) / np.sum(X * Y)), 1)
    put("th.Lambda.r", float(np.corrcoef(X, Y)[0, 1]), 2)
    # ------------------------------------------------------------------ structural
    lf = lambda k, a, q="life_fatigue": st[k]["albedo"][f"{a:.2f}"][q]
    rng("st.lf45", [lf(k, 0.45) for k in KEYS], 1)
    rng("st.lf45pct", [100 * (lf(k, 0.45) / 20 - 1) for k in KEYS], 0)
    rng("st.lf65", [lf(k, 0.65) for k in KEYS], 1)
    rng("st.ld45pct", [100 * (lf(k, 0.45, "life_subgrade") / lf(k, 0.14, "life_subgrade") - 1) for k in KEYS], 0)
    rng("st.lr45", [lf(k, 0.45, "life_rut") for k in KEYS], 0)
    rng("st.eac14", [lf(k, 0.14, "E_ac_mean") / 1000 for k in KEYS], 1)
    rng("st.eac45pct", [100 * (lf(k, 0.45, "E_ac_mean") / lf(k, 0.14, "E_ac_mean") - 1) for k in KEYS], 0)
    rng("st.ept45pct", [100 * (1 - lf(k, 0.45, "eps_t_trafficmean") / lf(k, 0.14, "eps_t_trafficmean")) for k in KEYS], 0)
    rng("st.nref", [st[k]["n_ref"] for k in KEYS], 0)
    rng("st.nrut", [st[k]["n_rut"] for k in KEYS], 1)
    put("st.gov", "fatigue" if all(st[k]["governing"] == "fatigue" for k in KEYS) else "mixed")
    rng("st.leff", [lg[k]["reference"]["L_eff"] for k in KEYS], 2)
    rng("st.leffpct", [100 * (lg[k]["reference"]["L_eff"] / 20 - 1) for k in KEYS], 1)
    # ------------------------------------------------------------------ kernel
    rng("k.hourly", [-kj["sites"][k]["K_aprp_hourly"] for k in KEYS], 2)
    rng("k.masked", [-kj["sites"][k]["K_aprp_snowmasked"] for k in KEYS], 2)
    rng("k.snowpct", [100 * (1 - kj["sites"][k]["K_aprp_snowmasked"] / kj["sites"][k]["K_aprp_hourly"]) for k in KEYS], 0)
    rng("k.clearratio", [kj["sites"][k]["K_clear"] / kj["sites"][k]["K_aprp_hourly"] for k in KEYS], 2)
    rng("k.simpleratio", [kj["sites"][k]["K_simple"] / kj["sites"][k]["K_aprp_hourly"] for k in KEYS], 2)
    rng("k.monthlyratio", [kj["sites"][k]["K_aprp_monthlymeans"] / kj["sites"][k]["K_aprp_hourly"] for k in KEYS], 2)
    # ------------------------------------------------------------------ ledger
    ref = {k: lg[k]["reference"] for k in KEYS}
    put("lg.phi", ref["durham"]["phi"], 2)
    put("lg.phipct", 100 * ref["durham"]["phi"], 0)
    put("lg.dalpha", ref["durham"]["mean_delta_alpha"], 3)
    put("lg.ncoats", REF.H / REF.Tr, 1)
    for m in ("static", "dynamic", "gtp", "stock"):
        for q in ("G_rf", "G_co", "G_st", "G_net", "R"):
            vals = [ref[k][m][q] for k in KEYS]
            if q == "G_rf" or q == "G_st":
                vals = [-v for v in vals]
            rng(f"lg.{m}.{q}", vals, 1 if q != "R" else 2)
            if q == "R":
                rng(f"lg.{m}.R1", vals, 1)
        for k in KEYS:
            for q in ("G_rf", "G_co", "G_st", "G_net", "R"):
                put(f"lg.{k}.{m}.{q}", ref[k][m][q], 1 if q != "R" else 2)
    for c in ("static", "dynamic", "stock", "gtp", "dynamic_TH20", "gtp_sustained_80", "stock_af055"):
        rng(f"coef.{c}", [lg[k]["coefficient"][c] for k in KEYS], 2)
    rng("kappa.100", [lg[k]["kappa"]["100"] for k in KEYS], 2)
    rng("kappa.20", [lg[k]["kappa"]["20"] for k in KEYS], 1)
    rng("kappa.500", [lg[k]["kappa"]["500"] for k in KEYS], 2)
    for vname in lg["durham"]["variants"]:
        dv = [lg[k]["variants"][vname]["G_net"] - ref[k]["static"]["G_net"] for k in KEYS]
        rng(f"var.{vname}", dv, 1)
        rng(f"var.{vname}.net", [lg[k]["variants"][vname]["G_net"] for k in KEYS], 1)
    hz = {h: [lg[k]["horizons"][h]["dynamic"] for k in KEYS] for h in lg["durham"]["horizons"]}
    for h in hz:
        rng(f"hz.{h}.dyn.net", [x["G_net"] for x in hz[h]], 1)
        rng(f"hz.{h}.dyn.rf", [-x["G_rf"] for x in hz[h]], 1)
        rng(f"hz.{h}.dyn.co", [x["G_co"] for x in hz[h]], 1)
        rng(f"hz.{h}.gtp.net", [lg[k]["horizons"][h]["gtp"]["G_net"] for k in KEYS], 1)
    # renewal sweep: net at T_r = 10 and 20
    for trv in (1.0, 5.0, 10.0, 20.0):
        vals = []
        for k in KEYS:
            rs = lg[k]["renewal_sweep"]
            i = int(np.argmin(np.abs(np.array(rs["Tr"]) - trv)))
            vals.append(rs["static"][i]["G_net"])
        rng(f"sweep.Tr{int(trv)}", vals, 1)
    # ------------------------------------------------------------------ closed form: payback time of a coat
    dA = REF.alphaf - REF.alpha0
    D = REF.tau * (1 - np.exp(-REF.Tr / REF.tau))
    put("bf.D", D, 2)
    put("bf.dA", dA, 2)
    Ts = [REF.g_co / (lg[k]["kappa"]["100"] * dA) for k in KEYS]
    rng("bf.Tstar", Ts, 1)
    rng("bf.Rclosed", [t / D for t in Ts], 1)
    for k in KEYS:
        put(f"bf.{k}.Tstar", REF.g_co / (lg[k]["kappa"]["100"] * dA), 1)
        put(f"kappa.{k}.100", lg[k]["kappa"]["100"], 2)
    # ------------------------------------------------------------------ design space
    for m in ("static", "dynamic", "gtp", "stock"):
        rng(f"ds.gstar.{m}", [ds["sites"][k]["ref"][m]["g_co_star"] for k in KEYS], 2)
        rng(f"ds.ratio.{m}", [ds["sites"][k]["ref"][m]["ratio"] for k in KEYS], 1)
        rng(f"ds.taustar.{m}", [ds["sites"][k]["tau_star"][f"{m}_Tr50"] for k in KEYS], 1)
        for k in KEYS:
            put(f"ds.{k}.gstar.{m}", ds["sites"][k]["ref"][m]["g_co_star"], 2)
    # renewal floor: smallest T_r at which some tau <= 20 yr reaches g* = 2.70 (dynamic)
    mt, mr = np.array(mf["taus"]), np.array(mf["trs"])
    floors = []
    for k in KEYS:
        G = np.array(mf["sites"][k]["dynamic"], float)
        ok = np.where(np.nanmax(G, axis=0) >= REF.g_co)[0]
        floors.append(float(mr[ok[0]]) if len(ok) else np.nan)
        put(f"ds.{k}.floor", floors[-1], 1)
    rng("ds.floor", floors, 1)
    # tau* at T_r = 10 (dynamic) from the manifold
    t10 = []
    for k in KEYS:
        G = np.array(mf["sites"][k]["dynamic"], float)
        j = int(np.argmin(np.abs(mr - 10.0)))
        col = G[:, j]
        t10.append(float(np.interp(REF.g_co, col, mt)) if col.max() > REF.g_co else np.nan)
    rng("ds.taustar10", t10, 1)
    # ------------------------------------------------------------------ uncertainty
    for des in ("A", "B"):
        for m in ("static", "dynamic", "gtp", "stock"):
            rng(f"sb.{des}.{m}.p", [100 * sb["results"][des][k][m]["p_source"] for k in KEYS], 0)
            rng(f"sb.{des}.{m}.med", [sb["results"][des][k][m]["q"][2] for k in KEYS], 1)
            rng(f"sb.{des}.{m}.q05", [sb["results"][des][k][m]["q"][0] for k in KEYS], 1)
            rng(f"sb.{des}.{m}.q95", [sb["results"][des][k][m]["q"][4] for k in KEYS], 1)
        names = sb["results"][des]["durham"]["dynamic"]["names"]
        for n in names:
            rng(f"sb.{des}.ST.{n}", [sb["results"][des][k]["dynamic"]["ST"][names.index(n)] for k in KEYS], 2)
    put("sb.N", sb["N"], 0)
    put("sb.nA", sb["N"] * (12 + 2), 0)
    put("sb.nB", sb["N"] * (11 + 2), 0)
    # ------------------------------------------------------------------ deep boundary
    rng("deep.tbot", [sw["sites"][k]["0.14"]["t_bot"] for k in KEYS if "t_bot" in sw["sites"][k]["0.14"]], 1)
    # ------------------------------------------------------------------ lower boundary, canyon
    boundary_numbers()
    canyon_numbers()
    conus_numbers()
    season_numbers()
    # ------------------------------------------------------------------ climate change (if available)
    climate_numbers()
    # ------------------------------------------------------------------ write
    lines = ["% generated by scripts/70_numbers.py -- do not edit"]
    for k in sorted(V):
        lines.append(f"\\defval{{{k}}}{{{V[k]}}}")
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
    tables(sites, th, st, lg, ds, kj)
    print(len(V), "values written")


def boundary_numbers():
    p = RES / f"boundary_test_{TAG}.json"
    if not p.exists():
        return
    bt = json.loads(p.read_text())
    lab = {"anchored8": "reference: anchored at 8\\,m", "adiabatic8": "adiabatic at 8\\,m",
           "anchored15": "anchored at 15\\,m"}
    rows, lr_dev, dfr = [], [], {}
    for k in ("goodridge", "durham", "tucson"):
        ref_lr = bt[k]["anchored8"]["0.45"]["life"] / bt[k]["anchored8"]["0.14"]["life"]
        for i, (var, name) in enumerate(lab.items()):
            u, t = bt[k][var]["0.14"], bt[k][var]["0.45"]
            lr = t["life"] / u["life"]
            lr_dev.append(100 * abs(lr / ref_lr - 1))
            fr = f"{u['frost_max_mean']:.2f}$\\rightarrow${t['frost_max_mean']:.2f}" if u["frost_max_mean"] > 0.01 else "--"
            dfr[(k, var)] = t["frost_max_mean"] - u["frost_max_mean"]
            site = C.SITE_SHORT[k] if i == 0 else ""
            rows.append(f"{site} & {name} & {fr} & {t['T_0.05'] - u['T_0.05']:.2f} & {t['T_0.85'] - u['T_0.85']:.2f} & "
                        f"{t['T_3.00'] - u['T_3.00']:.2f} & {lr:.3f} \\\\")
            put(f"bt.{k}.{var}.lr", lr, 3)
            put(f"bt.{k}.{var}.dT3", t["T_3.00"] - u["T_3.00"], 1)
            put(f"bt.{k}.{var}.dT085", t["T_0.85"] - u["T_0.85"], 1)
            put(f"bt.{k}.{var}.dTac", t["T_0.05"] - u["T_0.05"], 2)
        if k != "tucson":
            rows.append("\\addlinespace[2pt]")
    put("bt.lrdev.max", max(lr_dev), 1)
    for var, tag in (("anchored8", "a8"), ("adiabatic8", "ad"), ("anchored15", "a15")):
        devs = []
        for k in ("goodridge", "durham", "tucson"):
            u, t = bt[k][var]["0.14"], bt[k][var]["0.45"]
            if "depths" not in u:
                continue
            d = np.array(u["depths"])
            dT = np.array(t["T_profile"]) - np.array(u["T_profile"])
            nrm = dT / dT[int(np.argmin(np.abs(d - 0.05)))]
            m = (d >= 0.5) & (d <= 3.0)
            devs.append(float(np.max(np.abs(nrm[m] - 2 / np.pi * np.arctan(10.0 / (2 * d[m]))))))
        if devs:
            put(f"bt.strip.{tag}", 100 * max(devs), 0)
    put("bt.gdr.dfrost.ref", 100 * dfr[("goodridge", "anchored8")], 0)
    put("bt.gdr.dfrost.ad", 100 * dfr[("goodridge", "adiabatic8")], 0)
    put("bt.gdr.dfrost.15", 100 * dfr[("goodridge", "anchored15")], 0)
    (OUT / "tab_boundary_rows.tex").write_text(neg("\n".join(rows)).replace("\\vminus \\vminus ", "--") + "\n")


def conus_numbers():
    p = RES / f"conus_ledger_{TAG}.json"
    if not p.exists():
        return
    cs = json.loads(p.read_text())
    ck = json.loads((RES / "conus_kernel_v3.json").read_text())
    put("us.ncells", cs["ncells"], 0)
    put("us.ratio", cs["ratio"], 2)
    put("us.bias", 100 * (1 / cs["ratio"] - 1), 0)
    st = np.array([v["station_Km"] for v in ck["sites"].values()])
    gr = np.array([v["grid_Km"] for v in ck["sites"].values()])
    put("us.r", float(np.corrcoef(st, gr)[0, 1]), 2)
    rr = st / gr
    put("us.ratio.min", float(rr.min()), 2)
    put("us.ratio.max", float(rr.max()), 2)
    res_ = rr / cs["ratio"]
    put("us.resid.min", 100 * float(res_.min() - 1), 0)
    put("us.resid.max", 100 * float(res_.max() - 1), 0)
    for k, nd in (("R_static", 1), ("R_dynamic", 1), ("R_gtp", 0), ("gstar_dynamic", 2), ("Tstar", 1), ("K", 2),
                  ("Gnet_dynamic", 0)):
        d = cs[k]
        sgn = -1.0 if k == "K" else 1.0
        lo, hi = sorted((sgn * d["min"], sgn * d["max"]))
        put(f"us.{k}.min", lo, nd)
        put(f"us.{k}.max", hi, nd)
        put(f"us.{k}.med", sgn * d["med"], nd)
        p05, p95 = sorted((sgn * d["p05"], sgn * d["p95"]))
        put(f"us.{k}.p05", p05, nd)
        put(f"us.{k}.p95", p95, nd)
    for thr in ("3.0", "4.0"):
        put(f"us.share.R{thr.replace('.', 'p')}", 100 * cs[f"share_Rstatic_gt_{thr}"], 0)
    put("us.share.dyn1", 100 * cs["share_Rdyn_gt_1"], 0)
    from alrb.config import REF as _REF
    put("us.gfac.min", _REF.g_co / cs["gstar_dynamic"]["max"], 1)
    put("us.gfac.max", _REF.g_co / cs["gstar_dynamic"]["min"], 1)
    D = _REF.tau * (1 - np.exp(-_REF.Tr / _REF.tau))
    put("us.tfac.min", cs["Tstar"]["min"] / D, 1)
    put("us.tfac.max", cs["Tstar"]["max"] / D, 1)
    put("us.Kfac", cs["K"]["min"] / cs["K"]["max"], 1)
    zz = np.load(RES / f"conus_ledger_{TAG}.npz")
    kap = zz["kappa"][np.isfinite(zz["kappa"])]
    put("us.prem.min", 0.1 * 50 * float(kap.min()), 0)
    put("us.prem.max", 0.1 * 50 * float(kap.max()), 0)
    lgj = json.loads((RES / f"ledger_{TAG}.json").read_text())
    ks = [lgj[k]["kappa"]["100"] for k in KEYS]
    put("disc.prem.min", 0.1 * 50 * min(ks), 0)
    put("disc.prem.max", 0.1 * 50 * max(ks), 0)
    rows = []
    for k in KEYS:
        v = cs["validation"][k]
        c = ck["sites"][k]
        rows.append(f"{C.SITE_SHORT[k]} & {-c['station_Km']:.2f} & {-c['grid_Km']:.2f} & "
                    f"{c['station_Km'] / c['grid_Km']:.2f} & {c['grid_kt']:.2f} & {v['R_static_station']:.2f} & "
                    f"{v['R_static_map']:.2f} & {v['gstar_station']:.2f} & {v['gstar_map']:.2f} \\\\")
    (OUT / "tab_s8_rows.tex").write_text("\n".join(rows) + "\n")
    dev = [100 * abs(v["R_static_map"] / v["R_static_station"] - 1) for v in cs["validation"].values()]
    put("us.val.max", max(dev), 1)
    put("us.val.rms", float(np.sqrt(np.mean(np.square(dev)))), 1)


def season_numbers():
    p = RES / f"season_{TAG}.json"
    if not p.exists():
        return
    se = json.loads(p.read_text())
    f = {k: np.array(v) for k, v in se["stations"].items()}
    rng("season.st.gain", [100 * (v.max() - 1) for v in f.values()], 0)
    rng("season.st.octloss", [100 * (1 - v[9]) for v in f.values()], 0)
    rng("season.st.pen", [100 * (1 - v[9] / v.max()) for v in f.values()], 0)
    rng("season.st.rinc", [100 * (v[4] / v[9] - 1) for v in f.values()], 0)
    g = se["grid"]
    put("season.grid.pen.min", g["penalty_oct"]["min"], 0)
    put("season.grid.pen.max", g["penalty_oct"]["max"], 0)
    put("season.grid.pen.med", g["penalty_oct"]["med"], 0)
    put("season.grid.maylos.max", g["loss_may"]["max"], 1)
    put("season.grid.maylos.med", g["loss_may"]["med"], 1)
    b = g["best_month_share"]
    put("season.best.mar", 100 * b["3"], 0)
    put("season.best.apr", 100 * b["4"], 0)
    put("season.best.may", 100 * b["5"], 0)
    put("season.best.jun", 100 * b["6"], 0)


def canyon_numbers():
    p = RES / f"canyon_{TAG}.json"
    if not p.exists():
        return
    cy = json.loads(p.read_text())
    h = np.array(cy["h"])
    j = {x: int(np.argmin(np.abs(h - x))) for x in (0.2, 0.3, 0.5, 1.0, 2.0)}
    for x, jj in j.items():
        tag = f"{x:.1f}".replace(".", "p")
        put(f"cy.fgs.{tag}", cy["fgs"][jj], 2)
        put(f"cy.dif.{tag}", cy["diffuse"]["0.30"][jj], 2)
        for o in ("EW", "NS", "diag"):
            rng(f"cy.{o}.{tag}", [cy["sites"][k][o]["psi"][jj] for k in KEYS], 2)
        allo = [cy["sites"][k][o]["psi"][jj] for k in KEYS for o in ("EW", "NS", "diag")]
        rng(f"cy.all.{tag}", allo, 2)
        R = [cy["sites"][k]["R_open"] / cy["sites"][k][o]["psi"][jj] for k in KEYS for o in ("EW", "NS", "diag")]
        rng(f"cy.R.{tag}", R, 1)
        rng(f"cy.REW.{tag}", [cy["sites"][k]["R_open"] / cy["sites"][k]["EW"]["psi"][jj] for k in KEYS], 1)  # as in the figure
    # aspect ratio at which psi falls to 0.6 (lower end of the sensitivity range) and to 0.5
    for thr in (0.6, 0.5):
        hs = []
        for k in KEYS:
            for o in ("EW", "NS"):
                ps = np.array(cy["sites"][k][o]["psi"])
                hs.append(float(np.interp(-thr, -ps, h)))
        rng(f"cy.hat{int(thr * 10)}", hs, 2)
    # wall albedo effect at H/W = 1 (E-W street), across sites
    for aw in ("0.10", "0.60"):
        rng(f"cy.aw{aw.replace('.', 'p')}.1p0", [cy["sites"][k]["EW_by_aw"][aw][j[1.0]] for k in KEYS], 2)
    if "path_tracing" in cy:
        d = [abs(c["pt_psi"] - c["rad_psi"]) for c in cy["path_tracing"]]
        de = [abs(c["pt_esc"][i] - c["rad_esc"][i]) for c in cy["path_tracing"] for i in (0, 1)]
        put("cy.pt.maxdpsi", max(d), 3)
        put("cy.pt.maxdesc", max(de), 3)
        put("cy.pt.n", len(cy["path_tracing"]), 0)
    if "photon_mc" in cy:
        put("cy.mc.maxdesc", max(abs(c["mc_esc"][i] - c["rad_esc"][i]) for c in cy["photon_mc"] for i in (0, 1)), 4)
        put("cy.mc.maxdpsi", max(abs(c["mc_psi"] - c["rad_psi"]) for c in cy["photon_mc"]), 3)
    q2 = RES / f"canyon_fatecurve_{TAG}.json"
    if q2.exists():
        fc = json.loads(q2.read_text())
        hh = np.array(fc["h"])
        w = np.array(fc["walls"])
        fl = np.array(fc["floor"])
        j = int(np.argmax(w))
        put("cy.fc.wallmax", w[j], 2)
        put("cy.fc.wallmax.h", hh[j], 1)
        put("cy.fc.share.lo", 100 * w[0] / fl[0], 0)
        put("cy.fc.share.lo.h", hh[0], 1)
        put("cy.fc.share.hi", 100 * w[-1] / fl[-1], 0)
        put("cy.fc.share.hi.h", hh[-1], 0)
    # irradiance- vs kernel-weighted
    rng("cy.irrw", [cy["sites"][k]["EW_irradiance_weighted_h1"] for k in KEYS], 2)
    # where the additional reflection goes (Durham, E-W street), from the figure's data
    q = RES / f"canyon_fate_{TAG}.json"
    if q.exists():
        fate = json.loads(q.read_text())
        for hh, v in fate.items():
            tag = hh.replace(".", "p")
            put(f"cy.fate.{tag}.floor", -v["floor"], 2)
            put(f"cy.fate.{tag}.walls", v["walls"], 2)
            put(f"cy.fate.{tag}.esc", v["escape"], 2)
            put(f"cy.fate.{tag}.wallshare", 100 * v["walls"] / -v["floor"], 0)


def climate_numbers():
    p = RES / f"climate_{TAG}.json"
    if not p.exists():
        return
    cj = json.loads(p.read_text())
    S = cj["sites"]
    put("cc.nruns", cj["nruns"] + len(S), 0)
    put("cc.nmodels", len(cj["models"]), 0)
    for ssp in ("ssp245", "ssp585"):
        ks = [k for k in KEYS if ssp in S[k]]
        E = {k: S[k][ssp]["ensemble"] for k in ks}
        put(f"cc.{ssp}.nmin", min(E[k]["n"] for k in ks), 0)
        rng(f"cc.{ssp}.dta", [E[k]["dta"]["med"] for k in ks], 1)
        rng(f"cc.{ssp}.dK", [E[k]["dK_pct"]["med"] for k in ks], 1)
        put(f"cc.{ssp}.dK.absmax", max(max(abs(E[k]["dK_pct"]["min"]), abs(E[k]["dK_pct"]["max"])) for k in ks), 1)
        rng(f"cc.{ssp}.dDu", [E[k]["dDu_pct"]["med"] for k in ks], 0)
        put(f"cc.{ssp}.dDu.maxmodel", max(E[k]["dDu_pct"]["max"] for k in ks), 0)
        rng(f"cc.{ssp}.Rchg", [100 * (E[k]["R_ratio"]["med"] - 1) for k in ks], 1)
        put(f"cc.{ssp}.Rchg.medabsmax", max(abs(100 * (E[k]["R_ratio"]["med"] - 1)) for k in ks), 1)
        put(f"cc.{ssp}.Rchg.absmax", max(max(abs(100 * (E[k]["R_ratio"]["min"] - 1)),
                                             abs(100 * (E[k]["R_ratio"]["max"] - 1))) for k in ks), 1)
        rng(f"cc.{ssp}.R", [E[k]["R"]["med"] for k in ks], 2)
        rng(f"cc.{ssp}.benefit", [100 * E[k]["benefit"]["med"] for k in ks], 1)
        rng(f"cc.{ssp}.dbenefit", [100 * (E[k]["benefit"]["med"] - S[k]["stationary"]["benefit"]) for k in ks], 1)
        put(f"cc.{ssp}.dbenefit.absmax", max(abs(100 * (E[k]["benefit"]["med"] - S[k]["stationary"]["benefit"]))
                                             for k in ks), 1)
        rng(f"cc.{ssp}.Gst", [E[k]["G_st"]["med"] for k in ks], 2)
        g = S["goodridge"][ssp]["ensemble"]
        put(f"cc.{ssp}.gdr.frost0", g["frost_u0"]["med"], 2)
        put(f"cc.{ssp}.gdr.frost1", g["frost_u1"]["med"], 2)
        put(f"cc.{ssp}.gdr.dfrost0", 100 * (g["frost_t0"]["med"] - g["frost_u0"]["med"]), 0)
        put(f"cc.{ssp}.gdr.dfrost1", 100 * (g["frost_t1"]["med"] - g["frost_u1"]["med"]), 0)
        put(f"cc.{ssp}.gdr.fdays0", g["fdays_u0"]["med"], 0)
        put(f"cc.{ssp}.gdr.fdays1", g["fdays_u1"]["med"], 0)
    allR, allK = [], []
    for ssp in ("ssp245", "ssp585"):
        for k in KEYS:
            if ssp in S[k]:
                e = S[k][ssp]["ensemble"]
                allR += [abs(100 * (e["R_ratio"]["min"] - 1)), abs(100 * (e["R_ratio"]["max"] - 1))]
                allK += [abs(e["dK_pct"]["min"]), abs(e["dK_pct"]["max"])]
    put("cc.Rchg.absmax", max(allR), 1)
    put("cc.dK.absmax", max(allK), 1)
    st = {k: S[k]["stationary"] for k in KEYS}
    rng("cc.stat.R", [st[k]["R"] for k in KEYS], 2)
    rng("cc.stat.benefit", [100 * st[k]["benefit"] for k in KEYS], 1)
    rng("cc.stat.Gst", [st[k]["G_st"] for k in KEYS], 2)
    rng("cc.stat.dR.ledger", [100 * abs(st[k]["R"] / json.loads((RES / f"ledger_{TAG}.json").read_text())[k]
                                         ["reference"]["static"]["R"] - 1) for k in KEYS], 1)
    put("cc.stat.gdr.frost", st["goodridge"]["frost_u0"], 2)


def tables(sites, th, st, lg, ds, kj):
    # ---------------------------------------------------------------- sites
    rows = []
    for k in KEYS:
        s = sites[k]
        rows.append(f"{s['name']}, {s['state']} & {C.SITE_SHORT[k]} & {s['wban']} & {s['koppen']} & "
                    f"{s['lat']:.2f} & {s['lon']:.2f} & {s['elev']:.0f} & {s['maat']:.1f} & {s['fi']:.0f} & "
                    f"{s['precip']:.0f} & {s['sw']:.0f} & {s['kt']:.2f} & {-kj['sites'][k]['K_aprp_snowmasked']:.2f} \\\\")
    (OUT / "tab_sites_rows.tex").write_text("\n".join(rows) + "\n")
    # ---------------------------------------------------------------- per-site results
    rows = []
    for k in KEYS:
        a0, a1 = th[k]["albedo"]["0.14"], th[k]["albedo"]["0.45"]
        r = lg[k]["reference"]
        fr = f"{a0['frost_max']:.2f}$\\rightarrow${a1['frost_max']:.2f}" if a0["frost_max"] > 0.01 else "--"
        rows.append(
            f"{C.SITE_SHORT[k]} & {a1['ts_mean'] - a0['ts_mean']:.2f} & {th[k]['dT_depth_mean'][11]:.2f} & {fr} & "
            f"{st[k]['albedo']['0.45']['life_fatigue']:.1f} & {r['L_eff']:.2f} & "
            f"{r['static']['G_rf']:.1f} & {r['static']['G_co']:.1f} & {r['static']['G_st']:.2f} & "
            f"{r['static']['G_net']:.1f} & {r['dynamic']['G_net']:.1f} & {r['gtp']['G_net']:.1f} & "
            f"{r['dynamic']['R']:.2f} & {ds['sites'][k]['ref']['dynamic']['g_co_star']:.2f} \\\\")
    (OUT / "tab_results_rows.tex").write_text("\n".join(rows).replace("-", "\\vminus ").replace("\\vminus \\vminus ", "--") + "\n")
    supplementary(sites, th, st, lg, ds, kj)


def neg(x):
    return x.replace("-", "\\vminus ")


def supplementary(sites, th, st, lg, ds, kj):
    al = ["0.05", "0.14", "0.30", "0.45", "0.65"]
    # S1: response by albedo
    rows = []
    for k in KEYS:
        a = th[k]["albedo"]
        c = [f"{a[x]['ts_mean']:.1f}" for x in al]
        c += [f"{a[x]['frost_max']:.2f}" if a["0.14"]["frost_max"] > 0.01 else "--" for x in al]
        c += [f"{st[k]['albedo'][x]['life_fatigue']:.1f}" for x in al]
        rows.append(f"{C.SITE_SHORT[k]} & " + " & ".join(c) + " \\\\")
    (OUT / "tab_s1_rows.tex").write_text(neg("\n".join(rows)) + "\n")
    # S2: ledger under the four metrics
    rows = []
    for k in KEYS:
        r = lg[k]["reference"]
        c = []
        for m in ("static", "dynamic", "gtp", "stock"):
            c += [f"{r[m]['G_rf']:.1f}", f"{r[m]['G_co']:.1f}", f"{r[m]['G_st']:.2f}", f"{r[m]['G_net']:.1f}"]
        rows.append(f"{C.SITE_SHORT[k]} & " + " & ".join(c) + " \\\\")
    (OUT / "tab_s2_rows.tex").write_text(neg("\n".join(rows)) + "\n")
    # S3: kernels and coefficients
    rows = []
    for k in KEYS:
        q = kj["sites"][k]
        c = [f"{-q['K_aprp_hourly']:.2f}", f"{-q['K_aprp_snowmasked']:.2f}", f"{-q['K_aprp_monthlymeans']:.2f}",
             f"{-q['K_simple']:.2f}", f"{-q['K_clear']:.2f}",
             f"{-q['K_cmip6_mean']:.2f} ({-q['K_cmip6_max']:.2f}--{-q['K_cmip6_min']:.2f})"]
        co = lg[k]["coefficient"]
        c += [f"{co['static']:.2f}", f"{co['stock_af055']:.2f}", f"{co['gtp_sustained_80']:.2f}",
              f"{lg[k]['kappa']['100']:.2f}"]
        rows.append(f"{C.SITE_SHORT[k]} & " + " & ".join(c) + " \\\\")
    (OUT / "tab_s3_rows.tex").write_text("\n".join(rows) + "\n")
    # S4: Sobol total-effect indices, design B, dynamic GWP100
    sb = json.loads((RES / f"sobol_{TAG}.json").read_text())["results"]["B"]
    names = sb["durham"]["dynamic"]["names"]
    rows = []
    for k in KEYS:
        stv = sb[k]["dynamic"]["ST"]
        c = [f"{max(v, 0.0):.2f}" for v in stv]
        rows.append(f"{C.SITE_SHORT[k]} & " + " & ".join(c) + f" & {100 * sb[k]['dynamic']['p_source']:.0f} \\\\")
    (OUT / "tab_s4_rows.tex").write_text("\n".join(rows) + "\n")
    # S5: urban-form factor by site, orientation and aspect ratio
    q = RES / f"canyon_{TAG}.json"
    if q.exists():
        cy = json.loads(q.read_text())
        h = np.array(cy["h"])
        js = [int(np.argmin(np.abs(h - x))) for x in (0.25, 0.5, 1.0, 2.0)]
        j1 = int(np.argmin(np.abs(h - 1.0)))
        rows = []
        for k in KEYS:
            sc = cy["sites"][k]
            c = [f"{sc[o]['psi'][j]:.2f}" for o in ("EW", "NS", "diag") for j in js]
            c += [f"{sc['EW_by_aw']['0.10'][j1]:.2f}", f"{sc['EW_by_aw']['0.60'][j1]:.2f}",
                  f"{sc['R_open']:.2f}", f"{sc['R_open'] / sc['EW']['psi'][j1]:.1f}"]
            rows.append(f"{C.SITE_SHORT[k]} & " + " & ".join(c) + " \\\\")
        (OUT / "tab_s5_rows.tex").write_text("\n".join(rows) + "\n")
    # S7: state means over the contiguous United States (two blocks side by side)
    q = RES / f"conus_states_{TAG}.json"
    if q.exists():
        stt = json.loads(q.read_text())
        keys = list(stt)
        half = (len(keys) + 1) // 2

        def cells_of(k):
            v = stt[k]
            return [k, f"{v['n']}", f"{-v['K']:.2f}", f"{v['R_static']:.2f}", f"{v['R_dynamic']:.2f}",
                    f"{v['gstar_dynamic']:.2f}", f"{v['Tstar']:.1f}", f"{v['pen_oct']:.0f}"]
        rows = []
        for i in range(half):
            left = cells_of(keys[i])
            right = cells_of(keys[i + half]) if i + half < len(keys) else [""] * 8
            rows.append(" & ".join(left + right) + " \\\\")
        (OUT / "tab_s7_rows.tex").write_text("\n".join(rows) + "\n")
    # S6: projected climates by site
    q = RES / f"climate_{TAG}.json"
    if q.exists():
        cj = json.loads(q.read_text())["sites"]
        rows = []
        for k in KEYS:
            sc = cj[k]
            st0 = sc["stationary"]
            e2, e5 = sc["ssp245"]["ensemble"], sc["ssp585"]["ensemble"]
            fr = (f"{e5['frost_u0']['med']:.2f}$\\rightarrow${e5['frost_u1']['med']:.2f}"
                  if st0["frost_u0"] > 0.01 else "--")
            c = [f"{e2['dta']['med']:.1f}", f"{e5['dta']['med']:.1f}", fr,
                 f"{e2['dDu_pct']['med']:.0f}", f"{e5['dDu_pct']['med']:.0f}",
                 f"{100 * st0['benefit']:.1f}", f"{100 * e5['benefit']['med']:.1f}",
                 f"{e2['dK_pct']['med']:.1f}", f"{e5['dK_pct']['med']:.1f}",
                 f"{st0['G_st']:.2f}", f"{e5['G_st']['med']:.2f}",
                 f"{st0['R']:.2f}", f"{e2['R']['med']:.2f}", f"{e5['R']['med']:.2f}"]
            rows.append(f"{C.SITE_SHORT[k]} & " + " & ".join(c) + " \\\\")
        txt = "\n".join(rows).replace("-0.0 ", "0.0 ").replace("-0.00 ", "0.00 ")
        (OUT / "tab_s6_rows.tex").write_text(neg(txt).replace("\\vminus \\vminus ", "--") + "\n")


if __name__ == "__main__":
    main()

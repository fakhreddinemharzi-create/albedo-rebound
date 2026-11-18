"""Numbers and table rows of the analyses added in revision (discrete
structural renewals, Monte Carlo over streets, distortion of the gridded
rebound ratio), generated from the result files.

Writes
  manuscript/generated/numbers_rev.tex   \\defval{key}{value}
  manuscript/generated/tab_s9_rows.tex   discrete renewals by site
  manuscript/generated/tab_urban_rows.tex streets in canyons and warmer cities
  manuscript/generated/tab_sv_rows.tex   pavement-structure variants
  manuscript/generated/tab_mc_rows.tex   national statistics of the Monte Carlo
  manuscript/generated/tab_dist_rows.tex distortion factors
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import palette as C  # noqa: E402
from alrb.config import MS, RES, SITES  # noqa: E402

TAG = "v3"
OUT = MS / "generated"
OUT.mkdir(parents=True, exist_ok=True)
V = {}
KEYS = C.SITE_ORDER
CODE = C.SITE_CODE if hasattr(C, "SITE_CODE") else None


def f(x, nd=2):
    s = f"{x:.{nd}f}"
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "\\vminus ") if s.startswith("-") else s


def put(k, v, nd=2):
    V[k] = f(v, nd) if isinstance(v, (int, float, np.floating)) else str(v)


def pct(x, nd=0):
    """Share in per cent; values that would round to 100 without being 100 are written as > 99.9."""
    p = 100.0 * x
    if p >= 100.0 - 1e-9:
        return "100"
    if p > 99.95:
        return "$>$99.9"
    if p >= 90.0 or nd > 0:
        return f"{p:.1f}"
    return f"{p:.0f}"


def pct_mc(x, mode="round"):
    """Share in whole per cent for Monte Carlo statistics (their precision); ``floor`` for lower
    bounds and ``ceil`` for upper bounds, so that a rounded bound is never stronger than the value."""
    p = 100.0 * x
    if p >= 100.0 - 1e-9:
        return "100"
    if mode == "floor":
        return f"{int(np.floor(p + 1e-9))}"
    if mode == "ceil":
        return f"{min(100, int(np.ceil(p - 1e-9)))}"
    return "$>$99" if p >= 99.5 else f"{p:.0f}"


def load(name):
    return json.loads((RES / name).read_text())


def site_codes():
    rows = (OUT / "tab_s8_rows.tex").read_text().splitlines()
    codes = [r.split("&")[0].strip() for r in rows if "&" in r]
    return dict(zip(KEYS, codes))


def main():
    lg = load(f"ledger_{TAG}.json")
    ds = load(f"discrete_structure_{TAG}.json")
    mc = load(f"conus_mc_{TAG}.json")
    # ------------------------------------------------------------------ structural term in the burden
    sh = [100 * abs(lg[k]["reference"]["dynamic"]["G_st"]) / lg[k]["reference"]["dynamic"]["G_co"] for k in KEYS]
    put("rev.st.share.min", min(sh), 1)
    put("rev.st.share.max", max(sh), 1)
    # ------------------------------------------------------------------ discrete renewals
    g9 = {k: ds[k]["g9"] for k in KEYS}
    g70 = {k: ds[k]["g70"] for k in KEYS}
    ident = max(max(abs(g9[k]["prorata"]["static"][q] - g9[k]["flow"]["static"]) for q in ("min", "max")) for k in KEYS)
    assert ident < 1e-9, ident                      # pro rata = flow in the static convention, every age
    put("dsc.pr.dyn.dev", max(100 * abs(g9[k]["prorata"]["dynamic"]["mean"] / g9[k]["flow"]["dynamic"] - 1) for k in KEYS), 1)
    put("dsc.pr.dyn.dR.lo", -min(g9[k]["prorata"]["dynamic"]["dR_min_pct"] for k in KEYS), 1)
    put("dsc.pr.dyn.dR.hi", max(g9[k]["prorata"]["dynamic"]["dR_max_pct"] for k in KEYS), 1)
    put("dsc.pr.gtp.dR.lo", -min(g9[k]["prorata"]["gtp"]["dR_min_pct"] for k in KEYS), 1)
    put("dsc.pr.gtp.dR.hi", max(g9[k]["prorata"]["gtp"]["dR_max_pct"] for k in KEYS), 1)
    put("dsc.cf.dyn.dev", max(100 * abs(g9[k]["cashflow"]["dynamic"]["mean"] / g9[k]["flow"]["dynamic"] - 1) for k in KEYS), 1)
    # signed offsets of the mean discrete term from the flow (systematic, the same sign at every site)
    for m, mk in (("dynamic", "dyn"), ("gtp", "gtp")):
        dv = [100 * (g9[k]["prorata"][m]["mean"] / g9[k]["flow"][m] - 1) for k in KEYS]
        put(f"dsc.pr.{mk}.off.lo", min(dv), 1)
        put(f"dsc.pr.{mk}.off.hi", max(dv), 1)
        put(f"dsc.pr.{mk}.off.abs", max(abs(x) for x in dv), 1)
    put("dsc.cf.all.dev", max(100 * abs(g9[k]["cashflow"][m]["mean"] / g9[k]["flow"][m] - 1)
                              for k in KEYS for m in ("static", "dynamic", "gtp")), 1)
    put("dsc.cf.share.min", min(100 * g9[k]["cashflow"]["dynamic"]["share_avoided"] for k in KEYS), 0)
    put("dsc.cf.share.max", max(100 * g9[k]["cashflow"]["dynamic"]["share_avoided"] for k in KEYS), 0)
    put("dsc.cf.dyn.dR.lo", -min(g9[k]["cashflow"]["dynamic"]["dR_min_pct"] for k in KEYS), 0)
    put("dsc.cf.dyn.dR.hi", max(g9[k]["cashflow"]["dynamic"]["dR_max_pct"] for k in KEYS), 0)
    assert all(g9[k][mode][m]["R_min"] > 1 for k in KEYS for mode in ("prorata", "cashflow") for m in ("static", "dynamic", "gtp"))
    put("dsc.g70.cf.sink.min", min(100 * g70[k]["cashflow"]["dynamic"]["share_sink"] for k in KEYS), 0)
    put("dsc.g70.cf.sink.max", max(100 * g70[k]["cashflow"]["dynamic"]["share_sink"] for k in KEYS), 0)
    assert all(abs(g70[k]["cashflow"]["dynamic"]["share_sink"] - g9[k]["cashflow"]["dynamic"]["share_avoided"]) < 1e-9 for k in KEYS)
    assert all(g70[k]["prorata"][m]["R_min"] > 1 for k in KEYS for m in ("static", "dynamic", "gtp"))
    put("dsc.g70.pr.dR.lo", -min(g70[k]["prorata"]["dynamic"]["dR_min_pct"] for k in KEYS), 1)
    put("dsc.g70.pr.dR.hi", max(g70[k]["prorata"]["dynamic"]["dR_max_pct"] for k in KEYS), 1)
    # latest date, after the horizon, of a renewal that the treatment moves out of the analysis period
    slip = 0.0
    for k in KEYS:
        Le = ds[k]["L_eff"]
        for a0 in np.linspace(0, 20, 2001)[:-1]:
            for j in range(4):
                tu, tt = (20 - a0) + 20 * j, (1 - a0 / 20) * Le + Le * j
                if tu < 50 <= tt:
                    slip = max(slip, tt - 50)
    put("dsc.slip.max", np.ceil(slip), 0)
    # delay of the renewals of the treated street within the analysis period (yr)
    delay = [((1 - a0 / 20) * ds[k]["L_eff"] + ds[k]["L_eff"] * j) - ((20 - a0) + 20 * j)
             for k in KEYS for a0 in np.linspace(0, 20, 2001)[:-1] for j in range(4) if (20 - a0) + 20 * j < 50]
    put("dsc.delay.max", np.ceil(10 * max(delay)) / 10, 1)                     # rounded up: an upper bound
    codes = site_codes()
    rows = []
    for k in KEYS:
        o = g9[k]
        pr, cf = o["prorata"]["dynamic"], o["cashflow"]["dynamic"]
        rows.append(f"{codes[k]} & {ds[k]['L_eff']:.2f} & {f(o['flow']['static'])} & {f(o['flow']['dynamic'])} & "
                    f"{f(pr['mean'])} & {f(pr['min'])} to {f(pr['max'])} & {f(pr['dR_min_pct'], 1)} to {f(pr['dR_max_pct'], 1)} & "
                    f"{f(cf['mean'])} & {f(cf['min'])} to {f(cf['max'])} & {f(cf['dR_min_pct'], 0)} to {f(cf['dR_max_pct'], 0)} & "
                    f"{100 * cf['share_avoided']:.0f} \\\\")
    (OUT / "tab_s9_rows.tex").write_text("\n".join(rows) + "\n")
    # ------------------------------------------------------------------ streets in canyons and warmer cities
    ul = load(f"urban_ledger_{TAG}.json")
    cfgs = [n for n in ul[KEYS[0]] if n != "check"]
    put("urb.check.ledger", max(max(ul[k]["check"].values()) for k in KEYS) * 100, 2)
    uc = load(f"urban_column_{TAG}.json")
    for n in cfgs:
        nk = n.replace(".", "p")
        arr = lambda q: [(-ul[k][n]["dTs"] if q == "cool" else ul[k][n][q]) for k in KEYS]  # noqa: E731
        for q, nd in (("cool", 1), ("dTsmax", 1), ("st_factor", 2), ("life45", 2), ("psi", 2), ("sw14", 0), ("ts14", 1),
                      ("frost14", 2), ("dfrost", 2)):
            a = np.array(arr(q), float)
            put(f"urb.{nk}.{q}.min", a.min(), nd)
            put(f"urb.{nk}.{q}.max", a.max(), nd)
            put(f"urb.{nk}.{q}.med", float(np.median(a)), nd)
        put(f"urb.{nk}.lifepct.min", 100 * (min(arr("life45")) - 1), 0)
        put(f"urb.{nk}.lifepct.max", 100 * (max(arr("life45")) - 1), 0)
        for m, mn in (("static", "st"), ("dynamic", "dyn"), ("gtp", "gtp")):
            R = np.array([ul[k][n][m]["R"] for k in KEYS])
            Rlb = np.array([ul[k][n]["R_open_structure"][m] for k in KEYS])
            Gst = np.array([ul[k][n][m]["G_st"] for k in KEYS])
            put(f"urb.{nk}.R.{mn}.min", R.min(), 2)
            put(f"urb.{nk}.R.{mn}.max", R.max(), 2)
            put(f"urb.{nk}.R.{mn}.med", float(np.median(R)), 2)
            put(f"urb.{nk}.Rlb.{mn}.min", Rlb.min(), 2)
            put(f"urb.{nk}.Rlb.{mn}.max", Rlb.max(), 2)
            put(f"urb.{nk}.dRlb.{mn}", 100 * float(np.max(np.abs(R / Rlb - 1))), 1)
            put(f"urb.{nk}.Gst.{mn}.min", -Gst.max(), 2)
            put(f"urb.{nk}.Gst.{mn}.max", -Gst.min(), 2)
        Ro = np.array([ul[k]["open"]["dynamic"]["R"] for k in KEYS])
        Rn = np.array([ul[k][n]["dynamic"]["R"] for k in KEYS])
        put(f"urb.{nk}.Rfac.min", (Rn / Ro).min(), 2)
        put(f"urb.{nk}.Rfac.max", (Rn / Ro).max(), 2)
        put(f"urb.{nk}.dRpct", 100 * float(np.max(np.abs(Rn / Ro - 1))), 1)
    for a, b in (("urban0.5", "h0.5"), ("urban1", "h1")):
        put(f"urb.{a.replace('.', 'p')}.dRvsgeom", 100 * max(abs(ul[k][a]["dynamic"]["R"] / ul[k][b]["dynamic"]["R"] - 1)
                                                         for k in KEYS), 1)
    put("urb.wind.h0p5", ul[KEYS[0]]["urban0.5"]["wind"], 2)
    put("urb.wind.h1", ul[KEYS[0]]["urban1"]["wind"], 2)
    # validation of the canyon floor forcing against the fate curve (from 85_urban_column / alrb.urban)
    fc = load(f"canyon_fatecurve_{TAG}.json")
    put("urb.check.floor", max(ul["validation"]["max_abs_pct"], 0.01), 2)
    proxy = np.interp([0.25, 0.5, 1.0, 2.0], [0.0] + fc["h"], [1.0] + fc["floor"])
    for q, (n, pr) in enumerate(zip(("h0p25", "h0p5", "h1", "h2"), proxy)):
        put(f"urb.{n}.proxy", pr, 2)
    rows = []
    lab = {"open": ("Open street (reference)", "0", "--", "--"), "h0.25": ("Canyon", "0.25", "--", "--"),
           "h0.5": ("Canyon", "0.5", "--", "--"), "h1": ("Canyon", "1", "--", "--"), "h2": ("Canyon", "2", "--", "--"),
           "uhi3": ("Open street, warmer city", "0", "--", "3"),
           "urban0.5": ("Canyon, sheltered, warmer city", "0.5", f"{ul[KEYS[0]]['urban0.5']['wind']:.2f}", "3"),
           "urban1": ("Canyon, sheltered, warmer city", "1", f"{ul[KEYS[0]]['urban1']['wind']:.2f}", "3")}
    def rng_(a, nd):
        lo, hi = f(min(a), nd), f(max(a), nd)
        return lo if lo == hi else f"{lo}{' to ' if min(a) < 0 else '--'}{hi}"
    for n in ("open", "h0.25", "h0.5", "h1", "h2", "uhi3", "urban0.5", "urban1"):
        L, hh, wd, dt = lab[n]
        g = lambda q: [ul[k][n][q] for k in KEYS]  # noqa: E731
        rows.append(f"{L} & {hh} & {wd} & {dt} & {rng_(g('psi'), 2)} & {rng_(g('sw14'), 0)} & {rng_([-x for x in g('dTs')], 1)} & "
                    f"{rng_([100 * (x - 1) for x in g('life45')], 0)} & {rng_(g('st_factor'), 2)} & "
                    f"{rng_([ul[k][n]['static']['R'] for k in KEYS], 2)} & {rng_([ul[k][n]['dynamic']['R'] for k in KEYS], 2)} & "
                    f"{rng_([ul[k][n]['R_open_structure']['dynamic'] for k in KEYS], 2)} \\\\")
    (OUT / "tab_urban_rows.tex").write_text("\n".join(rows) + "\n")
    # ------------------------------------------------------------------ pavement-structure variants
    sv = load(f"structure_variants_{TAG}.json")
    names = sv["variants"]
    allf = np.array([[sv["sites"][k]["factor"][n] for n in names] for k in KEYS])
    put("sv.factor.min", allf.min(), 2)
    put("sv.factor.max", allf.max(), 2)
    put("sv.n", len(names), 0)
    # effect on the dynamic rebound ratio: the structural term changes by (factor - 1) times its reference value
    dR = np.array([[100 * (sv["sites"][k]["factor"][n] - 1) * lg[k]["reference"]["dynamic"]["G_st"] /
                    (lg[k]["reference"]["dynamic"]["G_co"] + lg[k]["reference"]["dynamic"]["G_st"]) for n in names] for k in KEYS])
    put("sv.dR.max", float(np.max(np.abs(dR))), 1)
    lab = {"sg_weak": "Weak subgrade, $M_R$ 35\\,MPa", "sg_stiff": "Stiff subgrade, $M_R$ 140\\,MPa",
           "gran_weak": "Granular layers $\\times$0.6", "gran_stiff": "Granular layers $\\times$1.6",
           "mix_stiff": "Asphalt, 4\\,\\% air voids", "mix_soft": "Asphalt, 9\\,\\% air voids",
           "ac_thin": "Asphalt 50\\,mm", "ac_thick": "Asphalt 200\\,mm",
           "wt_shallow": "Water table at 1.5\\,m", "wt_deep": "Water table at 6\\,m"}
    rows = [f"Reference structure & {rng_([100 * (sv['sites'][k]['ref']['life45'] - 1) for k in KEYS], 0)} & "
            f"{rng_([sv['sites'][k]['ref']['L_eff'] for k in KEYS], 2)} & 1 & 0 \\\\"]
    for q, n in enumerate(names):
        rows.append(f"{lab[n]} & {rng_([100 * (sv['sites'][k]['life45'][n] - 1) for k in KEYS], 0)} & "
                    f"{rng_([sv['sites'][k]['L_eff'][n] for k in KEYS], 2)} & {rng_(allf[:, q], 2)} & "
                    f"{rng_(dR[:, q], 1)} \\\\")
        put(f"sv.{n}.min", allf[:, q].min(), 2)
        put(f"sv.{n}.max", allf[:, q].max(), 2)
        lf = [100 * (sv["sites"][k]["life45"][n] - 1) for k in KEYS]
        put(f"sv.lf45.{n.replace('_', '')}.min", min(lf), 0)
        put(f"sv.lf45.{n.replace('_', '')}.max", max(lf), 0)
    (OUT / "tab_sv_rows.tex").write_text("\n".join(rows) + "\n")
    # ------------------------------------------------------------------ Monte Carlo
    p1 = mc["part1"]
    put("mc.nprod", p1["B"]["n_products"], 0)
    put("mc.nper", p1["B"]["n_streets"] / p1["B"]["n_products"] / p1["B"]["n_cells"], 0)
    put("mc.nstreets", p1["B"]["n_streets"] / 1e6, 1)
    put("mc.ntotal", (p1["B"]["n_streets"] + p1["A"]["n_streets"]) / 1e6, 0)
    MN = {"dynamic": "dyn", "static": "st", "gtp": "gtp"}
    XN = {"lcz": "lcz", "lcz_e": "lcze", "lcz_nom": "lcznom", "open": "open", "lowrise": "low", "mixed": "mix",
          "compact": "comp"}
    for d in ("A", "B"):
        for m, mn in MN.items():
            for x, xn in XN.items():
                for wn in ("paved", "pop", "area", "sunny"):
                    o = p1[d][x][wn][m]
                    key = f"mc.{d}.{mn}.{xn}" + ("" if wn == "paved" else f".{wn}")
                    for name, arr, nd in (("net", o["net"], 0), ("R", o["R"], 1), ("div", o["net_diverse"], 0),
                                          ("Rs", o["R_streets"], 1)):
                        put(f"{key}.{name}.lo", arr[0], nd)
                        put(f"{key}.{name}.med", arr[2], nd)
                        put(f"{key}.{name}.hi", arr[4], nd)
                    V[f"{key}.p"] = pct_mc(o["p_source"])
                    if "p_source_ci" in o:
                        V[f"{key}.p.cilo"] = pct_mc(o["p_source_ci"][0], "floor")
                        V[f"{key}.p.cihi"] = pct_mc(o["p_source_ci"][1], "ceil")
                        for qi, qn in enumerate(("lo", "med", "hi")):
                            put(f"{key}.net.{qn}.cilo", o["net_ci"][qi][0], 0)
                            put(f"{key}.net.{qn}.cihi", o["net_ci"][qi][1], 0)
                    if "share_streets_source_ci" in o:
                        V[f"{key}.streets.cilo"] = pct_mc(o["share_streets_source_ci"][0], "floor")
                        V[f"{key}.streets.cihi"] = pct_mc(o["share_streets_source_ci"][1], "ceil")
                    if "net_productonly" in o:
                        for qi, qn in ((0, "lo"), (2, "med"), (4, "hi")):
                            put(f"{key}.prodonly.{qn}", o["net_productonly"][qi], 0)
                        V[f"{key}.var.prod"] = f"{100 * o['var_share']['product']:.0f}"
                        V[f"{key}.var.sys"] = f"{100 * o['var_share']['systematic']:.0f}"
                        put(f"{key}.nonpos", 100 * o["share_burden_nonpositive"], 2)
                    if "lnR_variance" in o:
                        lv = o["lnR_variance"]
                        for k2 in ("street", "systematic", "product", "form"):
                            V[f"{key}.lnR.{k2[:4]}"] = f"{100 * lv[k2]:.0f}"
                        put(f"{key}.streetfold", lv["street_fold_med"], 1)
                        put(f"{key}.formfold", lv["form_fold_med"], 1)
                    V[f"{key}.pdiv"] = pct_mc(o["p_source_diverse"])
                    V[f"{key}.streets"] = pct_mc(o["share_streets_source"])
                    if "cells_psource" in o:
                        V[f"{key}.cells.min"] = pct_mc(o["cells_psource"]["min"], "floor")
                        V[f"{key}.cells.med"] = pct_mc(o["cells_psource"]["med"])
                        put(f"{key}.cellfold", o["cellfold_med"], 0)
        dn = [abs(p1[d]["lcz"][wn]["dynamic"]["net"][2] - p1[d]["lcz"]["paved"]["dynamic"]["net"][2]) for wn in ("pop", "area")]
        put(f"mc.{d}.wdnet", max(dn), 1)
        for s, v in p1[d]["by_class"].items():
            sk = s.replace(" ", "").replace("-", "")
            V[f"mc.{d}.cls.{sk}.src"] = pct_mc(v["share_source"])
            put(f"mc.{d}.cls.{sk}.R.med", v["R"][1], 1)
        V[f"mc.{d}.cls.compact.min"] = pct_mc(min(p1[d]["by_class"][c]["share_source"]
                                                 for c in ("compact high-rise", "compact mid-rise", "compact low-rise")), "floor")
    lh = p1["B"]["lcz_h"]
    put("lcz.h.med", lh["q"][2], 2)
    put("lcz.h.p05", lh["q"][0], 2)
    put("lcz.h.p95", lh["q"][4], 2)
    put("lcz.psi.med", lh["psi"][2], 2)
    put("lcz.psi.p05", lh["psi"][0], 2)
    put("lcz.psi.p95", lh["psi"][4], 2)
    sh = mc["lcz_paved_share"]
    # median aspect ratio of the streets under the nominal ranges of the zones (mixture of uniform laws)
    nom = {"compact high-rise": (2.0, 3.0), "compact mid-rise": (0.75, 2.0), "compact low-rise": (0.75, 1.5),
           "open high-rise": (0.75, 1.25), "open mid-rise": (0.3, 0.75), "open low-rise": (0.3, 0.75),
           "large low-rise": (0.1, 0.3), "heavy industry": (0.2, 0.5)}
    hx = np.linspace(0.0, 3.0, 30001)
    cdf = sum(sh[c] * np.clip((hx - a) / (b - a), 0, 1) for c, (a, b) in nom.items()) / sum(sh[c] for c in nom)
    put("lcz.nom.h.med", float(np.interp(0.5, cdf, hx)), 2)
    V["lcz.share.openlow"] = f"{100 * sh['open low-rise']:.1f}"
    V["lcz.share.large"] = f"{100 * sh['large low-rise']:.1f}"
    V["lcz.share.compact"] = f"{100 * (sh['compact high-rise'] + sh['compact mid-rise'] + sh['compact low-rise']):.1f}"
    V["lcz.share.openhighmid"] = f"{100 * (sh['open high-rise'] + sh['open mid-rise']):.1f}"
    V["lcz.share.industry"] = f"{100 * sh['heavy industry']:.1f}"
    lj = load("conus_lcz_v3.json")
    for d in ("A", "B"):
        o = p1[d]["lcz"]["paved"]["dynamic"]
        for qi, qn in ((0, "lo"), (2, "med"), (4, "hi")):
            put(f"macro.{d}.{qn}", o["net"][qi] * 1e8 / 1e9, 1)       # Mt CO2e per 100 km2 of treated street
    V["lcz.acc.oa"] = "80"         # overall accuracy of the map, pixel-based and city hold-out (readme of the data set)
    V["lcz.acc.city"] = "48"
    put("lcz.built.km2", lj["built_km2"] / 1e3, 0)
    put("lcz.paved.km2", lj["paved_km2"] / 1e3, 0)
    put("mc.sys.lo", p1["B"]["sysf"][0], 2)
    put("mc.sys.hi", p1["B"]["sysf"][2], 2)
    z = np.load(RES / f"conus_mc_{TAG}.npz")
    for d in ("A", "B"):
        r50 = z[f"{d}_lcz_R50"]
        put(f"mc.{d}.cellR50.min", r50.min(), 1)
        put(f"mc.{d}.cellR50.max", r50.max(), 1)
    rows = []
    lab = [("lcz", "paved", "LCZ form (reference)"), ("lcz_nom", "paved", "LCZ form, nominal $H/W$"),
           ("lcz_e", "paved", "LCZ form and paved land"), ("lowrise", "paved", "Assumed low-rise mix"),
           ("compact", "paved", "Assumed compact mix"), ("open", "paved", "Open streets only"),
           ("open", "sunny", "Open streets, sunniest cells")]
    q = lambda a, nd=0: f"{f(a[2], nd)} ({f(a[0], nd)} to {f(a[4], nd)})"  # noqa: E731
    for d, dl in (("B", "$T_r=3$\\,yr"), ("A", "$T_r$ 1--10\\,yr")):
        for i, (x, wn, lb_) in enumerate(lab):
            o, s, g = p1[d][x][wn]["dynamic"], p1[d][x][wn]["static"], p1[d][x][wn]["gtp"]
            rows.append(f"{dl if i == 0 else ''} & {lb_} & {pct_mc(o['share_streets_source'])} & {q(o['R_streets'], 1)} & "
                        f"{q(o['net'])} & {pct_mc(o['p_source'])} & {q(o['net_diverse'])} & {q(s['net'])} & "
                        f"{pct_mc(s['p_source'])} & {pct_mc(g['share_streets_source'])} \\\\")
        if d == "B":
            rows.append("\\addlinespace")
    (OUT / "tab_mc_rows.tex").write_text("\n".join(rows) + "\n")
    # ------------------------------------------------------------------ correlated coating properties
    co = mc["correlation"]
    for d in ("A", "B"):
        base = co[d]["independent"]
        coat = {n: c for n, c in co[d].items() if "Tr" not in n}           # correlations among coating properties
        dn = [c["net"][2] - base["net"][2] for n, c in coat.items() if n != "independent"]
        put(f"corr.{d}.dnet.min", min(dn), 1)
        put(f"corr.{d}.dnet.max", max(dn), 1)
        V[f"corr.{d}.p.min"] = pct_mc(min(c["p_source"] for c in coat.values()), "floor")
        V[f"corr.{d}.streets.min"] = pct_mc(min(c["share_streets_source"] for c in coat.values()), "floor")
        V[f"corr.{d}.streets.max"] = pct_mc(max(c["share_streets_source"] for c in coat.values()), "ceil")
        for n, c in co[d].items():
            nk = n.replace("+", "p").replace("-", "m").replace(".", "").replace("_", "")
            put(f"corr.{d}.{nk}.net", c["net"][2], 1)
            put(f"corr.{d}.{nk}.dnet", c["net"][2] - base["net"][2], 1)
            V[f"corr.{d}.{nk}.streets"] = pct_mc(c["share_streets_source"])
            V[f"corr.{d}.{nk}.p"] = pct_mc(c["p_source"])
            put(f"corr.{d}.{nk}.rho", c["rho_sample"], 2)
            put(f"corr.{d}.{nk}.dstreets", 100 * (c["share_streets_source"] - base["share_streets_source"]), 1)
        trc = [c for n, c in co[d].items() if "Tr" in n]
        if trc:
            put(f"corr.{d}.Trtau.dnetabs", max(abs(c["net"][2] - base["net"][2]) for c in trc), 1)
            V[f"corr.{d}.Trtau.p.min"] = pct_mc(min(c["p_source"] for c in trc), "floor")
    # streets sharing their properties within a cell: upper bounds of the change over both designs
    sd = [co[d]["shared_street"] for d in ("A", "B") if "shared_street" in co[d]]
    if sd:
        put("corr.shared.dnet.ub", np.ceil(10 * max(abs(c["net"][2] - co[d]["independent"]["net"][2])
                                                     for d, c in zip(("A", "B"), sd)) + 1e-9) / 10, 1)
        put("corr.shared.dstreets.ub", np.ceil(1000 * max(abs(c["share_streets_source"] - co[d]["independent"]["share_streets_source"])
                                                          for d, c in zip(("A", "B"), sd)) + 1e-9) / 10, 1)
    # ------------------------------------------------------------------ distortion
    di = mc["distortion"]
    xs, qs = z["dist_Tr_x"], z["dist_Tr_q"]
    put("dist.Tr.whole.min", qs[1].min(), 2)
    put("dist.Tr.whole.max", qs[1].max(), 2)
    for name in ("h", "tau", "Tr", "af", "gco"):
        for kk, v in di[name].items():
            key = f"dist.{name}.{kk.replace('.', 'p')}"
            put(key, v["med"], 2)
            put(key + ".lo", v["p05"], 2)
            put(key + ".hi", v["p95"], 2)
            put(key + ".spread", v["spread_pct"], 0)
    put("dist.coat.spread.max", max(v["spread_pct"] for n in ("tau", "Tr", "af", "gco") for v in di[n].values()), 0)
    put("dist.life.max", di["life_station"]["max_abs_pct"], 1)
    put("dist.cal.lo", -di["calibration"]["min_pct"], 0)
    put("dist.cal.hi", di["calibration"]["max_pct"], 0)
    for h in ("0.25", "0.5", "1"):
        hk = h.replace(".", "p")
        ps = di["psi_station"][h]
        put(f"dist.psi.any.{hk}", ps["any_station_max_abs_pct"], 1)
        put(f"dist.psi.any95.{hk}", ps["any_station_p95_abs_pct"], 1)
        put(f"dist.psi.tr.{hk}", ps["transfer_max_abs_pct"], 1)
        put(f"dist.psi.tr95.{hk}", ps["transfer_p95_abs_pct"], 1)
    for h, v in di["psi_regression_resid_pct"].items():
        put(f"dist.psi.resid.{h.replace('.', 'p')}", v, 1)
    put("dist.uhi3.pct", 100 * (di["uhi3"]["med"] - 1), 1)
    put("dist.uhi3.max", 100 * max(abs(di["uhi3"]["min"] - 1), abs(di["uhi3"]["max"] - 1)), 1)
    for name in ("tau", "Tr", "af", "gco"):
        put(f"dist.el.{name}", di["elasticity"][name]["med"], 2)
    put("dist.el.h", di["elasticity"]["h_per_0.1"]["med"], 0)
    jt = di["joint"]
    put("dist.joint.open3.R", jt["0|3"]["R"]["med"], 2)
    V["dist.joint.open3.paved"] = pct(jt["0|3"]["share_paved_source"])
    V["dist.joint.open3.pop"] = pct(jt["0|3"]["share_pop_source"])
    V["dist.joint.open3.sinkpaved"] = pct(1 - jt["0|3"]["share_paved_source"])
    mant, ex = f"{di['check_against_grid_ledger']:.0e}".split("e")
    V["dist.check"] = f"${int(mant)}\\times10^{{{int(ex)}}}$"
    put("dist.joint.q3.R", jt["0.25|3"]["R"]["med"], 2)
    put("dist.joint.q3.Rmin", jt["0.25|3"]["R"]["min"], 2)
    hh, tt = ["0", "0.25", "0.5", "1", "2"], ["0.5", "0.91", "1.5", "3"]
    rows = []
    for b in tt:
        cells = []
        for a in hh:
            v = jt[f"{a}|{b}"]
            cells.append(f"{v['factor']['med']:.2f} & {v['R']['med']:.1f} ({v['R']['p05']:.1f}--{v['R']['p95']:.1f})")
        rows.append(f"{b} & " + " & ".join(cells) + " \\\\")
    (OUT / "tab_dist_rows.tex").write_text("\n".join(rows) + "\n")
    # joint decay time and renewal interval
    for kk, v in di.get("joint_tau_Tr", {}).items():
        b, c = kk.split("|")
        key = f"dist.jt.{b.replace('.', 'p')}.{c.replace('.', 'p')}"
        put(key + ".R", v["R"]["med"], 2)
        put(key + ".Rlo", v["R"]["p05"], 2)
        put(key + ".Rhi", v["R"]["p95"], 2)
        V[key + ".paved"] = pct(v["share_paved_source"])
        V[key + ".sinkpaved"] = pct(1 - v["share_paved_source"])
    # ------------------------------------------------------------------ bounds (88_bounds_revision.py)
    bd = load("bounds_v3.json")
    sp = bd["seasonal_psi"]
    for tau, tk in (("0.91", "ref"), ("0.5", "fast"), ("3", "slow")):
        for o in ("EW", "NS", "diag"):
            for h in ("0.5", "1.0"):
                v = sp[tau][f"{o}|{h}"]
                put(f"bd.ps.{tk}.{o}.{h.replace('.', 'p')}.lo", 100 * (v["min"] - 1), 1)
                put(f"bd.ps.{tk}.{o}.{h.replace('.', 'p')}.hi", 100 * (v["max"] - 1), 1)
    put("bd.ps.ref.NS.max", 100 * (max(sp["0.91"]["NS|0.5"]["max"], sp["0.91"]["NS|1.0"]["max"]) - 1), 1)
    cl = bd["coat_loss"]
    put("bd.coat.per", 100 * cl["frac_per_renewal"], 1)
    put("bd.coat.tot.lo", 100 * cl["frac_total_min"], 1)
    put("bd.coat.tot.hi", 100 * cl["frac_total_max"], 1)
    put("bd.coat.n.lo", cl["renewals_min"], 1)
    put("bd.coat.n.hi", cl["renewals_max"], 1)
    put("bd.coat.extra", 100 * cl["extra_coat_frac"], 0)
    kl = bd["kernel_loo"]
    for k2 in ("loo_min", "loo_max", "loo_rms", "insample_min", "insample_max"):
        put(f"bd.kernel.{k2.replace('_', '')}", 100 * kl[k2], 1)
    for hk, v in bd["strain_tables"].items():
        k3 = f"{int(round(float(hk) * 1000)):03d}"
        put(f"bd.stt.{k3}.t", 100 * v["eps_t_max"], 1)
        put(f"bd.stt.{k3}.v", 100 * v["eps_v_max"], 1)
    # share of the absorption removed from the street that the walls take (fate curve: Durham, east-west)
    fc = load(f"canyon_fatecurve_{TAG}.json")
    hfc = np.array(fc["h"])
    wsh = np.array(fc["walls"]) / np.array(fc["floor"])
    for hv in (0.5, 1.0, 1.5, 2.0):
        put(f"cy.fc.wshare.{f'{hv:g}'.replace('.', 'p')}", 100 * float(np.interp(hv, hfc, wsh)), 0)
    put("cy.fc.wshare.half.h", float(np.interp(0.5, wsh, hfc)), 1)
    put("cy.fc.escape.1", float(np.interp(1.0, hfc, fc["escape"])), 2)
    lines = ["% generated by scripts/83_numbers_revision.py - do not edit"]
    lines += [f"\\defval{{{k}}}{{{v}}}" for k, v in sorted(V.items())]
    (OUT / "numbers_rev.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(V)} values")
    for k in sorted(V):
        if k.startswith(("dsc", "rev", "mc.n", "mc.B.dyn.lcz", "mc.A.dyn.lcz", "mc.B.dyn.open", "mc.A.dyn.open", "dist.el",
                         "dist.h", "dist.joint", "dist.life", "dist.cal", "dist.coat", "dist.psi", "dist.uhi", "mc.B.cls",
                         "mc.A.cls", "mc.B.cell", "mc.A.cell", "mc.B.w", "mc.A.w", "mc.sys", "urb.", "sv.", "corr.", "lcz.")) \
                and ".pop" not in k and ".area" not in k:
            print(f"  {k:28s} {V[k]}")


if __name__ == "__main__":
    main()

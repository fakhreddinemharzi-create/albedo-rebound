"""Monte Carlo over streets, products and systematic uncertainty across the
contiguous United States, and the distortion of the gridded rebound ratio
when the properties of a street deviate from the reference.

The gridded ledger of 67_conus_ledger.py places one reference street (open,
reference coating, structure of the nearest station) in the climate of every
one-degree cell.  Here every cell is populated with streets whose properties
are sampled:

  form      : the urban form of the cell from the Local Climate Zone map of
              Demuzere et al. (2020) (84_lcz_grid.py): an LCZ class is drawn in
              proportion to its paved (impervious, non-roof) area in the cell,
              and the sky-view factor of the street uniformly in the range of
              the class (Stewart and Oke, 2012); the aspect ratio follows from
              the sky-view factor of an infinitely long canyon floor,
              F = sqrt(1 + h^2) - h.  Assumed mixes of aspect-ratio classes
              (open streets only, low-rise, mixed, compact) are kept as
              alternatives.  Orientation and wall albedo are drawn uniformly.
  street    : albedo of the aged surface, emissions of a structural renewal,
              convective transfer, an urban heat-island offset of 0-3 K and a
              pavement structure (85_urban_column.py, 87_structure_variants.py);
  product   : albedo as laid, decay time, residual gain, emissions per coat
              (and the renewal interval, design A), independent or with rank
              correlations (Gaussian copula);
  systematic: efficacy, kernel multiplier and the calibration of the grid
              kernel (bootstrap of the ten station ratios).

The urban-form factor of a street is that of the nearest station at the drawn
geometry (64_canyon.py), transferred to the latitude of the cell with the
latitude dependence fitted across the ten stations.  The structural term of
a street in a canyon or in a warmer city is scaled by the factors simulated
with the canyon forcing (85_urban_column.py; the factor of H/W = 2 is used
for deeper streets).  Cells are weighted by their paved area (reference),
population or cell area, or by the paved area of the sunniest cells that
hold a quarter of it (largest snow-masked kernel); the sensitivity that
counts the paved land of class E as open streets also adds it to the
weights.  Alternative descriptions of the form: the nominal aspect-ratio
ranges of the classes instead of the conversion of their sky-view factor,
and assumed mixes of aspect-ratio classes.

Part 2 moves one property of the reference street at a time, and the aspect
ratio and the decay time together, away from the reference in every cell and
records the factor by which the rebound ratio changes; it also bounds the
nearest-station assumptions.  Part 3 repeats the Monte Carlo with correlated
coating properties, and with the street properties shared by the streets of a
cell.

usage: python 82_conus_mc.py [n_products] [streets_per_cell] [--part3]
Writes results/conus_mc_v3.json and results/conus_mc_v3.npz.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import ledger, ledger_batch as lb  # noqa: E402
from alrb.config import ALBEDO_GRID, REF, RES, SITES  # noqa: E402

TAG = "v3"
_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
NPROD = int(_ARGS[0]) if len(_ARGS) > 0 else 1200
NS = int(_ARGS[1]) if len(_ARGS) > 1 else 5
NBOOT = 2000
METRICS = ("static", "dynamic", "gtp")
NQUAD = 401
ORIENT = ("EW", "NS", "diag")
UHI_MAX = 3.0
# aspect-ratio classes and assumed mixes of treated streets over them
STRATA = [("open", 0.0, 0.0), ("sparse", 0.10, 0.30), ("openrise", 0.30, 0.75), ("compact", 0.75, 2.0),
          ("highrise", 2.0, 3.0)]
MIXES = {
    "open": [1.0, 0.0, 0.0, 0.0, 0.0],
    "lowrise": [0.0, 0.35, 0.55, 0.09, 0.01],
    "mixed": [0.0, 0.20, 0.45, 0.30, 0.05],
    "compact": [0.0, 0.05, 0.25, 0.55, 0.15],
}
# built LCZ classes: sky-view factor range and impervious (non-roof) surface fraction, % (midpoint),
# from Table 1 of Demuzere et al. (2020), after Stewart and Oke (2012); classes 7 and 9 are not mapped
LCZ = {1: (0.2, 0.4, 50.0), 2: (0.3, 0.6, 40.0), 3: (0.2, 0.6, 35.0), 4: (0.5, 0.7, 35.0), 5: (0.5, 0.8, 40.0),
       6: (0.6, 0.9, 35.0), 8: (0.7, 1.0, 45.0), 10: (0.6, 0.9, 30.0)}
LCZ_E = {15: (0.9, 1.0, 95.0)}              # bare rock or paved, as open paved streets (sensitivity)
# nominal aspect-ratio ranges of the classes (Stewart and Oke, 2012, Table 2; class 1, H/W > 2, capped at 3),
# an alternative to the conversion of the sky-view factor (sensitivity)
LCZ_HW = {1: (2.0, 3.0), 2: (0.75, 2.0), 3: (0.75, 1.5), 4: (0.75, 1.25), 5: (0.3, 0.75), 6: (0.3, 0.75),
          8: (0.1, 0.3), 10: (0.2, 0.5)}
LCZ_NAME = {1: "compact high-rise", 2: "compact mid-rise", 3: "compact low-rise", 4: "open high-rise",
            5: "open mid-rise", 6: "open low-rise", 8: "large low-rise", 10: "heavy industry", 15: "paved"}


def svf_to_h(F):
    F = np.clip(F, 1e-6, 1.0)
    return (1.0 - F * F) / (2.0 * F)


def life_tables():
    al = np.array(ALBEDO_GRID)
    tabs = []
    for s in SITES:
        t = []
        for tag in (f"{TAG}_h06", TAG, f"{TAG}_h16"):
            js = json.loads((RES / f"structural_{tag}.json").read_text())[s.key]
            t.append(np.array([js["albedo"][f"{a:.2f}"]["life_ref"] for a in al]) / js["albedo"]["0.14"]["life_ref"] * 20.0)
        tabs.append(t)
    return al, np.array([0.6, 1.0, 1.6]), np.array(tabs)


class Canyon:
    """psi(h, orientation, wall albedo) of every station, transferred to the
    latitude of a cell with the latitude dependence fitted across the
    stations (log psi linear in latitude for every h and orientation)."""

    def __init__(self):
        cy = json.loads((RES / f"canyon_{TAG}.json").read_text())
        sites = json.loads((RES / "sites.json").read_text())
        self.h = np.array(cy["h"])
        self.aw = np.array(cy["aw"])
        self.psi = np.array([[cy["sites"][s.key][o]["psi"] for o in ORIENT] for s in SITES])       # site, orient, h
        byaw = np.array([[cy["sites"][s.key]["EW_by_aw"][f"{a:.2f}"] for a in self.aw] for s in SITES])
        ref = byaw[:, list(np.round(self.aw, 2)).index(round(cy["aw_ref"], 2))]
        with np.errstate(invalid="ignore", divide="ignore"):
            self.awfac = np.where(ref[:, None, :] > 0, byaw / ref[:, None, :], 1.0)                 # site, aw, h
        self.lat = np.array([sites[s.key]["lat"] for s in SITES])
        X = np.column_stack([np.ones(len(SITES)), self.lat - 40.0])
        self.coef = np.zeros((len(ORIENT), len(self.h)))
        self.resid = np.zeros((len(ORIENT), len(self.h)))
        for o in range(len(ORIENT)):
            for j in range(1, len(self.h)):
                y = np.log(self.psi[:, o, j])
                b = np.linalg.lstsq(X, y, rcond=None)[0]
                self.coef[o, j] = b[1]
                self.resid[o, j] = float(np.std(y - X @ b, ddof=2))

    def _ih(self, h):
        j = np.clip(np.searchsorted(self.h, h) - 1, 0, len(self.h) - 2)
        return j, (h - self.h[j]) / (self.h[j + 1] - self.h[j])

    def __call__(self, site, h, orient, aw, dlat=0.0):
        j, w = self._ih(h)
        p = (1 - w) * self.psi[site, orient, j] + w * self.psi[site, orient, j + 1]
        k = np.clip(np.searchsorted(self.aw, aw) - 1, 0, len(self.aw) - 2)
        v = (aw - self.aw[k]) / (self.aw[k + 1] - self.aw[k])
        f = lambda kk: (1 - w) * self.awfac[site, kk, j] + w * self.awfac[site, kk, j + 1]  # noqa: E731
        c = (1 - w) * self.coef[orient, j] + w * self.coef[orient, j + 1]
        return p * ((1 - v) * f(k) + v * f(k + 1)) * np.exp(c * dlat)


class Structure:
    """Factors on the structural term of a street: canyon geometry and urban
    heat island from the simulations of 85_urban_column.py (per site), and the
    pavement-structure variants of 87_structure_variants.py."""

    def __init__(self):
        self.source = "simulated"
        f = RES / "urban_column_v3.json"
        fc = json.loads((RES / f"canyon_fatecurve_{TAG}.json").read_text())
        self.fh_proxy = np.concatenate([[0.0], fc["h"]])
        self.floor_proxy = np.concatenate([[1.0], fc["floor"]])
        if not f.exists():
            self.source = "proxy"
            return
        u = json.loads(f.read_text())
        al = np.array(u["albedos"])
        rate = {}
        for s in SITES:
            rate[s.key] = {}
            for name, cfg in u["sites"][s.key].items():
                L = np.array([min(cfg["albedo"][f"{a:.2f}"]["life_fatigue"], cfg["albedo"][f"{a:.2f}"]["life_subgrade"])
                              for a in al])
                L = L / L[0] * 20.0
                Leff = ledger.harmonic_life(lambda x, L=L: np.interp(x, al, L), ledger.Params())
                rate[s.key][name] = 1.0 / Leff - 1.0 / 20.0
        self.rate = rate
        self.hg = np.array([0.0, 0.25, 0.5, 1.0, 2.0])
        names = ["open", "h0.25", "h0.5", "h1", "h2"]
        self.geom = np.array([[rate[s.key][n] / rate[s.key]["open"] for n in names] for s in SITES])
        self.uhi = np.array([rate[s.key]["uhi3"] / rate[s.key]["open"] for s in SITES])
        self.urban = {n: np.array([rate[s.key][n] / rate[s.key]["open"] for s in SITES]) for n in ("urban0.5", "urban1")}
        v = RES / "structure_variants_v3.json"
        self.variants = None
        if v.exists():
            vj = json.loads(v.read_text())
            self.variant_names = ["reference"] + list(vj["sites"][SITES[0].key]["factor"].keys())
            self.variants = np.array([[1.0] + [vj["sites"][s.key]["factor"][n] for n in self.variant_names[1:]]
                                      for s in SITES])

    def geom_factor(self, site, h):
        if self.source == "proxy":
            return np.interp(h, self.fh_proxy, self.floor_proxy)
        j = np.clip(np.searchsorted(self.hg, h) - 1, 0, len(self.hg) - 2)
        w = np.clip((h - self.hg[j]) / (self.hg[j + 1] - self.hg[j]), 0.0, 1.0)
        return (1 - w) * self.geom[site, j] + w * self.geom[site, j + 1]

    def uhi_factor(self, site, dT):
        if self.source == "proxy":
            return np.ones_like(dT)
        return 1.0 + (self.uhi[site] - 1.0) * dT / UHI_MAX

    def variant_factor(self, site, rng, ns_shared=None):
        if self.variants is None:
            return np.ones(len(site))
        if ns_shared:                                   # one structure for all streets of a cell and realisation
            k = np.repeat(rng.integers(0, self.variants.shape[1], len(site) // ns_shared), ns_shared)
        else:
            k = rng.integers(0, self.variants.shape[1], len(site))
        return self.variants[site, k]


def wquant(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return np.interp(np.asarray(q) * c[-1], c, x[o])


def draw_products(rng, n, design, corr=None):
    """Coating products and systematic factors; ``corr`` maps a pair of
    (af, tau, gco, Tr) to a Spearman rank correlation (Gaussian copula; the
    renewal interval is fixed at three years in design B)."""
    names = ["af", "tau", "gco", "Tr"]
    R = np.eye(4)
    for (a, b), rs in (corr or {}).items():
        i, j = names.index(a), names.index(b)
        R[i, j] = R[j, i] = 2.0 * np.sin(np.pi * rs / 6.0)
    u = norm.cdf(rng.standard_normal((n, 4)) @ np.linalg.cholesky(R).T)
    P = dict(af=0.25 + 0.35 * u[:, 0], tau=10 ** (np.log10(0.5) + (np.log10(3.0) - np.log10(0.5)) * u[:, 1]),
             gco=1.63 + (3.39 - 1.63) * u[:, 2], resid=rng.uniform(0.0, 0.05, n),
             Tr=np.full(n, 3.0) if design == "B" else 1.0 + 9.0 * u[:, 3],
             eff=rng.uniform(0.60, 1.10, n), kmult=rng.uniform(0.85, 1.15, n))
    return P


class Setup:
    def __init__(self):
        z = np.load(RES / "conus_kernel_v3.npz")
        zl = np.load(RES / f"conus_ledger_{TAG}.npz")
        cj = json.loads((RES / "conus_kernel_v3.json").read_text())["sites"]
        self.ratios = np.array([cj[s.key]["station_Km"] / cj[s.key]["grid_Km"] for s in SITES])
        self.ratio = float(np.median(self.ratios))
        self.lat, self.lon = zl["lat"], zl["lon"]
        ok = zl["land"] & np.isfinite(zl["R_static"])
        self.ok = ok
        self.ia, self.ib = np.where(ok)
        self.nc = len(self.ia)
        self.near = zl["near"][self.ia, self.ib].astype(np.int64)
        self.zl = zl
        days = pd.date_range("2017-05-01", "2024-04-30", freq="D")
        self.Kd = 100.0 * self.ratio * z["Kmon"][:, self.ia, self.ib][days.month.to_numpy() - 1].T
        slat = np.array([s.lat for s in SITES])
        self.dlat = self.lat[self.ia] - slat[self.near]
        # land-cover areas, with those of cells without a ledger moved to the nearest ledger cell
        lz = np.load(RES / "conus_lcz_v3.npz")
        area = lz["area"]
        A = area[:, self.ia, self.ib].copy()
        oa, ob = np.where(~ok & (area[1:11].sum(0) > 0))
        for a, b in zip(oa, ob):
            d = (self.lat[self.ia] - self.lat[a]) ** 2 + ((self.lon[self.ib] - self.lon[b]) * np.cos(np.radians(self.lat[a]))) ** 2
            A[:, int(np.argmin(d))] += area[:, a, b]
        self.lcz_area = A                                                     # (18, nc), km2
        pop = np.load(RES / "conus_population_v3.npz")["pop"][self.ia, self.ib]
        paved = sum(A[c] * LCZ[c][2] / 100.0 for c in LCZ)
        paved_e = paved + A[15] * LCZ_E[15][2] / 100.0
        self.W = {"paved": paved / paved.sum(), "paved_e": paved_e / paved_e.sum(), "pop": pop / pop.sum(),
                  "area": np.cos(np.radians(self.lat[self.ia])) / np.cos(np.radians(self.lat[self.ia])).sum()}
        # deployment in the sunniest quarter of the paved area (largest snow-masked kernel)
        Kabs = np.abs(self.Kd.mean(axis=1))
        o = np.argsort(-Kabs)
        cum = np.cumsum(self.W["paved"][o])
        sunny = np.zeros(self.nc, bool)
        sunny[o[: int(np.searchsorted(cum, 0.25)) + 1]] = True
        self.W["sunny"] = np.where(sunny, paved, 0.0) / np.sum(np.where(sunny, paved, 0.0))
        self.sunny = sunny
        self.al, self.hs, self.Ltab = life_tables()
        self.cy = Canyon()
        self.st = Structure()

    def class_probs(self, with_e=False):
        cls = dict(LCZ)
        if with_e:
            cls.update(LCZ_E)
        keys = list(cls)
        pav = np.array([self.lcz_area[c] * cls[c][2] / 100.0 for c in keys]).T          # (nc, ncls)
        tot = pav.sum(1, keepdims=True)
        p = np.where(tot > 0, pav / np.maximum(tot, 1e-12), 0.0)
        p[tot[:, 0] == 0, keys.index(6)] = 1.0                                          # no built area: open low-rise
        return keys, cls, p


def draw_form(rng, setup, morph, cell):
    """Aspect ratio of every street under a morphology."""
    n = len(cell)
    if morph in MIXES:
        shares = np.array(MIXES[morph])
        s = rng.choice(len(STRATA), n, p=shares / shares.sum())
        lo = np.array([x[1] for x in STRATA])[s]
        hi = np.array([x[2] for x in STRATA])[s]
        return lo + (hi - lo) * rng.random(n), s
    keys, cls, p = setup.class_probs(with_e=(morph == "lcz_e"))
    cum = np.cumsum(p, axis=1)
    k = np.minimum((rng.random(n)[:, None] > cum[cell]).sum(1), len(keys) - 1)
    if morph == "lcz_nom":                                     # nominal aspect ratios of the classes
        lo = np.array([LCZ_HW[c][0] for c in keys])[k]
        hi = np.array([LCZ_HW[c][1] for c in keys])[k]
        return lo + (hi - lo) * rng.random(n), np.array(keys)[k]
    lo = np.array([cls[c][0] for c in keys])[k]
    hi = np.array([cls[c][1] for c in keys])[k]
    return svf_to_h(lo + (hi - lo) * rng.random(n)), np.array(keys)[k]


def evaluate_streets(setup, rng, design, nprod, ns, corr=None, shared=False):
    """``shared``: the street properties other than form and orientation are
    drawn once per cell and realisation (perfect correlation within a cell)."""
    P = draw_products(rng, nprod, design, corr)
    cal = np.array([np.median(rng.choice(setup.ratios, len(setup.ratios))) for _ in range(nprod)]) / setup.ratio
    sysf = P["eff"] * P["kmult"] * cal
    nc = setup.nc
    shape = (nprod, nc, ns)
    n = int(np.prod(shape))
    Pi, Ci, _ = np.meshgrid(np.arange(nprod), np.arange(nc), np.arange(ns), indexing="ij")
    Pi, Ci = Pi.ravel(), Ci.ravel()
    draw = (lambda f: np.repeat(f(nprod * nc), ns)) if shared else (lambda f: f(n))  # noqa: E731
    a0 = draw(lambda k: rng.uniform(0.09, 0.18, k))
    gst = draw(lambda k: 10 ** rng.uniform(np.log10(6.0), np.log10(70.0), k))
    hm = draw(lambda k: rng.uniform(0.60, 1.60, k))
    site = setup.near[Ci]
    r = lb.evaluate(setup.Kd, Ci, a0, P["af"][Pi], a0 + P["resid"][Pi], P["tau"][Pi], P["Tr"][Pi], P["gco"][Pi], gst,
                    hm, site, setup.al, setup.hs, setup.Ltab, nquad=NQUAD)
    street = dict(dT=draw(lambda k: rng.uniform(0.0, UHI_MAX, k)), orient=rng.integers(0, 3, n),
                  aw=draw(lambda k: rng.uniform(0.10, 0.60, k)),
                  fvar=setup.st.variant_factor(site, rng, ns if shared else None))
    return dict(P=P, sysf=sysf, shape=shape, Pi=Pi, Ci=Ci, site=site, r=r, street=street)


def terms(setup, E, h):
    """Credit (> 0) and burden of every street for the aspect ratios h."""
    s = E["street"]
    psi = setup.cy(E["site"], h, s["orient"], s["aw"], setup.dlat[E["Ci"]])
    fst = setup.st.geom_factor(E["site"], h) * setup.st.uhi_factor(E["site"], s["dT"]) * s["fvar"]
    out = {}
    for m in METRICS:
        credit = -E["r"]["G_rf"][m] * psi * E["sysf"][E["Pi"]]
        burden = E["r"]["G_co"][m] + E["r"]["G_st"][m] * fst
        out[m] = (credit.reshape(E["shape"]), burden.reshape(E["shape"]), psi)
    return out


def statistics(setup, E, T, wname, maps=None, mapkey=None):
    nprod, nc, ns = E["shape"]
    wc = setup.W[wname]
    q = lambda x: [float(v) for v in np.percentile(x, [5, 25, 50, 75, 95])]  # noqa: E731
    res = {}
    for m in METRICS:
        credit, burden, _ = T[m]
        net = burden - credit
        w3 = np.broadcast_to(wc[None, :, None] / ns, E["shape"])
        nat_net = np.sum(w3 * net, axis=(1, 2))
        nat_cr = np.sum(w3 * credit, axis=(1, 2))
        nat_bu = np.sum(w3 * burden, axis=(1, 2))
        cr0 = credit / E["sysf"][:, None, None]
        mean_cr0 = np.sum(w3 * cr0) / nprod
        mean_bu = np.sum(w3 * burden) / nprod
        div_net = mean_bu - E["sysf"] * mean_cr0
        wflat = (w3 / nprod).ravel()
        Rs = (burden / credit).ravel()
        res[m] = dict(net=q(nat_net), net_mean=float(nat_net.mean()), R=q(nat_bu / nat_cr),
                      p_source=float(np.mean(nat_net > 0)), net_diverse=q(div_net),
                      p_source_diverse=float(np.mean(div_net > 0)),
                      share_streets_source=float(np.sum(wflat * (net.ravel() > 0))),
                      R_streets=[float(v) for v in wquant(Rs, wflat, [0.05, 0.25, 0.5, 0.75, 0.95])])
        # Monte Carlo precision: bootstrap over the realisations (95 % intervals)
        bi = np.random.default_rng(5).integers(0, nprod, (NBOOT, nprod))
        bs = nat_net[bi]
        res[m]["net_ci"] = np.percentile(np.percentile(bs, [5, 50, 95], axis=1), [2.5, 97.5], axis=1).T.tolist()
        res[m]["p_source_ci"] = np.percentile(np.mean(bs > 0, axis=1), [2.5, 97.5]).tolist()
        share_r = np.sum(w3 * (net > 0), axis=(1, 2))                     # share of streets, per realisation
        res[m]["share_streets_source_ci"] = np.percentile(share_r[bi].mean(axis=1), [2.5, 97.5]).tolist()
        # sources of the interval of a one-product programme: the product alone (systematic factors at their
        # median) and the systematic factors alone (every street with its own product)
        fix = nat_bu - np.median(E["sysf"]) * (nat_cr / E["sysf"])
        res[m]["net_productonly"] = q(fix)
        res[m]["var_share"] = dict(product=float(np.var(fix) / np.var(nat_net)),
                                   systematic=float(np.var(div_net) / np.var(nat_net)))
        res[m]["share_burden_nonpositive"] = float(np.sum(wflat * (burden.ravel() <= 0)))
        if m == "dynamic" and maps is not None:
            src = (net > 0).mean(axis=(0, 2))
            Rq = np.empty((nc, 3))
            for c in range(nc):
                Rq[c] = np.percentile((burden[:, c] / credit[:, c]).ravel(), [5, 50, 95])
            maps[f"{mapkey}_psource"] = src
            maps[f"{mapkey}_R05"], maps[f"{mapkey}_R50"], maps[f"{mapkey}_R95"] = Rq.T
            res[m]["cells_psource"] = dict(min=float(src.min()), med=float(np.median(src)), max=float(src.max()),
                                           share_w_gt90=float(np.sum(wc * (src > 0.9))))
            res[m]["cellfold_med"] = float(np.median(Rq[:, 2] / Rq[:, 0]))
            # variance of ln R within a cell: between streets (within a realisation), systematic factors, product
            lnR = np.log(np.clip(burden, 1e-6, None) / credit)
            vt = lnR.transpose(1, 0, 2).reshape(nc, -1).var(axis=1)
            vw = lnR.var(axis=2, ddof=1).mean(axis=0)
            vs = float(np.var(np.log(E["sysf"])))
            lpsi = np.log(T[m][2]).reshape(E["shape"])
            vpsi = lpsi.transpose(1, 0, 2).reshape(nc, -1).var(axis=1)
            res[m]["lnR_variance"] = dict(street=float(np.sum(wc * vw / vt)), systematic=float(np.sum(wc * vs / vt)),
                                          product=float(np.sum(wc * (1 - (vw + vs) / vt))),
                                          form=float(np.sum(wc * vpsi / vt)),
                                          street_fold_med=float(np.median(np.exp(2 * 1.6449 * np.sqrt(vw)))),
                                          form_fold_med=float(np.median(np.exp(2 * 1.6449 * np.sqrt(vpsi)))))
            res[m]["hist_nat_net"] = nat_net.tolist()
    return res


def part1(setup, rng, maps):
    out = {}
    morphs = ["lcz", "lcz_e", "lcz_nom", "open", "lowrise", "mixed", "compact"]
    for design in ("B", "A"):
        E = evaluate_streets(setup, rng, design, NPROD, NS)
        res = dict(n_products=NPROD, n_streets=int(np.prod(E["shape"])), n_cells=setup.nc,
                   sysf=[float(v) for v in np.percentile(E["sysf"], [5, 50, 95])])
        for morph in morphs:
            h, lab = draw_form(rng, setup, morph, E["Ci"])
            T = terms(setup, E, h)
            res[morph] = {}
            for wname in ("paved", "pop", "area", "sunny"):
                mk = f"{design}_{morph}" if (wname == "paved" and morph in ("lcz", "open")) else None
                wuse = "paved_e" if (morph == "lcz_e" and wname == "paved") else wname   # class E adds paved area
                res[morph][wname] = statistics(setup, E, T, wuse, maps if mk else None, mk)
            if morph == "lcz":
                wc = setup.W["paved"][E["Ci"]] / NS / NPROD
                res["lcz_h"] = dict(mean=float(np.sum(wc * h)), q=[float(v) for v in wquant(h, wc, [0.05, 0.25, 0.5, 0.75, 0.95])],
                                    psi=[float(v) for v in wquant(T["dynamic"][2], wc, [0.05, 0.25, 0.5, 0.75, 0.95])])
        # by LCZ class: every street drawn in one class
        keys, cls, _ = setup.class_probs(with_e=True)
        res["by_class"] = {}
        for c in keys:
            n = len(E["Ci"])
            h = svf_to_h(cls[c][0] + (cls[c][1] - cls[c][0]) * rng.random(n))
            credit, burden, _ = terms(setup, E, h)["dynamic"]
            wflat = np.broadcast_to(setup.W["paved"][None, :, None] / NS / NPROD, E["shape"]).ravel()
            Rk = (burden / credit).ravel()
            res["by_class"][LCZ_NAME[c]] = dict(share_source=float(np.sum(wflat * (Rk > 1)) / np.sum(wflat)),
                                                R=[float(v) for v in wquant(Rk, wflat, [0.05, 0.5, 0.95])],
                                                h=[float(cls[c][0]), float(cls[c][1])])
        out[design] = res
        d = res["lcz"]["paved"]["dynamic"]
        print(f"design {design}: {res['n_streets'] / 1e6:.1f} M streets | measured form, paved weights, dynamic: national net "
              f"{d['net'][2]:.1f} [{d['net'][0]:.1f}, {d['net'][4]:.1f}], P(source) {d['p_source']:.3f}, streets source "
              f"{d['share_streets_source']:.4f}, street R {d['R_streets'][2]:.2f} [{d['R_streets'][0]:.2f}, {d['R_streets'][4]:.2f}]"
              f" | cells {d['cells_psource']['min']:.3f}-{d['cells_psource']['max']:.3f}", flush=True)
        for morph in morphs:
            for wname in ("paved", "pop", "area", "sunny"):
                d = res[morph][wname]["dynamic"]
                print(f"   {morph:8s} {wname:5s} net {d['net'][2]:6.1f} [{d['net'][0]:6.1f},{d['net'][4]:6.1f}] P {d['p_source']:.3f}"
                      f" | streets {d['share_streets_source']:.4f} R {d['R_streets'][2]:5.2f} | static P "
                      f"{res[morph][wname]['static']['p_source']:.3f} streets {res[morph][wname]['static']['share_streets_source']:.4f}"
                      f" | gtp streets {res[morph][wname]['gtp']['share_streets_source']:.4f}", flush=True)
        print("   by class:", {k: (round(v["share_source"], 4), round(v["R"][1], 2)) for k, v in res["by_class"].items()}, flush=True)
        print("   form:", res["lcz_h"], flush=True)
    return out


def part3(setup, rng):
    """Correlated coating properties (measured form, paved weights), with
    common random numbers: every case of a design uses the same draws, so
    that the cases differ only by the correlation."""
    cases = {"independent": None, "tau_gco+0.5": {("tau", "gco"): 0.5}, "tau_gco-0.5": {("tau", "gco"): -0.5},
             "af_tau-0.5": {("af", "tau"): -0.5}, "af_tau+0.5": {("af", "tau"): 0.5},
             "af_gco+0.5": {("af", "gco"): 0.5}, "Tr_tau+0.5": {("Tr", "tau"): 0.5}, "Tr_tau+0.8": {("Tr", "tau"): 0.8}}
    cases["shared_street"] = "shared"                  # street properties drawn once per cell and realisation
    out = {}
    for design in ("B", "A"):
        out[design] = {}
        seed = int(rng.integers(2 ** 31))
        for name, corr in cases.items():
            shared = corr == "shared"
            corr = None if shared else corr
            if design == "B" and corr and "Tr" in next(iter(corr)):
                continue                                   # renewal interval fixed in design B
            crn = np.random.default_rng(seed)
            E = evaluate_streets(setup, crn, design, NPROD // 2, 2, corr, shared=shared)
            if corr:
                (a, b), rs = next(iter(corr.items()))
                from scipy.stats import spearmanr
                rho = float(spearmanr(E["P"][a], E["P"][b])[0])
            else:
                rho = 0.0
            h, _ = draw_form(crn, setup, "lcz", E["Ci"])
            st = statistics(setup, E, terms(setup, E, h), "paved")
            d = st["dynamic"]
            out[design][name] = dict(rho_sample=rho, net=d["net"], p_source=d["p_source"],
                                     share_streets_source=d["share_streets_source"], R_streets=d["R_streets"],
                                     static_share=st["static"]["share_streets_source"])
            if shared or name == "independent":            # spread of the share of sources between cells
                cr, bu, _ = terms(setup, E, h)["dynamic"]
                cs = ((bu - cr) > 0).mean(axis=(0, 2))
                out[design][name]["cells_psource"] = dict(min=float(cs.min()), p05=float(np.percentile(cs, 5)),
                                                          med=float(np.median(cs)))
            print(f"corr {design} {name:12s} rho {rho:+.2f} net {d['net'][2]:6.1f} [{d['net'][0]:6.1f},{d['net'][4]:6.1f}] "
                  f"P {d['p_source']:.3f} streets {d['share_streets_source']:.4f}", flush=True)
    return out


def part2(setup, maps):
    nc = setup.nc
    cells = np.arange(nc)
    cy, stc = setup.cy, setup.st
    W = setup.W

    def R_of(h=0.0, tau=REF.tau, Tr=REF.Tr, af=REF.alphaf, gco=REF.g_co, life=None, metric="dynamic", psi_site=None,
             dlat=None, dT=0.0):
        site = setup.near if life is None else np.full(nc, life)
        rr = lb.evaluate(setup.Kd, cells, REF.alpha0, af, REF.alpha0, tau, Tr, gco, REF.g_st, 1.0, site, setup.al, setup.hs,
                         setup.Ltab, nquad=NQUAD)
        ps = setup.near if psi_site is None else np.full(nc, psi_site)
        hh = np.full(nc, float(h))
        psi = cy(ps, hh, np.zeros(nc, int), np.full(nc, 0.30), setup.dlat if dlat is None else dlat)
        fst = stc.geom_factor(setup.near, hh) * stc.uhi_factor(setup.near, np.full(nc, dT))
        return (rr["G_co"][metric] + rr["G_st"][metric] * fst) / (-rr["G_rf"][metric] * psi)

    zero = np.zeros(nc)
    R0 = R_of()
    chk = float(np.max(np.abs(R0 / setup.zl["R_dynamic"][setup.ia, setup.ib] - 1)))
    stat = lambda f: dict(med=float(np.median(f)), p05=float(np.percentile(f, 5)), p95=float(np.percentile(f, 95)),  # noqa: E731
                          min=float(f.min()), max=float(f.max()),
                          spread_pct=float(100 * (np.percentile(f, 95) / np.percentile(f, 5) - 1)))
    dist = dict(check_against_grid_ledger=chk)
    grids = dict(h=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0], tau=[0.5, 0.7, 1.5, 2.0, 3.0], Tr=[1.0, 2.0, 5.0, 7.0, 10.0],
                 af=[0.25, 0.35, 0.60], gco=[1.63, 3.39])
    for name, vals in grids.items():
        dist[name] = {f"{v:g}": stat(R_of(**{name: v}) / R0) for v in vals}
    curves = dict(h=np.linspace(0, 2, 41), tau=np.geomspace(0.5, 3.0, 25), Tr=np.arange(1.0, 10.01, 1.0))
    for name, xs in curves.items():
        F = np.array([R_of(**{name: float(x)}) / R0 for x in xs])
        maps[f"dist_{name}_x"] = xs
        maps[f"dist_{name}_q"] = np.percentile(F, [5, 50, 95], axis=1)
    hh, tt = [0.0, 0.25, 0.5, 1.0, 2.0], [0.5, 0.91, 1.5, 3.0]
    joint = {}
    for a in hh:
        for b in tt:
            Rj = R_of(h=a, tau=b)
            joint[f"{a:g}|{b:g}"] = dict(factor=stat(Rj / R0), R=stat(Rj), share_cells_source=float(np.mean(Rj > 1)),
                                         share_paved_source=float(np.sum(W["paved"] * (Rj > 1))),
                                         share_pop_source=float(np.sum(W["pop"] * (Rj > 1))))
    dist["joint"] = joint
    jt = {}
    for b in tt:
        for c in (1.0, 2.0, 3.0, 5.0, 10.0):
            Rj = R_of(tau=b, Tr=c)
            jt[f"{b:g}|{c:g}"] = dict(R=stat(Rj), share_cells_source=float(np.mean(Rj > 1)),
                                      share_paved_source=float(np.sum(W["paved"] * (Rj > 1))))
    dist["joint_tau_Tr"] = jt
    # nearest-station assumptions: structure and canyon factor of any other station; latitude transfer
    Rl = np.array([R_of(life=j) for j in range(len(SITES))])
    dist["life_station"] = dict(max_abs_pct=float(100 * np.max(np.abs(Rl / R0[None, :] - 1))),
                                p95_abs_pct=float(np.percentile(100 * np.abs(Rl / R0[None, :] - 1), 95)))
    dist["psi_station"] = {}
    for h in (0.25, 0.5, 1.0):
        Rn = R_of(h=h, dlat=zero)
        Rp = np.array([R_of(h=h, psi_site=j, dlat=zero) for j in range(len(SITES))])
        Rt = R_of(h=h)
        dist["psi_station"][f"{h:g}"] = dict(any_station_max_abs_pct=float(100 * np.max(np.abs(Rp / Rn[None, :] - 1))),
                                             any_station_p95_abs_pct=float(np.percentile(100 * np.abs(Rp / Rn[None, :] - 1), 95)),
                                             transfer_max_abs_pct=float(100 * np.max(np.abs(Rt / Rn - 1))),
                                             transfer_p95_abs_pct=float(np.percentile(100 * np.abs(Rt / Rn - 1), 95)))
    dist["psi_regression_resid_pct"] = {f"{h:g}": float(100 * np.max(cy.resid[:, int(np.argmin(np.abs(cy.h - h)))]))
                                        for h in (0.25, 0.5, 1.0, 2.0)}
    dist["calibration"] = dict(min_pct=float(100 * (setup.ratio / setup.ratios.max() - 1)),
                               max_pct=float(100 * (setup.ratio / setup.ratios.min() - 1)))
    # urban heat island on its own
    dist["uhi3"] = stat(R_of(dT=UHI_MAX) / R0)
    el = {}
    for name, x0, dx in (("tau", REF.tau, 0.02), ("Tr", REF.Tr, 0.05), ("af", REF.alphaf, 0.01), ("gco", REF.g_co, 0.05)):
        f = (np.log(R_of(**{name: x0 + dx})) - np.log(R_of(**{name: x0 - dx}))) / (np.log(x0 + dx) - np.log(x0 - dx))
        el[name] = stat(f)
    f = (np.log(R_of(h=0.1)) - np.log(R0)) / 0.1
    el["h_per_0.1"] = stat(100 * (np.exp(0.1 * f) - 1))
    dist["elasticity"] = el
    print("distortion: check", f"{chk:.1e}", "| h", {k: round(v["med"], 2) for k, v in dist["h"].items()},
          "| tau", {k: round(v["med"], 2) for k, v in dist["tau"].items()},
          "| Tr", {k: round(v["med"], 2) for k, v in dist["Tr"].items()}, flush=True)
    print("nearest station: life", dist["life_station"], "| psi", dist["psi_station"], "| resid",
          dist["psi_regression_resid_pct"], "| uhi3", round(dist["uhi3"]["med"], 4), flush=True)
    return dist, stat(R0)


def main():
    rng = np.random.default_rng(20261118)
    setup = Setup()
    if "--part3" in sys.argv:                     # correlations only, merged into the existing results
        f = RES / f"conus_mc_{TAG}.json"
        out = json.loads(f.read_text())
        out["correlation"] = part3(setup, np.random.default_rng(31))
        f.write_text(json.dumps(out, indent=1))
        return
    print(f"structural factors: {setup.st.source}; variants: "
          f"{None if setup.st.variants is None else setup.st.variant_names}", flush=True)
    if setup.st.source == "simulated":
        print("geometry factors (sites x h=0,.25,.5,1,2):", np.round(setup.st.geom, 3).tolist(), flush=True)
        print("uhi3 factors:", np.round(setup.st.uhi, 3).tolist(), flush=True)
    maps = {}
    out = dict(part1=part1(setup, rng, maps))
    out["correlation"] = part3(setup, rng)
    dist, r0 = part2(setup, maps)
    out["distortion"] = dist
    out["R0_dynamic"] = r0
    out["mixes"] = MIXES
    out["strata"] = STRATA
    out["lcz"] = {LCZ_NAME[c]: dict(svf=[v[0], v[1]], h=[float(svf_to_h(v[1])), float(svf_to_h(v[0]))], imperv=v[2])
                  for c, v in {**LCZ, **LCZ_E}.items()}
    keys, cls, p = setup.class_probs()
    share = np.array([np.sum(setup.W["paved"] * p[:, i]) for i in range(len(keys))])
    out["lcz_paved_share"] = {LCZ_NAME[c]: float(s) for c, s in zip(keys, share)}
    out["structure_source"] = setup.st.source
    if setup.st.source == "simulated":
        out["structure_factors"] = dict(h=setup.st.hg.tolist(), geom=setup.st.geom.tolist(), uhi3=setup.st.uhi.tolist(),
                                        urban={k: v.tolist() for k, v in setup.st.urban.items()},
                                        proxy=np.interp(setup.st.hg, setup.st.fh_proxy, setup.st.floor_proxy).tolist())
    (RES / f"conus_mc_{TAG}.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(RES / f"conus_mc_{TAG}.npz", ia=setup.ia, ib=setup.ib, lat=setup.lat, lon=setup.lon,
                        wpaved=setup.W["paved"], pop=setup.W["pop"], **maps)


if __name__ == "__main__":
    main()

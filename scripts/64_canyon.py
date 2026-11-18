"""Urban-form factor of a reflective street in a canyon (Sect. 4.6, 6.7).

1. Radiosity responses of the canyon cross section (alrb.canyon) for a grid
   of aspect ratios and wall albedos, floor albedo 0.14 -> 0.45.
2. Independent check by Monte Carlo path tracing (Mitsuba 3, irradiance
   meter over the opening of a long three-dimensional canyon).
3. Kernel-weighted urban-form factor at every site for three street
   orientations, from the hourly irradiance (Erbs diffuse fraction, sun
   position at 15-min sub-steps) and the snow-masked hourly kernel of the
   seven-year record used by the ledger.
4. Consequence for the static rebound ratio and the albedo payback time.

usage: python 64_canyon.py [--spp 4194304]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import canyon, driver, kernel, solar  # noqa: E402
from alrb.config import RES, SITE, SITES  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--spp", type=int, default=1 << 22)
_ap.add_argument("--skip-pt", action="store_true")
_A = _ap.parse_args()

A0, A1 = 0.14, 0.45
AW_REF = 0.30
H_GRID = np.round(np.concatenate([np.arange(0.0, 1.0, 0.05), np.arange(1.0, 3.01, 0.1)]), 3)
AW_GRID = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
ORIENT = {"EW": 90.0, "NS": 0.0, "diag": 45.0}      # street-axis azimuth (deg from north)
TP = np.radians(np.linspace(0.0, 89.75, 360))
TAG = "v3"


def response_table(aw):
    """Diffuse and beam responses (difference between the two floor albedos,
    per unit albedo) for every aspect ratio."""
    dif, beam = [], []
    for h in H_GRID:
        if h == 0.0:
            dif.append(1.0)
            beam.append(np.ones_like(TP))
            continue
        c = canyon.build(float(h), 80)
        d0, _, b0 = canyon.responses(c, A0, aw, TP)
        d1, _, b1 = canyon.responses(c, A1, aw, TP)
        dif.append((d1 - d0) / (A1 - A0))
        beam.append((b1 - b0) / (A1 - A0))
    return np.array(dif), np.array(beam)


def site_inputs(key):
    df = driver.load_forcing(key)
    K, _, _, kt, _ = kernel.hourly_kernel(df)
    mask = driver.snow_mask_pavement(df, key, days_after=1.0)
    K = K * (1.0 - mask)
    loc = df.index + pd.Timedelta(hours=SITE[key].tz)
    keep = (loc >= "2017-05-01") & (loc < "2024-05-01")
    sw = df["sw"].to_numpy()
    kd = solar.erbs_diffuse_fraction(kt)
    D = kd * sw
    B = sw - D
    zen, az, w = canyon.solar_angles(df.index, SITE[key].lat, SITE[key].lon, nsub=4)
    wsum = w.sum(1, keepdims=True)
    wn = np.where(wsum > 0, w / np.maximum(wsum, 1e-12), 0.25)
    return dict(K=K[keep], sw=sw[keep], D=D[keep], B=B[keep], zen=zen[keep], az=az[keep], wn=wn[keep],
                month=loc[keep].month.to_numpy())


def psi_series(inp, dif, beam_row, axis_deg):
    """Hourly psi numerator (W m-2 per unit albedo) for one aspect ratio."""
    tp = np.arctan(np.tan(np.minimum(inp["zen"], np.radians(89.9)))
                   * np.abs(np.sin(inp["az"] - np.radians(axis_deg))))
    rb = np.interp(np.minimum(tp, TP[-1]), TP, beam_row)
    rb = np.where(inp["zen"] < np.radians(90.0), rb, 0.0)
    return dif * inp["D"] + inp["B"] * np.sum(inp["wn"] * rb, axis=1)


def site_psi(inp, dif, beam):
    """Kernel-weighted psi for every aspect ratio and orientation, and the
    monthly values at H/W = 0.5 and 1."""
    K, sw = inp["K"], inp["sw"]
    ok = sw > 1.0
    wK = np.where(ok, K, 0.0)
    out = {}
    for name, ax in ORIENT.items():
        vals, monthly = [], {}
        for j, h in enumerate(H_GRID):
            num = psi_series(inp, dif[j], beam[j], ax)
            psi_t = np.where(ok, num / np.maximum(sw, 1e-9), 0.0)
            vals.append(float(np.sum(wK * psi_t) / np.sum(wK)))
            if h in (0.5, 1.0):
                m = inp["month"]
                monthly[f"{h:.1f}"] = [float(np.sum((wK * psi_t)[m == k]) / np.sum(wK[m == k]))
                                       if np.sum(wK[m == k]) != 0 else float("nan") for k in range(1, 13)]
        out[name] = dict(psi=vals, monthly=monthly)
    # irradiance-weighted (no kernel) for comparison
    num = psi_series(inp, dif[H_GRID.tolist().index(1.0)], beam[H_GRID.tolist().index(1.0)], ORIENT["EW"])
    out["EW_irradiance_weighted_h1"] = float(np.sum(num[ok]) / np.sum(sw[ok]))
    return out


# ------------------------------------------------------------------ path tracing
def pt_escape(h, alb_floor, alb_wall, theta_p_deg=None, spp=1 << 22, length=40.0, seed=0):
    """Shortwave flux leaving the opening of a canyon of length ``length``
    (in street widths), per unit horizontal irradiance, from Monte Carlo path
    tracing: irradiance meters on the central two widths of the floor and of
    both walls give the absorbed flux, and by translational symmetry of the
    long canyon the flux leaving the central section is the incident flux
    minus the absorbed flux.  Beam with projected zenith theta_p (sun in the
    cross section, on the side x > 1), or isotropic sky if theta_p_deg is None."""
    import mitsuba as mi
    mi.set_variant("llvm_ad_rgb")
    T = mi.ScalarTransform4f
    half = length / 2.0

    def diffuse(a):
        return {"type": "twosided", "material": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [a, a, a]}}}

    counter = [0]

    def rect(center, scale, rot_axis=None, rot_deg=0.0, bsdf=None, meter=False):
        counter[0] += 1
        tr = T().translate(center)
        if rot_axis is not None:
            tr = tr @ T().rotate(rot_axis, rot_deg)
        tr = tr @ T().scale(scale)
        d = {"type": "rectangle", "to_world": tr, "bsdf": bsdf}
        if meter:
            d["sensor"] = {"type": "irradiancemeter",
                           "sampler": {"type": "independent", "sample_count": spp, "seed": seed + counter[0]},
                           "film": {"type": "hdrfilm", "width": 1, "height": 1, "rfilter": {"type": "box"},
                                    "pixel_format": "rgb"}}
        return d

    black = diffuse(0.0)
    scene = {"type": "scene", "integrator": {"type": "path", "max_depth": 200, "rr_depth": 150}}
    yc = 1.0                                    # half-length of the metered central section
    yo = (half - yc) / 2.0                      # half-length of each outer piece
    for part, y0, sy, meter in (("c", 0.0, yc, True), ("n", yc + yo, yo, False), ("s", -yc - yo, yo, False)):
        # floor (z = 0, normal +z), left wall (x = 0, normal +x), right wall (x = 1, normal -x)
        scene["floor_" + part] = rect([0.5, y0, 0.0], [0.5, sy, 1.0], bsdf=diffuse(alb_floor), meter=meter)
        scene["wl_" + part] = rect([0.0, y0, h / 2], [h / 2, sy, 1.0], [0, 1, 0], 90.0, bsdf=diffuse(alb_wall),
                                   meter=meter)
        scene["wr_" + part] = rect([1.0, y0, h / 2], [h / 2, sy, 1.0], [0, 1, 0], -90.0, bsdf=diffuse(alb_wall),
                                   meter=meter)
    # black end walls, roofs and a black plane below everything
    scene["e0"] = rect([0.5, -half, h / 2], [0.5, h / 2, 1.0], [1, 0, 0], 90.0, bsdf=black)
    scene["e1"] = rect([0.5, half, h / 2], [0.5, h / 2, 1.0], [1, 0, 0], 90.0, bsdf=black)
    big = 2000.0
    scene["roofL"] = rect([-big / 2, 0.0, h], [big / 2, big, 1.0], bsdf=black)
    scene["roofR"] = rect([1.0 + big / 2, 0.0, h], [big / 2, big, 1.0], bsdf=black)
    scene["roofN"] = rect([0.5, half + big / 2, h], [0.5, big / 2, 1.0], bsdf=black)
    scene["roofS"] = rect([0.5, -half - big / 2, h], [0.5, big / 2, 1.0], bsdf=black)
    scene["under"] = rect([0.5, 0.0, -1e-3], [big, big, 1.0], bsdf=black)
    if theta_p_deg is None:
        scene["sky"] = {"type": "constant", "radiance": {"type": "rgb", "value": [1 / np.pi] * 3}}
    else:
        th = np.radians(theta_p_deg)
        scene["sun"] = {"type": "directional", "direction": [-np.sin(th), 0.0, -np.cos(th)],
                        "irradiance": {"type": "rgb", "value": [1.0 / np.cos(th)] * 3}}
    sc = mi.load_dict(scene)
    E = {}
    for shape in sc.shapes():
        if not shape.is_sensor():
            continue
        bb = shape.bbox()
        cx = 0.5 * (bb.min[0] + bb.max[0])
        name = "floor" if bb.max[2] - bb.min[2] < 1e-6 else ("wl" if cx < 0.5 else "wr")
        E[name] = float(np.array(mi.render(sc, sensor=shape.sensor(), spp=spp))[0, 0, 0])
    if theta_p_deg is not None:
        # the meters cannot see a delta (directional) emitter, so they return the
        # irradiance from reflections only; the direct beam on the metered
        # sections is added by casting shadow rays towards the sun in the same scene
        th = np.radians(theta_p_deg)
        sdir = np.array([np.sin(th), 0.0, np.cos(th)])
        rng = np.random.default_rng(seed + 17)
        n = 400_000
        u, v = rng.random(n), rng.random(n) * 2.0 - 1.0
        surf = {"floor": (np.column_stack([u, v, np.zeros(n)]), np.array([0.0, 0.0, 1.0])),
                "wl": (np.column_stack([np.zeros(n), v, u * h]), np.array([1.0, 0.0, 0.0])),
                "wr": (np.column_stack([np.ones(n), v, u * h]), np.array([-1.0, 0.0, 0.0]))}
        for name, (P, nrm) in surf.items():
            cosi = float(nrm @ sdir)
            if cosi <= 0:
                continue
            o = P + 1e-5 * nrm
            ray = mi.Ray3f(mi.Point3f(o[:, 0], o[:, 1], o[:, 2]),
                           mi.Vector3f(np.full(n, sdir[0]), np.full(n, sdir[1]), np.full(n, sdir[2])))
            blocked = np.array(sc.ray_test(ray))
            E[name] += (1.0 - blocked.mean()) * cosi / np.cos(th)
    absorbed = (1 - alb_floor) * E["floor"] + (1 - alb_wall) * h * (E["wl"] + E["wr"])
    return 1.0 - absorbed


def path_trace_check():
    cases = []
    for h in (0.5, 1.0, 2.0):
        for tp in (None, 0.0, 30.0, 55.0):
            cases.append((h, tp))
    out = []
    for h, tp in cases:
        t0 = time.time()
        e0 = pt_escape(h, A0, AW_REF, tp, spp=_A.spp, seed=1)
        e1 = pt_escape(h, A1, AW_REF, tp, spp=_A.spp, seed=2)
        c = canyon.build(h, 80)
        grid = np.radians([0.0 if tp is None else tp])
        d0, _, b0 = canyon.responses(c, A0, AW_REF, grid)
        d1, _, b1 = canyon.responses(c, A1, AW_REF, grid)
        rad0, rad1 = (d0, d1) if tp is None else (b0[0], b1[0])
        rec = dict(h=h, theta_p=tp, pt_esc=[e0, e1], rad_esc=[rad0, rad1],
                   pt_psi=(e1 - e0) / (A1 - A0), rad_psi=(rad1 - rad0) / (A1 - A0))
        out.append(rec)
        print(f"PT h={h} tp={tp}: esc {e0:.4f}/{e1:.4f} vs {rad0:.4f}/{rad1:.4f}; psi {rec['pt_psi']:.4f} vs "
              f"{rec['rad_psi']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    res = dict(h=H_GRID.tolist(), aw=list(AW_GRID), a0=A0, a1=A1, aw_ref=AW_REF, orient=ORIENT)
    tabs = {aw: response_table(aw) for aw in AW_GRID}
    res["fgs"] = canyon.floor_sky_view(np.maximum(H_GRID, 0)).tolist()
    res["diffuse"] = {f"{aw:.2f}": tabs[aw][0].tolist() for aw in AW_GRID}
    res["beam_tp_deg"] = np.degrees(TP[::12]).tolist()
    res["beam"] = {f"{aw:.2f}": tabs[aw][1][:, ::12].tolist() for aw in AW_GRID}
    print("tables done", flush=True)
    lj = json.loads((RES / f"ledger_{TAG}.json").read_text())
    sites = {}
    for s in SITES:
        inp = site_inputs(s.key)
        out = site_psi(inp, *tabs[AW_REF])
        # wall-albedo sensitivity for the E-W street
        out["EW_by_aw"] = {}
        for aw in AW_GRID:
            dif, beam = tabs[aw]
            vals = []
            for j in range(len(H_GRID)):
                num = psi_series(inp, dif[j], beam[j], ORIENT["EW"])
                ok = inp["sw"] > 1.0
                vals.append(float(np.sum(np.where(ok, inp["K"] * num / np.maximum(inp["sw"], 1e-9), 0.0))
                                  / np.sum(np.where(ok, inp["K"], 0.0))))
            out["EW_by_aw"][f"{aw:.2f}"] = vals
        ref = lj[s.key]["reference"]["static"]
        out["R_open"] = ref["R"]
        out["G"] = dict(G_rf=ref["G_rf"], G_co=ref["G_co"], G_st=ref["G_st"])
        sites[s.key] = out
        j05, j1 = H_GRID.tolist().index(0.5), H_GRID.tolist().index(1.0)
        print(f"{s.key:13s} psi(0.5) EW {out['EW']['psi'][j05]:.3f} NS {out['NS']['psi'][j05]:.3f} | psi(1) EW "
              f"{out['EW']['psi'][j1]:.3f} NS {out['NS']['psi'][j1]:.3f}", flush=True)
    res["sites"] = sites
    if not _A.skip_pt:
        res["path_tracing"] = path_trace_check()
    (RES / f"canyon_{TAG}.json").write_text(json.dumps(res, indent=1))

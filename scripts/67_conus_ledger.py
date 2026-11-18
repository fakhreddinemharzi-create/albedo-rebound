"""The reference ledger for every 1-degree land cell of the contiguous
United States (Sect. 6.8).

The daily kernel of each cell is its monthly snow-masked kernel from
66_conus_kernel.py, scaled by the median ratio of the station to the grid
kernel at the ten station cells (the station network defines the kernel of
the paper), and repeated over the seven ledger years; the structural life
function is that of the nearest station.  The ledger of Sect. 4.8 is then
evaluated exactly as at the stations.  At the station cells the result is
compared with the station ledger.

Writes results/conus_ledger_v3.npz and results/conus_ledger_v3.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb import ledger  # noqa: E402
from alrb.config import ALBEDO_GRID, REF, RES, SITES  # noqa: E402

TAG = "v3"


def life_fun(js):
    al = np.array(ALBEDO_GRID)
    L = np.array([js["albedo"][f"{a:.2f}"]["life_ref"] for a in al])
    L = L / L[list(al).index(0.14)] * 20.0
    return lambda a: np.interp(a, al, L)


def main():
    z = np.load(RES / "conus_kernel_v3.npz")
    cj = json.loads((RES / "conus_kernel_v3.json").read_text())["sites"]
    ratio = float(np.median([v["station_Km"] / v["grid_Km"] for v in cj.values()]))
    lat, lon, land, Kmon = z["lat"], z["lon"], z["land"], z["Kmon"]
    days = pd.date_range("2017-05-01", "2024-04-30", freq="D")
    mon = days.month.to_numpy() - 1
    st = json.loads((RES / f"structural_{TAG}.json").read_text())
    lfs = {s.key: life_fun(st[s.key]) for s in SITES}
    slat = np.array([s.lat for s in SITES])
    slon = np.array([s.lon for s in SITES])
    shape = land.shape
    out = {k: np.full(shape, np.nan) for k in ("R_static", "R_dynamic", "R_gtp", "Gnet_static", "Gnet_dynamic",
                                               "Grf_static", "Gst_static", "gstar_dynamic", "Tstar", "kappa", "K")}
    near = np.full(shape, -1)
    base = ledger.Params()
    for a in range(shape[0]):
        for b in range(shape[1]):
            if not land[a, b] or not np.all(np.isfinite(Kmon[:, a, b])):
                continue
            Kd = 100.0 * ratio * Kmon[mon, a, b]              # W m-2 per unit albedo
            d = (slat - lat[a]) ** 2 + ((slon - lon[b]) * np.cos(np.radians(lat[a]))) ** 2
            j = int(np.argmin(d))
            near[a, b] = j
            r = ledger.evaluate(Kd, base, lfs[SITES[j].key], metrics_list=("static", "dynamic", "gtp"))
            out["R_static"][a, b] = r["static"]["R"]
            out["R_dynamic"][a, b] = r["dynamic"]["R"]
            out["R_gtp"][a, b] = r["gtp"]["R"]
            out["Gnet_static"][a, b] = r["static"]["G_net"]
            out["Gnet_dynamic"][a, b] = r["dynamic"]["G_net"]
            out["Grf_static"][a, b] = r["static"]["G_rf"]
            out["Gst_static"][a, b] = r["static"]["G_st"]
            dyn = r["dynamic"]
            out["gstar_dynamic"][a, b] = REF.g_co * (-dyn["G_rf"] - dyn["G_st"]) / dyn["G_co"]
            kap = ledger.kappa(Kd, TH=100.0)
            out["kappa"][a, b] = kap
            out["Tstar"][a, b] = REF.g_co / (kap * (REF.alphaf - REF.alpha0))
            out["K"][a, b] = float(np.mean(Kd)) / 100.0
    np.savez_compressed(RES / f"conus_ledger_{TAG}.npz", lat=lat, lon=lon, land=land, near=near, ratio=ratio, **out)
    lg = json.loads((RES / f"ledger_{TAG}.json").read_text())
    ds = json.loads((RES / f"design_space_{TAG}.json").read_text())
    val = {}
    for s in SITES:
        a = int(np.argmin(np.abs(lat - s.lat)))
        b = int(np.argmin(np.abs(lon - s.lon)))
        val[s.key] = dict(R_static_map=float(out["R_static"][a, b]), R_static_station=lg[s.key]["reference"]["static"]["R"],
                          R_dynamic_map=float(out["R_dynamic"][a, b]),
                          R_dynamic_station=lg[s.key]["reference"]["dynamic"]["R"],
                          gstar_map=float(out["gstar_dynamic"][a, b]),
                          gstar_station=ds["sites"][s.key]["ref"]["dynamic"]["g_co_star"])
        v = val[s.key]
        print(f"{s.key:13s} R static map {v['R_static_map']:.2f} station {v['R_static_station']:.2f} | "
              f"g* map {v['gstar_map']:.2f} station {v['gstar_station']:.2f}", flush=True)
    m = land & np.isfinite(out["R_static"])
    summ = dict(ratio=ratio, ncells=int(m.sum()), validation=val)
    for k, v in out.items():
        x = v[m]
        summ[k] = dict(min=float(x.min()), max=float(x.max()), med=float(np.median(x)),
                       p05=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)))
    # area-weighted shares
    w = np.cos(np.radians(lat))[:, None] * np.ones(shape)
    for thr in (2.5, 3.0, 4.0):
        summ[f"share_Rstatic_gt_{thr}"] = float(np.sum(w[m] * (out["R_static"][m] > thr)) / np.sum(w[m]))
    summ["share_Rdyn_gt_1"] = float(np.sum(w[m] * (out["R_dynamic"][m] > 1)) / np.sum(w[m]))
    (RES / f"conus_ledger_{TAG}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: summ[k] for k in ("R_static", "R_dynamic", "gstar_dynamic", "Tstar", "K")}, indent=0))


if __name__ == "__main__":
    main()

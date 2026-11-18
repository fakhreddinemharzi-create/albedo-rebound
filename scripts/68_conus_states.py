"""State means of the gridded kernel and balance (Table S7).

Every one-degree land cell is assigned to the state containing its centre;
a state without a cell centre takes the cell nearest to its centroid.
Means are weighted by cell area (cosine of latitude).  The October penalty
is recomputed from the monthly kernel as in fig_season.py.
Writes results/conus_states_v3.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb.config import RAW, REF, RES  # noqa: E402

TAG = "v3"
W = np.exp(-np.arange(int(REF.Tr * 365.25)) / (REF.tau * 365.25))
ABBR = {"Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
        "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Idaho": "ID", "Illinois": "IL",
        "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
        "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
        "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
        "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND",
        "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
        "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI",
        "Wyoming": "WY"}


def timing(k, dates):
    kk = np.concatenate([k, k, k, k])
    base = np.mean(k) * W.sum()
    out = []
    for m in range(1, 13):
        starts = np.where((dates.month == m) & (dates.day == 1))[0]
        out.append(np.mean([np.sum(kk[i:i + len(W)] * W) for i in starts]) / base)
    return np.array(out)


def main():
    z = np.load(RES / f"conus_ledger_{TAG}.npz")
    zk = np.load(RES / "conus_kernel_v3.npz")
    lat, lon, land = z["lat"], z["lon"], z["land"]
    ok = land & np.isfinite(z["R_static"])
    days = pd.date_range("2017-05-01", "2024-04-30", freq="D")
    mon = days.month.to_numpy() - 1
    gj = json.loads((RAW / "ne" / "ne_50m_admin_1_states_provinces_lakes.geojson").read_text())
    LO, LA = np.meshgrid(lon, lat)
    pts = np.column_stack([LO.ravel(), LA.ravel()])
    out = {}
    for f in gj["features"]:
        pr = f["properties"]
        name = pr.get("name")
        if pr.get("adm0_a3") != "USA" or name not in ABBR:
            continue
        g = f["geometry"]
        rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [p[0] for p in g["coordinates"]]
        inside = np.zeros(len(pts), bool)
        allv = []
        for r in rings:
            a = np.asarray(r)
            allv.append(a)
            inside |= MPath(a).contains_points(pts)
        inside = inside.reshape(LA.shape) & ok
        if not inside.any():
            v = np.concatenate(allv)
            c = v.mean(0)
            d = (LA - c[1]) ** 2 + ((LO - c[0]) * np.cos(np.radians(c[1]))) ** 2
            d[~ok] = np.inf
            a, b = np.unravel_index(np.argmin(d), d.shape)
            inside = np.zeros_like(ok)
            inside[a, b] = True
        w = np.cos(np.radians(LA))[inside]
        rec = dict(n=int(inside.sum()))
        for key in ("K", "R_static", "R_dynamic", "gstar_dynamic", "Tstar"):
            rec[key] = float(np.sum(w * z[key][inside]) / np.sum(w))
        pens = []
        for a, b in zip(*np.where(inside)):
            fm = timing(zk["Kmon"][mon, a, b], days)
            pens.append(100 * (1 - fm[9] / fm.max()))
        rec["pen_oct"] = float(np.sum(w * np.array(pens)) / np.sum(w))
        out[ABBR[name]] = rec
    out = dict(sorted(out.items()))
    (RES / f"conus_states_{TAG}.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f"{k} n={v['n']:3d} K {-v['K']:.2f} R {v['R_static']:.2f}/{v['R_dynamic']:.2f} g* {v['gstar_dynamic']:.2f} "
              f"T* {v['Tstar']:.1f} pen {v['pen_oct']:.0f}")
    print(len(out), "states")


if __name__ == "__main__":
    main()

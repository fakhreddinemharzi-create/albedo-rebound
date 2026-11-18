"""Population of the one-degree cells of the contiguous United States.

Aggregates the 2020 Census county populations, located at the county
centres of population, to the cells of the gridded ledger; a county whose
centre falls in a cell without a ledger (coast) is assigned to the nearest
cell with one.  The population weights the cells in the Monte Carlo of
82_conus_mc.py, as a proxy of where streets are.

Input : RAW/census/CenPop2020_Mean_CO.txt
        (https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt)
Writes results/conus_population_v3.npz.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb.config import RAW, RES  # noqa: E402

EXCLUDE = {"02", "15", "72"}          # Alaska, Hawaii, Puerto Rico


def main():
    z = np.load(RES / "conus_ledger_v3.npz")
    lat, lon = z["lat"], z["lon"]
    ok = z["land"] & np.isfinite(z["R_static"])
    ia, ib = np.where(ok)
    pop = np.zeros(ok.shape)
    total = moved = 0.0
    with open(RAW / "census" / "CenPop2020_Mean_CO.txt", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["STATEFP"] in EXCLUDE:
                continue
            p, la, lo = float(r["POPULATION"]), float(r["LATITUDE"]), float(r["LONGITUDE"])
            d = (lat[ia] - la) ** 2 + ((lon[ib] - lo) * np.cos(np.radians(la))) ** 2
            k = int(np.argmin(d))
            if abs(lat[ia[k]] - la) > 0.5 or abs(lon[ib[k]] - lo) > 0.5:
                moved += p
            pop[ia[k], ib[k]] += p
            total += p
    np.savez_compressed(RES / "conus_population_v3.npz", lat=lat, lon=lon, pop=pop, total=total, moved=moved)
    print(f"population {total / 1e6:.1f} million in {int((pop > 0).sum())} of {int(ok.sum())} cells; "
          f"{100 * moved / total:.1f} % assigned to a neighbouring cell; "
          f"half of the population in {int(np.searchsorted(np.cumsum(np.sort(pop[ok])[::-1]), total / 2)) + 1} cells")


if __name__ == "__main__":
    main()

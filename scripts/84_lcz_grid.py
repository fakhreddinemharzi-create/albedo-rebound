"""Urban form of the one-degree cells of the contiguous United States from
the Local Climate Zone (LCZ) map of Demuzere et al. (2020).

The 100 m LCZ map (EPSG:4326 version, classes 1-17; classes 7 and 9 are not
mapped) is read in bands of one degree of latitude, and the area of every
class is summed over the cells of the gridded ledger (67_conus_ledger.py).

Input : RAW/lcz/CONUS_LCZ_map_NLCD_v1.0_epsg4326.tif
        (https://doi.org/10.6084/m9.figshare.11416950, CC BY 4.0)
Writes results/conus_lcz_v3.npz (area of every class in every cell, km2)
       and results/conus_lcz_v3.json (summary).
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb.config import RAW, RES  # noqa: E402

NCLASS = 18                         # 0 (no data) and classes 1-17
NAMES = {1: "compact high-rise", 2: "compact mid-rise", 3: "compact low-rise", 4: "open high-rise",
         5: "open mid-rise", 6: "open low-rise", 7: "lightweight low-rise", 8: "large low-rise",
         9: "sparsely built", 10: "heavy industry", 11: "dense trees", 12: "scattered trees",
         13: "bush, scrub", 14: "low plants", 15: "bare rock or paved", 16: "bare soil or sand", 17: "water"}
R_EARTH = 6371.0088


def main():
    z = np.load(RES / "conus_ledger_v3.npz")
    lat, lon = z["lat"], z["lon"]
    ok = z["land"] & np.isfinite(z["R_static"])
    lo0, la0 = lon[0] - 0.5, lat[0] - 0.5
    area = np.zeros((NCLASS, len(lat), len(lon)))
    with rasterio.open(RAW / "lcz" / "CONUS_LCZ_map_NLCD_v1.0_epsg4326.tif") as r:
        T = r.transform
        dx, dy = T.a, -T.e
        xs = T.c + (np.arange(r.width) + 0.5) * dx
        col_cell = np.floor(xs - lo0).astype(int)
        inx = (col_cell >= 0) & (col_cell < len(lon))
        ys = T.f - (np.arange(r.height) + 0.5) * dy
        row_cell = np.floor(ys - la0).astype(int)
        for a in range(len(lat)):
            rows = np.where(row_cell == a)[0]
            if not len(rows):
                continue
            w = Window(0, int(rows[0]), r.width, int(rows[-1] - rows[0] + 1))
            band = r.read(1, window=w)
            band = band[:, inx]
            cc = np.broadcast_to(col_cell[inx][None, :], band.shape)
            cls = np.clip(band, 0, NCLASS - 1).astype(np.int64)
            cnt = np.bincount((cc * NCLASS + cls).ravel(), minlength=len(lon) * NCLASS).reshape(len(lon), NCLASS)
            # pixel area at the latitude of each row, averaged over the band
            px = (np.radians(dx) * R_EARTH) * (np.radians(dy) * R_EARTH) * np.mean(np.cos(np.radians(ys[rows])))
            area[:, a, :] = cnt.T * px
            print(f"lat {lat[a]:5.1f}: {len(rows)} rows, built {area[1:11, a, :].sum():8.0f} km2", flush=True)
    built = area[1:11].sum(0)
    np.savez_compressed(RES / "conus_lcz_v3.npz", lat=lat, lon=lon, area=area, names=np.array(
        [NAMES.get(k, "none") for k in range(NCLASS)]))
    tot = {NAMES[k]: float(area[k][ok].sum()) for k in range(1, NCLASS)}
    b = sum(tot[NAMES[k]] for k in range(1, 11))
    summ = dict(source="Demuzere et al. (2020), doi:10.6084/m9.figshare.11416950",
                built_km2=b, built_outside_cells_km2=float(built[~ok].sum()),
                share_of_built={NAMES[k]: tot[NAMES[k]] / b for k in range(1, 11)},
                paved_km2=tot[NAMES[15]], cells_with_built=int(((built > 0) & ok).sum()), ncells=int(ok.sum()))
    (RES / "conus_lcz_v3.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

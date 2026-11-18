"""Hourly MERRA-2 and CERES SYN1deg series at the ten sites (NASA POWER Zarr)."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb.config import RAW, SITES  # noqa: E402
from alrb import io_power  # noqa: E402

VARS = {"merra2": ["TQV", "TO3", "SNODP", "PRECSNO", "FRSNO", "PS", "T2M", "TS", "TSOIL1", "GWETTOP", "GWETROOT", "PRECTOTCORR"],
        "syn1deg": ["TOA_SW_DWN", "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "ALLSKY_SFC_LW_DWN", "ALLSKY_SRF_ALB",
                    "AOD_55", "PW", "CLOUD_AMT", "ALLSKY_SFC_SW_UP"]}
OUT = RAW / "power"
OUT.mkdir(parents=True, exist_ok=True)


def one(args):
    s, prod = args
    dest = OUT / f"{s.key}_{prod}_hourly.parquet"
    if dest.exists():
        return f"{s.key} {prod} cached"
    cols = {}
    cells = {}
    for v in VARS[prod]:
        m = io_power.meta(prod, "hourly")
        if f"{v}/.zarray" not in m:
            continue
        ser, ll = io_power.series(prod, "hourly", v, s.lat, s.lon, "2016-12-31", "2025-01-01 23:00")
        cols[v] = ser
        cells[v] = ll
    df = pd.DataFrame(cols)
    df.attrs["cell"] = str(next(iter(cells.values())))
    df.to_parquet(dest)
    return f"{s.key} {prod}: {df.shape} cell {df.attrs['cell']}"


if __name__ == "__main__":
    jobs = [(s, p) for s in SITES for p in VARS]
    with ThreadPoolExecutor(4) as ex:
        for msg in ex.map(one, jobs):
            print(msg, flush=True)
    print("POWER_DONE")

"""Extract the GOES-R ABI land-surface temperature at the site pixels for
every hourly CONUS product of the reference years (streamed, not stored)."""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb.config import RAW, SITES, GOES_YEARS  # noqa: E402
from alrb import io_goes  # noqa: E402

OUT = RAW / "goes"
OUT.mkdir(parents=True, exist_ok=True)


def run(sat, year, workers=16):
    dest = OUT / f"lst_{sat}_{year}.parquet"
    if dest.exists():
        print("cached", dest, flush=True)
        return
    sites = [s for s in SITES if s.goes == sat]
    ex_ = io_goes.Extractor(sat, sites)
    ndays = 366 if year % 4 == 0 else 365
    keys = []
    with ThreadPoolExecutor(8) as ex:
        for ks in ex.map(lambda d: io_goes.list_keys(sat, year, d), range(1, ndays + 1)):
            keys += ks
    print(sat, year, len(keys), "files", flush=True)
    # locate on the first file (sequential), then stream the rest in parallel
    ex_.fetch(keys[0])
    recs = []
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for n, res in enumerate(ex.map(ex_.fetch, keys)):
            if res is None:
                continue
            t, rows = res
            for s, r in zip(sites, rows):
                recs.append((t, s.key) + r)
            if n % 1000 == 0:
                print(f"  {n}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(recs, columns=["t_j2000", "site", "lst_c", "dqf_c", "pqi_c", "lst_3x3", "n_good"])
    df["time"] = pd.Timestamp("2000-01-01 12:00:00", tz="UTC") + pd.to_timedelta(df.t_j2000, unit="s")
    df.drop(columns="t_j2000").to_parquet(dest)
    print("wrote", dest, len(df), flush=True)


if __name__ == "__main__":
    for sat in ("goes16", "goes18"):
        for y in GOES_YEARS:
            if sat == "goes18" and y < 2023:
                continue
            run(sat, y)
    print("GOES_DONE")

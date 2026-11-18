"""Monthly CMIP6 historical (1995-2014) shortwave fluxes at the site grid
cells (rsdt, rsut, rsds, rsus) for an independent APRP kernel."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alrb.config import RAW, SITES  # noqa: E402
from alrb import io_cmip6  # noqa: E402

MODELS = ["ACCESS-CM2", "CanESM5", "EC-Earth3", "GFDL-ESM4", "INM-CM5-0", "IPSL-CM6A-LR",
          "MIROC6", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
VARS = ["rsdt", "rsut", "rsds", "rsus"]
OUT = RAW / "cmip6"


def job(a):
    model, var, zstore = a
    dest = OUT / f"{model}_historical_{var}.parquet"
    if dest.exists():
        return "cached " + dest.name
    try:
        df = io_cmip6.points(zstore, var, [s.lat for s in SITES], [s.lon for s in SITES], (1995, 2014))
    except Exception as e:
        return f"FAILED {model} {var}: {e}"
    df.columns = [s.key for s in SITES]
    df.index = df.index.astype(str)
    df.to_parquet(dest)
    return f"{model} {var} {df.shape}"


if __name__ == "__main__":
    cat = pd.read_csv(io_cmip6.CATALOG)
    cat = cat[(cat.table_id == "Amon") & (cat.member_id == "r1i1p1f1") & cat.source_id.isin(MODELS)
              & (cat.experiment_id == "historical") & cat.variable_id.isin(VARS)]
    cat = cat.sort_values("version").groupby(["source_id", "variable_id"]).tail(1)
    jobs = [(r.source_id, r.variable_id, r.zstore) for r in cat.itertuples()]
    with ThreadPoolExecutor(4) as ex:
        for m in ex.map(job, jobs):
            print(m, flush=True)
